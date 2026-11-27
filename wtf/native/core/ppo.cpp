#include "ppo.hpp"
#include <cstring>
#include <numeric>
namespace wtfcore {
namespace {
std::vector<float> floats(const Value &v) {
    if (auto packed = std::get_if<Value::FloatArrayPtr>(&v.data))
        return **packed;
    std::vector<float> r;
    for (const auto &x : v.items())
        r.push_back(static_cast<float>(x.number()));
    return r;
}
Value float_values(const std::vector<float> &v) {
    return Value::float_array(v);
}
Value ordered_layer(const std::vector<float> &weights,
                    const std::vector<float> &bias, bool bias_first) {
    return bias_first ? object({{"bias", float_values(bias)},
                                {"weights", float_values(weights)}})
                      : object({{"weights", float_values(weights)},
                                {"bias", float_values(bias)}});
}
const char *metric_names[] = {"policy_loss", "value_loss", "entropy",
                              "approximate_kl", "clip_fraction"};
} // namespace
PPO::PPO(const Value &s)
    : config_(s.at("config")), last_metrics_(s.at("last_metrics")) {
    if (s.text("algorithm") != "ppo" || config_.text("type", "ppo") != "ppo")
        throw std::runtime_error("Native learner supports ordinary PPO only");
    if (s.contains("late") || s.contains("auxiliary") ||
        config_.contains("late_consequence") ||
        config_.contains("auxiliary_consequence"))
        throw std::runtime_error("Native PPO does not yet support late or "
                                 "auxiliary consequence heads");
    if (s.text("precision", "float64") != "float32")
        throw std::runtime_error("Native PPO requires float32 checkpoints");
    for (const auto &a : s.at("actions").items())
        actions_.push_back(a.str());
    if (actions_.empty())
        throw std::runtime_error("Native PPO requires actions");
    const auto &net = s.at("network").items(), &mom = s.at("moments").items(),
               &var = s.at("variances").items();
    if (net.empty() || net.size() != mom.size() || net.size() != var.size())
        throw std::runtime_error("Invalid native PPO network state");
    std::size_t previous = 0;
    for (std::size_t j = 0; j < net.size(); ++j) {
        Layer l;
        l.bias_first = net[j].fields().front().first == "bias";
        l.moment_bias_first = mom[j].fields().front().first == "bias";
        l.variance_bias_first = var[j].fields().front().first == "bias";
        l.weights = floats(net[j].at("weights"));
        l.bias = floats(net[j].at("bias"));
        l.mw = floats(mom[j].at("weights"));
        l.mb = floats(mom[j].at("bias"));
        l.vw = floats(var[j].at("weights"));
        l.vb = floats(var[j].at("bias"));
        if (l.bias.empty() || l.weights.empty() ||
            l.weights.size() % l.bias.size())
            throw std::runtime_error("Invalid native PPO layer shape");
        l.inputs = l.weights.size() / l.bias.size();
        if ((j && l.inputs != previous) || l.mw.size() != l.weights.size() ||
            l.vw.size() != l.weights.size() || l.mb.size() != l.bias.size() ||
            l.vb.size() != l.bias.size())
            throw std::runtime_error("Invalid native PPO optimizer shape");
        previous = l.bias.size();
        layers_.push_back(std::move(l));
    }
    if (previous != actions_.size() + 1)
        throw std::runtime_error("Invalid native PPO output shape");
    steps_ = s.at("steps").i64();
    updates_ = s.at("updates").i64();
    rollout_updates_ = s.at("rollout_updates").i64();
    if (steps_ < 0 || updates_ < 0 || rollout_updates_ < 0)
        throw std::runtime_error("Native PPO counters must be nonnegative");
    rng_.restore(s.at("rng_state"));
    training_rng_.restore(s.at("training_rng_state"));
    for (const auto &r : s.at("rollout").items())
        rollout_.push_back({numbers(r.at("observation")), r.integer("action"),
                            r.num("reward"), r.num("log_probability"),
                            r.num("value"), r.num("next_value"),
                            r.flag("terminated"), r.flag("truncated")});
    for (const auto &sample : rollout_) {
        if (sample.action < 0 ||
            static_cast<std::size_t>(sample.action) >= actions_.size() ||
            sample.observation.size() != layers_.front().inputs)
            throw std::runtime_error("Invalid native PPO rollout sample");
    }
    if (!s.get("pending").null()) {
        const auto &pending = s.at("pending").items();
        if (pending.size() != 4 ||
            pending[0].array_size() != layers_.front().inputs ||
            pending[1].as_int() < 0 ||
            static_cast<std::size_t>(pending[1].as_int()) >= actions_.size())
            throw std::runtime_error("Invalid native PPO pending sample");
        pending_.observation = numbers(pending[0]);
        pending_.action = pending[1].as_int();
        pending_.logp = pending[2].number();
        pending_.value = pending[3].number();
        pending_.valid = true;
        // An imported record keeps its exact scalar types until consumed.
        pending_.imported = s.at("pending");
    }
    settings_ = {
        config_.num("discount_factor"),   config_.num("gae_lambda"),
        config_.num("clip_ratio"),        config_.num("entropy_coefficient"),
        config_.num("value_coefficient"), config_.num("target_kl", 0.03),
        config_.num("max_grad_norm"),     config_.num("learning_rate"),
        config_.integer("rollout_steps"), config_.integer("batch_size"),
        config_.integer("epochs"),        config_.flag("normalize_advantages")};
    if (settings_.rollout_steps < 1 || settings_.batch_size < 1 ||
        settings_.epochs < 1)
        throw std::runtime_error("Invalid native PPO learning schedule");
}
const std::vector<std::vector<double>> &
PPO::forward(const std::vector<double> &observation, bool value_only) {
    if (observation.size() != layers_[0].inputs)
        throw std::runtime_error("Native PPO observation shape mismatch");
    // The previous next-value query commonly has the same inputs/weights as
    // this action query. Reuse its hidden activations, not another animal's
    // policy. Bitwise input comparison preserves signed-zero distinctions.
    bool cached = tape_valid_ && tape_updates_ == updates_ &&
                  tape_[0].size() == observation.size() &&
                  std::memcmp(tape_[0].data(), observation.data(),
                              observation.size() * sizeof(double)) == 0;
    if (cached && (value_only || tape_full_))
        return tape_;
    // Never expose a partially rewritten tape after allocation failure.
    tape_valid_ = false;
    if (!cached) {
        tape_.resize(layers_.size() + 1);
        tape_[0].assign(observation.begin(), observation.end());
    }
    for (std::size_t j = cached ? layers_.size() - 1 : 0; j < layers_.size();
         ++j) {
        const auto &l = layers_[j];
        const auto &x = tape_[j];
        auto &y = tape_[j + 1];
        bool last = j + 1 == layers_.size();
        std::size_t first = value_only && last ? l.bias.size() - 1 : 0;
        y.resize(l.bias.size() - first);
        for (std::size_t k = first; k < l.bias.size(); ++k) {
            Sum sum;
            for (std::size_t i = 0; i < l.inputs; ++i)
                sum.add(x[i] *
                        static_cast<double>(l.weights[k * l.inputs + i]));
            double v = static_cast<double>(l.bias[k]) + sum.result();
            y[k - first] = last ? v : std::tanh(v);
        }
    }
    tape_updates_ = updates_;
    tape_valid_ = true;
    tape_full_ = !value_only;
    return tape_;
}
void PPO::distribution(const std::vector<double> &outputs,
                       std::vector<double> &p, std::vector<double> &logs) {
    std::size_t n = outputs.size() - 1;
    p.resize(n);
    logs.resize(n);
    double maximum = *std::max_element(outputs.begin(), outputs.end() - 1);
    Sum sum;
    for (std::size_t i = 0; i < n; ++i) {
        p[i] = std::exp(outputs[i] - maximum);
        sum.add(p[i]);
    }
    double total = sum.result(), normalizer = std::log(total);
    for (std::size_t i = 0; i < n; ++i) {
        logs[i] = (outputs[i] - maximum) - normalizer;
        p[i] /= total;
    }
}
std::string PPO::choose(const std::vector<double> &observation, bool training,
                        bool random_baseline) {
    if (random_baseline)
        return actions_[rng_.below(actions_.size())];
    const auto &out = forward(observation).back();
    auto &p = probabilities_;
    auto &logs = log_probabilities_;
    distribution(out, p, logs);
    if (!training)
        return actions_[std::max_element(p.begin(), p.end()) - p.begin()];
    double draw = rng_.random(), cumulative = 0;
    std::size_t action = p.size() - 1;
    for (std::size_t i = 0; i < p.size(); ++i) {
        cumulative += p[i];
        if (draw < cumulative) {
            action = i;
            break;
        }
    }
    pending_.observation.assign(observation.begin(), observation.end());
    pending_.action = static_cast<int>(action);
    pending_.logp = logs[action];
    pending_.value = out.back();
    pending_.valid = true;
    pending_.imported = Value{};
    return actions_[action];
}
std::string PPO::trial_action(const std::vector<double> &observation,
                              Random &external_rng, Value &, const Value &) {
    const auto &out = forward(observation).back();
    std::vector<double> probabilities, logs;
    distribution(out, probabilities, logs);
    double draw = external_rng.random(), cumulative = 0;
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        cumulative += probabilities[i];
        if (draw < cumulative)
            return actions_[i];
    }
    return actions_.back();
}

void PPO::learn(const std::vector<double> &observation,
                const std::string &action, double reward,
                const std::vector<double> *next, bool terminated,
                bool truncated) {
    auto found = std::find(actions_.begin(), actions_.end(), action);
    if (found == actions_.end())
        throw std::runtime_error("Unknown native PPO action");
    if (steps_ == std::numeric_limits<std::int64_t>::max())
        throw std::runtime_error("Native PPO step counter capacity exceeded");
    int index = static_cast<int>(found - actions_.begin());
    double old, value;
    if (pending_.valid && pending_.action == index &&
        pending_.observation == observation) {
        old = pending_.logp;
        value = pending_.value;
    } else {
        const auto &out = forward(observation).back();
        auto &p = probabilities_;
        auto &logs = log_probabilities_;
        distribution(out, p, logs);
        old = logs[index];
        value = out.back();
    }
    pending_.valid = false;
    pending_.imported = Value{};
    if (!terminated && !next)
        throw std::runtime_error(
            "PPO requires a next observation for nonterminal experience");
    double nv = terminated ? 0 : forward(*next, true).back().back();
    rollout_.push_back(
        {observation, index, reward, old, value, nv, terminated, truncated});
    ++steps_;
    if (rollout_.size() >= static_cast<std::size_t>(settings_.rollout_steps) ||
        terminated || truncated)
        update();
}
void PPO::update() {
    if (rollout_updates_ == std::numeric_limits<std::int64_t>::max())
        throw std::runtime_error(
            "Native PPO rollout counter capacity exceeded");
    const std::size_t n = rollout_.size();
    auto &advantages = advantages_;
    auto &targets = targets_;
    advantages.resize(n);
    targets.resize(n);
    double trace = 0, discount = settings_.discount, lambda = settings_.lambda;
    for (std::size_t i = n; i-- > 0;) {
        const auto &s = rollout_[i];
        double bootstrap = s.terminated ? 0 : s.next_value;
        double delta = s.reward + discount * bootstrap - s.value;
        double continuation = s.terminated || s.truncated ? 0 : 1;
        trace = delta + discount * lambda * continuation * trace;
        advantages[i] = trace;
        targets[i] = trace + s.value;
    }
    if (settings_.normalize && n > 1) {
        Sum sum;
        for (double x : advantages)
            sum.add(x);
        double mean = sum.result() / n;
        Sum squares;
        for (double x : advantages)
            squares.add(std::pow(x - mean, 2.0));
        double deviation = std::sqrt(squares.result() / n + 1e-8);
        for (auto &x : advantages)
            x = (x - mean) / deviation;
    }
    auto &gradients = gradients_;
    if (gradients.empty()) {
        std::vector<Gradient> fresh;
        fresh.reserve(layers_.size());
        for (const auto &l : layers_)
            fresh.push_back({std::vector<double>(l.weights.size()),
                             std::vector<double>(l.bias.size())});
        gradients = std::move(fresh);
    }
    std::array<double, 5> totals{};
    std::size_t samples = 0;
    bool stopped = false;
    auto &indices = indices_;
    indices.resize(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::size_t batch_size = settings_.batch_size;
    for (int epoch = 0; epoch < settings_.epochs; ++epoch) {
        training_rng_.shuffle(indices);
        for (std::size_t offset = 0; offset < n; offset += batch_size) {
            std::size_t count = std::min(batch_size, n - offset);
            for (auto &g : gradients) {
                std::fill(g.w.begin(), g.w.end(), 0);
                std::fill(g.b.begin(), g.b.end(), 0);
            }
            std::array<double, 5> batch{};
            for (std::size_t a = offset; a < offset + count; ++a) {
                std::size_t i = indices[a];
                const auto &s = rollout_[i];
                const auto &tape = forward(s.observation);
                const auto &out = tape.back();
                auto &p = probabilities_;
                auto &logs = log_probabilities_;
                distribution(out, p, logs);
                double lr = std::fmax(-60,
                                      std::fmin(60, logs[s.action] - s.logp)),
                       ratio = std::exp(lr), clip = settings_.clip,
                       adv = advantages[i],
                       clipped =
                           std::fmax(1 - clip, std::fmin(1 + clip, ratio));
                bool inactive = (adv > 0 && ratio > 1 + clip) ||
                                (adv < 0 && ratio < 1 - clip);
                double pl = -std::fmin(ratio * adv, clipped * adv),
                       d = inactive ? 0 : -adv * ratio;
                Sum sum;
                for (std::size_t k = 0; k < p.size(); ++k)
                    sum.add(p[k] * logs[k]);
                double entropy = -sum.result();
                auto &delta = delta_;
                delta.resize(out.size());
                for (std::size_t k = 0; k < p.size(); ++k)
                    delta[k] =
                        d * ((static_cast<int>(k) == s.action ? 1 : 0) - p[k]) +
                        settings_.entropy * p[k] * (logs[k] + entropy);
                double error = out.back() - targets[i],
                       vl = 0.5 * error * error;
                delta.back() = settings_.value * error;
                for (std::size_t j = layers_.size(); j-- > 0;) {
                    const auto &l = layers_[j];
                    const auto &x = tape[j];
                    auto &g = gradients[j];
                    auto &previous = previous_;
                    previous.assign(j ? l.inputs : 0, 0.0);
                    for (std::size_t k = 0; k < l.bias.size(); ++k) {
                        double derivative = delta[k];
                        g.b[k] += derivative;
                        for (std::size_t h = 0; h < l.inputs; ++h) {
                            g.w[k * l.inputs + h] += derivative * x[h];
                            if (j)
                                previous[h] += derivative *
                                               static_cast<double>(
                                                   l.weights[k * l.inputs + h]);
                        }
                    }
                    if (j) {
                        delta.resize(l.inputs);
                        for (std::size_t h = 0; h < l.inputs; ++h)
                            delta[h] = previous[h] * (1 - x[h] * x[h]);
                    }
                }
                batch[0] += pl;
                batch[1] += vl;
                batch[2] += entropy;
                batch[3] += (ratio - 1) - lr;
                batch[4] += static_cast<double>(std::fabs(ratio - 1) > clip);
            }
            double target_kl = settings_.target_kl;
            if (target_kl && batch[3] / count > target_kl) {
                stopped = true;
                break;
            }
            Sum norm_sum;
            for (std::size_t j = 0; j < gradients.size(); ++j) {
                const auto &g = gradients[j];
                bool first = layers_[j].bias_first;
                for (const auto *row :
                     {first ? &g.b : &g.w, first ? &g.w : &g.b})
                    for (double x : *row)
                        norm_sum.add(
                            std::pow(x / static_cast<double>(count), 2.0));
            }
            double norm = std::sqrt(norm_sum.result()),
                   factor = std::fmin(1, settings_.max_grad / (norm + 1e-12)) /
                            count;
            if (updates_ == std::numeric_limits<std::int64_t>::max())
                throw std::runtime_error(
                    "Native PPO update counter capacity exceeded");
            ++updates_;
            double c1 = 1 - std::pow(0.9, static_cast<double>(updates_)),
                   c2 = 1 - std::pow(0.999, static_cast<double>(updates_)),
                   rate = settings_.rate;
            auto apply = [&](std::vector<float> &p, std::vector<float> &m,
                             std::vector<float> &v,
                             const std::vector<double> &gradient) {
                for (std::size_t k = 0; k < p.size(); ++k) {
                    double g = gradient[k] * factor;
                    m[k] = static_cast<float>(0.9 * static_cast<double>(m[k]) +
                                              0.1 * g);
                    v[k] = static_cast<float>(
                        0.999 * static_cast<double>(v[k]) + 0.001 * g * g);
                    p[k] = static_cast<float>(
                        static_cast<double>(p[k]) -
                        rate * (static_cast<double>(m[k]) / c1) /
                            (std::sqrt(static_cast<double>(v[k]) / c2) + 1e-8));
                }
            };
            for (std::size_t j = 0; j < layers_.size(); ++j) {
                auto &l = layers_[j];
                apply(l.weights, l.mw, l.vw, gradients[j].w);
                apply(l.bias, l.mb, l.vb, gradients[j].b);
            }
            samples += count;
            for (int k = 0; k < 5; ++k)
                totals[k] += batch[k];
        }
        if (stopped)
            break;
    }
    ++rollout_updates_;
    last_metrics_ = Value::object();
    if (samples)
        for (int k = 0; k < 5; ++k)
            last_metrics_[metric_names[k]] = totals[k] / samples;
    last_metrics_["early_stopped"] = stopped;
    rollout_.clear();
}
Value PPO::checkpoint() {
    Value net = Value::array(), mom = Value::array(), var = Value::array(),
          rollout = Value::array(), actions = Value::array();
    for (const auto &a : actions_)
        actions.push(a);
    for (const auto &l : layers_) {
        net.push(ordered_layer(l.weights, l.bias, l.bias_first));
        mom.push(ordered_layer(l.mw, l.mb, l.moment_bias_first));
        var.push(ordered_layer(l.vw, l.vb, l.variance_bias_first));
    }
    for (const auto &s : rollout_)
        rollout.push(object({{"observation", observation_values(s.observation)},
                             {"action", s.action},
                             {"reward", s.reward},
                             {"log_probability", s.logp},
                             {"value", s.value},
                             {"next_value", s.next_value},
                             {"terminated", s.terminated},
                             {"truncated", s.truncated}}));
    return object({{"algorithm", "ppo"},
                   {"precision", "float32"},
                   {"config", config_},
                   {"actions", actions},
                   {"network", net},
                   {"moments", mom},
                   {"variances", var},
                   {"rollout", rollout},
                   {"pending", pending_state()},
                   {"steps", steps_},
                   {"updates", updates_},
                   {"rollout_updates", rollout_updates_},
                   {"last_metrics", last_metrics_},
                   {"rng_state", rng_.checkpoint()},
                   {"training_rng_state", training_rng_.checkpoint()}});
}
Value PPO::pending_state() const {
    if (!pending_.valid)
        return Value{};
    if (!pending_.imported.null())
        return pending_.imported;
    return array({observation_values(pending_.observation), pending_.action,
                  pending_.logp, pending_.value});
}
Value PPO::observation_values(const std::vector<double> &observation) const {
    return world_observation_values(
        observation, config_.integer("observation_version", 1), world_inputs_);
}
Value PPO::model_state() const {
    Value network = Value::array();
    for (const auto &layer : layers_)
        network.push(
            ordered_layer(layer.weights, layer.bias, layer.bias_first));
    return object({{"precision", "float32"},
                   {"network", network},
                   {"config", config_},
                   {"updates", updates_}});
}
Value PPO::diagnostics(const std::vector<double> &observation,
                       bool activations) {
    const auto &tape = forward(observation);
    const auto &out = tape.back();
    auto &p = probabilities_;
    auto &logs = log_probabilities_;
    distribution(out, p, logs);
    Value probabilities = Value::object();
    Sum entropy;
    for (std::size_t i = 0; i < p.size(); ++i) {
        probabilities[actions_[i]] = p[i];
        entropy.add(p[i] * logs[i]);
    }
    Value info =
        object({{"algorithm", "ppo"},
                {"buffer_kind", "rollout"},
                {"buffer_size", static_cast<std::int64_t>(rollout_.size())},
                {"features", observation_values(observation)},
                {"action_probabilities", probabilities},
                {"state_value", out.back()},
                {"entropy", -entropy.result()},
                {"updates", updates_},
                {"rollout_updates", rollout_updates_},
                {"rollout_size", static_cast<std::int64_t>(rollout_.size())},
                {"training_metrics", last_metrics_}});
    if (activations) {
        Value hidden = Value::array();
        for (std::size_t j = 1; j + 1 < tape.size(); ++j) {
            Value layer = Value::array();
            for (double x : tape[j])
                layer.push(static_cast<double>(static_cast<float>(x)));
            hidden.push(layer);
        }
        info["activations"] = object({{"version", 1}, {"layers", hidden}});
    }
    return info;
}
Value PPO::inherit(const Value &mother, const Value &father,
                   std::uint64_t seed) {
    return inherit(
        mother, father,
        std::vector<std::uint32_t>{static_cast<std::uint32_t>(seed),
                                   static_cast<std::uint32_t>(seed >> 32)});
}
Value PPO::inherit(const Value &mother, const Value &father,
                   const std::vector<std::uint32_t> &seed_words) {
    PPO child(mother), other(father);
    if (child.actions_ != other.actions_ ||
        child.layers_.size() != other.layers_.size())
        throw std::runtime_error("Incompatible native PPO parents");
    auto cross_words = seed_words;
    if (cross_words.size() < 3)
        cross_words.resize(3, 0);
    cross_words[0] ^= 0x4F564552U;
    cross_words[1] ^= 0x524F5353U;
    cross_words[2] ^= 0x43U;
    while (cross_words.size() > 1 && cross_words.back() == 0)
        cross_words.pop_back();
    Random cross;
    cross.reseed_words(cross_words);
    for (std::size_t j = 0; j < child.layers_.size(); ++j) {
        auto &l = child.layers_[j];
        const auto &r = other.layers_[j];
        if (l.inputs != r.inputs || l.bias.size() != r.bias.size())
            throw std::runtime_error("Incompatible native PPO parent shapes");
        for (std::size_t i = 0; i < l.weights.size(); ++i)
            if (cross.bits(1))
                l.weights[i] = r.weights[i];
        for (std::size_t i = 0; i < l.bias.size(); ++i)
            if (cross.bits(1))
                l.bias[i] = r.bias[i];
        std::fill(l.mw.begin(), l.mw.end(), 0);
        std::fill(l.mb.begin(), l.mb.end(), 0);
        std::fill(l.vw.begin(), l.vw.end(), 0);
        std::fill(l.vb.begin(), l.vb.end(), 0);
    }
    child.rollout_.clear();
    child.pending_.valid = false;
    child.pending_.imported = Value{};
    child.steps_ = child.updates_ = child.rollout_updates_ = 0;
    child.last_metrics_ = Value::object();
    auto child_words = seed_words;
    if (child_words.empty())
        child_words.push_back(0);
    while (child_words.size() > 1 && child_words.back() == 0)
        child_words.pop_back();
    child.rng_.reseed_words(child_words);
    child_words[0] ^= 0x5F3759DFU;
    while (child_words.size() > 1 && child_words.back() == 0)
        child_words.pop_back();
    child.training_rng_.reseed_words(child_words);
    return child.checkpoint();
}
} // namespace wtfcore
