#include "value_policy.hpp"
#include "dense.hpp"
#include <cstring>
#include <numeric>
namespace wtfcore {
namespace {
using Vec = std::vector<double>;
void capacity(std::int64_t value, const char *name) {
    if (value == std::numeric_limits<std::int64_t>::max())
        throw std::runtime_error(std::string("Native ") + name +
                                 " counter capacity exceeded");
}
std::size_t best_index(const Vec &v) {
    if (v.empty())
        throw std::runtime_error("Empty native action values");
    return std::max_element(v.begin(), v.end()) - v.begin();
}
Value obs_value(const Vec &v, const Value &config, bool world) {
    return world_observation_values(v, config.integer("observation_version", 1),
                                    world);
}

Value named_values(const std::vector<std::string> &names, const Vec &v) {
    Value out = Value::object();
    for (std::size_t i = 0; i < names.size(); ++i)
        out[names[i]] = v.at(i);
    return out;
}
Value action_values(const std::vector<std::string> &names) {
    Value out = Value::array();
    for (const auto &s : names)
        out.push(s);
    return out;
}
Value append_layers(Value a, const Value &b) {
    for (const auto &l : b.items())
        a.push(l);
    return a;
}
Value slice_layers(const Value &v, std::size_t a, std::size_t b) {
    Value out = Value::array();
    for (auto i = a; i < b; ++i)
        out.push(v.items().at(i));
    return out;
}
DenseNetwork q_network(const Value &w) {
    return DenseNetwork(
        array({object({{"weights", w.at("w1")}, {"bias", w.at("b1")}}),
               object({{"weights", w.at("w2")}, {"bias", w.at("b2")}})}));
}
Value q_state(const DenseNetwork &n, const std::vector<std::string> &order) {
    Value out = Value::object();
    for (const auto &key : order)
        out[key] = Value::float_array(key == "w1"   ? n.layers[0].weights
                                      : key == "b1" ? n.layers[0].bias
                                      : key == "w2" ? n.layers[1].weights
                                                    : n.layers[1].bias);
    return out;
}
std::vector<DenseLayer> layer_rows(const Value &v) {
    std::vector<DenseLayer> out;
    for (const auto &r : v.items())
        out.push_back({dense_floats(r.at("weights")),
                       dense_floats(r.at("bias")), 0,
                       r.fields().front().first == "bias"});
    return out;
}
Value row_state(const std::vector<DenseLayer> &rows) {
    Value out = Value::array();
    for (const auto &r : rows)
        out.push(r.bias_first
                     ? object({{"bias", Value::float_array(r.bias)},
                               {"weights", Value::float_array(r.weights)}})
                     : object({{"weights", Value::float_array(r.weights)},
                               {"bias", Value::float_array(r.bias)}}));
    return out;
}
Value activation_layers(const DenseNetwork::Tape &tape) {
    Value out = Value::array();
    for (std::size_t i = 1; i + 1 < tape.size(); ++i) {
        std::vector<float> row;
        for (double x : tape[i])
            row.push_back(static_cast<float>(x));
        out.push(Value::float_array(std::move(row)));
    }
    return object({{"version", 1}, {"layers", out}});
}
} // namespace
struct ValuePolicy::Impl {
    struct Mark {
        std::uint64_t value;
        std::uint32_t index;
        std::uint8_t field, kind;
    };
    struct Sample {
        Vec observation, next;
        std::string action;
        double reward = 0;
        bool has_next = false, terminated = false, truncated = false;
        std::vector<Mark> types;
    };
    static void remember(Sample &sample, const Value &v, std::uint8_t field,
                         std::size_t index = 0) {
        if (auto p = std::get_if<std::int64_t>(&v.data))
            sample.types.push_back({static_cast<std::uint64_t>(*p),
                                    static_cast<std::uint32_t>(index), field,
                                    0});
        else if (auto p = std::get_if<std::uint64_t>(&v.data))
            sample.types.push_back(
                {*p, static_cast<std::uint32_t>(index), field, 1});
        else if (auto p = std::get_if<bool>(&v.data))
            sample.types.push_back({static_cast<std::uint64_t>(*p),
                                    static_cast<std::uint32_t>(index), field,
                                    2});
    }
    static Value remembered(const Mark &mark) {
        if (mark.kind == 1)
            return mark.value;
        if (mark.kind == 2)
            return mark.value != 0;
        std::int64_t value;
        std::memcpy(&value, &mark.value, sizeof value);
        return value;
    }
    Value config, last_metrics = Value::object(), pending, imported_rollout;
    std::string kind;
    std::vector<std::string> actions, q_order, q_target_order;
    DenseNetwork network, target;
    std::vector<DenseNetwork> critics, target_critics;
    std::vector<DenseLayer> moments, variances;
    std::vector<Sample> replay;
    Random rng, training_rng;
    std::int64_t steps = 0, updates = 0, rollout_updates = 0;
    int cursor = 0, quantiles = 1, replay_capacity = 0, batch_size = 0,
        learning_starts = 0, train_every = 0, target_every = 1;
    double epsilon = 0, decay = 0, minimum = 0, discount = 0, threshold = 1,
           rate = 0, max_grad = 1, temperature = 0, tau = 0;
    bool world_inputs = false;
    explicit Impl(const Value &s)
        : config(s.at("config")), kind(config.text("type")) {
        if (s.contains("auxiliary") || s.contains("late"))
            throw std::runtime_error(
                "This native value policy has no auxiliary/late heads");
        if (s.text("precision", "float64") != "float32")
            throw std::runtime_error(
                "Native value policies require FP32 storage");
        if (kind != "q_network" && kind != "dqn" && kind != "ddqn" &&
            kind != "qrdqn" && kind != "discrete_sac")
            throw std::runtime_error("Unsupported native value policy");
        for (const auto &a : s.at("actions").items())
            actions.push_back(a.str());
        if (actions.empty())
            throw std::runtime_error("Native policy needs actions");
        if (kind == "q_network") {
            network = q_network(s.at("weights"));
            target = q_network(s.at("target"));
            for (const auto &[key, ignored] : s.at("weights").fields()) {
                (void)ignored;
                q_order.push_back(key);
            }
            for (const auto &[key, ignored] : s.at("target").fields()) {
                (void)ignored;
                q_target_order.push_back(key);
            }
        } else if (kind == "discrete_sac") {
            const auto &net = s.at("network");
            auto count = net.items().size();
            if (count % 3)
                throw std::runtime_error("Invalid SAC bundle layer count");
            auto block = count / 3;
            network = DenseNetwork(slice_layers(net, 0, block));
            critics.emplace_back(slice_layers(net, block, 2 * block));
            critics.emplace_back(slice_layers(net, 2 * block, 3 * block));
            const auto &targets = s.at("target_critics").items();
            if (targets.size() != 2)
                throw std::runtime_error("SAC requires twin target critics");
            for (const auto &v : targets)
                target_critics.emplace_back(v);
            rollout_updates = s.at("rollout_updates").i64();
            pending = s.get("pending");
            imported_rollout = s.at("rollout");
        } else {
            network = DenseNetwork(s.at("network"));
            target = DenseNetwork(s.at("target_network"));
        }
        if (kind != "q_network") {
            moments = layer_rows(s.at("moments"));
            variances = layer_rows(s.at("variances"));
            last_metrics = s.at("last_metrics");
        }
        if (kind == "qrdqn")
            quantiles = config.integer("num_quantiles");
        if (quantiles < 1 ||
            network.layers.back().bias.size() !=
                actions.size() * static_cast<std::size_t>(quantiles))
            throw std::runtime_error("Invalid native action output shape");
        replay_capacity = config.integer("replay_capacity");
        batch_size = config.integer("batch_size");
        learning_starts = config.integer("learning_starts_after");
        train_every = config.integer("train_every_steps");
        if (replay_capacity < 1 || batch_size < 1 ||
            batch_size > replay_capacity || train_every < 1)
            throw std::runtime_error("Invalid native replay schedule");
        discount = config.num("discount_factor");
        rate = config.num("learning_rate");
        max_grad = config.num("max_grad_norm", 1);
        threshold = config.num("huber_threshold", 1);
        if (kind != "discrete_sac") {
            epsilon = s.num("epsilon");
            const auto &e = config.at("exploration");
            decay = e.num("decay_per_decision");
            minimum = e.num("minimum");
            target_every = config.integer("target_update_every_training_steps");
            if (target_every < 1)
                throw std::runtime_error("Invalid target update interval");
        } else {
            temperature = config.num("temperature");
            tau = config.num("target_tau");
        }
        cursor = s.integer("cursor");
        steps = s.at("steps").i64();
        updates = s.at("updates").i64();
        if (cursor < 0 || cursor >= replay_capacity || steps < 0 ||
            updates < 0 || rollout_updates < 0)
            throw std::runtime_error("Invalid native policy counters");
        rng.restore(s.at("rng_state"));
        training_rng.restore(s.at("training_rng_state"));
        for (const auto &e : s.at("replay").items()) {
            Sample item;
            item.observation = numbers(e.at("observation"));
            item.action = e.at("action").str();
            item.reward = e.num("reward");
            item.has_next = !e.get("next_observation").null();
            if (item.has_next)
                item.next = numbers(e.at("next_observation"));
            item.terminated = e.flag("terminated");
            item.truncated = e.flag("truncated");
            for (std::size_t j = 0; j < e.at("observation").array_size(); ++j)
                remember(item, e.at("observation").array_value(j), 0, j);
            if (item.has_next)
                for (std::size_t j = 0;
                     j < e.at("next_observation").array_size(); ++j)
                    remember(item, e.at("next_observation").array_value(j), 1,
                             j);
            remember(item, e.at("reward"), 2);
            validate(item);
            replay.push_back(std::move(item));
        }
        if (replay.size() > static_cast<std::size_t>(replay_capacity))
            throw std::runtime_error("Replay exceeds native capacity");
        if (kind == "discrete_sac") {
            for (std::size_t j = 0; j < critics.size(); ++j) {
                if (critics[j].layers.front().inputs !=
                        network.layers.front().inputs ||
                    critics[j].layers.back().bias.size() != actions.size() ||
                    !same_shape(critics[j], target_critics[j]))
                    throw std::runtime_error("Invalid SAC critic/target shape");
            }
        } else if (!same_shape(network, target))
            throw std::runtime_error("Invalid value target shape");
        if (kind != "q_network") {
            auto rows = parameter_rows();
            if (moments.size() != rows.size() ||
                variances.size() != rows.size())
                throw std::runtime_error("Invalid native optimizer rows");
            for (std::size_t j = 0; j < rows.size(); ++j)
                if (moments[j].weights.size() != rows[j]->weights.size() ||
                    moments[j].bias.size() != rows[j]->bias.size() ||
                    variances[j].weights.size() != rows[j]->weights.size() ||
                    variances[j].bias.size() != rows[j]->bias.size())
                    throw std::runtime_error("Invalid native optimizer shape");
        }
    }
    int action_index(const std::string &a) const {
        auto p = std::find(actions.begin(), actions.end(), a);
        if (p == actions.end())
            throw std::runtime_error("Unknown native action");
        return p - actions.begin();
    }
    static bool same_shape(const DenseNetwork &a, const DenseNetwork &b) {
        if (a.layers.size() != b.layers.size())
            return false;
        for (std::size_t i = 0; i < a.layers.size(); ++i)
            if (a.layers[i].inputs != b.layers[i].inputs ||
                a.layers[i].weights.size() != b.layers[i].weights.size() ||
                a.layers[i].bias.size() != b.layers[i].bias.size())
                return false;
        return true;
    }
    void validate(const Sample &s) const {
        if (s.observation.size() != network.layers.front().inputs ||
            (s.has_next && s.next.size() != network.layers.front().inputs))
            throw std::runtime_error(
                "Native replay observation width mismatch");
        (void)action_index(s.action);
    }
    std::vector<DenseLayer *> parameter_rows() {
        std::vector<DenseLayer *> out;
        for (auto &l : network.layers)
            out.push_back(&l);
        for (auto &n : critics)
            for (auto &l : n.layers)
                out.push_back(&l);
        return out;
    }
    Value network_state() const {
        Value out = network.checkpoint();
        for (const auto &n : critics)
            out = append_layers(std::move(out), n.checkpoint());
        return out;
    }
    Vec means(const Vec &v) const {
        if (quantiles == 1)
            return v;
        Vec out;
        for (std::size_t j = 0; j < v.size(); j += quantiles) {
            Sum s;
            for (int k = 0; k < quantiles; ++k)
                s.add(v[j + k]);
            out.push_back(s.result() / quantiles);
        }
        return out;
    }
    Value sample_state(const Sample &s) const {
        Value out = object(
            {{"observation", obs_value(s.observation, config, world_inputs)},
             {"action", s.action},
             {"reward", s.reward},
             {"next_observation",
              s.has_next ? obs_value(s.next, config, world_inputs) : Value{}},
             {"terminated", s.terminated},
             {"truncated", s.truncated}});
        for (const auto &mark : s.types) {
            if (mark.field == 2)
                out["reward"] = remembered(mark);
            else
                out[mark.field == 0 ? "observation" : "next_observation"]
                    .items()
                    .at(mark.index) = remembered(mark);
        }
        return out;
    }
    void adam(std::vector<DenseGradients *> &parts, std::size_t count) {
        Sum sum;
        for (const auto *part : parts)
            for (const auto &g : *part) {
                for (const auto *row :
                     {g.bias_first ? &g.b : &g.w, g.bias_first ? &g.w : &g.b})
                    for (double x : *row)
                        sum.add(std::pow(x / static_cast<double>(count), 2.0));
            }
        double norm = std::sqrt(sum.result()),
               factor = std::fmin(1.0, max_grad / (norm + 1e-12)) / count;
        capacity(updates, "optimizer update");
        ++updates;
        double c1 = 1.0 - std::pow(0.9, static_cast<double>(updates)),
               c2 = 1.0 - std::pow(0.999, static_cast<double>(updates));
        auto rows = parameter_rows();
        std::size_t j = 0;
        for (auto *part : parts)
            for (const auto &gradient : *part) {
                auto apply = [&](std::vector<float> &p, std::vector<float> &m,
                                 std::vector<float> &v, const Vec &g) {
                    for (std::size_t i = 0; i < p.size(); ++i) {
                        double value = g[i] * factor;
                        m[i] = static_cast<float>(
                            0.9 * static_cast<double>(m[i]) + 0.1 * value);
                        v[i] = static_cast<float>(
                            0.999 * static_cast<double>(v[i]) +
                            0.001 * value * value);
                        p[i] = static_cast<float>(
                            static_cast<double>(p[i]) -
                            rate * (static_cast<double>(m[i]) / c1) /
                                (std::sqrt(static_cast<double>(v[i]) / c2) +
                                 1e-8));
                    }
                };
                apply(rows[j]->weights, moments[j].weights,
                      variances[j].weights, gradient.w);
                apply(rows[j]->bias, moments[j].bias, variances[j].bias,
                      gradient.b);
                ++j;
            }
    }
    void legacy_update(const std::vector<std::size_t> &indices) {
        capacity(updates, "optimizer update");
        auto gradients = network.zeros();
        const auto &w2 = network.layers[1].weights;
        auto hidden = network.layers[0].bias.size(),
             inputs = network.layers[0].inputs;
        DenseNetwork::Tape tape, target_tape;
        for (auto i : indices) {
            const auto &s = replay[i];
            int a = action_index(s.action);
            network.forward_into(s.observation, tape, a, a + 1);
            const auto &h = tape[1];
            double value = tape.back()[0], expected = s.reward;
            if (!s.terminated) {
                if (!s.has_next)
                    throw std::runtime_error(
                        "Q learning needs nonterminal next observation");
                target.forward_into(s.next, target_tape);
                const auto &q = target_tape.back();
                expected += discount * (*std::max_element(q.begin(), q.end()));
            }
            double delta = std::fmax(-1.0, std::fmin(1.0, value - expected));
            gradients[1].b[a] += delta;
            for (std::size_t j = 0; j < hidden; ++j) {
                gradients[1].w[a * hidden + j] += delta * h[j];
                double dh = delta * static_cast<double>(w2[a * hidden + j]) *
                            (1.0 - std::pow(h[j], 2.0));
                gradients[0].b[j] += dh;
                for (std::size_t k = 0; k < inputs; ++k)
                    gradients[0].w[j * inputs + k] += dh * s.observation[k];
            }
        }
        double scale = rate / batch_size;
        for (std::size_t j = 0; j < network.layers.size(); ++j) {
            auto &l = network.layers[j];
            for (std::size_t i = 0; i < l.weights.size(); ++i)
                l.weights[i] =
                    static_cast<float>(static_cast<double>(l.weights[i]) -
                                       scale * gradients[j].w[i]);
            for (std::size_t i = 0; i < l.bias.size(); ++i)
                l.bias[i] = static_cast<float>(static_cast<double>(l.bias[i]) -
                                               scale * gradients[j].b[i]);
        }
        capacity(updates, "optimizer update");
        ++updates;
        if (updates % target_every == 0) {
            target = network;
            q_target_order = q_order;
        }
    }
    void value_update(const std::vector<std::size_t> &indices) {
        auto gradients = network.zeros();
        double total_loss = 0;
        DenseNetwork::Tape tape, next_tape, target_tape;
        Vec derivative;
        for (auto i : indices) {
            const auto &s = replay[i];
            network.forward_into(s.observation, tape);
            const auto &outputs = tape.back();
            int a = action_index(s.action);
            derivative.assign(outputs.size(), 0.0);
            if (kind == "dqn" || kind == "ddqn") {
                double expected = s.reward;
                if (!s.terminated) {
                    if (!s.has_next)
                        throw std::runtime_error(
                            "DQN needs nonterminal next observation");
                    double bootstrap;
                    if (kind == "ddqn") {
                        network.forward_into(s.next, next_tape);
                        auto best = best_index(next_tape.back());
                        target.forward_into(s.next, target_tape, best,
                                            best + 1);
                        bootstrap = target_tape.back()[0];
                    } else {
                        target.forward_into(s.next, target_tape);
                        const auto &q = target_tape.back();
                        bootstrap = q[best_index(q)];
                    }
                    expected += discount * bootstrap;
                }
                double error = outputs[a] - expected,
                       absolute = std::fabs(error);
                total_loss += absolute <= threshold
                                  ? 0.5 * error * error
                                  : threshold * (absolute - 0.5 * threshold);
                derivative[a] =
                    std::fmax(-threshold, std::fmin(threshold, error));
            } else {
                Vec targets(quantiles, s.reward);
                if (!s.terminated) {
                    if (!s.has_next)
                        throw std::runtime_error(
                            "QR-DQN needs nonterminal next observation");
                    auto current = means(network.predict(s.next));
                    auto best = best_index(current);
                    auto bootstrap = target.predict_range(
                        s.next, best * quantiles, (best + 1) * quantiles);
                    for (int q = 0; q < quantiles; ++q)
                        targets[q] = s.reward + discount * bootstrap[q];
                }
                double loss = 0;
                Vec g(quantiles, 0.0);
                for (int q = 0; q < quantiles; ++q) {
                    double tauq = (q + 0.5) / quantiles;
                    for (double t : targets) {
                        double delta = t - outputs[a * quantiles + q],
                               weight = std::fabs(
                                   tauq - static_cast<double>(delta < 0)),
                               absolute = std::fabs(delta),
                               huber = absolute <= threshold
                                           ? 0.5 * delta * delta
                                           : threshold *
                                                 (absolute - 0.5 * threshold);
                        loss += weight * huber / threshold;
                        g[q] -=
                            weight *
                            std::fmax(-threshold, std::fmin(threshold, delta)) /
                            threshold;
                    }
                }
                double scale = static_cast<double>(quantiles) * quantiles;
                total_loss += loss / scale;
                for (int q = 0; q < quantiles; ++q)
                    derivative[a * quantiles + q] = g[q] / scale;
            }
            network.accumulate_gradient(tape, derivative, gradients);
        }
        std::vector<DenseGradients *> parts{&gradients};
        adam(parts, indices.size());
        if (updates % target_every == 0)
            target = network;
        last_metrics = object({{kind == "qrdqn" ? "quantile_loss" : "td_loss",
                                total_loss / indices.size()}});
    }
    void sac_update(const std::vector<std::size_t> &indices) {
        capacity(rollout_updates, "rollout update");
        auto actor_gradient = network.zeros();
        std::array<DenseGradients, 2> critic_gradient{critics[0].zeros(),
                                                      critics[1].zeros()};
        double policy_loss = 0, critic_loss = 0, entropy_total = 0;
        for (auto i : indices) {
            const auto &s = replay[i];
            int action = action_index(s.action);
            double expected = s.reward;
            if (!s.terminated) {
                Vec p, logs;
                categorical_distribution(network.predict(s.next), p, logs);
                auto first = target_critics[0].predict(s.next),
                     second = target_critics[1].predict(s.next);
                Sum soft;
                for (std::size_t j = 0; j < actions.size(); ++j)
                    soft.add(p[j] * (std::min(first[j], second[j]) -
                                     temperature * logs[j]));
                expected += discount * soft.result();
            }
            std::array<Vec, 2> q;
            for (std::size_t k = 0; k < 2; ++k) {
                auto tape = critics[k].forward(s.observation);
                q[k] = tape.back();
                double error = q[k][action] - expected;
                Vec derivative(actions.size(), 0.0);
                derivative[action] = error;
                critics[k].accumulate_gradient(tape, derivative,
                                               critic_gradient[k]);
                critic_loss += 0.5 * error * error;
            }
            auto tape = network.forward(s.observation);
            Vec p, logs;
            categorical_distribution(tape.back(), p, logs);
            Sum value, entropy;
            for (std::size_t j = 0; j < actions.size(); ++j) {
                value.add(p[j] * std::min(q[0][j], q[1][j]));
                entropy.add(p[j] * logs[j]);
            }
            double average = value.result(), h = -entropy.result();
            Vec derivative;
            for (std::size_t j = 0; j < actions.size(); ++j)
                derivative.push_back(p[j] *
                                     (temperature * (logs[j] + h) -
                                      std::min(q[0][j], q[1][j]) + average));
            network.accumulate_gradient(tape, derivative, actor_gradient);
            policy_loss += -average - temperature * h;
            entropy_total += h;
        }
        std::vector<DenseGradients *> parts{
            &actor_gradient, &critic_gradient[0], &critic_gradient[1]};
        adam(parts, indices.size());
        double retained = 1 - tau;
        for (std::size_t j = 0; j < 2; ++j)
            for (std::size_t l = 0; l < critics[j].layers.size(); ++l) {
                auto apply = [&](std::vector<float> &left,
                                 const std::vector<float> &right) {
                    for (std::size_t i = 0; i < left.size(); ++i)
                        left[i] = static_cast<float>(
                            retained * static_cast<double>(left[i]) +
                            tau * static_cast<double>(right[i]));
                };
                apply(target_critics[j].layers[l].weights,
                      critics[j].layers[l].weights);
                apply(target_critics[j].layers[l].bias,
                      critics[j].layers[l].bias);
            }
        capacity(rollout_updates, "rollout update");
        ++rollout_updates;
        last_metrics = object({{"policy_loss", policy_loss / indices.size()},
                               {"critic_loss", critic_loss / indices.size()},
                               {"entropy", entropy_total / indices.size()}});
    }
};
ValuePolicy::ValuePolicy(const Value &s) : impl_(std::make_unique<Impl>(s)) {}
ValuePolicy::~ValuePolicy() = default;
ValuePolicy::ValuePolicy(ValuePolicy &&) noexcept = default;
ValuePolicy &ValuePolicy::operator=(ValuePolicy &&) noexcept = default;
std::string ValuePolicy::choose(const Vec &o, bool training,
                                bool random_baseline) {
    auto &p = *impl_;
    if (random_baseline)
        return p.actions[p.rng.below(p.actions.size())];
    if (p.kind != "discrete_sac") {
        if (training && p.rng.random() < p.epsilon)
            return p.actions[p.rng.below(p.actions.size())];
        return p.actions[best_index(p.means(p.network.predict(o)))];
    }
    Vec probabilities, logs;
    categorical_distribution(p.network.predict(o), probabilities, logs);
    if (!training)
        return p.actions[best_index(probabilities)];
    double draw = p.rng.random(), cumulative = 0;
    auto a = p.actions.size() - 1;
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        cumulative += probabilities[i];
        if (draw < cumulative) {
            a = i;
            break;
        }
    }
    p.pending = array({obs_value(o, p.config, p.world_inputs),
                       static_cast<std::int64_t>(a), logs[a], 0.0});
    return p.actions[a];
}
void ValuePolicy::learn(const Vec &o, const std::string &a, double reward,
                        const Vec *next, bool terminal, bool truncated) {
    auto &p = *impl_;
    if (p.kind == "discrete_sac" && !terminal && !next)
        throw std::runtime_error("SAC needs nonterminal next observation");
    Impl::Sample s{o,
                   next ? *next : Vec{},
                   a,
                   reward,
                   next != nullptr,
                   terminal,
                   truncated,
                   {}};
    p.validate(s);
    capacity(p.steps, "step");
    if (p.kind == "discrete_sac")
        p.pending = Value{};
    if (p.replay.size() < static_cast<std::size_t>(p.replay_capacity))
        p.replay.push_back(std::move(s));
    else
        p.replay[p.cursor] = std::move(s);
    p.cursor = (p.cursor + 1) % p.replay_capacity;
    ++p.steps;
    if (p.kind != "discrete_sac")
        p.epsilon = std::max(p.minimum, p.epsilon * p.decay);
    if (p.replay.size() < static_cast<std::size_t>(
                              std::max(p.batch_size, p.learning_starts)) ||
        p.steps % p.train_every)
        return;
    auto indices =
        sample_indices(p.training_rng, p.replay.size(), p.batch_size);
    if (p.kind == "q_network")
        p.legacy_update(indices);
    else if (p.kind == "discrete_sac")
        p.sac_update(indices);
    else
        p.value_update(indices);
}
Value ValuePolicy::checkpoint() const {
    const auto &p = *impl_;
    Value replay = Value::array();
    for (const auto &s : p.replay)
        replay.push(p.sample_state(s));
    Value out;
    if (p.kind == "q_network")
        out = object({{"precision", "float32"},
                      {"config", p.config},
                      {"actions", action_values(p.actions)},
                      {"weights", q_state(p.network, p.q_order)},
                      {"target", q_state(p.target, p.q_target_order)},
                      {"replay", replay},
                      {"cursor", p.cursor},
                      {"steps", p.steps},
                      {"updates", p.updates},
                      {"epsilon", p.epsilon},
                      {"rng_state", p.rng.checkpoint()},
                      {"training_rng_state", p.training_rng.checkpoint()}});
    else if (p.kind == "discrete_sac") {
        out = object(
            {{"algorithm", p.kind},
             {"precision", "float32"},
             {"config", p.config},
             {"actions", action_values(p.actions)},
             {"network", p.network_state()},
             {"moments", row_state(p.moments)},
             {"variances", row_state(p.variances)},
             {"rollout", p.imported_rollout},
             {"pending", p.pending},
             {"steps", p.steps},
             {"updates", p.updates},
             {"rollout_updates", p.rollout_updates},
             {"last_metrics", p.last_metrics},
             {"rng_state", p.rng.checkpoint()},
             {"training_rng_state", p.training_rng.checkpoint()},
             {"replay", replay},
             {"cursor", p.cursor},
             {"target_critics", array({p.target_critics[0].checkpoint(),
                                       p.target_critics[1].checkpoint()})}});
    } else
        out = object({{"algorithm", p.kind},
                      {"precision", "float32"},
                      {"config", p.config},
                      {"actions", action_values(p.actions)},
                      {"network", p.network.checkpoint()},
                      {"target_network", p.target.checkpoint()},
                      {"moments", row_state(p.moments)},
                      {"variances", row_state(p.variances)},
                      {"replay", replay},
                      {"cursor", p.cursor},
                      {"steps", p.steps},
                      {"updates", p.updates},
                      {"epsilon", p.epsilon},
                      {"last_metrics", p.last_metrics},
                      {"rng_state", p.rng.checkpoint()},
                      {"training_rng_state", p.training_rng.checkpoint()}});
    return out;
}
Value ValuePolicy::model_state() const {
    const auto &p = *impl_;
    Value out = object({{"precision", "float32"},
                        {"config", p.config},
                        {"updates", p.updates}});
    out[p.kind == "q_network" ? "weights" : "network"] =
        p.kind == "q_network" ? q_state(p.network, p.q_order)
                              : p.network_state();
    return out;
}
Value ValuePolicy::diagnostics(const Vec &o, bool activations) const {
    const auto &p = *impl_;
    auto tape = p.network.forward(o);
    auto outputs = tape.back();
    Value info;
    if (p.kind == "discrete_sac") {
        Vec probabilities, logs;
        categorical_distribution(outputs, probabilities, logs);
        Sum entropy;
        for (std::size_t i = 0; i < probabilities.size(); ++i)
            entropy.add(probabilities[i] * logs[i]);
        double h = -entropy.result();
        auto first = p.critics[0].predict(o), second = p.critics[1].predict(o);
        Sum value;
        for (std::size_t i = 0; i < p.actions.size(); ++i)
            value.add(probabilities[i] * std::min(first[i], second[i]));
        info = object(
            {{"algorithm", p.kind},
             {"buffer_kind", "replay"},
             {"buffer_size", static_cast<std::int64_t>(p.replay.size())},
             {"features", obs_value(o, p.config, p.world_inputs)},
             {"action_probabilities", named_values(p.actions, probabilities)},
             {"state_value", value.result() + p.temperature * h},
             {"entropy", h},
             {"updates", p.updates},
             {"rollout_updates", p.rollout_updates},
             {"rollout_size", 0},
             {"training_metrics", p.last_metrics},
             {"temperature", p.temperature},
             {"twin_q_values", array({named_values(p.actions, first),
                                      named_values(p.actions, second)})},
             {"critic_separate", true}});
    } else {
        auto q = p.means(outputs);
        info = object(
            {{"algorithm", p.kind == "q_network" ? "q_learning" : p.kind},
             {"features", obs_value(o, p.config, p.world_inputs)},
             {"action_values", named_values(p.actions, q)}});
        if (p.kind == "qrdqn") {
            Value quantile = Value::object();
            for (std::size_t j = 0; j < p.actions.size(); ++j)
                quantile[p.actions[j]] =
                    values(Vec(outputs.begin() + j * p.quantiles,
                               outputs.begin() + (j + 1) * p.quantiles));
            info["return_quantiles"] = quantile;
        }
        info["epsilon"] = p.epsilon;
        info["updates"] = p.updates;
        info["replay_size"] = static_cast<std::int64_t>(p.replay.size());
        if (p.kind != "q_network")
            info["training_metrics"] = p.last_metrics;
    }
    if (activations)
        info["activations"] = activation_layers(tape);
    return info;
}
std::int64_t ValuePolicy::updates() const { return impl_->updates; }
void ValuePolicy::use_world_inputs() { impl_->world_inputs = true; }
Value ValuePolicy::rng_state() const { return impl_->rng.checkpoint(); }
std::string ValuePolicy::trial_action(const Vec &o, Random &rng, Value &memory,
                                      const Value &context) const {
    (void)memory;
    (void)context;
    const auto &p = *impl_;
    if (p.kind != "discrete_sac") {
        if (rng.random() < p.epsilon)
            return p.actions[rng.below(p.actions.size())];
        return p.actions[best_index(p.means(p.network.predict(o)))];
    }
    Vec probabilities, logs;
    categorical_distribution(p.network.predict(o), probabilities, logs);
    double draw = rng.random(), cumulative = 0;
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        cumulative += probabilities[i];
        if (draw < cumulative)
            return p.actions[i];
    }
    return p.actions.back();
}
} // namespace wtfcore
