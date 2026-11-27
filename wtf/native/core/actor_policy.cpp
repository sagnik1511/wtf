#include "actor_policy.hpp"
#include <functional>
#include <numeric>
namespace wtfcore {
namespace {
using Vector = std::vector<double>;
using Gradient = std::vector<Vector>;
struct Sample {
    Vector observation;
    int action = 0;
    double reward = 0, logp = 0, value = 0, next_value = 0;
    bool terminated = false, truncated = false;
    Vector hidden;
};
void increment(std::int64_t &n) {
    if (n == std::numeric_limits<std::int64_t>::max())
        throw std::runtime_error("Native actor counter capacity exceeded");
    ++n;
}
double total(const Vector &v) {
    Sum s;
    for (double x : v)
        s.add(x);
    return s.result();
}
double dot(const Vector &a, const Vector &b) {
    if (a.size() != b.size())
        throw std::runtime_error("Actor dot width mismatch");
    Sum s;
    for (std::size_t i = 0; i < a.size(); ++i)
        s.add(a[i] * b[i]);
    return s.result();
}
void normalize(Vector &v) {
    if (v.size() <= 1)
        return;
    double mean = total(v) / v.size();
    Sum s;
    for (double x : v)
        s.add(std::pow(x - mean, 2.0));
    double scale = std::sqrt(s.result() / v.size() + 1e-8);
    for (auto &x : v)
        x = (x - mean) / scale;
}
std::size_t sample(const Vector &p, Random &r) {
    double draw = r.random(), cumulative = 0;
    std::size_t a = p.size() - 1;
    for (std::size_t i = 0; i < p.size(); ++i) {
        cumulative += p[i];
        if (draw < cumulative) {
            a = i;
            break;
        }
    }
    return a;
}
Vector logits_of(const Vector &out, bool value_head) {
    return value_head ? Vector(out.begin(), out.end() - 1) : out;
}
struct ActorLoss {
    Vector gradient;
    double policy = 0, entropy = 0, kl = 0, reference = 0;
};
ActorLoss actor_loss(const Vector &logits, int action, double advantage,
                     double entropy_coefficient,
                     const double *old_logp = nullptr, double clip = 0.2,
                     const Vector *reference = nullptr,
                     double reference_coefficient = 0) {
    Vector p, logs;
    categorical_distribution(logits, p, logs);
    ActorLoss r;
    r.entropy = -dot(p, logs);
    double derivative = -advantage;
    if (!old_logp)
        r.policy = -logs[action] * advantage;
    else {
        double lr = std::fmax(-60.0, std::fmin(60.0, logs[action] - *old_logp)),
               ratio = std::exp(lr),
               clipped = std::fmax(1 - clip, std::fmin(1 + clip, ratio));
        r.policy = -std::fmin(ratio * advantage, clipped * advantage);
        bool inactive = (advantage > 0 && ratio > 1 + clip) ||
                        (advantage < 0 && ratio < 1 - clip);
        derivative = inactive ? 0.0 : -advantage * ratio;
        r.kl = (ratio - 1) - lr;
    }
    r.gradient.resize(p.size());
    for (std::size_t i = 0; i < p.size(); ++i)
        r.gradient[i] = derivative * ((int(i) == action ? 1.0 : 0.0) - p[i]) +
                        entropy_coefficient * p[i] * (logs[i] + r.entropy);
    if (reference && reference_coefficient) {
        Vector rp, rl;
        categorical_distribution(*reference, rp, rl);
        Vector difference(p.size());
        Sum s;
        for (std::size_t i = 0; i < p.size(); ++i) {
            difference[i] = logs[i] - rl[i];
            s.add(p[i] * difference[i]);
        }
        r.reference = s.result();
        for (std::size_t i = 0; i < p.size(); ++i)
            r.gradient[i] +=
                reference_coefficient * p[i] * (difference[i] - r.reference);
    }
    return r;
}
Gradient zeros(const std::vector<std::vector<float> *> &blocks) {
    Gradient g;
    for (auto *p : blocks)
        g.emplace_back(p->size(), 0);
    return g;
}
void clear(Gradient &g) {
    for (auto &v : g)
        std::fill(v.begin(), v.end(), 0.0);
}
Vector backward(const DenseNetwork &network, const DenseNetwork::Tape &tape,
                Vector delta, Gradient &g, std::size_t offset = 0,
                bool input = false) {
    for (std::size_t j = network.layers.size(); j-- > 0;) {
        const auto &l = network.layers[j];
        const auto &x = tape[j];
        Vector previous((j || input) ? l.inputs : 0, 0);
        auto &dw = g[offset + 2 * j], &db = g[offset + 2 * j + 1];
        for (std::size_t k = 0; k < l.bias.size(); ++k) {
            double d = delta[k];
            db[k] += d;
            for (std::size_t h = 0; h < l.inputs; ++h) {
                dw[k * l.inputs + h] += d * x[h];
                if (j || input)
                    previous[h] +=
                        d * static_cast<double>(l.weights[k * l.inputs + h]);
            }
        }
        if (!j)
            return previous;
        delta.resize(l.inputs);
        for (std::size_t h = 0; h < l.inputs; ++h)
            delta[h] = previous[h] * (1 - x[h] * x[h]);
    }
    return {};
}
std::vector<std::vector<float> *> dense_blocks(DenseNetwork &network) {
    std::vector<std::vector<float> *> out;
    for (auto &l : network.layers) {
        out.push_back(&l.weights);
        out.push_back(&l.bias);
    }
    return out;
}
Value slice(const Value &v, std::size_t start, std::size_t stop) {
    Value r = Value::array();
    for (std::size_t i = start; i < stop; ++i)
        r.push(v.items().at(i));
    return r;
}
void append(Value &to, const Value &from) {
    for (const auto &v : from.items())
        to.push(v);
}
} // namespace
struct ActorPolicy::Impl {
    Value base, config, last_metrics, moment_layout, variance_layout,
        pending_imported;
    std::vector<std::pair<std::size_t, std::string>> block_keys;
    std::vector<std::size_t> parameter_order;
    std::vector<std::string> recurrent_order;
    std::string kind;
    std::vector<std::string> actions;
    DenseNetwork network, critic, reference;
    std::vector<std::vector<float>> moments, variances;
    std::vector<Sample> rollout;
    Sample pending;
    bool pending_valid = false, world_inputs = false;
    std::int64_t steps = 0, update_count = 0, rollout_updates = 0,
                 last_group_step = 0, policy_updates = 0;
    double baseline = 0;
    Random rng, training_rng;
    std::unique_ptr<Random> group_generator;
    Vector hidden;
    bool recurrent = false, lstm = false, bundle = false, value_head = false;
    std::size_t h = 0, inputs = 0;
    std::vector<float> wx, wh, bias;
    struct RecurrentTape {
        Vector x, old, a, b, c, d, e;
        DenseNetwork::Tape head;
    };
    struct Prediction {
        Vector output, memory;
        DenseNetwork::Tape dense;
        RecurrentTape recurrent;
    };
    explicit Impl(const Value &s)
        : base(s), config(s.at("config")), last_metrics(s.at("last_metrics")),
          moment_layout(s.at("moments")), variance_layout(s.at("variances")),
          kind(s.text("algorithm")), steps(s.at("steps").i64()),
          update_count(s.at("updates").i64()),
          rollout_updates(s.at("rollout_updates").i64()) {
        if (s.contains("auxiliary") || s.contains("late"))
            throw std::runtime_error(
                "Native actor adapters require the augmented PPO family");
        if (s.text("precision") != "float32")
            throw std::runtime_error(
                "Native actor requires FP32 stored arrays");
        if (steps < 0 || update_count < 0 || rollout_updates < 0)
            throw std::runtime_error("Negative native actor counter");
        if (kind != "a2c" && kind != "reinforce" && kind != "grpo" &&
            kind != "trpo" && kind != "recurrent_ppo" && kind != "lstm_ppo")
            throw std::runtime_error("Unsupported native actor policy");
        for (const auto &a : s.at("actions").items())
            actions.push_back(a.str());
        if (actions.empty())
            throw std::runtime_error("Empty native actor actions");
        recurrent = kind == "recurrent_ppo" || kind == "lstm_ppo";
        lstm = kind == "lstm_ppo";
        bundle = kind == "trpo";
        value_head = kind == "a2c" || recurrent;
        const auto &net = s.at("network");
        if (recurrent) {
            if (net.array_size() < 2 || config.integer("recurrent_size") < 1)
                throw std::runtime_error("Invalid recurrent actor layout");
            h = config.integer("recurrent_size");
            wx = dense_floats(net.items()[0].at("input_weights"));
            wh = dense_floats(net.items()[0].at("recurrent_weights"));
            bias = dense_floats(net.items()[0].at("bias"));
            std::size_t gates = lstm ? 4 : 3;
            if (wx.empty() || bias.size() != gates * h ||
                wh.size() != gates * h * h || wx.size() % (gates * h))
                throw std::runtime_error("Invalid recurrent actor shape");
            inputs = wx.size() / (gates * h);
            network = DenseNetwork(slice(net, 1, net.items().size()));
            if (network.layers.front().inputs != h)
                throw std::runtime_error("Recurrent actor head width mismatch");
            hidden = numbers(s.at("hidden"));
            if (hidden.size() != (lstm ? 2 * h : h))
                throw std::runtime_error("Invalid recurrent actor memory");
        } else if (bundle) {
            std::size_t block =
                (config.at("network").at("hidden_layers").items().size() + 1);
            if (net.items().size() != 2 * block)
                throw std::runtime_error("Invalid TRPO network bundle");
            network = DenseNetwork(slice(net, 0, block));
            critic = DenseNetwork(slice(net, block, 2 * block));
            policy_updates = s.at("policy_updates").i64();
            if (critic.layers.front().inputs != network.layers.front().inputs ||
                critic.layers.back().bias.size() != 1)
                throw std::runtime_error("Native TRPO critic shape mismatch");
        } else
            network = DenseNetwork(net);
        if (!recurrent)
            inputs = network.layers.front().inputs;
        if (network.layers.back().bias.size() !=
            actions.size() + std::size_t(value_head))
            throw std::runtime_error("Native actor output shape mismatch");
        if (kind == "grpo") {
            reference = DenseNetwork(s.at("reference_network"));
            if (reference.layers.size() != network.layers.size())
                throw std::runtime_error(
                    "Native GRPO reference shape mismatch");
            for (std::size_t j = 0; j < network.layers.size(); ++j)
                if (reference.layers[j].inputs != network.layers[j].inputs ||
                    reference.layers[j].bias.size() !=
                        network.layers[j].bias.size())
                    throw std::runtime_error(
                        "Native GRPO reference shape mismatch");
            group_generator = std::make_unique<Random>(0);
            group_generator->restore(s.at("group_rng_state"));
            last_group_step = s.at("last_group_step").i64();
        }
        if (s.contains("baseline"))
            baseline = s.num("baseline");
        rng.restore(s.at("rng_state"));
        training_rng.restore(s.at("training_rng_state"));
        auto blocks = parameters();
        for (std::size_t layer = 0; layer < net.array_size(); ++layer) {
            std::vector<std::string> keys =
                recurrent && layer == 0
                    ? std::vector<std::string>{"input_weights",
                                               "recurrent_weights", "bias"}
                    : std::vector<std::string>{"weights", "bias"};
            std::size_t offset = block_keys.size();
            for (const auto &key : keys) {
                block_keys.emplace_back(layer, key);
                moments.push_back(
                    dense_floats(s.at("moments").items().at(layer).at(key)));
                variances.push_back(
                    dense_floats(s.at("variances").items().at(layer).at(key)));
            }
            for (const auto &[key, ignored] : net.items()[layer].fields()) {
                (void)ignored;
                auto position = std::find(keys.begin(), keys.end(), key);
                if (position == keys.end())
                    throw std::runtime_error("Unexpected actor parameter key");
                parameter_order.push_back(offset +
                                          std::size_t(position - keys.begin()));
                if (recurrent && layer == 0)
                    recurrent_order.push_back(key);
            }
        }
        std::size_t k = moments.size();
        if (k != blocks.size() || variances.size() != k)
            throw std::runtime_error("Native actor optimizer block mismatch");
        for (k = 0; k < blocks.size(); ++k)
            if (moments[k].size() != blocks[k]->size() ||
                variances[k].size() != blocks[k]->size())
                throw std::runtime_error(
                    "Native actor optimizer shape mismatch");
        const char *buffer =
            (kind == "reinforce" || kind == "grpo") ? "trajectory" : "rollout";
        for (const auto &v : s.at(buffer).items())
            rollout.push_back(read_sample(v));
        if (!s.get("pending").null()) {
            pending_imported = s.at("pending");
            const auto &p = pending_imported.items();
            std::size_t width = (kind == "reinforce" || kind == "grpo")
                                    ? 3
                                    : (recurrent ? 5 : 4);
            if (p.size() != width)
                throw std::runtime_error("Invalid native actor pending record");
            pending.observation = numbers(p[0]);
            pending.action = p[1].as_int();
            pending.logp = p[2].number();
            if (width >= 4)
                pending.value = p[3].number();
            if (width == 5)
                pending.hidden = numbers(p[4]);
            validate(pending);
            pending_valid = true;
        }
        for (auto *layout : {&moment_layout, &variance_layout})
            for (auto &layer : layout->items())
                for (auto &[key, values] : layer.fields()) {
                    (void)key;
                    values = Value{};
                }
        if (kind != config.text("type"))
            throw std::runtime_error("Native actor algorithm/config mismatch");
        if (kind == "grpo") {
            if (config.integer("batch_size") < 1 ||
                config.integer("epochs") < 1 ||
                config.integer("group_size") < 2 ||
                config.integer("rollout_horizon") < 1 ||
                config.integer("train_every_steps") < 1 ||
                last_group_step < 0 || last_group_step > steps)
                throw std::runtime_error("Invalid native GRPO schedule");
        } else if (kind == "reinforce") {
            if (config.integer("max_episode_steps") < 1)
                throw std::runtime_error("Invalid native REINFORCE schedule");
        } else if (config.integer("rollout_steps") < 1 ||
                   (kind != "a2c" && config.integer("batch_size") < 1) ||
                   (recurrent && (config.integer("sequence_length") < 1 ||
                                  config.integer("epochs") < 1)) ||
                   (bundle && (config.integer("value_epochs") < 1 ||
                               config.integer("cg_iterations") < 1 ||
                               config.integer("backtrack_steps") < 1 ||
                               policy_updates < 0))) {
            throw std::runtime_error("Invalid native actor update schedule");
        }
        for (const char *key :
             {"network", "moments", "variances", "rollout", "trajectory",
              "pending", "reference_network", "rng_state", "training_rng_state",
              "group_rng_state", "last_metrics", "hidden"})
            if (base.contains(key))
                base[key] = Value{};
    }
    std::vector<std::vector<float> *> parameters() {
        std::vector<std::vector<float> *> p;
        if (recurrent) {
            p = {&wx, &wh, &bias};
        }
        auto d = dense_blocks(network);
        p.insert(p.end(), d.begin(), d.end());
        if (bundle) {
            d = dense_blocks(critic);
            p.insert(p.end(), d.begin(), d.end());
        }
        return p;
    }
    void validate(const Sample &s) const {
        if (s.observation.size() != inputs || s.action < 0 ||
            std::size_t(s.action) >= actions.size() ||
            (recurrent && s.hidden.size() != (lstm ? 2 * h : h)))
            throw std::runtime_error("Invalid native actor sample");
    }
    Sample read_sample(const Value &v) {
        Sample s;
        s.observation = numbers(v.at("observation"));
        s.action = v.integer("action");
        s.reward = v.num("reward");
        s.logp = v.num("log_probability");
        s.terminated = v.flag("terminated");
        s.truncated = v.flag("truncated");
        if (value_head || bundle) {
            s.value = v.num("value");
            s.next_value = v.num("next_value");
        }
        if (recurrent)
            s.hidden = numbers(v.at("hidden"));
        validate(s);
        return s;
    }
    Value observation_value(const Vector &v) const {
        return world_observation_values(
            v, config.integer("observation_version", 1), world_inputs);
    }
    Value sample_value(const Sample &s) const {
        Value v = object({{"observation", observation_value(s.observation)},
                          {"action", s.action},
                          {"reward", s.reward},
                          {"log_probability", s.logp}});
        if (value_head || bundle) {
            v["value"] = s.value;
            v["next_value"] = s.next_value;
        }
        v["terminated"] = s.terminated;
        v["truncated"] = s.truncated;
        if (recurrent)
            v["hidden"] = values(s.hidden);
        return v;
    }
    Value optimizer_value(const std::vector<std::vector<float>> &blocks,
                          const Value &layout) const {
        Value out = layout;
        for (std::size_t k = 0; k < block_keys.size(); ++k)
            out.items().at(block_keys[k].first)[block_keys[k].second] =
                Value::float_array(blocks.at(k));
        return out;
    }
    Value network_value() const {
        Value out = Value::array();
        if (recurrent) {
            Value row = Value::object();
            for (const auto &key : recurrent_order)
                row[key] =
                    Value::float_array(key == "input_weights"       ? wx
                                       : key == "recurrent_weights" ? wh
                                                                    : bias);
            out.push(std::move(row));
        }
        append(out, network.checkpoint());
        if (bundle)
            append(out, critic.checkpoint());
        return out;
    }
    void adam(const Gradient &gradient, std::size_t count) {
        auto p = parameters();
        if (gradient.size() != p.size() || !count)
            throw std::runtime_error("Invalid native actor gradient blocks");
        Sum normsum;
        for (auto block : parameter_order)
            for (double x : gradient[block])
                normsum.add(std::pow(x / static_cast<double>(count), 2.0));
        double norm = std::sqrt(normsum.result()),
               factor = std::fmin(1.0, config.num("max_grad_norm") /
                                           (norm + 1e-12)) /
                        count;
        increment(update_count);
        double c1 = 1 - std::pow(0.9, double(update_count)),
               c2 = 1 - std::pow(0.999, double(update_count)),
               rate = config.num("learning_rate");
        for (std::size_t j = 0; j < p.size(); ++j)
            for (std::size_t k = 0; k < p[j]->size(); ++k) {
                double g = gradient[j][k] * factor;
                auto &m = moments[j][k];
                auto &v = variances[j][k];
                m = float(0.9 * double(m) + 0.1 * g);
                v = float(0.999 * double(v) + 0.001 * g * g);
                (*p[j])[k] = float(double((*p[j])[k]) -
                                   rate * (double(m) / c1) /
                                       (std::sqrt(double(v) / c2) + 1e-8));
            }
    }
    void advantages(Vector &adv, Vector &targets) {
        adv.resize(rollout.size());
        targets.resize(rollout.size());
        double trace = 0;
        for (std::size_t i = rollout.size(); i-- > 0;) {
            auto &s = rollout[i];
            double bootstrap = s.terminated ? 0 : s.next_value,
                   delta = s.reward +
                           config.num("discount_factor") * bootstrap - s.value,
                   continuation = s.terminated || s.truncated ? 0 : 1;
            trace = delta + config.num("discount_factor") *
                                config.num("gae_lambda") * continuation * trace;
            adv[i] = trace;
            targets[i] = trace + s.value;
        }
        if (config.flag("normalize_advantages"))
            normalize(adv);
    }
    Prediction predict(const Vector &obs, const Vector *memory = nullptr,
                       bool value_only = false, bool tape = false);
    Vector recurrent_backward(const RecurrentTape &tape,
                              const Vector &output_gradient,
                              const Vector &future, Gradient &gradient);
    void update_a2c() {
        if (rollout_updates == std::numeric_limits<std::int64_t>::max())
            throw std::runtime_error(
                "Native rollout counter capacity exceeded");
        Vector adv, targets;
        advantages(adv, targets);
        auto g = zeros(parameters());
        double pl = 0, vl = 0, ent = 0;
        for (std::size_t i = 0; i < rollout.size(); ++i) {
            auto tape = network.forward(rollout[i].observation);
            auto out = tape.back();
            auto a = actor_loss(logits_of(out, true), rollout[i].action, adv[i],
                                config.num("entropy_coefficient"));
            double error = out.back() - targets[i];
            a.gradient.push_back(config.num("value_coefficient") * error);
            backward(network, tape, a.gradient, g);
            pl += a.policy;
            ent += a.entropy;
            vl += 0.5 * error * error;
        }
        adam(g, rollout.size());
        last_metrics = object({{"policy_loss", pl / rollout.size()},
                               {"value_loss", vl / rollout.size()},
                               {"entropy", ent / rollout.size()}});
        increment(rollout_updates);
        rollout.clear();
    }
    void update_reinforce() {
        if (rollout_updates == std::numeric_limits<std::int64_t>::max())
            throw std::runtime_error(
                "Native rollout counter capacity exceeded");
        Vector returns(rollout.size()), adv;
        double r = 0;
        for (std::size_t i = rollout.size(); i-- > 0;) {
            r = rollout[i].reward + config.num("discount_factor") * r;
            returns[i] = r;
        }
        double mean = total(returns) / returns.size();
        adv = returns;
        for (std::size_t i = 0; i < adv.size(); ++i)
            adv[i] = std::pow(config.num("discount_factor"), double(i)) *
                     (adv[i] - baseline);
        if (config.flag("normalize_advantages"))
            normalize(adv);
        auto g = zeros(parameters());
        double pl = 0, ent = 0;
        for (std::size_t i = 0; i < rollout.size(); ++i) {
            auto tape = network.forward(rollout[i].observation);
            auto a = actor_loss(tape.back(), rollout[i].action, adv[i],
                                config.num("entropy_coefficient"));
            backward(network, tape, a.gradient, g);
            pl += a.policy;
            ent += a.entropy;
        }
        std::size_t n = rollout.size();
        adam(g, 1);
        double decay = config.num("baseline_decay");
        baseline = decay * baseline + (1 - decay) * mean;
        last_metrics = object({{"policy_loss", pl / n},
                               {"entropy", ent / n},
                               {"mean_return", mean}});
        increment(rollout_updates);
        rollout.clear();
    }
    void update_group(const Value &trajectories) {
        if (rollout_updates == std::numeric_limits<std::int64_t>::max())
            throw std::runtime_error("Native group counter capacity exceeded");
        std::vector<std::vector<Sample>> group;
        Vector returns;
        for (const auto &t : trajectories.items()) {
            if (t.items().empty())
                throw std::runtime_error("Empty native GRPO trajectory");
            std::vector<Sample> row;
            Sum sum;
            std::size_t i = 0;
            for (const auto &v : t.items()) {
                auto s = read_sample(v);
                sum.add(std::pow(config.num("discount_factor"), double(i++)) *
                        s.reward);
                row.push_back(std::move(s));
            }
            returns.push_back(sum.result());
            group.push_back(std::move(row));
        }
        if (group.size() < 2)
            throw std::runtime_error("GRPO requires matched groups");
        double mean = total(returns) / returns.size();
        Sum squares;
        for (double r : returns)
            squares.add(std::pow(r - mean, 2.0));
        double deviation = std::sqrt(squares.result() / returns.size());
        last_group_step = steps;
        increment(rollout_updates);
        last_metrics = object({{"group_returns", values(returns)},
                               {"mean_return", mean},
                               {"return_std", deviation},
                               {"zero_variance", deviation < 1e-8},
                               {"early_stopped", false}});
        if (deviation < 1e-8)
            return;
        std::size_t count = 0;
        for (const auto &t : group)
            count += t.size();
        struct Weighted {
            const Sample *sample;
            double advantage, weight;
        };
        std::vector<Weighted> samples;
        for (std::size_t i = 0; i < group.size(); ++i) {
            double advantage = (returns[i] - mean) / (deviation + 1e-8),
                   weight = double(count) / (group.size() * group[i].size());
            for (const auto &s : group[i])
                samples.push_back({&s, advantage, weight});
        }
        std::vector<std::size_t> indices(count);
        std::iota(indices.begin(), indices.end(), 0);
        auto g = zeros(parameters());
        std::size_t size = config.integer("batch_size");
        for (int epoch = 0; epoch < config.integer("epochs"); ++epoch) {
            training_rng.shuffle(indices);
            for (std::size_t offset = 0; offset < count; offset += size) {
                clear(g);
                double kl = 0;
                std::size_t end = std::min(offset + size, count);
                for (std::size_t i = offset; i < end; ++i) {
                    const auto &w = samples[indices[i]];
                    const auto &s = *w.sample;
                    auto tape = network.forward(s.observation);
                    Vector ref;
                    if (config.num("kl_coefficient"))
                        ref = reference.predict(s.observation);
                    auto a = actor_loss(tape.back(), s.action, w.advantage,
                                        config.num("entropy_coefficient"),
                                        &s.logp, config.num("clip_ratio"),
                                        ref.empty() ? nullptr : &ref,
                                        config.num("kl_coefficient"));
                    for (auto &x : a.gradient)
                        x *= w.weight;
                    backward(network, tape, a.gradient, g);
                    kl += a.kl;
                }
                double gate = config.num("target_kl", 0.03);
                if (gate && kl / (end - offset) > gate) {
                    last_metrics["early_stopped"] = true;
                    return;
                }
                adam(g, end - offset);
            }
        }
    }
    void update_trpo();
    void update_recurrent();
    Value checkpoint() {
        Value s = base;
        s["network"] = network_value();
        s["moments"] = optimizer_value(moments, moment_layout);
        s["variances"] = optimizer_value(variances, variance_layout);
        Value samples = Value::array();
        for (const auto &r : rollout)
            samples.push(sample_value(r));
        s[(kind == "reinforce" || kind == "grpo") ? "trajectory" : "rollout"] =
            samples;
        s["pending"] = Value{};
        if (pending_valid) {
            if (!pending_imported.null())
                s["pending"] = pending_imported;
            else {
                Value p = array({observation_value(pending.observation),
                                 pending.action, pending.logp});
                if (value_head || bundle)
                    p.push(pending.value);
                if (recurrent)
                    p.push(values(pending.hidden));
                s["pending"] = p;
            }
        }
        s["steps"] = steps;
        s["updates"] = update_count;
        s["rollout_updates"] = rollout_updates;
        s["last_metrics"] = last_metrics;
        s["rng_state"] = rng.checkpoint();
        s["training_rng_state"] = training_rng.checkpoint();
        if (s.contains("baseline"))
            s["baseline"] = baseline;
        if (kind == "grpo") {
            s["reference_network"] = reference.checkpoint();
            s["group_rng_state"] = group_generator->checkpoint();
            s["last_group_step"] = last_group_step;
        }
        if (bundle)
            s["policy_updates"] = policy_updates;
        if (recurrent)
            s["hidden"] = values(hidden);
        return s;
    }
};
ActorPolicy::Impl::Prediction ActorPolicy::Impl::predict(const Vector &obs,
                                                         const Vector *memory,
                                                         bool value_only,
                                                         bool tape) {
    (void)tape;
    Prediction out;
    if (!recurrent && value_only) {
        out.output = {bundle ? critic.value(obs) : network.value(obs)};
        return out;
    }
    if (!recurrent) {
        out.dense = network.forward(obs);
        out.output = out.dense.back();
        if (bundle)
            out.output.push_back(critic.value(obs));
        else if (value_only)
            out.output = {out.output.back()};
        return out;
    }
    const auto &old = memory ? *memory : hidden;
    if (obs.size() != inputs || old.size() != (lstm ? 2 * h : h))
        throw std::runtime_error("Invalid recurrent input shape");
    auto &t = out.recurrent;
    t.x = obs;
    t.old = old;
    auto gate = [&](std::size_t group, const Vector &prior, bool sigmoid) {
        Vector r(h);
        for (std::size_t j = 0; j < h; ++j) {
            auto row = group * h + j;
            Sum sx, sh;
            for (std::size_t i = 0; i < inputs; ++i)
                sx.add(obs[i] * double(wx[row * inputs + i]));
            for (std::size_t i = 0; i < h; ++i)
                sh.add(prior[i] * double(wh[row * h + i]));
            double v = (double(bias[row]) + sx.result()) + sh.result();
            r[j] = sigmoid ? (v >= 0 ? 1.0 / (1.0 + std::exp(-v))
                                     : std::exp(v) / (1.0 + std::exp(v)))
                           : std::tanh(v);
        }
        return r;
    };
    Vector prior(old.begin(), old.begin() + h), current(h);
    if (lstm) {
        t.a = gate(0, prior, true);
        t.b = gate(1, prior, true);
        t.c = gate(2, prior, false);
        t.d = gate(3, prior, true);
        t.e.resize(h);
        for (std::size_t j = 0; j < h; ++j) {
            t.e[j] = t.b[j] * old[h + j] + t.a[j] * t.c[j];
            current[j] = t.d[j] * std::tanh(t.e[j]);
        }
        out.memory = current;
        out.memory.insert(out.memory.end(), t.e.begin(), t.e.end());
    } else {
        t.a = gate(0, prior, true);
        t.b = gate(1, prior, true);
        t.d.resize(h);
        for (std::size_t j = 0; j < h; ++j)
            t.d[j] = t.b[j] * old[j];
        t.c = gate(2, t.d, false);
        for (std::size_t j = 0; j < h; ++j)
            current[j] = (1 - t.a[j]) * t.c[j] + t.a[j] * old[j];
        out.memory = current;
    }
    if (value_only)
        out.output = {network.value(current)};
    else {
        out.dense = network.forward(current);
        out.output = out.dense.back();
        t.head = out.dense;
    }
    return out;
}
Vector ActorPolicy::Impl::recurrent_backward(const RecurrentTape &t,
                                             const Vector &output_gradient,
                                             const Vector &future,
                                             Gradient &g) {
    Vector derivative = backward(network, t.head, output_gradient, g, 3, true),
           previous(lstm ? 2 * h : h, 0.0), dh(h);
    for (std::size_t i = 0; i < h; ++i)
        dh[i] = derivative[i] + future[i];
    if (lstm) {
        Vector tc(h), dc(h), delta(4 * h);
        for (std::size_t j = 0; j < h; ++j) {
            tc[j] = std::tanh(t.e[j]);
            dc[j] = future[h + j] + dh[j] * t.d[j] * (1 - std::pow(tc[j], 2.0));
            delta[j] = dc[j] * t.c[j] * t.a[j] * (1 - t.a[j]);
            delta[h + j] = dc[j] * t.old[h + j] * t.b[j] * (1 - t.b[j]);
            delta[2 * h + j] = dc[j] * t.a[j] * (1 - std::pow(t.c[j], 2.0));
            delta[3 * h + j] = dh[j] * tc[j] * t.d[j] * (1 - t.d[j]);
            previous[h + j] = dc[j] * t.b[j];
        }
        for (std::size_t i = 0; i < h; ++i) {
            Sum sum;
            for (std::size_t gate = 0; gate < 4; ++gate)
                for (std::size_t j = 0; j < h; ++j)
                    sum.add(delta[gate * h + j] *
                            double(wh[(gate * h + j) * h + i]));
            previous[i] = sum.result();
        }
        for (std::size_t gate = 0; gate < 4; ++gate)
            for (std::size_t j = 0; j < h; ++j) {
                std::size_t row = gate * h + j;
                double d = delta[row];
                g[2][row] += d;
                for (std::size_t i = 0; i < inputs; ++i)
                    g[0][row * inputs + i] += d * t.x[i];
                for (std::size_t i = 0; i < h; ++i)
                    g[1][row * h + i] += d * t.old[i];
            }
    } else {
        Vector dz(h), dn(h), drh(h), dr(h);
        for (std::size_t i = 0; i < h; ++i) {
            dz[i] = dh[i] * (t.old[i] - t.c[i]) * t.a[i] * (1 - t.a[i]);
            dn[i] = dh[i] * (1 - t.a[i]) * (1 - t.c[i] * t.c[i]);
        }
        for (std::size_t i = 0; i < h; ++i) {
            Sum sum;
            for (std::size_t j = 0; j < h; ++j)
                sum.add(dn[j] * double(wh[(2 * h + j) * h + i]));
            drh[i] = sum.result();
            dr[i] = drh[i] * t.old[i] * t.b[i] * (1 - t.b[i]);
        }
        for (std::size_t i = 0; i < h; ++i) {
            Sum sum;
            for (std::size_t j = 0; j < h; ++j)
                sum.add(dz[j] * double(wh[j * h + i]) +
                        dr[j] * double(wh[(h + j) * h + i]));
            previous[i] = (dh[i] * t.a[i] + drh[i] * t.b[i]) + sum.result();
        }
        for (std::size_t gate = 0; gate < 3; ++gate) {
            const Vector &delta = gate == 0 ? dz : (gate == 1 ? dr : dn);
            const Vector &old = gate == 2 ? t.d : t.old;
            for (std::size_t j = 0; j < h; ++j) {
                std::size_t row = gate * h + j;
                double d = delta[j];
                g[2][row] += d;
                for (std::size_t i = 0; i < inputs; ++i)
                    g[0][row * inputs + i] += d * t.x[i];
                for (std::size_t i = 0; i < h; ++i)
                    g[1][row * h + i] += d * old[i];
            }
        }
    }
    return previous;
}
void ActorPolicy::Impl::update_recurrent() {
    if (rollout_updates == std::numeric_limits<std::int64_t>::max())
        throw std::runtime_error("Native rollout counter capacity exceeded");
    Vector adv, targets;
    advantages(adv, targets);
    std::size_t length = config.integer("sequence_length"),
                per_batch = std::max<std::size_t>(
                    1, config.integer("batch_size") / length);
    std::vector<std::vector<std::size_t>> sequences;
    for (std::size_t i = 0; i < rollout.size(); i += length) {
        std::vector<std::size_t> s;
        for (std::size_t j = i; j < std::min(i + length, rollout.size()); ++j)
            s.push_back(j);
        sequences.push_back(std::move(s));
    }
    auto g = zeros(parameters());
    std::array<double, 5> totals{};
    std::size_t samples = 0;
    bool stopped = false;
    const char *names[] = {"policy_loss", "value_loss", "entropy",
                           "approximate_kl", "clip_fraction"};
    for (int epoch = 0; epoch < config.integer("epochs"); ++epoch) {
        training_rng.shuffle(sequences);
        for (std::size_t offset = 0; offset < sequences.size();
             offset += per_batch) {
            clear(g);
            std::array<double, 5> metrics{};
            std::size_t count = 0;
            for (std::size_t si = offset;
                 si < std::min(offset + per_batch, sequences.size()); ++si) {
                const auto &indices = sequences[si];
                Vector memory = rollout[indices[0]].hidden;
                std::vector<std::pair<RecurrentTape, Vector>> tapes;
                for (auto i : indices) {
                    const auto &s = rollout[i];
                    auto prediction =
                        predict(s.observation, &memory, false, true);
                    memory = prediction.memory;
                    const auto &out = prediction.output;
                    auto a = actor_loss(logits_of(out, true), s.action, adv[i],
                                        config.num("entropy_coefficient"),
                                        &s.logp, config.num("clip_ratio"));
                    double error = out.back() - targets[i];
                    a.gradient.push_back(config.num("value_coefficient") *
                                         error);
                    tapes.push_back({std::move(prediction.recurrent),
                                     std::move(a.gradient)});
                    Vector p, logs;
                    categorical_distribution(logits_of(out, true), p, logs);
                    double lr = std::fmax(
                               -60.0, std::fmin(60.0, logs[s.action] - s.logp)),
                           ratio = std::exp(lr);
                    metrics[0] += a.policy;
                    metrics[1] += 0.5 * error * error;
                    metrics[2] += a.entropy;
                    metrics[3] += a.kl;
                    metrics[4] +=
                        double(std::fabs(ratio - 1) > config.num("clip_ratio"));
                    ++count;
                }
                Vector future(rollout[indices[0]].hidden.size(), 0);
                for (std::size_t i = tapes.size(); i-- > 0;)
                    future = recurrent_backward(tapes[i].first, tapes[i].second,
                                                future, g);
            }
            double gate = config.num("target_kl", 0.03);
            if (gate && metrics[3] / count > gate) {
                stopped = true;
                break;
            }
            adam(g, count);
            samples += count;
            for (std::size_t k = 0; k < 5; ++k)
                totals[k] += metrics[k];
        }
        if (stopped)
            break;
    }
    increment(rollout_updates);
    last_metrics = Value::object();
    if (samples)
        for (std::size_t k = 0; k < 5; ++k)
            last_metrics[names[k]] = totals[k] / samples;
    last_metrics["early_stopped"] = stopped;
    rollout.clear();
}
void ActorPolicy::Impl::update_trpo() {
    if (rollout_updates == std::numeric_limits<std::int64_t>::max())
        throw std::runtime_error("Native rollout counter capacity exceeded");
    Vector adv, targets;
    advantages(adv, targets);
    std::size_t count = rollout.size();
    std::vector<DenseNetwork::Tape> tapes;
    std::vector<Vector> probabilities, oldlogs;
    auto actor_gradient = zeros(dense_blocks(network));
    double before = 0;
    double entropy_coefficient = config.num("entropy_coefficient");
    auto objective = [&](const Vector &logits, const Sample &s,
                         double advantage, Vector *derivative) {
        Vector p, logs;
        categorical_distribution(logits, p, logs);
        double ratio = std::exp(
                   std::fmax(-60.0, std::fmin(60.0, logs[s.action] - s.logp))),
               entropy = -dot(p, logs);
        if (derivative) {
            derivative->resize(p.size());
            for (std::size_t i = 0; i < p.size(); ++i)
                (*derivative)[i] =
                    advantage * ratio *
                        ((int(i) == s.action ? 1.0 : 0.0) - p[i]) -
                    entropy_coefficient * p[i] * (logs[i] + entropy);
        }
        return ratio * advantage + entropy_coefficient * entropy;
    };
    for (std::size_t i = 0; i < count; ++i) {
        auto tape = network.forward(rollout[i].observation);
        Vector p, logs, d;
        categorical_distribution(tape.back(), p, logs);
        before += objective(tape.back(), rollout[i], adv[i], &d);
        backward(network, tape, d, actor_gradient);
        tapes.push_back(std::move(tape));
        probabilities.push_back(std::move(p));
        oldlogs.push_back(std::move(logs));
    }
    before /= count;
    auto flatten = [&](const Gradient &g) {
        Vector v;
        for (std::size_t j = 0; j < network.layers.size(); ++j) {
            std::size_t first = 2 * j + (network.layers[j].bias_first ? 1 : 0);
            std::size_t second = 2 * j + (network.layers[j].bias_first ? 0 : 1);
            v.insert(v.end(), g[first].begin(), g[first].end());
            v.insert(v.end(), g[second].begin(), g[second].end());
        }
        return v;
    };
    Vector gradient = flatten(actor_gradient);
    for (auto &v : gradient)
        v /= count;
    auto fisher = [&](const Vector &direction) {
        auto g = zeros(dense_blocks(network));
        std::vector<std::pair<Vector, Vector>> directions;
        std::size_t cursor = 0;
        for (const auto &layer : network.layers) {
            std::size_t n = layer.weights.size(), b = layer.bias.size();
            // Source split_direction deliberately uses weights/bias slices even
            // when serialized parameter dictionaries have a different order.
            directions.push_back({Vector(direction.begin() + cursor,
                                         direction.begin() + cursor + n),
                                  Vector(direction.begin() + cursor + n,
                                         direction.begin() + cursor + n + b)});
            cursor += n + b;
        }
        for (std::size_t sample = 0; sample < tapes.size(); ++sample) {
            const auto &tape = tapes[sample];
            Vector tangent(tape[0].size(), 0);
            for (std::size_t j = 0; j < network.layers.size(); ++j) {
                const auto &l = network.layers[j];
                const auto &x = tape[j];
                const auto &dw = directions[j].first;
                const auto &db = directions[j].second;
                Vector next(l.bias.size());
                for (std::size_t k = 0; k < next.size(); ++k) {
                    Sum sum;
                    for (std::size_t i = 0; i < x.size(); ++i)
                        sum.add(dw[k * x.size() + i] * x[i] +
                                double(l.weights[k * x.size() + i]) *
                                    tangent[i]);
                    double v = db[k] + sum.result();
                    next[k] = j + 1 < network.layers.size()
                                  ? v * (1 - std::pow(tape[j + 1][k], 2.0))
                                  : v;
                }
                tangent = std::move(next);
            }
            double mean = dot(probabilities[sample], tangent);
            Vector d(tangent.size());
            for (std::size_t i = 0; i < d.size(); ++i)
                d[i] = probabilities[sample][i] * (tangent[i] - mean);
            backward(network, tape, d, g);
        }
        auto out = flatten(g);
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] = out[i] / count + config.num("cg_damping") * direction[i];
        return out;
    };
    Vector direction(gradient.size(), 0), residual = gradient,
                                          search = residual;
    double rr = dot(residual, residual);
    for (int iteration = 0; iteration < config.integer("cg_iterations");
         ++iteration) {
        if (rr <= config.num("cg_tolerance"))
            break;
        auto applied = fisher(search);
        double denominator = dot(search, applied);
        if (denominator <= 0 || !std::isfinite(denominator))
            break;
        double scale = rr / denominator;
        for (std::size_t i = 0; i < direction.size(); ++i) {
            direction[i] = direction[i] + scale * search[i];
            residual[i] = residual[i] - scale * applied[i];
        }
        double following = dot(residual, residual);
        for (std::size_t i = 0; i < search.size(); ++i)
            search[i] = residual[i] + (following / rr) * search[i];
        rr = following;
    }
    double curvature = dot(direction, fisher(direction));
    Vector base_parameters = network.parameters();
    bool accepted = false;
    double kl = 0, improvement = 0;
    if (curvature > 0 && std::isfinite(curvature)) {
        double scale = std::sqrt(2 * config.num("max_kl") / curvature);
        Vector full = direction;
        for (auto &v : full)
            v = scale * v;
        double expected = dot(gradient, full);
        for (int n = 0; n < config.integer("backtrack_steps"); ++n) {
            double fraction =
                std::pow(config.num("backtrack_coefficient"), double(n));
            Vector candidate(base_parameters.size());
            for (std::size_t i = 0; i < candidate.size(); ++i)
                candidate[i] = base_parameters[i] + fraction * full[i];
            network.set_parameters(candidate);
            double value = 0;
            kl = 0;
            for (std::size_t i = 0; i < count; ++i) {
                auto out = network.predict(rollout[i].observation);
                Vector p, logs;
                categorical_distribution(out, p, logs);
                value += objective(out, rollout[i], adv[i], nullptr);
                Sum sum;
                for (std::size_t k = 0; k < p.size(); ++k)
                    sum.add(probabilities[i][k] * (oldlogs[i][k] - logs[k]));
                kl += sum.result();
            }
            kl = std::fmax(0, kl / count);
            improvement = value / count - before;
            if (std::isfinite(improvement) && std::isfinite(kl) &&
                kl <= config.num("max_kl") && improvement > 0 &&
                improvement >=
                    config.num("accept_ratio") * fraction * expected) {
                accepted = true;
                increment(policy_updates);
                break;
            }
        }
        if (!accepted) {
            network.set_parameters(base_parameters);
            kl = 0;
            improvement = 0;
        }
    }
    double loss = 0;
    std::vector<std::size_t> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    std::size_t batch = config.integer("batch_size"),
                critic_offset = 2 * network.layers.size();
    for (int epoch = 0; epoch < config.integer("value_epochs"); ++epoch) {
        training_rng.shuffle(indices);
        loss = 0;
        for (std::size_t offset = 0; offset < count; offset += batch) {
            auto g = zeros(parameters());
            std::size_t end = std::min(offset + batch, count);
            for (std::size_t j = offset; j < end; ++j) {
                std::size_t i = indices[j];
                auto tape = critic.forward(rollout[i].observation);
                double error = tape.back()[0] - targets[i];
                backward(critic, tape,
                         {config.num("value_coefficient") * error}, g,
                         critic_offset);
                loss += 0.5 * error * error;
            }
            adam(g, end - offset);
        }
    }
    last_metrics = object({{"empirical_kl", kl},
                           {"policy_accepted", accepted},
                           {"surrogate_improvement", improvement},
                           {"value_loss", loss / count}});
    increment(rollout_updates);
    rollout.clear();
}
ActorPolicy::ActorPolicy(const Value &s) : impl_(std::make_unique<Impl>(s)) {}
ActorPolicy::~ActorPolicy() = default;
ActorPolicy::ActorPolicy(ActorPolicy &&) noexcept = default;
ActorPolicy &ActorPolicy::operator=(ActorPolicy &&) noexcept = default;
std::string ActorPolicy::choose(const Vector &observation, bool training,
                                bool random_baseline) {
    auto &b = *impl_;
    if (random_baseline && !b.recurrent)
        return b.actions[b.rng.below(b.actions.size())];
    Vector before = b.hidden;
    auto prediction = b.predict(observation, b.recurrent ? &before : nullptr);
    if (b.recurrent)
        b.hidden = prediction.memory;
    Vector p, logs;
    categorical_distribution(
        logits_of(prediction.output, b.value_head || b.bundle), p, logs);
    if (random_baseline)
        return b.actions[b.rng.below(b.actions.size())];
    if (!training)
        return b.actions[std::max_element(p.begin(), p.end()) - p.begin()];
    std::size_t a = sample(p, b.rng);
    b.pending = Sample{};
    b.pending.observation = observation;
    b.pending.action = int(a);
    b.pending.logp = logs[a];
    if (b.value_head || b.bundle)
        b.pending.value = prediction.output.back();
    if (b.recurrent)
        b.pending.hidden = before;
    b.pending_valid = true;
    b.pending_imported = Value{};
    return b.actions[a];
}
void ActorPolicy::learn(const Vector &observation, const std::string &action,
                        double reward, const Vector *next, bool terminated,
                        bool truncated) {
    auto &b = *impl_;
    if (b.steps == std::numeric_limits<std::int64_t>::max())
        throw std::runtime_error("Native actor step counter capacity exceeded");
    if (b.kind == "reinforce" &&
        b.rollout.size() >= std::size_t(b.config.integer("max_episode_steps")))
        throw std::runtime_error("REINFORCE maximum episode steps exceeded");
    if (b.kind == "grpo") {
        increment(b.steps);
        b.pending_valid = false;
        b.pending_imported = Value{};
        return;
    }
    auto found = std::find(b.actions.begin(), b.actions.end(), action);
    if (found == b.actions.end())
        throw std::runtime_error("Unknown native actor action");
    int index = int(found - b.actions.begin());
    bool matches = b.pending_valid && b.pending.action == index &&
                   b.pending.observation == observation;
    if (b.recurrent && !matches)
        throw std::runtime_error("Recurrent PPO requires choose before learn");
    Sample s;
    if (matches)
        s = b.pending;
    else {
        auto prediction = b.predict(observation);
        Vector p, logs;
        categorical_distribution(
            logits_of(prediction.output, b.value_head || b.bundle), p, logs);
        s.observation = observation;
        s.action = index;
        s.logp = logs[index];
        if (b.value_head || b.bundle)
            s.value = prediction.output.back();
    }
    s.reward = reward;
    s.terminated = terminated;
    s.truncated = truncated;
    b.pending_valid = false;
    b.pending_imported = Value{};
    if (b.kind == "reinforce" &&
        b.rollout.size() >= std::size_t(b.config.integer("max_episode_steps")))
        throw std::runtime_error("REINFORCE maximum episode steps exceeded");
    if ((b.value_head || b.bundle) && !terminated) {
        if (!next)
            throw std::runtime_error(
                "Actor critic requires nonterminal next observation");
        s.next_value = b.predict(*next, b.recurrent ? &b.hidden : nullptr, true)
                           .output.back();
    }
    b.rollout.push_back(std::move(s));
    increment(b.steps);
    if (b.kind == "reinforce") {
        if (terminated || truncated)
            b.update_reinforce();
    } else if (b.rollout.size() >=
                   std::size_t(b.config.integer("rollout_steps")) ||
               terminated || truncated) {
        if (b.kind == "a2c")
            b.update_a2c();
        else if (b.kind == "trpo")
            b.update_trpo();
        else
            b.update_recurrent();
    }
    if (b.recurrent && (terminated || truncated))
        std::fill(b.hidden.begin(), b.hidden.end(), 0.0);
}
Value ActorPolicy::checkpoint() { return impl_->checkpoint(); }
Value ActorPolicy::model_state() {
    auto &b = *impl_;
    return object({{"precision", "float32"},
                   {"network", b.network_value()},
                   {"config", b.config},
                   {"updates", b.update_count}});
}
std::int64_t ActorPolicy::updates() const { return impl_->update_count; }
void ActorPolicy::use_world_inputs() { impl_->world_inputs = true; }
void ActorPolicy::set_context(const Value &context) { (void)context; }
bool ActorPolicy::needs_group() const {
    const auto &b = *impl_;
    return b.kind == "grpo" &&
           b.steps - b.last_group_step >= b.config.integer("train_every_steps");
}
void ActorPolicy::update_group(const Value &trajectories) {
    if (impl_->kind != "grpo")
        throw std::runtime_error("Group updates require GRPO");
    impl_->update_group(trajectories);
}
Random &ActorPolicy::group_rng() {
    if (!impl_->group_generator)
        throw std::runtime_error("Group RNG requires GRPO");
    return *impl_->group_generator;
}
Value ActorPolicy::rng_state() const { return impl_->rng.checkpoint(); }
std::vector<double> ActorPolicy::policy_logits(const Vector &observation) {
    auto &b = *impl_;
    auto p = b.predict(observation, b.recurrent ? &b.hidden : nullptr);
    return logits_of(p.output, b.value_head || b.bundle);
}
Value ActorPolicy::trial_memory() const { return values(impl_->hidden); }
std::string ActorPolicy::trial_action(const Vector &observation,
                                      Random &external_rng, Value &memory,
                                      const Value &context) {
    (void)context;
    auto &b = *impl_;
    Vector prior =
        b.recurrent ? (memory.null() ? b.hidden : numbers(memory)) : Vector{};
    auto prediction = b.predict(observation, b.recurrent ? &prior : nullptr);
    if (b.recurrent)
        memory = values(prediction.memory);
    Vector p, logs;
    categorical_distribution(
        logits_of(prediction.output, b.value_head || b.bundle), p, logs);
    return b.actions[sample(p, external_rng)];
}
Value ActorPolicy::diagnostics(const Vector &observation, bool activations) {
    auto &b = *impl_;
    auto prediction = b.predict(observation, b.recurrent ? &b.hidden : nullptr,
                                false, activations);
    Vector p, logs;
    categorical_distribution(
        logits_of(prediction.output, b.value_head || b.bundle), p, logs);
    Value probabilities = Value::object();
    for (std::size_t i = 0; i < b.actions.size(); ++i)
        probabilities[b.actions[i]] = p[i];
    double entropy = -dot(p, logs);
    Value info;
    if (b.kind == "reinforce" || b.kind == "grpo") {
        info = object(
            {{"algorithm", b.kind},
             {"features", b.observation_value(observation)},
             {"action_probabilities", probabilities},
             {"entropy", entropy},
             {"updates", b.update_count},
             {"rollout_updates", b.rollout_updates},
             {"buffer_kind", b.kind == "reinforce" ? "episode" : "group"},
             {"buffer_size", std::int64_t(b.rollout.size())},
             {"training_metrics", b.last_metrics}});
        if (b.kind == "reinforce")
            info["return_baseline"] = b.baseline;
        else {
            info["group_size"] = b.config.at("group_size");
            info["rollout_horizon"] = b.config.at("rollout_horizon");
        }
    } else if (b.recurrent) {
        info = object({{"algorithm", b.kind},
                       {"features", b.observation_value(observation)},
                       {"action_probabilities", probabilities},
                       {"state_value", prediction.output.back()},
                       {"entropy", entropy},
                       {"updates", b.update_count},
                       {"rollout_updates", b.rollout_updates},
                       {"buffer_kind", "sequence"},
                       {"buffer_size", std::int64_t(b.rollout.size())},
                       {"recurrent_size", int(b.h)}});
        if (b.lstm)
            info["recurrent_cell"] = "lstm";
        Sum norm;
        for (std::size_t i = 0; i < b.h; ++i)
            norm.add(b.hidden[i] * b.hidden[i]);
        info["memory_norm"] = std::sqrt(norm.result());
        if (b.lstm) {
            Sum cell;
            for (std::size_t i = b.h; i < 2 * b.h; ++i)
                cell.add(b.hidden[i] * b.hidden[i]);
            info["cell_norm"] = std::sqrt(cell.result());
        }
        info["training_metrics"] = b.last_metrics;
    } else {
        info = object({{"algorithm", b.kind},
                       {"buffer_kind", "rollout"},
                       {"buffer_size", std::int64_t(b.rollout.size())},
                       {"features", b.observation_value(observation)},
                       {"action_probabilities", probabilities},
                       {"state_value", prediction.output.back()},
                       {"entropy", entropy},
                       {"updates", b.update_count},
                       {"rollout_updates", b.rollout_updates},
                       {"rollout_size", std::int64_t(b.rollout.size())},
                       {"training_metrics", b.last_metrics}});
    }
    if (activations) {
        Value layers = Value::array();
        if (b.recurrent) {
            Vector current(prediction.memory.begin(),
                           prediction.memory.begin() + b.h);
            std::vector<float> f(current.begin(), current.end());
            layers.push(Value::float_array(std::move(f)));
        }
        for (std::size_t i = 1; i + 1 < prediction.dense.size(); ++i) {
            std::vector<float> f(prediction.dense[i].begin(),
                                 prediction.dense[i].end());
            layers.push(Value::float_array(std::move(f)));
        }
        info["activations"] = object({{"version", 1}, {"layers", layers}});
        if (b.recurrent) {
            std::vector<float> prior(b.hidden.begin(), b.hidden.end());
            info["activations"]["recurrent_prior"] =
                Value::float_array(std::move(prior));
        }
    }
    if (b.bundle) {
        info["critic_separate"] = true;
        info["policy_updates"] = b.policy_updates;
    }
    return info;
}
} // namespace wtfcore
