#include "augmented_ppo.hpp"
#include <numeric>
namespace wtfcore {
namespace {
using Vector = std::vector<double>;
using Gradient = std::vector<Vector>;
using Parameters = std::vector<std::vector<float> *>;
struct Sample {
    Vector observation;
    int action = 0;
    double reward = 0, logp = 0, value = 0, next_value = 0;
    bool terminated = false, truncated = false;
};
void increment(std::int64_t &x) {
    if (x == std::numeric_limits<std::int64_t>::max())
        throw std::runtime_error(
            "Native augmented PPO counter capacity exceeded");
    ++x;
}
Value slice(const Value &v, std::size_t a, std::size_t b) {
    Value out = Value::array();
    for (std::size_t i = a; i < b; ++i)
        out.push(v.items().at(i));
    return out;
}
void append(Value &to, const Value &from) {
    for (const auto &v : from.items())
        to.push(v);
}
Parameters parameters(DenseNetwork &network) {
    Parameters p;
    for (auto &l : network.layers) {
        p.push_back(&l.weights);
        p.push_back(&l.bias);
    }
    return p;
}
Gradient zeros(const Parameters &p) {
    Gradient g;
    for (auto *v : p)
        g.emplace_back(v->size(), 0.0);
    return g;
}
void clear(Gradient &g) {
    for (auto &v : g)
        std::fill(v.begin(), v.end(), 0.0);
}
std::size_t sample_action(const Vector &p, Random &rng) {
    double draw = rng.random(), sum = 0;
    std::size_t index = p.size() - 1;
    for (std::size_t i = 0; i < p.size(); ++i) {
        sum += p[i];
        if (draw < sum) {
            index = i;
            break;
        }
    }
    return index;
}
void normalize(Vector &v) {
    if (v.size() < 2)
        return;
    Sum s;
    for (double x : v)
        s.add(x);
    double mean = s.result() / v.size();
    s = {};
    for (double x : v)
        s.add(std::pow(x - mean, 2.0));
    double scale = std::sqrt(s.result() / v.size() + 1e-8);
    for (auto &x : v)
        x = (x - mean) / scale;
}
std::vector<const DenseLayer *> layer_refs(const DenseNetwork &network) {
    std::vector<const DenseLayer *> out;
    for (const auto &l : network.layers)
        out.push_back(&l);
    return out;
}
DenseNetwork::Tape forward_refs(const std::vector<const DenseLayer *> &layers,
                                const Vector &observation) {
    DenseNetwork::Tape tape{observation};
    for (std::size_t j = 0; j < layers.size(); ++j) {
        const auto &l = *layers[j];
        const auto &x = tape.back();
        if (x.size() != l.inputs)
            throw std::runtime_error("Augmented forward width mismatch");
        Vector out(l.bias.size());
        for (std::size_t k = 0; k < out.size(); ++k) {
            Sum sum;
            for (std::size_t i = 0; i < l.inputs; ++i)
                sum.add(x[i] * double(l.weights[k * l.inputs + i]));
            double v = double(l.bias[k]) + sum.result();
            out[k] = j + 1 == layers.size() ? v : std::tanh(v);
        }
        tape.push_back(std::move(out));
    }
    return tape;
}
void backward_refs(const std::vector<const DenseLayer *> &layers,
                   const DenseNetwork::Tape &tape, Vector delta,
                   const std::vector<Vector *> &gradient) {
    for (std::size_t j = layers.size(); j-- > 0;) {
        const auto &l = *layers[j];
        const auto &x = tape[j];
        auto &dw = *gradient[2 * j];
        auto &db = *gradient[2 * j + 1];
        Vector prior(j ? l.inputs : 0, 0.0);
        for (std::size_t k = 0; k < l.bias.size(); ++k) {
            double d = delta[k];
            db[k] += d;
            for (std::size_t i = 0; i < l.inputs; ++i) {
                dw[k * l.inputs + i] += d * x[i];
                if (j)
                    prior[i] += d * double(l.weights[k * l.inputs + i]);
            }
        }
        if (j) {
            delta.resize(l.inputs);
            for (std::size_t i = 0; i < l.inputs; ++i)
                delta[i] = prior[i] * (1 - x[i] * x[i]);
        }
    }
}
void backward(const DenseNetwork &network, const DenseNetwork::Tape &tape,
              Vector delta, Gradient &gradient, std::size_t offset = 0) {
    std::vector<Vector *> g;
    for (std::size_t i = 0; i < 2 * network.layers.size(); ++i)
        g.push_back(&gradient[offset + i]);
    backward_refs(layer_refs(network), tape, std::move(delta), g);
}
struct Optimizer {
    Value layout, variance_layout;
    std::vector<std::size_t> order;
    std::vector<std::vector<float>> moments, variances;
    std::int64_t updates = 0;
    Optimizer() = default;
    Optimizer(const Value &state, const Parameters &p)
        : layout(state.at("moments")), variance_layout(state.at("variances")),
          updates(state.at("updates").i64()) {
        if (updates < 0)
            throw std::runtime_error(
                "Negative augmented PPO optimizer counter");
        const auto &network = state.at("network").items();
        for (std::size_t j = 0; j < network.size(); ++j) {
            for (const auto *key : {"weights", "bias"}) {
                moments.push_back(
                    dense_floats(state.at("moments").items().at(j).at(key)));
                variances.push_back(
                    dense_floats(state.at("variances").items().at(j).at(key)));
            }
            for (const auto &[key, ignored] : network[j].fields()) {
                (void)ignored;
                if (key != "weights" && key != "bias")
                    throw std::runtime_error(
                        "Unexpected augmented parameter key");
                order.push_back(2 * j + (key == "bias" ? 1 : 0));
            }
        }
        if (p.size() != moments.size() || p.size() != variances.size())
            throw std::runtime_error("Augmented optimizer block mismatch");
        for (std::size_t i = 0; i < p.size(); ++i) {
            if (p[i]->size() != moments[i].size() ||
                p[i]->size() != variances[i].size())
                throw std::runtime_error("Augmented optimizer shape mismatch");
            for (float x : variances[i])
                if (!std::isfinite(x) || x < 0)
                    throw std::runtime_error(
                        "Invalid augmented optimizer variance");
        }
        for (auto *record : {&layout, &variance_layout})
            for (auto &layer : record->items())
                for (auto &entry : layer.fields())
                    entry.second = Value{};
    }
    Value values_of(const std::vector<std::vector<float>> &arrays,
                    bool variance = false) const {
        Value out = variance ? variance_layout : layout;
        for (std::size_t j = 0; j < out.items().size(); ++j) {
            out.items()[j]["weights"] = Value::float_array(arrays.at(2 * j));
            out.items()[j]["bias"] = Value::float_array(arrays.at(2 * j + 1));
        }
        return out;
    }
    void apply(const Parameters &p, const Gradient &g, std::size_t count,
               double rate, double max_norm) {
        if (!count || p.size() != g.size())
            throw std::runtime_error("Invalid augmented gradient count");
        Sum sum;
        for (auto block : order)
            for (double x : g[block])
                sum.add(std::pow(x / double(count), 2.0));
        double norm = std::sqrt(sum.result()),
               factor = std::fmin(1.0, max_norm / (norm + 1e-12)) / count;
        increment(updates);
        double c1 = 1 - std::pow(.9, double(updates)),
               c2 = 1 - std::pow(.999, double(updates));
        for (std::size_t j = 0; j < p.size(); ++j)
            for (std::size_t i = 0; i < p[j]->size(); ++i) {
                double d = g[j][i] * factor;
                auto &m = moments[j][i];
                auto &v = variances[j][i];
                m = float(.9 * double(m) + .1 * d);
                v = float(.999 * double(v) + .001 * d * d);
                (*p[j])[i] = float(double((*p[j])[i]) -
                                   rate * (double(m) / c1) /
                                       (std::sqrt(double(v) / c2) + 1e-8));
            }
    }
};
} // namespace
struct AugmentedPPO::Impl {
    Value base, config, last_metrics, pending_imported;
    std::string kind, scope;
    std::shared_ptr<const ObservationLayout> layout_owner;
    const ObservationLayout &layout;
    std::vector<std::string> actions;
    DenseNetwork network, critic, auxiliary, late;
    Optimizer optimizer, auxiliary_optimizer, late_optimizer;
    bool masked = false, central = false, has_auxiliary = false,
         has_late = false, world_inputs = false, pending_valid = false;
    std::vector<bool> mask;
    Vector context, next_context;
    std::size_t inputs = 0, policy_inputs = 0;
    std::int64_t steps = 0, rollout_updates = 0;
    double coefficient = 0, bound = 2;
    std::size_t ready_index = 0, late_features = 0;
    Random rng, training_rng;
    std::vector<Sample> rollout;
    Sample pending;
    explicit Impl(const Value &s)
        : base(s), config(s.at("config")), last_metrics(s.at("last_metrics")),
          kind(s.text("algorithm")),
          scope(config.text("consequence_scope", "context")),
          layout_owner(shared_observation_layout(config)),
          layout(*layout_owner), steps(s.at("steps").i64()),
          rollout_updates(s.at("rollout_updates").i64()) {
        if (s.text("precision") != "float32")
            throw std::runtime_error(
                "Native augmented PPO requires FP32 storage");
        if (kind != "ppo" && kind != "maskable_ppo" && kind != "mappo")
            throw std::runtime_error("Unsupported augmented PPO algorithm");
        if (config.text("type") != kind || steps < 0 || rollout_updates < 0)
            throw std::runtime_error("Invalid augmented PPO state");
        for (const auto &a : s.at("actions").items())
            actions.push_back(a.str());
        if (actions.empty())
            throw std::runtime_error("Empty augmented PPO actions");
        masked = kind == "maskable_ppo";
        central = kind == "mappo";
        has_auxiliary = s.contains("auxiliary");
        has_late = s.contains("late");
        inputs = layout.width();
        if (central) {
            std::size_t block =
                config.at("network").at("hidden_layers").items().size() + 1;
            network = DenseNetwork(slice(s.at("network"), 0, block));
            critic = DenseNetwork(
                slice(s.at("network"), block, s.at("network").items().size()));
            context = numbers(s.at("context"));
            next_context = numbers(s.at("next_context"));
            if (context.size() != 16 || next_context.size() != 16 ||
                critic.layers.front().inputs != inputs + 16 ||
                critic.layers.back().bias.size() != 1 ||
                network.layers.back().bias.size() != actions.size())
                throw std::runtime_error("Invalid centralized PPO shape");
        } else {
            network = DenseNetwork(s.at("network"));
            if (network.layers.back().bias.size() != actions.size() + 1)
                throw std::runtime_error("Invalid augmented PPO output shape");
        }
        policy_inputs = network.layers.front().inputs;
        if (policy_inputs != (has_late ? layout.physical_width : inputs))
            throw std::runtime_error(
                "Augmented PPO observation layout mismatch");
        optimizer = Optimizer(s, parameters_of_policy());
        rng.restore(s.at("rng_state"));
        training_rng.restore(s.at("training_rng_state"));
        if (masked) {
            for (const auto &v : s.at("mask").items())
                mask.push_back(v.boolean());
            validate_mask(mask);
        }
        if (has_late) {
            if (central || layout.version != 7 ||
                !config.flag("consequence_observation") ||
                layout.actions != actions)
                throw std::runtime_error(
                    "Invalid late consequence policy contract");
            late = DenseNetwork(s.at("late").at("network"));
            late_optimizer = Optimizer(s.at("late"), parameters(late));
            bound = config.at("late_consequence").num("max_adjustment", 2.0);
            if (!std::isfinite(bound) || bound <= 0 || bound > 10)
                throw std::runtime_error("Invalid late consequence bound");
            late_features =
                12 * ((!config.get("social_observation").null() ? 1 : 0) +
                      (scope == "both" ? 2 : 1));
            ready_index = layout.index("Self ready");
            if (late.layers.size() != 1 ||
                late.layers[0].inputs != late_features ||
                late.layers[0].bias.size() != actions.size())
                throw std::runtime_error("Invalid late consequence shape");
        }
        if (has_auxiliary) {
            if (central || !config.flag("consequence_observation"))
                throw std::runtime_error("Invalid auxiliary policy contract");
            const auto &state = s.at("auxiliary");
            if (state.text("precision", "float64") != "float32")
                throw std::runtime_error(
                    "Native auxiliary requires FP32 storage");
            auxiliary = DenseNetwork(state.at("network"));
            auxiliary_optimizer = Optimizer(state, parameters(auxiliary));
            coefficient = config.at("auxiliary_consequence").num("coefficient");
            std::size_t hidden = network.layers.back().inputs;
            if (auxiliary.layers.size() != 1 ||
                auxiliary.layers[0].inputs != hidden ||
                auxiliary.layers[0].bias.size() != 36)
                throw std::runtime_error("Invalid auxiliary consequence shape");
        }
        for (const auto &record : s.at("rollout").items()) {
            Sample v;
            v.observation = numbers(record.at("observation"));
            v.action = record.integer("action");
            v.reward = record.num("reward");
            v.logp = record.num("log_probability");
            v.value = record.num("value");
            v.next_value = record.num("next_value");
            v.terminated = record.flag("terminated");
            v.truncated = record.flag("truncated");
            validate_sample(v);
            rollout.push_back(std::move(v));
        }
        if (!s.get("pending").null()) {
            pending_imported = s.at("pending");
            const auto &p = pending_imported.items();
            if (p.size() != 4)
                throw std::runtime_error(
                    "Invalid augmented PPO pending record");
            pending.observation = numbers(p[0]);
            pending.action = p[1].as_int();
            pending.logp = p[2].number();
            pending.value = p[3].number();
            validate_sample(pending);
            pending_valid = true;
        }
        if (config.integer("rollout_steps") < 1 ||
            config.integer("batch_size") < 1 || config.integer("epochs") < 1)
            throw std::runtime_error("Invalid augmented PPO schedule");
        for (const auto *key :
             {"network", "moments", "variances", "rollout", "pending",
              "auxiliary", "late", "rng_state", "training_rng_state",
              "last_metrics", "context", "next_context", "mask"})
            if (base.contains(key))
                base[key] = Value{};
    }
    Parameters parameters_of_policy() {
        auto p = parameters(network);
        if (central) {
            auto q = parameters(critic);
            p.insert(p.end(), q.begin(), q.end());
        }
        return p;
    }
    void validate_mask(const std::vector<bool> &m) const {
        if (m.size() != actions.size() ||
            !std::any_of(m.begin(), m.end(), [](bool x) { return x; }))
            throw std::runtime_error("Missing or empty action mask");
    }
    void validate_sample(const Sample &s) const {
        if (s.observation.size() != inputs + (masked    ? actions.size()
                                              : central ? 16
                                                        : 0) ||
            s.action < 0 || std::size_t(s.action) >= actions.size())
            throw std::runtime_error("Invalid augmented PPO sample");
        if (masked) {
            std::vector<bool> m;
            for (std::size_t i = inputs; i < s.observation.size(); ++i)
                m.push_back(s.observation[i] != 0);
            validate_mask(m);
        }
    }
    Vector pack(const Vector &obs, bool next = false,
                const Value *external = nullptr) const {
        if (obs.size() < inputs)
            throw std::runtime_error("Invalid augmented observation width");
        Vector out(obs.begin(), obs.begin() + inputs);
        if (masked) {
            std::vector<bool> m = mask;
            if (external && !external->get("mask").null()) {
                m.clear();
                for (const auto &v : external->at("mask").items())
                    m.push_back(v.boolean());
            }
            validate_mask(m);
            for (bool x : m)
                out.push_back(x ? 1.0 : 0.0);
        } else if (central) {
            const Vector *ctx = next ? &next_context : &context;
            Vector temp;
            if (external &&
                !external->get(next ? "next_context" : "context").null()) {
                temp = numbers(external->at(next ? "next_context" : "context"));
                ctx = &temp;
            }
            if (ctx->size() != 16)
                throw std::runtime_error("Invalid centralized context");
            out.insert(out.end(), ctx->begin(), ctx->end());
        }
        return out;
    }
    Value observation_value(const Vector &obs) const {
        Vector core(obs.begin(), obs.begin() + std::min(inputs, obs.size()));
        Value out =
            world_observation_values(core, layout.version, world_inputs);
        for (std::size_t i = inputs; i < obs.size(); ++i)
            out.push(masked ? Value(obs[i] != 0) : Value(obs[i]));
        return out;
    }
    Vector physical(const Vector &obs) const {
        return Vector(obs.begin(), obs.begin() + policy_inputs);
    }
    struct Prediction {
        Vector out;
        DenseNetwork::Tape actor, critic;
    };
    Prediction predict(const Vector &obs, bool only_value = false) const {
        Prediction p;
        if (central) {
            if (only_value) {
                p.out = {critic.value(obs)};
                return p;
            }
            p.actor =
                network.forward(Vector(obs.begin(), obs.begin() + inputs));
            p.critic = critic.forward(obs);
            p.out = p.actor.back();
            p.out.push_back(p.critic.back()[0]);
        } else if (only_value)
            p.out = {network.value(physical(obs))};
        else {
            p.actor = network.forward(physical(obs));
            p.out = p.actor.back();
        }
        return p;
    }
    std::vector<Vector> feature_rows(const Vector &obs) const {
        Vector needs = {1.0, std::fmax(0.0, 1 - obs[0]),
                        std::fmax(0.0, 1 - obs[1]), obs.at(ready_index)};
        std::vector<Vector> rows;
        for (std::size_t i = 0; i < actions.size(); ++i) {
            std::vector<std::size_t> offsets;
            if (!config.get("social_observation").null())
                offsets.push_back(layout.physical_width +
                                  4 * action_group(actions[i]));
            for (std::size_t block = 0;
                 block < std::size_t(scope == "both" ? 2 : 1); ++block)
                offsets.push_back(layout.consequence_start +
                                  4 * (block * layout.actions.size() + i));
            Vector row;
            row.reserve(late_features);
            for (auto offset : offsets)
                for (std::size_t outcome = 1; outcome < 4; ++outcome)
                    for (double need : needs)
                        row.push_back(obs.at(offset) *
                                      obs.at(offset + outcome) * need);
            rows.push_back(std::move(row));
        }
        return rows;
    }
    Vector residuals(const Vector &obs,
                     const std::vector<Vector> *prepared = nullptr,
                     Vector *tanh_values = nullptr) const {
        auto rows = prepared ? std::vector<Vector>{} : feature_rows(obs);
        const auto &r = prepared ? *prepared : rows;
        Vector out(actions.size());
        if (tanh_values)
            tanh_values->resize(actions.size());
        const auto &w = late.layers[0].weights;
        for (std::size_t i = 0; i < out.size(); ++i) {
            Sum sum;
            for (std::size_t j = 0; j < late_features; ++j)
                sum.add(double(w[i * late_features + j]) * r[i][j]);
            double value = std::tanh(sum.result());
            out[i] = bound * value;
            if (tanh_values)
                (*tanh_values)[i] = value;
        }
        return out;
    }
    void distribution(const Vector &base_outputs, const Vector &obs, Vector &p,
                      Vector &logs, Vector *adjustments = nullptr,
                      Vector *final_logits = nullptr,
                      const std::vector<Vector> *prepared = nullptr,
                      Vector *tanh_values = nullptr) const {
        Vector logits(base_outputs.begin(), base_outputs.end() - 1);
        if (has_late) {
            auto a = residuals(obs, prepared, tanh_values);
            for (std::size_t i = 0; i < logits.size(); ++i)
                logits[i] = logits[i] + a[i];
            if (adjustments)
                *adjustments = std::move(a);
        }
        if (final_logits)
            *final_logits = logits;
        if (masked) {
            if (obs.size() != inputs + actions.size())
                throw std::runtime_error("Missing action mask");
            for (std::size_t i = 0; i < actions.size(); ++i)
                if (!obs[inputs + i])
                    logits[i] = -1e30;
        }
        categorical_distribution(logits, p, logs);
    }
    struct AuxiliarySample {
        Vector input;
        std::array<std::array<double, 4>, 12> targets{};
        bool active = false;
    };
    AuxiliarySample prepare_auxiliary(const Vector &obs) const {
        AuxiliarySample p;
        std::size_t start;
        std::vector<std::array<double, 4>> general, context_rows;
        std::size_t blocks = scope == "both" ? 2 : 1;
        if (layout.version == 7) {
            start = std::min(policy_inputs, layout.consequence_start);
            std::vector<std::vector<std::array<double, 4>>> rows;
            for (std::size_t block = 0; block < blocks; ++block) {
                std::vector<std::array<double, 4>> r(12);
                for (std::size_t i = 0; i < layout.actions.size(); ++i)
                    for (std::size_t j = 0; j < 4; ++j)
                        r[action_index(layout.actions[i])][j] =
                            obs.at(layout.consequence_start +
                                   4 * (block * layout.actions.size() + i) + j);
                rows.push_back(std::move(r));
            }
            if (scope == "both") {
                general = rows[0];
                context_rows = rows[1];
            } else if (scope == "species_action")
                general = rows[0];
            else
                context_rows = rows[0];
        } else {
            std::size_t width = blocks * 48;
            Vector block;
            if (layout.version == 6) {
                std::size_t raw_start = layout.raw_names.size() - width;
                start = layout.offset(raw_start);
                block = layout.decode_soft(obs, raw_start);
            } else {
                start = policy_inputs - width;
                block =
                    Vector(obs.begin() + start, obs.begin() + policy_inputs);
            }
            auto read = [&](std::size_t offset) {
                std::vector<std::array<double, 4>> r(12);
                for (std::size_t i = 0; i < 12; ++i)
                    for (std::size_t j = 0; j < 4; ++j)
                        r[i][j] = block.at(offset + 4 * i + j);
                return r;
            };
            if (scope == "both") {
                general = read(0);
                context_rows = read(48);
            } else if (scope == "species_action")
                general = read(0);
            else
                context_rows = read(0);
        }
        p.input.assign(obs.begin(), obs.begin() + start);
        p.input.resize(policy_inputs, 0.0);
        Sum confidence;
        for (std::size_t i = 0; i < 12; ++i) {
            if (!context_rows.empty() && context_rows[i][0] > 0)
                p.targets[i] = context_rows[i];
            else if (!general.empty())
                p.targets[i] = general[i];
            confidence.add(p.targets[i][0]);
        }
        p.active = confidence.result() != 0;
        return p;
    }
    void update();
    Value head_state(const DenseNetwork &n, const Optimizer &o,
                     bool precision) const {
        Value s = object({{"network", n.checkpoint()},
                          {"moments", o.values_of(o.moments)},
                          {"variances", o.values_of(o.variances, true)},
                          {"updates", o.updates}});
        if (precision)
            s["precision"] = "float32";
        return s;
    }
    Value checkpoint() {
        Value s = base;
        s["network"] = network.checkpoint();
        if (central)
            append(s["network"], critic.checkpoint());
        s["moments"] = optimizer.values_of(optimizer.moments);
        s["variances"] = optimizer.values_of(optimizer.variances, true);
        Value r = Value::array();
        for (const auto &v : rollout)
            r.push(object({{"observation", observation_value(v.observation)},
                           {"action", v.action},
                           {"reward", v.reward},
                           {"log_probability", v.logp},
                           {"value", v.value},
                           {"next_value", v.next_value},
                           {"terminated", v.terminated},
                           {"truncated", v.truncated}}));
        s["rollout"] = r;
        s["pending"] = Value{};
        if (pending_valid)
            s["pending"] =
                pending_imported.null()
                    ? array({observation_value(pending.observation),
                             pending.action, pending.logp, pending.value})
                    : pending_imported;
        s["steps"] = steps;
        s["updates"] = optimizer.updates;
        s["rollout_updates"] = rollout_updates;
        s["last_metrics"] = last_metrics;
        s["rng_state"] = rng.checkpoint();
        s["training_rng_state"] = training_rng.checkpoint();
        if (masked) {
            Value m = Value::array();
            for (bool x : mask)
                m.push(x);
            s["mask"] = m;
        }
        if (central) {
            s["context"] = values(context);
            s["next_context"] = values(next_context);
        }
        if (has_auxiliary)
            s["auxiliary"] = head_state(auxiliary, auxiliary_optimizer, true);
        if (has_late)
            s["late"] = head_state(late, late_optimizer, false);
        return s;
    }
};
void AugmentedPPO::Impl::update() {
    if (rollout_updates == std::numeric_limits<std::int64_t>::max())
        throw std::runtime_error("Augmented rollout counter capacity exceeded");
    std::size_t n = rollout.size();
    Vector advantages(n), targets(n);
    double trace = 0, discount = config.num("discount_factor"),
           lambda = config.num("gae_lambda");
    for (std::size_t i = n; i-- > 0;) {
        const auto &s = rollout[i];
        double bootstrap = s.terminated ? 0 : s.next_value,
               delta = s.reward + discount * bootstrap - s.value,
               continuation = s.terminated || s.truncated ? 0 : 1;
        trace = delta + discount * lambda * continuation * trace;
        advantages[i] = trace;
        targets[i] = trace + s.value;
    }
    if (config.flag("normalize_advantages"))
        normalize(advantages);
    auto g = zeros(parameters_of_policy());
    Gradient late_g = has_late ? zeros(parameters(late)) : Gradient{};
    std::vector<AuxiliarySample> auxiliary_samples;
    if (has_auxiliary)
        for (const auto &s : rollout)
            auxiliary_samples.push_back(prepare_auxiliary(s.observation));
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::array<double, 7> totals{};
    std::size_t samples = 0;
    bool stopped = false;
    std::size_t batch_size = config.integer("batch_size");
    const char *names[] = {"policy_loss",       "value_loss",
                           "entropy",           "approximate_kl",
                           "clip_fraction",     "auxiliary_loss",
                           "auxiliary_coverage"};
    for (int epoch = 0; epoch < config.integer("epochs"); ++epoch) {
        training_rng.shuffle(indices);
        for (std::size_t offset = 0; offset < n; offset += batch_size) {
            clear(g);
            if (has_late)
                clear(late_g);
            std::array<double, 7> metrics{};
            std::size_t end = std::min(offset + batch_size, n),
                        count = end - offset;
            for (std::size_t index = offset; index < end; ++index) {
                std::size_t i = indices[index];
                const auto &s = rollout[i];
                auto prediction = predict(s.observation);
                Vector p, logs, late_values;
                auto late_rows = has_late ? feature_rows(s.observation)
                                          : std::vector<Vector>{};
                distribution(prediction.out, s.observation, p, logs, nullptr,
                             nullptr, has_late ? &late_rows : nullptr,
                             has_late ? &late_values : nullptr);
                double log_ratio = std::fmax(
                           -60.0, std::fmin(60.0, logs[s.action] - s.logp)),
                       ratio = std::exp(log_ratio),
                       clip = config.num("clip_ratio"),
                       clipped =
                           std::fmax(1 - clip, std::fmin(1 + clip, ratio)),
                       adv = advantages[i];
                bool inactive = (adv > 0 && ratio > 1 + clip) ||
                                (adv < 0 && ratio < 1 - clip);
                double policy_loss = -std::fmin(ratio * adv, clipped * adv),
                       derivative = inactive ? 0.0 : -adv * ratio;
                Sum entropy_sum;
                for (std::size_t j = 0; j < p.size(); ++j)
                    entropy_sum.add(p[j] * logs[j]);
                double entropy = -entropy_sum.result();
                Vector delta(actions.size() + 1);
                for (std::size_t j = 0; j < actions.size(); ++j) {
                    delta[j] =
                        derivative * ((int(j) == s.action ? 1.0 : 0.0) - p[j]) +
                        config.num("entropy_coefficient") * p[j] *
                            (logs[j] + entropy);
                    if (masked && !s.observation[inputs + j])
                        delta[j] = 0.0;
                }
                double error = prediction.out.back() - targets[i];
                delta.back() = config.num("value_coefficient") * error;
                if (central) {
                    backward(network, prediction.actor,
                             Vector(delta.begin(), delta.end() - 1), g);
                    backward(critic, prediction.critic, {delta.back()}, g,
                             2 * network.layers.size());
                } else
                    backward(network, prediction.actor, delta, g);
                if (has_late) {
                    for (std::size_t a = 0; a < actions.size(); ++a) {
                        double value = late_values[a],
                               scale = delta[a] * bound * (1 - value * value);
                        for (std::size_t j = 0; j < late_features; ++j)
                            late_g[0][a * late_features + j] +=
                                scale * late_rows[a][j];
                    }
                }
                metrics[0] += policy_loss;
                metrics[1] += 0.5 * error * error;
                metrics[2] += entropy;
                metrics[3] += (ratio - 1) - log_ratio;
                metrics[4] += double(std::fabs(ratio - 1) > clip);
            }
            double target_kl = config.num("target_kl", 0.03);
            if (target_kl && metrics[3] / count > target_kl) {
                stopped = true;
                break;
            }
            Gradient aux_g;
            if (has_auxiliary) {
                aux_g = zeros(parameters(auxiliary));
                std::vector<const DenseLayer *> view;
                std::vector<Vector *> view_grad;
                for (std::size_t j = 0; j + 1 < network.layers.size(); ++j) {
                    view.push_back(&network.layers[j]);
                    view_grad.push_back(&g[2 * j]);
                    view_grad.push_back(&g[2 * j + 1]);
                }
                view.push_back(&auxiliary.layers[0]);
                view_grad.push_back(&aux_g[0]);
                view_grad.push_back(&aux_g[1]);
                for (std::size_t j = offset; j < end; ++j) {
                    const auto &sample = auxiliary_samples[indices[j]];
                    if (!sample.active)
                        continue;
                    metrics[6] += 1;
                    auto tape = forward_refs(view, sample.input);
                    const auto &out = tape.back();
                    Vector delta(36);
                    for (std::size_t action = 0; action < 12; ++action) {
                        double confidence = sample.targets[action][0];
                        for (std::size_t outcome = 0; outcome < 3; ++outcome) {
                            std::size_t k = action * 3 + outcome;
                            double target = sample.targets[action][outcome + 1],
                                   logit = out[k],
                                   probability =
                                       1 / (1 + std::exp(-std::fmax(
                                                    -60.0,
                                                    std::fmin(60.0, logit)))),
                                   scale = coefficient * confidence / 36;
                            metrics[5] +=
                                scale *
                                (std::fmax(logit, 0.0) - logit * target +
                                 std::log1p(std::exp(-std::fabs(logit))));
                            delta[k] = scale * (probability - target);
                        }
                    }
                    backward_refs(view, tape, delta, view_grad);
                }
            }
            optimizer.apply(parameters_of_policy(), g, count,
                            config.num("learning_rate"),
                            config.num("max_grad_norm"));
            if (has_late)
                late_optimizer.apply(parameters(late), late_g, count,
                                     config.num("learning_rate"),
                                     config.num("max_grad_norm"));
            if (has_auxiliary && metrics[6])
                auxiliary_optimizer.apply(parameters(auxiliary), aux_g, count,
                                          config.num("learning_rate"),
                                          config.num("max_grad_norm"));
            samples += count;
            for (std::size_t k = 0; k < (has_auxiliary ? 7 : 5); ++k)
                totals[k] += metrics[k];
        }
        if (stopped)
            break;
    }
    increment(rollout_updates);
    last_metrics = Value::object();
    if (samples)
        for (std::size_t k = 0; k < (has_auxiliary ? 7 : 5); ++k)
            last_metrics[names[k]] = totals[k] / samples;
    last_metrics["early_stopped"] = stopped;
    rollout.clear();
}
AugmentedPPO::AugmentedPPO(const Value &s) : impl_(std::make_unique<Impl>(s)) {}
AugmentedPPO::~AugmentedPPO() = default;
AugmentedPPO::AugmentedPPO(AugmentedPPO &&) noexcept = default;
AugmentedPPO &AugmentedPPO::operator=(AugmentedPPO &&) noexcept = default;
std::string AugmentedPPO::choose(const Vector &observation, bool training,
                                 bool random_baseline) {
    auto &b = *impl_;
    if (random_baseline) {
        if (b.masked) {
            std::vector<std::size_t> valid;
            for (std::size_t i = 0; i < b.mask.size(); ++i)
                if (b.mask[i])
                    valid.push_back(i);
            if (valid.empty())
                throw std::runtime_error("Empty random action mask");
            return b.actions[valid[b.rng.below(valid.size())]];
        }
        return b.actions[b.rng.below(b.actions.size())];
    }
    if (b.central && !training) {
        if (observation.size() < b.inputs)
            throw std::runtime_error("Invalid centralized observation width");
        auto logits = b.network.predict(
            Vector(observation.begin(), observation.begin() + b.inputs));
        Vector p, logs;
        categorical_distribution(logits, p, logs);
        return b.actions[std::max_element(p.begin(), p.end()) - p.begin()];
    }
    auto packed = b.pack(observation);
    auto prediction = b.predict(packed);
    Vector p, logs;
    b.distribution(prediction.out, packed, p, logs);
    if (!training)
        return b.actions[std::max_element(p.begin(), p.end()) - p.begin()];
    auto index = sample_action(p, b.rng);
    b.pending = Sample{};
    b.pending.observation = std::move(packed);
    b.pending.action = int(index);
    b.pending.logp = logs[index];
    b.pending.value = prediction.out.back();
    b.pending_valid = true;
    b.pending_imported = Value{};
    return b.actions[index];
}
void AugmentedPPO::learn(const Vector &observation, const std::string &action,
                         double reward, const Vector *next, bool terminated,
                         bool truncated) {
    auto &b = *impl_;
    if (b.steps == std::numeric_limits<std::int64_t>::max())
        throw std::runtime_error("Augmented step counter capacity exceeded");
    auto packed = b.pack(observation);
    auto found = std::find(b.actions.begin(), b.actions.end(), action);
    if (found == b.actions.end())
        throw std::runtime_error("Unknown augmented action");
    int index = int(found - b.actions.begin());
    Sample s;
    if (b.pending_valid && b.pending.action == index &&
        b.pending.observation == packed)
        s = b.pending;
    else {
        auto prediction = b.predict(packed);
        Vector p, logs;
        b.distribution(prediction.out, packed, p, logs);
        s.observation = packed;
        s.action = index;
        s.logp = logs[index];
        s.value = prediction.out.back();
    }
    b.pending_valid = false;
    b.pending_imported = Value{};
    s.reward = reward;
    s.terminated = terminated;
    s.truncated = truncated;
    if (!terminated) {
        if (!next)
            throw std::runtime_error(
                "Augmented PPO requires a nonterminal next observation");
        s.next_value = b.predict(b.pack(*next, true), true).out.back();
    }
    b.rollout.push_back(std::move(s));
    increment(b.steps);
    if (b.rollout.size() >= std::size_t(b.config.integer("rollout_steps")) ||
        terminated || truncated)
        b.update();
}
Value AugmentedPPO::checkpoint() { return impl_->checkpoint(); }
Value AugmentedPPO::model_state() {
    auto &b = *impl_;
    Value network = b.network.checkpoint();
    if (b.central)
        append(network, b.critic.checkpoint());
    Value state = object({{"precision", "float32"},
                          {"network", network},
                          {"config", b.config},
                          {"updates", b.optimizer.updates}});
    if (b.has_auxiliary)
        state["auxiliary"] = object({{"network", b.auxiliary.checkpoint()}});
    if (b.has_late)
        state["late"] = object({{"network", b.late.checkpoint()}});
    return state;
}
std::int64_t AugmentedPPO::updates() const { return impl_->optimizer.updates; }
void AugmentedPPO::use_world_inputs() { impl_->world_inputs = true; }
void AugmentedPPO::set_context(const Value &value) {
    auto &b = *impl_;
    if (b.masked && !value.get("mask").null()) {
        std::vector<bool> mask;
        for (const auto &v : value.at("mask").items())
            mask.push_back(v.boolean());
        b.validate_mask(mask);
        b.mask = std::move(mask);
    }
    if (b.central) {
        if (!value.get("context").null()) {
            auto v = numbers(value.at("context"));
            if (v.size() != 16)
                throw std::runtime_error("Invalid current critic context");
            b.context = std::move(v);
        }
        if (!value.get("next_context").null()) {
            auto v = numbers(value.at("next_context"));
            if (v.size() != 16)
                throw std::runtime_error("Invalid next critic context");
            b.next_context = std::move(v);
        }
    }
}
Value AugmentedPPO::rng_state() const { return impl_->rng.checkpoint(); }
Value AugmentedPPO::trial_memory() const { return Value{}; }
std::vector<double> AugmentedPPO::policy_logits(const Vector &observation) {
    auto &b = *impl_;
    if (b.central) {
        if (observation.size() < b.inputs)
            throw std::runtime_error("Invalid centralized observation width");
        return b.network.predict(
            Vector(observation.begin(), observation.begin() + b.inputs));
    }
    auto packed = b.pack(observation);
    auto prediction = b.predict(packed);
    Vector p, logs, final;
    b.distribution(prediction.out, packed, p, logs, nullptr, &final);
    if (b.masked)
        for (std::size_t i = 0; i < final.size(); ++i)
            if (!packed[b.inputs + i])
                final[i] = -1e30;
    return final;
}
std::string AugmentedPPO::trial_action(const Vector &observation,
                                       Random &external_rng, Value &memory,
                                       const Value &context) {
    (void)memory;
    auto &b = *impl_;
    if (b.central) {
        if (observation.size() < b.inputs)
            throw std::runtime_error("Invalid centralized observation width");
        auto logits = b.network.predict(
            Vector(observation.begin(), observation.begin() + b.inputs));
        Vector p, logs;
        categorical_distribution(logits, p, logs);
        return b.actions[sample_action(p, external_rng)];
    }
    auto packed = b.pack(observation, false, &context);
    auto prediction = b.predict(packed);
    Vector p, logs;
    b.distribution(prediction.out, packed, p, logs);
    return b.actions[sample_action(p, external_rng)];
}
Value AugmentedPPO::diagnostics(const Vector &observation, bool activations) {
    auto &b = *impl_;
    auto packed = b.pack(observation);
    auto prediction = b.predict(packed);
    Vector p, logs, adjustments, final;
    b.distribution(prediction.out, packed, p, logs,
                   b.has_late ? &adjustments : nullptr,
                   b.has_late ? &final : nullptr);
    Value probabilities = Value::object();
    for (std::size_t i = 0; i < b.actions.size(); ++i)
        probabilities[b.actions[i]] = p[i];
    Sum entropy;
    for (std::size_t i = 0; i < p.size(); ++i)
        entropy.add(p[i] * logs[i]);
    Value info = object(
        {{"algorithm", b.kind},
         {"buffer_kind", "rollout"},
         {"buffer_size", std::int64_t(b.rollout.size())},
         {"features", world_observation_values(observation, b.layout.version,
                                               b.world_inputs)},
         {"action_probabilities", probabilities},
         {"state_value", prediction.out.back()},
         {"entropy", -entropy.result()},
         {"updates", b.optimizer.updates},
         {"rollout_updates", b.rollout_updates},
         {"rollout_size", std::int64_t(b.rollout.size())},
         {"training_metrics", b.last_metrics}});
    if (b.has_late)
        info["late_consequence"] = object(
            {{"base_logits",
              values(Vector(prediction.out.begin(), prediction.out.end() - 1))},
             {"adjustments", values(adjustments)},
             {"final_logits", values(final)}});
    if (activations) {
        Value layers = Value::array();
        for (std::size_t i = 1; i + 1 < prediction.actor.size(); ++i) {
            std::vector<float> v(prediction.actor[i].begin(),
                                 prediction.actor[i].end());
            layers.push(Value::float_array(std::move(v)));
        }
        info["activations"] = object({{"version", 1}, {"layers", layers}});
    }
    if (b.masked) {
        Value m = Value::object();
        for (std::size_t i = 0; i < b.actions.size(); ++i)
            m[b.actions[i]] = bool(b.mask[i]);
        info["action_mask"] = m;
    }
    if (b.central) {
        info["critic_context"] = values(b.context);
        info["reward_scope"] = "species_mean";
    }
    return info;
}
} // namespace wtfcore
