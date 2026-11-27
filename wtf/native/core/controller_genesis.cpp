#include "controller_genesis.hpp"
namespace wtfcore {
namespace {
using Words = std::vector<std::uint32_t>;
Words seed_xor(Words words, std::uint64_t mask) {
    words.resize(std::max<std::size_t>(2, words.size()), 0);
    words[0] ^= static_cast<std::uint32_t>(mask);
    words[1] ^= static_cast<std::uint32_t>(mask >> 32);
    return words;
}
Random random_from(const Words &words) {
    Random result;
    result.reseed_words(words);
    return result;
}
Value zeros(std::size_t size) {
    return Value::float_array(std::vector<float>(size));
}
Value zero_tree(const Value &tree) {
    if (tree.is_array()) {
        Value result = Value::array();
        for (const auto &layer : tree.items())
            result.push(zero_tree(layer));
        return result;
    }
    Value result = Value::object();
    for (const auto &field : tree.fields())
        result[field.first] = zeros(field.second.array_size());
    return result;
}
std::vector<int> dimensions(const Value &layers) {
    std::vector<int> result;
    for (const auto &v : layers.items())
        result.push_back(v.as_int());
    return result;
}
Value mlp(const std::vector<int> &hidden, int actions, const Words &seed,
          bool value_head, int inputs) {
    auto rng = random_from(seed);
    auto sizes = hidden;
    sizes.insert(sizes.begin(), inputs);
    sizes.push_back(actions + static_cast<int>(value_head));
    Value result = Value::array();
    for (std::size_t layer = 1; layer < sizes.size(); ++layer) {
        int in = sizes[layer - 1], out = sizes[layer];
        double scale = std::sqrt(6.0 / (in + out));
        std::vector<float> weights(static_cast<std::size_t>(in) * out);
        for (std::size_t i = 0; i < weights.size(); ++i) {
            double value = rng.uniform(-scale, scale);
            if (layer + 1 == sizes.size() &&
                i < static_cast<std::size_t>(in) * actions)
                value *= 0.01;
            weights[i] = static_cast<float>(value);
        }
        result.push(object({{"weights", Value::float_array(std::move(weights))},
                            {"bias", zeros(out)}}));
    }
    return result;
}
void append(Value &target, const Value &other) {
    for (const auto &item : other.items())
        target.push(item);
}
Value initial_head(const Value &network, bool precision) {
    Value head = object({{"network", network},
                         {"moments", zero_tree(network)},
                         {"variances", zero_tree(network)},
                         {"updates", 0}});
    if (precision)
        head["precision"] = "float32";
    return head;
}
} // namespace
Value make_controller_state(const Value &config,
                            const std::vector<std::string> &actions,
                            const Words &base_seed,
                            const Words &individual_seed) {
    auto kind = config.at("type").str();
    int count = static_cast<int>(actions.size());
    ObservationLayout layout(config);
    int inputs = static_cast<int>(layout.width());
    auto hidden = dimensions(kind == "q_network"
                                 ? config.at("hidden_layers")
                                 : config.at("network").at("hidden_layers"));
    Value action_values = Value::array();
    for (const auto &name : actions)
        action_values.push(name);
    auto rng = random_from(individual_seed);
    auto training = random_from(seed_xor(individual_seed, 0x5F3759DF));
    if (kind == "q_network") {
        auto init = random_from(base_seed);
        int h = hidden.at(0);
        auto weights = [&](std::size_t size, double scale) {
            std::vector<float> values(size);
            for (auto &v : values)
                v = static_cast<float>(init.uniform(-1, 1) * scale);
            return Value::float_array(std::move(values));
        };
        Value parameters =
            object({{"w1", weights(static_cast<std::size_t>(inputs) * h,
                                   std::sqrt(6.0 / (inputs + h)))},
                    {"b1", zeros(h)},
                    {"w2", weights(static_cast<std::size_t>(h) * count,
                                   std::sqrt(6.0 / (h + count)))},
                    {"b2", zeros(count)}});
        return object({{"precision", "float32"},
                       {"config", config},
                       {"actions", action_values},
                       {"weights", parameters},
                       {"target", parameters},
                       {"replay", Value::array()},
                       {"cursor", 0},
                       {"steps", 0},
                       {"updates", 0},
                       {"epsilon", config.at("exploration").at("initial")},
                       {"rng_state", rng.checkpoint()},
                       {"training_rng_state", training.checkpoint()}});
    }
    bool q = kind == "dqn" || kind == "ddqn" || kind == "qrdqn";
    bool actor = kind == "reinforce" || kind == "grpo";
    bool recurrent = kind == "recurrent_ppo" || kind == "lstm_ppo";
    Value network;
    Value critics = Value::array();
    int policy_inputs = config.contains("late_consequence")
                            ? static_cast<int>(layout.physical_width)
                            : inputs;
    if (q) {
        int quantiles =
            kind == "qrdqn" ? config.at("num_quantiles").as_int() : 1;
        network = mlp(hidden, count * quantiles, base_seed, false, inputs);
    } else if (recurrent) {
        int h = config.at("recurrent_size").as_int();
        int gates = kind == "lstm_ppo" ? 4 : 3;
        auto cell_rng = random_from(seed_xor(base_seed, 0x6A09E667));
        auto weights = [&](std::size_t size, double scale) {
            std::vector<float> values(size);
            for (auto &v : values)
                v = static_cast<float>(cell_rng.uniform(-1, 1) * scale);
            return Value::float_array(std::move(values));
        };
        std::vector<float> bias(gates * h);
        if (gates == 4)
            std::fill(bias.begin() + h, bias.begin() + 2 * h, 1.0f);
        network = Value::array();
        network.push(
            object({{"input_weights",
                     weights(static_cast<std::size_t>(gates) * h * inputs,
                             std::sqrt(6.0 / (inputs + h)))},
                    {"recurrent_weights",
                     weights(static_cast<std::size_t>(gates) * h * h,
                             std::sqrt(3.0 / h))},
                    {"bias", Value::float_array(std::move(bias))}}));
        append(network, mlp(hidden, count, base_seed, true, h));
    } else if (kind == "trpo" || kind == "discrete_sac" || kind == "mappo") {
        network = mlp(hidden, count, base_seed, false, inputs);
        int n = kind == "discrete_sac" ? 2 : 1;
        for (int i = 0; i < n; ++i) {
            auto seed = seed_xor(
                base_seed, kind == "mappo" ? 0xC0171C : 0xA341316CULL + i);
            auto critic = mlp(
                kind == "mappo" ? dimensions(config.at("critic_hidden_layers"))
                                : hidden,
                kind == "discrete_sac" ? count : 1, seed, false,
                inputs + (kind == "mappo" ? 16 : 0));
            critics.push(critic);
            append(network, critic);
        }
    } else if (actor || kind == "ppo" || kind == "maskable_ppo" ||
               kind == "a2c") {
        network = mlp(hidden, count, base_seed, !actor, policy_inputs);
    } else
        throw std::runtime_error("Unsupported controller genesis policy: " +
                                 kind);
    Value state = object({{"algorithm", kind},
                          {"precision", "float32"},
                          {"config", config},
                          {"actions", action_values},
                          {"network", network}});
    if (q)
        state["target_network"] = network;
    state["moments"] = zero_tree(network);
    state["variances"] = zero_tree(network);
    state[q ? "replay" : actor ? "trajectory" : "rollout"] = Value::array();
    if (q)
        state["cursor"] = 0;
    else
        state["pending"] = Value{};
    state["steps"] = 0;
    state["updates"] = 0;
    if (q)
        state["epsilon"] = config.at("exploration").at("initial");
    else
        state["rollout_updates"] = 0;
    if (actor)
        state["baseline"] = 0.0;
    state["last_metrics"] = Value::object();
    state["rng_state"] = rng.checkpoint();
    state["training_rng_state"] = training.checkpoint();
    if (config.contains("auxiliary_consequence")) {
        int trunk = hidden.empty() ? policy_inputs : hidden.back();
        state["auxiliary"] = initial_head(
            mlp({}, 36, seed_xor(base_seed, 0x434F4E534551ULL), false, trunk),
            true);
    }
    if (config.contains("late_consequence")) {
        int sources =
            (!config.get("social_observation").null()) +
            (config.text("consequence_scope", "context") == "both" ? 2 : 1);
        auto late = Value::array();
        late.push(object(
            {{"weights", zeros(static_cast<std::size_t>(12) * sources * count)},
             {"bias", zeros(count)}}));
        state["late"] = initial_head(late, false);
    }
    if (kind == "grpo") {
        state["reference_network"] = network;
        state["group_rng_state"] =
            random_from(seed_xor(individual_seed, 0x0BADC0DE)).checkpoint();
        state["last_group_step"] = 0;
    }
    if (recurrent) {
        Value hidden_state = Value::array();
        int n =
            config.at("recurrent_size").as_int() * (kind == "lstm_ppo" ? 2 : 1);
        for (int i = 0; i < n; ++i)
            hidden_state.push(0.0);
        state["hidden"] = hidden_state;
    }
    if (kind == "trpo")
        state["policy_updates"] = 0;
    if (kind == "discrete_sac") {
        state["replay"] = Value::array();
        state["cursor"] = 0;
        state["target_critics"] = critics;
    }
    if (kind == "maskable_ppo") {
        Value mask = Value::array();
        for (int i = 0; i < count; ++i)
            mask.push(true);
        state["mask"] = mask;
    }
    if (kind == "mappo") {
        Value context = Value::array();
        for (int i = 0; i < 16; ++i)
            context.push(0.0);
        state["context"] = context;
        state["next_context"] = context;
    }
    return state;
}
} // namespace wtfcore
