#include "inheritance.hpp"
#include "dense.hpp"
#include "world.hpp"
#include <set>
namespace wtfcore {
namespace {
Value zero_parameters(const Value &source) {
    if (std::holds_alternative<Value::ObjectPtr>(source.data)) {
        Value out = Value::object();
        for (const auto &[key, value] : source.fields())
            out[key] = zero_parameters(value);
        return out;
    }
    if (source.is_array()) {
        if (source.array_size() && std::holds_alternative<Value::ObjectPtr>(
                                       source.array_value(0).data)) {
            Value out = Value::array();
            for (const auto &value : source.items())
                out.push(zero_parameters(value));
            return out;
        }
        return Value::float_array(
            std::vector<float>(source.array_size(), 0.0f));
    }
    throw std::runtime_error("Unexpected native parameter record");
}
void cross_parameters(Value &left, const Value &right, Random &rng) {
    if (std::holds_alternative<Value::ObjectPtr>(left.data)) {
        for (auto &[key, value] : left.fields())
            cross_parameters(value, right.at(key), rng);
        return;
    }
    if (!left.is_array() || !right.is_array() ||
        left.array_size() != right.array_size())
        throw std::runtime_error("Incompatible native parent arrays");
    if (left.array_size() &&
        std::holds_alternative<Value::ObjectPtr>(left.array_value(0).data)) {
        for (std::size_t i = 0; i < left.items().size(); ++i)
            cross_parameters(left.items()[i], right.items()[i], rng);
        return;
    }
    auto a = dense_floats(left), b = dense_floats(right);
    for (std::size_t i = 0; i < a.size(); ++i)
        if (rng.bits(1))
            a[i] = b[i];
    left = Value::float_array(std::move(a));
}
Value layer_slice(const Value &layers, std::size_t begin, std::size_t end) {
    Value out = Value::array();
    for (auto i = begin; i < end; ++i)
        out.push(layers.items().at(i));
    return out;
}
void reset_hidden(Value &state) {
    auto kind = state.at("config").text("type");
    if (kind == "recurrent_ppo" || kind == "lstm_ppo") {
        int count = state.at("config").integer("recurrent_size") *
                    (kind == "lstm_ppo" ? 2 : 1);
        state["hidden"] = values(std::vector<double>(count, 0.0));
    }
}
} // namespace
std::vector<std::uint32_t> offspring_seed_words(std::uint64_t seed, int id) {
    auto low = static_cast<std::uint64_t>(id) * 7919 + 13;
    std::vector<std::uint32_t> out{static_cast<std::uint32_t>(low),
                                   static_cast<std::uint32_t>(seed) ^
                                       static_cast<std::uint32_t>(low >> 32),
                                   static_cast<std::uint32_t>(seed >> 32)};
    while (out.size() > 1 && !out.back())
        out.pop_back();
    return out;
}
void reset_birth_state(Value &state) {
    for (const auto *key : {"replay", "rollout", "trajectory"})
        if (state.contains(key))
            state[key] = Value::array();
    for (const auto *key : {"steps", "updates", "rollout_updates", "cursor",
                            "last_group_step", "policy_updates"})
        if (state.contains(key))
            state[key] = 0;
    if (state.contains("pending"))
        state["pending"] = Value{};
    if (state.contains("last_metrics"))
        state["last_metrics"] = Value::object();
    if (state.contains("baseline"))
        state["baseline"] = 0.0;
    if (state.contains("moments")) {
        state["moments"] = zero_parameters(state.at("network"));
        state["variances"] = zero_parameters(state.at("network"));
    }
    for (const auto *key : {"auxiliary", "late"})
        if (state.contains(key)) {
            auto &head = state[key];
            head["moments"] = zero_parameters(head.at("network"));
            head["variances"] = zero_parameters(head.at("network"));
            head["updates"] = 0;
        }
    reset_hidden(state);
    if (state.contains("target"))
        state["target"] = state.at("weights");
    if (state.contains("target_network"))
        state["target_network"] = state.at("network");
    if (state.contains("reference_network"))
        state["reference_network"] = state.at("network");
    if (state.contains("target_critics")) {
        std::size_t block = state.at("config")
                                .at("network")
                                .at("hidden_layers")
                                .items()
                                .size() +
                            1;
        state["target_critics"] =
            array({layer_slice(state.at("network"), block, 2 * block),
                   layer_slice(state.at("network"), 2 * block, 3 * block)});
    }
}
Value inherit_controller_state(const Value &mother, const Value &father,
                               const std::vector<std::uint32_t> &seed,
                               bool crossover) {
    Value child = mother;
    for (const auto *key : {"replay", "rollout", "trajectory"})
        if (child.contains(key))
            child[key] = Value::array();
    if (child.contains("cursor"))
        child["cursor"] = 0;
    if (child.contains("pending"))
        child["pending"] = Value{};
    Random rng;
    rng.reseed_words(seed);
    child["rng_state"] = rng.checkpoint();
    auto training = seed;
    if (training.empty())
        training.push_back(0);
    training[0] ^= 0x5F3759DFU;
    rng.reseed_words(training);
    child["training_rng_state"] = rng.checkpoint();
    if (child.contains("group_rng_state")) {
        auto group = seed;
        if (group.empty())
            group.push_back(0);
        group[0] ^= 0x0BADC0DEU;
        rng.reseed_words(group);
        child["group_rng_state"] = rng.checkpoint();
        child["last_group_step"] = child.at("steps");
    }
    reset_hidden(child);
    if (!crossover)
        return child;
    if (mother.at("config").text("type") != father.at("config").text("type") ||
        !equivalent(mother.at("actions"), father.at("actions")))
        throw std::runtime_error(
            "Incompatible native parent controller settings");
    auto cross = seed;
    cross.resize(std::max<std::size_t>(3, cross.size()), 0);
    cross[0] ^= 0x4F564552U;
    cross[1] ^= 0x524F5353U;
    cross[2] ^= 0x43U;
    rng.reseed_words(cross);
    const char *parameter_key =
        child.at("config").text("type") == "q_network" ? "weights" : "network";
    cross_parameters(child[parameter_key], father.at(parameter_key), rng);
    for (const auto *key : {"auxiliary", "late"})
        if (child.contains(key))
            cross_parameters(child[key]["network"],
                             father.at(key).at("network"), rng);
    reset_birth_state(child);
    return child;
}
std::vector<Value *> mutation_parameter_arrays(Value &state) {
    std::vector<Value *> out;
    if (state.at("config").text("type") == "q_network") {
        out.push_back(&state["weights"]["w1"]);
        out.push_back(&state["weights"]["w2"]);
    } else
        for (auto &layer : state["network"].items())
            for (auto &[key, value] : layer.fields())
                if (key == "weights" || key == "input_weights" ||
                    key == "recurrent_weights")
                    out.push_back(&value);
    for (const auto *key : {"auxiliary", "late"})
        if (state.contains(key))
            for (auto &layer : state[key]["network"].items())
                out.push_back(&layer["weights"]);
    return out;
}
void mutate_birth_state(World &world, Value &child_state, const Value &birth,
                        const std::vector<std::uint32_t> &seed) {
    auto &child = world.animal(birth.integer("target"));
    const auto &rules = world.species(child).get("mutation");
    if (!rules.flag("enabled") || rules.num("probability_per_birth") <= 0 ||
        rules.num("noise_std") <= 0)
        return;
    auto words = seed;
    words.resize(std::max<std::size_t>(2, words.size()), 0);
    words[0] ^= 0x54494f4eU;
    words[1] ^= 0x4d555441U;
    Random rng;
    rng.reseed_words(words);
    if (rng.random() >= rules.num("probability_per_birth"))
        return;
    auto arrays = mutation_parameter_arrays(child_state);
    std::size_t total = 0;
    for (const auto *values : arrays)
        total += values->array_size();
    if (!total)
        return;
    int count = rng.randint(
        1, std::min<std::size_t>(rules.integer("max_changed_weights"), total));
    std::vector<std::size_t> selected;
    std::size_t setsize = 21;
    if (count > 5) {
        std::size_t size = 1;
        while (size < static_cast<std::size_t>(count) * 3)
            size *= 4;
        setsize += size;
    }
    if (total <= setsize) {
        std::vector<std::size_t> pool(total);
        for (std::size_t i = 0; i < total; ++i)
            pool[i] = i;
        for (int i = 0; i < count; ++i) {
            std::size_t j = rng.below(total - i);
            selected.push_back(pool[j]);
            pool[j] = pool[total - i - 1];
        }
    } else {
        std::set<std::size_t> seen;
        for (int i = 0; i < count; ++i) {
            std::size_t j = rng.below(total);
            while (seen.count(j))
                j = rng.below(total);
            seen.insert(j);
            selected.push_back(j);
        }
    }
    int changes = 0;
    for (auto offset : selected) {
        std::size_t layer = 0;
        while (offset >= arrays[layer]->array_size()) {
            offset -= arrays[layer]->array_size();
            ++layer;
        }
        auto &value = arrays[layer]->items()[offset];
        double before = value.number();
        double after =
            static_cast<float>(before + rng.gauss(0, rules.num("noise_std")));
        if (!std::isfinite(after))
            throw std::runtime_error("Mutation produced nonfinite weight");
        if (after != before) {
            value = after;
            ++changes;
        }
    }
    if (!changes)
        return;
    reset_birth_state(child_state);
    child["mutated"] = true;
    child["mutation_weights"] = changes;
    child["mutation_count"] = child.integer("mutation_count") + 1;
    auto name = child.text("species");
    world.frame["metrics"]["mutations"][name] =
        world.frame.at("metrics").at("mutations").integer(name) + 1;
    world.frame["events"].push(object({{"kind", "mutated"},
                                       {"actor", birth.at("actor")},
                                       {"target", birth.at("target")},
                                       {"amount", changes},
                                       {"cause", "weights"},
                                       {"mate", -1},
                                       {"position", Value{}}}));
}
} // namespace wtfcore
