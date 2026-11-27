#include "run_genesis.hpp"
#include "champions.hpp"
#include "context.hpp"
#include "controller_genesis.hpp"
#include "inheritance.hpp"
#include "observation_layout.hpp"
#include "social.hpp"
#include "world_genesis.hpp"
#include <set>
namespace wtfcore {
namespace {
const std::vector<std::string> sexes = {"female", "male"};
Value strings(const std::vector<std::string> &items) {
    Value out = Value::array();
    for (auto &s : items)
        out.push(s);
    return out;
}
std::vector<std::string> action_names(const Value &v) {
    std::vector<std::string> out;
    for (auto &a : v.items())
        out.push_back(a.str());
    return out;
}
std::vector<std::uint32_t> words(std::uint64_t n) {
    return {std::uint32_t(n), std::uint32_t(n >> 32)};
}
std::string criterion(const Value &c, const std::string &name,
                      const std::string &sex,
                      const std::string &fallback = "") {
    const auto &s = c.at("species").at(name);
    const auto &specific = s.get("champion_selection_by_sex").get(sex);
    return !specific.null()
               ? specific.str()
               : s.get("champion_selection")
                     .text("criterion", !fallback.empty() ? fallback
                                        : s.text("diet") == "resource"
                                            ? "lifespan"
                                            : "kills");
}
Value selection_policy(const Value &c, const std::string &name,
                       const std::string &metric) {
    const auto &s = c.at("species").at(name);
    const auto &p = s.get("champion_selection");
    if (metric == "balanced")
        return object(
            {{"version", 1},
             {"survival_steps_per_kill", p.contains("survival_steps_per_kill")
                                             ? p.at("survival_steps_per_kill")
                                             : Value(300)},
             {"max_survival_bonus", p.contains("max_survival_bonus")
                                        ? p.at("max_survival_bonus")
                                        : Value(2)}});
    return object(
        {{"version", 1},
         {"family", "founder_descendants"},
         {"diet", s.at("diet")},
         {"min_age_steps",
          p.contains("lineage_min_age_steps")
              ? p.at("lineage_min_age_steps")
              : Value(std::max(
                    1, s.get("reproduction").integer("min_age_steps", 1)))}});
}
Value initial_champions(const Value &c, const Value &models) {
    Value entries = Value::object();
    for (const auto &[name, groups] : models.fields())
        for (const auto &[sex, initial] : groups.fields()) {
            const auto &e = initial.get("champion");
            if (e.null() || e.get("source").text("sex") != sex)
                continue;
            auto metric = e.text("criterion");
            if ((metric == "balanced" || metric == "lineage") &&
                !equivalent(e.get("selection_policy"),
                            selection_policy(c, name, metric)))
                continue;
            Value entry = e;
            entry["model"] = initial.at("model");
            entries[name][sex] = object({{metric, entry}});
        }
    return entries;
}
Value packed_like(const Value &value, const Value &shape) {
    if (std::holds_alternative<Value::ObjectPtr>(shape.data)) {
        if (!std::holds_alternative<Value::ObjectPtr>(value.data) ||
            value.fields().size() != shape.fields().size())
            throw std::runtime_error("Invalid model parameter fields");
        Value out = Value::object();
        for (const auto &[k, v] : shape.fields())
            out[k] = packed_like(value.at(k), v);
        return out;
    }
    if (shape.is_array()) {
        if (!value.is_array() || value.array_size() != shape.array_size())
            throw std::runtime_error("Invalid model parameter dimensions");
        if (shape.array_size() && std::holds_alternative<Value::ObjectPtr>(
                                      shape.array_value(0).data)) {
            Value out = Value::array();
            for (std::size_t i = 0; i < shape.array_size(); ++i)
                out.push(
                    packed_like(value.array_value(i), shape.array_value(i)));
            return out;
        }
        std::vector<float> out;
        out.reserve(shape.array_size());
        for (std::size_t i = 0; i < shape.array_size(); ++i) {
            auto v = value.array_value(i);
            if (std::holds_alternative<bool>(v.data) ||
                !std::isfinite(v.number()))
                throw std::runtime_error("Nonfinite model weight");
            float f = static_cast<float>(v.number());
            if (!std::isfinite(f))
                throw std::runtime_error("FP32 model weight overflow");
            out.push_back(f);
        }
        return Value::float_array(std::move(out));
    }
    throw std::runtime_error("Invalid model parameter template");
}
Value early_state(const Value &c, const std::string &mode, const Value &models,
                  const Value &entries) {
    const auto &s = c.at("experiment").get("early_stopping");
    int patience = s.integer("patience_runs");
    if (!patience)
        return {};
    Value names = s.get("species");
    if (names.null()) {
        names = Value::array();
        for (const auto &[n, v] : c.at("species").fields()) {
            (void)v;
            names.push(n);
        }
    }
    Value criteria = Value::object(), best = Value::object();
    for (const auto &name : names.items())
        for (const auto &sex : sexes) {
            auto n = name.str();
            auto metric = criterion(
                c, n, sex,
                models.get(n).get(sex).get("provenance").text("criterion"));
            criteria[n][sex] = metric;
            const auto &e = entries.get(n).get(sex).get(metric);
            if (mode == "train" && !e.null())
                best[n][sex] =
                    object({{"criterion", metric}, {"rank", e.at("rank")}});
        }
    return object({{"patience_runs", patience},
                   {"species", names},
                   {"criteria", criteria},
                   {"best_ranks", best},
                   {"stagnant_runs", 0},
                   {"last_run", 0},
                   {"stopped", false},
                   {"reason", Value{}}});
}
} // namespace
Value native_model_signature(const Value &c,
                             const std::vector<std::string> &actions) {
    auto layout = shared_observation_layout(c);
    auto kind = c.text("type");
    int version = c.integer("observation_version", 1);
    Value out =
        object({{"observation_version", version},
                {"inputs", std::int64_t(layout->width())},
                {"algorithm", kind},
                {"hidden_layers", kind == "q_network"
                                      ? c.at("hidden_layers")
                                      : c.at("network").at("hidden_layers")},
                {"activation", "tanh"},
                {"actions", strings(actions)},
                {"value_head", kind == "ppo" || kind == "a2c" ||
                                   kind == "recurrent_ppo" ||
                                   kind == "maskable_ppo" || kind == "mappo" ||
                                   kind == "trpo" || kind == "lstm_ppo"}});
    if (version == 7) {
        out["observation_actions"] = strings(layout->actions);
        for (auto key : {"compact_layout_version", "movement_summary_version",
                         "scent_summary_version"})
            out[key] = 1;
    }
    if (c.contains("late_consequence")) {
        out["late_consequence_version"] = 1;
        const auto &late = c.at("late_consequence");
        out["late_consequence"] =
            object({{"max_adjustment", late.contains("max_adjustment")
                                           ? late.at("max_adjustment")
                                           : Value(2.0)}});
        out["policy_inputs"] = std::int64_t(layout->physical_width);
    }
    if (version == 6) {
        Value defaults =
            object({{"distance", array({0.0, 0.3, 1.0})},
                    {"energy", array({0.0, 0.25, 0.75, 1.0})},
                    {"health", array({0.0, 0.5, 1.0})},
                    {"level", array({0.0, 0.5, 1.0})},
                    {"depth", array({0.0, 0.1, 1.0})},
                    {"elevation", array({0.0, 0.1, 1.0})},
                    {"climb", array({-1.0, -0.1, 0.0, 0.1, 1.0})},
                    {"speed", array({0.0, 0.25, std::sqrt(2.0)})},
                    {"outcome", array({0.0, 0.001, 0.01, 0.1, 1.0})}});
        const auto &overrides = c.get("soft_categories");
        if (!overrides.null())
            for (const auto &[k, v] : overrides.fields())
                defaults[k] = v;
        out["soft_categories"] = defaults;
    }
    if (c.integer("observation_top_k", 1) != 1)
        out["observation_top_k"] = c.at("observation_top_k");
    if (c.integer("movement_history_steps"))
        out["movement_history_steps"] = c.at("movement_history_steps");
    if (c.contains("scent_observation")) {
        const auto &s = c.get("scent_observation");
        if (s.null())
            out["scent_observation"] = Value{};
        else if (s.text("mode", "separated") == "blended")
            out["scent_observation"] = object(
                {{"mode", "blended"}, {"radius", s.integer("radius", 2)}});
        else
            out["scent_observation"] =
                object({{"species", s.at("species")},
                        {"radius", s.integer("radius", 2)},
                        {"separate_sexes", s.flag("separate_sexes", true)}});
    }
    if (!c.get("social_observation").null()) {
        const auto &s = c.at("social_observation");
        out["social_observation"] =
            object({{"history_steps", s.integer("history_steps", 4)},
                    {"max_neighbors", s.integer("max_neighbors", 8)}});
    }
    if (c.contains("auxiliary_consequence"))
        out["auxiliary_consequence_version"] = 1;
    if (c.flag("consequence_observation")) {
        out["consequence_observation"] = true;
        auto scope = c.text("consequence_scope", "context");
        if (scope != "context")
            out["consequence_scope"] = scope;
    }
    if (kind == "recurrent_ppo" || kind == "lstm_ppo") {
        out["recurrent_size"] = c.at("recurrent_size");
        out["recurrent_cell"] = kind == "lstm_ppo" ? "lstm" : "gru";
    } else if (kind == "discrete_sac") {
        out["critic_count"] = 2;
        out["critic_outputs"] = std::int64_t(actions.size());
    } else if (kind == "trpo")
        out["critic_separate"] = true;
    else if (kind == "qrdqn")
        out["num_quantiles"] = c.at("num_quantiles");
    else if (kind == "mappo") {
        out["critic_hidden_layers"] = c.at("critic_hidden_layers");
        out["centralized_context_version"] = 1;
        out["centralized_inputs"] = 16;
    } else if (kind == "maskable_ppo")
        out["action_mask_version"] = 1;
    return out;
}
Value native_model_snapshot(const Value &brain, const Value &signature) {
    Value out = object(
        {{"signature", signature},
         {"precision", brain.contains("precision") ? brain.at("precision")
                                                   : Value("float64")},
         {"parameters", brain.at("config").text("type") == "q_network"
                            ? brain.at("weights")
                            : brain.at("network")}});
    for (auto key : {"auxiliary", "late"})
        if (brain.contains(key))
            out[std::string(key) + "_parameters"] = brain.at(key).at("network");
    return out;
}
void load_native_model_weights(Value &brain, const Value &model,
                               const Value &signature) {
    if (!equivalent(model.get("signature"), signature))
        throw std::runtime_error(
            "Saved brain algorithm, architecture, observation version or "
            "action order is incompatible");
    bool q = brain.at("config").text("type") == "q_network";
    auto parameters = packed_like(model.at("parameters"),
                                  brain.at(q ? "weights" : "network"));
    Value auxiliary, late;
    if (brain.contains("auxiliary"))
        auxiliary = packed_like(model.at("auxiliary_parameters"),
                                brain.at("auxiliary").at("network"));
    if (brain.contains("late")) {
        late = packed_like(model.at("late_parameters"),
                           brain.at("late").at("network"));
        for (std::size_t i = 0; i < late.array_value(0).at("bias").array_size();
             ++i)
            if (late.array_value(0).at("bias").array_value(i).number() != 0)
                throw std::runtime_error(
                    "Saved brain has invalid late-consequence parameters");
    }
    brain[q ? "weights" : "network"] = parameters;
    for (auto key : {"target", "target_network", "reference_network"})
        if (brain.contains(key))
            brain[key] = parameters;
    if (brain.contains("auxiliary"))
        brain["auxiliary"]["network"] = auxiliary;
    if (brain.contains("late"))
        brain["late"]["network"] = late;
    if (brain.contains("target_critics")) {
        std::size_t n = parameters.array_size() / 3;
        Value targets = Value::array();
        for (int c = 1; c <= 2; ++c) {
            Value block = Value::array();
            for (std::size_t i = 0; i < n; ++i)
                block.push(parameters.array_value(c * n + i));
            targets.push(block);
        }
        brain["target_critics"] = targets;
    }
}
Value start_native_run(Value s) {
    const auto config = s.at("config");
    auto mode = s.text("mode", "train");
    if (mode != "train" && mode != "evaluate" && mode != "random")
        throw std::runtime_error("mode must be train, evaluate, or random");
    bool initial = !s.contains("checkpoint_version");
    if (!initial && s.at("checkpoint_version").as_int() != 1)
        throw std::runtime_error("Unsupported checkpoint version");
    if (!initial && s.text("brain_precision", "float64") != "float32")
        throw std::runtime_error(
            "Native episode genesis requires FP32-stored brains");
    if (s.contains("freeze_evaluation_brains") &&
        !std::holds_alternative<bool>(s.at("freeze_evaluation_brains").data))
        throw std::runtime_error("Frozen brains require a boolean setting");
    const auto &settings = config.at("experiment");
    if (!initial) {
        auto default_field = [&](const std::string &key,
                                 const Value &fallback) {
            if (!s.contains(key))
                s[key] = fallback;
        };
        default_field("initial_models", Value::object());
        default_field("initial_weights", Value::object());
        default_field("run_initial_models", s.at("initial_models"));
        for (auto key : {"initial_models", "run_initial_models"})
            for (auto &[name, groups] : s[key].fields()) {
                (void)name;
                if (groups.contains("model")) {
                    auto shared = groups;
                    groups = object({{"female", shared}, {"male", shared}});
                }
            }
        for (auto &[name, groups] : s["initial_weights"].fields()) {
            (void)name;
            if (groups.contains("signature")) {
                auto shared = groups;
                groups = object({{"female", shared}, {"male", shared}});
            }
        }
        default_field("champion_artifacts", Value::object());
        for (auto &[name, groups] : s["champion_artifacts"].fields()) {
            (void)name;
            for (auto metric :
                 {"health", "kills", "lifespan", "food", "balanced", "lineage"})
                if (groups.contains(metric)) {
                    groups = Value::object();
                    break;
                }
        }
        default_field("legacy_brain_retention",
                      !settings.contains("chain_best_brains"));
        default_field("guard_lineage_chaining",
                      settings.flag("guard_lineage_chaining", false));
        default_field("improvement_only",
                      settings.flag("promote_only_improved", false));
        default_field("freeze_evaluation_brains", false);
        default_field("retired_updates", 0);
        default_field("inherited_updates", Value::object());
        default_field("randomization_seed", Value{});
        default_field("seed_rng_state", Value{});
        if (settings.text("world_seed_policy") == "random" &&
            s.get("seed_rng_state").null())
            throw std::runtime_error(
                "Randomized checkpoint is missing the seed generator state");
        default_field(
            "early_stopping",
            early_state(config, mode, s.at("initial_models"), Value::object()));
    }
    if (initial) {
        Value models = s.get("initial_models");
        if (models.null())
            models = Value::object();
        for (auto &[name, groups] : models.fields())
            if (groups.contains("model")) {
                auto model = groups;
                groups = object({{"female", model}, {"male", model}});
            }
        bool improvement = settings.flag("promote_only_improved", true);
        Value artifacts = Value::object();
        if (improvement)
            for (const auto &[name, groups] : models.fields())
                for (const auto &[sex, m] : groups.fields())
                    if (m.get("provenance").text("sex") == sex) {
                        auto p = m.at("provenance");
                        p["parent_artifact"] = m.get("champion_parent");
                        const auto &origin = m.get("champion_origin");
                        if (!origin.null())
                            for (const auto &[k, v] : origin.fields())
                                p[k] = v;
                        artifacts[name][sex][p.text("criterion")] = p;
                    }
        s["checkpoint_version"] = 1;
        s["brain_precision"] = "float32";
        s["initial_models"] = models;
        s["run_initial_models"] = models;
        s["initial_weights"] = Value::object();
        s["champion_artifacts"] = artifacts;
        s["legacy_brain_retention"] = false;
        s["guard_lineage_chaining"] =
            settings.flag("guard_lineage_chaining", true);
        s["improvement_only"] = improvement;
        s["freeze_evaluation_brains"] = s.flag("freeze_evaluation_brains");
        if (s.flag("freeze_evaluation_brains") && mode != "evaluate")
            throw std::runtime_error("Frozen brains require evaluate mode");
        s["mode"] = mode;
        s["run_index"] = 0;
        s["brains"] = Value::object();
        s["retired_updates"] = 0;
        s["inherited_updates"] = Value::object();
        s["history"] = Value::array();
        s["summaries"] = Value::array();
        s["run_seeds"] = Value::array();
        s["randomization_seed"] = settings.text("world_seed_policy") == "random"
                                      ? s.get("randomization_seed")
                                      : Value{};
        s["seed_rng_state"] = Value{};
    }
    int run = s.integer("run_index");
    bool improvement = s.flag("improvement_only"),
         legacy = s.flag("legacy_brain_retention");
    Value best =
        improvement
            ? (run > 0 ? s.at("champions")
                       : initial_champions(config, s.at("initial_models")))
            : Value{};
    auto retire = [&](const std::string &id) {
        const auto &b = s.at("brains").at(id);
        auto inherited = s.at("inherited_updates").get(id);
        s["retired_updates"] = s.at("retired_updates").i64() +
                               b.at("updates").i64() -
                               (inherited.null() ? 0 : inherited.i64());
        s["inherited_updates"].erase(id);
        s["brains"].erase(id);
    };
    if (run > 0 && !legacy) {
        auto models = s.at("initial_models");
        if (settings.flag("chain_best_brains", true) && mode == "train")
            for (const auto &[name, spec] : config.at("species").fields())
                for (const auto &sex : sexes) {
                    (void)spec;
                    const auto &original =
                        s.at("initial_models").get(name).get(sex);
                    auto metric =
                        criterion(config, name, sex,
                                  original.get("provenance").text("criterion"));
                    const auto &candidate =
                        s.at("champions").get(name).get(sex).get(metric);
                    if (candidate.null())
                        continue;
                    const auto &previous =
                        s.at("run_initial_models").get(name).get(sex);
                    if (improvement && !previous.null() &&
                        previous.get("provenance").text("criterion") ==
                            metric &&
                        equivalent(previous.at("provenance").get("rank"),
                                   candidate.at("rank"))) {
                        models[name][sex] = previous;
                        continue;
                    }
                    if (metric == "lineage" &&
                        s.flag("guard_lineage_chaining") &&
                        !candidate.get("lineage").flag("qualified") &&
                        !previous.null() &&
                        previous.get("provenance").flag("lineage_qualified")) {
                        auto retained = previous;
                        retained["provenance"]["retained_after_run"] = run;
                        retained["provenance"]["retention_reason"] =
                            "unqualified_lineage_candidate";
                        models[name][sex] = retained;
                        continue;
                    }
                    auto provenance = s.at("champion_artifacts")
                                          .get(name)
                                          .get(sex)
                                          .get(metric);
                    if (improvement && !provenance.null() &&
                        !equivalent(provenance.get("rank"),
                                    candidate.at("rank")))
                        provenance = Value{};
                    if (provenance.null()) {
                        const auto &parent = previous.get("provenance");
                        provenance =
                            object({{"artifact_id",
                                     "run:" + std::to_string(run) + ":" +
                                         std::to_string(
                                             s.at("world").at("seed").u64()) +
                                         ":" + name + ":" + sex + ":" + metric},
                                    {"path", Value{}},
                                    {"criterion", metric},
                                    {"score", candidate.at("score")},
                                    {"sex", sex},
                                    {"model_generation",
                                     parent.integer("model_generation") + 1},
                                    {"source_run",
                                     candidate.get("source").contains("run")
                                         ? candidate.at("source").at("run")
                                         : Value(run)},
                                    {"rank", candidate.at("rank")}});
                    }
                    provenance.erase("parent_artifact");
                    models[name][sex] =
                        object({{"model", candidate.at("model")},
                                {"provenance", provenance}});
                    if (metric == "lineage")
                        models[name][sex]["provenance"]["lineage_qualified"] =
                            candidate.get("lineage").flag("qualified");
                }
        s["run_initial_models"] = models;
        std::vector<std::string> ids;
        for (const auto &[id, b] : s.at("brains").fields()) {
            (void)b;
            ids.push_back(id);
        }
        for (const auto &id : ids)
            retire(id);
    }
    if (!improvement)
        s["champion_artifacts"] = Value::object();
    auto policy = settings.text("world_seed_policy");
    std::uint64_t seed;
    if (policy == "random") {
        if (s.get("randomization_seed").null())
            throw std::runtime_error(
                "Randomized native genesis needs randomization_seed");
        Random rng(s.at("randomization_seed").u64());
        if (!s.get("seed_rng_state").null())
            rng.restore(s.at("seed_rng_state"));
        std::set<std::uint32_t> used;
        for (const auto &v : s.at("run_seeds").items())
            used.insert(static_cast<std::uint32_t>(v.u64()));
        do {
            seed = rng.bits(32);
        } while (used.count(static_cast<std::uint32_t>(seed)));
        s["seed_rng_state"] = rng.checkpoint();
    } else
        seed = (config.at("seed").u64() + (policy == "increment" ? run : 0)) &
               0xffffffffULL;
    s["run_seeds"].push(seed);
    std::uint64_t brain_seed =
        policy == "random" ? seed : config.at("seed").u64();
    auto world_state = make_world(config, seed);
    bool social = false;
    for (const auto &[name, spec] : config.at("species").fields()) {
        (void)name;
        const auto &b = config.at("brains").at(spec.text("brain"));
        social |= !b.get("social_observation").null() ||
                  b.flag("consequence_observation");
    }
    if (social) {
        SocialState fresh(config);
        fresh.set_training(mode == "train");
        auto social_state = fresh.checkpoint();
        const auto &previous = s.get("world").get("social_state");
        if (!previous.null()) {
            SocialState old(config, previous);
            if (old.has_knowledge()) {
                if (mode == "train")
                    old.publish();
                social_state["knowledge"] = old.knowledge_checkpoint();
            }
        }
        world_state["social_state"] = social_state;
    }
    World world(world_state);
    std::set<std::string> live;
    for (const auto &a : world.animals())
        live.insert(std::to_string(a.integer("id")));
    std::vector<std::string> stale;
    for (const auto &[id, b] : s.at("brains").fields()) {
        (void)b;
        if (!live.count(id))
            stale.push_back(id);
    }
    for (const auto &id : stale)
        retire(id);
    Value signatures = Value::object();
    if (s.at("brains").fields().empty() ||
        !settings.flag("retain_brains_between_runs")) {
        s["brains"] = Value::object();
        int species_index = 0;
        for (const auto &[name, spec] : config.at("species").fields()) {
            auto b = config.at("brains").at(spec.text("brain"));
            double multiplier = spec.num("learning_rate_multiplier", 1.0);
            if (multiplier != 1.0)
                b["learning_rate"] = b.num("learning_rate") * multiplier;
            auto actions = action_names(spec.at("allowed_actions"));
            auto signature = native_model_signature(b, actions);
            std::uint64_t base_seed =
                brain_seed + 1000003ULL * (++species_index);
            std::map<std::string, Value> templates;
            for (const auto &a : world.animals()) {
                if (a.text("species") != name)
                    continue;
                auto sex = a.text("sex"), id = std::to_string(a.integer("id"));
                auto individual = words(
                    brain_seed + std::uint64_t(a.integer("id")) * 7919 + 13);
                Value brain;
                auto found = templates.find(sex);
                if (found != templates.end())
                    brain = inherit_controller_state(
                        found->second, found->second, individual, false);
                else {
                    brain = make_controller_state(
                        b, actions,
                        words(base_seed ^ (sex == "male" ? 0x4D414C45ULL : 0)),
                        individual);
                    const auto &initial_model =
                        s.at("run_initial_models").get(name).get(sex);
                    const auto &initial_weights =
                        s.at("initial_weights").get(name).get(sex);
                    if (!initial_model.null())
                        load_native_model_weights(
                            brain, initial_model.at("model"), signature);
                    else if (!legacy && !initial_weights.null())
                        load_native_model_weights(brain, initial_weights,
                                                  signature);
                    if (run == 0 &&
                        !s.at("initial_weights").get(name).contains(sex))
                        s["initial_weights"][name][sex] =
                            native_model_snapshot(brain, signature);
                    templates[sex] = brain;
                }
                s["brains"][id] = brain;
                signatures[id] = signature;
            }
        }
    } else
        for (const auto &[id, b] : s.at("brains").fields())
            signatures[id] = native_model_signature(
                b.at("config"), action_names(b.at("actions")));
    Champions champions(config, best, Value{});
    if (run == 0)
        s["early_stopping"] = early_state(
            config, mode, s.at("initial_models"),
            best.null() ? initial_champions(config, s.at("initial_models"))
                        : best);
    if (mode == "train") {
        Value source = object({{"experiment", config.at("name")},
                               {"mode", mode},
                               {"run", run + 1},
                               {"seed", seed},
                               {"tick", world.tick()}});
        std::map<int, Value> brains, sigs;
        for (const auto &[id, b] : s.at("brains").fields())
            brains[std::stoi(id)] = b;
        for (const auto &[id, v] : signatures.fields())
            sigs[std::stoi(id)] = v;
        champions.consider(world.animals(), brains, source, {}, sigs);
    }
    auto observed = observe(world);
    bool masks = false, central = false;
    for (const auto &[id, b] : s.at("brains").fields()) {
        (void)id;
        masks |= b.at("config").text("type") == "maskable_ppo";
        central |= b.at("config").text("type") == "mappo";
    }
    std::unique_ptr<DecisionContext> decision;
    if (masks)
        decision = std::make_unique<DecisionContext>(world);
    std::map<std::string, std::vector<double>> contexts;
    if (central)
        contexts = central_contexts(world);
    for (const auto &a : world.animals()) {
        auto id = std::to_string(a.integer("id"));
        auto &b = s["brains"][id];
        auto kind = b.at("config").text("type");
        if (kind == "maskable_ppo") {
            Value mask = Value::array();
            for (bool valid :
                 decision->action_mask(a, action_names(b.at("actions"))))
                mask.push(valid);
            b["mask"] = mask;
        } else if (kind == "mappo")
            b["context"] = b["next_context"] =
                values(contexts.at(a.text("species")));
    }
    s["champions"] = champions.entries();
    s["lineage"] = champions.lineage();
    s["world"] = world.checkpoint();
    s["finished_run"] = false;
    s["history"] = array({world_metrics(world)});
    Value observations = Value::object();
    for (const auto &[id, o] : observed)
        observations[std::to_string(id)] =
            world_observation_values(o, s.at("brains")
                                            .at(std::to_string(id))
                                            .at("config")
                                            .integer("observation_version", 1));
    return object({{"checkpoint", s},
                   {"observations", observations},
                   {"signatures", signatures}});
}
} // namespace wtfcore
