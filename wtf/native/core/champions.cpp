#include "champions.hpp"
#include <set>
namespace wtfcore {
namespace {
const std::vector<std::string> criteria = {"health", "kills",    "lifespan",
                                           "food",   "balanced", "lineage"};
bool greater_rank(const Value &a, const Value &b) {
    if (b.null())
        return true;
    const auto &x = a.items();
    const auto &y = b.items();
    for (std::size_t i = 0; i < std::min(x.size(), y.size()); ++i) {
        if (x[i].number() > y[i].number())
            return true;
        if (x[i].number() < y[i].number())
            return false;
    }
    return x.size() > y.size();
}
Value lineage_policy(const Value &config, const std::string &name) {
    const auto &spec = config.at("species").at(name);
    return object(
        {{"version", 1},
         {"family", "founder_descendants"},
         {"diet", spec.at("diet")},
         {"min_age_steps",
          spec.get("champion_selection").contains("lineage_min_age_steps")
              ? spec.at("champion_selection").at("lineage_min_age_steps")
              : Value(std::max(1, spec.get("reproduction")
                                      .integer("min_age_steps", 1)))}});
}
Value balanced_policy(const Value &config, const std::string &name) {
    const auto &p = config.at("species").at(name).get("champion_selection");
    return object(
        {{"version", 1},
         {"survival_steps_per_kill", p.contains("survival_steps_per_kill")
                                         ? p.at("survival_steps_per_kill")
                                         : Value(300)},
         {"max_survival_bonus", p.contains("max_survival_bonus")
                                    ? p.at("max_survival_bonus")
                                    : Value(2)}});
}
Value balanced_rank(const Value &a, const Value &p) {
    Value bonus = 0;
    if (a.num("kills") > 0) {
        Value fraction = a.num("age") / p.num("survival_steps_per_kill");
        bonus = minimum(fraction, p.at("max_survival_bonus"));
    }
    return array({add(a.at("kills"), bonus), a.at("kills"), a.at("age"),
                  a.at("energy_gained"), a.at("health")});
}
Value model_snapshot(const Value &brain, const Value &signature) {
    Value model = object(
        {{"signature", signature},
         {"precision", brain.contains("precision") ? brain.at("precision")
                                                   : Value("float64")},
         {"parameters", brain.at("config").text("type") == "q_network"
                            ? brain.at("weights")
                            : brain.at("network")}});
    if (brain.contains("auxiliary"))
        model["auxiliary_parameters"] = brain.at("auxiliary").at("network");
    if (brain.contains("late"))
        model["late_parameters"] = brain.at("late").at("network");
    return model;
}
} // namespace
Champions::Champions(const Value &config, const Value &entries,
                     const Value &lineage)
    : config_(config), entries_(Value::object()),
      lineage_(object(
          {{"version", 1},
           {"partial", lineage.flag("partial")},
           {"nodes",
            lineage.contains("nodes") ? lineage.at("nodes") : Value::object()},
           {"families", lineage.contains("families") ? lineage.at("families")
                                                     : Value::object()}})),
      policies_(Value::object()) {
    for (const auto &spec : config.at("species").fields())
        policies_[spec.first] = lineage_policy(config, spec.first);
    if (!lineage_.at("nodes").fields().empty() &&
        lineage_.at("families").fields().empty()) {
        for (const auto &node : lineage_.at("nodes").fields())
            add_node(std::stoi(node.first), node.second, Value{}, true);
    }
    if (!entries.null())
        for (const auto &group : entries.fields()) {
            bool legacy = false;
            for (const auto &metric : criteria)
                legacy |= group.second.contains(metric);
            if (!legacy) {
                entries_[group.first] = group.second;
                continue;
            }
            for (const auto &entry : group.second.fields()) {
                std::string sex = entry.second.get("source").text("sex");
                if (sex == "female" || sex == "male")
                    entries_[group.first][sex][entry.first] = entry.second;
            }
        }
}
void Champions::add_node(int id, const Value &node, const Value &previous,
                         bool is_new) {
    for (const auto &root : node.at("roots").items()) {
        std::string key = std::to_string(root.as_int());
        auto &families = lineage_["families"];
        if (!families.contains(key))
            families[key] = object({{"members", 0},
                                    {"viable_descendants", 0},
                                    {"kills", 0},
                                    {"energy_gained", 0.0},
                                    {"animal_steps", 0}});
        auto &family = families[key];
        family["members"] = family.integer("members") + int(is_new);
        family["viable_descendants"] =
            family.integer("viable_descendants") +
            int(node.flag("viable") && !previous.flag("viable") &&
                root.as_int() != id);
        for (const auto &field : {"kills", "energy_gained", "animal_steps"}) {
            Value delta = subtract(node.at(field), previous.contains(field)
                                                       ? previous.at(field)
                                                       : Value(0));
            family[field] = add(family.at(field), delta);
        }
    }
}
void Champions::observe_lineage(const Value::Array &animals) {
    std::vector<const Value *> ordered;
    for (const auto &a : animals)
        ordered.push_back(&a);
    std::stable_sort(ordered.begin(), ordered.end(),
                     [](const auto *a, const auto *b) {
                         return a->integer("id") < b->integer("id");
                     });
    for (const auto *ap : ordered) {
        const auto &a = *ap;
        int id = a.integer("id");
        std::string key = std::to_string(id);
        Value old = lineage_.at("nodes").get(key), node;
        if (old.null()) {
            std::vector<Value> parents;
            for (int parent : {a.integer("parent_id", -1),
                               a.integer("second_parent_id", -1)}) {
                const auto &candidate =
                    lineage_.at("nodes").get(std::to_string(parent));
                if (!candidate.null() &&
                    candidate.text("species") == a.text("species"))
                    parents.push_back(candidate);
            }
            std::set<int> roots;
            int generation = -1;
            for (const auto &p : parents) {
                generation = std::max(generation, p.integer("generation"));
                for (const auto &r : p.at("roots").items())
                    roots.insert(r.as_int());
            }
            Value root_array = Value::array();
            for (int root : roots)
                root_array.push(root);
            if (roots.empty())
                root_array.push(id);
            node = object({{"species", a.at("species")},
                           {"roots", root_array},
                           {"generation", generation + 1}});
            if (parents.size() < 2 && a.integer("parent_id", -1) >= 0)
                lineage_["partial"] = true;
        } else
            node = old;
        node["animal_steps"] = a.at("age");
        node["kills"] = a.at("kills");
        node["energy_gained"] = a.at("energy_gained");
        node["viable"] =
            a.integer("age") >=
            policies_.at(a.text("species")).integer("min_age_steps");
        add_node(id, node, old, old.null());
        lineage_["nodes"][key] = std::move(node);
    }
}
std::pair<Value, Value> Champions::lineage_rank(const Value &a,
                                                bool metadata) const {
    const auto &node = lineage_.at("nodes").at(std::to_string(a.integer("id")));
    bool predator = policies_.at(a.text("species")).text("diet") == "prey",
         qualified =
             node.flag("viable") &&
             (predator ? a.num("kills") > 0 : a.num("energy_gained") > 0);
    Value best;
    int best_root = -1;
    for (const auto &r : node.at("roots").items()) {
        int root = r.as_int();
        const auto &family = lineage_.at("families").at(std::to_string(root));
        Value primary =
                  predator ? family.at("kills") : family.at("animal_steps"),
              secondary = predator ? family.at("animal_steps")
                                   : family.at("energy_gained");
        double rate =
            (predator ? family.num("kills") : family.num("energy_gained")) /
            std::max(1.0, family.num("animal_steps"));
        Value rank = array(
            {qualified ? node.at("generation") : Value(-1),
             qualified ? primary : Value(0),
             qualified ? family.at("viable_descendants") : Value(0),
             qualified ? secondary : Value(0),
             qualified ? Value(rate) : Value(0),
             predator ? a.at("kills") : a.at("age"),
             predator ? a.at("age") : a.at("energy_gained"), a.at("health")});
        if (greater_rank(rank, best)) {
            best = rank;
            best_root = root;
        }
    }
    Value details;
    if (metadata) {
        const auto &family =
            lineage_.at("families").at(std::to_string(best_root));
        details = object({{"founder_id", best_root},
                          {"generation", node.at("generation")},
                          {"qualified", qualified},
                          {"partial_history", lineage_.at("partial")}});
        for (const auto &f : family.fields())
            details[f.first] = f.second;
        double time = std::max(1.0, family.num("animal_steps"));
        details["kills_per_1000_animal_steps"] =
            (1000 * family.num("kills")) / time;
        details["energy_per_animal_step"] = family.num("energy_gained") / time;
    }
    return {best, details};
}
void Champions::consider(const Value::Array &tracked,
                         const std::function<Value(int)> &brains,
                         const Value &source, const Value::Array &newborns,
                         const std::map<int, Value> &signatures) {
    Value::Array observed = tracked;
    observed.insert(observed.end(), newborns.begin(), newborns.end());
    observe_lineage(observed);
    struct Winner {
        Value rank;
        const Value *animal;
    };
    struct Group {
        std::string species, sex;
        std::vector<std::pair<std::string, Winner>> selected;
    };
    std::vector<Group> winners;
    for (const auto &a : tracked) {
        std::string species = a.text("species"), sex = a.text("sex");
        if (!entries_.contains(species))
            entries_[species] = Value::object();
        if (!entries_[species].contains(sex))
            entries_[species][sex] = Value::object();
        auto group =
            std::find_if(winners.begin(), winners.end(), [&](const auto &g) {
                return g.species == species && g.sex == sex;
            });
        if (group == winners.end()) {
            winners.push_back({species, sex, {}});
            group = std::prev(winners.end());
        }
        Value scores = object(
            {{"health", array({a.at("health"), a.at("age"), a.at("kills"),
                               a.at("energy_gained")})},
             {"kills", array({a.at("kills"), a.at("age"), a.at("energy_gained"),
                              a.at("health")})},
             {"lifespan", array({a.at("age"), a.at("kills"),
                                 a.at("energy_gained"), a.at("health")})},
             {"food", array({a.at("energy_gained"), a.at("age"), a.at("kills"),
                             a.at("health")})},
             {"lineage", lineage_rank(a, false).first}});
        if (config_.at("species").at(species).text("diet") == "prey")
            scores["balanced"] =
                balanced_rank(a, balanced_policy(config_, species));
        for (const auto &score : scores.fields()) {
            auto selected = std::find_if(
                group->selected.begin(), group->selected.end(),
                [&](const auto &v) { return v.first == score.first; });
            Value previous =
                selected != group->selected.end() ? selected->second.rank
                : entries_.at(species).at(sex).contains(score.first)
                    ? entries_.at(species).at(sex).at(score.first).at("rank")
                    : Value{};
            if (greater_rank(score.second, previous)) {
                if (selected == group->selected.end())
                    group->selected.push_back(
                        {score.first, {score.second, &a}});
                else
                    selected->second = {score.second, &a};
            }
        }
    }
    for (const auto &group : winners) {
        std::map<int, Value> models, metadata;
        for (const auto &selected : group.selected) {
            const auto &a = *selected.second.animal;
            int id = a.integer("id");
            if (!models.count(id)) {
                Value brain = brains(id);
                models[id] = model_snapshot(brain, signatures.at(id));
                Value meta = source;
                meta["animal_id"] = id;
                meta["species"] = a.at("species");
                meta["health"] = a.at("health");
                meta["alive"] = a.num("health") > 0;
                for (const auto *field :
                     {"age", "kills", "energy_gained", "parent_id"})
                    meta[field] = a.at(field);
                meta["body_generation"] = a.at("generation");
                for (const auto *field :
                     {"sex", "second_parent_id", "statistics_since_age"})
                    meta[field] = a.at(field);
                meta["optimizer_updates"] = brain.at("updates");
                if (a.integer("mutation_count")) {
                    for (const auto &field :
                         {"mutated", "mutation_count", "mutation_weights"})
                        meta[field] = a.at(field);
                }
                metadata[id] = std::move(meta);
            }
            const auto &metric = selected.first;
            Value entry = object({{"criterion", metric},
                                  {"score", selected.second.rank.items()[0]},
                                  {"rank", selected.second.rank},
                                  {"source", metadata.at(id)},
                                  {"model", models.at(id)}});
            if (metric == "balanced")
                entry["selection_policy"] =
                    balanced_policy(config_, group.species);
            else if (metric == "lineage") {
                entry["selection_policy"] = policies_.at(group.species);
                entry["lineage"] = lineage_rank(a, true).second;
            }
            entries_[group.species][group.sex][metric] = std::move(entry);
        }
    }
}
} // namespace wtfcore
