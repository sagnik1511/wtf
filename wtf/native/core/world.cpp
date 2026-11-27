#include "world.hpp"
#include "observation_layout.hpp"
#include "sensing.hpp"
#include "social.hpp"
#include <cmath>
#include <tuple>
namespace wtfcore {
namespace {
using Pos = std::pair<int, int>;
const int DX[] = {0, 1, 0, -1}, DY[] = {-1, 0, 1, 0};
Pos pos(const Value &a) { return {a.integer("x"), a.integer("y")}; }
bool walk(const std::string &a) {
    return a == "move_forward" || a == "move_backward" || a == "strafe_left" ||
           a == "strafe_right";
}
bool is_move(const std::string &a) { return walk(a) || a == "swim"; }
bool attempt(const std::string &a) {
    return is_move(a) || a == "attack" || a == "eat" || a == "reproduce" ||
           a == "challenge";
}
Pos front(const Value &a, int offset = 0) {
    int h = (a.integer("heading") + offset + 4) % 4;
    return {a.integer("x") + DX[h], a.integer("y") + DY[h]};
}
Value event(const std::string &kind, int actor, int target = -1,
            Value amount = 0.0, const std::string &cause = "", int mate = -1) {
    return object({{"kind", kind},
                   {"actor", actor},
                   {"target", target},
                   {"amount", amount},
                   {"cause", cause},
                   {"mate", mate},
                   {"position", Value{}}});
}
void increment(Value &v, const std::string &key, Value amount = 1) {
    v[key] = add(v.contains(key) ? v.at(key) : Value(0), amount);
}
int deviation(const Value &r, bool first) {
    std::string p = first ? "age" : "cooldown";
    int center = r.integer(first ? "min_age_steps" : "cooldown_steps");
    return r.get(p + "_deviation_pct").null()
               ? r.integer(p + "_deviation_steps")
               : static_cast<int>(center * r.num(p + "_deviation_pct") / 100 +
                                  0.5);
}
bool window(const Value &r) {
    return r.num("age_deviation_pct", r.num("age_deviation_steps")) ||
           r.num("cooldown_deviation_pct", r.num("cooldown_deviation_steps"));
}
int interval(const Value &r, Random &rng, bool first = false) {
    int c = r.integer(first ? "min_age_steps" : "cooldown_steps"),
        d = deviation(r, first);
    return d ? rng.randint(std::max(1, c - d), c + d) : c;
}
int due(const Value &a, const Value &r) {
    if (a.integer("reproduction_due_age", -1) >= 0)
        return a.integer("reproduction_due_age");
    bool first = a.integer("last_reproduced_age", -1) < 0;
    return (first ? 0 : a.integer("last_reproduced_age")) +
           std::max(1, r.integer(first ? "min_age_steps" : "cooldown_steps") -
                           deviation(r, first));
}
std::string personal(const World &world, const Value &a, const Value &r) {
    if (r.null() || !r.flag("enabled"))
        return "disabled";
    if (a.integer("age") < due(a, r))
        return a.integer("last_reproduced_age", -1) < 0 ? "immature"
                                                        : "cooldown";
    if (a.num("health") <= r.num("health_threshold"))
        return "low_health";
    if (a.num("energy") < r.num("min_energy") ||
        a.num("energy") <= r.num("energy_cost"))
        return "low_energy";
    if (world.depth(a.integer("x"), a.integer("y")))
        return "in_water";
    return "";
}
bool ready(Value &a, const Value &r, Random &rng) {
    if (!window(r))
        return a.integer("age") >= r.integer("min_age_steps") &&
               (a.integer("last_reproduced_age", -1) < 0 ||
                a.integer("age") - a.integer("last_reproduced_age") >=
                    r.integer("cooldown_steps"));
    if (a.integer("reproduction_due_age", -1) < 0) {
        bool f = a.integer("last_reproduced_age", -1) < 0;
        a["reproduction_due_age"] =
            (f ? 0 : a.integer("last_reproduced_age")) + interval(r, rng, f);
    }
    return a.integer("age") >= a.integer("reproduction_due_age");
}
std::vector<Pos> birth_cells(World &w, const Value &m, const Value &f,
                             const std::set<Pos> &occupied) {
    std::vector<Pos> out;
    for (const Value *p : {&m, &f})
        for (int i = 0; i < 4; ++i) {
            Pos q = {p->integer("x") + DX[i], p->integer("y") + DY[i]};
            if (w.birth_cell_allowed(*p, q.first, q.second) &&
                !occupied.count(q) &&
                std::find(out.begin(), out.end(), q) == out.end())
                out.push_back(q);
        }
    return out;
}
Value newborn(int id, const Value &m, const Value &f, const Value &b, Pos p,
              int heading, std::string sex) {
    return object(
        {{"id", id},
         {"species", m.at("species")},
         {"x", p.first},
         {"y", p.second},
         {"heading", heading},
         {"energy", b.at("initial_energy")},
         {"health", b.at("max_health")},
         {"age", 0},
         {"last_action", "rest"},
         {"last_reward", 0.0},
         {"parent_id", m.at("id")},
         {"generation",
          std::max(m.integer("generation"), f.integer("generation")) + 1},
         {"last_reproduced_age", -1},
         {"kills", 0},
         {"energy_gained", 0.0},
         {"statistics_since_age", 0},
         {"reproduction_due_age", -1},
         {"mutated", false},
         {"mutation_count",
          std::max(m.integer("mutation_count"), f.integer("mutation_count"))},
         {"mutation_weights", 0},
         {"sex", sex},
         {"second_parent_id", f.at("id")},
         {"drift_x", 0.0},
         {"drift_y", 0.0},
         {"mating_intent_until_tick", -1},
         {"last_mating_failure", ""}});
}
} // namespace
World::World(const Value &checkpoint,
             std::shared_ptr<const TerrainGrid> static_geography)
    : state(checkpoint), config(checkpoint.at("config")),
      frame(checkpoint.at("snapshot")) {
    width_ = config.at("world").integer("width");
    height_ = config.at("world").integer("height");
    if (width() < 1 || height() < 1 || width() > 256 || height() > 256)
        throw std::runtime_error("Invalid native world dimensions");
    if (frame.integer("width", width()) != width() ||
        frame.integer("height", height()) != height())
        throw std::runtime_error(
            "Native world snapshot dimensions differ from config");
    if (config.contains("terrain")) {
        if (static_geography && static_geography->width == width() &&
            static_geography->height == height() &&
            equivalent(static_geography->definition, config.at("terrain")))
            terrain_ = std::move(static_geography);
        else
            terrain_ = std::make_shared<const TerrainGrid>(config.at("terrain"),
                                                           width(), height());
    }
    blocked_.assign(static_cast<std::size_t>(width()) * height(), 0);
    for (const auto &point : frame.at("obstacles").items()) {
        if (point.items().size() != 2)
            throw std::runtime_error("Invalid obstacle coordinate");
        int x = point.items()[0].as_int(), y = point.items()[1].as_int();
        if (x >= 0 && y >= 0 && x < width() && y < height())
            blocked_[y * width() + x] = 1;
    }
    std::set<int> ids;
    for (const auto &a : animals()) {
        int id = a.integer("id");
        if (id < 0 || !ids.insert(id).second || a.integer("x") < 0 ||
            a.integer("y") < 0 || a.integer("x") >= width() ||
            a.integer("y") >= height() || a.integer("heading") < 0 ||
            a.integer("heading") > 3)
            throw std::runtime_error(
                "Invalid native animal identity or position");
    }
    rng.restore(state.at("rng_state"));
    index_animals();
    if (!state.get("social_state").null())
        social_ =
            std::make_unique<SocialState>(config, state.at("social_state"));
}
World::~World() = default;
std::shared_ptr<const ObservationLayout>
World::observation_layout(const std::string &name) {
    auto found = layouts_.find(name);
    if (found == layouts_.end())
        found = layouts_
                    .emplace(name, shared_observation_layout(
                                       config.at("brains").at(name)))
                    .first;
    return found->second;
}
void World::index_animals() {
    animal_index_.clear();
    const auto &bodies = animals();
    for (std::size_t i = 0; i < bodies.size(); ++i)
        if (!animal_index_.emplace(bodies[i].integer("id"), i).second)
            throw std::runtime_error("Duplicate native animal identity");
}
Value &World::animal(int id) {
    auto found = animal_index_.find(id);
    if (found == animal_index_.end())
        throw std::runtime_error("Missing native animal identity");
    return animals().at(found->second);
}
const Value &World::animal(int id) const {
    auto found = animal_index_.find(id);
    if (found == animal_index_.end())
        throw std::runtime_error("Missing native animal identity");
    return animals().at(found->second);
}
bool World::blocked(int x, int y) const {
    if (x < 0 || y < 0 || x >= width() || y >= height())
        return true;
    return blocked_[y * width() + x] != 0;
}
bool World::visible(int x, int y, int tx, int ty) const {
    int dx = std::abs(tx - x), dy = -std::abs(ty - y), sx = x < tx ? 1 : -1,
        sy = y < ty ? 1 : -1, e = dx + dy;
    while (x != tx || y != ty) {
        int e2 = 2 * e;
        if (e2 >= dy) {
            e += dy;
            x += sx;
        }
        if (e2 <= dx) {
            e += dx;
            y += sy;
        }
        if (blocked(x, y))
            return false;
    }
    return true;
}
WorldStep World::advance(const std::map<int, std::string> &actions) {
    if (actions.size() != animals().size())
        throw std::runtime_error(
            "Exactly one action required for every living animal");
    for (const auto &a : animals()) {
        auto it = actions.find(a.integer("id"));
        if (it == actions.end())
            throw std::runtime_error(
                "Exactly one action required for every living animal");
        bool allowed = false;
        for (const auto &s : species(a).at("allowed_actions").items())
            allowed |= s.str() == it->second;
        if (!allowed)
            throw std::runtime_error("Unsupported native action " + it->second);
    }
    if (social_)
        social_->begin(*this, actions);
    WorldStep result;
    auto &events = result.events;
    std::set<Pos> occupied;
    for (const auto &a : animals())
        occupied.insert(pos(a));
    std::vector<std::pair<Pos, std::vector<int>>> claims;
    for (auto &a : animals()) {
        int id = a.integer("id");
        auto action = actions.at(id);
        a["last_action"] = action;
        if (action == "turn_left")
            a["heading"] = (a.integer("heading") + 3) % 4;
        else if (action == "turn_right")
            a["heading"] = (a.integer("heading") + 1) % 4;
        else if (is_move(action)) {
            int off = action == "move_backward"  ? 2
                      : action == "strafe_left"  ? -1
                      : action == "strafe_right" ? 1
                                                 : 0;
            Pos q = front(a, off);
            std::string reason =
                movement_block_reason(a, q.first, q.second, action, &occupied);
            if (!reason.empty()) {
                std::string cause = action == "swim" ? "swim:" + reason
                                    : walk(action) && action != "move_forward"
                                        ? action + ":" + reason
                                    : has_terrain() ? reason
                                                    : "";
                events.push(event("movement_blocked", id, -1, 0.0, cause));
            } else {
                auto it =
                    std::find_if(claims.begin(), claims.end(),
                                 [&](const auto &v) { return v.first == q; });
                if (it == claims.end())
                    claims.push_back({q, {id}});
                else
                    it->second.push_back(id);
            }
        }
    }
    for (auto &claim : claims) {
        int winner = claim.second[rng.below(claim.second.size())];
        auto &a = animal(winner);
        auto action = actions.at(winner);
        if (has_terrain())
            terrain_charge(a,
                           terrain_move_cost(a, claim.first.first,
                                             claim.first.second, action),
                           events, action == "swim" ? "swim" : "uphill");
        a["x"] = claim.first.first;
        a["y"] = claim.first.second;
        events.push(event("moved", winner, -1, 0.0,
                          action == "move_forward" ? "" : action));
        for (int id : claim.second)
            if (id != winner) {
                auto ac = actions.at(id);
                events.push(event("movement_blocked", id, -1, 0.0,
                                  ac == "move_forward"
                                      ? "competing_claim"
                                      : ac + ":competing_claim"));
            }
    }
    river_drift(events);
    bool has_challenges =
        std::any_of(actions.begin(), actions.end(), [](const auto &entry) {
            return entry.second == "challenge";
        });
    CompetitionResult competition;
    if (has_challenges)
        competition = resolve_competition(actions, events);
    std::vector<int> eaters;
    for (const auto &a : animals())
        if (actions.at(a.integer("id")) == "eat")
            eaters.push_back(a.integer("id"));
    rng.shuffle(eaters);
    if (!competition.priority.empty()) {
        std::vector<int> ordered = competition.priority;
        for (int id : eaters)
            if (std::find(competition.priority.begin(),
                          competition.priority.end(),
                          id) == competition.priority.end())
                ordered.push_back(id);
        eaters = std::move(ordered);
    }
    for (int id : eaters) {
        if (competition.deaths.count(id))
            continue;
        auto &a = animal(id);
        const auto &b = body(a);
        Value *patch = nullptr;
        for (auto p : {pos(a), front(a)}) {
            for (auto &r : resources())
                if (pos(r) == p && r.num("quantity") > 0) {
                    patch = &r;
                    break;
                }
            if (patch)
                break;
        }
        double conv = b.num("energy_per_food_unit");
        if (patch && conv > 0) {
            Value limit = (b.num("max_energy") - a.num("energy")) / conv;
            Value amount = minimum(
                minimum(patch->at("quantity"), b.at("bite_size")), limit);
            (*patch)["quantity"] = subtract(patch->at("quantity"), amount);
            Value gained = multiply(amount, b.at("energy_per_food_unit"));
            a["energy"] = add(a.at("energy"), gained);
            if (gained.number() > 0)
                events.push(event(
                    "energy_gained", id, -1, gained,
                    actions.at(id) == "eat" ? "resource" : "competition_food"));
        }
    }
    std::map<Pos, int> positions;
    for (const auto &a : animals())
        if (a.num("health") > 0)
            positions[pos(a)] = a.integer("id");
    std::vector<std::pair<int, std::vector<int>>> attacks;
    for (const auto &a : animals())
        if (actions.at(a.integer("id")) == "attack" && a.num("health") > 0) {
            auto it = positions.find(front(a));
            if (it != positions.end() &&
                species(animal(it->second)).text("diet") == "resource" &&
                body(a).num("attack_damage") > 0) {
                auto found = std::find_if(
                    attacks.begin(), attacks.end(),
                    [&](const auto &p) { return p.first == it->second; });
                if (found == attacks.end())
                    attacks.push_back({it->second, {a.integer("id")}});
                else
                    found->second.push_back(a.integer("id"));
            }
        }
    std::set<int> attack_deaths;
    for (const auto &entry : attacks) {
        int target_id = entry.first;
        const auto &ids = entry.second;
        auto &target = animal(target_id);
        std::int64_t integer_damage = 0;
        bool floating_damage = false;
        Sum damage_sum;
        for (int id : ids) {
            const auto &amount = body(animal(id)).at("attack_damage");
            if (!floating_damage && amount.integer())
                integer_damage += amount.i64();
            else if (!floating_damage) {
                damage_sum.value =
                    static_cast<double>(integer_damage) + amount.number();
                floating_damage = true;
            } else if (amount.integer())
                damage_sum.value += amount.number();
            else
                damage_sum.add(amount.number());
        }
        Value attempted = floating_damage ? Value(damage_sum.result())
                                          : Value(integer_damage);
        Value damage = minimum(target.at("health"), attempted);
        for (int id : ids)
            events.push(
                event("damage_dealt", id, target_id,
                      damage.number() * (body(animal(id)).num("attack_damage") /
                                         attempted.number()),
                      "attack"));
        target["health"] = subtract(target.at("health"), damage);
        events.push(event("damage_taken", target_id, -1, damage, "attack"));
        if (target.num("health") <= 0) {
            attack_deaths.insert(target_id);
            int killer_id = ids[rng.below(ids.size())];
            auto &killer = animal(killer_id);
            const auto &b = body(killer);
            Value available = subtract(b.at("max_energy"), killer.at("energy"));
            Value gained = minimum(b.at("energy_per_kill"), available);
            killer["energy"] = add(killer.at("energy"), gained);
            Value killed = event("kill", killer_id, target_id);
            if (has_terrain() || has_challenges)
                killed["position"] = array({target.at("x"), target.at("y")});
            events.push(std::move(killed));
            events.push(
                event("energy_gained", killer_id, target_id, gained, "prey"));
        }
    }
    std::set<int> drowned;
    for (auto &a : animals()) {
        int id = a.integer("id");
        const auto &b = body(a);
        Value cost = b.at("living_cost");
        if (is_move(actions.at(id)))
            cost = add(cost, b.at("movement_cost"));
        if (actions.at(id) == "attack")
            cost = add(cost, b.at("attack_cost"));
        Value energy = subtract(a.at("energy"), cost);
        a["energy"] = maximum(Value(0.0), energy);
        result.rewards[id] =
            cost.number() * species(a).at("rewards").num("per_energy_cost");
        a["age"] = a.integer("age") + 1;
        if (has_terrain() && a.num("health") > 0) {
            Value water = depth_value(a.integer("x"), a.integer("y"));
            if (water.number() != 0) {
                const auto &spec = species(a);
                const auto &rules = spec.get("locomotion");
                terrain_charge(
                    a,
                    multiply(water, rules.contains("water_energy_cost")
                                        ? rules.at("water_energy_cost")
                                        : Value(0.25)),
                    events, "water");
                double health_gate =
                    rules.num("swim_min_health", b.num("max_health") * 0.5) *
                    water.number();
                bool swims = false;
                for (const auto &entry : spec.at("allowed_actions").items())
                    swims |= entry.str() == "swim";
                if (!swims || a.num("health") < health_gate ||
                    a.num("energy") <= 0) {
                    Value damage = minimum(
                        a.at("health"),
                        multiply(water, rules.contains("drowning_damage")
                                            ? rules.at("drowning_damage")
                                            : Value(2.0)));
                    a["health"] = subtract(a.at("health"), damage);
                    if (damage.number() != 0)
                        events.push(
                            event("damage_taken", id, -1, damage, "drowning"));
                    if (a.num("health") <= 0)
                        drowned.insert(id);
                }
            }
        }
        if (a.num("energy") <= 0 && a.num("health") > 0) {
            Value damage = minimum(a.at("health"), b.at("starvation_damage"));
            a["health"] = subtract(a.at("health"), damage);
            events.push(event("damage_taken", id, -1, damage, "starvation"));
        }
        if (a.num("health") <= 0)
            result.dead.push_back(id);
    }
    std::sort(result.dead.begin(), result.dead.end());
    for (int id : result.dead) {
        const auto &a = animal(id);
        std::string cause = attack_deaths.count(id)        ? "predation"
                            : competition.deaths.count(id) ? "competition"
                            : drowned.count(id)            ? "drowning"
                                                           : "starvation";
        Value death = event("died", id, -1, 0.0, cause);
        if (has_terrain() || has_challenges)
            death["position"] = array({a.at("x"), a.at("y")});
        events.push(std::move(death));
        increment(frame["metrics"]["deaths"], cause);
        state["lifespans"].push(object({{"id", id},
                                        {"species", a.at("species")},
                                        {"age", a.at("age")},
                                        {"cause", cause}}));
    }
    std::set<int> successful = competition.successful;
    for (const auto &e : events.items()) {
        int id = e.integer("actor");
        auto &a = animal(id);
        const auto &r = species(a).at("rewards");
        auto kind = e.text("kind"), cause = e.text("cause");
        double amount = e.num("amount");
        if (kind == "moved" ||
            (kind == "damage_dealt" && cause != "competition" && amount > 0) ||
            (kind == "energy_gained" && cause == "resource" && amount > 0))
            successful.insert(id);
        if (kind == "energy_spent")
            result.rewards[id] += amount * r.num("per_energy_cost");
        else if (kind == "energy_gained") {
            result.rewards[id] += amount * r.num("per_energy_gained");
            auto &totals = frame["metrics"]["energy_gained"];
            totals[a.text("species")] = add(totals.contains(a.text("species"))
                                                ? totals.at(a.text("species"))
                                                : Value(0.0),
                                            e.at("amount"));
            a["energy_gained"] = add(a.at("energy_gained"), e.at("amount"));
        } else if (kind == "damage_dealt")
            result.rewards[id] += amount * r.num("per_damage_dealt");
        else if (kind == "damage_taken")
            result.rewards[id] += amount * r.num("per_damage_taken");
        else if (kind == "died")
            result.rewards[id] += r.num("death");
        else if (kind == "kill") {
            increment(frame["metrics"]["kills"], a.text("species"));
            increment(a, "kills");
        }
    }
    last_dead_animals = Value::array();
    for (const auto &a : animals())
        if (std::binary_search(result.dead.begin(), result.dead.end(),
                               a.integer("id")))
            last_dead_animals.push(a);
    auto &av = animals();
    av.erase(std::remove_if(av.begin(), av.end(),
                            [&](const auto &a) {
                                return std::binary_search(result.dead.begin(),
                                                          result.dead.end(),
                                                          a.integer("id"));
                            }),
             av.end());
    index_animals();
    std::vector<int> parents, eligible;
    std::map<std::string, int> counts;
    for (auto &a : animals()) {
        int id = a.integer("id");
        parents.push_back(id);
        ++counts[a.text("species")];
        const auto &r = species(a).get("reproduction");
        bool active = !r.null() && r.flag("enabled"),
             is_ready = active && ready(a, r, rng);
        if (a.text("last_action") == "reproduce") {
            a["last_mating_failure"] = "";
            a["mating_intent_until_tick"] =
                is_ready && personal(*this, a, r).empty()
                    ? tick() + r.integer("intent_steps", 1) - 1
                    : -1;
        } else if (a.text("last_action") != "rest")
            a["mating_intent_until_tick"] = -1;
        if (!is_ready || (active && !personal(*this, a, r).empty()))
            a["mating_intent_until_tick"] = -1;
        if (is_ready && a.integer("mating_intent_until_tick", -1) >= tick() &&
            !depth(a.integer("x"), a.integer("y")) &&
            a.num("health") > r.num("health_threshold") &&
            a.num("energy") >= r.num("min_energy") &&
            a.num("energy") > r.num("energy_cost"))
            eligible.push_back(id);
    }
    rng.shuffle(eligible);
    occupied.clear();
    for (const auto &a : animals())
        occupied.insert(pos(a));
    std::map<std::tuple<std::string, int, int>, int> males;
    for (int id : eligible) {
        const auto &a = animal(id);
        if (a.text("sex") == "male")
            males[{a.text("species"), a.integer("x"), a.integer("y")}] = id;
    }
    std::set<int> used;
    for (int mother_id : eligible) {
        if (animal(mother_id).text("sex") != "female")
            continue;
        const auto mother = animal(mother_id);
        const auto &spec = species(mother);
        const auto &r = spec.at("reproduction");
        std::string name = mother.text("species");
        if (counts[name] >= r.integer("max_population"))
            continue;
        int radius = r.integer("mate_radius", 2);
        std::vector<int> candidates;
        for (int dy = -radius; dy <= radius; ++dy)
            for (int dx = -radius; dx <= radius; ++dx) {
                if (dx * dx + dy * dy > radius * radius)
                    continue;
                auto it = males.find(
                    {name, mother.integer("x") + dx, mother.integer("y") + dy});
                if (it != males.end() && !used.count(it->second)) {
                    const auto &mate = animal(it->second);
                    if (visible(mother.integer("x"), mother.integer("y"),
                                mate.integer("x"), mate.integer("y")) &&
                        mating_land_route(mother, mate, radius))
                        candidates.push_back(it->second);
                }
            }
        if (candidates.empty())
            continue;
        rng.shuffle(candidates);
        int father_id = -1;
        std::vector<Pos> free;
        for (int id : candidates) {
            free = birth_cells(*this, mother, animal(id), occupied);
            if (!free.empty()) {
                father_id = id;
                break;
            }
        }
        if (father_id < 0)
            continue;
        Pos p = free[rng.below(free.size())];
        int child_id = state.integer("next_animal_id");
        if (child_id == std::numeric_limits<int>::max())
            throw std::runtime_error(
                "Native animal identity capacity exceeded");
        state["next_animal_id"] = child_id + 1;
        int heading = rng.below(4);
        std::string sex = rng.below(2) ? "male" : "female";
        Value child = newborn(child_id, mother, animal(father_id),
                              spec.at("body"), p, heading, sex);
        animals().push_back(std::move(child));
        animal_index_[child_id] = animals().size() - 1;
        occupied.insert(p);
        ++counts[name];
        for (int id : {mother_id, father_id}) {
            auto &parent = animal(id);
            used.insert(id);
            parent["mating_intent_until_tick"] = -1;
            parent["last_mating_failure"] = "";
            parent["energy"] =
                subtract(parent.at("energy"), r.at("energy_cost"));
            parent["last_reproduced_age"] = parent.at("age");
            if (window(r))
                parent["reproduction_due_age"] =
                    parent.integer("age") + interval(r, rng);
        }
        increment(frame["metrics"]["births"], name);
        events.push(event("born", mother_id, child_id, 0.0,
                          "paired_reproduction", father_id));
        for (int id : {mother_id, father_id}) {
            successful.insert(id);
            result.rewards[id] += spec.at("rewards").num("per_offspring");
            result.rewards[id] += r.num("energy_cost") *
                                  spec.at("rewards").num("per_energy_cost");
        }
    }
    std::map<Pos, int> final_occupied;
    for (const auto &a : animals())
        final_occupied[pos(a)] = a.integer("id");
    for (int id : parents) {
        auto &a = animal(id);
        if (a.text("last_action") != "reproduce" || used.count(id))
            continue;
        const auto &r = species(a).get("reproduction");
        std::string reason = personal(*this, a, r);
        if (reason.empty() &&
            counts[a.text("species")] >= r.integer("max_population"))
            reason = "population_cap";
        std::vector<int> nearby, reachable, readies;
        if (reason.empty()) {
            int radius = r.integer("mate_radius", 2);
            for (int dy = -radius; dy <= radius; ++dy)
                for (int dx = -radius; dx <= radius; ++dx) {
                    auto it = final_occupied.find(
                        {a.integer("x") + dx, a.integer("y") + dy});
                    if (dx * dx + dy * dy <= radius * radius &&
                        it != final_occupied.end()) {
                        const auto &m = animal(it->second);
                        if (m.text("species") == a.text("species") &&
                            m.text("sex") != a.text("sex"))
                            nearby.push_back(it->second);
                    }
                }
            if (nearby.empty())
                reason = "no_nearby_mate";
        }
        if (reason.empty()) {
            for (int mate : nearby) {
                const auto &m = animal(mate);
                if (visible(a.integer("x"), a.integer("y"), m.integer("x"),
                            m.integer("y")) &&
                    mating_land_route(a, m, r.integer("mate_radius", 2)))
                    reachable.push_back(mate);
            }
            if (reachable.empty())
                reason = "blocked_route";
        }
        if (reason.empty()) {
            for (int mate : reachable)
                if (used.count(mate)) {
                    reason = "mate_unavailable";
                    break;
                }
        }
        if (reason.empty()) {
            for (int mate : reachable)
                if (personal(*this, animal(mate), r).empty())
                    readies.push_back(mate);
            if (readies.empty())
                reason = "mate_not_ready";
        }
        if (reason.empty()) {
            bool willing = false;
            for (int mate : readies)
                willing |= animal(mate).integer("mating_intent_until_tick",
                                                -1) >= tick();
            reason = willing ? "no_birth_space" : "awaiting_consent";
        }
        a["last_mating_failure"] = reason;
        events.push(event("mating_failed", id, -1, 0.0, reason));
    }
    for (const auto &entry : actions) {
        int id = entry.first;
        const auto &action = entry.second;
        bool dead =
            std::binary_search(result.dead.begin(), result.dead.end(), id);
        Value *ap = nullptr;
        if (dead) {
            for (auto &a : last_dead_animals.items())
                if (a.integer("id") == id) {
                    ap = &a;
                    break;
                }
        } else
            ap = &animal(id);
        auto &a = *ap;
        const auto &r = species(a).at("rewards");
        if (action == "reproduce" && dead) {
            a["last_mating_failure"] = "parent_died";
            events.push(event("mating_failed", id, -1, 0.0, "parent_died"));
        }
        if (attempt(action) && !successful.count(id)) {
            double penalty = r.num("failed_action");
            result.rewards[id] += penalty;
            if (dead && penalty)
                events.push(event("action_failed", id, -1, penalty, action));
        }
        a["last_reward"] = result.rewards.at(id);
    }
    const auto &r = config.at("resources");
    for (auto &patch : resources()) {
        Value grown = add(patch.at("quantity"), r.at("growth_per_step"));
        patch["quantity"] = minimum(r.at("capacity_per_patch"), grown);
    }
    frame["tick"] = tick() + 1;
    advance_scent(*this, last_dead_animals);
    if (social_)
        social_->finish(*this, events, last_dead_animals);
    frame["events"] = events;
    frame["metrics"] = world_metrics(*this);
    return result;
}
Value World::checkpoint() {
    state["config"] = config;
    frame["metrics"] = world_metrics(*this);
    state["snapshot"] = frame;
    state["rng_state"] = rng.checkpoint();
    if (social_)
        state["social_state"] = social_->checkpoint();
    return state;
}
} // namespace wtfcore
