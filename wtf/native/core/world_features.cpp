#include "world.hpp"
#include <cmath>
#include <deque>
#include <tuple>

namespace wtfcore {
namespace {
using Pos = CellPosition;
constexpr int directions[4][2] = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
Pos position(const Value &a) { return {a.integer("x"), a.integer("y")}; }
Value number(const Value &v, const char *key, Value fallback) {
    return v.contains(key) ? v.at(key) : fallback;
}
Value make_event(const std::string &kind, int actor, int target = -1,
                 Value amount = 0.0, const std::string &cause = "",
                 int mate = -1, Value location = Value{}) {
    return object({{"kind", kind},
                   {"actor", actor},
                   {"target", target},
                   {"amount", amount},
                   {"cause", cause},
                   {"mate", mate},
                   {"position", location}});
}
Value location(const Value &a) { return array({a.at("x"), a.at("y")}); }
int round_even(double value) {
    double low = std::floor(value), fraction = value - low;
    return static_cast<int>(fraction < 0.5             ? low
                            : fraction > 0.5           ? low + 1
                            : std::fmod(low, 2.0) == 0 ? low
                                                       : low + 1);
}
Value scalar(double value, bool floating) {
    return floating ? Value(value) : Value(static_cast<std::int64_t>(value));
}
} // namespace
TerrainGrid::TerrainGrid(const Value &value, int w, int h)
    : width(w), height(h), definition(value), elevation(w * h, 0.0),
      depth(w * h, 0.0), flow_x(w * h, 0.0), flow_y(w * h, 0.0),
      numeric_kinds(w * h, 0) {
    if (value.contains("map_file"))
        throw std::runtime_error(
            "Native geography requires a resolved map definition");
    if (value.integer("width") != w || value.integer("height") != h ||
        value.at("rows").items().size() != static_cast<std::size_t>(h))
        throw std::runtime_error(
            "Native geography dimensions differ from world");
    for (int y = 0; y < h; ++y) {
        const auto &row = value.at("rows").items()[y].str();
        if (row.size() != static_cast<std::size_t>(w))
            throw std::runtime_error("Invalid native geography row");
        for (int x = 0; x < w; ++x) {
            const auto &cell = value.at("legend").at(std::string(1, row[x]));
            auto i = static_cast<std::size_t>(y * w + x);
            Value e = number(cell, "elevation", 0),
                  d = number(cell, "water_depth", 0);
            if (!std::isfinite(e.number()) || e.number() < 0 ||
                e.number() > 100 || !std::isfinite(d.number()) ||
                d.number() < 0 || d.number() > 10)
                throw std::runtime_error("Invalid native terrain scalar");
            if (cell.contains("flow")) {
                const auto &f = cell.at("flow").items();
                if (f.size() != 2)
                    throw std::runtime_error("Invalid native river flow");
                for (const auto &v : f)
                    if (!std::isfinite(v.number()) || v.number() < -4 ||
                        v.number() > 4 || (d.number() <= 0 && v.number() != 0))
                        throw std::runtime_error("Invalid native river flow");
            }
            if (d.number() > 0 && cell.flag("blocked"))
                throw std::runtime_error(
                    "Native terrain cannot be both blocked and wet");
            if (e.number() != 0) {
                elevation[i] = e.number();
                if (!e.integer())
                    numeric_kinds[i] |= 1;
            }
            if (d.number() > 0) {
                depth[i] = d.number();
                if (!d.integer())
                    numeric_kinds[i] |= 2;
                const auto f =
                    cell.contains("flow") ? cell.at("flow") : array({0, 0});
                if (f.items().size() != 2)
                    throw std::runtime_error("Invalid native river flow");
                flow_x[i] = f.items()[0].number();
                flow_y[i] = f.items()[1].number();
                if (!f.items()[0].integer())
                    numeric_kinds[i] |= 4;
                if (!f.items()[1].integer())
                    numeric_kinds[i] |= 8;
            }
        }
    }
}
Value World::elevation_value(int x, int y) const {
    if (!terrain_ || x < 0 || y < 0 || x >= width_ || y >= height_)
        return 0;
    auto i = static_cast<std::size_t>(y * width_ + x);
    return scalar(terrain_->elevation[i], terrain_->numeric_kinds[i] & 1);
}
Value World::depth_value(int x, int y) const {
    if (!terrain_ || x < 0 || y < 0 || x >= width_ || y >= height_)
        return 0;
    auto i = static_cast<std::size_t>(y * width_ + x);
    return scalar(terrain_->depth[i], terrain_->numeric_kinds[i] & 2);
}
std::pair<Value, Value> World::flow_values(int x, int y) const {
    if (!terrain_ || x < 0 || y < 0 || x >= width_ || y >= height_)
        return {0, 0};
    auto i = static_cast<std::size_t>(y * width_ + x);
    return {scalar(terrain_->flow_x[i], terrain_->numeric_kinds[i] & 4),
            scalar(terrain_->flow_y[i], terrain_->numeric_kinds[i] & 8)};
}
std::pair<double, double> World::flow(int x, int y) const {
    auto value = flow_values(x, y);
    return {value.first.number(), value.second.number()};
}
double World::light() const {
    const auto &cycle = config.get("day_night");
    if (!cycle.flag("enabled"))
        return 1.0;
    int period = cycle.integer("cycle_steps");
    if (period <= 0)
        throw std::runtime_error("Invalid native day/night period");
    double phase = std::fmod(cycle.num("start_phase", 0.25) +
                                 static_cast<double>(tick() % period) / period,
                             1.0);
    return phase > 0.25 && phase < 0.75
               ? std::cos(2 * 3.14159265358979323846 * (phase - 0.5))
               : 0.0;
}
void World::refresh_bodies() const {
    Value fresh = Value::object();
    double sunlight = light();
    for (const auto &[name, spec] : config.at("species").fields()) {
        Value b = spec.at("body");
        const auto &rules = spec.get("circadian");
        if (rules.boolean()) {
            for (const auto *field :
                 {"vision_radius", "living_cost", "movement_cost"}) {
                double night = rules.num(
                    std::string("night_") + field + "_multiplier", 1.0);
                double value = b.num(field) * (night + (1 - night) * sunlight);
                b[field] =
                    std::string(field) == "vision_radius"
                        ? Value(std::max(
                              1, std::min(256, static_cast<int>(
                                                   std::floor(value + 0.5)))))
                        : Value(value);
            }
        }
        fresh[name] = std::move(b);
    }
    cycle_bodies_ = std::move(fresh);
    body_tick_ = tick();
}
const Value &World::body_for_species(const std::string &name) const {
    if (!config.get("day_night").flag("enabled"))
        return config.at("species").at(name).at("body");
    if (body_tick_ != tick())
        refresh_bodies();
    return cycle_bodies_.at(name);
}
const Value &World::body(const Value &a) const {
    return body_for_species(a.text("species"));
}
CellPosition World::movement_destination(const Value &a,
                                         const std::string &action) const {
    int offset = action == "move_backward"  ? 2
                 : action == "strafe_left"  ? -1
                 : action == "strafe_right" ? 1
                                            : 0;
    int heading = (a.integer("heading") + offset + 4) % 4;
    return {a.integer("x") + directions[heading][0],
            a.integer("y") + directions[heading][1]};
}
Value World::terrain_move_cost(const Value &a, int x, int y,
                               const std::string &action) const {
    if (!terrain_)
        return 0.0;
    const auto &rules = species(a).get("locomotion");
    Value gain = maximum(
        Value(0.0), subtract(elevation_value(x, y),
                             elevation_value(a.integer("x"), a.integer("y"))));
    Value cost = multiply(gain, number(rules, "uphill_energy_cost", 1.0));
    if (action == "swim") {
        Value water = maximum(depth_value(a.integer("x"), a.integer("y")),
                              depth_value(x, y));
        cost =
            add(cost, multiply(water, number(rules, "swim_energy_cost", 1.5)));
    }
    return cost;
}
std::string World::movement_reason(const Value &a, int x, int y,
                                   const std::string &action,
                                   bool occupied) const {
    if (blocked(x, y))
        return "obstacle";
    if (occupied)
        return "occupied";
    if (!terrain_)
        return action == "swim" ? "not_near_water" : "";
    const auto &rules = species(a).get("locomotion");
    double source = depth(a.integer("x"), a.integer("y")), target = depth(x, y);
    if ((action == "move_forward" || action == "move_backward" ||
         action == "strafe_left" || action == "strafe_right") &&
        (source || target))
        return "requires_swim";
    if (action == "swim") {
        double water = std::max(source, target);
        if (!water)
            return "not_near_water";
        double minimum =
            rules.num("swim_min_health", body(a).num("max_health") * 0.5) *
            water;
        if (a.num("health") < minimum)
            return "health";
    }
    double gain = elevation(x, y) - elevation(a.integer("x"), a.integer("y"));
    if (gain > rules.num("max_climb", 1.0))
        return "slope";
    auto cost = terrain_move_cost(a, x, y, action);
    if (cost.number() != 0 && a.num("energy") <= cost.number())
        return "energy";
    return "";
}
std::string
World::movement_block_reason(const Value &a, int x, int y,
                             const std::string &action,
                             const std::set<CellPosition> *occupied) const {
    return movement_reason(a, x, y, action,
                           occupied && occupied->count({x, y}));
}
void World::terrain_charge(Value &a, const Value &amount, Value &events,
                           const std::string &cause) {
    if (amount.number() != 0) {
        a["energy"] = maximum(Value(0.0), subtract(a.at("energy"), amount));
        events.push(
            make_event("energy_spent", a.integer("id"), -1, amount, cause));
    }
}
int World::water_sight_cells(int x, int y, int tx, int ty) const {
    if (!terrain_)
        return 0;
    int dx = std::abs(tx - x), dy = -std::abs(ty - y), sx = x < tx ? 1 : -1,
        sy = y < ty ? 1 : -1, error = dx + dy, count = 0;
    while (x != tx || y != ty) {
        int twice = 2 * error;
        if (twice >= dy) {
            error += dy;
            x += sx;
        }
        if (twice <= dx) {
            error += dx;
            y += sy;
        }
        if ((x != tx || y != ty) && depth(x, y))
            ++count;
    }
    return count;
}
bool World::mating_land_route(const Value &a, const Value &b,
                              int radius) const {
    if (!terrain_)
        return true;
    auto start = position(a), end = position(b);
    if (depth(start.first, start.second) || depth(end.first, end.second))
        return false;
    double limit = std::min(species(a).get("locomotion").num("max_climb", 1.0),
                            species(b).get("locomotion").num("max_climb", 1.0));
    std::deque<std::pair<Pos, int>> queue{{start, 0}};
    std::set<Pos> seen{start};
    while (!queue.empty()) {
        auto [p, hops] = queue.front();
        queue.pop_front();
        if (p == end)
            return true;
        if (hops >= 2 * radius)
            continue;
        for (const auto &d : directions) {
            Pos q{p.first + d[0], p.second + d[1]};
            if (seen.count(q) || blocked(q.first, q.second) ||
                depth(q.first, q.second))
                continue;
            if (std::abs(elevation(q.first, q.second) -
                         elevation(p.first, p.second)) > limit)
                continue;
            seen.insert(q);
            queue.push_back({q, hops + 1});
        }
    }
    return false;
}
bool World::birth_cell_allowed(const Value &parent, int x, int y) const {
    if (blocked(x, y))
        return false;
    return !terrain_ ||
           (!depth(x, y) &&
            std::abs(elevation(x, y) -
                     elevation(parent.integer("x"), parent.integer("y"))) <=
                species(parent).get("locomotion").num("max_climb", 1.0));
}
void World::river_drift(Value &events) {
    if (!terrain_)
        return;
    struct Path {
        int id;
        std::vector<Pos> offsets;
        bool active = true;
    };
    std::vector<Path> paths;
    std::size_t longest = 0;
    for (auto &a : animals()) {
        auto [x, y] = position(a);
        if (!depth(x, y)) {
            a["drift_x"] = 0.0;
            a["drift_y"] = 0.0;
            continue;
        }
        auto current = flow(x, y);
        double tx = a.num("drift_x") + current.first,
               ty = a.num("drift_y") + current.second;
        int nx = static_cast<int>(tx >= 0 ? std::floor(tx + 1e-12)
                                          : std::ceil(tx - 1e-12));
        int ny = static_cast<int>(ty >= 0 ? std::floor(ty + 1e-12)
                                          : std::ceil(ty - 1e-12));
        a["drift_x"] = std::abs(tx - nx) < 1e-12 ? 0.0 : tx - nx;
        a["drift_y"] = std::abs(ty - ny) < 1e-12 ? 0.0 : ty - ny;
        int n = std::max(std::abs(nx), std::abs(ny));
        if (n) {
            Path path{a.integer("id"), {}, true};
            for (int k = 1; k <= n; ++k)
                path.offsets.push_back(
                    {round_even(double(k * nx) / n) -
                         round_even(double((k - 1) * nx) / n),
                     round_even(double(k * ny) / n) -
                         round_even(double((k - 1) * ny) / n)});
            longest = std::max(longest, path.offsets.size());
            paths.push_back(std::move(path));
        }
    }
    for (std::size_t step = 0; step < longest; ++step) {
        std::set<Pos> occupied;
        for (const auto &a : animals())
            occupied.insert(position(a));
        std::vector<std::pair<Pos, std::vector<int>>> claims;
        for (auto &path : paths) {
            if (!path.active || step >= path.offsets.size())
                continue;
            auto &a = animal(path.id);
            auto [x, y] = position(a);
            if (!depth(x, y)) {
                a["drift_x"] = 0.0;
                a["drift_y"] = 0.0;
                path.active = false;
                continue;
            }
            auto [dx, dy] = path.offsets[step];
            Pos q{x + dx, y + dy};
            bool corner =
                dx && dy && (blocked(x + dx, y) || blocked(x, y + dy));
            if (corner ||
                !movement_block_reason(a, q.first, q.second, "drift", &occupied)
                     .empty()) {
                events.push(make_event("drift_blocked", path.id));
                path.active = false;
                continue;
            }
            auto claim = std::find_if(
                claims.begin(), claims.end(),
                [&](const auto &entry) { return entry.first == q; });
            if (claim == claims.end())
                claims.push_back({q, {path.id}});
            else
                claim->second.push_back(path.id);
        }
        for (const auto &[q, ids] : claims) {
            int winner = ids[rng.below(ids.size())];
            auto &a = animal(winner);
            terrain_charge(a, terrain_move_cost(a, q.first, q.second, "drift"),
                           events, "uphill");
            a["x"] = q.first;
            a["y"] = q.second;
            events.push(make_event("drifted", winner, -1, 0.0, "current"));
            if (!depth(q.first, q.second)) {
                a["drift_x"] = 0.0;
                a["drift_y"] = 0.0;
            }
            for (int id : ids)
                if (id != winner) {
                    events.push(make_event("drift_blocked", id, -1, 0.0,
                                           "competing_claim"));
                    for (auto &path : paths)
                        if (path.id == id)
                            path.active = false;
                }
        }
    }
}
namespace {
bool local_food(const World &world, const Value &a, Pos &found) {
    const auto &b = world.body(a);
    if (world.species(a).text("diet") != "resource" ||
        a.num("energy") >= b.num("max_energy") || b.num("bite_size") <= 0 ||
        b.num("energy_per_food_unit") <= 0)
        return false;
    for (auto p : {position(a), world.movement_destination(a, "move_forward")})
        for (const auto &patch : world.resources())
            if (position(patch) == p && patch.num("quantity") > 0) {
                found = p;
                return true;
            }
    return false;
}
} // namespace
std::pair<int, std::string> World::challenge_opportunity(
    const Value &a, const std::map<CellPosition, int> &positions) const {
    const auto &rules = species(a).get("competition");
    if (!rules.flag("enabled") || a.num("health") <= 0 ||
        a.num("energy") <= rules.num("energy_cost", 1.0) ||
        depth(a.integer("x"), a.integer("y")))
        return {-1, ""};
    auto eligible = [&](int id) {
        if (id < 0 || id == a.integer("id"))
            return false;
        const auto &target = animal(id);
        return target.text("species") == a.text("species") &&
               target.num("health") > 0 &&
               !depth(target.integer("x"), target.integer("y"));
    };
    auto target = positions.find(movement_destination(a, "move_forward"));
    if (target != positions.end() && eligible(target->second)) {
        Pos left, right;
        if (rules.flag("food", true) && local_food(*this, a, left) &&
            local_food(*this, animal(target->second), right) && left == right)
            return {target->second, "food"};
        if (rules.flag("blocked_path", true))
            return {target->second, "blocked_path"};
        if (rules.flag("aggression", true))
            return {target->second, "aggression"};
    }
    if (rules.flag("aggression", true)) {
        int best = -1;
        for (const auto &d : directions) {
            auto candidate =
                positions.find({a.integer("x") + d[0], a.integer("y") + d[1]});
            if (candidate != positions.end() && eligible(candidate->second) &&
                (best < 0 || candidate->second < best))
                best = candidate->second;
        }
        if (best >= 0)
            return {best, "aggression"};
    }
    return {-1, ""};
}
void World::contest_advance(int winner_id, int loser_id, Value &events,
                            std::set<Pos> &occupied) {
    auto &winner = animal(winner_id);
    auto &loser = animal(loser_id);
    Pos destination = position(loser);
    bool present = occupied.erase(destination) != 0;
    auto reason =
        movement_block_reason(winner, destination.first, destination.second,
                              "move_forward", &occupied);
    if (present)
        occupied.insert(destination);
    if (!reason.empty())
        return;
    Pos retreat{};
    bool has_retreat = false;
    if (loser.num("health") > 0) {
        std::vector<Pos> choices;
        for (const auto &d : directions)
            choices.push_back(
                {loser.integer("x") + d[0], loser.integer("y") + d[1]});
        auto key = [&](Pos p) {
            int dx = p.first - winner.integer("x"),
                dy = p.second - winner.integer("y");
            return std::make_tuple(-(dx * dx + dy * dy), p);
        };
        std::sort(choices.begin(), choices.end(),
                  [&](Pos a, Pos b) { return key(a) < key(b); });
        for (auto p : choices)
            if (movement_block_reason(loser, p.first, p.second, "move_forward",
                                      &occupied)
                    .empty()) {
                retreat = p;
                has_retreat = true;
                break;
            }
        if (!has_retreat)
            return;
    }
    for (int phase = 0; phase < 2; ++phase) {
        if (phase == 0 && !has_retreat)
            continue;
        auto &a = phase == 0 ? loser : winner;
        Pos target = phase == 0 ? retreat : destination;
        Value cost = add(
            body(a).at("movement_cost"),
            terrain_move_cost(a, target.first, target.second, "move_forward"));
        terrain_charge(a, cost, events, "competition_movement");
        occupied.erase(position(a));
        a["x"] = target.first;
        a["y"] = target.second;
        occupied.insert(target);
        events.push(make_event(phase == 0 ? "displaced" : "contest_moved",
                               a.integer("id"), -1, 0.0, "competition", -1,
                               array({target.first, target.second})));
    }
}
CompetitionResult
World::resolve_competition(const std::map<int, std::string> &actions,
                           Value &events) {
    std::map<Pos, int> positions;
    std::vector<int> challengers;
    for (const auto &a : animals()) {
        positions[position(a)] = a.integer("id");
        if (actions.at(a.integer("id")) == "challenge")
            challengers.push_back(a.integer("id"));
    }
    std::map<int, std::pair<int, std::string>> candidates;
    for (int id : challengers) {
        auto &a = animal(id);
        auto candidate = challenge_opportunity(a, positions);
        const auto &rules = species(a).get("competition");
        if (rules.flag("enabled"))
            terrain_charge(a, number(rules, "energy_cost", 1.0), events,
                           "competition_attempt");
        if (candidate.second == "food" &&
            actions.at(candidate.first) != "eat" &&
            actions.at(candidate.first) != "challenge") {
            if (rules.flag("blocked_path", true))
                candidate.second = "blocked_path";
            else if (rules.flag("aggression", true))
                candidate.second = "aggression";
            else
                candidate = {-1, ""};
        }
        candidates[id] = candidate;
        events.push(make_event("challenge_attempt", id, candidate.first, 0.0,
                               candidate.second.empty() ? "unavailable"
                                                        : candidate.second,
                               -1, location(a)));
    }
    CompetitionResult result;
    std::set<int> engaged;
    std::set<Pos> occupied;
    bool occupancy_ready = false;
    for (int id : challengers) {
        const auto &[target_id, kind] = candidates.at(id);
        if (target_id < 0 || engaged.count(id) || engaged.count(target_id))
            continue;
        auto &a = animal(id);
        auto &target = animal(target_id);
        if (a.num("health") <= 0 || target.num("health") <= 0)
            continue;
        if (kind == "aggression"
                ? std::abs(a.integer("x") - target.integer("x")) +
                          std::abs(a.integer("y") - target.integer("y")) !=
                      1
                : movement_destination(a, "move_forward") != position(target))
            continue;
        engaged.insert(id);
        engaged.insert(target_id);
        for (auto pair :
             {std::make_pair(id, target_id), std::make_pair(target_id, id)}) {
            if (actions.at(pair.first) == "challenge" &&
                candidates.at(pair.first).first == pair.second) {
                result.successful.insert(pair.first);
                events.push(make_event("challenge_resolved", pair.first,
                                       pair.second, 0.0, kind));
            }
        }
        auto strength = [&](const Value &animal) {
            const auto &b = body(animal);
            return animal.num("health") / b.num("max_health") +
                   animal.num("energy") / b.num("max_energy");
        };
        double left = strength(a), right = strength(target);
        int winner_id = left > right   ? id
                        : right > left ? target_id
                                       : (rng.below(2) ? target_id : id);
        int loser_id = winner_id == id ? target_id : id;
        auto &winner = animal(winner_id);
        auto &loser = animal(loser_id);
        const auto &rules = species(winner).at("competition");
        Value damage =
            minimum(loser.at("health"), number(rules, "damage", 5.0));
        loser["health"] = subtract(loser.at("health"), damage);
        events.push(make_event("contest_won", winner_id, loser_id, damage, kind,
                               -1, location(winner)));
        events.push(make_event("contest_lost", loser_id, winner_id, damage,
                               kind, -1, location(loser)));
        if (damage.number() != 0) {
            events.push(make_event("damage_dealt", winner_id, loser_id, damage,
                                   "competition"));
            events.push(make_event("damage_taken", loser_id, winner_id, damage,
                                   "competition"));
        }
        if (loser.num("health") <= 0) {
            result.deaths.insert(loser_id);
            if (occupancy_ready)
                occupied.erase(position(loser));
        }
        if (kind == "food")
            result.priority.push_back(winner_id);
        else if (kind == "blocked_path" && winner_id == id) {
            if (!occupancy_ready) {
                for (const auto &animal : animals())
                    if (animal.num("health") > 0)
                        occupied.insert(position(animal));
                occupancy_ready = true;
            }
            contest_advance(winner_id, loser_id, events, occupied);
        }
    }
    return result;
}
} // namespace wtfcore
