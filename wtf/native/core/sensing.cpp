#include "context.hpp"
#include "observation_layout.hpp"
#include "social.hpp"
#include "world.hpp"
#include <array>
#include <cmath>
#include <set>
#include <tuple>
namespace wtfcore {
namespace {
constexpr int directions[4][2] = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
using Point = std::pair<int, int>;
double norm(double x, double y) {
    // Compensate both products and their sum, as CPython's vector norm does.
    // This matters on arm64 macOS, where long double has double precision.
    x = std::abs(x);
    y = std::abs(y);
    if (x < y)
        std::swap(x, y);
    if (!x)
        return 0.0;
    const double xx = x * x, yy = y * y, sum = xx + yy;
    const double error =
        std::fma(x, x, -xx) + std::fma(y, y, -yy) + (xx - sum + yy);
    const double root = std::sqrt(sum);
    return root + (std::fma(-root, root, sum) + error) / (2 * root);
}

bool action(const Value &s, const std::string &name) {
    for (const auto &a : s.at("allowed_actions").items())
        if (a.str() == name)
            return true;
    return false;
}
Point relative(Point p, int h) {
    return {p.first * directions[h][0] + p.second * directions[h][1],
            -p.first * directions[h][1] + p.second * directions[h][0]};
}
std::pair<double, double> rotate(double x, double y, int h) {
    return {x * directions[h][0] + y * directions[h][1],
            -x * directions[h][1] + y * directions[h][0]};
}
int due(const Value &a, const Value &r) {
    int d = a.integer("reproduction_due_age", -1);
    if (d >= 0)
        return d;
    bool first = a.integer("last_reproduced_age", -1) < 0;
    std::string prefix = first ? "age" : "cooldown";
    int center = r.integer(first ? "min_age_steps" : "cooldown_steps");
    int deviation =
        r.get(prefix + "_deviation_pct").null()
            ? r.integer(prefix + "_deviation_steps")
            : int(center * r.num(prefix + "_deviation_pct") / 100 + 0.5);
    return (first ? 0 : a.integer("last_reproduced_age")) +
           std::max(1, center - deviation);
}
struct Target {
    int id, x, y, order;
};
struct SpeciesView {
    const Value *rules, *body, *brain;
    std::shared_ptr<const ObservationLayout> plan;
    int category, radius, radius_squared, bonus, limit, field_degrees, top_k,
        version, history_steps, mate_radius_squared, max_population,
        intent_steps;
    int population = 0;
    bool reproduce, attack, eat, swim, observe_attack, observe_eat,
        observe_swim;
    double max_energy, max_health, attack_damage, bite_size, energy_per_food;
    bool blended = false;
    int smell_radius = 0;
    std::vector<std::vector<int>> scent_groups;
    std::vector<std::size_t> members;
};
struct AnimalView {
    const std::string *sex;
    std::size_t species;
    int id, x, y, heading;
    bool male, birth_space = false;
    double energy, health, intent;
    std::array<double, 3> status;
};
// This read-only projection belongs to one observation phase. State,
// visibility, reproduction eligibility and scent deposits are never cached
// across steps.
struct ObservationPhase {
    World &world;
    DecisionContext decisions;
    int width, height, tick, channels, scent_lifetime = 1;
    std::vector<SpeciesView> species;
    std::vector<AnimalView> animals;
    std::array<std::vector<Target>, 3> targets;
    std::vector<int> occupied, first_occupant;
    std::set<Point> food_cells;
    std::map<int, int> scent_cells;
    std::map<int, std::size_t> visual_rows;
    std::vector<std::tuple<int, int, Target>> candidates;
    std::vector<unsigned> visible_stamp, scent_stamp;
    std::vector<unsigned char> visible_value;
    std::vector<std::pair<Point, int>> scent_queue;
    unsigned observer_stamp = 0;

    explicit ObservationPhase(World &w)
        : world(w), decisions(w), width(w.width()), height(w.height()),
          tick(w.tick()),
          channels(2 * int(w.config.at("species").fields().size())),
          occupied(width * height, -1), first_occupant(width * height, -1),
          visible_stamp(width * height), scent_stamp(width * height),
          visible_value(width * height) {
        const auto &configuration = w.config.at("species").fields();
        std::vector<std::string> names;
        for (const auto &[name, _] : configuration)
            names.push_back(name);
        std::sort(names.begin(), names.end());
        std::map<std::string, std::size_t> species_index;
        species.reserve(configuration.size());
        bool has_scent = false;
        const bool has_water =
            w.has_terrain() && std::any_of(w.geography()->depth.begin(),
                                           w.geography()->depth.end(),
                                           [](double d) { return d != 0; });
        for (const auto &[name, settings] : configuration) {
            SpeciesView view{};
            view.rules = &settings.get("reproduction");
            view.body = &w.body_for_species(name);
            view.brain = &w.config.at("brains").at(settings.text("brain"));
            const auto &body = *view.body, &brain = *view.brain;
            view.plan = w.observation_layout(settings.text("brain"));
            view.category = settings.text("diet") == "resource" ? 1 : 2;
            view.radius = body.integer("vision_radius");
            view.bonus = 0;
            if (has_water)
                view.bonus = body.integer("water_vision_bonus");
            view.limit = view.radius + view.bonus;
            view.radius_squared = view.limit * view.limit;
            view.field_degrees = body.integer("field_of_view_degrees", 360);
            view.top_k = brain.integer("observation_top_k", 1);
            view.version = brain.integer("observation_version", 1);
            view.history_steps = brain.integer("movement_history_steps");
            int mate_radius = view.rules->integer("mate_radius", 2);
            view.mate_radius_squared = mate_radius * mate_radius;
            view.max_population = view.rules->integer("max_population");
            view.intent_steps = view.rules->integer("intent_steps", 1);
            view.reproduce = !view.rules->null() &&
                             view.rules->flag("enabled") &&
                             action(settings, "reproduce");
            view.attack = action(settings, "attack");
            view.eat = action(settings, "eat");
            view.swim = action(settings, "swim");
            view.max_energy = body.num("max_energy");
            view.max_health = body.num("max_health");
            view.attack_damage = body.num("attack_damage");
            view.bite_size = body.num("bite_size");
            view.energy_per_food = body.num("energy_per_food_unit");
            view.observe_attack = view.observe_eat = view.observe_swim =
                brain.get("observation_actions").null();
            if (!brain.get("observation_actions").null())
                for (const auto &entry :
                     brain.at("observation_actions").items()) {
                    view.observe_attack |= entry.str() == "attack";
                    view.observe_eat |= entry.str() == "eat";
                    view.observe_swim |= entry.str() == "swim";
                }
            const auto &scent = brain.get("scent_observation");
            if (!scent.null()) {
                has_scent = true;
                view.blended = scent.text("mode", "separated") == "blended";
                view.smell_radius = scent.integer("radius", 2);
                if (view.blended) {
                    view.scent_groups.emplace_back();
                    for (int channel = 0; channel < channels; ++channel)
                        view.scent_groups.back().push_back(channel);
                } else {
                    for (const auto &entry : scent.at("species").items()) {
                        int base = 2 * int(std::find(names.begin(), names.end(),
                                                     entry.str()) -
                                           names.begin());
                        if (scent.flag("separate_sexes", true)) {
                            view.scent_groups.push_back({base});
                            view.scent_groups.push_back({base + 1});
                        } else {
                            view.scent_groups.push_back({base, base + 1});
                        }
                    }
                }
            }
            species_index[name] = species.size();
            species.push_back(std::move(view));
        }
        const auto &source = static_cast<const World &>(w).animals();
        animals.reserve(source.size());
        for (const auto &a : source) {
            auto index = species_index.at(a.text("species"));
            auto &spec = species[index];
            AnimalView animal{&a.at("sex").str(),
                              index,
                              a.integer("id"),
                              a.integer("x"),
                              a.integer("y"),
                              a.integer("heading"),
                              a.text("sex") == "male",
                              false,
                              a.num("energy"),
                              a.num("health"),
                              0,
                              {0, 0, 0}};
            animal.intent = std::min(
                1.0,
                double(std::max(0, a.integer("mating_intent_until_tick", -1) -
                                       tick + 1)) /
                    spec.intent_steps);
            if (spec.reproduce) {
                int deadline = due(a, *spec.rules),
                    last = a.integer("last_reproduced_age", -1),
                    age = a.integer("age");
                animal.status = {
                    last >= 0
                        ? 1
                        : std::min(1.0, double(age) / std::max(1, deadline)),
                    last < 0
                        ? 0
                        : std::min(1.0,
                                   double(std::max(0, deadline - (age + 1))) /
                                       std::max(1, deadline - last)),
                    decisions.eligible(a) ? 1.0 : 0.0};
            }
            int cell = animal.y * width + animal.x;
            occupied[cell] = int(animals.size());
            if (first_occupant[cell] < 0)
                first_occupant[cell] = int(animals.size());
            auto &category = targets[spec.category];
            category.push_back(
                {animal.id, animal.x, animal.y, int(category.size())});
            spec.members.push_back(animals.size());
            ++spec.population;
            animals.push_back(std::move(animal));
        }
        for (const auto &resource : static_cast<const World &>(w).resources()) {
            double quantity = resource.num("quantity");
            int x = resource.integer("x"), y = resource.integer("y");
            if (quantity > 0)
                food_cells.insert({x, y});
            if (quantity > 0.001)
                targets[0].push_back({-1, x, y, int(targets[0].size())});
        }
        if (has_scent) {
            scent_lifetime =
                w.config.at("scent_trails").integer("lifetime_steps");
            const auto &deposits = w.state.get("sensory_state").get("scent");
            if (!deposits.null())
                for (const auto &entry : deposits.items())
                    scent_cells[entry.items()[0].as_int()] =
                        entry.items()[1].as_int();
        }
        std::size_t largest = 0;
        for (const auto &category : targets)
            largest = std::max(largest, category.size());
        candidates.reserve(largest);
    }
    bool in_bounds(Point cell) const {
        return cell.first >= 0 && cell.second >= 0 && cell.first < width &&
               cell.second < height;
    }
    int occupant(Point cell) const {
        return in_bounds(cell) ? occupied[cell.second * width + cell.first]
                               : -1;
    }
    bool sees(const AnimalView &observer, int x, int y) {
        const auto &spec = species[observer.species];
        int cell = y * width + x;
        bool bounded = in_bounds({x, y});
        if (bounded && visible_stamp[cell] == observer_stamp)
            return visible_value[cell];
        int dx = x - observer.x, dy = y - observer.y;
        bool field = true;
        if (spec.field_degrees != 360) {
            int dot = dx * directions[observer.heading][0] +
                      dy * directions[observer.heading][1];
            double c = spec.field_degrees == 180 ? 0 : (2 - std::sqrt(3.0)) / 4;
            field = dot >= 0 && (!c || dot * dot >= (dx * dx + dy * dy) * c);
        }
        int range =
            spec.radius +
            std::min(spec.bonus, spec.bonus ? world.water_sight_cells(
                                                  observer.x, observer.y, x, y)
                                            : 0);
        bool visible = field && dx * dx + dy * dy <= range * range &&
                       world.visible(observer.x, observer.y, x, y);
        if (bounded) {
            visible_stamp[cell] = observer_stamp;
            visible_value[cell] = visible;
        }
        return visible;
    }
};
void mating(ObservationPhase &phase, const AnimalView &a,
            std::vector<double> &out, int version) {
    const auto &spec = phase.species[a.species];
    const AnimalView *best = nullptr;
    int bestdist = 0;
    bool pair = false, consent = false;
    for (auto index : spec.members) {
        const auto &b = phase.animals[index];
        if (*b.sex == *a.sex)
            continue;
        int dx = b.x - a.x, dy = b.y - a.y, distance = dx * dx + dy * dy;
        if (distance > spec.radius_squared || !phase.sees(a, b.x, b.y))
            continue;
        if (!best || std::make_pair(distance, b.id) <
                         std::make_pair(bestdist, best->id)) {
            best = &b;
            bestdist = distance;
        }
        if (!a.status[2] || !b.status[2] ||
            spec.population >= spec.max_population ||
            distance > spec.mate_radius_squared ||
            !phase.world.mating_land_route(
                phase.world.animal(a.id), phase.world.animal(b.id),
                spec.rules->integer("mate_radius", 2)))
            continue;
        if (phase.decisions.birth_space(phase.world.animal(a.id),
                                        phase.world.animal(b.id))) {
            pair = true;
            consent = consent || b.intent > 0;
        }
    }
    out.push_back(a.male ? 1 : -1);
    out.insert(out.end(), a.status.begin(), a.status.end());
    if (best) {
        double dx = double(best->x - a.x) / std::max(1, spec.limit),
               dy = double(best->y - a.y) / std::max(1, spec.limit);
        if (version >= 5) {
            auto p = rotate(dx, dy, a.heading);
            dx = p.first;
            dy = p.second;
        }
        out.push_back(dx);
        out.push_back(dy);
        if (version != 7)
            out.push_back(std::sqrt(double(bestdist)) /
                          std::max(1, spec.limit));
        out.push_back(1);
        out.push_back(best->status[2]);
    } else {
        auto zero = version >= 5 ? rotate(0.0, 0.0, a.heading)
                                 : std::make_pair(0.0, 0.0);
        out.push_back(zero.first);
        out.push_back(zero.second);
        out.insert(out.end(), version == 7 ? 2 : 3, 0);
    }
    out.push_back(pair);
    if (version >= 4) {
        out.push_back(a.intent);
        out.push_back(best ? best->intent : 0);
        out.push_back(consent);
    }
}
std::vector<Target> nearest(ObservationPhase &phase, const AnimalView &a,
                            const std::vector<Target> &targets, int k,
                            int radius) {
    auto &candidates = phase.candidates;
    candidates.clear();
    for (const auto &target : targets) {
        if (target.id == a.id)
            continue;
        int dx = target.x - a.x, dy = target.y - a.y,
            distance = dx * dx + dy * dy;
        if (distance <= radius * radius)
            candidates.emplace_back(distance, target.order, target);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto &left, const auto &right) {
                  return std::make_pair(std::get<0>(left), std::get<1>(left)) <
                         std::make_pair(std::get<0>(right), std::get<1>(right));
              });
    std::vector<Target> out;
    out.reserve(std::min<std::size_t>(k, candidates.size()));
    for (const auto &candidate : candidates) {
        const auto &target = std::get<2>(candidate);
        if (phase.sees(a, target.x, target.y)) {
            out.push_back(target);
            if (int(out.size()) == k)
                break;
        }
    }
    return out;
}
Value &sensory(World &w) {
    auto &s = w.state["sensory_state"];
    if (s.null())
        s = object({{"version", 1}, {"visual", Value::array()}});
    return s;
}
void history(ObservationPhase &phase, const AnimalView &a,
             const std::array<std::vector<Target>, 3> &targets, int steps,
             int k, int radius, int version, std::vector<double> &out) {
    World &w = phase.world;
    auto &visual = sensory(w)["visual"];
    if (visual.null())
        visual = Value::array();
    auto row_index = phase.visual_rows.find(a.id);
    if (row_index == phase.visual_rows.end()) {
        phase.visual_rows[a.id] = visual.items().size();
        visual.push(array({a.id, Value::array()}));
    }
    auto *frames = &visual.items()[phase.visual_rows.at(a.id)].items()[1];
    auto &fs = frames->items();
    while (!fs.empty() && fs.front().items()[0].as_int() < phase.tick - steps)
        fs.erase(fs.begin());
    Value current = Value::array();
    for (int c = 0; c < 2; ++c)
        for (auto t : targets[c + 1]) {
            // Python's personal sighting map resolves a position to its first
            // animal.
            if (phase.in_bounds({t.x, t.y})) {
                int first = phase.first_occupant[t.y * phase.width + t.x];
                if (first >= 0)
                    t.id = phase.animals[first].id;
            }
            auto point = array({t.id, c, t.x, t.y});
            bool replaced = false;
            for (auto &entry : current.items())
                if (entry.items()[0].as_int() == t.id) {
                    entry = point;
                    replaced = true;
                    break;
                }
            if (!replaced)
                current.push(point);
        }
    Value row = array({phase.tick, current});
    if (!fs.empty() && fs.back().items()[0].as_int() == phase.tick)
        fs.back() = row;
    else
        fs.push_back(row);
    std::vector<std::pair<int, Value>> latest;
    std::set<int> seen;
    for (auto it = fs.rbegin(); it != fs.rend(); ++it)
        for (const auto &p : it->items()[1].items())
            if (seen.insert(p.items()[0].as_int()).second)
                latest.push_back({it->items()[0].as_int(), p});
    for (int c = 0; c < 2; ++c) {
        std::vector<int> ids;
        for (const auto &p : current.items())
            if (p.items()[1].as_int() == c)
                ids.push_back(p.items()[0].as_int());
        for (const auto &p : latest)
            if (p.second.items()[1].as_int() == c &&
                std::find(ids.begin(), ids.end(),
                          p.second.items()[0].as_int()) == ids.end())
                ids.push_back(p.second.items()[0].as_int());
        for (int slot = 0; slot < k; ++slot) {
            if (slot >= int(ids.size())) {
                auto start = out.size();
                out.insert(out.end(), version == 7 ? 7 : 7 + 3 * steps, 0);
                if (version >= 5 && version != 7) {
                    auto z = rotate(0.0, 0.0, a.heading);
                    out[start] = z.first;
                    out[start + 1] = z.second;
                    out[start + 2] = z.first;
                    out[start + 3] = z.second;
                    for (int i = 0; i < steps; ++i) {
                        out[start + 7 + 3 * i] = z.first;
                        out[start + 8 + 3 * i] = z.second;
                    }
                }
                continue;
            }
            int id = ids[slot], when = 0;
            Value point;
            for (const auto &p : latest)
                if (p.second.items()[0].as_int() == id) {
                    when = p.first;
                    point = p.second;
                    break;
                }
            double dx = double(point.items()[2].as_int() - a.x) / radius,
                   dy = double(point.items()[3].as_int() - a.y) / radius;
            auto last = rotate(dx, dy, a.heading);
            std::map<int, std::pair<double, double>> points;
            points[when - phase.tick] = last;
            std::vector<double> lags;
            for (int lag = 1; lag <= steps; ++lag) {
                bool found = false;
                for (const auto &f : fs)
                    if (f.items()[0].as_int() == phase.tick - lag)
                        for (const auto &p : f.items()[1].items())
                            if (p.items()[0].as_int() == id) {
                                auto q =
                                    rotate(double(p.items()[2].as_int() - a.x) /
                                               radius,
                                           double(p.items()[3].as_int() - a.y) /
                                               radius,
                                           a.heading);
                                points[-lag] = q;
                                if (version < 5)
                                    lags.insert(
                                        lags.end(),
                                        {double(p.items()[2].as_int() - a.x) /
                                             radius,
                                         double(p.items()[3].as_int() - a.y) /
                                             radius,
                                         1});
                                else
                                    lags.insert(lags.end(),
                                                {q.first, q.second, 1});
                                found = true;
                            }
                if (!found) {
                    auto z = version >= 5 ? rotate(0.0, 0.0, a.heading)
                                          : std::make_pair(0.0, 0.0);
                    lags.insert(lags.end(), {z.first, z.second, 0});
                }
            }
            double confidence = 1 - double(phase.tick - when) / (steps + 1),
                   visible = 0;
            for (const auto &p : current.items())
                if (p.items()[0].as_int() == id)
                    visible = 1;
            if (version == 7) {
                double vx = 0, vy = 0, reliability = 0;
                if (points.size() > 1) {
                    auto first = *points.begin(), end = *points.rbegin();
                    double x = end.second.first - first.second.first,
                           y = end.second.second - first.second.second;
                    int span = end.first - first.first;
                    vx = x / span;
                    vy = y / span;
                    Sum path_sum;
                    auto prev = points.begin();
                    for (auto it = std::next(prev); it != points.end(); ++it) {
                        path_sum.add(
                            norm(it->second.first - prev->second.first,
                                 it->second.second - prev->second.second));
                        prev = it;
                    }
                    double path = path_sum.result();
                    reliability = double(points.size() - 1) / steps *
                                  (path ? std::min(1.0, norm(x, y) / path) : 1);
                }
                out.insert(out.end(), {last.first, last.second, vx, vy,
                                       confidence, visible, reliability});
            } else {
                double vx = 0, vy = 0;
                for (auto it = fs.rbegin(); it != fs.rend(); ++it) {
                    int t = it->items()[0].as_int();
                    if (t >= when)
                        continue;
                    bool found = false;
                    for (const auto &p : it->items()[1].items())
                        if (p.items()[0].as_int() == id) {
                            vx = double(point.items()[2].as_int() -
                                        p.items()[2].as_int()) /
                                 ((when - t) * radius);
                            vy = double(point.items()[3].as_int() -
                                        p.items()[3].as_int()) /
                                 ((when - t) * radius);
                            found = true;
                            break;
                        }
                    if (found)
                        break;
                }
                if (version >= 5) {
                    auto p = rotate(vx, vy, a.heading);
                    vx = p.first;
                    vy = p.second;
                }
                out.insert(out.end(), {version >= 5 ? last.first : dx,
                                       version >= 5 ? last.second : dy, vx, vy,
                                       double(phase.tick - when) / steps,
                                       confidence, visible});
                out.insert(out.end(), lags.begin(), lags.end());
            }
        }
    }
}
std::vector<std::string> species_names(World &w) {
    std::vector<std::string> out;
    for (auto &p : w.config.at("species").fields())
        out.push_back(p.first);
    std::sort(out.begin(), out.end());
    return out;
}
void scent(ObservationPhase &phase, const AnimalView &a,
           std::vector<double> &out) {
    auto &w = phase.world;
    // Preserve lazy sensory-state creation even if no scent has been deposited.
    sensory(w);
    const auto &spec = phase.species[a.species];
    const auto &groups = spec.scent_groups;
    using Key = std::tuple<double, double, int, int>;
    std::vector<Key> best(groups.size(), {0, 0, 0, 0});
    std::vector<Point> patch(groups.size());
    std::vector<double> grid(groups.size() * 9);
    constexpr int sectors[9][2] = {{0, 0}, {0, -1}, {1, -1}, {1, 0},  {1, 1},
                                   {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}};
    auto &queue = phase.scent_queue;
    queue.clear();
    Point start{a.x, a.y};
    queue.push_back({start, 0});
    phase.scent_stamp[start.second * phase.width + start.first] =
        phase.observer_stamp;
    for (std::size_t head = 0; head < queue.size(); ++head) {
        auto entry = queue[head];
        auto p = entry.first;
        int distance = entry.second;
        if (w.blocked(p.first, p.second) || w.depth(p.first, p.second))
            continue;
        int dx = p.first - start.first, dy = p.second - start.second,
            cell = p.second * phase.width + p.first;
        if (spec.version != 7 || dx || dy)
            for (std::size_t i = 0; i < groups.size(); ++i) {
                double strength = 0, fresh = 0;
                for (int group : groups[i]) {
                    auto found =
                        phase.scent_cells.find(cell * phase.channels + group);
                    if (found == phase.scent_cells.end())
                        continue;
                    double value =
                        std::max(0.0, 1 - double(phase.tick - found->second) /
                                              phase.scent_lifetime);
                    strength = spec.blended ? std::min(1.0, strength + value)
                                            : std::max(strength, value);
                    fresh = std::max(fresh, value);
                }
                if (spec.version != 7) {
                    Point direction{(dx > 0) - (dx < 0), (dy > 0) - (dy < 0)};
                    if (spec.version >= 5) {
                        auto r = relative(direction, a.heading);
                        direction = {r.second, -r.first};
                    }
                    int sector = 0;
                    while (sectors[sector][0] != direction.first ||
                           sectors[sector][1] != direction.second)
                        ++sector;
                    grid[i * 9 + sector] =
                        std::max(grid[i * 9 + sector], strength);
                }
                Key key{strength, fresh, -(dx * dx + dy * dy), -cell};
                if (strength && key > best[i]) {
                    best[i] = key;
                    patch[i] = {dx, dy};
                }
            }
        if (distance < spec.smell_radius)
            for (const auto &direction : directions) {
                Point next{p.first + direction[0], p.second + direction[1]};
                if (!phase.in_bounds(next))
                    continue;
                int cell_index = next.second * phase.width + next.first;
                if (phase.scent_stamp[cell_index] != phase.observer_stamp) {
                    phase.scent_stamp[cell_index] = phase.observer_stamp;
                    queue.push_back({next, distance + 1});
                }
            }
    }
    if (spec.version != 7) {
        out.insert(out.end(), grid.begin(), grid.end());
        return;
    }
    for (std::size_t i = 0; i < groups.size(); ++i) {
        if (!std::get<0>(best[i]))
            out.insert(out.end(), 4, 0);
        else {
            Point p = relative(patch[i], a.heading);
            double distance = norm(patch[i].first, patch[i].second);
            out.insert(out.end(), {p.first / distance, p.second / distance,
                                   std::get<0>(best[i]), std::get<1>(best[i])});
        }
    }
}

} // namespace
void advance_scent(World &w, const Value &dead) {
    const auto &trails = w.config.get("scent_trails");
    if (trails.null() || !trails.flag("enabled"))
        return;
    auto &s = sensory(w)["scent"];
    if (s.null())
        s = Value::array();
    auto &cells = s.items();
    int lifetime = trails.integer("lifetime_steps");
    cells.erase(std::remove_if(cells.begin(), cells.end(),
                               [&](const Value &r) {
                                   return w.tick() - r.items()[1].as_int() >=
                                          lifetime;
                               }),
                cells.end());
    auto names = species_names(w);
    int channels = 2 * int(names.size());
    auto deposit = [&](const Value &a) {
        if (w.blocked(a.integer("x"), a.integer("y")) ||
            w.depth(a.integer("x"), a.integer("y")))
            return;
        int group =
            2 * int(std::find(names.begin(), names.end(), a.text("species")) -
                    names.begin()) +
            (a.text("sex") == "male");
        int key =
            (a.integer("y") * w.width() + a.integer("x")) * channels + group;
        cells.erase(std::remove_if(cells.begin(), cells.end(),
                                   [&](const Value &r) {
                                       return r.items()[0].as_int() == key;
                                   }),
                    cells.end());
        cells.push_back(array({key, w.tick()}));
    };
    for (const auto &a : w.animals())
        deposit(a);
    if (!dead.null())
        for (const auto &a : dead.items())
            deposit(a);
}
std::map<int, std::vector<double>> observe(World &w) {
    ObservationPhase phase(w);
    if (w.social())
        w.social()->prepare(w);
    if (!w.state.get("sensory_state").null()) {
        std::set<int> living;
        for (const auto &animal : phase.animals)
            living.insert(animal.id);
        auto &visual = sensory(w)["visual"].items();
        visual.erase(std::remove_if(visual.begin(), visual.end(),
                                    [&](const Value &row) {
                                        return !living.count(
                                            row.items()[0].as_int());
                                    }),
                     visual.end());
    }
    if (!w.state.get("sensory_state").get("visual").null()) {
        const auto &visual = w.state.at("sensory_state").at("visual").items();
        for (std::size_t index = 0; index < visual.size(); ++index)
            phase.visual_rows[visual[index].items()[0].as_int()] = index;
    }
    std::map<int, std::vector<double>> result;
    for (const auto &a : phase.animals) {
        ++phase.observer_stamp;
        const auto &spec = phase.species[a.species];
        int k = spec.top_k, v = spec.version, h = a.heading,
            radius = spec.limit;
        auto &out = result[a.id];
        out.reserve(64 + 32 * k);
        out = {a.energy / spec.max_energy, a.health / spec.max_health,
               double(directions[h][0]), double(directions[h][1])};
        for (int offset = 0; offset < 4; ++offset) {
            const auto &direction = directions[(h + offset) % 4];
            Point cell{a.x + direction[0], a.y + direction[1]};
            out.push_back(w.has_terrain()
                              ? !w.movement_reason(w.animal(a.id), cell.first,
                                                   cell.second, "move_forward",
                                                   phase.occupant(cell) >= 0)
                                     .empty()
                              : (w.blocked(cell.first, cell.second) ||
                                 phase.occupant(cell) >= 0));
        }
        std::array<std::vector<Target>, 3> selected;
        for (int category = 0; category < 3; ++category) {
            selected[category] =
                nearest(phase, a, phase.targets[category], k, radius);
            for (int slot = 0; slot < k; ++slot) {
                if (slot >= int(selected[category].size())) {
                    auto zero =
                        v >= 5 ? rotate(0.0, 0.0, h) : std::make_pair(0.0, 0.0);
                    out.push_back(zero.first);
                    out.push_back(zero.second);
                    out.insert(out.end(), v == 7 ? 1 : 2, 0);
                    continue;
                }
                const auto &target = selected[category][slot];
                double dx = double(target.x - a.x) / radius,
                       dy = double(target.y - a.y) / radius;
                if (v >= 5) {
                    auto point = rotate(dx, dy, h);
                    out.push_back(point.first);
                    out.push_back(point.second);
                } else {
                    out.push_back(dx);
                    out.push_back(dy);
                }
                if (v != 7)
                    out.push_back(norm(target.x - a.x, target.y - a.y) /
                                  radius);
                out.push_back(1);
            }
        }
        if (v >= 2) {
            double elevation = w.elevation(a.x, a.y);
            auto flow = w.flow(a.x, a.y);
            auto projected =
                v >= 5 ? rotate(flow.first / 4, flow.second / 4, h)
                       : std::make_pair(flow.first / 4, flow.second / 4);
            out.insert(out.end(), {w.depth(a.x, a.y) / 10, projected.first,
                                   projected.second, elevation / 100});
            for (int offset = 0; offset < 4; ++offset) {
                const auto &d = directions[(h + offset) % 4];
                int x = a.x + d[0], y = a.y + d[1];
                out.push_back(w.depth(x, y) / 10);
                out.push_back((w.elevation(x, y) - elevation) / 100);
            }
        }
        if (v >= 3)
            mating(phase, a, out, v);
        if (v >= 5) {
            Point front{a.x + directions[h][0], a.y + directions[h][1]};
            int target = phase.occupant(front);
            bool attack =
                spec.attack && spec.category == 2 && target >= 0 &&
                phase.species[phase.animals[target].species].category == 1 &&
                spec.attack_damage > 0;
            bool eat = spec.eat && a.energy < spec.max_energy &&
                       spec.bite_size > 0 && spec.energy_per_food > 0 &&
                       (phase.food_cells.count(front) ||
                        phase.food_cells.count({a.x, a.y}));
            if (v != 7 || spec.observe_attack)
                out.push_back(attack);
            if (v != 7 || spec.observe_eat)
                out.push_back(eat);
            if (v != 7 || spec.observe_swim)
                out.push_back(spec.swim &&
                              w.movement_reason(w.animal(a.id), front.first,
                                                front.second, "swim",
                                                phase.occupant(front) >= 0)
                                  .empty());
        }
        if (spec.history_steps)
            history(phase, a, selected, spec.history_steps, k, radius, v, out);
        if (!spec.brain->get("scent_observation").null())
            scent(phase, a, out);
        if (w.social()) {
            const auto &plan = spec.plan;
            auto social =
                w.social()->observe(w, w.animal(a.id), *spec.brain,
                                    v == 7 ? &plan->actions : nullptr);
            out.insert(out.end(), social.begin(), social.end());
        }
        if (v == 6)
            out = spec.plan->encode_soft(out);
    }
    return result;
}
Value world_metrics(World &w) {
    Value population = Value::object();
    for (const auto &s : w.config.at("species").fields())
        population[s.first] = 0;
    for (const auto &a : w.animals())
        population[a.text("species")] =
            population.at(a.text("species")).as_int() + 1;
    // CPython 3.12/3.13 sum preserves its integer prefix and compensates only
    // later float operands. Integer operands in the float phase bypass the
    // correction, so converting every quantity to double changes exact metrics.
    std::int64_t integer_food = 0;
    bool floating_food = false;
    Sum food_sum;
    for (const auto &resource : w.resources()) {
        const auto &quantity = resource.at("quantity");
        if (!floating_food && quantity.integer()) {
            integer_food += quantity.i64();
        } else if (!floating_food) {
            food_sum.value =
                static_cast<double>(integer_food) + quantity.number();
            floating_food = true;
        } else if (quantity.integer()) {
            food_sum.value += quantity.number();
        } else {
            food_sum.add(quantity.number());
        }
    }
    Value food = floating_food ? Value(food_sum.result()) : Value(integer_food);
    Value result = object(
        {{"tick", w.tick()}, {"populations", population}, {"food", food}});
    const auto &old = w.frame.at("metrics");
    for (const auto *key :
         {"deaths", "energy_gained", "kills", "births", "mutations"})
        result[key] = old.at(key);
    return result;
}
} // namespace wtfcore
