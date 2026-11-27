#include "world_genesis.hpp"
#include "sensing.hpp"
#include <deque>
#include <optional>
#include <set>

namespace wtfcore {
namespace {
using Pos = CellPosition;
constexpr int directions[4][2] = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
Pos position(const Value &a) { return {a.integer("x"), a.integer("y")}; }
std::string zone(Pos p, const Value &bounds) {
    const auto &b = bounds.items();
    int left = b[0].as_int(), top = b[1].as_int(), right = b[2].as_int(),
        bottom = b[3].as_int();
    if (left < p.first && p.first < right && top < p.second &&
        p.second < bottom)
        return "inside";
    if (p.first < left || p.first > right || p.second < top ||
        p.second > bottom)
        return "outside";
    return "boundary";
}
template <class T>
std::vector<T> sample(Random &rng, const std::vector<T> &population,
                      std::size_t count) {
    if (count > population.size())
        throw std::runtime_error("Insufficient native placement cells");
    std::size_t setsize = 21;
    if (count > 5) {
        std::size_t power = 1;
        while (power < 3 * count)
            power *= 4;
        setsize += power;
    }
    std::vector<T> result;
    result.reserve(count);
    if (population.size() <= setsize) {
        auto pool = population;
        for (std::size_t i = 0; i < count; ++i) {
            auto j = rng.below(population.size() - i);
            result.push_back(pool[j]);
            pool[j] = pool[population.size() - i - 1];
        }
    } else {
        std::set<std::size_t> selected;
        for (std::size_t i = 0; i < count; ++i) {
            auto j = rng.below(population.size());
            while (selected.count(j))
                j = rng.below(population.size());
            selected.insert(j);
            result.push_back(population[j]);
        }
    }
    return result;
}
Value founder(int id, const std::string &species, Pos p, int heading,
              const Value &body, const std::string &sex) {
    return object({{"id", id},
                   {"species", species},
                   {"x", p.first},
                   {"y", p.second},
                   {"heading", heading},
                   {"energy", body.at("initial_energy")},
                   {"health", body.at("max_health")},
                   {"age", 0},
                   {"last_action", "rest"},
                   {"last_reward", 0.0},
                   {"parent_id", -1},
                   {"generation", 0},
                   {"last_reproduced_age", -1},
                   {"kills", 0},
                   {"energy_gained", 0.0},
                   {"statistics_since_age", 0},
                   {"reproduction_due_age", -1},
                   {"mutated", false},
                   {"mutation_count", 0},
                   {"mutation_weights", 0},
                   {"sex", sex},
                   {"second_parent_id", -1},
                   {"drift_x", 0.0},
                   {"drift_y", 0.0},
                   {"mating_intent_until_tick", -1},
                   {"last_mating_failure", ""}});
}
std::vector<std::pair<Pos, Pos>> pair_cells(World &world,
                                            const std::string &name,
                                            const std::set<Pos> &occupied) {
    const auto &spec = world.config.at("species").at(name),
               &placement = world.config.get("placement");
    double climb = spec.get("locomotion").num("max_climb", 1.0);
    std::set<Pos> cells;
    for (int y = 0; y < world.height(); ++y)
        for (int x = 0; x < world.width(); ++x) {
            Pos p{x, y};
            if (!occupied.count(p) && !world.blocked(x, y) &&
                !world.depth(x, y) &&
                (placement.null() ||
                 zone(p, placement.at("enclosure_bounds")) ==
                     placement.at("species").text(name)))
                cells.insert(p);
        }
    std::vector<Pos> left;
    for (auto p : cells)
        if ((p.first + p.second) % 2 == 0)
            left.push_back(p);
    Random rng(world.state.at("seed").u64() ^ 0x50414952ULL);
    rng.shuffle(left);
    std::map<Pos, Pos> matched_left, matched_right;
    std::vector<Pos> order;
    std::size_t needed = spec.integer("initial_population") / 2;
    for (auto start : left) {
        std::deque<Pos> queue{start};
        std::set<Pos> seen{start};
        std::map<Pos, Pos> previous;
        std::optional<Pos> found;
        while (!queue.empty() && !found) {
            auto current = queue.front();
            queue.pop_front();
            for (const auto &d : directions) {
                Pos target{current.first + d[0], current.second + d[1]};
                if (!cells.count(target) || previous.count(target))
                    continue;
                if (world.has_terrain() &&
                    std::abs(world.elevation(current.first, current.second) -
                             world.elevation(target.first, target.second)) >
                        climb)
                    continue;
                previous[target] = current;
                if (!matched_right.count(target)) {
                    found = target;
                    break;
                }
                auto other = matched_right.at(target);
                if (!seen.count(other)) {
                    seen.insert(other);
                    queue.push_back(other);
                }
            }
        }
        if (!found)
            continue;
        while (found) {
            auto current = previous.at(*found);
            std::optional<Pos> old;
            auto before = matched_left.find(current);
            if (before != matched_left.end())
                old = before->second;
            else
                order.push_back(current);
            matched_left[current] = *found;
            matched_right[*found] = current;
            found = old;
        }
        if (matched_left.size() == needed) {
            std::vector<std::pair<Pos, Pos>> result;
            for (auto p : order)
                result.push_back({p, matched_left.at(p)});
            return result;
        }
    }
    throw std::runtime_error("Insufficient adjacent dry native spawn pairs");
}
void adjacent_pairs(World &world) {
    std::size_t index = 0;
    for (const auto &[name, spec] : world.config.at("species").fields()) {
        std::size_t species_index = index++;
        if (spec.get("spawn").text("mode", "random") != "adjacent_pairs")
            continue;
        std::vector<int> founders, females, males;
        for (const auto &a : world.animals())
            if (a.text("species") == name) {
                int id = a.integer("id");
                founders.push_back(id);
                if (a.text("sex") == "female")
                    females.push_back(id);
                if (a.text("sex") == "male")
                    males.push_back(id);
            }
        if (females.size() != males.size())
            throw std::runtime_error(
                "Adjacent native spawning requires even population");
        if (founders.empty())
            continue;
        std::map<int, Pos> originals;
        for (int id : founders)
            originals[id] = position(world.animal(id));
        std::set<Pos> other_cells;
        for (const auto &a : world.animals())
            if (a.text("species") != name)
                other_cells.insert(position(a));
        auto occupied = other_cells;
        for (int id : females)
            occupied.insert(position(world.animal(id)));
        Random rng(world.state.at("seed").u64() ^ 0x50414952ULL ^
                   (species_index * 7919));
        const auto &placement = world.config.get("placement");
        double climb = spec.get("locomotion").num("max_climb", 1.0);
        for (std::size_t i = 0; i < females.size(); ++i) {
            const auto &female = world.animal(females[i]);
            std::vector<Pos> candidates;
            for (const auto &d : directions) {
                Pos target{female.integer("x") + d[0],
                           female.integer("y") + d[1]};
                if (occupied.count(target) ||
                    world.blocked(target.first, target.second))
                    continue;
                if (!placement.null() &&
                    zone(target, placement.at("enclosure_bounds")) !=
                        placement.at("species").text(name))
                    continue;
                if (world.has_terrain() &&
                    (world.depth(target.first, target.second) ||
                     std::abs(world.elevation(target.first, target.second) -
                              world.elevation(female.integer("x"),
                                              female.integer("y"))) > climb))
                    continue;
                candidates.push_back(target);
            }
            if (candidates.empty()) {
                for (int id : founders) {
                    auto &a = world.animal(id);
                    a["x"] = originals.at(id).first;
                    a["y"] = originals.at(id).second;
                }
                auto pairs = pair_cells(world, name, other_cells);
                for (std::size_t j = 0; j < females.size(); ++j) {
                    auto &mother = world.animal(females[j]);
                    auto &father = world.animal(males[j]);
                    mother["x"] = pairs[j].first.first;
                    mother["y"] = pairs[j].first.second;
                    father["x"] = pairs[j].second.first;
                    father["y"] = pairs[j].second.second;
                }
                break;
            }
            Pos target = candidates[rng.below(candidates.size())];
            auto &male = world.animal(males[i]);
            male["x"] = target.first;
            male["y"] = target.second;
            occupied.insert(target);
        }
    }
}
} // namespace
Value make_world(const Value &config, std::uint64_t seed) {
    int width = config.at("world").integer("width"),
        height = config.at("world").integer("height");
    if (width < 3 || width > 256 || height < 3 || height > 256)
        throw std::runtime_error("Invalid native world dimensions");
    std::vector<Pos> cells;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            cells.push_back({x, y});
    Random layout(seed);
    layout.shuffle(cells);
    std::set<Pos> obstacles;
    const auto &o = config.at("obstacles");
    if (o.text("generation") == "explicit") {
        for (const auto &entry : o.at("explicit_cells").items())
            obstacles.insert(
                {entry.items()[0].as_int(), entry.items()[1].as_int()});
    } else {
        auto count =
            static_cast<std::size_t>(cells.size() * o.num("occupied_fraction"));
        if (count > cells.size())
            throw std::runtime_error("Invalid native obstacle fraction");
        obstacles.insert(cells.begin(), cells.begin() + count);
    }
    std::shared_ptr<const TerrainGrid> terrain;
    if (config.contains("terrain")) {
        terrain = std::make_shared<const TerrainGrid>(config.at("terrain"),
                                                      width, height);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                const auto &row =
                    config.at("terrain").at("rows").items()[y].str();
                if (config.at("terrain")
                        .at("legend")
                        .at(std::string(1, row[x]))
                        .flag("blocked"))
                    obstacles.insert({x, y});
            }
    }
    std::vector<Pos> free;
    for (auto p : cells)
        if (!obstacles.count(p) &&
            (!terrain || !terrain->depth[p.second * width + p.first]))
            free.push_back(p);
    const auto &resources = config.at("resources"),
               &placement = config.get("placement");
    std::vector<Pos> resource_cells;
    if (!placement.null()) {
        std::map<std::string, std::vector<Pos>> pools;
        const auto &b = placement.at("enclosure_bounds").items();
        int left = b[0].as_int(), top = b[1].as_int(), right = b[2].as_int(),
            bottom = b[3].as_int();
        int boundary = placement.at("resources").integer("boundary_width");
        for (auto p : free) {
            auto side = zone(p, placement.at("enclosure_bounds"));
            if (side == "boundary")
                continue;
            int distance = side == "inside"
                               ? std::min({p.first - left, right - p.first,
                                           p.second - top, bottom - p.second})
                               : std::max({left - p.first, p.first - right,
                                           top - p.second, p.second - bottom});
            pools[distance <= boundary ? side : "far"].push_back(p);
        }
        int count = resources.integer("patch_count");
        int near = static_cast<int>(
            count * placement.at("resources").num("boundary_fraction") + 0.5);
        int inside = static_cast<int>(
            near * placement.at("resources").num("inside_fraction") + 0.5);
        for (auto [side, n] : std::vector<std::pair<std::string, int>>{
                 {"inside", inside},
                 {"outside", near - inside},
                 {"far", count - near}}) {
            auto chosen = sample(layout, pools[side], n);
            resource_cells.insert(resource_cells.end(), chosen.begin(),
                                  chosen.end());
        }
    } else
        resource_cells = sample(layout, free, resources.integer("patch_count"));
    Value patches = Value::array();
    for (auto p : resource_cells)
        patches.push(
            object({{"x", p.first},
                    {"y", p.second},
                    {"quantity", resources.at("initial_quantity_per_patch")}}));
    layout.shuffle(free);
    std::map<std::string, std::vector<Pos>> spawn_pools;
    if (!placement.null())
        for (auto p : free) {
            auto side = zone(p, placement.at("enclosure_bounds"));
            if (side == "inside" || side == "outside")
                spawn_pools[side].push_back(p);
        }
    Value animals = Value::array(), populations = Value::object();
    std::size_t species_index = 0;
    for (const auto &[name, spec] : config.at("species").fields()) {
        int population = spec.integer("initial_population");
        populations[name] = population;
        std::vector<std::string> sexes;
        for (int n = 0; n < population; ++n)
            sexes.push_back(n % 2 == 0 ? "female" : "male");
        Random sex_rng(config.at("seed").u64() ^ 0x534558ULL ^
                       (species_index++ * 7919));
        sex_rng.shuffle(sexes);
        for (const auto &sex : sexes) {
            auto &pool = placement.null()
                             ? free
                             : spawn_pools[placement.at("species").text(name)];
            if (pool.empty())
                throw std::runtime_error("Insufficient native founder cells");
            auto p = pool.back();
            pool.pop_back();
            int id = static_cast<int>(animals.items().size()),
                heading = static_cast<int>(layout.below(4));
            animals.push(founder(id, name, p, heading, spec.at("body"), sex));
        }
    }
    Value obstacles_value = Value::array();
    for (auto p : obstacles)
        obstacles_value.push(array({p.first, p.second}));
    Value metrics = object({{"tick", 0},
                            {"populations", populations},
                            {"food", 0},
                            {"deaths", Value::object()},
                            {"energy_gained", Value::object()},
                            {"kills", Value::object()},
                            {"births", Value::object()},
                            {"mutations", Value::object()}});
    Value snapshot =
        object({{"width", width},
                {"height", height},
                {"resource_capacity", resources.at("capacity_per_patch")},
                {"obstacles", obstacles_value},
                {"tick", 0},
                {"resources", patches},
                {"animals", animals},
                {"events", Value::array()},
                {"metrics", metrics}});
    Random world_rng(seed ^ 0x12345678ULL);
    Value result =
        object({{"config", config},
                {"seed", seed},
                {"snapshot", snapshot},
                {"rng_state", world_rng.checkpoint()},
                {"lifespans", Value::array()},
                {"next_animal_id", static_cast<int>(animals.items().size())}});
    bool tracking = config.get("scent_trails").flag("enabled");
    for (const auto &[name, brain] : config.at("brains").fields())
        tracking |= brain.integer("movement_history_steps") != 0;
    if (tracking) {
        result["sensory_state"] =
            object({{"version", 1}, {"visual", Value::array()}});
        if (config.get("scent_trails").flag("enabled"))
            result["sensory_state"]["scent"] = Value::array();
    }
    World world(result);
    adjacent_pairs(world);
    if (config.get("scent_trails").flag("enabled"))
        advance_scent(world, Value::array());
    return world.checkpoint();
}
} // namespace wtfcore
