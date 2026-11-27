#include "context.hpp"
namespace wtfcore {
namespace {
constexpr int DX[] = {0, 1, 0, -1};
constexpr int DY[] = {-1, 0, 1, 0};
CellPosition position(const Value &a) {
    return {a.integer("x"), a.integer("y")};
}
bool allows(const Value &spec, const std::string &action) {
    for (const auto &a : spec.at("allowed_actions").items())
        if (a.str() == action)
            return true;
    return false;
}
} // namespace
DecisionContext::DecisionContext(World &world) : world_(world) {
    for (const auto &a : world.animals()) {
        positions_[position(a)] = a.integer("id");
        occupied_.insert(position(a));
        ++populations_[a.text("species")];
        eligible_[a.integer("id")] = eligible(a);
    }
    for (const auto &r : world.resources())
        food_[position(r)] = r.num("quantity");
}
bool DecisionContext::eligible(const Value &a) const {
    const auto &spec = world_.species(a);
    const auto *rules = &spec.get("reproduction");
    if (!rules || rules->null() || !rules->flag("enabled", false) ||
        !allows(spec, "reproduce"))
        return false;
    int due = a.integer("reproduction_due_age", -1);
    if (due < 0) {
        bool first = a.integer("last_reproduced_age", -1) < 0;
        std::string prefix = first ? "age" : "cooldown";
        int center = rules->integer(first ? "min_age_steps" : "cooldown_steps");
        const auto *pct = &rules->get(prefix + "_deviation_pct");
        int deviation = pct && !pct->null()
                            ? int(center * pct->number() / 100 + 0.5)
                            : rules->integer(prefix + "_deviation_steps", 0);
        due = (first ? 0 : a.integer("last_reproduced_age")) +
              std::max(1, center - deviation);
    }
    return a.integer("age") + 1 >= due &&
           a.num("health") > rules->num("health_threshold") &&
           a.num("energy") >= rules->num("min_energy") &&
           a.num("energy") > rules->num("energy_cost") &&
           (!world_.has_terrain() ||
            world_.depth(a.integer("x"), a.integer("y")) == 0);
}
bool DecisionContext::birth_space(const Value &first,
                                  const Value &second) const {
    for (const auto *parent : {&first, &second}) {
        auto p = position(*parent);
        for (int d = 0; d < 4; ++d) {
            int x = p.first + DX[d], y = p.second + DY[d];
            if (!occupied_.count({x, y}) &&
                world_.birth_cell_allowed(*parent, x, y))
                return true;
        }
    }
    return false;
}
bool DecisionContext::reproduction_available(const Value &a) const {
    if (!eligible_.at(a.integer("id")))
        return false;
    const auto &rules = world_.species(a).at("reproduction");
    if (populations_.at(a.text("species")) >= rules.integer("max_population"))
        return false;
    int radius = rules.integer("mate_radius", 2);
    auto p = position(a);
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > radius * radius)
                continue;
            auto found = positions_.find({p.first + dx, p.second + dy});
            if (found == positions_.end() || found->second == a.integer("id"))
                continue;
            const auto &mate = world_.animal(found->second);
            if (mate.text("species") != a.text("species") ||
                mate.text("sex") == a.text("sex") ||
                !eligible_.at(found->second))
                continue;
            if (world_.visible(p.first, p.second, mate.integer("x"),
                               mate.integer("y")) &&
                world_.mating_land_route(a, mate, radius) &&
                birth_space(a, mate))
                return true;
        }
    return false;
}
std::vector<bool>
DecisionContext::action_mask(const Value &a,
                             const std::vector<std::string> &actions) const {
    const auto &body = world_.body(a);
    auto front = world_.movement_destination(a, "move_forward");
    std::vector<bool> result;
    result.reserve(actions.size());
    bool any = false;
    for (const auto &action : actions) {
        bool valid = true;
        if (action == "move_forward" || action == "move_backward" ||
            action == "strafe_left" || action == "strafe_right" ||
            action == "swim") {
            auto p = action == "swim" ? front
                                      : world_.movement_destination(a, action);
            valid = world_
                        .movement_block_reason(a, p.first, p.second, action,
                                               &occupied_)
                        .empty();
        } else if (action == "challenge")
            valid = world_.challenge_opportunity(a, positions_).first >= 0;
        else if (action == "reproduce")
            valid = reproduction_available(a);
        else if (action == "eat") {
            auto at = food_.find(position(a)), ahead = food_.find(front);
            valid = a.num("energy") < body.num("max_energy") &&
                    body.num("bite_size") > 0 &&
                    body.num("energy_per_food_unit") > 0 &&
                    ((at != food_.end() && at->second > 0) ||
                     (ahead != food_.end() && ahead->second > 0));
        } else if (action == "attack") {
            auto found = positions_.find(front);
            valid = found != positions_.end() &&
                    world_.species(world_.animal(found->second)).text("diet") ==
                        "resource" &&
                    world_.species(a).text("diet") == "prey" &&
                    body.num("attack_damage") > 0;
        }
        result.push_back(valid);
        any = any || valid;
    }
    if (!any)
        throw std::runtime_error(
            "Masked policy has no available action; allow rest or turning");
    return result;
}
std::map<std::string, std::vector<double>> central_contexts(World &world) {
    double area = double(world.width()) * world.height();
    const auto &resources = world.config.at("resources");
    double capacity =
        resources.num("patch_count") * resources.num("capacity_per_patch");
    double max_steps = world.config.at("world").num("max_steps");
    Sum food;
    std::int64_t integer_food = 0;
    bool floating_food = false;
    for (const auto &r : world.resources()) {
        const auto &quantity = r.at("quantity");
        if (!floating_food && quantity.integer())
            integer_food += quantity.i64();
        else if (!floating_food) {
            food.value = static_cast<double>(integer_food) + quantity.number();
            floating_food = true;
        } else if (quantity.integer())
            food.value += quantity.number();
        else
            food.add(quantity.number());
    }
    double food_total =
        floating_food ? food.result() : static_cast<double>(integer_food);
    std::vector<double> prefix = {
        world.tick() / max_steps, capacity ? food_total / capacity : 0.0,
        world.animals().size() / area,
        world.frame.at("obstacles").items().size() / area};
    std::map<std::string, std::vector<double>> result;
    for (const auto &[name, spec] : world.config.at("species").fields()) {
        (void)spec;
        auto values = prefix;
        for (bool allied : {true, false}) {
            std::array<Sum, 5> sums;
            std::int64_t ages = 0;
            std::size_t n = 0;
            for (const auto &a : world.animals()) {
                if ((a.text("species") == name) != allied)
                    continue;
                const auto &body = world.body(a);
                ++n;
                sums[0].add(a.num("energy") / body.num("max_energy"));
                sums[1].add(a.num("health") / body.num("max_health"));
                sums[2].add(a.num("x") / std::max(1, world.width() - 1));
                sums[3].add(a.num("y") / std::max(1, world.height() - 1));
                ages += a.integer("age");
            }
            if (!n)
                values.insert(values.end(), 6, 0.0);
            else {
                values.push_back(n / area);
                for (int i = 0; i < 4; ++i)
                    values.push_back(sums[i].result() / n);
                values.push_back(double(ages) / n / max_steps);
            }
        }
        result.emplace(name, std::move(values));
    }
    return result;
}
} // namespace wtfcore
