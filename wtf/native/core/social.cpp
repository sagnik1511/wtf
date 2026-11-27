#include "social.hpp"
#include "world.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <list>
#include <optional>
#include <set>
#include <unordered_map>
namespace wtfcore {
namespace {
using Key = ConsequenceModel::Context;
using Labels = ConsequenceModel::Labels;
using Rates = ConsequenceModel::Rates;
using Pos = std::pair<int, int>;
constexpr const char *actions[] = {
    "move_forward", "move_backward", "strafe_left", "strafe_right",
    "turn_left",    "turn_right",    "rest",        "eat",
    "attack",       "swim",          "reproduce",   "challenge"};
constexpr int groups[] = {0, 0, 0, 0, 5, 5, 5, 2, 3, 1, 4, 3};
constexpr int directions[4][2] = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
int action_index(const std::string &name) {
    for (int i = 0; i < 12; ++i)
        if (name == actions[i])
            return i;
    throw std::runtime_error("Unknown social action");
}
void fields(const Value &value, std::initializer_list<const char *> names) {
    if (!std::holds_alternative<Value::ObjectPtr>(value.data) ||
        value.fields().size() != names.size())
        throw std::runtime_error("Invalid social/consequence fields");
    for (auto name : names)
        if (!value.contains(name))
            throw std::runtime_error("Missing social/consequence field");
}
std::int64_t integer(const Value &v) {
    if (!v.integer())
        throw std::runtime_error("Expected exact social integer");
    return v.i64();
}
void increment(std::int64_t &n) {
    if (n == std::numeric_limits<std::int64_t>::max())
        throw std::runtime_error("Native social counter capacity exceeded");
    ++n;
}
Key key_from(const Value &value, bool bounded = true) {
    if (!value.is_array() || value.array_size() != 12)
        throw std::runtime_error("Invalid consequence context");
    Key key{};
    for (std::size_t i = 0; i < 12; ++i) {
        if (!value.array_value(i).integer())
            throw std::runtime_error("Invalid consequence context type");
        key[i] = value.array_value(i).as_int();
    }
    if (key[0] < 0 || key[0] >= 12)
        throw std::runtime_error("Invalid consequence action");
    if (bounded)
        for (std::size_t i = 1; i < 12; ++i)
            if (key[i] < -2 || key[i] > 3)
                throw std::runtime_error("Invalid consequence context range");
    return key;
}
Value key_state(const Key &key) {
    Value out = Value::array();
    for (int x : key)
        out.push(x);
    return out;
}
Value labels_state(const Labels &labels) {
    Value out = Value::array();
    for (auto x : labels)
        out.push(x);
    return out;
}
Labels labels_from(const Value &value) {
    if (!value.is_array() || value.array_size() != 3)
        throw std::runtime_error("Invalid social labels");
    Labels labels{};
    for (int i = 0; i < 3; ++i) {
        labels[i] = integer(value.array_value(i));
        if (labels[i] < 0)
            throw std::runtime_error("Negative social labels");
    }
    return labels;
}
struct Row {
    std::array<double, 4> values{};
    unsigned floating = 0;
    static Row restore(const Value &v, bool totals) {
        if (!v.is_array() || v.array_size() != 4)
            throw std::runtime_error("Invalid consequence row");
        Row out;
        for (int i = 0; i < 4; ++i) {
            const auto x = v.array_value(i);
            if (!x.integer() && !std::holds_alternative<double>(x.data))
                throw std::runtime_error("Invalid consequence row type");
            out.values[i] = x.number();
            if (!std::isfinite(out.values[i]))
                throw std::runtime_error("Nonfinite consequence row");
            if (std::holds_alternative<double>(x.data))
                out.floating |= 1U << i;
        }
        if ((totals ? out.values[0] < 0 : out.values[0] <= 0) ||
            out.values[0] > (totals ? 9007199254740992.0 : 65536.0))
            throw std::runtime_error("Invalid consequence evidence count");
        for (int i = 1; i < 4; ++i)
            if (out.values[i] < 0 || out.values[i] > out.values[0])
                throw std::runtime_error("Invalid consequence outcome count");
        return out;
    }
    Value state() const {
        Value out = Value::array();
        for (int i = 0; i < 4; ++i)
            out.push(floating & (1U << i)
                         ? Value(values[i])
                         : Value(static_cast<std::int64_t>(values[i])));
        return out;
    }
    void add(const Row &other) {
        for (int i = 0; i < 4; ++i)
            values[i] += other.values[i];
        floating |= other.floating;
    }
    void learn(const Labels &labels, double threshold) {
        if (values[0] >= threshold) {
            for (auto &v : values)
                v *= 0.5;
            floating = 15;
        }
        values[0] += 1;
        for (int i = 0; i < 3; ++i)
            values[i + 1] += static_cast<int>(labels[i] > 0);
    }
    Rates rates() const {
        if (!values[0])
            return {};
        return {values[0] / (values[0] + 4), values[1] / values[0],
                values[2] / values[0], values[3] / values[0]};
    }
};
class Table {
  public:
    using Entry = std::pair<Key, Row>;
    std::list<Entry> rows;
    std::map<Key, std::list<Entry>::iterator> lookup;
    Table() = default;
    Table(const Table &other) : rows(other.rows) { reindex(); }
    Table &operator=(const Table &other) {
        if (this != &other) {
            rows = other.rows;
            reindex();
        }
        return *this;
    }
    Table(Table &&) = default;
    Table &operator=(Table &&) = default;
    void reindex() {
        lookup.clear();
        for (auto i = rows.begin(); i != rows.end(); ++i)
            lookup.emplace(i->first, i);
    }
    const Row *find(const Key &key) const {
        auto p = lookup.find(key);
        return p == lookup.end() ? nullptr : &p->second->second;
    }
    void append(const Key &key, const Row &row) {
        if (lookup.count(key))
            throw std::runtime_error("Duplicate consequence context");
        rows.push_back({key, row});
        lookup.emplace(key, std::prev(rows.end()));
    }
    void trim(std::size_t capacity) {
        while (rows.size() > capacity) {
            lookup.erase(rows.front().first);
            rows.pop_front();
        }
    }
    void learn(const Key &key, const Labels &labels, std::size_t capacity) {
        auto p = lookup.find(key);
        if (p == lookup.end()) {
            rows.push_back({key, {}});
            p = lookup.emplace(key, std::prev(rows.end())).first;
        } else
            rows.splice(rows.end(), rows, p->second);
        p->second->second.learn(labels, 65535);
        trim(capacity);
    }
    Value state() const {
        Value out = Value::array();
        for (const auto &[key, row] : rows)
            out.push(array({key_state(key), row.state()}));
        return out;
    }
};
using Totals = std::array<Row, 12>;
Value totals_state(const Totals &rows) {
    Value out = Value::array();
    for (const auto &r : rows)
        out.push(r.state());
    return out;
}
Totals totals_restore(const Value &value) {
    if (!value.is_array() || value.array_size() != 12)
        throw std::runtime_error("Invalid consequence total actions");
    Totals out;
    for (int i = 0; i < 12; ++i)
        out[i] = Row::restore(value.array_value(i), true);
    return out;
}
} // namespace
struct ConsequenceModel::Impl {
    struct Species {
        std::string name;
        Table published;
        std::optional<Table> staged;
        Totals totals{};
        std::optional<Totals> staged_totals;
        mutable std::optional<std::array<Rates, 12>> cache;
    };
    Value contracts = Value::object();
    std::vector<Species> species;
    std::map<std::string, std::size_t> indices;
    std::vector<std::size_t> totals_order, staged_order, staged_totals_order;
    std::int64_t updates = 0, published_updates = 0;
    int horizon = 16, capacity = 512;
    bool episode = false, persistent = false;
    explicit Impl(const Value &config) {
        const auto &settings = config.get("consequence_learning");
        horizon = settings.integer("horizon_steps", 16);
        capacity = settings.integer("max_contexts", 512);
        episode = settings.text("update_schedule", "continuous") == "episode";
        persistent = settings.flag("persistent_action_totals");
        if (horizon < 1 || horizon > 64 || capacity < 1 || capacity > 8192)
            throw std::runtime_error("Invalid native consequence bounds");
        for (const auto &[name, spec] : config.at("species").fields())
            if (config.at("brains")
                    .at(spec.text("brain"))
                    .flag("consequence_observation")) {
                Value contract = Value::object();
                for (auto key : {"diet", "body", "locomotion", "circadian",
                                 "reproduction", "allowed_actions"})
                    contract[key] = spec.get(key);
                contracts[name] = contract;
                indices[name] = species.size();
                totals_order.push_back(species.size());
                Species entry;
                entry.name = name;
                species.push_back(std::move(entry));
            }
    }
    Species &get(const std::string &name) {
        return species.at(indices.at(name));
    }
    const Species &get(const std::string &name) const {
        return species.at(indices.at(name));
    }
    void table_restore(Table &table, Totals *seed, const Value &rows) {
        if (!rows.is_array() || rows.array_size() > 8192)
            throw std::runtime_error("Oversized consequence model");
        for (const auto &entry : rows.items()) {
            if (!entry.is_array() || entry.array_size() != 2)
                throw std::runtime_error("Invalid consequence table row");
            Key key = key_from(entry.array_value(0));
            Row row = Row::restore(entry.array_value(1), false);
            table.append(key, row);
            if (seed)
                (*seed)[key[0]].add(row);
        }
        table.trim(capacity);
    }
    void restore_model(const Value &state) {
        auto version = integer(state.at("version"));
        if (version == 3) {
            fields(state, {"version", "model", "action_totals",
                           "staged_action_totals"});
            if (!persistent || !state.at("model").contains("version") ||
                (state.at("model").integer("version") != 1 &&
                 state.at("model").integer("version") != 2))
                throw std::runtime_error(
                    "Invalid persistent consequence wrapper");
            restore_model(state.at("model"));
            const auto &totals = state.at("action_totals");
            const auto &staged = state.at("staged_action_totals");
            if (totals.fields().size() != species.size() ||
                staged.fields().size() != staged_order.size())
                throw std::runtime_error("Invalid consequence total species");
            for (auto &entry : species) {
                if (!totals.contains(entry.name))
                    throw std::runtime_error(
                        "Missing consequence total species");
                entry.totals = totals_restore(totals.at(entry.name));
                if (entry.staged) {
                    if (!staged.contains(entry.name))
                        throw std::runtime_error(
                            "Missing staged total species");
                    entry.staged_totals = totals_restore(staged.at(entry.name));
                }
            }
            totals_order.clear();
            for (const auto &[name, _] : totals.fields())
                totals_order.push_back(indices.at(name));
            staged_totals_order.clear();
            for (const auto &[name, _] : staged.fields()) {
                auto index = indices.at(name);
                if (!species[index].staged)
                    throw std::runtime_error("Unexpected staged totals");
                staged_totals_order.push_back(index);
            }
            for (const auto &entry : species) {
                auto validate = [](const Table &table, const Totals &totals) {
                    Totals retained{};
                    for (const auto &[key, row] : table.rows)
                        retained[key[0]].add(row);
                    for (int a = 0; a < 12; ++a)
                        for (int p = 0; p < 4; ++p)
                            if (totals[a].values[p] < retained[a].values[p])
                                throw std::runtime_error(
                                    "Persistent totals omit retained evidence");
                };
                validate(entry.published, entry.totals);
                if (entry.staged)
                    validate(*entry.staged, *entry.staged_totals);
            }
            return;
        }
        if (version != 1 && version != 2)
            throw std::runtime_error("Unsupported consequence model version");
        if (version == 2) {
            fields(state, {"version", "horizon_steps", "contracts", "updates",
                           "tables", "update_schedule", "published_updates",
                           "staged_tables"});
            if (!episode || state.text("update_schedule") != "episode")
                throw std::runtime_error(
                    "Invalid episode consequence schedule");
        } else
            fields(state, {"version", "horizon_steps", "contracts", "updates",
                           "tables"});
        if (integer(state.at("horizon_steps")) != horizon ||
            !equivalent(state.at("contracts"), contracts))
            throw std::runtime_error(
                "Incompatible consequence physical/horizon contract");
        updates = integer(state.at("updates"));
        if (updates < 0)
            throw std::runtime_error("Negative consequence updates");
        published_updates = updates;
        const auto &tables = state.at("tables");
        if (tables.fields().size() != species.size())
            throw std::runtime_error("Invalid consequence species");
        for (auto &entry : species) {
            if (!tables.contains(entry.name))
                throw std::runtime_error("Missing consequence species");
            table_restore(entry.published, persistent ? &entry.totals : nullptr,
                          tables.at(entry.name));
        }
        if (version == 2) {
            published_updates = integer(state.at("published_updates"));
            if (published_updates < 0 || published_updates > updates)
                throw std::runtime_error(
                    "Invalid published consequence updates");
            const auto &staged = state.at("staged_tables");
            for (const auto &[name, rows] : staged.fields()) {
                auto index = indices.at(name);
                auto &entry = species[index];
                entry.staged.emplace();
                if (persistent)
                    entry.staged_totals.emplace();
                table_restore(*entry.staged,
                              persistent ? &*entry.staged_totals : nullptr,
                              rows);
                staged_order.push_back(index);
                if (persistent)
                    staged_totals_order.push_back(index);
            }
        }
    }
    Value model_state(bool exporting) const {
        Value tables = Value::object();
        for (const auto &entry : species)
            tables[entry.name] =
                (exporting && entry.staged ? *entry.staged : entry.published)
                    .state();
        Value state = object({{"version", 1},
                              {"horizon_steps", horizon},
                              {"contracts", contracts},
                              {"updates", updates},
                              {"tables", tables}});
        if (episode && !exporting) {
            state["version"] = 2;
            state["update_schedule"] = "episode";
            state["published_updates"] = published_updates;
            Value staged = Value::object();
            for (auto index : staged_order)
                staged[species[index].name] = species[index].staged->state();
            state["staged_tables"] = staged;
        }
        return state;
    }
    Value state(bool exporting) const {
        auto model = model_state(exporting);
        if (!persistent)
            return model;
        Value totals = Value::object(), staged = Value::object();
        for (auto index : totals_order) {
            const auto &entry = species[index];
            totals[entry.name] = totals_state(exporting && entry.staged_totals
                                                  ? *entry.staged_totals
                                                  : entry.totals);
        }
        if (!exporting)
            for (auto index : staged_totals_order)
                staged[species[index].name] =
                    totals_state(*species[index].staged_totals);
        return object({{"version", 3},
                       {"model", model},
                       {"action_totals", totals},
                       {"staged_action_totals", staged}});
    }
};
ConsequenceModel::ConsequenceModel(const Value &config, const Value &state)
    : impl_(std::make_unique<Impl>(config)) {
    if (!state.null())
        impl_->restore_model(state);
}
ConsequenceModel::~ConsequenceModel() = default;
ConsequenceModel::ConsequenceModel(ConsequenceModel &&) noexcept = default;
ConsequenceModel &
ConsequenceModel::operator=(ConsequenceModel &&) noexcept = default;
bool ConsequenceModel::has_species(const std::string &species) const {
    return impl_->indices.count(species);
}
int ConsequenceModel::horizon() const { return impl_->horizon; }
void ConsequenceModel::learn(const std::string &name, const Context &key,
                             const Labels &labels) {
    auto &p = *impl_;
    if (key[0] < 0 || key[0] >= 12)
        throw std::runtime_error("Invalid consequence action");
    if (p.updates == std::numeric_limits<std::int64_t>::max())
        throw std::runtime_error("Native consequence update capacity exceeded");
    auto index = p.indices.at(name);
    auto &s = p.species[index];
    Table *table = &s.published;
    if (p.episode) {
        if (!s.staged) {
            s.staged = s.published;
            p.staged_order.push_back(index);
        }
        table = &*s.staged;
    }
    if (p.persistent) {
        Totals *totals = &s.totals;
        if (p.episode) {
            if (!s.staged_totals) {
                s.staged_totals = s.totals;
                p.staged_totals_order.push_back(index);
            }
            totals = &*s.staged_totals;
        }
        (*totals)[key[0]].learn(labels, 4503599627370496.0);
    }
    table->learn(key, labels, p.capacity);
    increment(p.updates);
    if (!p.episode) {
        s.cache.reset();
        p.published_updates = p.updates;
    }
}
ConsequenceModel::Rates ConsequenceModel::predict(const std::string &name,
                                                  const Context &key) const {
    const auto *row = impl_->get(name).published.find(key);
    return row ? row->rates() : Rates{};
}
ConsequenceModel::Rates
ConsequenceModel::predict_action(const std::string &name, int action) const {
    if (action < 0 || action >= 12)
        throw std::runtime_error("Invalid consequence action");
    auto &s = impl_->get(name);
    if (!s.cache) {
        Totals rows = s.totals;
        if (!impl_->persistent) {
            rows = {};
            for (const auto &[key, row] : s.published.rows)
                rows[key[0]].add(row);
        }
        std::array<Rates, 12> rates{};
        for (int i = 0; i < 12; ++i)
            rates[i] = rows[i].rates();
        s.cache = rates;
    }
    return (*s.cache)[action];
}
Value ConsequenceModel::checkpoint() const { return impl_->state(false); }
Value ConsequenceModel::export_state() const { return impl_->state(true); }
void ConsequenceModel::publish() {
    auto &p = *impl_;
    for (auto index : p.staged_order) {
        auto &s = p.species[index];
        s.published = std::move(*s.staged);
        s.staged.reset();
        s.cache.reset();
    }
    p.staged_order.clear();
    for (auto index : p.staged_totals_order) {
        auto &s = p.species[index];
        s.totals = *s.staged_totals;
        s.staged_totals.reset();
    }
    p.staged_totals_order.clear();
    p.published_updates = p.updates;
}
namespace {
int floor_bucket(int value) {
    return value >= 0 ? value / 8 : -((-value + 7) / 8);
}
struct Sight {
    int x, y, radius, bonus, heading, degrees;
};
struct Actor {
    int id, x, y, heading, age;
    std::string species, diet;
    double energy, health;
    const Value *body;
    Sight sight;
    bool danger_ready = false, danger = false;
    std::array<std::optional<Key>, 12> contexts;
    std::map<Pos, bool> visibility;
};
std::optional<std::pair<int, int>> social_contract(const Value &v) {
    if (!v.contains("social_observation"))
        return {};
    const auto &c = v.at("social_observation");
    if (!std::holds_alternative<Value::ObjectPtr>(c.data))
        throw std::runtime_error("Invalid social observation contract");
    for (const auto &[key, _] : c.fields())
        if (key != "history_steps" && key != "max_neighbors")
            throw std::runtime_error("Invalid social observation contract key");
    int history = c.integer("history_steps", 4),
        neighbors = c.integer("max_neighbors", 8);
    if (history < 1 || history > 16 || neighbors < 1 || neighbors > 16)
        throw std::runtime_error("Invalid social observation bounds");
    return std::make_pair(history, neighbors);
}
double norm(double x, double y) {
    x = std::abs(x);
    y = std::abs(y);
    if (x < y)
        std::swap(x, y);
    if (!x)
        return 0;
    double xx = x * x, yy = y * y, sum = xx + yy,
           error = std::fma(x, x, -xx) + std::fma(y, y, -yy) + (xx - sum + yy),
           root = std::sqrt(sum);
    return root + (std::fma(-root, root, sum) + error) / (2 * root);
}
int flow_bin(double x, double y) {
    double largest = std::max(std::fabs(x), std::fabs(y));
    if (largest >= 1.5)
        return 3;
    if (largest < 0.25)
        return 0;
    return std::min(3, static_cast<int>(norm(x, y) * 2));
}
bool in_field(const Sight &s, Pos target) {
    if (s.degrees == 360)
        return true;
    if (s.degrees != 150 && s.degrees != 180)
        throw std::runtime_error("Invalid social visual field");
    int dx = target.first - s.x, dy = target.second - s.y,
        dot = dx * directions[s.heading][0] + dy * directions[s.heading][1];
    double cosine = s.degrees == 180 ? 0 : (2 - std::sqrt(3.0)) / 4;
    return dot >= 0 && (!cosine || dot * dot >= (dx * dx + dy * dy) * cosine);
}
int depth_bin(double d) { return d <= 0 ? 0 : d <= 0.5 ? 1 : d <= 1 ? 2 : 3; }

} // namespace
struct SocialState::Impl {
    struct Sighting {
        int tick;
        std::array<std::int64_t, 24> counts{};
    };
    struct ObserverHistory {
        int id;
        std::deque<Sighting> history;
    };
    struct Window {
        int tick;
        Key key;
        Labels opening;
    };
    struct Pending {
        int id;
        std::string species;
        Labels totals{};
        std::deque<Window> queue;
    };
    struct Witness {
        int observer;
        std::vector<int> neighbors;
    };
    Value config;
    bool training = true, before_valid = false;
    std::unique_ptr<ConsequenceModel> knowledge;
    std::vector<ObserverHistory> sightings;
    std::unordered_map<int, std::size_t> sighting_index;
    std::vector<Pending> pending;
    std::unordered_map<int, std::size_t> pending_index;
    std::map<int, Pos> before_positions;
    std::map<int, std::string> before_actions;
    std::vector<Witness> witnesses;
    World *phase_world = nullptr;
    int phase_tick = -1;
    std::vector<Actor> actors;
    std::unordered_map<int, std::size_t> actor_index;
    std::map<Pos, std::vector<std::size_t>> buckets;
    std::map<Pos, double> food;
    explicit Impl(const Value &configuration) : config(configuration) {
        bool configured = false;
        for (const auto &[_, spec] : config.at("species").fields())
            configured |= config.at("brains")
                              .at(spec.text("brain"))
                              .flag("consequence_observation");
        if (config.get("consequence_learning").flag("enabled") && configured)
            knowledge = std::make_unique<ConsequenceModel>(config);
    }
    void indices() {
        sighting_index.clear();
        for (std::size_t i = 0; i < sightings.size(); ++i)
            sighting_index[sightings[i].id] = i;
        pending_index.clear();
        for (std::size_t i = 0; i < pending.size(); ++i)
            pending_index[pending[i].id] = i;
    }
    const Value &brain(const std::string &species) const {
        return config.at("brains").at(
            config.at("species").at(species).text("brain"));
    }
    void prepare(World &w) {
        phase_world = &w;
        phase_tick = w.tick();
        actors.clear();
        actor_index.clear();
        buckets.clear();
        food.clear();
        bool water = false;
        if (w.has_terrain())
            for (double depth : w.geography()->depth)
                if (depth > 0) {
                    water = true;
                    break;
                }
        for (const auto &a : static_cast<const World &>(w).animals()) {
            const auto &body = w.body(a);
            Actor view;
            view.id = a.integer("id");
            view.x = a.integer("x");
            view.y = a.integer("y");
            view.heading = a.integer("heading");
            view.age = a.integer("age");
            view.species = a.text("species");
            view.diet = w.config.at("species").at(view.species).text("diet");
            view.energy = a.num("energy");
            view.health = a.num("health");
            view.body = &body;
            view.sight = {view.x,
                          view.y,
                          body.integer("vision_radius"),
                          water ? body.integer("water_vision_bonus", 0) : 0,
                          view.heading,
                          body.integer("field_of_view_degrees", 360)};
            actor_index[view.id] = actors.size();
            buckets[{floor_bucket(view.x), floor_bucket(view.y)}].push_back(
                actors.size());
            actors.push_back(std::move(view));
        }
        for (const auto &r : static_cast<const World &>(w).resources())
            food[{r.integer("x"), r.integer("y")}] = r.num("quantity");
    }
    void ensure(World &w) {
        if (phase_world != &w || phase_tick != w.tick())
            prepare(w);
    }
    bool visible(World &w, Actor &observer, Pos target) {
        auto old = observer.visibility.find(target);
        if (old != observer.visibility.end())
            return old->second;
        const auto &s = observer.sight;
        int dx = target.first - s.x, dy = target.second - s.y,
            distance = dx * dx + dy * dy;
        bool result = in_field(s, target) &&
                      distance <= (s.radius + s.bonus) * (s.radius + s.bonus);
        if (result) {
            int allowed =
                s.radius +
                (s.bonus ? std::min(s.bonus,
                                    w.water_sight_cells(s.x, s.y, target.first,
                                                        target.second))
                         : 0);
            result = distance <= allowed * allowed &&
                     w.visible(s.x, s.y, target.first, target.second);
        }
        observer.visibility[target] = result;
        return result;
    }
    bool danger(World &w, Actor &a) {
        if (a.danger_ready)
            return a.danger;
        a.danger_ready = true;
        for (int by = floor_bucket(a.y - 2); by <= floor_bucket(a.y + 2); ++by)
            for (int bx = floor_bucket(a.x - 2); bx <= floor_bucket(a.x + 2);
                 ++bx) {
                auto p = buckets.find({bx, by});
                if (p == buckets.end())
                    continue;
                for (auto index : p->second) {
                    const auto &other = actors[index];
                    if (other.id != a.id && other.diet != a.diet &&
                        std::abs(other.x - a.x) <= 2 &&
                        std::abs(other.y - a.y) <= 2 &&
                        visible(w, a, {other.x, other.y})) {
                        a.danger = true;
                        return true;
                    }
                }
            }
        return false;
    }
    const Key &context(World &w, Actor &a, int action) {
        auto &cache = a.contexts[action];
        if (cache)
            return *cache;
        Pos source{a.x, a.y},
            target =
                action < 4 || action == 9
                    ? w.movement_destination(w.animal(a.id), actions[action])
                    : source;
        double sd = w.depth(source.first, source.second),
               td = w.depth(target.first, target.second),
               climb = w.elevation(target.first, target.second) -
                       w.elevation(source.first, source.second);
        auto flow = w.flow(source.first, source.second);
        bool blocked = w.blocked(target.first, target.second);
        const auto &spec = w.config.at("species").at(a.species);
        const auto &rules = spec.get("locomotion");
        double minimum =
            rules.num("swim_min_health", a.body->num("max_health") * 0.5) *
            std::max(sd, td);
        auto resource = food.find(target);
        cache = Key{
            action,
            std::min(3,
                     std::max(0, static_cast<int>(4 * a.energy /
                                                  a.body->num("max_energy")))),
            std::min(3,
                     std::max(0, static_cast<int>(4 * a.health /
                                                  a.body->num("max_health")))),
            depth_bin(sd),
            depth_bin(td),
            std::max(-2, std::min(2, static_cast<int>(climb))),
            flow_bin(flow.first, flow.second),
            static_cast<int>(a.health >= minimum),
            static_cast<int>(!blocked && resource != food.end() &&
                             resource->second > 0.1),
            static_cast<int>(blocked),
            static_cast<int>(a.age >=
                             spec.get("reproduction").integer("min_age_steps")),
            static_cast<int>(danger(w, a))};
        return *cache;
    }
    ObserverHistory &history(int id) {
        auto p = sighting_index.find(id);
        if (p != sighting_index.end())
            return sightings[p->second];
        sighting_index[id] = sightings.size();
        sightings.push_back({id, {}});
        return sightings.back();
    }
    Pending &pending_for(int id, const std::string &species) {
        auto p = pending_index.find(id);
        if (p != pending_index.end())
            return pending[p->second];
        pending_index[id] = pending.size();
        pending.push_back({id, species, {}, {}});
        return pending.back();
    }
    void restore(const Value &state) {
        fields(state,
               {"version", "training", "knowledge", "sightings", "pending"});
        if (integer(state.at("version")) != 1 ||
            !std::holds_alternative<bool>(state.at("training").data))
            throw std::runtime_error(
                "Invalid social checkpoint version/training");
        training = state.flag("training");
        if (!knowledge && (!state.at("knowledge").null() ||
                           !state.at("pending").fields().empty()))
            throw std::runtime_error("Unexpected consequence state");
        if (knowledge) {
            if (state.at("knowledge").null())
                throw std::runtime_error("Missing consequence state");
            knowledge = std::make_unique<ConsequenceModel>(
                config, state.at("knowledge"));
        }
        auto id = [](const std::string &key) {
            if (key.empty() ||
                !std::all_of(key.begin(), key.end(), [](unsigned char c) {
                    return c >= '0' && c <= '9';
                }))
                throw std::runtime_error("Invalid social identity");
            std::size_t consumed = 0;
            long n = std::stol(key, &consumed);
            if (consumed != key.size() || n > std::numeric_limits<int>::max())
                throw std::runtime_error("Invalid social identity capacity");
            return static_cast<int>(n);
        };
        for (const auto &[name, rows] : state.at("sightings").fields()) {
            ObserverHistory entry{id(name), {}};
            if (!rows.is_array() || rows.array_size() > 16)
                throw std::runtime_error("Oversized social sighting history");
            int last = -1;
            for (const auto &row : rows.items()) {
                if (!row.is_array() || row.array_size() != 2)
                    throw std::runtime_error("Invalid social sighting row");
                int tick = row.array_value(0).as_int();
                const auto &counts = row.items()[1];
                if (tick <= last || !counts.is_array() ||
                    counts.array_size() != 24)
                    throw std::runtime_error(
                        "Invalid social sighting ticks/counts");
                Sighting item{tick, {}};
                for (int i = 0; i < 24; ++i) {
                    item.counts[i] = integer(counts.array_value(i));
                    if (item.counts[i] < 0 || item.counts[i] > 16)
                        throw std::runtime_error(
                            "Invalid social sighting count");
                }
                std::int64_t n = 0;
                for (int i = 0; i < 24; i += 4) {
                    n += item.counts[i];
                    for (int j = 1; j < 4; ++j)
                        if (item.counts[i + j] > item.counts[i])
                            throw std::runtime_error(
                                "Inconsistent social sighting outcomes");
                }
                if (n > 16)
                    throw std::runtime_error(
                        "Inconsistent social neighbor counts");
                entry.history.push_back(item);
                last = tick;
            }
            auto existing = sighting_index.find(entry.id);
            if (existing == sighting_index.end()) {
                sighting_index[entry.id] = sightings.size();
                sightings.push_back(std::move(entry));
            } else
                sightings[existing->second] = std::move(entry);
        }
        for (const auto &[name, row] : state.at("pending").fields()) {
            if (!row.is_array() || row.array_size() != 3)
                throw std::runtime_error("Invalid pending social entry");
            Pending entry{id(name),
                          row.array_value(0).str(),
                          labels_from(row.array_value(1)),
                          {}};
            const auto &queue = row.items()[2];
            if (!knowledge || !knowledge->has_species(entry.species) ||
                !queue.is_array() ||
                queue.array_size() >
                    static_cast<std::size_t>(knowledge->horizon()))
                throw std::runtime_error("Invalid pending consequence history");
            int last = -1;
            for (const auto &window : queue.items()) {
                if (!window.is_array() || window.array_size() != 3)
                    throw std::runtime_error(
                        "Invalid pending consequence window");
                int tick = window.array_value(0).as_int();
                auto key = key_from(window.array_value(1), false);
                auto opening = labels_from(window.array_value(2));
                if (tick <= last)
                    throw std::runtime_error(
                        "Invalid pending consequence tick");
                for (int i = 0; i < 3; ++i)
                    if (opening[i] > entry.totals[i])
                        throw std::runtime_error(
                            "Invalid pending consequence opening");
                entry.queue.push_back({tick, key, opening});
                last = tick;
            }
            auto existing = pending_index.find(entry.id);
            if (existing == pending_index.end()) {
                pending_index[entry.id] = pending.size();
                pending.push_back(std::move(entry));
            } else
                pending[existing->second] = std::move(entry);
        }
        indices();
    }
};
SocialState::SocialState(const Value &config, const Value &state)
    : impl_(std::make_unique<Impl>(config)) {
    if (!state.null())
        impl_->restore(state);
}
SocialState::~SocialState() = default;
SocialState::SocialState(SocialState &&) noexcept = default;
SocialState &SocialState::operator=(SocialState &&) noexcept = default;
void SocialState::prepare(World &w) { impl_->prepare(w); }
void SocialState::begin(World &w, const std::map<int, std::string> &chosen) {
    auto &p = *impl_;
    p.prepare(w);
    p.before_positions.clear();
    p.before_actions = chosen;
    p.witnesses.clear();
    for (auto &a : p.actors) {
        p.before_positions[a.id] = {a.x, a.y};
        auto contract = social_contract(p.brain(a.species));
        if (contract) {
            std::vector<std::tuple<int, int>> nearest;
            int radius = a.sight.radius + a.sight.bonus;
            for (int by = floor_bucket(a.y - radius);
                 by <= floor_bucket(a.y + radius); ++by)
                for (int bx = floor_bucket(a.x - radius);
                     bx <= floor_bucket(a.x + radius); ++bx) {
                    auto bucket = p.buckets.find({bx, by});
                    if (bucket == p.buckets.end())
                        continue;
                    for (auto index : bucket->second) {
                        const auto &other = p.actors[index];
                        if (other.id != a.id && other.species == a.species &&
                            p.visible(w, a, {other.x, other.y})) {
                            int dx = other.x - a.x, dy = other.y - a.y;
                            nearest.emplace_back(dx * dx + dy * dy, other.id);
                        }
                    }
                }
            std::sort(nearest.begin(), nearest.end());
            Impl::Witness witness{a.id, {}};
            for (int i = 0; i < std::min<int>(contract->second, nearest.size());
                 ++i)
                witness.neighbors.push_back(std::get<1>(nearest[i]));
            p.witnesses.push_back(std::move(witness));
        }
        if (p.training && p.knowledge && p.knowledge->has_species(a.species)) {
            auto &entry = p.pending_for(a.id, a.species);
            entry.queue.push_back(
                {w.tick(), p.context(w, a, action_index(chosen.at(a.id))),
                 entry.totals});
        }
    }
    p.before_valid = true;
}
void SocialState::finish(World &w, const Value &events, const Value &dead) {
    auto &p = *impl_;
    if (!p.before_valid)
        throw std::runtime_error("Social finish requires begin");
    p.before_valid = false;
    p.prepare(w);
    std::map<int, const Value *> all;
    std::set<int> dead_ids;
    for (const auto &a : dead.items()) {
        int id = a.integer("id");
        all[id] = &a;
        dead_ids.insert(id);
    }
    for (const auto &a : static_cast<const World &>(w).animals())
        all[a.integer("id")] = &a;
    std::map<int, Labels> outcomes;
    std::vector<const Value *> births;
    for (const auto &e : events.items()) {
        auto kind = e.text("kind");
        int actor = e.integer("actor");
        if ((kind == "damage_taken" && e.num("amount") > 0) || kind == "died")
            outcomes[actor][0] = 1;
        else if (kind == "energy_gained" && e.num("amount") > 0)
            outcomes[actor][1] = 1;
        else if (kind == "born") {
            outcomes[actor][2] = 1;
            int mate = e.integer("mate", -1);
            if (mate >= 0)
                outcomes[mate][2] = 1;
            births.push_back(&e);
        }
    }
    for (const auto &record : p.witnesses) {
        auto index = p.actor_index.find(record.observer);
        if (index == p.actor_index.end())
            continue;
        auto &observer = p.actors[index->second];
        auto contract = social_contract(p.brain(observer.species));
        if (!contract)
            throw std::runtime_error("Social observer lost its contract");
        Impl::Sighting current{w.tick(), {}};
        for (int id : record.neighbors) {
            auto found = all.find(id);
            if (found == all.end())
                continue;
            const auto &subject = *found->second;
            Pos position{subject.integer("x"), subject.integer("y")};
            const auto &prior = p.before_positions.at(id);
            if (!p.visible(w, observer, position) ||
                (position != prior && !p.visible(w, observer, prior)))
                continue;
            int offset = groups[action_index(p.before_actions.at(id))] * 4;
            ++current.counts[offset];
            auto labels = outcomes[id];
            if (labels[2]) {
                bool visible_birth = false;
                for (auto e : births)
                    if ((id == e->integer("actor") ||
                         id == e->integer("mate", -1)) &&
                        p.actor_index.count(e->integer("target"))) {
                        const auto &child =
                            p.actors[p.actor_index.at(e->integer("target"))];
                        if (p.visible(w, observer, {child.x, child.y})) {
                            visible_birth = true;
                            break;
                        }
                    }
                labels[2] = visible_birth;
            }
            for (int i = 0; i < 3; ++i)
                current.counts[offset + 1 + i] += labels[i];
        }
        auto &history = p.history(observer.id).history;
        history.push_back(current);
        while (!history.empty() &&
               history.front().tick <= w.tick() - contract->first)
            history.pop_front();
    }
    p.sightings.erase(std::remove_if(p.sightings.begin(), p.sightings.end(),
                                     [&](const auto &entry) {
                                         return !p.actor_index.count(entry.id);
                                     }),
                      p.sightings.end());
    if (p.training && p.knowledge) {
        for (auto &entry : p.pending) {
            for (int i = 0; i < 3; ++i) {
                auto value = outcomes[entry.id][i];
                if (entry.totals[i] >
                    std::numeric_limits<std::int64_t>::max() - value)
                    throw std::runtime_error(
                        "Native social outcome capacity exceeded");
                entry.totals[i] += value;
            }
            bool died = dead_ids.count(entry.id);
            while (!entry.queue.empty() &&
                   (died || w.tick() - entry.queue.front().tick >=
                                p.knowledge->horizon())) {
                const auto opening = entry.queue.front();
                entry.queue.pop_front();
                Labels labels{};
                for (int i = 0; i < 3; ++i)
                    labels[i] = entry.totals[i] - opening.opening[i];
                p.knowledge->learn(entry.species, opening.key, labels);
            }
        }
        p.pending.erase(std::remove_if(p.pending.begin(), p.pending.end(),
                                       [&](const auto &entry) {
                                           return dead_ids.count(entry.id);
                                       }),
                        p.pending.end());
    }
    p.indices();
}
std::vector<double>
SocialState::observe(World &w, const Value &animal, const Value &templ,
                     const std::vector<std::string> *compact_actions) {
    auto &p = *impl_;
    p.ensure(w);
    int id = animal.integer("id");
    auto index = p.actor_index.find(id);
    if (index == p.actor_index.end())
        throw std::runtime_error("Social observer must be alive");
    auto &a = p.actors[index->second];
    std::vector<double> features;
    auto contract = social_contract(templ);
    if (contract) {
        std::array<std::int64_t, 24> total{};
        auto old = p.sighting_index.find(id);
        if (old != p.sighting_index.end())
            for (const auto &row : p.sightings[old->second].history)
                if (w.tick() - contract->first < row.tick &&
                    row.tick <= w.tick())
                    for (int i = 0; i < 24; ++i)
                        total[i] += row.counts[i];
        for (int i = 0; i < 24; i += 4) {
            double n = static_cast<double>(total[i]);
            features.push_back(n / (n + 4));
            for (int j = 1; j < 4; ++j)
                features.push_back(n ? total[i + j] / n : 0.0);
        }
    }
    if (templ.flag("consequence_observation")) {
        if (!p.knowledge)
            throw std::runtime_error("Missing consequence knowledge");
        std::string scope = templ.text("consequence_scope", "context");
        if (scope != "context" && scope != "species_action" && scope != "both")
            throw std::runtime_error("Invalid consequence scope");
        const auto &allowed =
            w.config.at("species").at(a.species).at("allowed_actions").items();
        std::array<bool, 12> enabled{};
        for (const auto &item : allowed)
            enabled[action_index(item.str())] = true;
        std::vector<int> rows;
        if (compact_actions) {
            for (const auto &name : *compact_actions)
                rows.push_back(action_index(name));
        } else
            for (int i = 0; i < 12; ++i)
                rows.push_back(i);
        for (int block = 0; block < (scope == "both" ? 2 : 1); ++block) {
            bool general =
                scope == "species_action" || (scope == "both" && block == 0);
            for (int action : rows) {
                Rates values{};
                if (enabled[action])
                    values =
                        general ? p.knowledge->predict_action(a.species, action)
                                : p.knowledge->predict(a.species,
                                                       p.context(w, a, action));
                features.insert(features.end(), values.begin(), values.end());
            }
        }
    }
    return features;
}
Value SocialState::checkpoint() const {
    const auto &p = *impl_;
    Value sightings = Value::object(), pending = Value::object();
    for (const auto &entry : p.sightings) {
        Value history = Value::array();
        for (const auto &row : entry.history) {
            Value counts = Value::array();
            for (auto n : row.counts)
                counts.push(n);
            history.push(array({row.tick, counts}));
        }
        sightings[std::to_string(entry.id)] = history;
    }
    for (const auto &entry : p.pending) {
        Value queue = Value::array();
        for (const auto &row : entry.queue)
            queue.push(array(
                {row.tick, key_state(row.key), labels_state(row.opening)}));
        pending[std::to_string(entry.id)] =
            array({entry.species, labels_state(entry.totals), queue});
    }
    return object(
        {{"version", 1},
         {"training", p.training},
         {"knowledge", p.knowledge ? p.knowledge->checkpoint() : Value{}},
         {"sightings", sightings},
         {"pending", pending}});
}
bool SocialState::training() const { return impl_->training; }
void SocialState::set_training(bool training) {
    auto &p = *impl_;
    if (p.training != training) {
        p.pending.clear();
        p.pending_index.clear();
    }
    p.training = training;
}
bool SocialState::has_knowledge() const { return bool(impl_->knowledge); }
Value SocialState::knowledge_checkpoint() const {
    return impl_->knowledge ? impl_->knowledge->checkpoint() : Value{};
}
Value SocialState::knowledge_export() const {
    return impl_->knowledge ? impl_->knowledge->export_state() : Value{};
}
void SocialState::publish() {
    if (impl_->knowledge)
        impl_->knowledge->publish();
}
} // namespace wtfcore
