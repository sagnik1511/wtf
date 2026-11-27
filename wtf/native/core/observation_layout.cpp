#include "observation_layout.hpp"
#include <cstring>
#include <map>
#include <mutex>
namespace wtfcore {
namespace {
bool starts(const std::string &s, const std::string &p) {
    return s.compare(0, p.size(), p) == 0;
}
bool ends(const std::string &s, const std::string &p) {
    return s.size() >= p.size() &&
           s.compare(s.size() - p.size(), p.size(), p) == 0;
}
std::vector<std::pair<std::string, std::string>> channels(const Value &c) {
    const auto &s = c.get("scent_observation");
    std::vector<std::pair<std::string, std::string>> out;
    if (s.null())
        return out;
    if (s.text("mode", "separated") == "blended") {
        out.emplace_back("blended", "any");
        return out;
    }
    for (const auto &v : s.at("species").items()) {
        if (s.flag("separate_sexes", true)) {
            out.emplace_back(v.str(), "female");
            out.emplace_back(v.str(), "male");
        } else
            out.emplace_back(v.str(), "any");
    }
    return out;
}
std::vector<std::string> raw_names(const Value &c, int version) {
    std::vector<std::string> out = {
        "Energy",        "Health",        "Heading x",    "Heading y",
        "Blocked front", "Blocked right", "Blocked back", "Blocked left"};
    int k = c.integer("observation_top_k", 1),
        h = c.integer("movement_history_steps");
    for (const auto *category : {"Food", "Prey", "Predator"})
        for (int n = 1; n <= k; ++n)
            for (const auto &part : std::vector<std::string>{
                     version == 5 ? "forward" : "dx",
                     version == 5 ? "right" : "dy", "distance", "present"})
                out.push_back(std::string(category) + " " + std::to_string(n) +
                              " " + part);
    if (version >= 2) {
        for (const auto &name : std::vector<std::string>{
                 "Water depth", version == 5 ? "Flow forward" : "Flow x",
                 version == 5 ? "Flow right" : "Flow y", "Elevation"})
            out.push_back(name);
        for (const auto *side : {"Front", "Right", "Back", "Left"}) {
            out.push_back(std::string(side) + " water");
            out.push_back(std::string(side) + " climb");
        }
    }
    if (version >= 3)
        for (const auto &name : std::vector<std::string>{
                 "Sex", "Maturity", "Cooldown", "Self ready",
                 version == 5 ? "Mate forward" : "Mate dx",
                 version == 5 ? "Mate right" : "Mate dy", "Mate distance",
                 "Mate present", "Mate ready", "Pair opportunity"})
            out.push_back(name);
    if (version >= 4)
        for (const auto *name :
             {"Self consent", "Mate consent", "Consenting pair"})
            out.emplace_back(name);
    if (version == 5)
        for (const auto *name :
             {"Attack possible now", "Eat possible now", "Swim possible now"})
            out.emplace_back(name);
    if (h)
        for (const auto *category : {"Prey", "Predator"})
            for (int n = 1; n <= k; ++n) {
                std::string prefix =
                    std::string(category) + " track " + std::to_string(n) + " ";
                for (const auto &part : std::vector<std::string>{
                         version == 5 ? "last forward" : "last dx",
                         version == 5 ? "last right" : "last dy",
                         version == 5 ? "velocity forward" : "velocity x",
                         version == 5 ? "velocity right" : "velocity y", "age",
                         "confidence", "visible"})
                    out.push_back(prefix + part);
                for (int lag = 1; lag <= h; ++lag)
                    for (const auto &part : std::vector<std::string>{
                             version == 5 ? "forward" : "dx",
                             version == 5 ? "right" : "dy", "seen"})
                        out.push_back(prefix + "lag " + std::to_string(lag) +
                                      " " + part);
            }
    const std::vector<std::string>
        world = {"here",  "north",     "northeast", "east",     "southeast",
                 "south", "southwest", "west",      "northwest"},
        body = {"here", "front",     "front-right", "right",     "back-right",
                "back", "back-left", "left",        "front-left"};
    for (const auto &[name, sex] : channels(c)) {
        std::string prefix =
            name == "blended" &&
                    c.at("scent_observation").text("mode") == "blended"
                ? "Scent blended"
                : "Scent " + name + " " + sex;
        for (const auto &part : version == 5 ? body : world)
            out.push_back(prefix + " " + part);
    }
    if (!c.get("social_observation").null())
        for (const auto *group :
             {"walk", "swim", "feed", "combat", "reproduce", "idle"})
            for (const auto *part : {"confidence", "harm", "feeding", "birth"})
                out.push_back(std::string("Collective ") + group + " " + part);
    if (c.flag("consequence_observation")) {
        std::string scope = c.text("consequence_scope", "context");
        std::vector<std::string> prefixes =
            scope == "both"
                ? std::vector<std::string>{"Species action", "Consequence"}
                : std::vector<std::string>{scope == "species_action"
                                               ? "Species action"
                                               : "Consequence"};
        for (const auto &prefix : prefixes)
            for (const auto &action : canonical_actions())
                for (const auto *part :
                     {"confidence", "harm", "feeding", "birth"})
                    out.push_back(prefix + " " + action + " " + part);
    }
    return out;
}
std::string scalar_kind(const std::string &name) {
    if (name == "Energy")
        return "energy";
    if (name == "Health")
        return "health";
    if (name == "Elevation")
        return "elevation";
    if (name == "Maturity" || name == "Cooldown" || name == "Self consent" ||
        name == "Mate consent" || ends(name, " age") ||
        ends(name, " confidence") || starts(name, "Scent "))
        return "level";
    if (name == "Water depth" || ends(name, " water"))
        return "depth";
    if (ends(name, " climb"))
        return "climb";
    if ((starts(name, "Collective ") || starts(name, "Species action ") ||
         starts(name, "Consequence ")) &&
        (ends(name, " harm") || ends(name, " feeding") || ends(name, " birth")))
        return "outcome";
    return "raw";
}
const std::map<std::string, std::vector<double>> &defaults() {
    static const std::map<std::string, std::vector<double>> d = {
        {"distance", {0, .3, 1}},
        {"energy", {0, .25, .75, 1}},
        {"health", {0, .5, 1}},
        {"level", {0, .5, 1}},
        {"depth", {0, .1, 1}},
        {"elevation", {0, .1, 1}},
        {"climb", {-1, -.1, 0, .1, 1}},
        {"speed", {0, .25, std::sqrt(2.0)}},
        {"outcome", {0, .001, .01, .1, 1}}};
    return d;
}
const std::map<std::string, std::vector<std::string>> &labels() {
    static const std::map<std::string, std::vector<std::string>> d = {
        {"distance", {"near", "medium", "far"}},
        {"energy", {"critical", "hungry", "comfortable", "full"}},
        {"health", {"critical", "weak", "healthy"}},
        {"level", {"none", "partial", "strong"}},
        {"depth", {"dry", "shallow", "deep"}},
        {"elevation", {"low", "middle", "high"}},
        {"climb",
         {"steep-down", "gentle-down", "level", "gentle-up", "steep-up"}},
        {"speed", {"still", "gentle", "strong"}},
        {"outcome", {"zero", "very-low", "low", "moderate", "high"}}};
    return d;
}
} // namespace
std::vector<double> soft_memberships(double value,
                                     const std::vector<double> &knots) {
    std::vector<double> out(knots.size());
    if (value <= knots.front())
        out[0] = 1;
    else if (value >= knots.back())
        out.back() = 1;
    else {
        auto upper =
            std::upper_bound(knots.begin(), knots.end(), value) - knots.begin();
        double right =
            (value - knots[upper - 1]) / (knots[upper] - knots[upper - 1]);
        out[upper - 1] = 1 - right;
        out[upper] = right;
    }
    return out;
}
std::vector<double> soft_direction(double f, double r) {
    std::vector<double> out(8);
    if (f || r) {
        constexpr double pi = 3.141592653589793;
        double angle = std::fmod(std::atan2(r, f), 2 * pi);
        if (angle < 0)
            angle += 2 * pi;
        if (angle == 0)
            angle = 0.0;
        double sector = angle / (pi / 4);
        int index = static_cast<int>(sector) % 8;
        double fraction = sector - static_cast<int>(sector);
        out[index] = 1 - fraction;
        out[(index + 1) % 8] = fraction;
    }
    return out;
}
ObservationLayout::ObservationLayout(const Value &c) {
    version = c.integer("observation_version", 1);
    k = c.integer("observation_top_k", 1);
    history = c.integer("movement_history_steps");
    raw_names = wtfcore::raw_names(c, version >= 5 ? 5 : version);
    raw_core = (version >= 5   ? 48
                : version == 4 ? 45
                : version == 3 ? 42
                : version == 2 ? 32
                               : 20) +
               12 * (k - 1);
    tracking =
        (history ? 2 * k * (7 + 3 * history) : 0) + 9 * channels(c).size();
    if (version == 7) {
        actions = c.get("observation_actions").null()
                      ? canonical_actions()
                      : std::vector<std::string>{};
        if (!c.get("observation_actions").null())
            for (const auto &action : c.at("observation_actions").items())
                actions.push_back(action.str());
        for (std::size_t i = 0; i < raw_core; ++i) {
            const auto &name = raw_names[i];
            std::string opportunity = name == "Attack possible now" ? "attack"
                                      : name == "Eat possible now"  ? "eat"
                                      : name == "Swim possible now" ? "swim"
                                                                    : "";
            if (!ends(name, " distance") &&
                (opportunity.empty() ||
                 std::find(actions.begin(), actions.end(), opportunity) !=
                     actions.end())) {
                core_indices.push_back(i);
                names.push_back(name);
            }
        }
        if (history)
            for (const auto *category : {"Prey", "Predator"})
                for (int n = 1; n <= k; ++n)
                    for (const auto *part :
                         {"last forward", "last right", "trend forward",
                          "trend right", "freshness", "visible",
                          "motion reliability"})
                        names.push_back(std::string(category) + " track " +
                                        std::to_string(n) + " " + part);
        scent_start = names.size();
        for (const auto &[name, sex] : channels(c))
            for (const auto *part : {"direction forward", "direction right",
                                     "strength", "freshness"})
                names.push_back("Scent " + name + " " + sex + " " + part);
        physical_width = names.size();
        social_start = raw_core + tracking;
        if (!c.get("social_observation").null())
            for (std::size_t i = social_start; i < social_start + 24; ++i)
                names.push_back(raw_names.at(i));
        consequence_start = names.size();
        if (c.flag("consequence_observation")) {
            std::string scope = c.text("consequence_scope", "context");
            std::vector<std::string> prefixes =
                scope == "both"
                    ? std::vector<std::string>{"Species action", "Consequence"}
                    : std::vector<std::string>{scope == "species_action"
                                                   ? "Species action"
                                                   : "Consequence"};
            for (const auto &p : prefixes)
                for (const auto &a : actions)
                    for (const auto *part :
                         {"confidence", "harm", "feeding", "birth"})
                        names.push_back(p + " " + a + " " + part);
        }
        return;
    }
    if (version != 6) {
        names = raw_names;
        physical_width = raw_core + tracking;
        social_start = physical_width;
        consequence_start =
            social_start + (!c.get("social_observation").null() ? 24 : 0);
        return;
    }
    std::map<std::string, std::vector<double>> knots = defaults();
    for (const auto &[key, value] : c.get("soft_categories").null()
                                        ? Value::Object{}
                                        : c.at("soft_categories").fields())
        knots.at(key) = numbers(value);
    const std::vector<std::string> bearings = {
        "front", "front-right", "right", "back-right",
        "back",  "back-left",   "left",  "front-left"};
    for (std::size_t i = 0; i < raw_names.size();) {
        ObservationField field;
        field.raw = i;
        field.offset = names.size();
        field.label = raw_names[i];
        field.kind = scalar_kind(field.label);
        if (field.label == "Flow forward" ||
            ends(field.label, "velocity forward")) {
            field.count = 2;
            field.kind = "vector";
        } else if (ends(field.label, " forward") &&
                   (starts(field.label, "Food ") ||
                    starts(field.label, "Prey ") ||
                    starts(field.label, "Predator ") ||
                    starts(field.label, "Mate "))) {
            field.count =
                i + 2 < raw_names.size() && ends(raw_names[i + 2], " distance")
                    ? 3
                    : 2;
            field.kind = "position";
        }
        if (field.kind == "position" || field.kind == "vector") {
            auto key = field.kind == "position" ? "distance" : "speed";
            field.knots = knots.at(key);
            field.label = field.label.substr(0, field.label.rfind(' '));
            for (const auto &part : labels().at(key))
                names.push_back(
                    field.label +
                    (field.kind == "position" ? " range " : " speed ") + part);
            for (const auto &part : bearings)
                names.push_back(field.label + " direction " + part);
        } else if (field.kind == "raw")
            names.push_back(field.label);
        else {
            field.knots = knots.at(field.kind);
            for (const auto &part : labels().at(field.kind))
                names.push_back(field.label + " " + part);
        }
        field.width = names.size() - field.offset;
        if ((starts(field.label, "Species action ") ||
             starts(field.label, "Consequence ")) &&
            ends(field.label, " confidence"))
            confidence_offsets.push_back(field.offset);
        if (field.kind == "raw")
            zeroes.push_back(0.0);
        else if ((field.kind == "position" || field.kind == "vector") &&
                 field.count == 3)
            zeroes.insert(zeroes.end(), 11, 0.0);
        else {
            auto basis = soft_memberships(0.0, field.knots);
            zeroes.insert(zeroes.end(), basis.begin(), basis.end());
            if (field.kind == "position" || field.kind == "vector")
                zeroes.insert(zeroes.end(), 8, 0.0);
        }
        i += field.count;
        fields.push_back(std::move(field));
    }
    physical_width = offset(raw_core + tracking);
    social_start = physical_width;
    consequence_start = offset(raw_core + tracking +
                               (!c.get("social_observation").null() ? 24 : 0));
}
std::size_t ObservationLayout::offset(std::size_t raw) const {
    if (raw == raw_names.size())
        return names.size();
    auto p = std::lower_bound(
        fields.begin(), fields.end(), raw,
        [](const auto &f, std::size_t n) { return f.raw < n; });
    if (p == fields.end() || p->raw != raw)
        throw std::runtime_error("Invalid semantic field boundary");
    return p->offset;
}
std::size_t ObservationLayout::index(const std::string &name) const {
    auto p = std::find(names.begin(), names.end(), name);
    if (p == names.end())
        throw std::runtime_error("Missing observation field: " + name);
    return p - names.begin();
}
std::vector<double>
ObservationLayout::encode_soft(const std::vector<double> &values) const {
    auto out = zeroes;
    for (const auto &f : fields) {
        double value = values.at(f.raw);
        if (f.kind == "raw")
            out[f.offset] = value;
        else if (f.kind == "position" || f.kind == "vector") {
            if (f.count == 3 && !values.at(f.raw + 3))
                continue;
            double x = value, y = values.at(f.raw + 1);
            if (f.count != 3 && !(x || y))
                continue;
            double distance =
                f.count == 3 ? values.at(f.raw + 2) : exact_hypot(x, y);
            auto basis = soft_memberships(distance, f.knots);
            std::copy(basis.begin(), basis.end(), out.begin() + f.offset);
            auto direction = soft_direction(x, y);
            std::copy(direction.begin(), direction.end(),
                      out.begin() + f.offset + basis.size());
        } else if (value || std::signbit(value)) {
            auto basis = soft_memberships(value, f.knots);
            std::copy(basis.begin(), basis.end(), out.begin() + f.offset);
        }
    }
    return out;
}
std::vector<double>
ObservationLayout::decode_soft(const std::vector<double> &values,
                               std::size_t raw_start) const {
    std::vector<double> out;
    for (const auto &f : fields) {
        if (f.raw < raw_start)
            continue;
        if (f.kind == "raw") {
            out.push_back(values.at(f.offset));
            continue;
        }
        Sum weighted;
        for (std::size_t j = 0; j < f.knots.size(); ++j)
            weighted.add(values.at(f.offset + j) * f.knots[j]);
        double distance = weighted.result();
        if (f.kind == "position" || f.kind == "vector") {
            auto first = f.offset + f.knots.size();
            bool any = false;
            std::size_t anchor = 0;
            for (std::size_t j = 0; j < 8; ++j) {
                any |= values.at(first + j) != 0;
                if (values.at(first + j) > values.at(first + anchor))
                    anchor = j;
            }
            double angle = 0;
            if (any) {
                Sum sum;
                for (int j = 0; j < 8; ++j) {
                    int wrapped = (j - static_cast<int>(anchor) + 4) % 8;
                    if (wrapped < 0)
                        wrapped += 8;
                    sum.add((static_cast<int>(anchor) + wrapped - 4) *
                            values.at(first + j));
                }
                angle = sum.result() * 3.141592653589793 / 4;
            }
            out.push_back(distance * std::cos(angle));
            out.push_back(distance * std::sin(angle));
            if (f.count == 3)
                out.push_back(distance);
        } else
            out.push_back(distance);
    }
    return out;
}

namespace {
bool layout_key_equal(const Value &a, const Value &b) {
    if (a.data.index() != b.data.index())
        return false;
    if (auto x = std::get_if<double>(&a.data)) {
        double y = std::get<double>(b.data);
        return std::memcmp(x, &y, sizeof(double)) == 0;
    }
    if (auto x = std::get_if<Value::ArrayPtr>(&a.data)) {
        const auto &y = b.items();
        if ((*x)->size() != y.size())
            return false;
        for (std::size_t i = 0; i < y.size(); ++i)
            if (!layout_key_equal((**x)[i], y[i]))
                return false;
        return true;
    }
    if (auto x = std::get_if<Value::ObjectPtr>(&a.data)) {
        if ((*x)->size() != b.fields().size())
            return false;
        for (const auto &[key, value] : **x)
            if (!b.contains(key) || !layout_key_equal(value, b.at(key)))
                return false;
        return true;
    }
    return equivalent(a, b);
}
} // namespace
std::shared_ptr<const ObservationLayout>
shared_observation_layout(const Value &configuration) {
    struct Entry {
        Value key;
        std::weak_ptr<const ObservationLayout> plan;
    };
    static std::mutex mutex;
    static std::vector<Entry> cache;
    Value key = Value::object();
    for (const auto *field :
         {"observation_version", "observation_top_k", "observation_actions",
          "movement_history_steps", "scent_observation", "social_observation",
          "consequence_observation", "consequence_scope", "soft_categories"})
        if (configuration.contains(field))
            key[field] = configuration.at(field);
    std::lock_guard<std::mutex> lock(mutex);
    for (std::size_t i = 0; i < cache.size();) {
        auto plan = cache[i].plan.lock();
        if (!plan) {
            cache.erase(cache.begin() + i);
            continue;
        }
        if (layout_key_equal(cache[i].key, key)) {
            auto entry = std::move(cache[i]);
            cache.erase(cache.begin() + i);
            cache.push_back(std::move(entry));
            return plan;
        }
        ++i;
    }
    auto plan = std::make_shared<const ObservationLayout>(configuration);
    cache.push_back({std::move(key), plan});
    if (cache.size() > 128)
        cache.erase(cache.begin());
    return plan;
}
} // namespace wtfcore
