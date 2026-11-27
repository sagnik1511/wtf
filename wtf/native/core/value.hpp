#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>
namespace wtfcore {
// Native-owned boundary tree. No PyObject is retained during execution.
// Container copies share immutable storage; every mutable accessor detaches.
// Model/optimizer arithmetic uses separate private typed buffers in PPO.
struct Value {
    using Array = std::vector<Value>;
    using Object = std::vector<std::pair<std::string, Value>>;
    using ArrayPtr = std::shared_ptr<Array>;
    using ObjectPtr = std::shared_ptr<Object>;
    // Immutable copies of existing FP32 model parameters; Python still receives
    // ordinary float lists. Mutable access explicitly detaches and expands.
    using FloatArray = std::vector<float>;
    using FloatArrayPtr = std::shared_ptr<const FloatArray>;
    std::variant<std::monostate, bool, std::int64_t, std::uint64_t, double,
                 std::string, ArrayPtr, ObjectPtr, FloatArrayPtr>
        data;
    Value() = default;
    Value(bool x) : data(x) {}
    Value(int x) : data(static_cast<std::int64_t>(x)) {}
    Value(std::int64_t x) : data(x) {}
    Value(std::uint64_t x) : data(x) {}
    Value(double x) : data(x) {}
    Value(const char *x) : data(std::string(x)) {}
    Value(std::string x) : data(std::move(x)) {}
    Value(Array x) : data(std::make_shared<Array>(std::move(x))) {}
    Value(Object x) : data(std::make_shared<Object>(std::move(x))) {}
    static Value array() { return Array{}; }
    static Value object() { return Object{}; }
    static Value float_array(FloatArray values) {
        Value result;
        result.data = std::make_shared<const FloatArray>(std::move(values));
        return result;
    }
    bool is_array() const {
        return std::holds_alternative<ArrayPtr>(data) ||
               std::holds_alternative<FloatArrayPtr>(data);
    }
    std::size_t array_size() const {
        if (auto p = std::get_if<FloatArrayPtr>(&data))
            return (*p)->size();
        return items().size();
    }
    Value array_value(std::size_t index) const {
        if (auto p = std::get_if<FloatArrayPtr>(&data))
            return static_cast<double>((**p).at(index));
        return items().at(index);
    }
    bool null() const { return std::holds_alternative<std::monostate>(data); }
    bool integer() const {
        return std::holds_alternative<std::int64_t>(data) ||
               std::holds_alternative<std::uint64_t>(data);
    }
    double number() const {
        if (auto p = std::get_if<double>(&data))
            return *p;
        if (auto p = std::get_if<std::int64_t>(&data))
            return static_cast<double>(*p);
        if (auto p = std::get_if<std::uint64_t>(&data))
            return static_cast<double>(*p);
        if (auto p = std::get_if<bool>(&data))
            return *p ? 1.0 : 0.0;
        throw std::runtime_error("Expected native number");
    }
    std::int64_t i64() const {
        if (auto p = std::get_if<std::int64_t>(&data))
            return *p;
        if (auto p = std::get_if<std::uint64_t>(&data)) {
            if (*p > static_cast<std::uint64_t>(
                         std::numeric_limits<std::int64_t>::max()))
                throw std::runtime_error(
                    "Native signed integer is out of range");
            return static_cast<std::int64_t>(*p);
        }
        double value = number();
        // The upper endpoint is exclusive: converting INT64_MAX to double
        // rounds up to 2**63, which is already outside the signed range.
        if (!std::isfinite(value) || std::trunc(value) != value ||
            value < -9223372036854775808.0 || value >= 9223372036854775808.0)
            throw std::runtime_error("Native signed integer is out of range");
        return static_cast<std::int64_t>(value);
    }
    std::uint64_t u64() const {
        if (auto p = std::get_if<std::uint64_t>(&data))
            return *p;
        if (auto p = std::get_if<std::int64_t>(&data)) {
            if (*p < 0)
                throw std::runtime_error(
                    "Native unsigned integer is out of range");
            return static_cast<std::uint64_t>(*p);
        }
        double value = number();
        if (!std::isfinite(value) || std::trunc(value) != value || value < 0 ||
            value >= 18446744073709551616.0)
            throw std::runtime_error("Native unsigned integer is out of range");
        return static_cast<std::uint64_t>(value);
    }
    int as_int() const {
        double value = number();
        if (!std::isfinite(value) || std::trunc(value) != value ||
            value < std::numeric_limits<int>::min() ||
            value > std::numeric_limits<int>::max())
            throw std::runtime_error("Native integer is out of range");
        return static_cast<int>(value);
    }
    bool boolean() const {
        if (null())
            return false;
        if (auto p = std::get_if<bool>(&data))
            return *p;
        if (auto p = std::get_if<std::string>(&data))
            return !p->empty();
        if (auto p = std::get_if<ArrayPtr>(&data))
            return !(*p)->empty();
        if (auto p = std::get_if<ObjectPtr>(&data))
            return !(*p)->empty();
        if (auto p = std::get_if<FloatArrayPtr>(&data))
            return !(*p)->empty();
        return number() != 0;
    }
    const std::string &str() const { return std::get<std::string>(data); }
    Array &items() {
        if (auto packed = std::get_if<FloatArrayPtr>(&data)) {
            Array expanded;
            expanded.reserve((*packed)->size());
            for (float number : **packed)
                expanded.emplace_back(static_cast<double>(number));
            data = std::make_shared<Array>(std::move(expanded));
        }
        auto &p = std::get<ArrayPtr>(data);
        if (p.use_count() != 1)
            p = std::make_shared<Array>(*p);
        return *p;
    }
    const Array &items() const { return *std::get<ArrayPtr>(data); }
    Object &fields() {
        auto &p = std::get<ObjectPtr>(data);
        if (p.use_count() != 1)
            p = std::make_shared<Object>(*p);
        return *p;
    }
    const Object &fields() const { return *std::get<ObjectPtr>(data); }
    bool contains(const std::string &key) const {
        const auto *p = std::get_if<ObjectPtr>(&data);
        if (!p)
            return false;
        return std::any_of((*p)->begin(), (*p)->end(),
                           [&](const auto &x) { return x.first == key; });
    }
    Value &operator[](const std::string &key) {
        if (null())
            data = std::make_shared<Object>();
        for (auto &entry : fields())
            if (entry.first == key)
                return entry.second;
        fields().emplace_back(key, Value{});
        return fields().back().second;
    }
    const Value &at(const std::string &key) const {
        for (const auto &entry : fields())
            if (entry.first == key)
                return entry.second;
        throw std::runtime_error("Missing native field: " + key);
    }
    const Value &get(const std::string &key) const {
        static const Value none;
        if (auto p = std::get_if<ObjectPtr>(&data))
            for (const auto &entry : **p)
                if (entry.first == key)
                    return entry.second;
        return none;
    }
    double num(const std::string &key, double fallback = 0) const {
        const auto &v = get(key);
        return v.null() ? fallback : v.number();
    }
    int integer(const std::string &key, int fallback = 0) const {
        const auto &value = get(key);
        return value.null() ? fallback : value.as_int();
    }
    bool flag(const std::string &key, bool fallback = false) const {
        const auto &v = get(key);
        return v.null() ? fallback : v.boolean();
    }
    std::string text(const std::string &key,
                     const std::string &fallback = "") const {
        const auto &v = get(key);
        return v.null() ? fallback : v.str();
    }
    void erase(const std::string &key) {
        auto &v = fields();
        v.erase(std::remove_if(v.begin(), v.end(),
                               [&](const auto &x) { return x.first == key; }),
                v.end());
    }
    void push(Value value) { items().push_back(std::move(value)); }
};
inline Value
object(std::initializer_list<std::pair<std::string, Value>> fields) {
    return Value::Object(fields);
}
inline Value array(std::initializer_list<Value> values) {
    return Value::Array(values);
}
inline std::vector<double> numbers(const Value &value) {
    std::vector<double> out;
    out.reserve(value.array_size());
    if (auto packed = std::get_if<Value::FloatArrayPtr>(&value.data)) {
        for (float x : **packed)
            out.push_back(static_cast<double>(x));
    } else {
        for (const auto &x : value.items())
            out.push_back(x.number());
    }
    return out;
}
inline Value values(const std::vector<double> &v) {
    Value out = Value::array();
    for (double x : v)
        out.push(x);
    return out;
}
inline double exact_hypot(double x, double y) {
    x = std::abs(x);
    y = std::abs(y);
    if (x < y)
        std::swap(x, y);
    if (!x)
        return 0.0;
    double xx = x * x, yy = y * y, total = xx + yy;
    double error =
        std::fma(x, x, -xx) + std::fma(y, y, -yy) + (xx - total + yy);
    double root = std::sqrt(total);
    return root + (std::fma(-root, root, total) + error) / (2 * root);
}
// World headings remain integer raw cues in every observation generation.
// V6 expands energy (four hats) and health (three hats) before those cues.
inline Value world_observation_values(const std::vector<double> &observation,
                                      int version, bool enabled = true) {
    Value out = values(observation);
    std::size_t heading = version == 6 ? 7 : 2;
    if (enabled && observation.size() >= heading + 2) {
        for (std::size_t i = heading; i < heading + 2; ++i) {
            if (!std::isfinite(observation[i]) ||
                std::trunc(observation[i]) != observation[i] ||
                observation[i] < -1 || observation[i] > 1)
                throw std::runtime_error("Invalid canonical world heading");
            out.items()[i] = static_cast<int>(observation[i]);
        }
    }
    return out;
}

inline bool equivalent(const Value &a, const Value &b) {
    if (a.null() || b.null())
        return a.null() && b.null();
    if (a.is_array() || b.is_array()) {
        if (!a.is_array() || !b.is_array() || a.array_size() != b.array_size())
            return false;
        for (std::size_t i = 0; i < a.array_size(); ++i)
            if (!equivalent(a.array_value(i), b.array_value(i)))
                return false;
        return true;
    }
    if (auto x = std::get_if<Value::ObjectPtr>(&a.data)) {
        auto y = std::get_if<Value::ObjectPtr>(&b.data);
        if (!y || (*x)->size() != (*y)->size())
            return false;
        for (const auto &[key, value] : **x)
            if (!b.contains(key) || !equivalent(value, b.at(key)))
                return false;
        return true;
    }
    if (auto x = std::get_if<std::string>(&a.data)) {
        auto y = std::get_if<std::string>(&b.data);
        return y && *x == *y;
    }
    if (std::holds_alternative<Value::ObjectPtr>(b.data) ||
        std::holds_alternative<std::string>(b.data))
        return false;
    return a.number() == b.number();
}
inline Value add(const Value &a, const Value &b) {
    if (a.integer() && b.integer())
        return a.i64() + b.i64();
    return a.number() + b.number();
}
inline Value subtract(const Value &a, const Value &b) {
    if (a.integer() && b.integer())
        return a.i64() - b.i64();
    return a.number() - b.number();
}
inline Value multiply(const Value &a, const Value &b) {
    if (a.integer() && b.integer())
        return a.i64() * b.i64();
    return a.number() * b.number();
}
inline const Value &minimum(const Value &a, const Value &b) {
    return b.number() < a.number() ? b : a;
}
inline const Value &maximum(const Value &a, const Value &b) {
    return b.number() > a.number() ? b : a;
}
} // namespace wtfcore
