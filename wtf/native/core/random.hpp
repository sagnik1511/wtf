#pragma once
#include "value.hpp"
#include <array>
#include <cmath>
#include <limits>
namespace wtfcore {
class Random {
    std::array<std::uint32_t, 624> mt{};
    int index = 624;
    Value gaussian;
    void initialize(std::uint32_t seed) {
        mt[0] = seed;
        for (int i = 1; i < 624; ++i)
            mt[i] = 1812433253U * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i;
        index = 624;
    }

  public:
    explicit Random(std::uint64_t seed = 0) { reseed(seed); }
    void reseed(std::uint64_t seed) {
        reseed_words({static_cast<std::uint32_t>(seed),
                      static_cast<std::uint32_t>(seed >> 32)});
    }
    void reseed_words(std::vector<std::uint32_t> key) {
        while (key.size() > 1 && !key.back())
            key.pop_back();
        if (key.empty())
            key.push_back(0);
        initialize(19650218U);
        int length = static_cast<int>(key.size()), i = 1, j = 0;
        for (int k = std::max(624, length); k; --k) {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525U)) +
                    key[j] + j;
            if (++i >= 624) {
                mt[0] = mt[623];
                i = 1;
            }
            if (++j >= length)
                j = 0;
        }
        for (int k = 623; k; --k) {
            mt[i] =
                (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941U)) - i;
            if (++i >= 624) {
                mt[0] = mt[623];
                i = 1;
            }
        }
        mt[0] = 0x80000000U;
        gaussian = Value{};
    }
    void restore(const Value &state) {
        const auto &s = state.items();
        if (s.size() != 3 || s[0].as_int() != 3 || s[1].items().size() != 625)
            throw std::runtime_error("Invalid native RNG state");
        for (int i = 0; i < 624; ++i)
            mt[i] = static_cast<std::uint32_t>(s[1].items()[i].i64());
        index = s[1].items()[624].as_int();
        if (index < 0 || index > 624)
            throw std::runtime_error("Invalid RNG index");
        gaussian = s[2];
    }
    Value checkpoint() const {
        Value state = Value::array();
        state.push(3);
        Value words = Value::array();
        for (auto x : mt)
            words.push(static_cast<std::int64_t>(x));
        words.push(index);
        state.push(std::move(words));
        state.push(gaussian);
        return state;
    }
    std::uint32_t word() {
        if (index >= 624) {
            for (int i = 0; i < 624; ++i) {
                auto y =
                    (mt[i] & 0x80000000U) | (mt[(i + 1) % 624] & 0x7fffffffU);
                mt[i] = mt[(i + 397) % 624] ^ (y >> 1) ^
                        ((y & 1) ? 0x9908b0dfU : 0);
            }
            index = 0;
        }
        auto y = mt[index++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= y >> 18;
        return y;
    }
    double random() {
        std::uint32_t a = word() >> 5, b = word() >> 6;
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
    }
    std::uint64_t bits(int k) {
        if (k < 0 || k > 64)
            throw std::runtime_error("Native RNG supports up to 64 bits");
        if (!k)
            return 0;
        if (k <= 32)
            return word() >> (32 - k);
        auto low = static_cast<std::uint64_t>(word());
        return low | (static_cast<std::uint64_t>(word() >> (64 - k)) << 32);
    }
    std::uint64_t below(std::uint64_t n) {
        if (!n)
            throw std::runtime_error("Empty random choice");
        int k = 0;
        for (auto t = n; t; t >>= 1)
            ++k;
        auto r = bits(k);
        while (r >= n)
            r = bits(k);
        return r;
    }
    int randint(int a, int b) {
        return a +
               static_cast<int>(below(static_cast<std::uint64_t>(b) - a + 1));
    }
    double uniform(double a, double b) { return a + (b - a) * random(); }
    template <class T> void shuffle(std::vector<T> &v) {
        for (std::size_t i = v.size(); i > 1; --i)
            std::swap(v[i - 1], v[below(i)]);
    }
    double gauss(double mean, double sigma) {
        double z;
        if (!gaussian.null()) {
            z = gaussian.number();
            gaussian = Value{};
        } else {
            double angle = random() * 6.283185307179586,
                   radius = std::sqrt(-2.0 * std::log(1.0 - random()));
            z = std::cos(angle) * radius;
            gaussian = std::sin(angle) * radius;
        }
        return mean + z * sigma;
    }
};
struct Sum {
    double value = 0, correction = 0;
    void add(double x) {
        double n = value + x;
        correction += std::fabs(value) >= std::fabs(x) ? (value - n) + x
                                                       : (x - n) + value;
        value = n;
    }
    double result() const {
        return correction && std::isfinite(correction) ? value + correction
                                                       : value;
    }
};
} // namespace wtfcore
