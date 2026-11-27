#pragma once
#include "random.hpp"
#include <numeric>
namespace wtfcore {
struct DenseLayer {
    std::vector<float> weights, bias;
    std::size_t inputs = 0;
    bool bias_first = false;
};
struct DenseGradient {
    std::vector<double> w, b;
    bool bias_first = false;
};
using DenseGradients = std::vector<DenseGradient>;
inline std::vector<float> dense_floats(const Value &v) {
    if (auto p = std::get_if<Value::FloatArrayPtr>(&v.data))
        return **p;
    std::vector<float> out;
    out.reserve(v.array_size());
    for (const auto &x : v.items())
        out.push_back(static_cast<float>(x.number()));
    return out;
}
// Shared exact affine/backprop primitives. Controllers own networks and all
// optimizer/temporal state; this class never shares mutable buffers with peers.
class DenseNetwork {
  public:
    using Tape = std::vector<std::vector<double>>;
    std::vector<DenseLayer> layers;
    DenseNetwork() = default;
    explicit DenseNetwork(const Value &state) {
        std::size_t previous = 0;
        for (const auto &record : state.items()) {
            DenseLayer layer{dense_floats(record.at("weights")),
                             dense_floats(record.at("bias")), 0};
            if (layer.bias.empty() || layer.weights.empty() ||
                layer.weights.size() % layer.bias.size())
                throw std::runtime_error("Invalid dense layer shape");
            layer.inputs = layer.weights.size() / layer.bias.size();
            layer.bias_first = record.fields().front().first == "bias";
            if (previous && previous != layer.inputs)
                throw std::runtime_error("Disconnected dense layers");
            previous = layer.bias.size();
            layers.push_back(std::move(layer));
        }
        if (layers.empty())
            throw std::runtime_error("Empty dense network");
    }
    Value checkpoint() const {
        Value out = Value::array();
        for (const auto &l : layers)
            out.push(l.bias_first
                         ? object({{"bias", Value::float_array(l.bias)},
                                   {"weights", Value::float_array(l.weights)}})
                         : object({{"weights", Value::float_array(l.weights)},
                                   {"bias", Value::float_array(l.bias)}}));
        return out;
    }
    void forward_into(
        const std::vector<double> &observation, Tape &tape,
        std::size_t start = 0,
        std::size_t stop = std::numeric_limits<std::size_t>::max()) const {
        if (layers.empty() || observation.size() != layers.front().inputs)
            throw std::runtime_error("Dense input width mismatch");
        if (stop == std::numeric_limits<std::size_t>::max())
            stop = layers.back().bias.size();
        if (start > stop || stop > layers.back().bias.size())
            throw std::runtime_error("Dense output range mismatch");
        tape.resize(layers.size() + 1);
        tape[0] = observation;
        for (std::size_t j = 0; j < layers.size(); ++j) {
            const auto &l = layers[j];
            const auto &x = tape[j];
            auto &out = tape[j + 1];
            bool last = j + 1 == layers.size();
            std::size_t first = last ? start : 0,
                        end = last ? stop : l.bias.size();
            out.resize(end - first);
            for (std::size_t k = first; k < end; ++k) {
                Sum sum;
                for (std::size_t i = 0; i < l.inputs; ++i)
                    sum.add(x[i] *
                            static_cast<double>(l.weights[k * l.inputs + i]));
                double value = static_cast<double>(l.bias[k]) + sum.result();
                out[k - first] = last ? value : std::tanh(value);
            }
        }
    }
    Tape forward(const std::vector<double> &observation) const {
        Tape tape;
        forward_into(observation, tape);
        return tape;
    }
    std::vector<double> predict(const std::vector<double> &observation) const {
        return forward(observation).back();
    }
    std::vector<double> predict_range(const std::vector<double> &observation,
                                      std::size_t start,
                                      std::size_t stop) const {
        Tape tape;
        forward_into(observation, tape, start, stop);
        return tape.back();
    }
    double value(const std::vector<double> &observation) const {
        auto last = layers.back().bias.size();
        return predict_range(observation, last - 1, last)[0];
    }
    DenseGradients zeros() const {
        DenseGradients out;
        for (const auto &l : layers)
            out.push_back({std::vector<double>(l.weights.size()),
                           std::vector<double>(l.bias.size()), l.bias_first});
        return out;
    }
    static void clear_gradients(DenseGradients &gradients) {
        for (auto &g : gradients) {
            std::fill(g.w.begin(), g.w.end(), 0.0);
            std::fill(g.b.begin(), g.b.end(), 0.0);
        }
    }
    std::vector<double>
    accumulate_gradient(const Tape &tape, std::vector<double> delta,
                        DenseGradients &gradients,
                        bool return_input_gradient = false) const {
        if (tape.size() != layers.size() + 1 ||
            gradients.size() != layers.size() ||
            delta.size() != layers.back().bias.size())
            throw std::runtime_error("Dense gradient shape mismatch");
        for (std::size_t j = layers.size(); j-- > 0;) {
            const auto &l = layers[j];
            const auto &x = tape[j];
            auto &g = gradients[j];
            std::vector<double> previous(
                (j || return_input_gradient) ? l.inputs : 0, 0.0);
            for (std::size_t k = 0; k < l.bias.size(); ++k) {
                double derivative = delta[k];
                g.b[k] += derivative;
                for (std::size_t h = 0; h < l.inputs; ++h) {
                    g.w[k * l.inputs + h] += derivative * x[h];
                    if (j || return_input_gradient)
                        previous[h] +=
                            derivative *
                            static_cast<double>(l.weights[k * l.inputs + h]);
                }
            }
            if (j) {
                delta.resize(l.inputs);
                for (std::size_t h = 0; h < l.inputs; ++h)
                    delta[h] = previous[h] * (1 - x[h] * x[h]);
            } else if (return_input_gradient)
                return previous;
        }
        return {};
    }
    std::vector<double> parameters() const {
        std::vector<double> out;
        for (const auto &l : layers) {
            const auto &first = l.bias_first ? l.bias : l.weights;
            const auto &second = l.bias_first ? l.weights : l.bias;
            for (float x : first)
                out.push_back(x);
            for (float x : second)
                out.push_back(x);
        }
        return out;
    }
    void set_parameters(const std::vector<double> &values) {
        std::size_t n = 0;
        for (const auto &l : layers)
            n += l.weights.size() + l.bias.size();
        if (values.size() != n)
            throw std::runtime_error("Dense parameter size mismatch");
        std::size_t i = 0;
        for (auto &l : layers) {
            auto &first = l.bias_first ? l.bias : l.weights;
            auto &second = l.bias_first ? l.weights : l.bias;
            for (auto &x : first)
                x = static_cast<float>(values[i++]);
            for (auto &x : second)
                x = static_cast<float>(values[i++]);
        }
    }
};
inline void categorical_distribution(const std::vector<double> &logits,
                                     std::vector<double> &probabilities,
                                     std::vector<double> &logs) {
    if (logits.empty())
        throw std::runtime_error("Empty categorical policy");
    probabilities.resize(logits.size());
    logs.resize(logits.size());
    double maximum = *std::max_element(logits.begin(), logits.end());
    Sum sum;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - maximum);
        sum.add(probabilities[i]);
    }
    double total = sum.result(), normalizer = std::log(total);
    for (std::size_t i = 0; i < logits.size(); ++i) {
        logs[i] = (logits[i] - maximum) - normalizer;
        probabilities[i] /= total;
    }
}
inline std::vector<std::size_t>
sample_indices(Random &rng, std::size_t population, std::size_t count) {
    if (count > population)
        throw std::runtime_error("Sample exceeds population");
    std::size_t setsize = 21;
    if (count > 5) {
        std::size_t size = 1;
        while (size < count * 3)
            size *= 4;
        setsize += size;
    }
    std::vector<std::size_t> out;
    out.reserve(count);
    if (population <= setsize) {
        std::vector<std::size_t> pool(population);
        std::iota(pool.begin(), pool.end(), 0);
        for (std::size_t i = 0; i < count; ++i) {
            auto j = rng.below(population - i);
            out.push_back(pool[j]);
            pool[j] = pool[population - i - 1];
        }
    } else {
        std::vector<std::size_t> selected;
        for (std::size_t i = 0; i < count; ++i) {
            auto j = rng.below(population);
            while (std::find(selected.begin(), selected.end(), j) !=
                   selected.end())
                j = rng.below(population);
            selected.push_back(j);
            out.push_back(j);
        }
    }
    return out;
}
} // namespace wtfcore
