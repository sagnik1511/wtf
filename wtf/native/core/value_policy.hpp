#pragma once
#include "random.hpp"
#include <memory>
namespace wtfcore {
// Independent off-policy replay/targets/optimizers. No Python owners or calls.
class ValuePolicy {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    explicit ValuePolicy(const Value &checkpoint);
    ~ValuePolicy();
    ValuePolicy(ValuePolicy &&) noexcept;
    ValuePolicy &operator=(ValuePolicy &&) noexcept;
    std::string choose(const std::vector<double> &observation,
                       bool training = true, bool random_baseline = false);
    void learn(const std::vector<double> &observation,
               const std::string &action, double reward,
               const std::vector<double> *next, bool terminated,
               bool truncated);
    Value checkpoint() const;
    Value model_state() const;
    Value diagnostics(const std::vector<double> &observation,
                      bool activations = false) const;
    std::int64_t updates() const;
    void use_world_inputs();
    Value rng_state() const;
    std::string trial_action(const std::vector<double> &observation,
                             Random &external_rng, Value &memory,
                             const Value &context = Value{}) const;
};
} // namespace wtfcore
