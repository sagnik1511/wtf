#pragma once
#include "dense.hpp"
#include "observation_layout.hpp"
#include <memory>
namespace wtfcore {
class AugmentedPPO {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    explicit AugmentedPPO(const Value &checkpoint);
    ~AugmentedPPO();
    AugmentedPPO(AugmentedPPO &&) noexcept;
    AugmentedPPO &operator=(AugmentedPPO &&) noexcept;
    std::string choose(const std::vector<double> &observation,
                       bool training = true, bool random_baseline = false);
    void learn(const std::vector<double> &observation,
               const std::string &action, double reward,
               const std::vector<double> *next, bool terminated,
               bool truncated);
    Value checkpoint();
    Value model_state();
    std::int64_t updates() const;
    Value diagnostics(const std::vector<double> &observation,
                      bool activations = false);
    void use_world_inputs();
    void set_context(const Value &context);
    Value rng_state() const;
    Value trial_memory() const;
    std::vector<double> policy_logits(const std::vector<double> &observation);
    std::string trial_action(const std::vector<double> &observation,
                             Random &external_rng, Value &memory,
                             const Value &context = Value{});
};
} // namespace wtfcore
