#pragma once
#include "random.hpp"
#include <array>
namespace wtfcore {
// Private FP32 model/Adam arrays; all arithmetic and rollout storage are
// double.
class PPO {
    struct Layer {
        std::vector<float> weights, bias, mw, mb, vw, vb;
        std::size_t inputs = 0;
        bool bias_first = false, moment_bias_first = false,
             variance_bias_first = false;
    };
    struct Step {
        std::vector<double> observation;
        int action;
        double reward, logp, value, next_value;
        bool terminated, truncated;
    };
    Value config_, last_metrics_;
    bool world_inputs_ = false;
    Value observation_values(const std::vector<double> &observation) const;
    std::vector<std::string> actions_;
    std::vector<Layer> layers_;
    std::vector<Step> rollout_;
    struct Pending {
        std::vector<double> observation;
        int action = 0;
        double logp = 0, value = 0;
        bool valid = false;
        Value imported;
    } pending_;
    struct Settings {
        double discount, lambda, clip, entropy, value, target_kl, max_grad,
            rate;
        int rollout_steps, batch_size, epochs;
        bool normalize;
    } settings_{};
    struct Gradient {
        std::vector<double> w, b;
    };
    // Private scratch is bounded by this network and its largest rollout, is
    // released with the controller, and never enters snapshots. Construction,
    // restoration and newborn inheritance start with a cold forward cache.
    // Hidden reuse requires exact input bits and the unchanged update counter;
    // another input (including diagnostics) replaces it, and Adam invalidates
    // it. Pending choice data has separate storage and cannot be clobbered.
    std::vector<std::vector<double>> tape_;
    std::int64_t tape_updates_ = -1;
    bool tape_valid_ = false, tape_full_ = false;
    std::vector<Gradient> gradients_;
    std::vector<double> probabilities_, log_probabilities_, delta_, previous_;
    std::vector<double> advantages_, targets_;
    std::vector<std::size_t> indices_;
    Value pending_state() const;
    Random rng_, training_rng_;
    std::int64_t steps_ = 0, updates_ = 0, rollout_updates_ = 0;
    const std::vector<std::vector<double>> &
    forward(const std::vector<double> &observation, bool value_only = false);
    static void distribution(const std::vector<double> &outputs,
                             std::vector<double> &p, std::vector<double> &logs);
    void update();

  public:
    explicit PPO(const Value &checkpoint);
    std::string choose(const std::vector<double> &observation,
                       bool training = true, bool random_baseline = false);
    void learn(const std::vector<double> &observation,
               const std::string &action, double reward,
               const std::vector<double> *next, bool terminated,
               bool truncated);
    Value checkpoint();
    Value model_state() const;
    void use_world_inputs() { world_inputs_ = true; }
    std::int64_t updates() const { return updates_; }
    Value diagnostics(const std::vector<double> &observation,
                      bool activations = false);
    Value rng_state() const { return rng_.checkpoint(); }
    std::string trial_action(const std::vector<double> &observation,
                             Random &external_rng, Value &memory,
                             const Value &context = Value{});
    static Value inherit(const Value &mother, const Value &father,
                         std::uint64_t seed);
    static Value inherit(const Value &mother, const Value &father,
                         const std::vector<std::uint32_t> &seed_words);
};
} // namespace wtfcore
