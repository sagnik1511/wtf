#pragma once
#include "random.hpp"
#include <memory>
namespace wtfcore {
// Type-erased private learners. Only metadata is common; model, RNG, optimizer,
// replay and temporal buffers remain owned by each concrete controller.
class Controller {
    Value config_;
    std::vector<std::string> actions_;
    std::string kind_;

  public:
    explicit Controller(const Value &state)
        : config_(state.at("config")), kind_(config_.text("type")) {
        for (const auto &a : state.at("actions").items())
            actions_.push_back(a.str());
    }
    virtual ~Controller() = default;
    const Value &config() const { return config_; }
    const std::vector<std::string> &actions() const { return actions_; }
    const std::string &kind() const { return kind_; }
    virtual std::string choose(const std::vector<double> &observation,
                               bool training = true,
                               bool random_baseline = false) = 0;
    virtual void learn(const std::vector<double> &observation,
                       const std::string &action, double reward,
                       const std::vector<double> *next, bool terminated,
                       bool truncated) = 0;
    virtual Value checkpoint() = 0;
    virtual Value model_state() = 0;
    virtual Value diagnostics(const std::vector<double> &observation,
                              bool activations = false) = 0;
    virtual std::int64_t updates() const = 0;
    virtual void use_world_inputs() = 0;
    virtual Value rng_state() const = 0;
    virtual Value trial_memory() const { return {}; }
    virtual std::string trial_action(const std::vector<double> &observation,
                                     Random &external_rng, Value &memory,
                                     const Value &context = Value{}) = 0;
    virtual void set_context(const Value &) {}
    virtual bool needs_group() const { return false; }
    virtual Random &group_rng() {
        throw std::runtime_error("Controller has no group RNG");
    }
    virtual std::vector<double> policy_logits(const std::vector<double> &) {
        throw std::runtime_error("Controller has no grouped actor");
    }
    virtual void update_group(const Value &) {
        throw std::runtime_error("Controller has no grouped update");
    }
};
std::unique_ptr<Controller> restore_controller(const Value &state);
} // namespace wtfcore
