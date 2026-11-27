#include "controller.hpp"
#include "actor_policy.hpp"
#include "augmented_ppo.hpp"
#include "ppo.hpp"
#include "value_policy.hpp"
namespace wtfcore {
namespace {
template <class Policy> class IndependentController : public Controller {
  protected:
    Policy policy_;

  public:
    explicit IndependentController(const Value &state)
        : Controller(state), policy_(state) {}
    std::string choose(const std::vector<double> &o, bool train,
                       bool random) override {
        return policy_.choose(o, train, random);
    }
    void learn(const std::vector<double> &o, const std::string &a,
               double reward, const std::vector<double> *n, bool term,
               bool trunc) override {
        policy_.learn(o, a, reward, n, term, trunc);
    }
    Value checkpoint() override { return policy_.checkpoint(); }
    Value model_state() override { return policy_.model_state(); }
    Value diagnostics(const std::vector<double> &o, bool activations) override {
        return policy_.diagnostics(o, activations);
    }
    std::int64_t updates() const override { return policy_.updates(); }
    void use_world_inputs() override { policy_.use_world_inputs(); }
    Value rng_state() const override { return policy_.rng_state(); }
    std::string trial_action(const std::vector<double> &o, Random &rng,
                             Value &memory, const Value &context) override {
        return policy_.trial_action(o, rng, memory, context);
    }
};
class ActorController : public IndependentController<ActorPolicy> {
  public:
    using IndependentController<ActorPolicy>::IndependentController;
    void set_context(const Value &v) override { policy_.set_context(v); }
    bool needs_group() const override { return policy_.needs_group(); }
    Random &group_rng() override { return policy_.group_rng(); }
    Value trial_memory() const override { return policy_.trial_memory(); }
    std::vector<double> policy_logits(const std::vector<double> &o) override {
        return policy_.policy_logits(o);
    }
    void update_group(const Value &v) override { policy_.update_group(v); }
};
class AugmentedController : public IndependentController<AugmentedPPO> {
  public:
    using IndependentController<AugmentedPPO>::IndependentController;
    void set_context(const Value &v) override { policy_.set_context(v); }
    Value trial_memory() const override { return policy_.trial_memory(); }
    std::vector<double> policy_logits(const std::vector<double> &o) override {
        return policy_.policy_logits(o);
    }
};
} // namespace
std::unique_ptr<Controller> restore_controller(const Value &state) {
    const auto &kind = state.at("config").at("type").str();
    if (kind == "maskable_ppo" || kind == "mappo" ||
        (kind == "ppo" &&
         (state.contains("auxiliary") || state.contains("late"))))
        return std::make_unique<AugmentedController>(state);
    if (kind == "ppo")
        return std::make_unique<IndependentController<PPO>>(state);
    if (kind == "q_network" || kind == "dqn" || kind == "ddqn" ||
        kind == "qrdqn" || kind == "discrete_sac")
        return std::make_unique<IndependentController<ValuePolicy>>(state);
    if (kind == "a2c" || kind == "reinforce" || kind == "grpo" ||
        kind == "trpo" || kind == "recurrent_ppo" || kind == "lstm_ppo")
        return std::make_unique<ActorController>(state);
    throw std::runtime_error("Native controller not implemented: " + kind);
}
} // namespace wtfcore
