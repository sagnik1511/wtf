#include "grpo_trials.hpp"
#include "context.hpp"
#include "dense.hpp"
#include "inheritance.hpp"
namespace wtfcore {
namespace {
int sample(const std::vector<double> &probabilities, Random &rng) {
    double draw = rng.random(), cumulative = 0;
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        cumulative += probabilities[i];
        if (draw < cumulative)
            return static_cast<int>(i);
    }
    return static_cast<int>(probabilities.size()) - 1;
}
} // namespace
Value collect_group(World &world,
                    const std::map<int, std::unique_ptr<Controller>> &brains,
                    int focal_id) {
    auto found = brains.find(focal_id);
    if (found == brains.end())
        throw std::runtime_error("GRPO focal agent must have a controller");
    Controller &focal = *found->second;
    bool alive = false;
    for (const auto &a : world.animals())
        alive |= a.integer("id") == focal_id;
    if (!alive)
        throw std::runtime_error("GRPO focal agent must be alive");
    int horizon =
        std::min(focal.config().integer("rollout_horizon"),
                 world.config.at("world").integer("max_steps") - world.tick());
    if (horizon < 1)
        throw std::runtime_error("No simulation steps remain for a GRPO trial");
    Value initial_world = world.checkpoint();
    initial_world["lifespans"] = Value::array();
    std::map<int, Value> initial_rng;
    bool has_masks = false;
    for (const auto &a : world.animals()) {
        int id = a.integer("id");
        auto &brain = *brains.at(id);
        initial_rng[id] = brain.rng_state();
        has_masks |= brain.kind() == "maskable_ppo";
    }
    Value group = Value::array();
    for (int group_index = 0;
         group_index < focal.config().integer("group_size"); ++group_index) {
        World trial(initial_world, world.geography());
        std::map<int, Controller *> policies;
        std::map<int, std::unique_ptr<Controller>> newborns;
        std::map<int, Random> generators;
        std::map<int, Value> memories;
        for (const auto &[id, state] : initial_rng) {
            policies[id] = brains.at(id).get();
            generators[id].restore(state);
            memories[id] = policies.at(id)->trial_memory();
        }
        generators[focal_id].reseed(focal.group_rng().bits(64));
        auto observations = observe(trial);
        Value trajectory = Value::array();
        for (int step = 0; step < horizon; ++step) {
            const auto &observation = observations.at(focal_id);
            std::vector<double> probabilities, logs;
            categorical_distribution(focal.policy_logits(observation),
                                     probabilities, logs);
            int index = sample(probabilities, generators.at(focal_id));
            std::unique_ptr<DecisionContext> context;
            if (has_masks)
                context = std::make_unique<DecisionContext>(trial);
            std::map<int, std::string> actions;
            for (const auto &[id, obs] : observations) {
                auto &policy = *policies.at(id);
                if (id == focal_id) {
                    actions[id] = focal.actions().at(index);
                    continue;
                }
                Value extra;
                if (policy.kind() == "maskable_ppo") {
                    extra = Value::object();
                    Value mask = Value::array();
                    for (bool allowed : context->action_mask(trial.animal(id),
                                                             policy.actions()))
                        mask.push(allowed);
                    extra["mask"] = std::move(mask);
                }
                actions[id] = policy.trial_action(obs, generators.at(id),
                                                  memories[id], extra);
            }
            auto result = trial.advance(actions);
            // Python World.step returns next observations before caller-owned
            // newborn mutation updates metadata. Keep that phase boundary
            // exact.
            auto next_observations = observe(trial);
            for (const auto &event : result.events.items()) {
                if (event.text("kind") != "born")
                    continue;
                int id = event.integer("target"),
                    mother = event.integer("actor"),
                    father = event.integer("mate");
                auto seed =
                    offspring_seed_words(trial.state.at("seed").u64(), id);
                auto child_state = inherit_controller_state(
                    policies.at(mother)->checkpoint(),
                    policies.at(father)->checkpoint(), seed);
                mutate_birth_state(trial, child_state, event, seed);
                newborns[id] = restore_controller(child_state);
                newborns.at(id)->use_world_inputs();
                policies[id] = newborns.at(id).get();
                memories[id] = policies.at(id)->trial_memory();
                generators[id].reseed_words(seed);
            }
            bool died = false;
            for (int id : result.dead) {
                died |= id == focal_id;
                policies.erase(id);
                newborns.erase(id);
                generators.erase(id);
                memories.erase(id);
            }
            trajectory.push(
                object({{"observation",
                         world_observation_values(
                             observation,
                             focal.config().integer("observation_version", 1))},
                        {"action", index},
                        {"reward", result.rewards.at(focal_id)},
                        {"log_probability", logs.at(index)},
                        {"terminated", died},
                        {"truncated", !died && step + 1 == horizon}}));
            if (died)
                break;
            observations = std::move(next_observations);
        }
        group.push(std::move(trajectory));
    }
    return group;
}
} // namespace wtfcore
