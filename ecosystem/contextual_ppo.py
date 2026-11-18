"""PPO variants with local action masks or a centralized training critic."""

import math
from .brain import INPUTS, observation_inputs
from .ppo import MLP, PPOBrain, loss_and_gradient, policy_distribution

CENTRAL_INPUTS = 16


def action_mask(world, animal, actions, occupied=None, populations=None):
    """Immediate conditions only, before simultaneous movement/combat resolution.

    Turning/rest remain possible. An approaching target can still move into an
    unmasked attack's range only on a later decision; this changes the task.
    """
    front = world.front(animal)
    if occupied is None:
        occupied = {a.position: a for a in world.animals.values()}
    body = world.body(animal)
    result = []
    for action in actions:
        valid = True
        if action == "move_forward":
            valid = not world.movement_block_reason(
                animal, front, "move_forward", occupied
            )
        elif action == "reproduce":
            valid = world.reproduction_available(animal, occupied, populations)
        elif action == "swim":
            valid = not world.movement_block_reason(animal, front, "swim", occupied)
        elif action == "eat":
            valid = (
                animal.energy < body["max_energy"]
                and body["bite_size"] > 0
                and body["energy_per_food_unit"] > 0
                and any(
                    p in world.resources and world.resources[p].quantity > 0
                    for p in (animal.position, front)
                )
            )
        elif action == "attack":
            target = occupied.get(front)
            valid = (
                target is not None
                and world.config["species"][target.species]["diet"] == "resource"
                and world.config["species"][animal.species]["diet"] == "prey"
                and body["attack_damage"] > 0
            )
        result.append(bool(valid))
    if not any(result):
        raise ValueError("Masked policy has no available action; allow rest or turning")
    return tuple(result)


def central_contexts(world):
    """Fixed-size team-relative summaries, computed once per species per step.

    Tick, food, global density, obstacle density; then allied/opponent population,
    energy, health and centroid x/y. Privileged context goes to critics only.
    """
    animals = list(world.animals.values())
    capacity = (
        world.config["resources"]["patch_count"]
        * world.config["resources"]["capacity_per_patch"]
    )
    prefix = (
        world.tick / world.config["world"]["max_steps"],
        (
            sum(r.quantity for r in world.resources.values()) / capacity
            if capacity
            else 0.0
        ),
        len(animals) / (world.width * world.height),
        len(world.obstacles) / (world.width * world.height),
    )

    def summary(group):
        if not group:
            return (0.0,) * 6
        n = len(group)
        return (
            n / (world.width * world.height),
            sum(a.energy / world.body(a)["max_energy"] for a in group) / n,
            sum(a.health / world.body(a)["max_health"] for a in group) / n,
            sum(a.x / max(1, world.width - 1) for a in group) / n,
            sum(a.y / max(1, world.height - 1) for a in group) / n,
            sum(a.age for a in group) / n / world.config["world"]["max_steps"],
        )

    return {
        name: prefix
        + summary([a for a in animals if a.species == name])
        + summary([a for a in animals if a.species != name])
        for name in world.config["species"]
    }


class LocalMLP(MLP):
    """Masks accompany stored samples but are never neural input features."""

    def forward(self, observation):
        return super().forward(observation[: self.sizes[0]])

    def _predict(self, observation, value_only=False, output_range=None):
        return super()._predict(observation[: self.sizes[0]], value_only, output_range)


class MaskablePPOBrain(PPOBrain):
    algorithm = "maskable_ppo"

    def __init__(self, config, actions, base_seed, individual_seed):
        super().__init__(config, actions, base_seed, individual_seed)
        self.mask = (True,) * len(actions)

    def _new_network(self, config, actions, seed):
        return LocalMLP(
            config["network"]["hidden_layers"],
            actions,
            seed,
            inputs=observation_inputs(config),
        )

    def _pack(self, observation):
        return tuple(observation[: self.network.sizes[0]]) + tuple(self.mask)

    def distribution(self, outputs, observation):
        mask = observation[self.network.sizes[0] :]
        if len(mask) != len(self.actions) or not any(mask):
            raise ValueError("Missing or empty action mask")
        return policy_distribution(
            [v if valid else -1e30 for v, valid in zip(outputs[:-1], mask)]
        )

    def choose(self, observation, explore=True, random_baseline=False):
        if random_baseline:
            return self.rng.choice(
                [a for a, valid in zip(self.actions, self.mask) if valid]
            )
        return super().choose(self._pack(observation), explore, False)

    def action_probabilities(self, observation):
        return super().action_probabilities(self._pack(observation))

    def learn(self, experience):
        return super().learn(
            experience._replace(
                observation=self._pack(experience.observation),
                next_observation=(
                    None
                    if experience.next_observation is None
                    else self._pack(experience.next_observation)
                ),
            )
        )

    def objective(self, outputs, step, advantage, target):
        mask = step.observation[self.network.sizes[0] :]
        masked = [v if valid else -1e30 for v, valid in zip(outputs[:-1], mask)] + [
            outputs[-1]
        ]
        loss, gradient, info = loss_and_gradient(
            masked, step.action, step.log_probability, advantage, target, self.config
        )
        for i, valid in enumerate(mask):
            if not valid:
                gradient[i] = 0.0
        return loss, gradient, info

    def diagnostics(self, observation):
        d = super().diagnostics(self._pack(observation))
        d.update(features=observation, action_mask=dict(zip(self.actions, self.mask)))
        return d

    def checkpoint(self):
        state = super().checkpoint()
        state["mask"] = list(self.mask)
        return state

    @classmethod
    def restore(cls, state):
        b = super().restore(state)
        b.mask = tuple(state["mask"])
        return b


class CentralNetwork:
    """Independent actor and critic; actor receives only its versioned local inputs."""

    def __init__(self, config, actions, seed):
        self.inputs = observation_inputs(config)
        self.actor = MLP(
            config["network"]["hidden_layers"],
            actions,
            seed,
            value_head=False,
            inputs=self.inputs,
        )
        self.critic = MLP(
            config["critic_hidden_layers"],
            1,
            seed ^ 0xC0171C,
            value_head=False,
            inputs=self.inputs + CENTRAL_INPUTS,
        )

    @property
    def layers(self):
        return self.actor.layers + self.critic.layers

    @layers.setter
    def layers(self, values):
        n = len(self.actor.layers)
        self.actor.layers, self.critic.layers = values[:n], values[n:]

    def zeros(self):
        return self.actor.zeros() + self.critic.zeros()

    clear_gradients = staticmethod(MLP.clear_gradients)

    def predict(self, observation):
        return self.actor.predict(observation[: self.inputs]) + [
            self.value(observation)
        ]

    def value(self, observation):
        return self.critic.predict(observation)[0]

    def forward(self, observation):
        a, at = self.actor.forward(observation[: self.inputs])
        v, vt = self.critic.forward(observation)
        return a + v, (at, vt)

    def accumulate_gradient(self, tape, gradient, gradients):
        n = len(self.actor.layers)
        self.actor.accumulate_gradient(tape[0], gradient[:-1], gradients[:n])
        self.critic.accumulate_gradient(tape[1], gradient[-1:], gradients[n:])

    def checkpoint(self):
        return self.actor.checkpoint() + self.critic.checkpoint()


class MAPPOBrain(PPOBrain):
    """CTDE PPO variant: individual actors/critics and mean species rewards."""

    algorithm = "mappo"

    def __init__(self, config, actions, base_seed, individual_seed):
        super().__init__(config, actions, base_seed, individual_seed)
        self.context = self.next_context = (0.0,) * CENTRAL_INPUTS

    def _new_network(self, config, actions, seed):
        return CentralNetwork(config, actions, seed)

    def choose(self, observation, explore=True, random_baseline=False):
        if not explore and not random_baseline:
            probabilities = self.action_probabilities(observation)
            return self.actions[
                max(range(len(probabilities)), key=probabilities.__getitem__)
            ]
        return super().choose(
            tuple(observation) + self.context, explore, random_baseline
        )

    def action_probabilities(self, observation):
        # No privileged features are needed to run the actor.
        return policy_distribution(self.network.actor.predict(observation))[0]

    def learn(self, experience):
        return super().learn(
            experience._replace(
                observation=tuple(experience.observation) + self.context,
                next_observation=(
                    None
                    if experience.next_observation is None
                    else tuple(experience.next_observation) + self.next_context
                ),
            )
        )

    def diagnostics(self, observation):
        d = super().diagnostics(tuple(observation) + self.context)
        d.update(
            features=observation,
            critic_context=list(self.context),
            reward_scope="species_mean",
        )
        return d

    def checkpoint(self):
        state = super().checkpoint()
        state.update(context=list(self.context), next_context=list(self.next_context))
        return state

    @classmethod
    def restore(cls, state):
        b = super().restore(state)
        b.context, b.next_context = tuple(state["context"]), tuple(
            state["next_context"]
        )
        return b
