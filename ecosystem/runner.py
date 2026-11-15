"""Experiment lifecycle, training, evaluation and exact JSON checkpoints."""

import copy
import random
import secrets
from .brain import Experience
from .brains import create_brain, restore_brain, inherit_brain, offspring_seed, load_model_weights, model_snapshot
from .config import validate
from .world import World
from .library import ChampionBook, load_initial_models


class Experiment:
    def __init__(self, config, mode="train", library_root=None):
        if mode not in ("train", "evaluate", "random"):
            raise ValueError("mode must be train, evaluate, or random")
        self.config = copy.deepcopy(validate(config))
        self.mode = mode
        self.legacy_brain_retention = False
        self.initial_models = load_initial_models(self.config, **({"root": library_root} if library_root is not None else {}))
        self.run_initial_models = copy.deepcopy(self.initial_models)
        self.champion_artifacts = {}
        self.initial_weights = {}
        randomized = self.config["experiment"]["world_seed_policy"] == "random"
        self.randomization_seed = secrets.randbits(64) if randomized else None
        self.seed_rng = random.Random(self.randomization_seed) if randomized else None
        self.run_index = 0
        self.brains = {}
        self.retired_updates = 0
        self.inherited_updates = {}
        self.history = []
        self.summaries = []
        self.run_seeds = []
        self.last_record = None
        self._start_run()

    def _start_run(self):
        c = self.config
        if self.run_index > 0 and not self.legacy_brain_retention:
            models = copy.deepcopy(self.initial_models)
            if c["experiment"].get("chain_best_brains", True) and self.mode == "train":
                for species, spec in c["species"].items():
                    metric = self.initial_models.get(species, {}).get("provenance", {}).get(
                        "criterion", "lifespan" if spec["diet"] == "resource" else "kills")
                    candidate = self.champions.entries.get(species, {}).get(metric)
                    if candidate is None:
                        continue
                    provenance = self.champion_artifacts.get(species, {}).get(metric)
                    if provenance is None:
                        parent = self.run_initial_models.get(species, {}).get("provenance", {})
                        provenance = dict(artifact_id=f"run:{self.run_index}:{self.world.seed}:{species}:{metric}",
                            path=None, criterion=metric, score=candidate["score"],
                            model_generation=parent.get("model_generation", 0) + 1,
                            source_run=self.run_index)
                    models[species] = dict(model=copy.deepcopy(candidate["model"]),
                                           provenance=copy.deepcopy(provenance))
            self.run_initial_models = models
            self._retire_brains(list(self.brains))
        self.champion_artifacts = {}
        policy = c["experiment"]["world_seed_policy"]
        if policy == "random":
            used = set(self.run_seeds)
            seed = self.seed_rng.getrandbits(32)
            while seed in used:
                seed = self.seed_rng.getrandbits(32)
        else:
            seed = (
                c["seed"] + (self.run_index if policy == "increment" else 0)
            ) % 2**32
        # Retained brains keep their learning and individual RNG states.
        # Newly initialized random-mode brains use the actual run seed.
        brain_seed = seed if policy == "random" else c["seed"]
        self.run_seeds.append(seed)
        self.world = World(c, seed)
        self._retire_brains(set(self.brains) - set(self.world.animals))
        if not self.brains or not c["experiment"]["retain_brains_between_runs"]:
            self.brains = {}
            for species_index, (species, spec) in enumerate(c["species"].items()):
                brain_config = copy.deepcopy(c["brains"][spec["brain"]])
                multiplier = spec.get("learning_rate_multiplier", 1.0)
                if multiplier != 1.0:
                    brain_config["learning_rate"] *= multiplier
                base_seed = brain_seed + 1000003 * (species_index + 1)
                for i, a in self.world.animals.items():
                    if a.species == species:
                        self.brains[i] = create_brain(
                            brain_config,
                            spec["allowed_actions"],
                            base_seed,
                            brain_seed + i * 7919 + 13,
                        )
                        if species in self.run_initial_models:
                            load_model_weights(self.brains[i], self.run_initial_models[species]["model"])
                        elif not self.legacy_brain_retention and species in self.initial_weights:
                            load_model_weights(self.brains[i], self.initial_weights[species])
                        if self.run_index == 0 and species not in self.initial_weights:
                            self.initial_weights[species] = model_snapshot(self.brains[i])
        self.champions = ChampionBook(c)
        if self.mode == "train":
            self.champions.consider(self.world.animals.values(), self.brains, self._champion_source())
        self.observations = self.world.observe_all()
        self.finished_run = False
        self.history = [self.world.metrics()]
        self.last_record = None

    @property
    def done(self):
        return (
            self.finished_run
            and self.run_index + 1 >= self.config["experiment"]["runs"]
        )

    def step(self):
        if self.finished_run:
            if self.done:
                return self.snapshot()
            self.run_index += 1
            self._start_run()
        if self.mode == "train":
            grouped = [
                (i, self.brains[i])
                for i in self.world.animals
                if self.brains[i].config["type"] == "grpo"
            ]
            if grouped:
                from .grouped_rollouts import collect_group

                start = self.world.tick % len(grouped)
                budget = self.config["experiment"].get("grpo_updates_per_step", 1)
                for i, brain in grouped[start:] + grouped[:start]:
                    if brain.needs_group:
                        brain.update_group(collect_group(self.world, self.brains, i))
                        budget -= 1
                        if not budget:
                            break
        tracked_animals = tuple(self.world.animals.values()) if self.mode == "train" else ()
        actions = {
            i: self.brains[i].choose(
                o, explore=self.mode == "train", random_baseline=self.mode == "random"
            )
            for i, o in self.observations.items()
        }
        result = self.world.step(actions)
        limit = self.world.tick >= self.config["world"]["max_steps"]
        for i, observation in self.observations.items():
            if self.mode == "train":
                self.brains[i].learn(
                    Experience(
                        observation,
                        actions[i],
                        result.rewards[i],
                        result.observations.get(i),
                        i in result.dead_ids,
                        limit and i not in result.dead_ids,
                    )
                )
        if self.mode == "train":
            # Includes the final action and terminal learning of dead animals,
            # before deceased descendants' controllers are released.
            self.champions.consider(tracked_animals, self.brains, self._champion_source())
        # Parents receive the current step's feedback before model inheritance.
        for event in result.events:
            if event.kind == "born":
                self.brains[event.target] = inherit_brain(
                    self.brains[event.actor], offspring_seed(self.world, event.target)
                )
                self.inherited_updates[event.target] = self.brains[event.target].updates
        founders = sum(s["initial_population"] for s in self.config["species"].values())
        self._retire_brains(i for i in result.dead_ids if i >= founders)
        self.observations = result.observations
        self.last_record = None
        self.finished_run = limit or not self.world.animals
        if (
            self.world.tick % self.config["experiment"]["record_metrics_every_steps"]
            == 0
            or self.finished_run
        ):
            self.last_record = self.world.metrics()
            self.history.append(self.last_record)
            # Browser history is bounded; CLI streams all recorded rows to disk.
            self.history = self.history[-2000:]
        if self.finished_run:
            self.summaries.append(
                dict(
                    run=self.run_index + 1,
                    seed=self.world.seed,
                    initialized_from={s: m["provenance"] for s, m in self.run_initial_models.items()},
                    **self.world.metrics(),
                    lifespans=copy.deepcopy(self.world.lifespans),
                    survivors=[
                        dict(id=a.id, species=a.species, age=a.age)
                        for a in self.world.animals.values()
                    ],
                )
            )
        return None

    def _champion_source(self):
        return dict(experiment=self.config["name"], mode=self.mode,
                    run=self.run_index + 1, seed=self.world.seed, tick=self.world.tick)

    def _retire_brains(self, ids):
        # Keep only the bounded founding population for cross-run retention;
        # deceased descendants must not accumulate models/replay forever.
        for i in ids:
            self.retired_updates += self.brains.pop(
                i
            ).updates - self.inherited_updates.pop(i, 0)

    def next_run(self):
        if not self.finished_run or self.done:
            return False
        self.run_index += 1
        self._start_run()
        return True

    @property
    def total_updates(self):
        # Inherited optimizer counters do not represent new training work.
        return self.retired_updates + sum(
            b.updates - self.inherited_updates.get(i, 0) for i, b in self.brains.items()
        )

    def snapshot(self, selected=None):
        data = self.world.snapshot()
        data.update(
            run=self.run_index + 1,
            total_runs=self.config["experiment"]["runs"],
            done=self.done,
            finished_run=self.finished_run,
            mode=self.mode,
            history=self.history,
            summaries=self.summaries,
            name=self.config["name"],
            species={
                k: {"diet": v["diet"], "body": v["body"]}
                for k, v in self.config["species"].items()
            },
            total_updates=self.total_updates,
        )
        if selected in self.world.animals:
            brain = self.brains[selected]
            o = self.observations[selected]
            data["selected"] = dict(id=selected, **brain.diagnostics(o))
        return data

    def checkpoint(self, include_brains=True):
        state = dict(
            checkpoint_version=1,
            initial_models=copy.deepcopy(self.initial_models),
            initial_weights=copy.deepcopy(self.initial_weights),
            run_initial_models=copy.deepcopy(self.run_initial_models),
            champion_artifacts=copy.deepcopy(self.champion_artifacts),
            legacy_brain_retention=self.legacy_brain_retention,
            champions=copy.deepcopy(self.champions.entries),
            retired_updates=self.retired_updates,
            inherited_updates={str(i): n for i, n in self.inherited_updates.items()},
            randomization_seed=self.randomization_seed,
            seed_rng_state=self.seed_rng.getstate() if self.seed_rng else None,
            config=copy.deepcopy(self.config),
            mode=self.mode,
            run_index=self.run_index,
            world=self.world.checkpoint(),
            history=copy.deepcopy(self.history),
            summaries=copy.deepcopy(self.summaries),
            run_seeds=self.run_seeds[:],
            finished_run=self.finished_run,
        )
        if include_brains:
            state["brains"] = {str(i): b.checkpoint() for i, b in self.brains.items()}
        return state

    def write_checkpoint(self, path):
        """Atomic streamed checkpoint; .gz selects the compact lossless format."""
        from .checkpoint_io import write_checkpoint
        write_checkpoint(path, self.checkpoint(include_brains=False),
                         ((animal_id,brain.checkpoint()) for animal_id,brain in self.brains.items()))

    @classmethod
    def restore(cls, state, mode=None):
        if state.get("checkpoint_version") != 1:
            raise ValueError("Unsupported checkpoint version")
        e = cls.__new__(cls)
        e.config = copy.deepcopy(validate(state["config"]))
        e.initial_models = copy.deepcopy(state.get("initial_models", {}))
        e.legacy_brain_retention = state.get("legacy_brain_retention", "chain_best_brains" not in e.config["experiment"])
        e.initial_weights = copy.deepcopy(state.get("initial_weights", {}))
        e.run_initial_models = copy.deepcopy(state.get("run_initial_models", e.initial_models))
        e.champion_artifacts = copy.deepcopy(state.get("champion_artifacts", {}))
        e.champions = ChampionBook(e.config, state.get("champions"))
        e.retired_updates = state.get("retired_updates", 0)
        e.inherited_updates = {
            int(i): n for i, n in state.get("inherited_updates", {}).items()
        }
        e.mode = mode or state["mode"]
        if e.mode not in ("train", "evaluate", "random"):
            raise ValueError("Invalid checkpoint mode")
        for key in ("run_index", "history", "summaries", "run_seeds", "finished_run"):
            setattr(e, key, copy.deepcopy(state[key]))
        e.randomization_seed = state.get("randomization_seed")
        e.seed_rng = None
        if e.config["experiment"]["world_seed_policy"] == "random":
            if state.get("seed_rng_state") is None:
                raise ValueError(
                    "Randomized checkpoint is missing the seed generator state"
                )

            def tuples(value):
                return (
                    tuple(tuples(x) for x in value)
                    if isinstance(value, (list, tuple))
                    else value
                )

            e.seed_rng = random.Random(0)
            e.seed_rng.setstate(tuples(state["seed_rng_state"]))
        e.world = World.restore(state["world"])
        e.brains = {int(i): restore_brain(b) for i, b in state["brains"].items()}
        if "champions" not in state and e.mode == "train":
            e.champions.consider(e.world.animals.values(), e.brains, e._champion_source())
        e.observations = e.world.observe_all()
        e.last_record = None
        return e
