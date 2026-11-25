"""Experiment lifecycle, training, evaluation and exact JSON checkpoints."""

# Black owns code layout; long literal payloads, URLs and regression names retain
# their exact text.
# pylint: disable=line-too-long


import copy
from contextlib import nullcontext
import random
import secrets
from wtf.native.numeric import precision_scope
from wtf.learning.brain import Experience
from wtf.learning.brains import (
    create_brain,
    inherit_brain,
    restore_brain,
    mix_parent_brains,
    offspring_seed,
    load_model_weights,
    model_snapshot,
)
from wtf.config import validate
from wtf.experiment.early_stopping import EarlyStopping
from wtf.simulation.world import World
from wtf.storage.library import (
    ChampionBook,
    initial_champions,
    load_initial_models,
    selection_criterion,
    sex_models,
    SEXES,
    CRITERIA,
)


class Experiment:
    """Own episode scheduling, independent brains, selection and exact recovery."""

    # This record/owner deliberately keeps its related fields and lifecycle operations
    # together without artificial wrappers.
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, config, mode="train", library_root=None, *, freeze_evaluation_brains=False
    ):

        # Exact types intentionally reject bool/subclass substitutions in persisted
        # numeric contracts.
        # pylint: disable=unidiomatic-typecheck

        if mode not in ("train", "evaluate", "random"):
            raise ValueError("mode must be train, evaluate, or random")
        if type(freeze_evaluation_brains) is not bool or (
            freeze_evaluation_brains and mode != "evaluate"
        ):
            raise ValueError(
                "Frozen brains require evaluate mode and a boolean setting"
            )
        self._cuda_training = None
        self.config = copy.deepcopy(validate(config))
        self.guard_lineage_chaining = self.config["experiment"].get(
            "guard_lineage_chaining", True
        )
        self.mode = mode
        self.freeze_evaluation_brains = freeze_evaluation_brains
        self.improvement_only = self.config["experiment"].get(
            "promote_only_improved", True
        )
        self.brain_precision = "float32"
        self.legacy_brain_retention = False
        self.initial_models = load_initial_models(
            self.config, **({"root": library_root} if library_root is not None else {})
        )
        self.early_stopping = EarlyStopping(self.config, mode, self.initial_models)
        self.run_initial_models = copy.deepcopy(self.initial_models)
        self.champion_artifacts = {}
        if self.improvement_only:
            for species, groups in self.initial_models.items():
                for sex, initial in groups.items():
                    if initial["provenance"].get("sex") != sex:
                        continue
                    provenance = copy.deepcopy(initial["provenance"])
                    provenance["parent_artifact"] = copy.deepcopy(
                        initial.get("champion_parent")
                    )
                    provenance.update(copy.deepcopy(initial.get("champion_origin", {})))
                    self.champion_artifacts.setdefault(species, {})[sex] = {
                        provenance["criterion"]: provenance
                    }
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

    # Backend controls are explicit execution options, independent of saved config.
    # pylint: disable-next=too-many-arguments
    def set_training_device(
        self,
        device="cpu",
        *,
        max_agents=64,
        workspace_mb=256,
        precision="fp64",
        profile=False,
        inference="auto",
    ):
        """Select a transient backend; checkpoints retain portable CPU-owned state."""
        # The optional CUDA loader is never imported on ordinary CPU runs.
        # pylint: disable=import-outside-toplevel,protected-access
        if device not in ("cpu", "cuda"):
            raise ValueError("Training device must be cpu or cuda")
        if device == "cpu":
            if self._cuda_training is not None:
                self._cuda_training.close()
            self._cuda_training = None
            return
        if self.mode != "train":
            raise ValueError("CUDA acceleration currently supports training mode only")
        from wtf.learning.cuda_training import CudaTraining

        if not any(CudaTraining.supports(brain) for brain in self.brains.values()):
            raise ValueError(
                "CUDA requires a supported DQN/DDQN or dense policy brain with FP32-stored weights"
            )
        accelerator = CudaTraining(
            max_agents=max_agents,
            workspace_mb=workspace_mb,
            precision=precision,
            profile=profile,
            inference=inference,
        )
        if self._cuda_training is not None:
            self._cuda_training.close()
        self._cuda_training = accelerator

    def _start_run(self):

        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-nested-blocks
        # pylint: disable=too-many-statements
        # This internal collaborator intentionally inspects the owning component's
        # private state.
        # pylint: disable=protected-access

        c = self.config
        best_entries = None
        if self.improvement_only:
            best_entries = (
                self.champions.entries
                if self.run_index > 0
                else initial_champions(c, self.initial_models)
            )
        if self.run_index > 0 and not self.legacy_brain_retention:
            models = copy.deepcopy(self.initial_models)
            if c["experiment"].get("chain_best_brains", True) and self.mode == "train":
                for species, spec in c["species"].items():
                    for sex in SEXES:
                        original = self.initial_models.get(species, {}).get(sex, {})
                        metric = selection_criterion(
                            c,
                            species,
                            original.get("provenance", {}).get("criterion"),
                            sex,
                        )
                        candidate = (
                            self.champions.entries.get(species, {})
                            .get(sex, {})
                            .get(metric)
                        )
                        if candidate is None:
                            continue
                        previous = self.run_initial_models.get(species, {}).get(sex)
                        if (
                            self.improvement_only
                            and previous
                            and previous["provenance"].get("criterion") == metric
                            and previous["provenance"].get("rank")
                            == list(candidate["rank"])
                        ):
                            models.setdefault(species, {})[sex] = copy.deepcopy(
                                previous
                            )
                            continue
                        if (
                            metric == "lineage"
                            and self.guard_lineage_chaining
                            and not candidate.get("lineage", {}).get("qualified", False)
                            and previous
                            and previous["provenance"].get("lineage_qualified", False)
                        ):
                            retained = copy.deepcopy(previous)
                            retained["provenance"].update(
                                retained_after_run=self.run_index,
                                retention_reason="unqualified_lineage_candidate",
                            )
                            models.setdefault(species, {})[sex] = retained
                            continue
                        provenance = (
                            self.champion_artifacts.get(species, {})
                            .get(sex, {})
                            .get(metric)
                        )
                        if (
                            self.improvement_only
                            and provenance is not None
                            and provenance.get("rank") != list(candidate["rank"])
                        ):
                            provenance = None
                        if provenance is None:
                            parent = (
                                self.run_initial_models.get(species, {})
                                .get(sex, {})
                                .get("provenance", {})
                            )
                            provenance = {
                                "artifact_id": f"run:{self.run_index}:{self.world.seed}:{species}:{sex}:{metric}",
                                "path": None,
                                "criterion": metric,
                                "score": candidate["score"],
                                "sex": sex,
                                "model_generation": parent.get("model_generation", 0)
                                + 1,
                                "source_run": candidate["source"].get(
                                    "run", self.run_index
                                ),
                                "rank": list(candidate["rank"]),
                            }
                        provenance = {
                            k: v
                            for k, v in provenance.items()
                            if k != "parent_artifact"
                        }
                        models.setdefault(species, {})[sex] = {
                            "model": copy.deepcopy(candidate["model"]),
                            "provenance": copy.deepcopy(provenance),
                        }
                        if metric == "lineage":
                            models[species][sex]["provenance"]["lineage_qualified"] = (
                                candidate.get("lineage", {}).get("qualified", False)
                            )
            self.run_initial_models = models
            self._retire_brains(list(self.brains))
        if not self.improvement_only:
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
        previous_social = getattr(getattr(self, "world", None), "_social", None)
        self.world = World(c, seed)
        if self.world._social is not None:
            self.world._social.training = self.mode == "train"
            if previous_social is not None and previous_social.knowledge is not None:
                if self.mode == "train":
                    previous_social.knowledge.publish()
                self.world._social.knowledge = previous_social.knowledge
        self._retire_brains(set(self.brains) - set(self.world.animals))
        if not self.brains or not c["experiment"]["retain_brains_between_runs"]:
            self.brains = {}
            for species_index, (species, spec) in enumerate(c["species"].items()):
                brain_config = copy.deepcopy(c["brains"][spec["brain"]])
                multiplier = spec.get("learning_rate_multiplier", 1.0)
                if multiplier != 1.0:
                    brain_config["learning_rate"] *= multiplier
                base_seed = brain_seed + 1000003 * (species_index + 1)
                # Templates live only during initialization, before any learning.
                # Every founder owns detached parameters, buffers and RNGs.
                templates = {}
                for i, a in self.world.animals.items():
                    if a.species == species:
                        individual_seed = brain_seed + i * 7919 + 13
                        if a.sex in templates:
                            self.brains[i] = inherit_brain(
                                templates[a.sex], individual_seed
                            )
                            continue
                        with precision_scope(self.brain_precision):
                            self.brains[i] = create_brain(
                                brain_config,
                                spec["allowed_actions"],
                                base_seed ^ (0x4D414C45 if a.sex == "male" else 0),
                                individual_seed,
                            )
                        if a.sex in self.run_initial_models.get(species, {}):
                            load_model_weights(
                                self.brains[i],
                                self.run_initial_models[species][a.sex]["model"],
                            )
                        elif (
                            not self.legacy_brain_retention
                            and a.sex in self.initial_weights.get(species, {})
                        ):
                            load_model_weights(
                                self.brains[i], self.initial_weights[species][a.sex]
                            )
                        if self.run_index == 0:
                            self.initial_weights.setdefault(species, {})
                            if a.sex not in self.initial_weights[species]:
                                self.initial_weights[species][a.sex] = model_snapshot(
                                    self.brains[i]
                                )
                        templates[a.sex] = self.brains[i]
        self.champions = ChampionBook(c, best_entries)
        if self.run_index == 0 and self.early_stopping.active:
            self.early_stopping.seed(
                best_entries
                if best_entries is not None
                else initial_champions(c, self.initial_models)
            )
        if self.mode == "train":
            self.champions.consider(
                self.world.animals.values(), self.brains, self._champion_source()
            )
        self.observations = self.world.observe_all()
        self._refresh_context(mating_context=self.world._mate_context)
        self.finished_run = False
        self.history = [self.world.metrics()]
        self.last_record = None

    def _refresh_context(self, contexts=None, mating_context=None):

        # Keep imports lazy so optional tools and training dependencies load only on
        # this path.
        # pylint: disable=import-outside-toplevel
        # This internal collaborator intentionally inspects the owning component's
        # private state.
        # pylint: disable=protected-access

        contextual = [
            i
            for i in self.world.animals
            if self.brains[i].config["type"] in ("mappo", "maskable_ppo")
        ]
        if not contextual:
            return
        from wtf.learning.contextual_ppo import action_mask, central_contexts

        if contexts is None:
            contexts = (
                central_contexts(self.world)
                if any(self.brains[i].config["type"] == "mappo" for i in contextual)
                else {}
            )
        has_masks = any(
            self.brains[i].config["type"] == "maskable_ppo" for i in contextual
        )
        # Observation production has just built these same physical indexes.
        # Standalone refreshes rebuild them, so direct same-tick edits stay visible.
        if has_masks and mating_context is None:
            mating_context = self.world._prepare_mating_context()
        occupied, populations = mating_context[:2] if has_masks else (None, None)
        for i in contextual:
            brain, animal = self.brains[i], self.world.animals[i]
            if brain.config["type"] == "mappo":
                brain.context = brain.next_context = contexts[animal.species]
            else:
                brain.mask = action_mask(
                    self.world,
                    animal,
                    brain.actions,
                    occupied,
                    populations,
                    mating_context,
                )

    @property
    def done(self):
        """Return whether the configured experiment has finished."""
        return self.finished_run and (
            self.run_index + 1 >= self.config["experiment"]["runs"]
            or self.early_stopping.stopped
        )

    def step(self):
        """Choose actions, advance the world and update independent learners."""

        # Keyword dict construction deliberately detects duplicate keys in dynamic **
        # expansions; a literal would silently overwrite them.
        # pylint: disable=use-dict-literal

        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements
        # Keep imports lazy so optional tools and training dependencies load only on
        # this path.
        # pylint: disable=import-outside-toplevel
        # This internal collaborator intentionally inspects the owning component's
        # private state.
        # pylint: disable=protected-access

        if self.finished_run:
            if self.done:
                return self.snapshot()
            self.run_index += 1
            self._start_run()
        training = self.mode == "train"
        if training and any(
            b["type"] == "grpo" for b in self.config["brains"].values()
        ):
            grouped = [
                (i, self.brains[i])
                for i in self.world.animals
                if self.brains[i].config["type"] == "grpo"
            ]
            if grouped:
                from wtf.learning.grouped_rollouts import collect_group

                start = self.world.tick % len(grouped)
                budget = self.config["experiment"].get("grpo_updates_per_step", 1)
                for i, brain in grouped[start:] + grouped[:start]:
                    if brain.needs_group:
                        brain.update_group(collect_group(self.world, self.brains, i))
                        budget -= 1
                        if not budget:
                            break
        tracked_animals = tuple(self.world.animals.values()) if training else ()
        from wtf.learning.batched import choose_actions

        actions = choose_actions(
            self.brains,
            self.observations,
            explore=training,
            random_baseline=self.mode == "random",
            accelerator=self._cuda_training,
            allow_batch=self._cuda_training is not None
            or any(
                self.config["brains"][spec["brain"]]["type"]
                in ("ppo", "a2c", "reinforce", "grpo", "maskable_ppo", "mappo")
                for spec in self.config["species"].values()
            ),
        )
        result = self.world.step(actions)
        limit = self.world.tick >= self.config["world"]["max_steps"]
        contexts = None
        if training:
            team_rewards = {}
            if any(self.brains[i].config["type"] == "mappo" for i in self.observations):
                from wtf.learning.contextual_ppo import central_contexts

                contexts = central_contexts(self.world)
                teams = {}
                for animal in tracked_animals:
                    teams.setdefault(animal.species, []).append(
                        result.rewards[animal.id]
                    )
                team_rewards = {
                    s: sum(values) / len(values) for s, values in teams.items()
                }
                for animal in tracked_animals:
                    if self.brains[animal.id].config["type"] == "mappo":
                        self.brains[animal.id].next_context = contexts[animal.species]
            prior_species = (
                {a.id: a.species for a in tracked_animals} if team_rewards else {}
            )
            accelerator = self._cuda_training
            with accelerator.batch() if accelerator is not None else nullcontext():
                for i, observation in self.observations.items():
                    experience = Experience(
                        observation,
                        actions[i],
                        (
                            team_rewards[prior_species[i]]
                            if self.brains[i].config["type"] == "mappo"
                            else result.rewards[i]
                        ),
                        result.observations.get(i),
                        i in result.dead_ids,
                        limit and i not in result.dead_ids,
                    )
                    if accelerator is not None and accelerator.accepts(self.brains[i]):
                        self.brains[i].learn(experience, defer=accelerator)
                    else:
                        self.brains[i].learn(experience)
        if training:
            # Includes the final action and terminal learning of dead animals,
            # before deceased descendants' controllers are released.
            self.champions.consider(
                tracked_animals,
                self.brains,
                self._champion_source(),
                newborns=(
                    self.world.animals[event.target]
                    for event in result.events
                    if event.kind == "born"
                ),
            )
        # Parents receive the current step's feedback before model inheritance.
        mutation_events = []
        for event in result.events:
            if event.kind == "born":
                if self.mode == "evaluate" and self.freeze_evaluation_brains:
                    # Birth remains physical; the child's model is an exact copy
                    # of its same-sex parent, with independent transient state.
                    parent = (
                        event.actor
                        if self.world.animals[event.target].sex == "female"
                        else event.mate
                    )
                    self.brains[event.target] = inherit_brain(
                        self.brains[parent], offspring_seed(self.world, event.target)
                    )
                else:
                    self.brains[event.target] = mix_parent_brains(
                        self.brains[event.actor],
                        self.brains[event.mate],
                        offspring_seed(self.world, event.target),
                    )
                if training:
                    from wtf.learning.mutation import mutate_birth

                    self.brains[event.target], mutation_event = mutate_birth(
                        self.world,
                        event,
                        self.brains[event.actor],
                        self.brains[event.target],
                    )
                    if mutation_event is not None:
                        mutation_events.append(mutation_event)
                self.inherited_updates[event.target] = self.brains[event.target].updates
        result.events.extend(mutation_events)
        self._release_dead_brains(result.dead_ids)
        self.observations = result.observations
        self._refresh_context(contexts, self.world._mate_context)
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
                    initialized_from={
                        s: {sex: m["provenance"] for sex, m in groups.items()}
                        for s, groups in self.run_initial_models.items()
                    },
                    **copy.deepcopy(self.last_record),
                    lifespans=copy.deepcopy(self.world.lifespans),
                    survivors=[
                        {"id": a.id, "species": a.species, "age": a.age}
                        for a in self.world.animals.values()
                    ],
                )
            )
            self.early_stopping.finish_run(self.run_index + 1, self.champions.entries)
            if self.early_stopping.patience:
                self.summaries[-1]["early_stopping"] = self.early_stopping.checkpoint()
        return None

    def _champion_source(self):
        return {
            "experiment": self.config["name"],
            "mode": self.mode,
            "run": self.run_index + 1,
            "seed": self.world.seed,
            "tick": self.world.tick,
        }

    def _release_dead_brains(self, ids):
        if self.legacy_brain_retention:
            # Old checkpoints may reuse each founder's exact controller next run.
            founders = sum(
                s["initial_population"] for s in self.config["species"].values()
            )
            self._retire_brains(i for i in ids if i >= founders)
        else:
            # Modern chaining retains detached champions, not dead controllers.
            self._retire_brains(ids)

    def _retire_brains(self, ids):
        # Preserve cumulative work while dropping the controller, optimizer,
        # experience buffers and random generators for each retired animal.
        for i in ids:
            self.retired_updates += self.brains.pop(
                i
            ).updates - self.inherited_updates.pop(i, 0)

    def next_run(self):
        """Start the next scheduled world after finalizing the current episode."""
        if not self.finished_run or self.done:
            return False
        self.run_index += 1
        self._start_run()
        return True

    @property
    def total_updates(self):
        # Inherited optimizer counters do not represent new training work.
        """Count accumulated learning updates across current and retired animals."""
        return self.retired_updates + sum(
            b.updates - self.inherited_updates.get(i, 0) for i, b in self.brains.items()
        )

    def snapshot(self, selected=None):
        """Build a serializable view of the current simulation state."""

        # Keyword dict construction deliberately detects duplicate keys in dynamic **
        # expansions; a literal would silently overwrite them.
        # pylint: disable=use-dict-literal

        data = self.world.snapshot()
        data.update(
            run=self.run_index + 1,
            total_runs=self.config["experiment"]["runs"],
            early_stopping=self.early_stopping.checkpoint(),
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

    def _checkpoint_metadata(self, *, detached):
        # The file writer consumes this borrowed view synchronously, before any
        # further simulation step. Public snapshots retain their detached contract.
        clone = copy.deepcopy if detached else lambda value: value
        state = {
            "checkpoint_version": 1,
            "brain_precision": self.brain_precision,
            "initial_models": clone(self.initial_models),
            "initial_weights": clone(self.initial_weights),
            "run_initial_models": clone(self.run_initial_models),
            "champion_artifacts": clone(self.champion_artifacts),
            "legacy_brain_retention": self.legacy_brain_retention,
            "champions": clone(self.champions.entries),
            "lineage": self.champions.lineage.checkpoint(),
            "guard_lineage_chaining": self.guard_lineage_chaining,
            "improvement_only": self.improvement_only,
            "freeze_evaluation_brains": self.freeze_evaluation_brains,
            "early_stopping": self.early_stopping.checkpoint(),
            "retired_updates": self.retired_updates,
            "inherited_updates": {str(i): n for i, n in self.inherited_updates.items()},
            "randomization_seed": self.randomization_seed,
            "seed_rng_state": self.seed_rng.getstate() if self.seed_rng else None,
            "config": clone(self.config),
            "mode": self.mode,
            "run_index": self.run_index,
            "world": self.world.checkpoint(),
            "history": clone(self.history),
            "summaries": clone(self.summaries),
            "run_seeds": self.run_seeds[:],
            "finished_run": self.finished_run,
        }
        return state

    def checkpoint(self, include_brains=True):
        """Serialize state required to resume this component."""
        state = self._checkpoint_metadata(detached=True)
        if include_brains:
            state["brains"] = {str(i): b.checkpoint() for i, b in self.brains.items()}
        return state

    def write_checkpoint(self, path, *, shared_metadata=False):
        """Atomic streamed checkpoint; .gz selects the compact lossless format."""

        # Keep imports lazy so optional tools and training dependencies load only on
        # this path.
        # pylint: disable=import-outside-toplevel

        from wtf.storage.checkpoint_io import write_checkpoint

        # Preserve subclass/instance checkpoint hooks; only the stock snapshot
        # builder supports the private synchronous borrowed-metadata contract.
        metadata = (
            self._checkpoint_metadata(detached=False)
            if getattr(self.checkpoint, "__func__", None) is Experiment.checkpoint
            else self.checkpoint(include_brains=False)
        )
        write_checkpoint(
            path,
            metadata,
            (
                (animal_id, brain.checkpoint())
                for animal_id, brain in self.brains.items()
            ),
            shared_metadata=shared_metadata,
        )

    @classmethod
    def restore(cls, state, mode=None):
        """Reconstruct the component from its persisted continuation state."""

        # This internal collaborator intentionally inspects the owning component's
        # private state.
        # pylint: disable=protected-access
        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements

        if state.get("checkpoint_version") != 1:
            raise ValueError("Unsupported checkpoint version")
        e = cls.__new__(cls)
        e._cuda_training = None
        e.brain_precision = state.get("brain_precision", "float64")
        with precision_scope(e.brain_precision):
            pass  # Validate before constructing any controllers.
        e.config = copy.deepcopy(validate(state["config"]))
        # Old checkpoints retain their original unguarded promotion behavior.
        e.guard_lineage_chaining = state.get(
            "guard_lineage_chaining",
            e.config["experiment"].get("guard_lineage_chaining", False),
        )
        e.improvement_only = state.get(
            "improvement_only",
            e.config["experiment"].get("promote_only_improved", False),
        )
        e.initial_models = sex_models(state.get("initial_models", {}))
        e.legacy_brain_retention = state.get(
            "legacy_brain_retention", "chain_best_brains" not in e.config["experiment"]
        )
        e.initial_weights = copy.deepcopy(state.get("initial_weights", {}))
        for species, weights in e.initial_weights.items():
            if "signature" in weights:
                e.initial_weights[species] = {
                    sex: copy.deepcopy(weights) for sex in ("female", "male")
                }
        e.run_initial_models = sex_models(
            state.get("run_initial_models", e.initial_models)
        )
        e.champion_artifacts = copy.deepcopy(state.get("champion_artifacts", {}))
        for species, groups in list(e.champion_artifacts.items()):
            if any(metric in groups for metric in CRITERIA):
                e.champion_artifacts[species] = {}
        e.champions = ChampionBook(
            e.config,
            state.get("champions"),
            state.get("lineage", {"partial": True}),
        )
        e.retired_updates = state.get("retired_updates", 0)
        e.inherited_updates = {
            int(i): n for i, n in state.get("inherited_updates", {}).items()
        }
        e.freeze_evaluation_brains = state.get("freeze_evaluation_brains", False)
        e.mode = mode or state["mode"]
        if e.mode not in ("train", "evaluate", "random"):
            raise ValueError("Invalid checkpoint mode")
        for key in ("run_index", "history", "summaries", "run_seeds", "finished_run"):
            setattr(e, key, copy.deepcopy(state[key]))
        e.early_stopping = EarlyStopping(
            e.config, e.mode, e.initial_models, state.get("early_stopping")
        )
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
        if e.world._social is not None:
            if e.world._social.training != (e.mode == "train"):
                e.world._social.pending.clear()
            e.world._social.training = e.mode == "train"
        e.brains = {}
        for raw_id, brain_state in state["brains"].items():
            animal_id = int(raw_id)
            if not e.legacy_brain_retention and animal_id not in e.world.animals:
                # Account for old saved dead controllers without rebuilding
                # their networks, optimizer state and experience buffers.
                e.retired_updates += brain_state["updates"] - e.inherited_updates.pop(
                    animal_id, 0
                )
            else:
                e.brains[animal_id] = restore_brain(brain_state)
        legacy_champions = any(
            any(k in groups for k in CRITERIA)
            for groups in state.get("champions", {}).values()
        )
        if ("champions" not in state or legacy_champions) and e.mode == "train":
            e.champions.consider(
                e.world.animals.values(), e.brains, e._champion_source()
            )
        e.observations = e.world.observe_all()
        e._refresh_context(mating_context=e.world._mate_context)
        e.last_record = None
        return e
