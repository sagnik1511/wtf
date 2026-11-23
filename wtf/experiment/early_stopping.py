"""Bounded, checkpointable patience over selected species/sex champion ranks."""

import copy

from wtf.storage.library import SEXES, selection_criterion


class EarlyStopping:
    """Track champion improvement and patience across completed episodes."""

    # This record/owner deliberately keeps its related fields and lifecycle operations
    # together without artificial wrappers.
    # pylint: disable=too-many-instance-attributes

    def __init__(self, config, mode, initial_models=None, state=None):
        settings = config["experiment"].get("early_stopping", {})
        self.patience = settings.get("patience_runs", 0)
        self.active = self.patience > 0 and mode == "train"
        self.species = list(settings.get("species", config["species"]))
        initial_models = initial_models or {}
        self.criteria = {
            species: {
                sex: selection_criterion(
                    config,
                    species,
                    initial_models.get(species, {})
                    .get(sex, {})
                    .get("provenance", {})
                    .get("criterion"),
                    sex,
                )
                for sex in SEXES
            }
            for species in self.species
        }
        state = state or {}
        self.best = copy.deepcopy(state.get("best_ranks", {}))
        self.stagnant_runs = state.get("stagnant_runs", 0)
        self.last_run = state.get("last_run", 0)
        self.stopped = state.get("stopped", False)
        self.reason = copy.deepcopy(state.get("reason"))

    def _records(self, entries):
        for species, groups in self.criteria.items():
            for sex, criterion in groups.items():
                entry = entries.get(species, {}).get(sex, {}).get(criterion)
                if entry is not None:
                    yield species, sex, criterion, entry["rank"]

    def seed(self, entries):
        """Known warm-start records establish a baseline without consuming patience."""
        if self.active:
            for species, sex, criterion, rank in self._records(entries):
                self.best.setdefault(species, {}).setdefault(
                    sex, {"criterion": criterion, "rank": list(rank)}
                )

    def finish_run(self, run, entries):
        """Update improvement counters after a completed episode."""
        if not self.active or self.stopped or run <= self.last_run:
            return
        improved = False
        for species, sex, criterion, rank in self._records(entries):
            previous = self.best.get(species, {}).get(sex)
            if (
                previous is None
                or previous["criterion"] != criterion
                or tuple(rank) > tuple(previous["rank"])
            ):
                self.best.setdefault(species, {})[sex] = {
                    "criterion": criterion,
                    "rank": list(rank),
                }
                improved = True
        self.stagnant_runs = 0 if improved else self.stagnant_runs + 1
        self.last_run = run
        if self.stagnant_runs >= self.patience:
            self.stopped = True
            self.reason = {
                "cause": "no_champion_improvement",
                "run": run,
                "stagnant_runs": self.stagnant_runs,
                "patience_runs": self.patience,
                "species": self.species[:],
            }

    def checkpoint(self):
        """Serialize state required to resume this component."""
        if not self.patience and not self.stopped:
            return None
        return {
            "patience_runs": self.patience,
            "species": self.species[:],
            "criteria": copy.deepcopy(self.criteria),
            "best_ranks": copy.deepcopy(self.best),
            "stagnant_runs": self.stagnant_runs,
            "last_run": self.last_run,
            "stopped": self.stopped,
            "reason": copy.deepcopy(self.reason),
        }
