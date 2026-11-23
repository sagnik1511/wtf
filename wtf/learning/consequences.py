"""Bounded species-level physical-outcome estimates; no policy or reward sharing."""

from collections import OrderedDict
import copy
import json
import lzma
import math
from pathlib import Path
from wtf.simulation.actions import ACTIONS

MAX_FILE_BYTES = 16 * 1024 * 1024


class Consequences:
    """Bounded empirical outcome model per species, with deterministic LRU updates."""

    # This record/owner deliberately keeps its related fields and lifecycle operations
    # together without artificial wrappers.
    # pylint: disable=too-many-instance-attributes

    def __init__(self, config):
        settings = config.get("consequence_learning", {})
        self.update_schedule = settings.get("update_schedule", "continuous")
        self.capacity = settings.get("max_contexts", 512)
        self.horizon = settings.get("horizon_steps", 16)
        self.contracts = {
            name: {
                key: copy.deepcopy(spec.get(key))
                for key in (
                    "diet",
                    "body",
                    "locomotion",
                    "circadian",
                    "reproduction",
                    "allowed_actions",
                )
            }
            for name, spec in config["species"].items()
            if config["brains"][spec["brain"]].get("consequence_observation", False)
        }
        self.tables = {name: OrderedDict() for name in self.contracts}
        self.updates = 0
        self.published_updates = 0
        self.staged_tables = {}

    def learn(self, species, key, labels):
        """Accumulate physical outcome counts in the configured publication table."""
        if self.update_schedule == "episode":
            # Copy only a species receiving its first new evidence this episode.
            if species not in self.staged_tables:
                self.staged_tables[species] = OrderedDict(
                    (key, list(row)) for key, row in self.tables[species].items()
                )
            table = self.staged_tables[species]
        else:
            table = self.tables[species]
        values = table.pop(key, [0, 0, 0, 0])
        # Bound counters too; halving avoids overflow without keeping raw samples.
        if values[0] >= 65535:
            values = [v * 0.5 for v in values]
        values[0] += 1
        for i, value in enumerate(labels, 1):
            values[i] += int(value > 0)
        table[key] = values
        if len(table) > self.capacity:
            table.popitem(last=False)
        self.updates += 1
        if self.update_schedule == "continuous":
            self.published_updates = self.updates

    def publish(self):
        """Publish completed-episode evidence before the next world observes."""
        for species, table in self.staged_tables.items():
            self.tables[species] = table
        self.staged_tables.clear()
        self.published_updates = self.updates

    def predict(self, species, key):
        """Read confidence and outcome rates from published population knowledge."""
        row = self.tables[species].get(key)
        if row is None:
            return [0.0] * 4
        n = row[0]
        return [n / (n + 4), *(value / n for value in row[1:])]

    @staticmethod
    def _table_state(tables):
        return {
            name: [[list(key), list(row)] for key, row in table.items()]
            for name, table in tables.items()
        }

    def _state(self, tables):
        return {
            "version": 1,
            "horizon_steps": self.horizon,
            "contracts": copy.deepcopy(self.contracts),
            "updates": self.updates,
            "tables": self._table_state(tables),
        }

    def checkpoint(self):
        """Serialize state required to resume this component."""
        state = self._state(self.tables)
        if self.update_schedule == "episode":
            state.update(
                version=2,
                update_schedule="episode",
                published_updates=self.published_updates,
                staged_tables=self._table_state(self.staged_tables),
            )
        return state

    def export_state(self):
        # A completed-run warm start receives all finished windows, including
        # the final episode's evidence, without publishing into its old replay.
        """Build the aggregate population-knowledge warm-start representation."""
        return self._state(
            {
                name: self.staged_tables.get(name, table)
                for name, table in self.tables.items()
            }
        )

    @classmethod
    def restore(cls, config, state):
        """Reconstruct the component from its persisted continuation state."""

        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # pylint: disable=too-many-boolean-expressions
        # Exact types intentionally reject bool/subclass substitutions in persisted
        # numeric contracts.
        # pylint: disable=unidiomatic-typecheck

        result = cls(config)
        if isinstance(state, dict) and state.get("version") == 2:
            required = {
                "version",
                "horizon_steps",
                "contracts",
                "updates",
                "tables",
                "update_schedule",
                "published_updates",
                "staged_tables",
            }
            if (
                type(state["version"]) is not int
                or set(state) != required
                or result.update_schedule != "episode"
                or state["update_schedule"] != "episode"
                or type(state["published_updates"]) is not int
                or type(state["updates"]) is not int
                or not 0 <= state["published_updates"] <= state["updates"]
                or not isinstance(state["staged_tables"], dict)
                or set(state["staged_tables"]) - set(result.tables)
            ):
                raise ValueError("Invalid episode consequence checkpoint")
            base = {
                key: state[key]
                for key in (
                    "version",
                    "horizon_steps",
                    "contracts",
                    "updates",
                    "tables",
                )
            }
            base["version"] = 1
            result = cls.restore(config, base)
            staged = {
                **base,
                "tables": {
                    name: state["staged_tables"].get(name, []) for name in result.tables
                },
            }
            restored_staged = cls.restore(config, staged)
            result.staged_tables = {
                name: restored_staged.tables[name] for name in state["staged_tables"]
            }
            result.published_updates = state["published_updates"]
            return result
        if not isinstance(state, dict) or set(state) != {
            "version",
            "horizon_steps",
            "contracts",
            "updates",
            "tables",
        }:
            raise ValueError("Invalid consequence model")
        if (
            type(state["version"]) is not int
            or state["version"] != 1
            or type(state["horizon_steps"]) is not int
            or state["horizon_steps"] != result.horizon
            or state["contracts"] != result.contracts
        ):
            raise ValueError("Incompatible consequence model physical/horizon contract")
        if (
            type(state["updates"]) is not int
            or state["updates"] < 0
            or not isinstance(state["tables"], dict)
            or set(state["tables"]) != set(result.tables)
        ):
            raise ValueError("Invalid consequence model counters/species")
        for name, rows in state["tables"].items():
            if not isinstance(rows, list) or len(rows) > 8192:
                raise ValueError("Oversized consequence model")
            for key, row in rows:
                if (
                    len(key) != 12
                    or any(type(v) is not int for v in key)
                    or not 0 <= key[0] < len(ACTIONS)
                    or any(not -2 <= v <= 3 for v in key[1:])
                ):
                    raise ValueError("Invalid consequence context")
                if (
                    len(row) != 4
                    or any(
                        type(v) not in (int, float) or not math.isfinite(v) for v in row
                    )
                    or not 0 < row[0] <= 65536
                    or any(not 0 <= v <= row[0] for v in row[1:])
                ):
                    raise ValueError("Invalid consequence outcome counts")
                if tuple(key) in result.tables[name]:
                    raise ValueError("Duplicate consequence context")
                result.tables[name][tuple(key)] = list(row)
            while len(result.tables[name]) > result.capacity:
                result.tables[name].popitem(last=False)
        result.updates = state["updates"]
        result.published_updates = result.updates
        return result

    def save(self, path):
        """Persist the component using its supported storage representation."""
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        try:
            with lzma.open(temporary, "wt", encoding="utf-8", preset=6) as stream:
                json.dump(
                    self.export_state(), stream, separators=(",", ":"), allow_nan=False
                )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, config, path):
        """Read persisted data and validate its supported contract."""
        path = Path(path)
        if path.is_dir():
            path = path / "consequences.json.xz"
        with lzma.open(path, "rb") as stream:
            raw = stream.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise ValueError("Oversized consequence model file")
        return cls.restore(config, json.loads(raw))
