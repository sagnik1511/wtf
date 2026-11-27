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
        self.persistent_actions = settings.get("persistent_action_totals", False)
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
        self._action_cache = {}
        self.action_totals = (
            {name: [[0, 0, 0, 0] for _ in ACTIONS] for name in self.contracts}
            if self.persistent_actions
            else {}
        )
        self.staged_action_totals = {}

    def learn(self, species, key, labels):
        """Accumulate physical outcome counts in the configured publication table."""
        # Keep total/context updates and publication invalidation atomic per outcome.
        # pylint: disable=too-many-branches
        if self.update_schedule == "episode":
            # Copy only a species receiving its first new evidence this episode.
            if species not in self.staged_tables:
                self.staged_tables[species] = OrderedDict(
                    (key, list(row)) for key, row in self.tables[species].items()
                )
            table = self.staged_tables[species]
        else:
            table = self.tables[species]
        if self.persistent_actions:
            if self.update_schedule == "episode":
                if species not in self.staged_action_totals:
                    self.staged_action_totals[species] = [
                        row[:] for row in self.action_totals[species]
                    ]
                totals = self.staged_action_totals[species]
            else:
                totals = self.action_totals[species]
            row = totals[key[0]]
            # Preserve old evidence across context eviction; rescale only near the
            # exact-integer limit of portable double-precision interchange.
            if row[0] >= 2**52:
                row[:] = [value * 0.5 for value in row]
            row[0] += 1
            for i, value in enumerate(labels, 1):
                row[i] += int(value > 0)
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
            self._action_cache.pop(species, None)
            self.published_updates = self.updates

    def publish(self):
        """Publish completed-episode evidence before the next world observes."""
        for species, table in self.staged_tables.items():
            self.tables[species] = table
            self._action_cache.pop(species, None)
        self.staged_tables.clear()
        for species, totals in self.staged_action_totals.items():
            self.action_totals[species] = totals
        self.staged_action_totals.clear()
        self.published_updates = self.updates

    def predict(self, species, key):
        """Read confidence and outcome rates from published population knowledge."""
        row = self.tables[species].get(key)
        if row is None:
            return [0.0] * 4
        n = row[0]
        return [n / (n + 4), *(value / n for value in row[1:])]

    def predict_action(self, species, action):
        """Published species/action knowledge, independent of query animal state.

        Aggregate retained contexts once per species/publication revision; the
        derived cache adds no checkpoint fields and never sees staged outcomes.
        """
        # Reject bool as an action index, consistently with persisted enum fields.
        # pylint: disable=unidiomatic-typecheck
        if type(action) is not int or not 0 <= action < len(ACTIONS):
            raise ValueError("Invalid consequence action")
        return list(self.action_rates(species)[action])

    def action_rates(self, species):
        """Borrow immutable published rows until that species publishes again."""
        rows = self._action_cache.get(species)
        if rows is None:
            rows = (
                [row[:] for row in self.action_totals[species]]
                if self.persistent_actions
                else [[0, 0, 0, 0] for _ in ACTIONS]
            )
            if not self.persistent_actions:
                for key, values in self.tables[species].items():
                    row = rows[key[0]]
                    for i, value in enumerate(values):
                        row[i] += value
            # Immutable cached rates share float values across consumers/replay
            # observations; return a fresh list so callers cannot mutate the cache.
            rows = tuple(
                (
                    (
                        values[0] / (values[0] + 4),
                        *(value / values[0] for value in values[1:]),
                    )
                    if values[0]
                    else (0.0, 0.0, 0.0, 0.0)
                )
                for values in rows
            )
            self._action_cache[species] = rows
        return rows

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
        return self._with_action_totals(
            state, self.action_totals, self.staged_action_totals
        )

    def _with_action_totals(self, model, totals, staged):
        """Versioned wrapper keeps old bounded-table snapshots readable."""
        if not self.persistent_actions:
            return model
        return {
            "version": 3,
            "model": model,
            "action_totals": copy.deepcopy(totals),
            "staged_action_totals": copy.deepcopy(staged),
        }

    def export_state(self):
        # A completed-run warm start receives all finished windows, including
        # the final episode's evidence, without publishing into its old replay.
        """Build the aggregate population-knowledge warm-start representation."""
        model = self._state(
            {
                name: self.staged_tables.get(name, table)
                for name, table in self.tables.items()
            }
        )
        totals = {
            name: self.staged_action_totals.get(name, rows)
            for name, rows in self.action_totals.items()
        }
        return self._with_action_totals(model, totals, {})

    @classmethod
    def restore(cls, config, state):
        """Reconstruct the component from its persisted continuation state."""

        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # Published/staged and legacy validation stay together so their counts
        # cannot be accepted independently of their matching context tables.
        # pylint: disable=too-many-boolean-expressions,too-many-branches,too-many-locals
        # Exact types intentionally reject bool/subclass substitutions in persisted
        # numeric contracts.
        # pylint: disable=unidiomatic-typecheck

        result = cls(config)
        if isinstance(state, dict) and state.get("version") == 3:
            if (
                type(state["version"]) is not int
                or not isinstance(state.get("model"), dict)
                or state["model"].get("version") not in (1, 2)
            ):
                raise ValueError("Invalid persistent consequence model")
            if not result.persistent_actions or set(state) != {
                "version",
                "model",
                "action_totals",
                "staged_action_totals",
            }:
                raise ValueError(
                    "Persistent consequence totals require enabled matching configuration"
                )
            result = cls.restore(config, state["model"])
            totals = cls._restore_action_totals(
                state["action_totals"], set(result.tables)
            )
            staged = cls._restore_action_totals(
                state["staged_action_totals"], set(result.staged_tables)
            )
            for tables, counts in (
                (result.tables, totals),
                (result.staged_tables, staged),
            ):
                for name, rows in tables.items():
                    retained = [[0, 0, 0, 0] for _ in ACTIONS]
                    for key, values in rows.items():
                        for index, value in enumerate(values):
                            retained[key[0]][index] += value
                    if any(
                        total < part
                        for total_row, part_row in zip(counts[name], retained)
                        for total, part in zip(total_row, part_row)
                    ):
                        raise ValueError(
                            "Persistent totals cannot omit retained context evidence"
                        )
            result.action_totals, result.staged_action_totals = totals, staged
            return result
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
            if result.persistent_actions:
                result.staged_action_totals = {
                    name: restored_staged.action_totals[name]
                    for name in state["staged_tables"]
                }
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
                if result.persistent_actions:
                    total = result.action_totals[name][key[0]]
                    for index, value in enumerate(row):
                        total[index] += value
            while len(result.tables[name]) > result.capacity:
                result.tables[name].popitem(last=False)
        result.updates = state["updates"]
        result.published_updates = result.updates
        return result

    @staticmethod
    def _restore_action_totals(value, species):
        """Validate bounded fixed-width published/staged counts without aliases."""
        if not isinstance(value, dict) or set(value) != species:
            raise ValueError("Invalid consequence total species")
        result = {}
        for name, rows in value.items():
            if not isinstance(rows, list) or len(rows) != len(ACTIONS):
                raise ValueError("Invalid consequence total actions")
            for row in rows:
                if (
                    not isinstance(row, list)
                    or len(row) != 4
                    or any(
                        type(v) not in (int, float) or not math.isfinite(v) for v in row
                    )
                    or not 0 <= row[0] <= 2**53
                    or any(not 0 <= v <= row[0] for v in row[1:])
                ):
                    raise ValueError("Invalid persistent consequence counts")
            result[name] = [row[:] for row in rows]
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
