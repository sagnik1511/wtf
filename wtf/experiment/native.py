"""Python queries and episode setup for the standalone C++ simulation loop.

World stepping, sensing, every built-in learner, lifecycle and episode setup
run natively. Python validates configuration and reads/writes existing artifacts.
Query copies never become the native session's live owners.
"""

from array import array
import copy
import importlib
import os
import random
import secrets

from wtf.config import validate
from wtf.experiment.runner import Experiment
from wtf.learning.ppo import MLP, PPOBrain
from wtf.simulation.world import World
from wtf.learning.brains import model_signature
from wtf.native.numeric import precision_of


def _engine_module():
    """Load the optional extension only when native execution is requested."""
    try:
        engine = importlib.import_module("wtf.native._engine")
    except ImportError as error:
        raise RuntimeError(
            "The standalone C++ engine is unavailable. Build it with "
            "`.venv/bin/python -m work.build_native_engine` from the checkout, "
            "or install a source build with WTF_BUILD_ENGINE=1. "
            "Use a supported CPython interpreter."
        ) from error
    required = (
        "create",
        "bootstrap",
        "next_run",
        "run",
        "checkpoint",
        "observations",
        "snapshot",
    )
    if any(not hasattr(engine, name) for name in required):
        raise RuntimeError(
            "The installed standalone C++ engine is outdated; rebuild it with "
            "`.venv/bin/python -m work.build_native_engine`."
        )
    return engine


def _validate_mode(mode):
    """Reject unsupported execution modes before native initialization."""
    if mode not in ("train", "evaluate", "random"):
        raise ValueError("The C++ engine requires train, evaluate or random mode")


def _has_python_overrides(instance):
    """Detect instance replacements for built-in class methods before import."""
    return any(callable(getattr(type(instance), name, None)) for name in vars(instance))


def _require_native_owner(instance, expected, label):
    """Validate executable collaborators before importing their plain state."""
    # Exact built-in owners are required; native calls cannot dispatch subclasses.
    # pylint: disable=unidiomatic-typecheck
    if instance is not None and (
        type(instance) is not expected or _has_python_overrides(instance)
    ):
        raise ValueError(f"The C++ engine does not support Python {label} overrides")


def _validate_runtime_owners(experiment):
    """Check private world/bookkeeping/RNG owners once, outside the hot loop."""
    # The import boundary validates the collaborators whose state it transfers.
    # pylint: disable=protected-access
    # These optional execution imports belong to native selection, not root import.
    # pylint: disable=import-outside-toplevel
    from wtf.simulation import geography, social, tracking
    from wtf.learning import consequences
    from wtf.experiment import early_stopping, lineage
    from wtf.storage import library

    world = experiment.world
    for owner, expected, label in (
        (world.geography, geography.Geography, "geography"),
        (world._social, social.SocialState, "social"),
        (
            getattr(world._social, "knowledge", None),
            consequences.Consequences,
            "consequence",
        ),
        (world._tracking, tracking.Tracking, "tracking"),
        (
            getattr(world._tracking, "visual", None),
            tracking.VisualMemory,
            "visual-memory",
        ),
        (getattr(world._tracking, "scent", None), tracking.ScentField, "scent"),
        (experiment.champions, library.ChampionBook, "champion"),
        (experiment.champions.lineage, lineage.LineageBook, "lineage"),
        (experiment.early_stopping, early_stopping.EarlyStopping, "early-stopping"),
        (world.rng, random.Random, "random-generator"),
        (experiment.seed_rng, random.Random, "random-generator"),
    ):
        _require_native_owner(owner, expected, label)
    for owner in experiment.brains.values():
        for label in ("rng", "training_rng", "group_rng"):
            _require_native_owner(
                getattr(owner, label, None), random.Random, "random-generator"
            )


def _policy_head_networks(brain):
    """Validate optional executable heads and expose their standard networks."""
    # Head modules load only for native validation.
    # pylint: disable=import-outside-toplevel
    from wtf.learning.auxiliary_consequence import AuxiliaryConsequence
    from wtf.learning.late_consequence import LateConsequence

    networks = []
    for field, expected in (
        ("auxiliary", AuxiliaryConsequence),
        ("late", LateConsequence),
    ):
        head = getattr(brain, field, None)
        _require_native_owner(head, expected, "head")
        if head is not None:
            networks.append(head.network)
    return networks


def _validate_policy(brain):
    """Reject custom Python hooks instead of silently bypassing their behavior."""
    # Native imports are explicit and optional; ordinary Python runs stay lazy.
    # pylint: disable=import-outside-toplevel
    from wtf.learning.brain import Brain
    from wtf.learning.brains import advanced_brain
    from wtf.learning import policy_gradients
    from wtf.learning.contextual_ppo import LocalMLP, CentralNetwork
    from wtf.learning.policy_networks import PolicyBundle
    from wtf.learning.recurrent_ppo import GRUNetwork
    from wtf.learning.lstm_ppo import LSTMNetwork
    from wtf.learning.ppo import PhysicalMLP

    kind = brain.config["type"]
    expected = {
        "q_network": Brain,
        "ppo": PPOBrain,
        "a2c": policy_gradients.A2CBrain,
        "reinforce": policy_gradients.ReinforceBrain,
        "grpo": policy_gradients.GRPOBrain,
    }.get(kind)
    if expected is None:
        expected = advanced_brain(kind)
    # Exact classes are intentional: native calls cannot honor subclass overrides.
    # pylint: disable=unidiomatic-typecheck
    if type(brain) is not expected or _has_python_overrides(brain):
        raise ValueError("The C++ engine does not support Python policy overrides")
    pending = [
        getattr(brain, key, None)
        for key in ("network", "target_network", "reference_network")
    ]
    pending.extend(getattr(brain, "target_critics", ()))
    pending.extend(_policy_head_networks(brain))
    seen = set()
    while pending:
        network = pending.pop()
        if network is None or id(network) in seen:
            continue
        seen.add(id(network))
        if type(network) not in (
            MLP,
            PhysicalMLP,
            LocalMLP,
            CentralNetwork,
            PolicyBundle,
            GRUNetwork,
            LSTMNetwork,
        ) or _has_python_overrides(network):
            raise ValueError("The C++ engine does not support Python network overrides")
        pending.extend(
            getattr(network, key, None) for key in ("actor", "critic", "head")
        )
        pending.extend(getattr(network, "critics", ()))


def _validate_world_observation(observation, version=1):
    """Native world serialization requires the canonical integer heading pair."""
    offset = 7 if version == 6 else 2
    if len(observation) < offset + 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value not in (-1, 0, 1)
        for value in observation[offset : offset + 2]
    ):
        raise ValueError(
            "The C++ engine requires canonical integer world headings; "
            "use the Python engine for custom observation samples"
        )


def _validate_existing_observations(experiment):
    """Check live and buffered world inputs before native ownership begins."""
    for animal_id, observation in experiment.observations.items():
        _validate_world_observation(
            observation,
            experiment.brains[animal_id].config.get("observation_version", 1),
        )
    for brain in experiment.brains.values():
        version = brain.config.get("observation_version", 1)
        for field in ("rollout", "trajectory", "replay"):
            for sample in getattr(brain, field, ()):
                _validate_world_observation(sample.observation, version)
                following = getattr(sample, "next_observation", None)
                if following is not None:
                    _validate_world_observation(following, version)
        pending = getattr(brain, "pending", None)
        if pending is not None:
            _validate_world_observation(pending[0], version)


def _pack_import_models(state):
    """Pack only explicitly FP32 model arrays in this detached import payload.

    Parameters already underwent FP32 storage rounding in their owning brain.
    Observations, rewards, RNGs and FP64 legacy snapshots remain untouched.
    """
    packed = {}

    def parameters(value):
        cached = packed.get(id(value))
        if cached is not None:
            return cached[1]
        if isinstance(value, dict):
            # Keep shared parameter blocks shared across the boundary conversion.
            packed[id(value)] = (value, value)
            for key, item in value.items():
                value[key] = parameters(item)
        elif isinstance(value, list):
            if not value or not isinstance(value[0], (dict, list)):
                result = array("f", value)
                packed[id(value)] = (value, result)
                return result
            packed[id(value)] = (value, value)
            for index, item in enumerate(value):
                value[index] = parameters(item)
        return value

    # This explicit checkpoint-field allowlist also appears in lifecycle tests;
    # inferring parameter arrays could silently cast observations or metadata.
    # pylint: disable=duplicate-code
    fields = (
        "weights",
        "target",
        "network",
        "moments",
        "variances",
        "target_network",
        "reference_network",
        "target_critics",
    )
    for brain in state["brains"].values():
        if brain.get("precision") == "float32":
            for field in fields:
                if field in brain:
                    brain[field] = parameters(brain[field])
            for head in ("auxiliary", "late"):
                if head in brain:
                    for field in ("network", "moments", "variances"):
                        brain[head][field] = parameters(brain[head][field])

    visited = set()

    def models(value):
        if not isinstance(value, dict) or id(value) in visited:
            return
        visited.add(id(value))
        if value.get("precision") == "float32" and "parameters" in value:
            for field in ("parameters", "auxiliary_parameters", "late_parameters"):
                if field in value:
                    value[field] = parameters(value[field])
        for child in value.values():
            if isinstance(child, dict):
                models(child)

    for field in (
        "champions",
        "initial_models",
        "run_initial_models",
        "initial_weights",
    ):
        models(state.get(field, {}))
    return state


class NativeExperiment:
    """Own a native session while retaining detached, Pythonic query access.

    ``run`` makes one extension call and never restores Python world or brain
    objects. ``to_python``/``query`` and the ``world``/``brains`` properties create
    independent query copies; editing those copies does not edit this session.
    ``next_run`` performs champion chaining and world/controller initialization
    natively. Python restoration is reserved for explicit queries and legacy imports.
    """

    def __init__(
        self, config, mode="train", library_root=None, *, freeze_evaluation_brains=False
    ):
        # File lookup and config validation stay in the Python wrapper.
        # pylint: disable=import-outside-toplevel
        from wtf.storage.library import load_initial_models

        validated = copy.deepcopy(validate(config))
        _validate_mode(mode)
        if not isinstance(freeze_evaluation_brains, bool) or (
            freeze_evaluation_brains and mode != "evaluate"
        ):
            raise ValueError(
                "Frozen brains require evaluate mode and a boolean setting"
            )
        engine = _engine_module()
        metadata = {
            "config": validated,
            "mode": mode,
            "initial_models": load_initial_models(
                validated,
                **({"root": library_root} if library_root is not None else {}),
            ),
            "freeze_evaluation_brains": freeze_evaluation_brains,
        }
        if validated["experiment"]["world_seed_policy"] == "random":
            metadata["randomization_seed"] = secrets.randbits(64)
        self._engine = engine
        self._handle = engine.bootstrap(metadata)
        self._config = validated
        self._mode = mode
        self._run_index = 0
        self._report = engine.run(self._handle, 0, "", {})

    @classmethod
    def from_experiment(cls, experiment):
        """Copy an existing ordinary experiment into a native-owned session."""
        obj = cls.__new__(cls)
        obj._attach(experiment, _engine_module())
        return obj

    @classmethod
    def restore(cls, state, mode=None):
        """Restore a logical checkpoint through the ordinary compatibility path."""
        return cls.from_experiment(Experiment.restore(state, mode=mode))

    def _attach(self, experiment, engine):
        # This boundary validates transient controls that config cannot describe.
        # pylint: disable=protected-access
        # Native execution cannot preserve subclass step or checkpoint overrides.
        # pylint: disable=unidiomatic-typecheck
        if type(experiment) is not Experiment or _has_python_overrides(experiment):
            raise ValueError("The C++ engine requires an ordinary Experiment")
        _validate_mode(experiment.mode)
        # Compiled execution cannot dispatch user-installed Python hooks.
        # Mutating a detached query remains supported; custom execution uses
        # the Python engine instead of silently changing its behavior here.
        if type(experiment.world) is not World or _has_python_overrides(
            experiment.world
        ):
            raise ValueError("The C++ engine does not support Python world overrides")
        for brain in experiment.brains.values():
            _validate_policy(brain)
        _validate_runtime_owners(experiment)
        if experiment.brain_precision != "float32" or any(
            precision_of(brain) != "float32" for brain in experiment.brains.values()
        ):
            raise ValueError("The C++ engine requires FP32-stored brains")
        if experiment._cuda_training is not None:
            raise ValueError("The C++ engine does not support CUDA training")
        _validate_existing_observations(experiment)
        payload = {
            "checkpoint": _pack_import_models(experiment.checkpoint()),
            "observations": {
                str(animal_id): list(observation)
                for animal_id, observation in experiment.observations.items()
            },
            "signatures": {
                str(animal_id): model_signature(brain.config, brain.actions)
                for animal_id, brain in experiment.brains.items()
            },
        }
        handle = engine.create(payload)
        self._engine = engine
        self._handle = handle
        self._config = copy.deepcopy(experiment.config)
        self._run_index = experiment.run_index
        self._mode = experiment.mode
        self._report = {
            "tick": experiment.world.tick,
            "finished_run": experiment.finished_run,
            "total_updates": experiment.total_updates,
            "early_stopped": experiment.early_stopping.stopped,
        }

    def run(self, steps=None, capture_path=None, options=None):
        """Run up to ``steps`` native steps, stopping at the episode boundary.

        ``None`` runs the remaining episode. Capture and execution options are
        passed to C++ in the same call; reporting does not export full state.
        """
        if steps is not None and (
            isinstance(steps, bool) or not isinstance(steps, int) or steps < -1
        ):
            raise ValueError("steps must be a nonnegative integer, -1, or None")
        if options is not None and not isinstance(options, dict):
            raise TypeError("options must be a dictionary or None")
        path = "" if capture_path is None else os.fspath(capture_path)
        if not isinstance(path, str):
            raise TypeError("capture_path must be a text filesystem path")
        report = self._engine.run(
            self._handle,
            -1 if steps is None else steps,
            path,
            {} if options is None else copy.deepcopy(options),
        )
        self._report = dict(report)
        return copy.deepcopy(self._report)

    def checkpoint(self):
        """Export the native session as a complete ordinary logical checkpoint."""
        return self._engine.checkpoint(self._handle)

    def to_python(self):
        """Construct an independent ordinary Experiment for queries or handoff."""
        return Experiment.restore(self.checkpoint())

    def query(self):
        """Return the same detached query copy as ``to_python``."""
        return self.to_python()

    def snapshot(self, diagnostics=False):
        """Query native world state, optionally including policy diagnostics."""
        return self._engine.snapshot(self._handle, diagnostics)

    def next_run(self, *, champion_artifacts=None):
        """Initialize the next native episode, retaining file-export provenance.

        File writers may supply updated champion artifact metadata; native model,
        optimizer, physical and random state remain authoritative.
        """
        if not self.finished_run or self.done:
            return False
        handle = self._engine.next_run(self._handle, champion_artifacts)
        if handle is None:
            return False
        self._handle = handle
        self._run_index += 1
        self._report = self._engine.run(handle, 0, "", {})
        return True

    @property
    def config(self):
        """Return a detached copy of the configured experiment."""
        return copy.deepcopy(self._config)

    @property
    def mode(self):
        """Return the session's execution mode."""
        return self._mode

    @property
    def run_index(self):
        """Return the zero-based episode index without exporting state."""
        return self._run_index

    @property
    def finished_run(self):
        """Return whether the current native episode has finished."""
        return bool(self._report["finished_run"])

    @property
    def done(self):
        """Return whether all configured episodes have finished."""
        return self.finished_run and (
            self._run_index + 1 >= self._config["experiment"]["runs"]
            or self._report.get("early_stopped", False)
        )

    @property
    def total_updates(self):
        """Return cumulative learning work from the cheap native report."""
        return self._report["total_updates"]

    @property
    def observations(self):
        """Query detached observations using ordinary integer animal IDs."""
        return {
            int(animal_id): observation
            for animal_id, observation in self._engine.observations(
                self._handle
            ).items()
        }

    @property
    def world(self):
        """Restore and return a detached Python World query copy."""
        return self.to_python().world

    @property
    def brains(self):
        """Restore and return detached Python brain query copies."""
        return self.to_python().brains
