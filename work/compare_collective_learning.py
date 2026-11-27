"""Matched training and frozen-policy checks, retaining summaries instead of replays.

Each arm/seed runs in an isolated process. A source snapshot supplies the legacy
baseline; all arms read the same captured environment, seeds and physical rules.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("legacy", "persistent", "auxiliary")


def digest(value):
    """Compact exact-state check without writing model artifacts."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def worker(args):
    """One bounded arm/seed with no playback, checkpoint or champion files."""
    # Controlled comparison needs the same physics and private saved champions.
    # pylint: disable=import-outside-toplevel,protected-access,too-many-locals
    sys.path.insert(0, str(args.source.resolve()))
    from wtf.config import validate
    from wtf.experiment.runner import Experiment
    from wtf.learning.brains import load_model_weights, model_snapshot
    from wtf.learning.consequences import Consequences

    config = json.loads(args.environment.read_text())
    config["seed"] = args.seed
    config["world"]["max_steps"] = args.steps
    config["experiment"].update(runs=args.episodes, world_seed_policy="increment")
    config["recording"]["enabled"] = False
    # The baseline is the original bounded-table/observation-only treatment.
    config["consequence_learning"].pop("persistent_action_totals", None)
    for brain in config["brains"].values():
        brain.pop("auxiliary_consequence", None)
    if args.arm != "legacy":
        config["consequence_learning"]["persistent_action_totals"] = True
    if args.arm == "auxiliary":
        for brain in config["brains"].values():
            brain["auxiliary_consequence"] = {"coefficient": args.coefficient}
    config = validate(config)
    experiment = Experiment(config)
    started = time.perf_counter()
    while not experiment.done:
        experiment.step()
    training_seconds = time.perf_counter() - started
    knowledge = experiment.world._social.knowledge
    evidence = {
        "updates": knowledge.updates,
        "contexts": {s: len(rows) for s, rows in knowledge.tables.items()},
    }
    evaluations = []
    for seed in args.evaluation_seeds:
        evaluation_config = copy.deepcopy(config)
        evaluation_config["seed"] = seed
        evaluation_config["experiment"].update(runs=1, world_seed_policy="fixed")
        evaluation = Experiment(
            evaluation_config, mode="evaluate", freeze_evaluation_brains=True
        )
        for animal_id, animal in evaluation.world.animals.items():
            metric = (
                config["species"][animal.species]
                .get("champion_selection", {})
                .get("criterion", "lifespan")
            )
            group = experiment.champions.entries.get(animal.species, {}).get(
                animal.sex, {}
            )
            if metric not in group:
                raise ValueError(
                    f"Missing same-sex {animal.species}/{animal.sex}/{metric} champion"
                )
            load_model_weights(evaluation.brains[animal_id], group[metric]["model"])
        frozen = [
            (brain, digest(model_snapshot(brain)))
            for brain in evaluation.brains.values()
        ]
        evaluation.world._social.knowledge = Consequences.restore(
            evaluation.config, knowledge.export_state()
        )
        knowledge_before = digest(evaluation.world._social.knowledge.checkpoint())
        evaluation.observations = evaluation.world.observe_all()
        while not evaluation.done:
            evaluation.step()
        assert evaluation.total_updates == 0
        assert all(
            not hasattr(brain, "auxiliary") or brain.auxiliary.updates == 0
            for brain, _ in frozen
        )
        assert all(
            digest(model_snapshot(brain)) == expected for brain, expected in frozen
        )
        assert (
            digest(evaluation.world._social.knowledge.checkpoint()) == knowledge_before
        )
        evaluations.append(compact_summary(evaluation.summaries[0]))
    return {
        "arm": args.arm,
        "coefficient": args.coefficient if args.arm == "auxiliary" else None,
        "environment_sha256": hashlib.sha256(args.environment.read_bytes()).hexdigest(),
        "seed": args.seed,
        "steps": args.steps,
        "episodes": args.episodes,
        "training_seconds": training_seconds,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "training_updates": experiment.total_updates,
        "evidence": evidence,
        "training": [compact_summary(row) for row in experiment.summaries],
        "evaluation": evaluations,
    }


def compact_summary(summary):
    """Retain ecological counters and exact physical outcome summaries only."""
    return {
        key: summary[key]
        for key in (
            "run",
            "seed",
            "tick",
            "populations",
            "food",
            "deaths",
            "energy_gained",
            "kills",
            "births",
        )
    }


def main():
    """Run arms sequentially, printing progress and saving only compact metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--environment",
        type=Path,
        default=ROOT / "environments/land_bridge_behaviors.json",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 200, 300])
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[900, 901])
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--coefficient", type=float, default=0.1)
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".cache/collective-comparison.json"
    )
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.steps < 1 or args.episodes < 2 or not 0 < args.coefficient <= 1:
        parser.error(
            "Use positive steps, at least two episodes and coefficient in (0,1]"
        )
    if args.arm:
        print(json.dumps(worker(args), allow_nan=False))
        return
    if args.baseline is None or not (args.baseline / "wtf").is_dir():
        parser.error("--baseline must contain the unchanged wtf package")
    results = []
    for seed in args.seeds:
        for arm in ARMS:
            source = args.baseline if arm == "legacy" else ROOT
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--arm",
                arm,
                "--seed",
                str(seed),
                "--source",
                str(source.resolve()),
                "--environment",
                str(args.environment.resolve()),
                "--steps",
                str(args.steps),
                "--episodes",
                str(args.episodes),
                "--coefficient",
                str(args.coefficient),
                "--evaluation-seeds",
                *(str(value) for value in args.evaluation_seeds),
            ]
            print(f"Running {arm} seed={seed} ...", flush=True)
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                text=True,
                capture_output=True,
                check=True,
            )
            row = json.loads(completed.stdout)
            results.append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(results, indent=2, allow_nan=False) + "\n"
            )
            print(
                f"  {row['training_seconds']:.2f}s training; held-out kills "
                f"{[sum(s['kills'].values()) for s in row['evaluation']]}; "
                f"births {[sum(s['births'].values()) for s in row['evaluation']]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
