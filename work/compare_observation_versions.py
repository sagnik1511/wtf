"""Compare ordinary PPO observation versions on matched worlds with compact reports."""

import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from unittest.mock import patch
from tqdm import tqdm
from work.compare_collective_learning import compact_summary, digest


def run_episode(experiment, *, progress=False):
    """Advance experiments, retaining only their built-in compact summaries."""
    with tqdm(
        total=experiment.config["world"]["max_steps"]
        * experiment.config["experiment"]["runs"],
        disable=not progress,
        desc="PPO steps",
        file=sys.stderr,
    ) as progress_bar:
        while not experiment.done:
            experiment.step()
            progress_bar.update()


def frozen_sampler(seed):
    """Seeded sampling is local to evaluation, including new animal identities."""
    streams = {}

    def choose(brains, observations, **_kwargs):
        result = {}
        for identity, observation in observations.items():
            brain = brains[identity]
            if identity not in streams:
                streams[identity] = random.Random((seed << 32) ^ identity)
            rng = streams[identity]
            result[identity] = rng.choices(
                brain.actions, weights=brain.action_probabilities(observation)
            )[0]
        return result

    return choose


def version_config(config, version):
    """Copy only the representation treatment; older arms have no V6-only keys."""
    result = copy.deepcopy(config)
    for brain in result["brains"].values():
        if brain["type"] != "ppo":
            raise ValueError(
                "This comparison isolates ordinary PPO; every brain must use ppo"
            )
        brain["observation_version"] = version
        if version < 6:
            brain.pop("soft_categories", None)
    return result


def worker(args):
    """One isolated learner version and training seed; frozen holdouts follow."""
    # Keep package imports local so the orchestration process stays lightweight.
    # Keep the matched-study protocol explicit beside its treatment configuration.
    # pylint: disable=import-outside-toplevel,protected-access,too-many-locals,duplicate-code
    from wtf.config import load, validate
    from wtf.experiment.runner import Experiment
    from wtf.learning.brains import load_model_weights, model_snapshot
    from wtf.learning.consequences import Consequences

    config = copy.deepcopy(load(args.environment))
    config["seed"] = args.seed
    config["world"]["max_steps"] = args.steps
    config["experiment"].update(runs=args.episodes, world_seed_policy="increment")
    config["recording"]["enabled"] = False
    config = version_config(config, args.version)
    config = validate(config)
    common = copy.deepcopy(config)
    for brain in common["brains"].values():
        brain.pop("observation_version", None)
        brain.pop("soft_categories", None)
    experiment = Experiment(config)
    initial_world = digest(experiment.world.snapshot())
    started = time.perf_counter()
    run_episode(experiment, progress=args.progress)
    training_seconds = time.perf_counter() - started
    knowledge = experiment.world._social.knowledge if experiment.world._social else None
    evaluations = []
    for seed in args.evaluation_seeds:
        for mode in ("greedy", "stochastic"):
            evaluation_config = copy.deepcopy(config)
            evaluation_config["seed"] = seed
            evaluation_config["experiment"].update(runs=1, world_seed_policy="fixed")
            evaluation = Experiment(
                evaluation_config, mode="evaluate", freeze_evaluation_brains=True
            )
            for identity, animal in evaluation.world.animals.items():
                metric = (
                    config["species"][animal.species]
                    .get("champion_selection", {})
                    .get("criterion", "lifespan")
                )
                champion = experiment.champions.entries[animal.species][animal.sex][
                    metric
                ]
                load_model_weights(evaluation.brains[identity], champion["model"])
            frozen = [
                (brain, digest(model_snapshot(brain)))
                for brain in evaluation.brains.values()
            ]
            if knowledge is not None:
                evaluation.world._social.knowledge = Consequences.restore(
                    evaluation.config, knowledge.export_state()
                )
            knowledge_before = (
                digest(evaluation.world._social.knowledge.checkpoint())
                if knowledge is not None
                else None
            )
            evaluation.observations = evaluation.world.observe_all()
            if mode == "stochastic":
                with patch(
                    "wtf.learning.batched.choose_actions",
                    side_effect=frozen_sampler(seed),
                ):
                    run_episode(evaluation)
            else:
                run_episode(evaluation)
            assert evaluation.total_updates == 0
            assert all(
                digest(model_snapshot(brain)) == state for brain, state in frozen
            )
            if knowledge is not None:
                assert (
                    digest(evaluation.world._social.knowledge.checkpoint())
                    == knowledge_before
                )
            evaluations.append(
                {"mode": mode, **compact_summary(evaluation.summaries[0])}
            )
    from wtf.simulation.soft_observations import soft_contract

    return {
        "version": args.version,
        "seed": args.seed,
        "steps": args.steps,
        "episodes": args.episodes,
        "common_config_sha256": digest(common),
        "soft_categories": (
            {name: soft_contract(brain) for name, brain in config["brains"].items()}
            if args.version == 6
            else None
        ),
        "initial_world_sha256": initial_world,
        "training_seconds": training_seconds,
        "training_updates": experiment.total_updates,
        "training": [compact_summary(s) for s in experiment.summaries],
        "evaluation": evaluations,
    }


def main():
    """Run sequential pairs and print results while retaining a small JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default="environments/seaside_150.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 200, 300])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[900, 901])
    parser.add_argument(
        "--output", type=Path, default=Path(".cache/v4-v5-comparison.json")
    )
    parser.add_argument(
        "--versions", type=int, nargs=2, choices=[4, 5, 6], default=[4, 5]
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--version", type=int, choices=[4, 5, 6], help=argparse.SUPPRESS
    )
    parser.add_argument("--seed", type=int, default=100, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.steps < 1 or args.episodes < 1:
        parser.error("steps and episodes must be positive")
    if args.version is not None:
        print(json.dumps(worker(args), allow_nan=False))
        return
    rows = []
    for seed in args.seeds:
        pair = []
        for version in args.versions:
            print(
                f"Running V{version}, seed {seed}: {args.episodes} episodes × {args.steps} steps",
                flush=True,
            )
            command = [
                sys.executable,
                "-m",
                "work.compare_observation_versions",
                "--version",
                str(version),
                "--seed",
                str(seed),
                "--environment",
                args.environment,
                "--steps",
                str(args.steps),
                "--episodes",
                str(args.episodes),
                "--evaluation-seeds",
                *(str(s) for s in args.evaluation_seeds),
            ]
            if args.progress:
                command.append("--progress")
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                check=True,
            )
            row = json.loads(completed.stdout)
            pair.append(row)
            rows.append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
            holdouts = [
                (s["mode"], s["seed"], s["populations"], s["kills"], s["births"])
                for s in row["evaluation"]
            ]
            print(
                f"  {row['training_seconds']:.2f}s training; "
                f"births {[s['births'] for s in row['training']]}; holdouts {holdouts}",
                flush=True,
            )
        assert pair[0]["common_config_sha256"] == pair[1]["common_config_sha256"]
        assert pair[0]["initial_world_sha256"] == pair[1]["initial_world_sha256"]
    print(
        f"All matched configuration/world checks passed. Report: {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
