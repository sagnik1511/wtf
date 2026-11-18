"""Run: python -m ecosystem.cli environments/baseline.json --runs 2."""

import argparse
import csv
import json
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .config import load
from .runner import Experiment
from .checkpoint_io import read_checkpoint
from .recording import Recorder, settings
from .progress import RunProgress
from .library import (
    DEFAULT_LIBRARY,
    export_champions,
    apply_brain_overrides,
    apply_champion_overrides,
)


def main():
    parser = argparse.ArgumentParser(
        description="WTF — Wildlife Trial Framework: run RL ecosystem experiments"
    )
    parser.add_argument("environment", nargs="?", default="environments/baseline.json")
    parser.add_argument("--mode", choices=("train", "evaluate", "random"), default=None)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--seed",
        type=int,
        help="Use a reproducible base seed (defaults seed policy to increment)",
    )
    parser.add_argument(
        "--seed-policy",
        choices=("random", "fixed", "increment"),
        help="Override the environment seed policy",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume a compact .json.gz or legacy JSON checkpoint; environment and overrides are ignored",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Skip playback recording (metrics and checkpoints still saved)",
    )
    display = parser.add_mutually_exclusive_group()
    display.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        help="Force tqdm output, including when stderr is redirected",
    )
    display.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable training progress output",
    )
    parser.set_defaults(progress=None)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument(
        "--library-dir",
        type=Path,
        help="Optional shared library for selectors and an extra champion copy; run-local brains are always saved",
    )
    parser.add_argument(
        "--checkpoint-format",
        choices=("binary", "compact", "json"),
        default="binary",
        help="Checkpoint output: binary arrays + XZ (default), compact JSON, or legacy JSON",
    )
    parser.add_argument(
        "--no-library", action="store_true", help="Skip saving new champion models"
    )
    parser.add_argument(
        "--brains-from",
        type=Path,
        help="Source run name in results/ or experiment path; defaults to its latest archived numbered run",
    )
    parser.add_argument(
        "--brain-run", type=int, help="Archived run number within --brains-from"
    )
    parser.add_argument(
        "--brain",
        action="append",
        default=[],
        metavar="SPECIES[:SEX]=MODEL",
        help="Repeat per species/sex (prey:male=lifespan): criterion (including balanced or lineage) from --brains-from, JSON path, best:metric, latest:metric, or fresh",
    )
    parser.add_argument(
        "--champion",
        action="append",
        default=[],
        metavar="SPECIES[:SEX]=CRITERION",
        help="Choose the champion used for later-run chaining, without loading saved starting weights",
    )
    chain = parser.add_mutually_exclusive_group()
    chain.add_argument(
        "--chain-brains",
        dest="chain_brains",
        action="store_true",
        help="Initialize later runs from the previous run's champions (default)",
    )
    chain.add_argument(
        "--no-chain-brains",
        dest="chain_brains",
        action="store_false",
        help="Reset every run to the original starting models",
    )
    parser.set_defaults(chain_brains=None)
    args = parser.parse_args()
    if args.resume:
        if (
            any(
                v is not None
                for v in (
                    args.runs,
                    args.steps,
                    args.seed,
                    args.seed_policy,
                    args.brains_from,
                    args.brain_run,
                    args.chain_brains,
                )
            )
            or args.brain
            or args.champion
        ):
            parser.error("Configuration overrides cannot be used with --resume")
        experiment = Experiment.restore(read_checkpoint(args.resume), mode=args.mode)
    else:
        config = load(args.environment)
        if args.chain_brains is not None:
            config["experiment"]["chain_best_brains"] = args.chain_brains
        else:
            config["experiment"].setdefault("chain_best_brains", True)
        if args.runs is not None:
            config["experiment"]["runs"] = args.runs
        if args.steps is not None:
            config["world"]["max_steps"] = args.steps
        if args.seed is not None:
            if args.seed_policy == "random":
                parser.error("--seed cannot be combined with --seed-policy random")
            config["seed"] = args.seed
            config["experiment"]["world_seed_policy"] = args.seed_policy or "increment"
        elif args.seed_policy is not None:
            config["experiment"]["world_seed_policy"] = args.seed_policy
        try:
            apply_brain_overrides(
                config,
                args.brain,
                args.brains_from,
                args.brain_run,
                root=args.library_dir or DEFAULT_LIBRARY,
            )
            apply_champion_overrides(config, args.champion)
            experiment = Experiment(
                config,
                args.mode or "train",
                library_root=args.library_dir or DEFAULT_LIBRARY,
            )
        except (ValueError, OSError) as error:
            parser.error(str(error))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = args.output / f'{experiment.config["name"]}_{stamp}_{uuid.uuid4().hex[:6]}'
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "checkpoints").mkdir()

    def save(name, data):
        (folder / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

    save("environment.json", experiment.config)
    save(
        "manifest.json",
        {
            "mode": experiment.mode,
            "python": sys.version,
            "platform": platform.platform(),
            "resumed_from": str(args.resume) if args.resume else None,
            "initialized_from": {
                species: {sex: value["provenance"] for sex, value in groups.items()}
                for species, groups in experiment.initial_models.items()
            },
            "randomization_seed": experiment.randomization_seed,
            "checkpoint_version": {"json": 1, "compact": 2, "binary": 3}[
                args.checkpoint_format
            ],
            "checkpoint_format": args.checkpoint_format,
            "observation_version": max(
                experiment.config["brains"][spec["brain"]].get("observation_version", 1)
                for spec in experiment.config["species"].values()
            ),
            "guard_lineage_chaining": experiment.guard_lineage_chaining,
            "playback_recording": not args.no_record
            and settings(experiment.config)["enabled"],
        },
    )
    save("run_seeds.json", experiment.run_seeds)
    recorder = (
        Recorder(folder, experiment.config, experiment.mode)
        if not args.no_record and settings(experiment.config)["enabled"]
        else None
    )
    progress = RunProgress(experiment.mode, enabled=args.progress)
    completed = False
    try:
        run_experiment(
            experiment,
            folder,
            save,
            recorder,
            progress,
            library_root=args.library_dir,
            save_brains=not args.no_library,
            checkpoint_format=args.checkpoint_format,
        )
        completed = True
    finally:
        progress.close()
        if recorder is not None:
            recorder.close(completed=completed)
    print(f"Results: {folder.resolve()}")


def run_experiment(
    experiment,
    folder,
    save,
    recorder,
    progress=None,
    library_root=None,
    save_brains=True,
    checkpoint_format="compact",
):
    suffix = {"binary": ".bin.xz", "compact": ".json.gz", "json": ".json"}[
        checkpoint_format
    ]
    species = list(experiment.config["species"])
    with (folder / "metrics.csv").open("w", newline="") as handle:
        fields = (
            ["run", "seed", "tick", "food", "starvation", "predation"]
            + (["drowning"] if "terrain" in experiment.config else [])
            + [
                f"{s}_{metric}"
                for s in species
                for metric in (
                    "population",
                    "energy_gained",
                    "kills",
                    "births",
                    "mutations",
                )
            ]
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        def record(m):
            row = dict(
                run=experiment.run_index + 1,
                seed=experiment.world.seed,
                tick=m["tick"],
                food=m["food"],
                starvation=m["deaths"].get("starvation", 0),
                predation=m["deaths"].get("predation", 0),
            )
            if "terrain" in experiment.config:
                row["drowning"] = m["deaths"].get("drowning", 0)
            for s in species:
                row.update(
                    {
                        f"{s}_population": m["populations"][s],
                        f"{s}_energy_gained": m["energy_gained"].get(s, 0),
                        f"{s}_kills": m["kills"].get(s, 0),
                        f"{s}_births": m["births"].get(s, 0),
                        f"{s}_mutations": m.get("mutations", {}).get(s, 0),
                    }
                )
            writer.writerow(row)

        record(experiment.world.metrics())
        if recorder is not None:
            recorder.begin_run(experiment)
        if progress is not None:
            progress.start(experiment)
        while not experiment.done:
            if experiment.finished_run:
                experiment.next_run()
                save("run_seeds.json", experiment.run_seeds)
                record(experiment.world.metrics())
                if recorder is not None:
                    recorder.begin_run(experiment)
                if progress is not None:
                    progress.start(experiment)
            experiment.step()
            if progress is not None:
                progress.tick(experiment)
            if recorder is not None:
                recorder.capture(experiment, force=experiment.finished_run)
            if experiment.last_record:
                record(experiment.last_record)
            if experiment.finished_run:
                if recorder is not None:
                    recorder.end_run(experiment)
                run = experiment.run_index + 1
                if save_brains and experiment.mode == "train":
                    try:
                        export_champions(
                            experiment,
                            folder / f"champions/run_{run:04d}.json",
                            label=folder.name,
                        )
                        if library_root is not None:
                            export_champions(
                                experiment,
                                folder / f"champions/shared_{run:04d}.json",
                                root=library_root,
                                label=folder.name,
                            )
                    except (OSError, ValueError):
                        # A library failure must not discard trained controllers.
                        experiment.write_checkpoint(
                            folder / ("checkpoints/library_error" + suffix)
                        )
                        raise
                print(
                    f"Run {run} (seed {experiment.world.seed}): {experiment.world.metrics()}",
                    flush=True,
                )
                if (
                    not experiment.done
                    and run % experiment.config["experiment"]["checkpoint_every_runs"]
                    == 0
                ):
                    experiment.write_checkpoint(
                        folder / f"checkpoints/run_{run:04d}{suffix}"
                    )
            if experiment.last_record is not None or experiment.finished_run:
                handle.flush()
    save("run_seeds.json", experiment.run_seeds)
    save("summary.json", experiment.summaries)
    experiment.write_checkpoint(folder / ("checkpoints/final" + suffix))


if __name__ == "__main__":
    main()
