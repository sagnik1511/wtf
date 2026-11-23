"""Run: python -m wtf.cli environments/baseline.json --runs 2."""

import argparse
import csv
import json
import math
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from wtf.config import load
from wtf.experiment.runner import Experiment
from wtf.storage.checkpoint_io import read_checkpoint
from wtf.storage.recording import (
    Recorder,
    settings,
    DEFAULT_RECORDING,
    RECORDING_PROFILES,
)
from wtf.progress import RunProgress
from wtf.storage.library import (
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
    parser.add_argument(
        "--recording-profile",
        choices=RECORDING_PROFILES,
        help="Training capture: full (configured cadence), compact (diagnostics >=50 steps, no activations), or training-only (no replay); full also works for evaluation",
    )
    parser.add_argument(
        "--record-last-episodes",
        type=int,
        metavar="K",
        help="Record the final K scheduled episodes, capped at total; -1 records all (default), 0 disables playback. Metrics/brains/checkpoints still cover every episode",
    )
    parser.add_argument("--consequence-update-schedule", choices=("continuous", "episode"),
        help="Override enabled consequence publication timing for a fresh experiment; episode publishes only at next-episode start")
    parser.add_argument(
        "--consequences-from",
        type=Path,
        help="Load compatible population consequence knowledge from a result folder or consequences.json.xz; independent policy weights are unchanged",
    )
    parser.add_argument(
        "--freeze-brains",
        action="store_true",
        help="Evaluation only: keep offspring weights identical to the same-sex parent's brain; no crossover or mutation. Saved in recovery checkpoints",
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
        help="Initialize later runs from matching best-so-far champions (default)",
    )
    chain.add_argument(
        "--no-chain-brains",
        dest="chain_brains",
        action="store_false",
        help="Reset every run to the original starting models",
    )
    parser.set_defaults(chain_brains=None)
    parser.add_argument(
        "--early-stop-runs",
        type=int,
        help="Stop after K consecutive runs without a selected champion improvement; 0 disables",
    )
    parser.add_argument(
        "--early-stop-species",
        action="append",
        default=None,
        metavar="SPECIES",
        help="Monitor this species (repeat to monitor several); default: any configured species",
    )
    parser.add_argument(
        "--checkpoint-keep-last",
        type=int,
        help="Keep latest N new recovery checkpoints, including final; 0 keeps all (default 2)",
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        help="Periodic full recovery interval; 0 disables (default 500)",
    )
    parser.add_argument(
        "--brain-keep-last",
        type=int,
        help="Keep N recent qualifying best-brain milestones; 0 keeps all (default 2)",
    )
    parser.add_argument(
        "--brain-save-every-steps",
        type=int,
        help="Persist improved changed-weight champions every K steps; 0 disables (default 250)",
    )
    parser.add_argument(
        "--brain-save-min-score-delta",
        type=float,
        help="Minimum primary-score gain for periodic brain saves (default 0)",
    )
    checkpoint_metadata = parser.add_mutually_exclusive_group()
    checkpoint_metadata.add_argument(
        "--shared-checkpoint-metadata",
        action="store_true",
        help="Opt in to run-local shared metadata for binary recovery checkpoints",
    )
    checkpoint_metadata.add_argument(
        "--self-contained-checkpoints",
        action="store_true",
        help="Embed metadata instead of using the run-local binary checkpoint pool",
    )
    args = parser.parse_args()
    if args.record_last_episodes is not None and args.record_last_episodes < -1:
        parser.error("--record-last-episodes must be -1 or nonnegative")
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
                    args.early_stop_runs,
                    args.early_stop_species,
                )
            )
            or args.brain
            or args.champion
        ):
            parser.error("Configuration overrides cannot be used with --resume")
        experiment = Experiment.restore(read_checkpoint(args.resume), mode=args.mode)
    else:
        config = load(args.environment)
        if args.record_last_episodes is not None:
            config.setdefault("recording", dict(DEFAULT_RECORDING))[
                "record_last_episodes"
            ] = args.record_last_episodes
        if args.recording_profile is not None:
            config.setdefault("recording", dict(DEFAULT_RECORDING))[
                "profile"
            ] = args.recording_profile
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
        if args.early_stop_runs is not None or args.early_stop_species is not None:
            stop = config["experiment"].setdefault("early_stopping", {})
            if args.early_stop_runs is not None:
                stop["patience_runs"] = args.early_stop_runs
            if args.early_stop_species is not None:
                stop["species"] = args.early_stop_species
            if "patience_runs" not in stop:
                parser.error(
                    "--early-stop-species needs --early-stop-runs or configured patience_runs"
                )
        try:
            apply_brain_overrides(
                config,
                args.brain,
                args.brains_from,
                args.brain_run,
                root=args.library_dir or DEFAULT_LIBRARY,
            )
            apply_champion_overrides(config, args.champion)
            if args.consequence_update_schedule is not None:
                if not config.get("consequence_learning", {}).get("enabled", False):
                    parser.error("--consequence-update-schedule requires enabled consequence learning")
                config["consequence_learning"]["update_schedule"]=args.consequence_update_schedule
            experiment = Experiment(
                config,
                args.mode or "train",
                library_root=args.library_dir or DEFAULT_LIBRARY,
            )
        except (ValueError, OSError) as error:
            parser.error(str(error))
    if args.resume and args.consequence_update_schedule is not None:
        parser.error("--consequence-update-schedule cannot override exact resume")
    if args.consequences_from is not None:
        if args.resume:
            parser.error("--consequences-from cannot replace exact-resume knowledge")
        if (
            experiment.world._social is None
            or experiment.world._social.knowledge is None
        ):
            parser.error(
                "--consequences-from requires enabled consequence learning and inputs"
            )
        from wtf.learning.consequences import Consequences

        try:
            experiment.world._social.knowledge = Consequences.load(
                experiment.config, args.consequences_from
            )
            experiment.observations = experiment.world.observe_all()
        except (ValueError, OSError, EOFError) as error:
            parser.error(str(error))
    if args.freeze_brains and experiment.mode != "evaluate":
        parser.error("--freeze-brains requires --mode evaluate")
    if args.freeze_brains:
        experiment.freeze_evaluation_brains = True
    if args.recording_profile is not None:
        if experiment.mode != "train" and args.recording_profile != "full":
            parser.error(
                "compact and training-only recording profiles require train mode"
            )
        experiment.config.setdefault("recording", dict(DEFAULT_RECORDING))[
            "profile"
        ] = args.recording_profile
    if args.record_last_episodes is not None:
        experiment.config.setdefault("recording", dict(DEFAULT_RECORDING))[
            "record_last_episodes"
        ] = args.record_last_episodes
    recording_options = settings(experiment.config, experiment.mode)
    if args.no_record:
        recording_options["enabled"] = False
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = args.output / f'{experiment.config["name"]}_{stamp}_{uuid.uuid4().hex[:6]}'
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "checkpoints").mkdir()

    def save(name, data):
        (folder / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

    storage_options = dict(experiment.config["experiment"])
    for name in (
        "checkpoint_keep_last",
        "checkpoint_every_steps",
        "brain_keep_last",
        "brain_save_every_steps",
        "brain_save_min_score_delta",
    ):
        value = getattr(args, name)
        if value is not None:
            if value < 0 or (
                name == "brain_save_min_score_delta" and not math.isfinite(value)
            ):
                parser.error(name.replace("_", "-") + " must be nonnegative and finite")
            storage_options[name] = value
    if args.self_contained_checkpoints:
        storage_options["checkpoint_shared_metadata"] = False
    elif args.shared_checkpoint_metadata:
        storage_options["checkpoint_shared_metadata"] = True
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
            "checkpoint_version": (
                4
                if args.checkpoint_format == "binary"
                and storage_options.get("checkpoint_shared_metadata", False)
                else {"json": 1, "compact": 2, "binary": 3}[args.checkpoint_format]
            ),
            "storage_policy": storage_options,
            "checkpoint_format": args.checkpoint_format,
            "observation_version": max(
                experiment.config["brains"][spec["brain"]].get("observation_version", 1)
                for spec in experiment.config["species"].values()
            ),
            "guard_lineage_chaining": experiment.guard_lineage_chaining,
            "promote_only_improved": experiment.improvement_only,
            "early_stopping": experiment.config["experiment"].get("early_stopping"),
            "freeze_evaluation_brains": experiment.freeze_evaluation_brains,
            "recording_profile": recording_options["profile"],
            "recording_settings": recording_options,
            "playback_recording": recording_options["enabled"],
        },
    )
    save("run_seeds.json", experiment.run_seeds)
    recorder = (
        Recorder(folder, experiment.config, experiment.mode)
        if recording_options["enabled"]
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
            storage_options=storage_options,
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
    storage_options=None,
):
    suffix = {"binary": ".bin.xz", "compact": ".json.gz", "json": ".json"}[
        checkpoint_format
    ]
    from wtf.storage.retention import ArtifactRetention

    storage_options = (
        dict(experiment.config["experiment"])
        if storage_options is None
        else storage_options
    )
    retention = ArtifactRetention(folder, storage_options, checkpoint_format)
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

        metrics = experiment.world.metrics()
        record(metrics)
        if recorder is not None and (
            not experiment.finished_run
            or (experiment.done and not experiment.early_stopping.stopped)
        ):
            # Retain the legacy final-frame view for ordinary completed resumes,
            # while skipping already-finished episodes before the next run.
            recorder.begin_run(experiment, metrics=metrics)
        if progress is not None and not experiment.finished_run:
            progress.start(experiment, metrics=metrics)
        while not experiment.done:
            if experiment.finished_run:
                experiment.next_run()
                save("run_seeds.json", experiment.run_seeds)
                metrics = experiment.world.metrics()
                record(metrics)
                if recorder is not None:
                    recorder.begin_run(experiment, metrics=metrics)
                if progress is not None:
                    progress.start(experiment, metrics=metrics)
            experiment.step()
            if progress is not None:
                progress.tick(experiment)
            if recorder is not None:
                recorder.capture(
                    experiment,
                    force=experiment.finished_run,
                    metrics=experiment.last_record,
                )
            if experiment.last_record:
                record(experiment.last_record)
            if not experiment.finished_run:
                checkpoint_interval = storage_options.get("checkpoint_every_steps", 500)
                if (
                    checkpoint_interval
                    and experiment.world.tick % checkpoint_interval == 0
                ):
                    retention.checkpoint(
                        experiment,
                        f"run_{experiment.run_index + 1:04d}_step_{experiment.world.tick:09d}",
                    )
                if save_brains and experiment.mode == "train":
                    retention.step_brains(experiment)
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
                    f"Run {run} (seed {experiment.world.seed}): {experiment.last_record}",
                    flush=True,
                )
                if experiment.early_stopping.stopped:
                    print(
                        f"Early stopping after run {run}: no selected champion improvement "
                        f"for {experiment.early_stopping.stagnant_runs} consecutive runs.",
                        flush=True,
                    )
                if (
                    not experiment.done
                    and run % experiment.config["experiment"]["checkpoint_every_runs"]
                    == 0
                ):
                    retention.checkpoint(experiment, f"run_{run:04d}")
            if experiment.last_record is not None or experiment.finished_run:
                handle.flush()
    save("run_seeds.json", experiment.run_seeds)
    save("summary.json", experiment.summaries)
    if experiment.early_stopping.patience:
        save("early_stopping.json", experiment.early_stopping.checkpoint())
    retention.checkpoint(experiment, "final")
    if (
        experiment.world._social is not None
        and experiment.world._social.knowledge is not None
    ):
        experiment.world._social.knowledge.save(folder / "consequences.json.xz")


if __name__ == "__main__":
    main()
