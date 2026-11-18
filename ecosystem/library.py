"""Compact per-species/sex champions and immutable, compatible model generations."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import uuid

from .brains import model_signature, model_snapshot
from .lineage import LineageBook, lineage_policy

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_LIBRARY = PROJECT / "brain_library"
CRITERIA = ("health", "kills", "lifespan", "food", "balanced", "lineage")
SEXES = ("female", "male")


def selection_criterion(config, species, fallback=None, sex=None):
    spec = config["species"][species]
    if sex in spec.get("champion_selection_by_sex", {}):
        return spec["champion_selection_by_sex"][sex]
    return spec.get("champion_selection", {}).get(
        "criterion", fallback or ("lifespan" if spec["diet"] == "resource" else "kills")
    )


def balanced_policy(config, species):
    policy = config["species"][species].get("champion_selection", {})
    return dict(
        version=1,
        survival_steps_per_kill=policy.get("survival_steps_per_kill", 300),
        max_survival_bonus=policy.get("max_survival_bonus", 2),
    )


def balanced_rank(age, kills, energy, health, policy):
    bonus = (
        min(age / policy["survival_steps_per_kill"], policy["max_survival_bonus"])
        if kills > 0
        else 0
    )
    return (kills + bonus, kills, age, energy, health)


def source_balanced_rank(source, policy):
    return balanced_rank(
        source["age"],
        source["kills"],
        source.get("energy_gained", 0),
        source.get("health", 0),
        policy,
    )


def ranks(animal):
    return dict(
        health=(animal.health, animal.age, animal.kills, animal.energy_gained),
        kills=(animal.kills, animal.age, animal.energy_gained, animal.health),
        lifespan=(animal.age, animal.kills, animal.energy_gained, animal.health),
        food=(animal.energy_gained, animal.age, animal.kills, animal.health),
    )


class ChampionBook:
    """Bounded copies of model weights at the moment a new record is achieved."""

    def __init__(self, config, entries=None, lineage_state=None):
        self.config = config
        self.lineage = LineageBook(config, lineage_state)
        self.entries = {}
        for species, groups in (entries or {}).items():
            if any(k in CRITERIA for k in groups):
                # Legacy archives only establish a sex winner when recorded.
                for metric, entry in groups.items():
                    sex = entry.get("source", {}).get("sex")
                    if sex in SEXES:
                        self.entries.setdefault(species, {}).setdefault(sex, {})[
                            metric
                        ] = copy.deepcopy(entry)
            else:
                self.entries[species] = copy.deepcopy(groups)

    def consider(self, animals, brains, source, newborns=()):
        animals = tuple(animals)
        self.lineage.observe(animals + tuple(newborns))
        # Resolve all ranking comparisons first: intermediate record breakers
        # need no model copies if another animal wins before this pass ends.
        policies, winners = {}, {}
        for animal in animals:
            champions = self.entries.setdefault(animal.species, {}).setdefault(
                animal.sex, {}
            )
            selected = winners.setdefault((animal.species, animal.sex), {})
            scores = ranks(animal)
            scores["lineage"] = self.lineage.rank(animal)[0]
            if self.config["species"][animal.species]["diet"] == "prey":
                if animal.species not in policies:
                    policies[animal.species] = balanced_policy(
                        self.config, animal.species
                    )
                scores["balanced"] = balanced_rank(
                    animal.age,
                    animal.kills,
                    animal.energy_gained,
                    animal.health,
                    policies[animal.species],
                )
            for key, rank in scores.items():
                previous = (
                    selected[key][0]
                    if key in selected
                    else tuple(champions[key]["rank"]) if key in champions else None
                )
                if previous is None or rank > previous:
                    selected[key] = rank, animal
        for (species, sex), selected in winners.items():
            snapshots, metadata = {}, {}
            for key, (rank, animal) in selected.items():
                if animal.id not in snapshots:
                    brain = brains[animal.id]
                    snapshots[animal.id] = model_snapshot(brain)
                    metadata[animal.id] = dict(
                        source,
                        animal_id=animal.id,
                        species=animal.species,
                        health=animal.health,
                        alive=animal.health > 0,
                        age=animal.age,
                        kills=animal.kills,
                        energy_gained=animal.energy_gained,
                        parent_id=animal.parent_id,
                        body_generation=animal.generation,
                        sex=animal.sex,
                        second_parent_id=animal.second_parent_id,
                        statistics_since_age=animal.statistics_since_age,
                        optimizer_updates=brain.updates,
                    )
                    if animal.mutation_count:
                        metadata[animal.id].update(
                            mutated=animal.mutated,
                            mutation_count=animal.mutation_count,
                            mutation_weights=animal.mutation_weights,
                        )
                entry = dict(
                    criterion=key,
                    score=rank[0],
                    rank=list(rank),
                    source=metadata[animal.id],
                    model=snapshots[animal.id],
                )
                if key == "balanced":
                    entry["selection_policy"] = policies[species]
                elif key == "lineage":
                    entry["selection_policy"] = lineage_policy(self.config, species)
                    entry["lineage"] = self.lineage.rank(animal)[1]
                self.entries[species][sex][key] = entry


def _component(value):
    if re.fullmatch(r"[A-Za-z0-9_-]+", value):
        return value
    return "species_" + hashlib.sha256(value.encode()).hexdigest()[:16]


def catalog(root=DEFAULT_LIBRARY):
    entries = []
    root = Path(root).resolve()
    if not root.exists():
        return entries
    for path in sorted(root.glob("*/manifest.json")):
        if path.parent.name.startswith("."):
            continue
        resolved = path.resolve()
        if root not in resolved.parents:
            raise ValueError("Library manifest outside brain library")
        manifest = json.loads(resolved.read_text())
        if manifest.get("library_version") not in (1, 2):
            raise ValueError(f"Unsupported library version: {path}")
        for entry in manifest["entries"]:
            model_path = (path.parent / entry["file"]).resolve()
            if path.parent.resolve() not in model_path.parents:
                raise ValueError("Saved brain path outside its generation")
            entries.append(
                dict(entry, path=str(model_path), created_at=manifest["created_at"])
            )
    return entries


def read_artifact(path):
    artifact = json.loads(Path(path).read_text())
    if artifact.get("brain_artifact_version") not in (1, 2):
        raise ValueError(f"Unsupported saved-brain format: {path}")
    if artifact.get("criterion") not in CRITERIA:
        raise ValueError(f"Invalid saved-brain criterion: {path}")
    required = ("artifact_id", "model_generation", "source", "model", "score", "rank")
    if any(key not in artifact for key in required):
        raise ValueError(f"Incomplete saved-brain metadata: {path}")
    if (
        type(artifact["model_generation"]) is not int
        or artifact["model_generation"] < 1
    ):
        raise ValueError(f"Invalid model generation: {path}")
    if not isinstance(artifact["source"], dict) or "species" not in artifact["source"]:
        raise ValueError(f"Missing saved-brain species: {path}")
    if (
        artifact["brain_artifact_version"] == 2
        and artifact["source"].get("sex") not in SEXES
    ):
        raise ValueError(f"Missing saved-brain sex: {path}")
    return artifact


def sex_models(models):
    """Migrate older species-wide initializations without changing live weights."""
    result = {}
    for species, value in models.items():
        result[species] = (
            {sex: copy.deepcopy(value) for sex in SEXES}
            if "model" in value
            else copy.deepcopy(value)
        )
    return result


def load_initial_models(config, root=DEFAULT_LIBRARY):
    """Resolve independent species/sex selections; checkpoints embed weights."""
    selected, choices, artifacts = {}, None, {}
    for species, spec in config["species"].items():
        requests = spec.get("initial_brains", {})
        signature = model_signature(
            config["brains"][spec["brain"]], spec["allowed_actions"]
        )
        requested = False
        for sex in SEXES:
            explicit = sex in requests
            request = requests.get(sex, spec.get("initial_brain"))
            if request is None:
                continue
            requested = True
            if "path" in request:
                path = Path(request["path"]).expanduser()
                if not path.is_absolute():
                    path = PROJECT / path
            else:
                if choices is None:
                    choices = catalog(root)
                matches = [
                    e
                    for e in choices
                    if e["species"] == species
                    and e["criterion"] == request["metric"]
                    and e["signature"] == signature
                    and (
                        request["metric"] != "balanced"
                        or e.get("selection_policy") == balanced_policy(config, species)
                    )
                    and (
                        request["metric"] != "lineage"
                        or e.get("selection_policy") == lineage_policy(config, species)
                    )
                ]
                matching_sex = [
                    e
                    for e in matches
                    if e.get("sex", e.get("source", {}).get("sex")) == sex
                ]
                matches = matching_sex or [
                    e
                    for e in matches
                    if e.get("sex", e.get("source", {}).get("sex")) is None
                ]
                if not matches:
                    if explicit:
                        raise ValueError(
                            f'No compatible {request["metric"]} champion for {species}:{sex}'
                        )
                    continue
                key = (
                    (lambda e: (tuple(e["rank"]), e["created_at"], e["artifact_id"]))
                    if request["selection"] == "best"
                    else (lambda e: (e["created_at"], e["artifact_id"]))
                )
                path = Path(max(matches, key=key)["path"])
            cache_key = path.resolve()
            if cache_key not in artifacts:
                artifacts[cache_key] = read_artifact(path)
            artifact = artifacts[cache_key]
            if artifact["source"]["species"] != species:
                raise ValueError(
                    f"Saved brain species does not match {species}: {path}"
                )
            source_sex = artifact["source"].get("sex")
            if source_sex is not None and source_sex != sex:
                if explicit:
                    raise ValueError(
                        f"Saved brain sex does not match {species}:{sex}: {path}"
                    )
                continue
            if artifact["model"]["signature"] != signature:
                raise ValueError(
                    f"Saved brain is incompatible with {species} algorithm, architecture or actions: {path}"
                )
            entry = dict(
                model=artifact["model"],
                provenance=dict(
                    artifact_id=artifact["artifact_id"],
                    path=str(path.resolve()),
                    sex=source_sex,
                    criterion=artifact["criterion"],
                    score=artifact["score"],
                    model_generation=artifact["model_generation"],
                ),
            )
            if selection_criterion(config, species, sex=sex) == "balanced":
                entry["provenance"]["chaining_criterion"] = "balanced"
                entry["provenance"]["balanced_score"] = source_balanced_rank(
                    artifact["source"], balanced_policy(config, species)
                )[0]
            selected.setdefault(species, {})[sex] = entry
        if requested and not selected.get(species):
            raise ValueError(
                f'No compatible {spec.get("initial_brain", {}).get("metric", "saved")} champions for {species}'
            )
    return selected


def source_index(source_experiment, source_run=None):
    indexes = []
    source_folder = Path(source_experiment).expanduser()
    if not source_folder.is_absolute():
        if len(source_folder.parts) == 1:
            source_folder = PROJECT / "results" / source_folder
        elif not source_folder.is_dir():
            source_folder = PROJECT / source_folder
    for path in (source_folder / "champions").glob("run_*.json"):
        match = re.fullmatch(r"run_(\d+)", path.stem)
        if match:
            indexes.append((int(match[1]), path))
    if source_run is not None:
        indexes = [item for item in indexes if item[0] == source_run]
    if not indexes:
        raise ValueError(f"No archived champions found in {source_experiment}")
    index_path = max(indexes)[1].resolve()
    source = json.loads(index_path.read_text())
    if source.get("library_version") not in (1, 2):
        raise ValueError("Unsupported source champion index")
    return index_path, source


def apply_brain_overrides(
    config, requests=(), source_experiment=None, source_run=None, root=DEFAULT_LIBRARY
):
    """Invocation-only selections: SPECIES[:SEX]=MODEL; broad then sex-specific."""
    selections = {}
    for request in requests:
        if "=" not in request:
            raise ValueError(
                "--brain must be SPECIES[:SEX]=MODEL (for example prey:male=food)"
            )
        group, model = request.split("=", 1)
        species, sex = group, None
        if group not in config["species"]:
            species, separator, sex = group.rpartition(":")
            if not separator or species not in config["species"] or sex not in SEXES:
                raise ValueError(f"Unknown species or sex in --brain: {group}")
        if (species, sex) in selections:
            raise ValueError(f"Duplicate --brain selection for {group}")
        if not model.strip():
            raise ValueError(f"Empty --brain model for {group}")
        selections[species, sex] = model
    if source_run is not None and (source_experiment is None or source_run < 1):
        raise ValueError("--brain-run needs --brains-from and a positive run number")
    explicit_requests = set(selections)
    source = None
    if source_experiment is not None:
        index_path, source = source_index(source_experiment, source_run)
        for species in config["species"]:
            # Default each group independently, preserving configured criteria.
            if (species, None) not in selections:
                for sex in SEXES:
                    selections.setdefault(
                        (species, sex), selection_criterion(config, species, sex=sex)
                    )

    def source_path(entry):
        if "model_directory" in source:
            directory = (index_path.parent / source["model_directory"]).resolve()
            if index_path.parent.resolve() not in directory.parents:
                raise ValueError("Model directory outside source run")
        else:
            directory = Path(source["library_directory"]).resolve()
        path = (directory / entry["file"]).resolve()
        if directory not in path.parents:
            raise ValueError("Source champion path outside its generation")
        if not path.is_file():
            directory = (Path(root) / source["generation_id"]).resolve()
            path = (directory / entry["file"]).resolve()
            if directory not in path.parents or not path.is_file():
                raise ValueError(
                    f'Archived champion file is unavailable: {entry["file"]}'
                )
        try:
            return str(path.relative_to(PROJECT))
        except ValueError:
            return str(path)

    # Broad selections apply first, regardless of argument order.
    for (species, sex), model in sorted(
        selections.items(), key=lambda item: item[0][1] is not None
    ):
        spec = config["species"][species]
        targets = SEXES if sex is None else (sex,)
        if sex is None:
            spec.pop("initial_brain", None)
            spec.pop("initial_brains", None)
        bank = spec.setdefault("initial_brains", {})
        if model == "fresh":
            for target in targets:
                bank[target] = None
            continue
        metric = (
            model.split(":", 1)[1]
            if model.startswith(("best:", "latest:"))
            else model if model in CRITERIA else None
        )
        if metric is not None and metric not in CRITERIA:
            raise ValueError(f"Unknown champion criterion: {metric}")
        if metric == "balanced" and spec["diet"] != "prey":
            raise ValueError("Balanced selection requires a predator species")
        if metric:
            for target in targets:
                spec.setdefault("champion_selection_by_sex", {})[target] = metric
        if model.startswith(("best:", "latest:")):
            selection, metric = model.split(":", 1)
            if sex is None:
                spec["initial_brain"] = dict(selection=selection, metric=metric)
            else:
                bank[sex] = dict(selection=selection, metric=metric)
        elif model in CRITERIA:
            if source is None:
                raise ValueError(
                    "A bare champion criterion needs --brains-from; use best:food for the library"
                )
            found = False
            for target in targets:
                same_species = [e for e in source["entries"] if e["species"] == species]
                exact = [
                    e
                    for e in same_species
                    if e.get("sex", e.get("source", {}).get("sex")) == target
                ]
                candidates = exact or [
                    e
                    for e in same_species
                    if e.get("sex", e.get("source", {}).get("sex")) is None
                ]
                entry = next((e for e in candidates if e["criterion"] == model), None)
                if model == "balanced" and (
                    entry is None
                    or entry.get("selection_policy") != balanced_policy(config, species)
                ):
                    entry = max(
                        candidates,
                        key=lambda e: source_balanced_rank(
                            e["source"], balanced_policy(config, species)
                        ),
                        default=None,
                    )
                if entry is None:
                    if sex is not None and (species, sex) in explicit_requests:
                        raise ValueError(
                            f"Source run has no {species}:{target} {model} champion"
                        )
                    bank[target] = None
                else:
                    found = True
                    bank[target] = dict(path=source_path(entry))
            if not found and sex is None:
                raise ValueError(f"Source run has no {species} {model} champion")
        else:
            if sex is None:
                spec["initial_brain"] = dict(path=model)
            else:
                bank[sex] = dict(path=model)
    return config


def apply_champion_overrides(config, requests=()):
    """Choose chaining criteria without requesting any initial saved weights."""
    selected = {}
    for request in requests:
        group, separator, criterion = request.partition("=")
        species, sex = group, None
        if group not in config["species"]:
            species, _, sex = group.rpartition(":")
        if (
            not separator
            or species not in config["species"]
            or sex not in (None, *SEXES)
            or criterion not in CRITERIA
        ):
            raise ValueError("--champion requires SPECIES[:SEX]=CRITERION")
        if (species, sex) in selected:
            raise ValueError(f"Duplicate --champion selection for {group}")
        if criterion == "balanced" and config["species"][species]["diet"] != "prey":
            raise ValueError("Balanced selection requires a predator species")
        selected[species, sex] = criterion
    for (species, sex), criterion in sorted(
        selected.items(), key=lambda item: item[0][1] is not None
    ):
        choices = config["species"][species].setdefault("champion_selection_by_sex", {})
        for target in SEXES if sex is None else (sex,):
            choices[target] = criterion
    return config


def export_champions(experiment, destination, root=None, label=None):
    """Publish a complete generation atomically; never overwrite old models.

    Artifacts contain weights and metadata only, never replay/optimizer/RNG
    buffers. By default models live beside their run-local index, making the
    experiment folder self-contained. An explicit root publishes a shared copy.
    """
    if not experiment.champions.entries:
        return None
    destination = Path(destination).resolve()
    generation = (
        "gen_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    local_models = root is None
    if local_models:
        final = destination.with_suffix("")
        root = final.parent
    else:
        root = Path(root).resolve()
        final = root / generation
    root.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError(f"Champion generation already exists: {final}")
    staging = root / ("." + generation)
    staging.mkdir()
    manifest = dict(
        library_version=2,
        generation_id=generation,
        created_at=datetime.now(timezone.utc).isoformat(),
        label=label,
        entries=[],
    )
    try:
        for species, groups in experiment.champions.entries.items():
            for sex, champions in groups.items():
                initial = (
                    experiment.run_initial_models.get(species, {})
                    .get(sex, {})
                    .get("provenance")
                )
                for criterion, entry in champions.items():
                    artifact_id = (
                        f"{generation}:{_component(species)}:{sex}:{criterion}"
                    )
                    artifact = dict(
                        brain_artifact_version=2,
                        artifact_id=artifact_id,
                        model_generation=(initial["model_generation"] if initial else 0)
                        + 1,
                        parent_artifact=copy.deepcopy(initial),
                        experiment_id=label,
                        training_config=experiment.config,
                        run_complete=experiment.finished_run,
                        continued_from_run=(
                            experiment.run_index
                            if experiment.run_index
                            and (
                                experiment.config["experiment"].get(
                                    "chain_best_brains", True
                                )
                                if not experiment.legacy_brain_retention
                                else experiment.config["experiment"][
                                    "retain_brains_between_runs"
                                ]
                            )
                            else None
                        ),
                        **entry,
                    )
                    relative = Path(_component(species)) / sex / (criterion + ".json")
                    target = staging / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(
                        json.dumps(artifact, separators=(",", ":"), allow_nan=False)
                        + "\n"
                    )
                    manifest["entries"].append(
                        dict(
                            artifact_id=artifact_id,
                            species=species,
                            sex=sex,
                            criterion=criterion,
                            score=entry["score"],
                            rank=entry["rank"],
                            signature=entry["model"]["signature"],
                            file=relative.as_posix(),
                            source=entry["source"],
                            model_generation=artifact["model_generation"],
                        )
                    )
                    if criterion in ("balanced", "lineage"):
                        manifest["entries"][-1]["selection_policy"] = entry[
                            "selection_policy"
                        ]
                    if criterion == "lineage":
                        manifest["entries"][-1]["lineage"] = entry["lineage"]
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        staging.rename(final)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    local = dict(manifest, library_directory=str(final))
    if local_models:
        local["model_directory"] = final.name
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(local, indent=2, allow_nan=False) + "\n")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    if local_models or not experiment.champion_artifacts:
        experiment.champion_artifacts = {}
        for entry in manifest["entries"]:
            experiment.champion_artifacts.setdefault(entry["species"], {}).setdefault(
                entry["sex"], {}
            )[entry["criterion"]] = dict(
                artifact_id=entry["artifact_id"],
                path=str(final / entry["file"]),
                criterion=entry["criterion"],
                score=entry["score"],
                sex=entry["sex"],
                model_generation=entry["model_generation"],
                source_run=experiment.run_index + 1,
            )
    return final


def localize_champions(index_path):
    """Copy a legacy shared archive into its run, preserving artifact identities."""
    index_path = Path(index_path).resolve()
    source = json.loads(index_path.read_text())
    if source.get("library_version") not in (1, 2):
        raise ValueError("Unsupported source champion index")
    if "model_directory" in source:
        return (index_path.parent / source["model_directory"]).resolve()
    origin = Path(source["library_directory"]).resolve()
    final = index_path.with_suffix("")
    staging = index_path.parent / (".localize_" + uuid.uuid4().hex)
    if final.exists():
        manifest = json.loads((final / "manifest.json").read_text())
        if manifest["generation_id"] != source["generation_id"]:
            raise ValueError(
                "A different champion generation already occupies this run"
            )
    else:
        staging.mkdir()
        try:
            for entry in source["entries"]:
                path = (origin / entry["file"]).resolve()
                target = (staging / entry["file"]).resolve()
                if origin not in path.parents or staging not in target.parents:
                    raise ValueError("Champion path outside archive")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            (staging / "manifest.json").write_text(json.dumps(source, indent=2) + "\n")
            staging.rename(final)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    updated = dict(source, model_directory=final.name, library_directory=str(final))
    temporary = index_path.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(updated, indent=2, allow_nan=False) + "\n")
        temporary.replace(index_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return final


def main():
    parser = argparse.ArgumentParser(
        description="List saved WTF brain generations and compatible champion paths"
    )
    parser.add_argument("--directory", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument(
        "--run", help="Experiment folder name in results/ or an explicit path"
    )
    parser.add_argument(
        "--run-number", type=int, help="Numbered run inside that experiment"
    )
    parser.add_argument(
        "--localize",
        action="store_true",
        help="Copy a legacy shared archive into the selected run",
    )
    args = parser.parse_args()
    if args.run:
        path, manifest = source_index(args.run, args.run_number)
        if args.localize:
            localize_champions(path)
            path, manifest = source_index(args.run, args.run_number)
        directory = (
            (path.parent / manifest["model_directory"]).resolve()
            if "model_directory" in manifest
            else Path(manifest["library_directory"])
        )
        entries = [
            dict(e, path=str(directory / e["file"])) for e in manifest["entries"]
        ]
    else:
        if args.localize or args.run_number is not None:
            parser.error("--localize and --run-number require --run")
        entries = catalog(args.directory)
    for entry in entries:
        print(
            f"{entry['species']}:{entry.get('sex', entry.get('source', {}).get('sex')) or 'legacy'} · {entry['signature']['algorithm']} · {entry['criterion']}={entry['score']} · model generation {entry['model_generation']}\n  {entry['path']}"
        )


if __name__ == "__main__":
    main()
