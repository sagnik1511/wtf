"""Compact per-species champions and immutable, compatible model generations."""
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

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_LIBRARY = PROJECT / 'brain_library'
CRITERIA = ('health', 'kills', 'lifespan', 'food', 'balanced')


def selection_criterion(config, species, fallback=None):
    spec = config['species'][species]
    return spec.get('champion_selection', {}).get('criterion',
                fallback or ('lifespan' if spec['diet']=='resource' else 'kills'))


def balanced_policy(config, species):
    policy = config['species'][species].get('champion_selection', {})
    return dict(version=1, survival_steps_per_kill=policy.get('survival_steps_per_kill', 300),
                max_survival_bonus=policy.get('max_survival_bonus', 2))


def balanced_rank(age, kills, energy, health, policy):
    bonus = min(age/policy['survival_steps_per_kill'],policy['max_survival_bonus']) if kills>0 else 0
    return (kills+bonus,kills,age,energy,health)


def source_balanced_rank(source, policy):
    return balanced_rank(source['age'],source['kills'],source.get('energy_gained',0),source.get('health',0),policy)



def ranks(animal):
    return dict(health=(animal.health, animal.age, animal.kills, animal.energy_gained),
                kills=(animal.kills, animal.age, animal.energy_gained, animal.health),
                lifespan=(animal.age, animal.kills, animal.energy_gained, animal.health),
                food=(animal.energy_gained, animal.age, animal.kills, animal.health))


class ChampionBook:
    """Bounded copies of model weights at the moment a new record is achieved."""
    def __init__(self, config, entries=None):
        self.config = config
        self.entries = copy.deepcopy(entries or {})

    def consider(self, animals, brains, source):
        # Resolve all ranking comparisons first: intermediate record breakers
        # need no model copies if another animal wins before this pass ends.
        policies, winners = {}, {}
        for animal in animals:
            champions = self.entries.setdefault(animal.species, {})
            selected = winners.setdefault(animal.species, {})
            scores = ranks(animal)
            if self.config['species'][animal.species]['diet']=='prey':
                if animal.species not in policies:
                    policies[animal.species] = balanced_policy(self.config,animal.species)
                scores['balanced'] = balanced_rank(animal.age,animal.kills,animal.energy_gained,animal.health,policies[animal.species])
            for key, rank in scores.items():
                previous = selected[key][0] if key in selected else tuple(champions[key]['rank']) if key in champions else None
                if previous is None or rank > previous:
                    selected[key] = rank, animal
        for species, selected in winners.items():
            snapshots, metadata = {}, {}
            for key, (rank, animal) in selected.items():
                if animal.id not in snapshots:
                    brain = brains[animal.id]
                    snapshots[animal.id] = model_snapshot(brain)
                    metadata[animal.id] = dict(source, animal_id=animal.id, species=animal.species,
                        health=animal.health, alive=animal.health > 0, age=animal.age, kills=animal.kills,
                        energy_gained=animal.energy_gained, parent_id=animal.parent_id,
                        body_generation=animal.generation,
                        statistics_since_age=animal.statistics_since_age,
                        optimizer_updates=brain.updates)
                entry = dict(criterion=key, score=rank[0], rank=list(rank),
                             source=metadata[animal.id], model=snapshots[animal.id])
                if key=='balanced':entry['selection_policy'] = policies[species]
                self.entries[species][key] = entry


def _component(value):
    if re.fullmatch(r'[A-Za-z0-9_-]+', value):
        return value
    return 'species_' + hashlib.sha256(value.encode()).hexdigest()[:16]


def catalog(root=DEFAULT_LIBRARY):
    entries = []
    root = Path(root).resolve()
    if not root.exists():
        return entries
    for path in sorted(root.glob('*/manifest.json')):
        if path.parent.name.startswith('.'):
            continue
        resolved = path.resolve()
        if root not in resolved.parents:
            raise ValueError('Library manifest outside brain library')
        manifest = json.loads(resolved.read_text())
        if manifest.get('library_version') != 1:
            raise ValueError(f'Unsupported library version: {path}')
        for entry in manifest['entries']:
            model_path = (path.parent / entry['file']).resolve()
            if path.parent.resolve() not in model_path.parents:
                raise ValueError('Saved brain path outside its generation')
            entries.append(dict(entry, path=str(model_path),
                                created_at=manifest['created_at']))
    return entries


def read_artifact(path):
    artifact = json.loads(Path(path).read_text())
    if artifact.get('brain_artifact_version') != 1:
        raise ValueError(f'Unsupported saved-brain format: {path}')
    if artifact.get('criterion') not in CRITERIA:
        raise ValueError(f'Invalid saved-brain criterion: {path}')
    required = ('artifact_id', 'model_generation', 'source', 'model', 'score', 'rank')
    if any(key not in artifact for key in required):
        raise ValueError(f'Incomplete saved-brain metadata: {path}')
    if type(artifact['model_generation']) is not int or artifact['model_generation'] < 1:
        raise ValueError(f'Invalid model generation: {path}')
    if not isinstance(artifact['source'], dict) or 'species' not in artifact['source']:
        raise ValueError(f'Missing saved-brain species: {path}')
    return artifact


def load_initial_models(config, root=DEFAULT_LIBRARY):
    """Resolve selections once; checkpoints embed their weights and provenance."""
    selected = {}
    choices = None
    for species, spec in config['species'].items():
        request = spec.get('initial_brain')
        if request is None:
            continue
        if 'path' in request:
            path = Path(request['path']).expanduser()
            if not path.is_absolute():
                path = PROJECT / path
        else:
            if choices is None:
                choices = catalog(root)
            signature = model_signature(config['brains'][spec['brain']], spec['allowed_actions'])
            matches = [entry for entry in choices if entry['species'] == species
                       and entry['criterion'] == request['metric'] and entry['signature'] == signature
                       and (request['metric']!='balanced' or entry.get('selection_policy')==balanced_policy(config,species))]
            if not matches:
                raise ValueError(f'No compatible {request["metric"]} champion for {species}; train and archive this brain architecture first')
            key = (lambda entry: (tuple(entry['rank']), entry['created_at'], entry['artifact_id'])) if request['selection'] == 'best' else (lambda entry: (entry['created_at'], entry['artifact_id']))
            path = Path(max(matches, key=key)['path'])
        artifact = read_artifact(path)
        if artifact['source']['species'] != species:
            raise ValueError(f'Saved brain species does not match {species}: {path}')
        if artifact['model']['signature'] != model_signature(config['brains'][spec['brain']], spec['allowed_actions']):
            raise ValueError(f'Saved brain is incompatible with {species} algorithm, architecture or actions: {path}')
        selected[species] = dict(model=artifact['model'],
            provenance=dict(artifact_id=artifact['artifact_id'], path=str(path.resolve()),
                criterion=artifact['criterion'], score=artifact['score'],
                model_generation=artifact['model_generation']))
        if spec.get('champion_selection',{}).get('criterion')=='balanced':
            selected[species]['provenance']['chaining_criterion'] = 'balanced'
            selected[species]['provenance']['balanced_score'] = source_balanced_rank(artifact['source'],balanced_policy(config,species))[0]
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
    if source.get("library_version") != 1:
        raise ValueError("Unsupported source champion index")
    return index_path, source


def apply_brain_overrides(config, requests=(), source_experiment=None,
                          source_run=None, root=DEFAULT_LIBRARY):
    """Apply invocation-only selections without editing environment files."""
    selections = {}
    for request in requests:
        if "=" not in request:
            raise ValueError("--brain must be SPECIES=MODEL (for example prey=food)")
        species, model = request.split("=", 1)
        if species not in config["species"]:
            raise ValueError(f"Unknown species in --brain: {species}")
        if species in selections:
            raise ValueError(f"Duplicate --brain selection for {species}")
        if not model.strip():
            raise ValueError(f"Empty --brain model for {species}")
        selections[species] = model
    if source_run is not None and (source_experiment is None or source_run < 1):
        raise ValueError("--brain-run needs --brains-from and a positive run number")
    source = None
    if source_experiment is not None:
        index_path, source = source_index(source_experiment, source_run)
        for species, spec in config["species"].items():
            selections.setdefault(species, selection_criterion(config,species))
    for species, model in selections.items():
        spec = config["species"][species]
        if model == "fresh":
            spec.pop("initial_brain", None)
        elif model.startswith(("best:", "latest:")):
            selection, metric = model.split(":", 1)
            if metric not in CRITERIA:
                raise ValueError(f"Unknown champion criterion: {metric}")
            spec.setdefault('champion_selection',{})['criterion'] = metric
            spec["initial_brain"] = dict(selection=selection, metric=metric)
        elif model in CRITERIA:
            if source is None:
                raise ValueError("A bare champion criterion needs --brains-from; use best:food for the library")
            entry = next((e for e in source["entries"]
                          if e["species"] == species and e["criterion"] == model), None)
            if model=='balanced':
                if spec['diet']!='prey':raise ValueError('Balanced selection requires a predator species')
                policy = balanced_policy(config,species)
                # Older runs cannot recreate an unarchived individual's weights.
                # Rescore only available snapshots, without changing that archive.
                if entry is None or entry.get('selection_policy')!=policy:
                    candidates = [e for e in source['entries'] if e['species']==species]
                    entry = max(candidates,key=lambda e:source_balanced_rank(e['source'],policy),default=None)
            if entry is None:
                raise ValueError(f"Source run has no {species} {model} champion")
            spec.setdefault('champion_selection',{})['criterion'] = model
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
                    raise ValueError(f"Archived champion file is unavailable: {entry['file']}")
            try:
                path = path.relative_to(PROJECT)
            except ValueError:
                pass
            spec["initial_brain"] = dict(path=str(path))
        else:
            spec["initial_brain"] = dict(path=model)
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
    generation = 'gen_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8]
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
    staging = root / ('.' + generation)
    staging.mkdir()
    manifest = dict(library_version=1, generation_id=generation,
        created_at=datetime.now(timezone.utc).isoformat(), label=label,
        entries=[])
    try:
        for species, champions in experiment.champions.entries.items():
            initial = experiment.run_initial_models.get(species, {}).get('provenance')
            for criterion, entry in champions.items():
                artifact_id = f'{generation}:{_component(species)}:{criterion}'
                artifact = dict(brain_artifact_version=1, artifact_id=artifact_id,
                    model_generation=(initial['model_generation'] if initial else 0) + 1,
                    parent_artifact=copy.deepcopy(initial), experiment_id=label,
                    training_config=experiment.config, run_complete=experiment.finished_run,
                    continued_from_run=experiment.run_index if experiment.run_index and (experiment.config['experiment'].get('chain_best_brains', True) if not experiment.legacy_brain_retention else experiment.config['experiment']['retain_brains_between_runs']) else None,
                    **entry)
                relative = Path(_component(species)) / (criterion + '.json')
                target = staging / relative
                target.parent.mkdir(exist_ok=True)
                target.write_text(json.dumps(artifact, separators=(',', ':'), allow_nan=False) + '\n')
                manifest['entries'].append(dict(artifact_id=artifact_id, species=species,
                    criterion=criterion, score=entry['score'], rank=entry['rank'],
                    signature=entry['model']['signature'], file=relative.as_posix(),
                    source=entry['source'], model_generation=artifact['model_generation']))
                if criterion=='balanced':manifest['entries'][-1]['selection_policy'] = entry['selection_policy']
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        staging.rename(final)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    local = dict(manifest, library_directory=str(final))
    if local_models:
        local["model_directory"] = final.name
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    try:
        temporary.write_text(json.dumps(local, indent=2, allow_nan=False) + '\n')
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    if local_models or not experiment.champion_artifacts:
        experiment.champion_artifacts = {}
        for entry in manifest['entries']:
            experiment.champion_artifacts.setdefault(entry['species'], {})[entry['criterion']] = dict(
                artifact_id=entry['artifact_id'], path=str(final / entry['file']),
                criterion=entry['criterion'], score=entry['score'],
                model_generation=entry['model_generation'], source_run=experiment.run_index + 1)
    return final



def localize_champions(index_path):
    """Copy a legacy shared archive into its run, preserving artifact identities."""
    index_path = Path(index_path).resolve()
    source = json.loads(index_path.read_text())
    if source.get("library_version") != 1:
        raise ValueError("Unsupported source champion index")
    if "model_directory" in source:
        return (index_path.parent / source["model_directory"]).resolve()
    origin = Path(source["library_directory"]).resolve()
    final = index_path.with_suffix("")
    staging = index_path.parent / ('.localize_' + uuid.uuid4().hex)
    if final.exists():
        manifest = json.loads((final / 'manifest.json').read_text())
        if manifest['generation_id'] != source['generation_id']:
            raise ValueError('A different champion generation already occupies this run')
    else:
        staging.mkdir()
        try:
            for entry in source['entries']:
                path = (origin / entry['file']).resolve()
                target = (staging / entry['file']).resolve()
                if origin not in path.parents or staging not in target.parents:
                    raise ValueError('Champion path outside archive')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            (staging / 'manifest.json').write_text(json.dumps(source, indent=2) + '\n')
            staging.rename(final)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    updated = dict(source, model_directory=final.name, library_directory=str(final))
    temporary = index_path.with_suffix('.json.tmp')
    try:
        temporary.write_text(json.dumps(updated, indent=2, allow_nan=False) + '\n')
        temporary.replace(index_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return final


def main():
    parser = argparse.ArgumentParser(description='List saved WTF brain generations and compatible champion paths')
    parser.add_argument('--directory', type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument('--run', help='Experiment folder name in results/ or an explicit path')
    parser.add_argument('--run-number', type=int, help='Numbered run inside that experiment')
    parser.add_argument('--localize', action='store_true', help='Copy a legacy shared archive into the selected run')
    args = parser.parse_args()
    if args.run:
        path, manifest = source_index(args.run, args.run_number)
        if args.localize:
            localize_champions(path)
            path, manifest = source_index(args.run, args.run_number)
        directory = (path.parent / manifest['model_directory']).resolve() if 'model_directory' in manifest else Path(manifest['library_directory'])
        entries = [dict(e, path=str(directory / e['file'])) for e in manifest['entries']]
    else:
        if args.localize or args.run_number is not None:
            parser.error('--localize and --run-number require --run')
        entries = catalog(args.directory)
    for entry in entries:
        print(f"{entry['species']} · {entry['signature']['algorithm']} · {entry['criterion']}={entry['score']} · model generation {entry['model_generation']}\n  {entry['path']}")


if __name__ == '__main__':
    main()
