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
CRITERIA = ('health', 'kills', 'lifespan', 'food')


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
        for animal in animals:
            champions = self.entries.setdefault(animal.species, {})
            scores = ranks(animal)
            wins = [key for key, rank in scores.items()
                    if key not in champions or rank > tuple(champions[key]['rank'])]
            if not wins:
                continue
            brain = brains[animal.id]
            # A single detached model copy can serve multiple winning criteria.
            model = model_snapshot(brain)
            metadata = dict(source, animal_id=animal.id, species=animal.species,
                health=animal.health, alive=animal.health > 0, age=animal.age, kills=animal.kills,
                energy_gained=animal.energy_gained, parent_id=animal.parent_id,
                body_generation=animal.generation,
                statistics_since_age=animal.statistics_since_age,
                optimizer_updates=brain.updates)
            for key in wins:
                rank = scores[key]
                champions[key] = dict(criterion=key, score=rank[0], rank=list(rank),
                                     source=metadata, model=model)


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
                       and entry['criterion'] == request['metric'] and entry['signature'] == signature]
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
    return selected


def export_champions(experiment, destination, root=DEFAULT_LIBRARY, label=None):
    """Publish a complete generation atomically; never overwrite old models.

    Artifacts contain weights and metadata only, never replay/optimizer/RNG
    buffers. The run-local manifest points to the immutable library artifacts.
    """
    if not experiment.champions.entries:
        return None
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    generation = 'gen_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8]
    staging = root / ('.' + generation)
    final = root / generation
    staging.mkdir()
    manifest = dict(library_version=1, generation_id=generation,
        created_at=datetime.now(timezone.utc).isoformat(), label=label,
        entries=[])
    try:
        for species, champions in experiment.champions.entries.items():
            initial = experiment.initial_models.get(species, {}).get('provenance')
            for criterion, entry in champions.items():
                artifact_id = f'{generation}:{_component(species)}:{criterion}'
                artifact = dict(brain_artifact_version=1, artifact_id=artifact_id,
                    model_generation=(initial['model_generation'] if initial else 0) + 1,
                    parent_artifact=copy.deepcopy(initial), experiment_id=label,
                    training_config=experiment.config, run_complete=experiment.finished_run,
                    continued_from_run=experiment.run_index if experiment.run_index and experiment.config['experiment']['retain_brains_between_runs'] else None,
                    **entry)
                relative = Path(_component(species)) / (criterion + '.json')
                target = staging / relative
                target.parent.mkdir(exist_ok=True)
                target.write_text(json.dumps(artifact, separators=(',', ':'), allow_nan=False) + '\n')
                manifest['entries'].append(dict(artifact_id=artifact_id, species=species,
                    criterion=criterion, score=entry['score'], rank=entry['rank'],
                    signature=entry['model']['signature'], file=relative.as_posix(),
                    source=entry['source'], model_generation=artifact['model_generation']))
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        staging.rename(final)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    local = dict(manifest, library_directory=str(final))
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    try:
        temporary.write_text(json.dumps(local, indent=2, allow_nan=False) + '\n')
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return final


def main():
    parser = argparse.ArgumentParser(description='List saved WTF brain generations and compatible champion paths')
    parser.add_argument('--directory', type=Path, default=DEFAULT_LIBRARY)
    args = parser.parse_args()
    for entry in catalog(args.directory):
        print(f"{entry['species']} · {entry['signature']['algorithm']} · {entry['criterion']}={entry['score']} · model generation {entry['model_generation']}\n  {entry['path']}")


if __name__ == '__main__':
    main()
