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
from .recording import Recorder, settings


def main():
    parser = argparse.ArgumentParser(description='Run a reproducible learning ecosystem experiment')
    parser.add_argument('environment', nargs='?', default='environments/baseline.json')
    parser.add_argument('--mode', choices=('train', 'evaluate', 'random'), default=None)
    parser.add_argument('--runs', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume', type=Path, help='Resume an exact JSON checkpoint; environment and overrides are ignored')
    parser.add_argument('--no-record', action='store_true', help='Skip playback recording (metrics and checkpoints still saved)')
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    if args.resume:
        if any(v is not None for v in (args.runs, args.steps, args.seed)):
            parser.error('Configuration overrides cannot be used with --resume')
        experiment = Experiment.restore(json.loads(args.resume.read_text()), mode=args.mode)
    else:
        config = load(args.environment)
        if args.runs is not None:
            config['experiment']['runs'] = args.runs
        if args.steps is not None:
            config['world']['max_steps'] = args.steps
        if args.seed is not None:
            config['seed'] = args.seed
        experiment = Experiment(config, args.mode or 'train')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder = args.output / f'{experiment.config["name"]}_{stamp}_{uuid.uuid4().hex[:6]}'
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'checkpoints').mkdir()
    def save(name, data):
        (folder / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    save('environment.json', experiment.config)
    save('manifest.json', {'mode': experiment.mode, 'python': sys.version, 'platform': platform.platform(), 'resumed_from': str(args.resume) if args.resume else None, 'checkpoint_version': 1, 'observation_version': 1, 'playback_recording': not args.no_record and settings(experiment.config)['enabled']})
    recorder = Recorder(folder, experiment.config, experiment.mode) if not args.no_record and settings(experiment.config)['enabled'] else None
    completed = False
    try:
        run_experiment(experiment, folder, save, recorder)
        completed = True
    finally:
        if recorder is not None:
            recorder.close(completed=completed)
    print(f'Results: {folder.resolve()}')


def run_experiment(experiment, folder, save, recorder):
    species = list(experiment.config['species'])
    with (folder / 'metrics.csv').open('w', newline='') as handle:
        fields = ['run', 'seed', 'tick', 'food', 'starvation', 'predation'] + [f'{s}_{metric}' for s in species for metric in ('population', 'energy_gained', 'kills')]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        def record(m):
            row = dict(run=experiment.run_index + 1, seed=experiment.world.seed, tick=m['tick'], food=m['food'], starvation=m['deaths'].get('starvation', 0), predation=m['deaths'].get('predation', 0))
            for s in species:
                row.update({f'{s}_population': m['populations'][s], f'{s}_energy_gained': m['energy_gained'].get(s, 0), f'{s}_kills': m['kills'].get(s, 0)})
            writer.writerow(row)
        record(experiment.world.metrics())
        if recorder is not None:
            recorder.begin_run(experiment)
        while not experiment.done:
            if experiment.finished_run:
                experiment.next_run()
                record(experiment.world.metrics())
                if recorder is not None:
                    recorder.begin_run(experiment)
            experiment.step()
            if recorder is not None:
                recorder.capture(experiment, force=experiment.finished_run)
            if experiment.last_record:
                record(experiment.last_record)
            if experiment.finished_run:
                if recorder is not None:
                    recorder.end_run(experiment)
                run = experiment.run_index + 1
                print(f'Run {run}: {experiment.world.metrics()}', flush=True)
                if run % experiment.config['experiment']['checkpoint_every_runs'] == 0:
                    save(f'checkpoints/run_{run:04d}.json', experiment.checkpoint())
            handle.flush()
    save('run_seeds.json', experiment.run_seeds)
    save('summary.json', experiment.summaries)
    save('checkpoints/final.json', experiment.checkpoint())


if __name__ == '__main__':
    main()
