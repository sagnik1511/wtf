"""Stream playback frames to bounded chunks, without retaining a run in RAM.

This module depends on neither the simulation nor the learning implementation.
Only committed chunks are advertised in the atomically replaced manifest.
"""
import copy
import csv
import json
from pathlib import Path

DEFAULT_RECORDING = dict(enabled=True, frame_every_steps=1, frames_per_chunk=64, diagnostics_every_steps=10)


def settings(config):
    return dict(DEFAULT_RECORDING, **config.get('recording', {}))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, separators=(',', ':'), allow_nan=False)
        handle.write('\n')
    temporary.replace(path)


class Recorder:
    def __init__(self, folder, config, mode):
        self.folder = Path(folder) / 'playback'
        self.folder.mkdir(parents=True, exist_ok=False)
        self.options = settings(config)
        self.manifest = dict(recording_version=1, name=config['name'], mode=mode, status='writing', config=copy.deepcopy(config), recording=self.options, runs=[])
        self.stream = None
        self.run = None
        self.chunk_frames = 0
        self.last_tick = None
        self.last_diagnostic_tick = None
        self._publish()

    def _publish(self):
        write_json(self.folder / 'recording.json', self.manifest)

    def begin_run(self, experiment):
        if self.run is not None:
            self.end_run(experiment=None)
        world = experiment.world
        self.run = dict(run=experiment.run_index + 1, seed=world.seed, obstacles=sorted(world.obstacles), chunk_count=0, frame_count=0, first_tick=world.tick, last_tick=world.tick, complete=experiment.finished_run)
        self.manifest['runs'].append(self.run)
        self.last_tick = self.last_diagnostic_tick = None
        self._publish()
        self.capture(experiment, force=True)

    def _open_chunk(self):
        name = f'run_{self.run["run"]:04d}_chunk_{self.run["chunk_count"]:06d}.json'
        self.chunk_path = self.folder / name
        self.stream = self.chunk_path.with_suffix('.json.tmp').open('w')
        self.stream.write('[')
        self.chunk_frames = 0

    def _commit_chunk(self):
        if self.stream is None:
            return
        self.stream.write(']\n')
        self.stream.close()
        self.stream = None
        self.chunk_path.with_suffix('.json.tmp').replace(self.chunk_path)
        self.run['chunk_count'] += 1
        self.run['frame_count'] += self.chunk_frames
        self.run['last_tick'] = self.last_tick
        self._publish()

    def capture(self, experiment, force=False):
        tick = experiment.world.tick
        if tick == self.last_tick or (not force and tick % self.options['frame_every_steps']):
            return False
        if self.stream is None:
            self._open_chunk()
        frame = experiment.world.snapshot()
        for key in ('obstacles', 'width', 'height', 'resource_capacity'):
            frame.pop(key)
        frame['total_updates'] = experiment.total_updates
        frame['finished_run'] = experiment.finished_run
        diagnostic_interval = self.options['diagnostics_every_steps']
        if diagnostic_interval and (self.chunk_frames == 0 or force or self.last_diagnostic_tick is None or tick - self.last_diagnostic_tick >= diagnostic_interval):
            frame['diagnostics'] = {}
            for i, a in experiment.world.animals.items():
                b = experiment.brains[i]
                o = experiment.observations[i]
                frame['diagnostics'][str(i)] = dict(tick=tick, **b.diagnostics(o))
            self.last_diagnostic_tick = tick
        if self.chunk_frames:
            self.stream.write(',\n')
        # Encode a single frame with the C encoder, never an entire run.
        self.stream.write(json.dumps(frame, separators=(',', ':'), allow_nan=False))
        self.chunk_frames += 1
        self.last_tick = tick
        if self.chunk_frames >= self.options['frames_per_chunk']:
            self._commit_chunk()
        return True

    def end_run(self, experiment=None):
        if self.run is None:
            return
        if experiment is not None:
            self.capture(experiment, force=True)
            self.run['complete'] = experiment.finished_run
        self._commit_chunk()
        self._publish()
        self.run = None

    def close(self, completed=True):
        self.end_run()
        self.manifest['status'] = 'complete' if completed else 'interrupted'
        self._publish()


class RecordingStore:
    """Read-only, disk-backed catalog. Never loads brains or whole recordings."""
    def __init__(self, results):
        self.results = Path(results).resolve()

    def _folder(self, recording_id):
        if not isinstance(recording_id, str) or not recording_id or Path(recording_id).name != recording_id or recording_id in ('.', '..'):
            raise ValueError('Invalid recording id')
        folder = (self.results / recording_id / 'playback').resolve()
        if self.results not in folder.parents:
            raise ValueError('Recording path outside results folder')
        return folder

    def catalog(self):
        records = []
        for p in sorted(self.results.glob('*/playback/recording.json'), reverse=True):
            try:
                # Use the same containment checks as direct requests (symlinks too).
                m = self.metadata(p.parent.parent.name)
                if any(r['chunk_count'] for r in m['runs']):
                    records.append(dict(id=p.parent.parent.name, name=m['name'], mode=m['mode'], status=m['status'], runs=len(m['runs'])))
            except (OSError, ValueError, KeyError):
                continue
        return records

    def metadata(self, recording_id):
        m = json.loads((self._folder(recording_id) / 'recording.json').read_text())
        if m.get('recording_version') != 1:
            raise ValueError('Unsupported recording version')
        return m

    def metrics(self, recording_id, run, max_points=2048):
        """Bounded, lightweight history independent of decoded world chunks.

        Existing CLI recordings already have metrics.csv. Standalone playback
        exports without that file are read one committed chunk at a time.
        """
        if type(run) is not int or type(max_points) is not int or not 2 <= max_points <= 2048:
            raise ValueError('Invalid run or metrics sample limit')
        manifest = self.metadata(recording_id)
        entry = next((r for r in manifest['runs'] if r['run'] == run), None)
        if entry is None or not entry['chunk_count']:
            raise FileNotFoundError('Recorded run not found')
        folder = self._folder(recording_id)
        metrics_path = (folder.parent / 'metrics.csv').resolve()
        if metrics_path.parent != folder.parent:
            raise ValueError('Metrics path outside recording folder')

        def rows():
            if metrics_path.is_file():
                with metrics_path.open(newline='') as handle:
                    for row in csv.DictReader(handle):
                        if int(row['run']) == run:
                            yield dict(tick=int(row['tick']), food=float(row['food']), populations={name: int(row[f'{name}_population']) for name in manifest['config']['species']})
            else:
                for index in range(entry['chunk_count']):
                    frames = json.loads(self.chunk(recording_id, run, index).read_text())
                    for frame in frames:
                        yield frame['metrics']
                    del frames

        points, last, stride, count = [], None, 1, 0
        for metric in rows():
            if not entry['first_tick'] <= metric['tick'] <= entry['last_tick']:
                continue
            # Store only chart values, never agents, events or diagnostics.
            last = dict(tick=metric['tick'], food=metric['food'], populations=metric['populations'])
            if count % stride == 0:
                points.append(last)
            count += 1
            if len(points) > max_points - 1:
                points = points[::2]
                stride *= 2
        if last is not None and (not points or points[-1]['tick'] != last['tick']):
            points.append(last)
        return points

    def chunk(self, recording_id, run, index):
        m = self.metadata(recording_id)
        if type(run) is not int or type(index) is not int:
            raise ValueError('Run and chunk must be integers')
        entry = next((r for r in m['runs'] if r['run'] == run), None)
        if entry is None or not 0 <= index < entry['chunk_count']:
            raise FileNotFoundError('Recorded chunk not found')
        folder = self._folder(recording_id)
        path = (folder / f'run_{run:04d}_chunk_{index:06d}.json').resolve()
        if path.parent != folder:
            raise ValueError('Chunk path outside recording folder')
        return path
