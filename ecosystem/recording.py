"""Stream playback frames to bounded chunks, without retaining a run in RAM.

This module depends on neither the simulation nor the learning implementation.
Only committed chunks are advertised in the atomically replaced manifest.
"""

import copy
from collections import OrderedDict
from threading import RLock
import csv
import gzip
import json
import lzma
from pathlib import Path
from .replay_io import chunk_path, load_chunk
from .replay_codec import DeltaEncoder

DEFAULT_RECORDING = dict(
    enabled=True,
    frame_every_steps=1,
    frames_per_chunk=64,
    diagnostics_every_steps=10,
    compression="xz",
    compression_level=6,
    encoding="binary",
)


def settings(config):
    return dict(DEFAULT_RECORDING, **config.get("recording", {}))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


class Recorder:
    def __init__(self, folder, config, mode):
        self.folder = Path(folder) / "playback"
        self.folder.mkdir(parents=True, exist_ok=False)
        self.options = settings(config)
        self.manifest = dict(
            recording_version=1,
            name=config["name"],
            mode=mode,
            status="writing",
            config=copy.deepcopy(config),
            recording=self.options,
            storage_compression=self.options["compression"],
            storage_format=self.options["encoding"],
            runs=[],
        )
        self.stream = None
        self.run = None
        self.chunk_frames = 0
        self.last_tick = None
        self.last_diagnostic_tick = None
        self._publish()

    def _publish(self):
        write_json(self.folder / "recording.json", self.manifest)

    def begin_run(self, experiment):
        if self.run is not None:
            self.end_run(experiment=None)
        world = experiment.world
        self.run = dict(
            run=experiment.run_index + 1,
            seed=world.seed,
            brain_precision=experiment.brain_precision,
            obstacles=sorted(world.obstacles),
            chunk_count=0,
            frame_count=0,
            first_tick=world.tick,
            last_tick=world.tick,
            complete=experiment.finished_run,
        )
        self.run["initialized_from"] = {
            s: {sex: m["provenance"] for sex, m in groups.items()}
            for s, groups in experiment.run_initial_models.items()
        }
        self.manifest["runs"].append(self.run)
        self.last_tick = self.last_diagnostic_tick = None
        self._publish()
        self.capture(experiment, force=True)

    def _open_chunk(self):
        binary = self.options["encoding"] == "binary"
        name = f'run_{self.run["run"]:04d}_chunk_{self.run["chunk_count"]:06d}' + (
            ".bin" if binary else ".json"
        )
        if self.options["compression"] in ("gzip", "xz"):
            name += ".gz" if self.options["compression"] == "gzip" else ".xz"
        self.chunk_path = self.folder / name
        self.chunk_temporary = self.chunk_path.with_suffix(
            self.chunk_path.suffix + ".tmp"
        )
        if binary:
            from .binary_io import RecordWriter

            if self.options["compression"] == "xz":
                self.stream = lzma.open(
                    self.chunk_temporary,
                    "wb",
                    filters=[
                        dict(
                            id=lzma.FILTER_LZMA2,
                            preset=self.options["compression_level"],
                            dict_size=1 << 20,
                        )
                    ],
                )
            else:
                self.stream = (
                    gzip.open(
                        self.chunk_temporary,
                        "wb",
                        compresslevel=self.options["compression_level"],
                    )
                    if self.options["compression"] == "gzip"
                    else self.chunk_temporary.open("wb")
                )
            self.encoder = DeltaEncoder()
            self.binary_writer = RecordWriter(self.stream)
            self.binary_writer.write(dict(kind="replay", delta=self.encoder.header()))
            self.chunk_frames = 0
            return
        self.binary_writer = None
        if self.options["compression"] == "xz":
            self.stream = lzma.open(
                self.chunk_temporary,
                "wt",
                encoding="utf-8",
                filters=[
                    dict(
                        id=lzma.FILTER_LZMA2,
                        preset=self.options["compression_level"],
                        dict_size=1 << 20,
                    )
                ],
            )
        else:
            self.stream = (
                gzip.open(
                    self.chunk_temporary,
                    "wt",
                    encoding="utf-8",
                    compresslevel=self.options["compression_level"],
                )
                if self.options["compression"] == "gzip"
                else self.chunk_temporary.open("w", encoding="utf-8")
            )
        self.encoder = DeltaEncoder() if self.options["encoding"] == "delta" else None
        self.stream.write(
            json.dumps(self.encoder.header(), separators=(",", ":"))[:-1]
            + ',"frames":['
            if self.encoder
            else "["
        )
        self.chunk_frames = 0

    def _commit_chunk(self):
        if self.stream is None:
            return
        if self.binary_writer is None:
            self.stream.write("]}\n" if self.encoder else "]\n")
        else:
            self.binary_writer.write(dict(kind="replay_end", frames=self.chunk_frames))
        self.stream.close()
        self.stream = None
        self.encoder = None
        self.binary_writer = None
        self.chunk_temporary.replace(self.chunk_path)
        self.run["chunk_count"] += 1
        self.run["frame_count"] += self.chunk_frames
        self.run["last_tick"] = self.last_tick
        self._publish()

    def capture(self, experiment, force=False):
        tick = experiment.world.tick
        if tick == self.last_tick or (
            not force and tick % self.options["frame_every_steps"]
        ):
            return False
        if self.stream is None:
            self._open_chunk()
        frame = experiment.world.snapshot(include_static=False)
        frame["total_updates"] = experiment.total_updates
        frame["finished_run"] = experiment.finished_run
        diagnostic_interval = self.options["diagnostics_every_steps"]
        if diagnostic_interval and (
            self.chunk_frames == 0
            or force
            or self.last_diagnostic_tick is None
            or tick - self.last_diagnostic_tick >= diagnostic_interval
        ):
            frame["diagnostics"] = {}
            for i, a in experiment.world.animals.items():
                b = experiment.brains[i]
                o = experiment.observations[i]
                frame["diagnostics"][str(i)] = dict(tick=tick, **b.diagnostics(o))
            self.last_diagnostic_tick = tick
        packet = self.encoder.encode(frame) if self.encoder else frame
        if self.binary_writer is not None:
            self.binary_writer.write(packet)
        else:
            if self.chunk_frames:
                self.stream.write(",\n")
            self.stream.write(
                json.dumps(packet, separators=(",", ":"), allow_nan=False)
            )
        self.chunk_frames += 1
        self.last_tick = tick
        if self.chunk_frames >= self.options["frames_per_chunk"]:
            self._commit_chunk()
        return True

    def end_run(self, experiment=None):
        if self.run is None:
            return
        if experiment is not None:
            self.capture(experiment, force=True)
            self.run["complete"] = experiment.finished_run
        self._commit_chunk()
        self._publish()
        self.run = None

    def close(self, completed=True):
        self.end_run()
        self.manifest["status"] = "complete" if completed else "interrupted"
        self._publish()


class RecordingStore:
    """Read-only, disk-backed catalog. Never loads brains or whole recordings."""

    def __init__(self, results):
        self.results = Path(results).resolve()
        self.telemetry_cache = OrderedDict()
        self._cache_lock = RLock()

    def _folder(self, recording_id):
        if (
            not isinstance(recording_id, str)
            or not recording_id
            or Path(recording_id).name != recording_id
            or recording_id in (".", "..")
        ):
            raise ValueError("Invalid recording id")
        folder = (self.results / recording_id / "playback").resolve()
        if self.results not in folder.parents:
            raise ValueError("Recording path outside results folder")
        return folder

    def catalog(self):
        records = []
        for p in sorted(self.results.glob("*/playback/recording.json"), reverse=True):
            try:
                # Use the same containment checks as direct requests (symlinks too).
                m = self.metadata(p.parent.parent.name)
                if any(r["chunk_count"] for r in m["runs"]):
                    records.append(
                        dict(
                            id=p.parent.parent.name,
                            name=m["name"],
                            mode=m["mode"],
                            status=m["status"],
                            runs=len(m["runs"]),
                        )
                    )
            except (OSError, ValueError, KeyError):
                continue
        return records

    def metadata(self, recording_id):
        m = json.loads((self._folder(recording_id) / "recording.json").read_text())
        if m.get("recording_version") != 1:
            raise ValueError("Unsupported recording version")
        return m

    def analysis(self, recording_id, run):
        if type(run) is not int:
            raise ValueError("Run must be an integer")
        from .analyze import analyze_recording

        return analyze_recording(self._folder(recording_id), run=run)

    def telemetry(self, recording_id, run, page):
        """At most 128 summaries, with four stat-validated cached pages."""
        from .telemetry import PAGE_SIZE, summarize

        if type(run) is not int or type(page) is not int or page < 0:
            raise ValueError("Invalid telemetry run or page")
        manifest = self.metadata(recording_id)
        entry = next((r for r in manifest["runs"] if r["run"] == run), None)
        start = page * PAGE_SIZE
        if entry is None or start >= entry["frame_count"]:
            raise FileNotFoundError("Recorded telemetry page not found")
        end = min(start + PAGE_SIZE, entry["frame_count"])
        size = manifest["recording"]["frames_per_chunk"]
        first_chunk = start // size
        folder = self._folder(recording_id)
        paths = {
            i: chunk_path(folder, run, i)
            for i in range(max(0, first_chunk - 1), (end - 1) // size + 1)
        }
        key = (
            recording_id,
            run,
            page,
            end,
            (folder / "recording.json").stat().st_mtime_ns,
            tuple(
                (str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths.values()
            ),
        )
        with self._cache_lock:
            if key in self.telemetry_cache:
                self.telemetry_cache.move_to_end(key)
                return copy.deepcopy(self.telemetry_cache[key])
        previous = load_chunk(paths[first_chunk - 1])[-1] if first_chunk else None
        points = []
        for index in range(first_chunk, (end - 1) // size + 1):
            frames = load_chunk(paths[index])
            for local, frame in enumerate(frames):
                position = index * size + local
                if position >= end:
                    break
                if position >= start:
                    points.append(summarize(manifest["config"], frame, previous))
                previous = frame
            del frames
        result = dict(page=page, page_size=PAGE_SIZE, start_frame=start, points=points)
        with self._cache_lock:
            self.telemetry_cache[key] = copy.deepcopy(result)
            while len(self.telemetry_cache) > 4:
                self.telemetry_cache.popitem(last=False)
        return result

    def metrics(self, recording_id, run, max_points=2048):
        """Bounded, lightweight history independent of decoded world chunks.

        Existing CLI recordings already have metrics.csv. Standalone playback
        exports without that file are read one committed chunk at a time.
        """
        if (
            type(run) is not int
            or type(max_points) is not int
            or not 2 <= max_points <= 2048
        ):
            raise ValueError("Invalid run or metrics sample limit")
        manifest = self.metadata(recording_id)
        entry = next((r for r in manifest["runs"] if r["run"] == run), None)
        if entry is None or not entry["chunk_count"]:
            raise FileNotFoundError("Recorded run not found")
        folder = self._folder(recording_id)
        metrics_path = (folder.parent / "metrics.csv").resolve()
        if metrics_path.parent != folder.parent:
            raise ValueError("Metrics path outside recording folder")

        def rows():
            if metrics_path.is_file():
                with metrics_path.open(newline="") as handle:
                    for row in csv.DictReader(handle):
                        if int(row["run"]) == run:
                            yield dict(
                                tick=int(row["tick"]),
                                food=float(row["food"]),
                                populations={
                                    name: int(row[f"{name}_population"])
                                    for name in manifest["config"]["species"]
                                },
                            )
            else:
                for index in range(entry["chunk_count"]):
                    frames = load_chunk(self.chunk(recording_id, run, index))
                    for frame in frames:
                        yield frame["metrics"]
                    del frames

        points, last, stride, count = [], None, 1, 0
        for metric in rows():
            if not entry["first_tick"] <= metric["tick"] <= entry["last_tick"]:
                continue
            # Store only chart values, never agents, events or diagnostics.
            last = dict(
                tick=metric["tick"],
                food=metric["food"],
                populations=metric["populations"],
            )
            if count % stride == 0:
                points.append(last)
            count += 1
            if len(points) > max_points - 1:
                points = points[::2]
                stride *= 2
        if last is not None and (not points or points[-1]["tick"] != last["tick"]):
            points.append(last)
        return points

    def chunk(self, recording_id, run, index):
        m = self.metadata(recording_id)
        if type(run) is not int or type(index) is not int:
            raise ValueError("Run and chunk must be integers")
        entry = next((r for r in m["runs"] if r["run"] == run), None)
        if entry is None or not 0 <= index < entry["chunk_count"]:
            raise FileNotFoundError("Recorded chunk not found")
        folder = self._folder(recording_id)
        return chunk_path(folder, run, index)
