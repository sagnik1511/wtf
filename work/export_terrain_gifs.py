"""Clip full recorded viewer frames into small documentation GIFs (FFmpeg required)."""

# Black owns code layout; long literal payloads, URLs and regression names retain
# their exact text.
# pylint: disable=line-too-long


import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

CLIPS = [
    (
        "river-crossing",
        "Screen Recording 2026-11-17 at 17.38.51.mov",
        10.0,
        "river_crossing_20261117T120742Z_bb0e94",
    ),
    (
        "highlands",
        "Screen Recording 2026-11-17 at 17.40.05.mov",
        18.0,
        "highlands_20261117T120742Z_a3668d",
    ),
    (
        "enclosed-valleys",
        "Screen Recording 2026-11-17 at 17.41.08.mov",
        10.0,
        "enclosed_valleys_20261117T120742Z_1cfefe",
    ),
]


def probe(path):
    """Inspect the source media streams needed for the clip export."""

    # Keep independent numeric/compatibility or benchmark paths explicit to preserve
    # their exact contracts.
    # pylint: disable=duplicate-code

    return json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_read_frames:format=duration",
                "-of",
                "json",
                str(path),
            ]
        )
    )


def run(args):
    """Execute the controlled media export operation."""
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + args, check=True
    )


def main():
    """Parse command-line options and run the requested operation."""

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-locals
    # The callback is consumed synchronously within this iteration; it never escapes
    # the loop.
    # pylint: disable=cell-var-from-loop

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/terrains"))
    args = parser.parse_args()
    sources = [
        (name, args.source_dir / source, start, recording)
        for name, source, start, recording in CLIPS
    ]
    missing = [p.name for _, p, _, _ in sources if not p.is_file()]
    if missing:
        parser.error("Missing source recordings: " + ", ".join(missing))
    args.output.mkdir(parents=True, exist_ok=True)
    cache = Path(".cache/terrain-gifs")
    cache.mkdir(parents=True, exist_ok=True)
    report = []
    with tempfile.TemporaryDirectory(prefix="palette-", dir=cache) as directory:
        for name, source, start, recording in sources:
            duration = 3.5
            fps = 12
            width = 1280
            palette = Path(directory) / (name + ".png")
            video = f"fps={fps},scale={width}:-2:flags=lanczos"
            run(
                [
                    "-ss",
                    str(start),
                    "-t",
                    str(duration),
                    "-i",
                    str(source),
                    "-vf",
                    video + ",palettegen=max_colors=128:stats_mode=full",
                    "-frames:v",
                    "1",
                    str(palette),
                ]
            )
            output = args.output / (name + ".gif")
            run(
                [
                    "-ss",
                    str(start),
                    "-t",
                    str(duration),
                    "-i",
                    str(source),
                    "-i",
                    str(palette),
                    "-filter_complex",
                    f"[0:v]{video}[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                    "-an",
                    "-loop",
                    "0",
                    str(output),
                ]
            )
            result = probe(output)
            stream = result["streams"][0]
            assert int(stream["nb_read_frames"]) == 42, result
            assert 3 <= float(result["format"]["duration"]) <= 4, result
            assert stream["width"] == 1280 and stream["height"] == 718, result
            digest = hashlib.sha256()
            with source.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
            report.append(
                {
                    "asset": output.name,
                    "source_file": source.name,
                    "source_sha256": digest.hexdigest(),
                    "source_interval_seconds": [start, start + duration],
                    "recording": recording,
                    "run": 1,
                    "source_dimensions": [3400, 1906],
                    "dimensions": [stream["width"], stream["height"]],
                    "fps": fps,
                    "frames": 42,
                    "duration_seconds": float(result["format"]["duration"]),
                    "palette_colors": 128,
                    "loop": True,
                    "crop": None,
                    "bytes": output.stat().st_size,
                }
            )
            print(output, output.stat().st_size, "bytes", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
