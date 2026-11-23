"""Encode full-viewer replay frame exports as compact documentation GIFs."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


CLIPS = {
    "simulation-playback": ("baseline", "simulation-playback.gif"),
    "species-metrics": ("baseline", "species-metrics.gif"),
    "day-night": ("day_night", "day-night.gif"),
    "animal-brain": ("baseline", "animal-brain.gif"),
    "archived-brain": ("baseline", "archived-brain.gif"),
    "river-crossing": ("river_crossing", "terrains/river-crossing.gif"),
    "highlands": ("highlands", "terrains/highlands.gif"),
    "enclosed-valleys": ("enclosed_valleys", "terrains/enclosed-valleys.gif"),
}
ROOT = Path(__file__).resolve().parents[1]


def main():
    """Parse command-line options and run the requested operation."""

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-locals

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--light-frames-dir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets")
    args = parser.parse_args()
    manifest = []
    renderer = hashlib.sha256()
    for source in sorted((ROOT / "viewer").rglob("*")):
        if source.is_file():
            renderer.update(source.relative_to(ROOT).as_posix().encode())
            renderer.update(source.read_bytes())
    for name, (preset, destination) in CLIPS.items():
        folder = args.frames_dir / name
        capture = json.loads((folder / "capture.json").read_text())
        count = int(capture["frames"])
        duration = float(capture["duration_seconds"])
        if count < 2 or not 3 <= duration <= 4:
            raise ValueError(f"{name}: expected multiple frames and 3–4 seconds")
        sequence = hashlib.sha256()
        for index in range(count):
            sequence.update((folder / f"{index:03d}.jpg").read_bytes())
        output = args.output / destination
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(count / duration),
                "-i",
                str(folder / "%03d.jpg"),
                "-filter_complex",
                "[0:v]split[v][p];[p]palettegen=max_colors=128:stats_mode=diff[pal];"
                "[v][pal]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                "-frames:v",
                str(count),
                "-loop",
                "0",
                str(output),
            ],
            check=True,
        )
        probe = json.loads(
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
                    str(output),
                ]
            )
        )
        stream = probe["streams"][0]
        assert int(stream["nb_read_frames"]) == count, probe
        assert abs(float(probe["format"]["duration"]) - duration) < 0.02, probe
        manifest.append(
            {
                "asset": destination,
                "source": "current viewer replay frame exports; no desktop video recording",
                "preset": f"environments/{preset}.json",
                "preset_sha256": hashlib.sha256(
                    (ROOT / f"environments/{preset}.json").read_bytes()
                ).hexdigest(),
                "viewer_sha256": renderer.hexdigest(),
                "frame_sequence_sha256": sequence.hexdigest(),
                "seed": 100,
                "runs": 1,
                "steps": 300,
                "mode": "train",
                "sampled_steps": [int(step) for step in capture["steps"]],
                "theme": capture["theme"],
                "dimensions": [stream["width"], stream["height"]],
                "frames": count,
                "fps": count / duration,
                "duration_seconds": duration,
                "palette_colors": 128,
                "loop": True,
                "crop": None,
                "bytes": output.stat().st_size,
            }
        )
        if args.light_frames_dir:
            light_folder = args.light_frames_dir / name
            light_capture = json.loads((light_folder / "capture.json").read_text())
            if (
                light_capture["steps"] != capture["steps"]
                or light_capture["frames"] != count
            ):
                raise ValueError(f"{name}: light/dark replay samples differ")
            light_path = args.output / "themed" / f"{name}-light.webp"
            light_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-framerate",
                    str(count / duration),
                    "-i",
                    str(light_folder / "%03d.jpg"),
                    "-filter_complex",
                    "[0:v]split[v][p];[p]palettegen=max_colors=128:stats_mode=diff[pal];"
                    "[v][pal]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                    "-c:v",
                    "libwebp_anim",
                    "-lossless",
                    "1",
                    "-compression_level",
                    "6",
                    "-frames:v",
                    str(count),
                    "-loop",
                    "0",
                    str(light_path),
                ],
                check=True,
            )
            light_hash = hashlib.sha256()
            for index in range(count):
                light_hash.update((light_folder / f"{index:03d}.jpg").read_bytes())
            manifest[-1].update(
                {
                    "light_asset": light_path.relative_to(args.output).as_posix(),
                    "light_bytes": light_path.stat().st_size,
                    "light_frame_sequence_sha256": light_hash.hexdigest(),
                    "theme_selection": "site palette; GitHub picture media; original GIF fallback",
                }
            )
        print(f"{destination}: {output.stat().st_size:,} bytes")
    (args.output / "playback-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    terrain = [
        {**entry, "asset": Path(entry["asset"]).name}
        for entry in manifest
        if entry["asset"].startswith("terrains/")
    ]
    (args.output / "terrains/manifest.json").write_text(
        json.dumps(terrain, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
