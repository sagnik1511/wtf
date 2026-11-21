"""Read-only installed assets, with a checkout fallback; no writable package paths."""

from importlib.resources import files
from pathlib import Path, PurePosixPath

SOURCE_ROOT = Path(__file__).resolve().parent.parent


def asset_root():
    packaged = files("ecosystem").joinpath("assets")
    return packaged if packaged.joinpath("environments").is_dir() else SOURCE_ROOT


def asset(relative):
    relative = str(relative)
    parts = PurePosixPath(relative).parts
    if not parts or relative.startswith("/") or "\\" in relative or ".." in parts:
        raise ValueError("Invalid runtime asset path")
    root = asset_root()
    target = root.joinpath(*parts)
    if isinstance(target, Path):
        resolved = target.resolve()
        boundary = (
            root.joinpath(parts[0]).resolve() if len(parts) > 1 else root.resolve()
        )
        if boundary not in resolved.parents:
            raise ValueError("Runtime asset outside asset directory")
    return target


def preset_names():
    # The explicit distribution manifest excludes exploratory/local presets.
    return sorted(
        p.name[:-5]
        for p in asset("environments").iterdir()
        if p.is_file()
        and p.name.endswith(".json")
        and not p.name.startswith(("escape_intent_", "mating_intent_"))
    )


def preset_text(name):
    name = str(name)
    if name.startswith("environments/"):
        name = name[len("environments/") :]
    if name.endswith(".json"):
        name = name[:-5]
    if name not in preset_names():
        raise FileNotFoundError(f"Unknown bundled preset: {name}")
    return asset(f"environments/{name}.json").read_text(encoding="utf-8")
