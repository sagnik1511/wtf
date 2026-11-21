"""Installed-package smoke test. Run from an empty writable directory, not source.

Uses only the installed package/stdlib. Starts one temporary localhost viewer,
checks all bundled presets, a short CLI run, replay/diagnostics and checkpoint
restore. It never builds, publishes, or starts a Docker container.
"""

import json
import importlib.util
import os
import sysconfig
from pathlib import Path
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

import wtf
from wtf import Experiment
from wtf import load_config
from wtf.storage.checkpoint_io import read_checkpoint
from wtf.storage.recording import RecordingStore
from wtf.storage.replay_io import load_chunk, chunk_path
from wtf.resources import asset, preset_names
from wtf.server import make_handler, State


def main():
    workspace = Path.cwd()
    assert (
        importlib.util.find_spec("ecosystem") is None
    ), "Legacy namespace leaked into installation"
    assert (
        "site-packages" in Path(wtf.__file__).parts
    ), "Expected a fresh installed package"
    assert (
        workspace not in Path(wtf.__file__).resolve().parents
    ), "Run outside the source/package directory"
    executable = Path(sysconfig.get_path("scripts")) / (
        "wtf.exe" if os.name == "nt" else "wtf"
    )
    for command in (
        ["--help"],
        ["run", "--help"],
        ["view", "--help"],
        ["analyze", "--help"],
        ["presets"],
    ):
        subprocess.run(
            [str(executable), *command], check=True, stdout=subprocess.DEVNULL
        )
    names = preset_names()
    assert len(names) >= 27
    for name in names:
        config = load_config(name)
        assert "species" in config
    subprocess.run(
        [
            sys.executable,
            "-m",
            "wtf",
            "run",
            "maskable_ppo",
            "--runs",
            "1",
            "--steps",
            "5",
            "--seed",
            "100",
            "--no-progress",
        ],
        check=True,
    )
    folder = next((workspace / "results").glob("maskable_ppo_*"))
    state = read_checkpoint(folder / "checkpoints/final.bin.xz")
    restored = Experiment.restore(state)
    assert restored.done and restored.world.tick == 5
    store = RecordingStore(workspace / "results")
    metadata = store.metadata(folder.name)
    frames = load_chunk(chunk_path(folder / "playback", 1, 0))
    assert len(frames) == 6 and frames[-1]["tick"] == 5
    assert frames[0]["diagnostics"]
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), make_handler(State(workspace / "results"))
    )
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        for path in (
            "/",
            "/app.js",
            "/theme.mjs",
            "/style.css",
            "/day-night.mjs",
            "/branding/wtf-symbol-light.svg",
            "/branding/wtf-symbol-dark.svg",
            "/branding/wtf-symbol-light.png",
            "/branding/wtf-symbol-dark.png",
            "/branding/favicon.svg",
            "/branding/favicon.png",
            "/branding/wtf-symbol-motion-light.svg",
            "/branding/wtf-symbol-motion-dark.svg",
        ):
            with urlopen(base + path) as response:
                assert response.status == 200 and response.read()
                if path.endswith(".svg"):
                    assert response.headers.get_content_type() == "image/svg+xml"
                elif path.endswith(".png"):
                    assert response.headers.get_content_type() == "image/png"
        with urlopen(base + "/api/recordings") as response:
            assert any(r["id"] == folder.name for r in json.load(response))
        with urlopen(base + f"/api/recordings/{folder.name}") as response:
            assert json.load(response)["config"]["name"] == "maskable_ppo"
        # This is the viewer's ordinary decoded chunk endpoint.
        with urlopen(
            base + f"/api/recordings/{folder.name}/runs/1/chunks/0"
        ) as response:
            assert len(json.load(response)) == 6
        try:
            urlopen(base + "/../LICENSE")
        except HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError("Static asset traversal accepted")
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    from wtf.native.kernels import status

    if os.environ.get("WTF_KERNELS") == "native":
        assert status()["backend"] == "native"
    report = {
        "package": str(Path(wtf.__file__).resolve()),
        "presets": len(names),
        "frames": len(frames),
        "viewer": "HTTP assets/catalog/chunk passed",
        "checkpoint": "restored",
        "backend": status()["backend"],
    }
    (workspace / "installation-check.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
