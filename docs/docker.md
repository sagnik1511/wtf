# Docker training and playback

[Installation](../INSTALL.md) · [Setup](running.md) · [Python examples](../examples/README.md) · [Compatibility](compatibility.md)

The [Dockerfile](../Dockerfile) builds one image with Python, maintained presets,
optional native CPU kernels, viewer assets and examples. Training and playback
are separate commands. No Node build, GPU runtime or database is required.

## Use the launcher

Install Docker with Compose support. From the checkout:

```sh
sh docker/wtf.sh build
sh docker/wtf.sh train
sh docker/wtf.sh view
```

The training command finishes before playback starts. Open
<http://127.0.0.1:8000>. The viewer runs in the foreground; stop it with Ctrl+C.
Use `sh docker/wtf.sh view 8001` if port 8000 is occupied, or
`sh docker/wtf.sh down` to remove the Compose containers/network. Results remain
in the checkout's ignored `results/` directory. The launcher resolves the
checkout path and maps the calling user's UID/GID for bind-mount ownership.

Custom training arguments replace the default preset/run settings:

```sh
sh docker/wtf.sh train environments/day_night.json --runs 1 --seed 100 --progress
sh docker/wtf.sh train environments/maskable_ppo.json --runs 5 --seed 100
```

The launcher always appends `--output /data/results`, so artifacts remain on the
host. Container paths for old runs use `/data/results/EXPERIMENT`, for example
`--brains-from /data/results/EXPERIMENT`. The image contains maintained presets;
after editing a bundled preset, rebuild the image. For local JSONs, put them in
`results/configs/` and pass `/data/results/configs/my_world.json`, or add an
explicit read-only config mount in a local Compose override.

## Image and Compose details

| Component | Behavior |
| --- | --- |
| Builder | Python 3.11 slim plus a compiler; builds kernels inside Linux, never copies host `.so` files |
| Runtime | Matching Python image, tqdm, non-root user, sources and runtime assets; no compiler installation |
| `train` service | Opt-in Compose profile, writable `./results:/data/results`, CLI entry point |
| `viewer` service | Read-only result mount and root filesystem, temporary `/tmp`, HTTP health check |
| Networking | Server binds `0.0.0.0` inside the container; published port binds only `127.0.0.1` on the host |
| Build context | Default-deny `.dockerignore`; excludes local experiments, weights, caches, credentials, paper and host native binaries |

Native kernels are on by default. Set `WTF_BUILD_NATIVE=0` when building to use
the Python fallback and skip compiler/build-tool installation. The runtime
never contains the compiler toolchain. Python's patch image and dependency ranges resolve at build
time; preserve the built image digest for reproducibility.

Direct Compose equivalents:

```sh
docker compose build
docker compose run --rm train
docker compose up viewer
```

Direct Compose defaults UID/GID to 1000; customize `WTF_UID` and `WTF_GID` when
needed, or use the launcher. `WTF_PORT` selects the published viewer port.
An ordinary `docker compose up` starts only the viewer, not a training campaign.
The image's default entry point is training; playback uses
`python -m ecosystem.server --host 0.0.0.0 --results /data/results`.

## Validation scope

The files and launcher are provided without building an image or starting a
container. Shell syntax, configuration structure, server bind-address parsing
and local example behavior are checked separately. A real Docker build and
container networking/volume smoke test remain release checks for a Docker host.
