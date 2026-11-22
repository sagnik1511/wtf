<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/branding/wtf-lockup-dark.svg" type="image/svg+xml" />
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/branding/wtf-lockup-light.svg" type="image/svg+xml" />
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/branding/wtf-lockup-dark.png" />
    <img src="docs/assets/branding/wtf-lockup-light.png" width="280" alt="WTF — Wildlife Trial Framework" />
  </picture>
</h1>

A Python reinforcement-learning ecosystem with independently learning prey and
predators, growing resources, obstacles and paired male/female reproduction. Environments are JSON;
a JavaScript canvas viewer replays saved runs. Add named prey/predator species
through JSON; the presets start with one of each.

The simulator records first and the viewer plays later, keeping training and
rendering independent. Replay chunks use structured binary deltas and lossless XZ compression to reduce disk usage.

<picture class="replay-preview">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/themed/simulation-playback-light.webp" type="image/webp" />
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/simulation-playback.gif" type="image/gif" />
  <img src="docs/assets/simulation-playback.gif" alt="Saved ecosystem playback with prey, predators, obstacles and event highlights" />
</picture>

A short replay of a saved experiment. See the [live species charts](docs/recording.md#per-step-species-dashboard)
for population, energy, births and hunting trends.

For Python and Docker setup, see [INSTALL.md](INSTALL.md).
The Python import is `wtf`; the distribution is `wtf-wildlife-trial-framework`.
See [package layout and import migration](docs/package-layout.md) for the
responsibility-based modules and the 0.0.2.0 import changes.

## Tech stack

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![JavaScript ES modules](https://img.shields.io/badge/JavaScript-ES_modules-F7DF1E?logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules)
[![HTML5 Canvas](https://img.shields.io/badge/HTML5-Canvas-E34F26?logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)
[![CSS interface](https://img.shields.io/badge/CSS-interface-663399?logo=css&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS)
[![C optional kernels](https://img.shields.io/badge/C-optional_kernels-A8B9CC?logo=c&logoColor=black)](docs/native-kernels.md)
[![JSON environment presets](https://img.shields.io/badge/JSON-environment_presets-292929?logo=json&logoColor=white)](docs/configuration.md)
[![Docker optional installation](https://img.shields.io/badge/Docker-optional-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/get-started/get-docker/)
[![Zensical documentation](https://img.shields.io/badge/Zensical-documentation-526CFE)](https://zensical.org/)

Python owns simulation, custom RL models, learning and artifact storage. Optional
CPython C kernels accelerate CPU hot paths. Plain JavaScript, HTML, CSS, Canvas
and SVG provide the replay viewer; JSON defines worlds, species and policies.
Zensical builds the documentation website.

### Software and requirements

| Software | When you need it |
| --- | --- |
| [Python 3.12+](https://www.python.org/downloads/) and [tqdm](https://tqdm.github.io/) | Required for simulation and the local playback server. `tqdm` is the only Python runtime dependency; install it through [requirements.txt](requirements.txt). |
| Modern browser with [JavaScript modules](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules), [Canvas](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API) and [SVG](https://developer.mozilla.org/en-US/docs/Web/SVG) | Required to use the viewer. |
| [Git](https://git-scm.com/downloads) | Clone the source checkout; documentation builds also use Git to filter published files. |
| [Docker](https://docs.docker.com/get-started/get-docker/) and [Compose](https://docs.docker.com/compose/install/) | Optional container-based installation. |
| [GCC](https://gcc.gnu.org/) or [Clang](https://clang.llvm.org/), Python development headers and [setuptools](https://setuptools.pypa.io/en/latest/) | Optional native CPU kernels, validated on CPython 3.12–3.13. See [native setup](INSTALL.md#optional-native-cpu-kernels). |
| [Node.js 20+](https://nodejs.org/en/download) | Optional JavaScript tests and viewer illustration tooling. |
| [Zensical](https://zensical.org/) | Optional local documentation builds; pinned versions are in [requirements-docs.txt](requirements-docs.txt). |
| [FFmpeg and ffprobe](https://ffmpeg.org/download.html) | Optional documentation GIF exports. |

The viewer runs directly in the browser without npm packages or a frontend build.
Models use the project's own implementations; NumPy, PyTorch and a GPU are not
required. [Installation](INSTALL.md) covers commands and platform details;
[development setup](docs/development.md#mandatory-formatting) covers formatter tools.

## Quick start

Requires Python 3.12+. Install the [published package](https://pypi.org/project/wtf-wildlife-trial-framework/0.0.2.9/)
in a fresh writable working directory:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install wtf-wildlife-trial-framework==0.0.2.9
.venv/bin/wtf run baseline --runs 1 --steps 200 --seed 100 --progress
.venv/bin/wtf view --results results
```

Presets, terrain maps and viewer assets are included; a checkout is optional.
For source development or Docker, follow [installation](INSTALL.md).

Open <http://127.0.0.1:8000>, select the saved experiment and click **Play recording**.
You can close the simulator before starting playback. Python packages stay in
`.venv`; the viewer needs no packages or build step.

## Keep learned brains

Completed training runs archive health, kills, lifespan and food-gain champions
for each species/sex group inside `results/<run-name>/champions/`. Models keep their scores and ancestry.
Use the longest-lived prey champions and predator hunting champions to start a new generation:

```sh
.venv/bin/python -m wtf.cli environments/baseline.json --runs 1 \
  --brains-from PREVIOUS_RUN_NAME --brain prey=lifespan --brain predator=kills
```

Replace `PREVIOUS_RUN_NAME` with a folder name under `results/` containing compatible Q champions. See the
[saved-brain guide](docs/brain-library.md) to choose another criterion, algorithm
or exact model generation. Species-wide selectors resolve matching female and male champions separately.
With multiple runs, each group’s champion seeds the next run by default; add `--no-chain-brains` to restart from the original models.

## Documentation

Browse the [documentation hub](docs/index.md) for a reading path, or use the
[GitHub Pages documentation](https://sagnik1511.github.io/wtf/) for the published guides. The site adds search, grouped navigation and **light/dark modes** to these same
guides. Use the sun/moon icon in the top-right header to switch appearance; the
initial theme follows your system preference.

| Area | Entry points |
| --- | --- |
| Get started | [Installation](INSTALL.md), [complete tutorial](docs/tutorial.md), [run/resume](docs/running.md) |
| Worlds and species | [Customization](docs/customization.md), [configuration](docs/configuration.md), [geography](docs/geography.md), [species](docs/species.md) |
| Learning and behavior | [Algorithms](docs/algorithms.md), [observations](docs/observations.md), [champions](docs/brain-library.md), [evaluation](docs/evaluation.md) |
| Viewer and storage | [Playback](docs/recording.md), [appearance](docs/viewer-visual-reference.md), [explainability](docs/explainability.md), [artifact structures](docs/replay-data-structures.md) |
| Development | [Architecture](docs/architecture.md), [testing](docs/development.md), [extensions](docs/extending-worlds-and-actions.md), [packaging](docs/packaging.md) |
| Reference | [Configuration fields](docs/configuration-reference.md), [commands](docs/command-reference.md), [visual guide](docs/visual-guide.md), [compatibility](docs/compatibility.md) |

## Project layout

```text
wtf/     Python simulation, learning, archives and analysis
viewer/        JavaScript playback engine
environments/  JSON experiment presets
maps/          Editable river, highland and valley layouts
docs/          Detailed guides
tests/         Python and JavaScript checks
```

Generated `results/`, `brain_library/`, `.cache/` and `.venv/` remain local and are
ignored by Git. Observed survival and champion scores are useful diagnostics;
checking learning quality requires matched evaluation on unseen world seeds.

## License

Copyright © 2026 Sagnik Roy. Released under the [MIT License](LICENSE).
