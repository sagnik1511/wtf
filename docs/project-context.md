# Project context for contributors and coding agents

[Project overview](../README.md) · [Agent instructions](../AGENTS.md) · [Development](development.md)

This is maintained project context, not a transcript or experiment log. Verify
current defaults in source and the chosen JSON preset before describing a run.
Local results and running processes change independently of Git.

For diagrams matched to common questions, start with the
[visual guide](visual-guide.md). The [physics gallery](physics.md) and
[recorded terrain previews](geography.md#recorded-terrain-previews) show spatial
behavior and actual playback.

## Purpose and decisions

WTF — Wildlife Trial Framework — explores independently learning prey and
predators in a grid world with food, obstacles, energy, health, death and
paired male/female reproduction. Both parents must choose `reproduce` in the same
step and satisfy physical eligibility. [Reward accounting](rewards.md) covers
energy-demand penalties, failed choices and successful-parent feedback. It is an experimental ecosystem, not a calibrated biological model.
The priorities established during development are:

- Configurable experiments as JSON, with Python simulation and JavaScript rendering.
- Optional editable geography maps add swim-only rivers, fractional currents
  and elevation costs. Flat brains use 20 features; terrain version 2 uses 32.
  [Geography guide](geography.md) covers physics and saved-model compatibility.
  Landscape presets include meandering rivers, a branching delta, alpine passes
  and lake islands, with editable map JSON and generated static diagrams.
- Arbitrary named species grouped into resource/prey diets; presets keep one
  species of each. [Species guide](species.md) covers adding more and display colors.
- Record first, play later: training can finish and exit before opening the viewer.
- Private animal brains, inherited learned weights, run-local species/sex champion archives and
  optional matching-sex champion chaining between runs.
- Small disk footprints and bounded replay memory, preserving viewer features and
  numerical information. Binary replay data need not be human-readable; metrics do.
- Charts that follow the current replay step, with distinct whole-run analysis.
- Concise README with detailed guides and four-component rollout records under docs.

There are no intended universal prey/predator health, learning-rate, reproduction
or champion-selection settings: these vary by environment. Inspect the selected
preset and metadata captured with the experiment.

## Code navigation

| Area | Entry points | Read when changing |
| --- | --- | --- |
| Physical world, birth/death | `ecosystem/world.py`, `mutation.py` | [Architecture](architecture.md), [mutation](mutation.md) |
| Current observations/caches | `ecosystem/perception.py`, visibility helpers in `world.py` | [Performance](performance.md) |
| Experiments and configuration | `ecosystem/cli.py`, `config.py`, `runner.py`, `environments/`, schema | [Running](running.md), [configuration](configuration.md) |
| Learning | `brains.py`, `brain.py`, `ppo.py`, `policy_gradients.py`, `grouped_rollouts.py`, `recurrent_ppo.py`, `qrdqn.py`, `contextual_ppo.py` | [Algorithms](algorithms.md), [advanced variants](advanced-algorithms.md) |
| Optional native compute | `ecosystem/_kernels.c`, `_kernels_*.h`, `kernels.py`, `work/build_native.py` | [CPU kernels](native-kernels.md) |
| Brain inheritance/archive | `ecosystem/library.py`, `runner.py` | [Saved brains](brain-library.md) |
| Replay/checkpoint formats | `recording.py`, `replay_codec.py`, `replay_io.py`, `binary_io.py`, `checkpoint_io.py` in `ecosystem/` | [Binary storage](binary-storage.md), [recordings](recording.md) |
| Read-only playback | `ecosystem/server.py`, `viewer/api.mjs`, `playback.mjs`, `binary-codec.mjs`, `app.js` | [Recordings](recording.md) |
| Metrics and interpretation | `ecosystem/telemetry.py`, `analyze.py`, viewer dashboard modules | [Analysis](analysis.md) |

## Semantics worth preserving

World actions resolve together; death terminates an animal's experience, while
an episode time limit is a truncation. Different algorithms have different
bootstrap rules. Evaluation freezes learning and newborn mutation. GRPO trial
worlds must not change real-world controllers, RNGs or event counts. Mutated
newborns get independent mixed weights. All newborn training state resets after
paired crossover; the mutation guide describes the additional rare changes.
None of these features imply a shared species policy.

A fresh invocation, a warm start and exact resume are different operations:

- A fresh start initializes controllers using the selected config/CLI. To ensure
  no saved weights are used, inspect `initial_brain` and use species `fresh`
  selectors as appropriate.
- `--brains-from RUN_NAME` selects archived brains; `--brain SPECIES[:SEX]=CRITERION`
  chooses the group/metric and `--brain-run N` chooses a numbered archived run.
- `--no-chain-brains` returns each run to its original initialization. Chaining
  normally uses the previous run's matching species/sex champions; verify config/CLI overrides.
- `--resume CHECKPOINT` restores full saved state, unlike loading champion weights.

Binary chunks use independent keyframes plus deltas and compression. The server
reads requested chunks; the viewer retains one decoded world chunk plus bounded
supporting data. Do not decompress a whole experiment at startup. Legacy replay
and checkpoint readers remain necessary. Dynamic observation caches must account
for moved/dead/newborn animals and changed food, even when the observer stays put.

Optional `placement` selects founder inside/outside zones and exact boundary-food
quotas; it never confines later movement or resamples checkpoint bodies. See the
[layout guide](geography.md#enclosed-valley-starting-layout) and
`ecosystem/placement.py`. Existing configs without it preserve seeded layouts.

Optional `body.water_vision_bonus` extends only rays crossing water, with wall
occlusion intact. Cache static ray counts per world and key dynamic results by
both base radius and bonus; restore caches cold. See
[directional sight](geography.md#directional-sight-across-water).

[Physics illustrated](physics.md) links the tracked SVG cases and portable
interactive explorer under `docs/assets/physics/`. Refresh their controlled
physics snapshots with `python -m work.build_physics_docs`; they are not run
replays or evidence of learning.

New experiments use packed float32 model/optimizer storage with double arithmetic
and unchanged world precision. Optional [CPU kernels](native-kernels.md) accelerate
dense/PPO/GRU/quantile math, perception, flat-world phases and replay/dashboard
helpers, with a selectable Python fallback and no format change. The
[second report](native-kernels-v2.md) records strict type/arithmetic/cache boundaries
and native-to-native benchmarks; terrain-specific body physics stays Python. Checkpoints without precision metadata retain
float64 continuation, including future births/episodes. New warm starts convert
compatible saved weights to the new run's precision. See
[numeric precision](running.md#numeric-precision). Binary codecs narrow numeric
arrays only losslessly and preserve arbitrary counters/seeds; update Python and
JS readers together when opening new binary array tags.

## Local operations and common pitfalls

Commands run from the repository root. Use repo-relative paths in maintained docs;
when an absolute checkout example is needed, use `/path/to/wtf`. Start playback with
`./run.sh`; the default URL is `http://127.0.0.1:8000`. A catalog entry plus chunk
404 can indicate an older Python process that cannot read new binary chunks.
Current servers advertise `X-WTF-Replay-Protocol: binary-v1`; check files and API
responses before diagnosing corruption. Restart the relevant server and refresh
when updating readers. Do not assume a port or PID from previous sessions.

`results/`, `brain_library/`, `.cache/`, `.venv/` and generated benchmark outputs
are local/ignored. Read committed metadata before analyzing an active run; only
published chunks are available. Keep tests and temporary reports out of tracked
source. Intentional screenshots/GIFs belong in `docs/assets/`, with provenance.

## Reusable workflows and upkeep

[AGENTS.md](../AGENTS.md#repo-local-skills) links the repository skills. Invoke
`$wtf-analyze-run`, `$wtf-optimize` or `$wtf-release-docs` in a session that exposes
these skills, or ask the agent to read the corresponding `SKILL.md` directly.
They are stored in the repo, not installed into a personal global skill folder.
Each skill keeps its own instructions and UI metadata together:

```text
AGENTS.md                         Shared agent instructions
.agents/skills/
  wtf-analyze-run/
    SKILL.md                      Saved-run analysis workflow
    agents/openai.yaml            Skill display name and invocation prompt
  wtf-optimize/
    SKILL.md                      Compute, cache and storage workflow
    agents/openai.yaml
  wtf-release-docs/
    SKILL.md                      Versioned rollout workflow
    agents/openai.yaml
docs/project-context.md           Stable decisions and code map
```

Add scripts, references or assets inside an individual skill only when its
workflow needs them. The current skills link existing guides rather than copying
them. Skill-creator validation used PyYAML in the ignored `.cache/skill-tools/`
virtual environment; this is optional authoring tooling, not a runtime package.

The built-in `skill-creator` skill was used to author these workflows. Earlier
media excerpts used FFmpeg directly; no separate image-generation or video skill
was used. Tool availability and skill discovery depend on the coding environment.

Keep this file for stable decisions, the guides for detailed behavior, release
records for implementation history and ignored result folders for measurements.
Update affected links and workflow instructions when commands/formats change.
