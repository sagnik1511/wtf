# Project context for contributors and coding agents

[Project overview](../README.md) · [Agent instructions](../AGENTS.md) · [Development](development.md)

This is maintained project context, not a transcript or experiment log. Verify
current defaults in source and the chosen JSON preset before describing a run.
Local results and running processes change independently of Git.

For diagrams matched to common questions, start with the
[visual guide](visual-guide.md). The [physics gallery](physics.md) and
[recorded terrain previews](geography.md#recorded-terrain-previews) show spatial
behavior and actual playback.

## Customizing or extending the setup

The Python namespace is `wtf`, organized by [package ownership](package-layout.md). Use `from wtf import Experiment, load_config` for the small public API; implementations live under `simulation`, `learning`, `experiment`, `storage`, `inspection` and `native`.

Start with the [customization handbook](customization.md) and [command definitions](command-reference.md) to distinguish JSON,
CLI, viewer preferences and source-code extensions. The
[complete field reference](configuration-reference.md) defines every schema
field, allowed range/choice, required status and declared default, including
all algorithm variants and terrain symbols. Meanings live in schema descriptions;
regenerate with `python -m work.build_config_reference` and check with `--check`.
New settings must update both Python validation and schema definitions.

For more species, follow [the complete recipe](species.md); arbitrary names use
existing diets, exact species identity controls mating/archives and live state
stays independent. For new algorithms, follow [the policy integration checklist](extending-policies.md):
there is no dynamic policy registry today. Include creation/restore dispatch,
signatures, champion I/O, child-state reset/mutation, diagnostics, archive
explanations, viewer semantics and cross-algorithm test fixtures. Documentation
examples of an unregistered type are contributor recipes, not supported presets.
Do not infer support from a schema enum or successful constructor alone.
The [world/action extension guide](extending-worlds-and-actions.md) separates
new map symbols and action subsets from genuinely new physics/actions, including
masks, native fallbacks, reward events, replay and viewer integration.

## Purpose and decisions

WTF — Wildlife Trial Framework — explores independently learning prey and
predators in a grid world with food, obstacles, energy, health, death and
paired male/female reproduction. Both parents must consent through `reproduce`, either in the same step or within
a configured resting intent window, and satisfy physical eligibility. [Reward accounting](rewards.md) covers
energy-demand penalties, failed choices and successful-parent feedback. It is an experimental ecosystem, not a calibrated biological model.
The priorities established during development are:

- Configurable experiments as JSON, with Python simulation and JavaScript rendering.
- Optional editable geography maps add swim-only rivers, fractional currents
  and elevation costs. Flat brains use 20 features; terrain version 2 uses 32.
  Optional version 3 uses 42 inputs including local mating readiness; the
  meandering-river preset opts in. [Mating observations](mating-observations.md)
  describes compatibility and guarded lineage promotion. Configurable
  [top-k observations](observations.md) add 12 inputs per additional target slot;
  defaults remain one target per category.
  [Geography guide](geography.md) covers physics and saved-model compatibility.
  Landscape presets include meandering rivers, a branching delta, alpine passes
  and lake islands, with editable map JSON and generated static diagrams.
- Arbitrary named species grouped into resource/prey diets; presets keep one
  species of each. [Species guide](species.md) covers adding more and display colors.
- Optional [day–night cycles](day-night.md) change sunlight, species-specific vision
  and energy costs; playback derives matching lighting from saved ticks.
- Record first, play later: training can finish and exit before opening the viewer.
- Private animal brains, inherited learned weights, run-local species/sex champion archives and
  optional matching-sex champion chaining between runs. Fresh experiments retain
  best-so-far ranks by default (`promote_only_improved`); binary/XZ champion
  weights are deduplicated in the experiment's shared `champions/models/` pool.
  Optional [early stopping](running.md#stop-on-performance-plateaus) watches selected
  champion ranks for configurable species; its patience survives exact resume.
  Optional `lineage` curation tracks unique founder-family descendants and
  proven biological generations; its compact counters persist in checkpoints
  but reset between independently spawned episodes. See [saved brains](brain-library.md#select-successful-biological-lineages).
- Small disk footprints and bounded replay memory, preserving viewer features and
  numerical information. Binary replay data need not be human-readable; metrics do.
- Charts that follow the current replay step, with distinct whole-run analysis.
- Concise README with detailed guides and four-component rollout records under docs.

There are no intended universal prey/predator health, learning-rate, reproduction
or champion-selection settings: these vary by environment. Inspect the selected
preset and metadata captured with the experiment.

## Code navigation

[Source call and caller graphs](source-graphs.md) provide a generated, searchable
reference for Python, JavaScript and native components. Build with
`python -m work.build_source_docs`; keep the generated source reference ignored.
Read the static-analysis limitations before inferring runtime dependencies.


| Area | Entry points | Read when changing |
| --- | --- | --- |
| Physical world, birth/death | `wtf/simulation/world.py`, `mutation.py` | [Architecture](architecture.md), [mutation](mutation.md) |
| Current observations/caches | `wtf/simulation/perception.py`, visibility helpers in `world.py` | [Performance](performance.md) |
| Experiments and configuration | `wtf/cli.py`, `config.py`, `runner.py`, `environments/`, schema | [Running](running.md), [configuration](configuration.md) |
| Learning | `brains.py`, `brain.py`, `ppo.py`, `policy_gradients.py`, `grouped_rollouts.py`, `recurrent_ppo.py`, `qrdqn.py`, `contextual_ppo.py` | [Algorithms](algorithms.md), [advanced variants](advanced-algorithms.md) |
| Optional native compute | `wtf/native/_kernels.c`, `_kernels_*.h`, `kernels.py`, `work/build_native.py` | [CPU kernels](native-kernels.md) |
| Brain inheritance/archive | `wtf/storage/library.py`, `runner.py` | [Saved brains](brain-library.md) |
| Replay/checkpoint formats | `recording.py`, `replay_codec.py`, `replay_io.py`, `binary_io.py`, `checkpoint_io.py` in `wtf/` | [Binary storage](binary-storage.md), [recordings](recording.md) |
| Read-only playback | `wtf/server.py`, `viewer/api.mjs`, `playback.mjs`, `binary-codec.mjs`, `app.js` | [Recordings](recording.md) |
| Actual sampled animal activity | `activation_diagnostics.py`, per-brain diagnostics, `viewer/brain-activations.mjs` | [Recorded animal activations](explainability.md#recorded-animal-activations) |
| Archived-model explanation | `wtf/inspection/explanations.py`, `explainability.py`, `viewer/explainability.mjs` | [Brain explanations](explainability.md) |
| Metrics and interpretation | `wtf/inspection/telemetry.py`, `analyze.py`, viewer dashboard modules | [Analysis](analysis.md) |

The [feedback review](feedback-review.md) describes the shared physical reward
contract across environments and algorithms. Keep reward coefficients in species
JSON and avoid map-name/algorithm branches in environmental feedback; MAPPO's
documented species-mean adapter belongs in the learner delivery path.

## Semantics worth preserving

Optional [same-species competition](competition.md) is initiated by `challenge`
and stays disabled in existing presets. Contests finish within a step, use
ordinary damage/cost feedback and explicit displacement events, and do not grant
predation kills or kill energy. Keep failed-action and hunting metrics distinct.


World actions resolve together; death terminates an animal's experience, while
an episode time limit is a truncation. Different algorithms have different
bootstrap rules. Evaluation freezes learning and newborn mutation. GRPO trial
worlds must not change real-world controllers, RNGs or event counts. Mutated
newborns get independent mixed weights. All newborn training state resets after
paired crossover; the mutation guide describes the additional rare changes.
None of these features imply a shared species policy. See
[trial resets and independent inference batches](training-workflow-performance.md)
for group-private state, frozen model qualification, scalar fallbacks and native
workspace ownership; batching must never merge weights or action RNG streams.

Founder initialization can reuse a pristine species/sex template only while
constructing the run. Every animal must retain detached model/training state and
its original individual RNG seed; templates are discarded before learning.
Completed-step reporting may pass metrics explicitly between consumers while the
physical state is unchanged. Standalone reads stay fresh; neither observation
nor reporting reuse may assume that an unchanged tick means unchanged state.
See [initialization and reporting ownership](performance.md#founder-initialization-and-shared-reporting)
and the [phase-local mating context](performance.md#decision-phase-mating-data-and-birth-availability).

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
`wtf/simulation/placement.py`. Existing configs without it preserve seeded layouts.

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
them. Optional skill-authoring dependencies belong in ignored tooling environments.

Keep this file for stable decisions, the guides for detailed behavior, release
records for implementation history and ignored result folders for measurements.
Update affected links and workflow instructions when commands/formats change.

## Reproduction and consent

The [mating-intent guide](mating-intent.md) describes consent windows, physical
gates, failure reasons and legacy defaults. [Escape and consent](escape-and-consent.md)
covers backward/sideways walking and observation version 4. Version 3 retains
mating readiness without the added consent inputs; action and input signatures
must match archived models.

## Documentation ownership

Package guides describe supported mechanics, APIs, configuration and workflows.
Release records describe adopted software/configuration changes and their tests.
Experiment identifiers, observations and campaign histories belong in ignored
results or the separate paper workspace when relevant to the manuscript.
The [calibration guide](calibration.md) is a current recommendation reference:
update supported settings and tradeoffs after evaluation, without adding a trial
log. Running or analyzing an experiment alone does not create a release.

## Python distribution boundary

`pyproject.toml` and `setup.py` build an `wtf` package and its responsibility-based subpackages;
`packaging_assets.json` whitelists maintained presets/maps/viewer/schema data.
`resources.py` resolves installed assets via importlib.resources with a checkout
fallback. Writable results and library paths belong to the current working
directory or explicit CLI arguments, never the package installation. Preserve
`wtf` commands and isolated [installation checks](packaging.md) when extending
the setup; [Ubuntu coverage](linux.md) must distinguish configured CI from passes.

## Optional tracking senses

[Visual memory and ground scent](tracking.md) are independent opt-ins. Per-brain
`movement_history_steps` retains only personally selected sightings, with
identity matching, missing-step masks and fading last-seen estimates. Root
`scent_trails` owns sparse merged cell/species/sex deposits; per-brain
`scent_observation` chooses a blended all-animal field or legacy ordered channels,
plus a dry-path smell radius. Omitted mode preserves separated saved-brain
signatures; the tracking preset explicitly selects blended inputs. Neither
sense provides individual scent targets or hidden world trajectories. Append
layouts participate in model signatures and inspector/XAI labels. Preserve
versioned `world.sensory_state` for exact resume and independent GRPO trials;
record only the existing sampled input vectors, not full scent maps.

The viewer's movement heatmap follows replay time using one reusable `Uint32Array`
and prior positions. Normal steps and same-chunk rewinds reuse loaded frames;
large jumps call the read-only movement-prefix endpoint, which streams saved
chunks and retains no per-frame grids. Keep prefix reconstruction and full-run
CLI analysis movement rules shared, and keep future frames out of the display.
