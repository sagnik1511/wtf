# Package and saved-artifact compatibility

[Saved brains](brain-library.md) · [Checkpoint structure](checkpoint-data-structures.md) · [Replay structure](replay-data-structures.md)

Package releases, internal rollout numbers and file-format versions describe
different contracts. Package version [0.0.2.9 is published on PyPI](https://pypi.org/project/wtf-wildlife-trial-framework/0.0.2.9/)
with a portable wheel and source archive. The current checkout prepares
`0.1.0.0`; this does not establish a PyPI upload.
Four-component records under `docs/releases/` track implementation rollouts,
not importable-package versions. Binary magic, checkpoint envelopes, logical
state and observation versions each have their own version identifiers.

## Optional training capture and frozen evaluation

Package 0.0.2.9 supports `recording.profile` and evaluation `--freeze-brains`;
they are absent from the older 0.0.2.0 package. Replay/champion encodings are
unchanged. Old checkpoints restore with the freeze policy disabled; new logical
checkpoints retain it. Use feature-aware source when restoring profile-configured
checkpoints: older strict configuration validators reject `recording.profile`,
and older runtimes do not preserve the frozen-birth policy. Training-only outputs
contain no replay to convert; record a fresh evaluation for visualization.
See [training and inference](running.md#train-without-replay-then-record-inference).

## Scheduled-tail recording in current source

`--record-last-episodes` / `recording.record_last_episodes` is a source feature
after published package 0.0.2.9. Older CLIs reject the flag and older strict
configuration readers reject checkpoints containing the optional field. Existing
checkpoints without it default to recording all episodes. Replay v1 and chunk
encodings are unchanged: recorded episodes keep their original numbers, and
existing viewer readers already support that subset. No older replay is deleted.
See [selection rules](recording.md#record-only-the-final-scheduled-episodes).

## Persistent consequence totals and private auxiliary heads

Current source optionally reads logical consequence state v3, wrapping v1/v2
context tables with fixed-size published/staged species/action totals. Legacy
imports seed totals only from surviving rows; lost evidence cannot be recovered.
Restoring v3 requires `persistent_action_totals: true`. Omission of this setting
keeps legacy table exports and behavior. This logical version is separate from
the experiment checkpoint envelope version and binary codec magic.

PPO/masked-PPO `auxiliary_consequence` adds a versioned signature field and
`auxiliary_parameters` in champion models; recovery state also stores private
head optimizer arrays/counters. The actor's input/output layout is unchanged.
Champions with and without the head are not silently interchangeable. Existing
headless artifacts retain their readers. Use feature-aware source for the new
configuration/state; published package 0.0.2.9 predates it. Replay frame formats
remain compatible; optional diagnostic scalar fields may be ignored by old UI.
See [collective learning](social-learning.md#persistent-speciesaction-totals).

## Preset sensory promotion

Maintained run presets now select V7 compact observations with three-step visual
history and blended scent. Their previous V1–V6/scent-free brain signatures do
not match the new defaults. Start fresh, load matching V7 weights, or run the
original captured environment for older warm starts. Exact checkpoint resumes
and existing playback use their saved configuration; no artifacts are converted.
The minimal baseline, archived-model example and named V5/V6 controls remain
available. See [the migration guide](configuration.md#v7-sensory-preset-defaults).

## Historical sensory contracts

<a id="observation-v6"></a>
<a id="observation-v5"></a>

Readers preserve earlier numeric/soft layouts and their exact saved signatures.
Those layouts, older inspector sheets and settings are documented in
[historical reference](archive/index.md), not presented as current defaults.
Current source models use the [V7 contract](learning/v7-observations.md).

## Optional forward vision

`species.<name>.body.field_of_view_degrees` accepts 150, 180 or 360; omission keeps
legacy 360° sight. No observation width, policy signature or per-frame replay
field is added. The body setting is saved once in experiment/replay metadata;
use the current viewer for the correct forward-field overlay. Older strict source
or package validators reject the new configuration field. Existing recordings
without it retain the original appearance and full visual range rules.

The field changes perception, not physical movement or mating. Compatible policy
weights may be reused, but performance under 360° does not establish performance
under 150° or 180°. Consequence knowledge uses an exact physical/body contract and
rejects a mismatching visual-field setting. See [observation semantics](observations.md#forward-visual-field).

## Python runtime

The `ecosystem` → `wtf` import migration changes Python paths, not artifact formats. See the [complete import map](package-layout.md#migrate-existing-imports); old namespace aliases are not shipped.

The minimum is Python 3.12. Native CPU kernels support CPython 3.12–3.13 and must
be rebuilt for each interpreter/OS/architecture. Newer Python versions use the
reference backend until native equivalence is validated. Wheel metadata rejects
older Python installations; source-checkout imports also report the minimum.

Artifact formats and model signatures are unchanged by this runtime migration.
Existing supported checkpoints/brains/replays retain their readers. Moving a
checkpoint from Python 3.9–3.11 to 3.12+ can change its future numerical trajectory:
Python changed float `sum()` in 3.12. Native and reference calculations match
within each validated interpreter; this does not promise identical results
across Python versions. Keep the old interpreter and source revision when exact
historical continuation is required.

## Current reader capabilities

This table describes the current source readers, not a historical published
package matrix. The published 0.0.2.9 snapshot predates V7, private auxiliary/late
heads and other subsequently added source features. Its earlier installation
smoke check does not establish support for those additions. Use a current
source installation or a separately validated newer artifact for this guide.
Platform coverage remains limited to the runtimes actually tested. See
[package verification](packaging.md).

| Artifact | Current source can read | Compatibility boundary |
| --- | --- | --- |
| Experiment checkpoint | Legacy logical state v1 JSON; compact `wtf-checkpoint-v2` envelope; binary checkpoint envelopes v3 (standalone) and v4 (optional shared metadata), inside supported plain/gzip/XZ transport | Decoders normalize to logical state v1; controllers/configuration must still be supported |
| Champion brain | `brain_artifact_version` 1, 2 and 3; library manifests 1 and 2 | v3 metadata references binary/XZ weights; older readers do not necessarily support it |
| Replay | `recording_version: 1`, legacy full JSON chunks, `wtf-delta-v1` chunks and current structured binary chunks, with supported none/gzip/XZ compression | Current binary magic is `WTFB` codec v1; keep the recording manifest, chunks and run metadata together |
| Brain inputs | Supported observation versions 1–7 with configured top-k | Algorithm, architecture, action order, observation version/top-k, optional movement/scent layout and other signature fields must match |

Use the loader appropriate to the artifact. A checkpoint restores world,
controllers, optimizer state, buffers and RNGs. A champion initializes weights
in a fresh world and starts fresh training state. A replay is for visualization
and analysis; it cannot restore training or recover unsaved activations.
Species/sex champion compatibility rules apply independently of file format.

For example, restore a checkpoint through the normal decoder:

```python
from wtf.storage.checkpoint_io import read_checkpoint
from wtf.experiment.runner import Experiment

experiment = Experiment.restore(read_checkpoint("results/EXPERIMENT/checkpoints/final.bin.xz"))
```

A final checkpoint may already be complete. Restore an unfinished checkpoint to
continue its remaining schedule; champion warm starts use `--brains-from` instead.
Do not rename a champion or replay file to make it act as a checkpoint.

## Release policy

- Each published package release must state which artifact versions it reads and
  writes. Adding a reader does not mean an older package can read new output.
- Patch releases should retain supported artifact readers and public APIs. A
  required incompatibility must be explicit in release notes; never silently
  reinterpret unknown fields/versions or substitute an incompatible model.
- Before `1.0`, minor releases may change APIs or simulation semantics. Document
  changed signatures/formats and a conversion path when one exists. Historical
  rollout numbers alone are not evidence of cross-version compatibility.
- Test representative legacy fixtures, current write/read round trips, complete
  checkpoint continuation and Python/JavaScript replay decoding before release.
  Do not label a historical package/artifact combination supported without tests.
- Loading successfully promises structural compatibility, not identical future
  trajectories. Exact continuation is strongest with the same source revision,
  Python/runtime, numeric backend and dependencies. Changed physics or floating
  arithmetic can change trajectories even when the file loads correctly.

Record the package/source revision, Python version, backend and original JSON
with experiments you intend to reproduce. Preserve original files before using
storage conversion tools. A new viewer can display only information actually
recorded; older recordings without activation, sex or event-position fields
retain their documented missing-data behavior.

## Moving artifacts between machines or Docker

Copy the whole experiment directory, including relative champion model pools and
metadata. Do not copy only an index or remove referenced model blobs. Results
mounted under `/data/results` in Docker retain their internal relative links.
CPU numeric weights and replay files are portable data; compiled `.so`/`.pyd`
kernels are platform/Python-specific and must be rebuilt or omitted on the
receiving machine. Use the Python fallback when the native backend is unsupported.

Optional [tracking senses](tracking.md) append inputs without changing base
observation versions. Enabled checkpoints require sensory state version 1 and a
runtime supporting those senses; older releases cannot resume these runs. Legacy
checkpoints without the options retain their existing restore path. Replays use
the current binary format; updated inspectors decode the optional input blocks.

Blended scent uses an explicit `scent_observation.mode: "blended"` signature
and nine inputs for all animals. Omitted mode retains the original separated
species/sex layout and canonical signature; old brains, exact checkpoints and
recordings retain their original behavior. Switching scent mode requires a
compatible brain, even when both layouts happen to have nine inputs. The
maintained tracking preset now selects blended mode; use an old run's recorded
configuration to resume its separated-scent checkpoint.

Binary checkpoint header version 4 requires its run-local `checkpoints/_metadata/`
pool and a runtime with shared-metadata support. Version 3 and older checkpoint
loaders remain supported by new runtimes. See [retention and portability](artifact-retention.md#common-metadata-and-portability). Replay formats and saved model signatures are unchanged.


## Discrete SAC, TRPO and LSTM PPO

Current source adds distinct `discrete_sac`, `trpo` and `lstm_ppo` model signatures.
They require readers that dispatch these identifiers; published 0.0.2.9 cannot
restore them. Existing algorithms and artifact formats remain unchanged. SAC
champions include both online critics; exact recovery also restores target
critics and replay. TRPO stores separate actor/value networks and accepted-policy
counters. LSTM recovery includes both hidden/cell memory and pending sequences;
GRU signatures are incompatible. See [complete policy contracts](additional-policies.md).

## Optional collective consequence artifacts

Current source adds an independent version-1 `consequences.json.xz` aggregate
export. Import validates exact species physical contracts and the prediction
horizon; it supplies no policy weights or optimizer state. Exact recovery instead
uses the single `world.social_state` checkpoint record, including bounded local
and pending histories. These optional input contracts require an implementation
supporting collective learning; older package versions cannot restore a new
configuration with unknown fields, despite unchanged base checkpoint/replay
container versions. Existing configurations/readers retain legacy behavior.
See [collective learning](social-learning.md).

## Experimental CUDA backend

The [optional source CUDA backend](cuda.md) changes execution, not artifact formats.
CPU remains the default; restored checkpoints start on CPU until CUDA is selected
again. Device buffers are transient. GPU math can round differently, so trajectories
need not match CPU bit-for-bit. CUDA compilation/execution and speedups require
Linux/NVIDIA validation; this prototype is not a published GPU-support guarantee.

The revised CUDA source uses kernel ABI 3; rebuild local shared libraries after
pulling. Optional `--cuda-precision fp32` changes GPU arithmetic, while model and
optimizer storage remains FP32 as before. Backend choices remain transient;
resume explicitly selects CUDA/precision again. Ordinary/masked PPO now publish
device-trained weights and Adam state at the completed rollout-update boundary.

DQN/DDQN CUDA updates now publish online/target weights and private Adam state
before returning to world consumers. `--cuda-optimizer cpu` keeps the host-optimizer
comparison path; this transient option is not serialized. Rebuild native CPU kernels
to enable the exact bulk FP32 gradient-widening helper. Existing CPU extensions
without the helper retain the previous conversion fallback.


## Compact V7 models

Observation V7 has distinct numeric summaries, an ordered species-action contract,
and optional late-consequence adapter metadata. Changing version, action order,
summary settings or enabling the adapter requires matching weights. V1–V6
readers and saved artifacts remain supported. V7 champions retain the adapter's
weights; checkpoints additionally retain its private Adam moments/counter and
unchanged sampled evidence. No legacy replay or checkpoint migration is needed.


## Native episode backend

The [C++ CPU engine](native-engine.md) uses the existing logical checkpoint,
model-signature, metrics and replay contracts. Current source covers all 14
built-in CPU learners, V1–V7 sensing, terrain/day-night/contests, configured
social/consequence knowledge, and the auxiliary/late heads supported by
ordinary/masked PPO. Native genesis and next-run chaining preserve the same
seed, founder, inheritance, champion and publication contracts. The 0.1.0.0
release is unpublished; local validation and workflow benchmarks are documented.

| Boundary | Compatibility rule |
| --- | --- |
| Stored model/optimizer precision | Native live controllers require FP32 storage and retain FP64 arithmetic; exact legacy FP64-stored continuation uses Python |
| Policy/input contract | Existing signature checks still apply; a backend switch does not make another algorithm, layout or action order compatible |
| Recovery/brains/replay | Existing formats/readers remain authoritative; no migration or discarded playback data is required |
| Native library | Rebuild for the interpreter ABI, operating system and architecture; never copy a host `.so`/`.pyd` to another machine |
| Custom Python hooks | Execute with the Python engine; C++ cannot honor Python subclass or monkeypatch overrides inside its loop |
| CUDA | Use the Python engine; the standalone C++ episode backend is CPU-only |

Fresh native construction and episode chaining do not construct Python worlds or
learners. Explicit query/import/restore compatibility and batch artifact
projections still use Python. Detached query edits cannot mutate the active
native owner. Existing Python readers can inspect or resume its logical
checkpoints without a compiler.

Exact differential checks on a tested interpreter/platform are required for
execution parity. They do not establish universal cross-platform floating-point
agreement across libm implementations or compiler versions. Python remains the
portable default and fallback for unsupported execution contracts.
