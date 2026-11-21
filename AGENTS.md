# Working on WTF

WTF means **Wildlife Trial Framework**. Start with
[project context](docs/project-context.md); read the relevant guide linked there
rather than loading every document or result into context.

## Project boundaries

- Python owns simulation and learning; plain JavaScript owns saved-run playback.
  Keep training independent of the read-only viewer/server.
- Environments belong in `environments/*.json`, validated by `ecosystem/config.py`
  and `environment.schema.json`. Prefer CLI selectors for run-specific brain
  initialization; avoid hardcoding a result path into Python.
- Species names are arbitrary configuration keys. Use diet for prey/predator
  behavior and exact species identity for mating, metrics and champions; avoid
  hardcoded `prey`/`predator` names in generic paths.
- Animals learn independently. Birth requires an eligible same-species male/female
  pair consenting through `reproduce` (same-step by default, or overlapping configured
  mating-intent windows while resting) and mixes their parameters into an independent child with fresh training
  state. Rare birth mutation changes only that child. Preserve RNG isolation
  and exact checkpoint continuation when changing these paths. Champion pools,
  model loading and run-to-run chaining are species/sex-specific; never substitute
  a known opposite-sex winner for an unavailable group. Fresh experiments retain
  best-so-far ranks and only replace a champion on strict improvement, including
  existing tie-breakers. Keep old-checkpoint resume behavior explicit. Champion
  metadata is JSON; version-3 weights use a shared content-addressed binary/XZ
  pool. Preserve legacy artifact loading and relative-path portability.
- Terrain is opt-in. Preserve flat observation version 1 (20 inputs); terrain
  version 2 has 32 inputs and a distinct model contract. Optional version 3 adds
  mating readiness for 42 inputs, with terrain padding on flat maps. These sizes
  assume default `observation_top_k: 1`; each extra slot adds 12 inputs. Include k
  in model compatibility and preserve nearest-only legacy signatures. Version 4
  appends three local consent signals (45 inputs at k=1); preserve versions 1–3. Preserve
  RNG-free observation/mask eligibility and reject incompatible saved weights.
  Keep map geometry in
  metadata, not repeated frames. GRPO may share read-only geography, while bodies,
  obstacles, RNGs and transient observation caches remain independent. Record
  final death positions after drift rather than inferring one-cell movement.
- Optimize disk, RAM and repeated computation while preserving recorded values,
  playback controls, chart history, random seeking and legacy readers. Caches of
  dynamic observations must invalidate when relevant animals/resources change.
- Results, model libraries, caches and virtual environments stay ignored. Small,
  intentional documentation assets under `docs/assets/` are tracked. Keep work
  inside this checkout; do not commit source screen recordings or experiment data.
- Keep shared examples portable: use repo-relative paths or `/path/to/wtf`,
  without personal usernames or machine-specific home paths.

## Documentation boundaries

Keep package Markdown focused on current mechanics, interfaces, configuration,
architecture and reusable workflows. Do not add individual run names/IDs,
observation tables, campaign progress or trial histories to general guides.
Implementation and regression/benchmark evidence may explain adopted improvements.

The local, Git-ignored paper workspace under `docs/paper/` may retain observation
history and supporting evidence. It is separate from package documentation; do not link
ordinary operational guides to studies unless that connection is necessary.
Store other experiment reports in ignored results or caches by default.

`docs/calibration.md` is the living exception for recommendations: after relevant
evaluation, update the current best-supported settings, objective, applicable
conditions and tradeoffs when warranted. Keep it a current decision reference,
not an experiment log; inconclusive trials do not require another entry.

Running/analyzing an experiment or adding paper observations is not a software
release and must not create/extend a rollout record. Release notes cover actual
implementation, interface or adopted configuration changes and their validation.
Keep existing version/commit history accurate without recording experiment outcomes.

## Documentation visuals

Use visuals selectively when they make structure, ownership, spatial relations
or state transitions quicker to understand. Prefer connected/nested blocks for
containers, records, arrays and references; use diagrams to explain relationships,
with values as supporting examples. Do not add a figure to every guide or turn
prose into boxes. Keep tables for comparisons/field lists and prose or commands
for straightforward instructions. Reuse/link existing figures before duplicating
them. See [visual documentation guidance](docs/visual-guide.md#when-a-visual-helps).

## Working and checking

Use `.venv/bin/python` and install runtime dependencies only in `.venv`.
`requirements.txt` currently contains tqdm; learning code has no ML-framework
dependency. Preserve progress reporting during training. Prefer code inspection
for optimization requests; do not start profiling or expensive training without
scope that calls for it.

### Mandatory formatting after development updates

After every code-changing development update, run the formatters over **all
project source files**, including tracked files and new non-ignored source files,
even when only one file changed. Format before final validation and before
reporting the update complete. Follow [the formatting workflow](docs/development.md#mandatory-formatting).

- Python: Black, including simulation, tests and `work/` tools.
- C/C++ and CPython extensions: clang-format, including native headers.
- JavaScript (`.js`, `.mjs`, `.cjs`), CSS, HTML and maintained JSON: Prettier.
- Shell scripts: shfmt, preserving each script's shell dialect.

Exclude virtual environments, dependencies, results, checkpoints/model archives,
build products, caches and generated documentation/media. Preserve existing
formatter configuration; formatting must not change program behavior or rewrite
generated assets. Keep formatter dependencies isolated in the project's dev
environment. If a formatter is unavailable or fails, resolve it when authorized
or report the exact blocker; never silently skip a language or claim success.
Run the corresponding check modes after formatting; if validation needs another
code edit, repeat formatting and the relevant checks. Include formatting status
in the final handoff. Documentation-only edits use documentation validation.

Inspect `git status` before editing and preserve existing work. Select relevant
checks from [development](docs/development.md). The standard suites are:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -v
node --test tests/*.test.mjs
```

Use temporary/ignored outputs for verification. A documentation-only change needs
link/asset/skill checks rather than a training run. Restart a playback server after
Python reader/server changes; it can otherwise serve new JS with old readers.
Stop only a process you started or that the user asked you to stop.

For an adopted implementation/configuration rollout, update the relevant guide
and [release hub](docs/releases/README.md). Versions have four components. Mark uncommitted work Pending; record
only real commit hashes. Do not create commits, tags or package-version changes
just to maintain release docs.

## Repo-local skills

These reusable workflows live in `.agents/skills/`:

- [wtf-analyze-run](.agents/skills/wtf-analyze-run/SKILL.md): saved experiment analysis.
- [wtf-optimize](.agents/skills/wtf-optimize/SKILL.md): simulation/storage optimization.
- [wtf-release-docs](.agents/skills/wtf-release-docs/SKILL.md): rollout records and links.

Read a skill when its workflow matches the task. It supplements the guides;
it does not authorize new experiments, data conversion or publishing.
