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
- Animals learn independently. Birth copies a parent's brain into a separate
  controller. Rare birth mutation changes only that child. Preserve RNG isolation
  and exact checkpoint continuation when changing these paths.
- Optimize disk, RAM and repeated computation while preserving recorded values,
  playback controls, chart history, random seeking and legacy readers. Caches of
  dynamic observations must invalidate when relevant animals/resources change.
- Results, model libraries, caches and virtual environments stay ignored. Small,
  intentional documentation assets under `docs/assets/` are tracked. Keep work
  inside this checkout; do not commit source screen recordings or experiment data.
- Keep shared examples portable: use repo-relative paths or `/path/to/wtf`,
  without personal usernames or machine-specific home paths.

## Working and checking

Use `.venv/bin/python` and install runtime dependencies only in `.venv`.
`requirements.txt` currently contains tqdm; learning code has no ML-framework
dependency. Preserve progress reporting during training. Prefer code inspection
for optimization requests; do not start profiling or expensive training without
scope that calls for it.

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

Update the relevant guide and [release hub](docs/releases/README.md) for a
rollout. Versions have four components. Mark uncommitted work Pending; record
only real commit hashes. Do not create commits, tags or package-version changes
just to maintain release docs.

## Repo-local skills

These reusable workflows live in `.agents/skills/`:

- [wtf-analyze-run](.agents/skills/wtf-analyze-run/SKILL.md): saved experiment analysis.
- [wtf-optimize](.agents/skills/wtf-optimize/SKILL.md): simulation/storage optimization.
- [wtf-release-docs](.agents/skills/wtf-release-docs/SKILL.md): rollout records and links.

Read a skill when its workflow matches the task. It supplements the guides;
it does not authorize new experiments, data conversion or publishing.
