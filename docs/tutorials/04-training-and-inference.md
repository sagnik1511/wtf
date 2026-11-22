# 4. Train with less output, then record inference

[Tutorial hub](../tutorial.md) · [Previous: restore and view](03-restore-and-view.md) · [Next: compare saved policies](05-compare-saved-policies.md)

Use this workflow when you want a longer training campaign without keeping its
world playback. You will retain metrics, recovery checkpoints and best brains,
then record fresh evaluation worlds with fixed learned weights. Requires package
0.0.2.9+ or current source, Python 3.12+, and an activated virtual environment.
All commands below also work outside a checkout with bundled presets.

## Choose what to record

| Need | Training profile |
| --- | --- |
| Every configured world frame and inspector sample | `full`, the unchanged default |
| World playback, with fewer inspector samples and no activations | `compact` |
| Metrics, best brains and recovery state; no world playback | `training-only` |

Compact preserves the configured frame cadence. Its regular diagnostic interval
is at least 50 steps, with extra chunk-start/final samples; diagnostic interval
zero stays disabled. It is an explicit reduction in saved inspector history.
Training-only discards the opportunity to watch the original training episodes.
Neither profile changes decisions, learning, rewards, world seeds or reproduction.
See [capture rules](../recording.md#training-recording-profiles) for all overrides.

## Train and retain recovery data

```sh
python -m wtf run ppo --runs 3 --steps 1000 --seed 100 \
  --recording-profile training-only --progress \
  --checkpoint-every-steps 500 --checkpoint-keep-last 2 \
  --brain-save-every-steps 250 --brain-keep-last 2
```

This is a workflow example, not a calibrated training budget. For a quick smoke
check, use `--runs 1 --steps 5`; such a short run says nothing about learning or
reproduction. Each invocation prints a new result folder. Use that path wherever
`results/TRAINING_RUN` appears below.

Expect `environment.json`, `manifest.json`, `run_seeds.json`, `metrics.csv`,
`summary.json`, `champions/` and `checkpoints/`, with **no `playback/` directory**.
The two keep counts refer to different things: recent recovery files versus
recent qualifying best-brain milestones. Numbered run champion archives remain.
`--no-library` would suppress brain exports, so leave it off for this workflow.

```python
import json
from pathlib import Path

training = Path("results/TRAINING_RUN")
manifest = json.loads((training / "manifest.json").read_text())
assert manifest["mode"] == "train"
assert manifest["recording_profile"] == "training-only"
assert not manifest["playback_recording"]
assert not (training / "playback").exists()
assert (training / "champions" / "run_0001.json").is_file()
print(json.loads((training / "summary.json").read_text()))
```

For interrupted work, resume an unfinished recovery checkpoint through
`--resume`; the saved training profile persists. Keep referenced shared metadata
with the checkpoint when enabled. A completed final checkpoint has no remaining
training schedule. [Retention](../artifact-retention.md) covers rotation and
[exact resume](../running.md#resume-an-exact-experiment) covers continuation.

## Record a new inference experiment

Use the training environment so the observation layout, action order, network
architecture and species names match the saved brains:

```sh
python -m wtf run results/TRAINING_RUN/environment.json \
  --brains-from results/TRAINING_RUN \
  --mode evaluate --freeze-brains --recording-profile full \
  --runs 3 --steps 1000 --seed 1000 --progress
```

Use a different base seed for held-out worlds. `--brains-from` defaults to the
latest numbered run archive and the configured per-species/sex champion criteria.
Add `--brain-run 1` to select an earlier numbered archive, or use the existing
`--brain SPECIES[:SEX]=CRITERION` selectors. Compatibility and missing-group rules
are described in [saved brains](../brain-library.md); never substitute a
wrong-sex or incompatible model just to fill a missing bank entry.

`evaluate` performs no optimizer learning or birth mutation. `--freeze-brains`
also disables birth crossover: newborns receive independent copies of their
same-sex parent's model. Mating decisions and physical births still operate;
recurrent memory still evolves during inference. Without the flag, ordinary
evaluation retains crossover and can introduce new child weight combinations.

The training-only configuration resolves to full configured recording during
evaluation; the explicit `full` flag makes that choice visible. Existing
`recording.enabled: false` or `--no-record` still suppress recording. The new
folder holds the inference recording, metrics and recovery state. Evaluation
does not export replacement champions or alter the source artifacts.

## Verify and inspect

Replace `results/EVALUATION_RUN` with the second command's output folder:

```python
import json
from pathlib import Path
from wtf import Experiment
from wtf.storage.checkpoint_io import read_checkpoint

folder = Path("results/EVALUATION_RUN")
manifest = json.loads((folder / "manifest.json").read_text())
assert manifest["mode"] == "evaluate"
assert manifest["freeze_evaluation_brains"]
assert manifest["recording_profile"] == "full"
assert manifest["playback_recording"]
evaluation = Experiment.restore(read_checkpoint(folder / "checkpoints/final.bin.xz"))
assert evaluation.freeze_evaluation_brains
assert (
    evaluation.total_updates == 0
)  # Fresh champion warm start, not checkpoint resume.
print(manifest["initialized_from"])
print(evaluation.run_seeds)
```

```sh
python -m wtf view --results results
```

Choose the new evaluation recording and its run selector. Inspect an animal's
observation, prediction bars and recorded activations; inspect population and
food trends over time. Training-only output itself is absent from the recording
list. Hidden activations are available only if the original low-level activation
setting is enabled.

This playback shows **new inference episodes**, not historical training actions.
It cannot fill gaps in discarded training playback. Metrics and champion ranks
alone do not establish improvement; continue with the
[matched-policy comparison](05-compare-saved-policies.md) to separate world-seed
variation from differences between saved banks.
