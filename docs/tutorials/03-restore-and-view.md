# 3. Restore state and view the recording

[Tutorial hub](../tutorial.md) · [Previous: run and record](02-run-record-analyze.md) · [Compatibility](../compatibility.md)

## Read metrics and restore a checkpoint

Replace `EXPERIMENT` with the directory printed by the example:

```python
import json
from pathlib import Path
from wtf.storage.checkpoint_io import read_checkpoint
from wtf.experiment.runner import Experiment

folder = Path("results/EXPERIMENT")
summary = json.loads((folder / "summary.json").read_text())[0]
print({key: summary[key] for key in ("tick", "populations", "births", "kills", "deaths")})

restored = Experiment.restore(read_checkpoint(folder / "checkpoints/final.bin.xz"))
assert restored.done
assert restored.world.tick == summary["tick"]
```

The final checkpoint is completed, so `step()` does not start another episode.
To continue unfinished work, save and restore before the configured experiment
finishes. The current `read_checkpoint` normalizes supported on-disk envelopes;
do not hand-decode binary arrays or confuse file-envelope and logical versions.

## Start a fresh world with learned champions

An exact resume and a new world with saved weights are different operations:

```sh
.venv/bin/python -m wtf.cli environments/maskable_ppo.json \
  --runs 1 --seed 101 --brains-from results/EXPERIMENT
```

The species, sex and brain signatures must match. New-world initialization uses
matching archived champions and fresh training state; it does not restore the
old world, optimizer, replay buffers or RNG state. See the
[brain guide](../brain-library.md) for selection criteria and missing-group rules.

## Open the offline viewer

```sh
./run.sh
```

Open <http://127.0.0.1:8000>, refresh the saved-recording list, choose your example
and click **Load recording**. Python training has already finished. The viewer
loads one replay chunk at a time; selecting another run or seeking does not
restore learning checkpoints. Use `./run.sh --results PATH` for a different root.

Click an animal to inspect its recorded observations, action scores and sampled
brain activity. **Explain saved brain** is a separate evaluation of the archived
species/sex champion; it is not guaranteed to be that animal's live model. The
[appearance reference](../viewer-visual-reference.md) shows each visual and its
meaning. Day/night lighting appears only for enabled recordings; interface
light/dark preference is independent of simulated time.

## Share and retain results

Copy the whole experiment directory to preserve relative references. Keep the
original JSON, source/package revision and runtime/backend information with
experiments you need to reproduce. Results stay Git-ignored. Do not embed model
weights or binary replay data in a tutorial. Use the
[compatibility policy](../compatibility.md) before moving data across versions.
