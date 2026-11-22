# 5. Compare saved policy banks on matched worlds

[Tutorial hub](../tutorial.md) · [Previous: training and inference](04-training-and-inference.md) · [Evaluation method](../evaluation.md)

Compare two compatible saved brain banks without learning during the comparison.
You will run each bank on the same held-out seed schedule, save complete
inference playback, and print paired endpoint metrics. Use package 0.0.2.9+ in an
activated Python 3.12+ environment. The CLI and Python APIs work without a checkout.

## Prepare the two conditions

Use `results/BANK_A` and `results/BANK_B` for two completed training experiments
from [Tutorial 4](04-training-and-inference.md). Replace both placeholders in the
code below. Both need compatible champions for the environment's species/sex
groups, algorithms, observation layout, action order and network architecture.
This tutorial compares the loaded weights, not different model architectures.

Choose one common environment for both conditions. The script uses bank A's
saved environment, overriding only run count, horizon, seed schedule, mode,
recording profile and initial brain source. Do not change population, food,
terrain, sight, rewards or mating radius between the two evaluations.
`--freeze-brains` keeps offspring in their same-sex parent bank; no learning,
crossover, mutation or adaptive run-to-run champion chaining occurs.

The example uses three held-out seeds and 1,000 steps per episode. Change the
seed range if it overlaps training. Three worlds illustrate the workflow;
they are not enough to establish generalization or statistical significance.
Different populations can still produce different early-extinction times.

## Run both banks and save a comparison index

Save this code as `compare_policies.py` in your writable working directory and
run `python compare_policies.py`. It uses the installed CLI so ordinary streaming
metrics, progress, checkpoint retention and recording rules remain in effect.
The two new evaluation folders and their small index stay under `results/`.

```python
import csv
import json
from pathlib import Path
import subprocess
import sys
import uuid

banks = {"A": Path("results/BANK_A"), "B": Path("results/BANK_B")}
environment = banks["A"] / "environment.json"
output = Path("results") / f"comparison_{uuid.uuid4().hex[:8]}"
output.mkdir(parents=True, exist_ok=False)

runs, steps, base_seed = 3, 1000, 1000
folders = {}
for label, bank in banks.items():
    before = {p for p in output.iterdir() if p.is_dir()}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "wtf",
            "run",
            str(environment),
            "--brains-from",
            str(bank),
            "--mode",
            "evaluate",
            "--freeze-brains",
            "--recording-profile",
            "full",
            "--runs",
            str(runs),
            "--steps",
            str(steps),
            "--seed",
            str(base_seed),
            "--seed-policy",
            "increment",
            "--output",
            str(output),
            "--progress",
        ],
        check=True,
    )
    created = {p for p in output.iterdir() if p.is_dir()} - before
    assert len(created) == 1
    folders[label] = created.pop()
    manifest = json.loads((folders[label] / "manifest.json").read_text())
    assert manifest["mode"] == "evaluate"
    assert manifest["freeze_evaluation_brains"]
    assert manifest["playback_recording"]

(output / "comparison.json").write_text(
    json.dumps({k: v.name for k, v in folders.items()}, indent=2) + "\n"
)
by_seed = {}
for label, folder in folders.items():
    summaries = json.loads((folder / "summary.json").read_text())
    assert len(summaries) == runs
    by_seed[label] = {row["seed"]: row for row in summaries}
    assert len(by_seed[label]) == runs
assert set(by_seed["A"]) == set(by_seed["B"]) == set(range(base_seed, base_seed + runs))

writer = csv.writer(sys.stdout)
writer.writerow(
    [
        "seed",
        "species",
        "steps_A",
        "steps_B",
        "population_A",
        "population_B",
        "births_A",
        "births_B",
        "kills_A",
        "kills_B",
    ]
)
for seed in sorted(by_seed["A"]):
    a, b = by_seed["A"][seed], by_seed["B"][seed]
    assert set(a["populations"]) == set(b["populations"])
    for species in a["populations"]:
        writer.writerow(
            [
                seed,
                species,
                a["tick"],
                b["tick"],
                a["populations"][species],
                b["populations"][species],
                a.get("births", {}).get(species, 0),
                b.get("births", {}).get(species, 0),
                a.get("kills", {}).get(species, 0),
                b.get("kills", {}).get(species, 0),
            ]
        )
print(f"Viewer root: {output}")
```

Use only one instance of this script per output folder. The unique directory
also keeps evaluation folders from being confused with source training artifacts.
Inspect `manifest.json` → `initialized_from` to confirm the selected archive,
species/sex groups and criteria. The script uses each source directory's latest
numbered archive. Pin `--brain-run N` in the argument list when a particular
archive is part of the comparison design.

## Inspect behavior as well as endpoint counts

Use the printed viewer root:

```sh
python -m wtf view --results results/comparison_ID
```

`comparison.json` maps A/B to recording folder names. Select each recording and
its run number, checking the displayed seed before comparing. Look at population,
food and health trends, deaths by cause, observed hunts and mating failures.
Select an animal to inspect its inputs and action scores. A visually interesting
trajectory is not evidence that its bank performs better on other worlds.

For detailed recorded behavior, replace the folder below with one of the two
evaluation folders and run the analyzer for each condition:

```sh
python -m wtf analyze results/comparison_ID/EVALUATION_FOLDER
```

Endpoint birth and kill counts depend on the number of animals and their time
alive. Different extinction times also mean different exposure. Use the
[behavior analysis](../analysis.md) rates, energy coverage and coverage flags
alongside the printed table. Survivors are censored at the episode horizon;
their current age is not a completed lifespan.

Both species' banks change together in this example. That measures an ecosystem
pairing; it does not isolate predator improvement or prey improvement. To isolate
one species, load the same opposing species/sex champions in both conditions
using explicit `--brain` selections. More independent training replicas and
held-out seeds are required before attributing a stable advantage to a model.

Equal starting seeds hold world initialization constant for this compatible
configuration. Different decisions, births and deaths can subsequently consume
RNG draws differently, so future trajectories need not stay synchronized.
Keep both evaluation folders and their manifests with any conclusions; do not
promote settings merely because one run has more births or kills.
