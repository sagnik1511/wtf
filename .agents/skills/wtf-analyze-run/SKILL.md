---
name: wtf-analyze-run
description: Analyze saved Wildlife Trial Framework experiments, species behavior and learning evidence from a results folder without starting new training.
---

# Analyze a saved WTF experiment

Work from the repository root. Read [analysis semantics](../../../docs/analysis.md)
and the relevant parts of [project context](../../../docs/project-context.md).

Resolve the supplied run name/path; inspect its metadata, captured configuration,
completion status and available numbered runs. Do not substitute the current
preset for the configuration actually used. Distinguish experiment folders from
individual runs. Inspect champion provenance/chaining when continuation matters.

Use the existing analyzer instead of reimplementing replay decoding:

```sh
.venv/bin/python -m ecosystem.analyze results/EXPERIMENT --window-steps 250
# Add --run N for a single numbered run.
```

The CLI writes `analysis.json` inside the ignored experiment folder by default.
For a read-only investigation call `ecosystem.analyze.analyze_recording` directly
or use the existing read-only API; see the analysis guide. Stream committed chunks
and report partial coverage for writing/sampled recordings. Never load every
frame or restore all model weights merely to summarize behavior.

Compare early/late windows and distributions across available runs. Useful
species measures include population, survival/extinction, births/death causes,
productive feeding/hunting success, starvation/energy coverage, action/reward
trends, learning updates and mutations. Normalize event counts by animal-steps
where available; growing populations alone can increase totals. Do not invent
missing samples or convert absent predictions into zero probability.

State observations, plausible explanations and unresolved causes separately.
Survival and kills alone do not prove learning: changing seeds, inherited weights,
reproduction, changed physics and simultaneous opponent learning confound them.
Recommend matched frozen-policy comparisons across held-out seeds when useful;
run new experiments only when requested. Lead with the meaningful findings and
specific next configuration choices, with measured values and coverage limits.
