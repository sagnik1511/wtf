# Flow diagram sources and SVG exports

[Visual documentation guide](../../visual-guide.md) · [Asset index](../README.md)

The guides embed **27 SVG diagrams** across 17 pages. Readers need no Mermaid
plugin or JavaScript. These show ownership, data dependencies and decision order;
they do not depict measured performance or guarantee learning outcomes.

Local `.mmd` files are ignored by Git and are not required to read the docs.
They preserve a small, portable Mermaid flowchart subset. The
builder parses that subset and uses Graphviz to lay it out as SVG; it does not
require or emulate the full Mermaid renderer. Labels, decisions, edges, edge
labels and flat subgraphs are preserved, with counts checked in the manifest.
Unsupported source syntax is rejected. Generated SVGs have accessible titles,
light/dark styles and an opaque light fallback for simpler image renderers.

| Diagram | Documentation | Local source filename |
| --- | --- | --- |
| [Runtime boundaries](runtime-boundaries.svg) | [Guide](../../architecture.md#runtime-boundaries) | `runtime-boundaries.mmd` |
| [One simulation step](one-simulation-step.svg) | [Guide](../../architecture.md#one-simulation-step) | `one-simulation-step.mmd` |
| [Private learning loop](private-learning-loop.svg) | [Guide](../../algorithms.md#private-learning-loop) | `private-learning-loop.mmd` |
| [GRPO trial isolation](grpo-trial-isolation.svg) | [Guide](../../algorithms.md#grpo-trial-isolation) | `grpo-trial-isolation.mmd` |
| [Recurrent decision state](recurrent-decision-state.svg) | [Guide](../../advanced-algorithms.md#recurrent-decision-state) | `recurrent-decision-state.mmd` |
| [Distributional action values](distributional-action-values.svg) | [Guide](../../advanced-algorithms.md#distributional-action-values) | `distributional-action-values.mmd` |
| [Masked decisions](masked-decisions.svg) | [Guide](../../advanced-algorithms.md#masked-decisions) | `masked-decisions.mmd` |
| [MAPPO information boundaries](mappo-information-boundaries.svg) | [Guide](../../advanced-algorithms.md#mappo-information-boundaries) | `mappo-information-boundaries.mmd` |
| [Run, continue, or view](run-continue-or-view.svg) | [Guide](../../running.md#run-continue-or-view) | `run-continue-or-view.mmd` |
| [From configuration to experiment](from-configuration-to-experiment.svg) | [Guide](../../configuration.md#from-configuration-to-experiment) | `from-configuration-to-experiment.mmd` |
| [Reproduction gates](reproduction-gates.svg) | [Guide](../../configuration.md#reproduction-gates) | `reproduction-gates.mmd` |
| [Names, roles and private brains](names-roles-and-private-brains.svg) | [Guide](../../species.md#names-roles-and-private-brains) | `names-roles-and-private-brains.mmd` |
| [Weights versus exact state](weights-versus-exact-state.svg) | [Guide](../../brain-library.md#weights-versus-exact-state) | `weights-versus-exact-state.mmd` |
| [Champion chaining between runs](champion-chaining-between-runs.svg) | [Guide](../../brain-library.md#champion-chaining-between-runs) | `champion-chaining-between-runs.mmd` |
| [Child-only mutation path](child-only-mutation-path.svg) | [Guide](../../mutation.md#child-only-mutation-path) | `child-only-mutation-path.mmd` |
| [Physical costs and learning feedback](physical-costs-and-learning-feedback.svg) | [Guide](../../rewards.md#physical-costs-and-learning-feedback) | `physical-costs-and-learning-feedback.mmd` |
| [Seeking and charts use separate data](seeking-and-charts-use-separate-data.svg) | [Guide](../../recording.md#seeking-and-charts-use-separate-data) | `seeking-and-charts-use-separate-data.mmd` |
| [Two payloads, one bounded file at a time](two-payloads-one-bounded-file-at-a-time.svg) | [Guide](../../binary-storage.md#two-payloads-one-bounded-file-at-a-time) | `two-payloads-one-bounded-file-at-a-time.mmd` |
| [Observation cache dependencies](observation-cache-dependencies.svg) | [Guide](../../performance.md#observation-cache-dependencies) | `observation-cache-dependencies.mmd` |
| [From recorded behavior to an assessment](from-recorded-behavior-to-an-assessment.svg) | [Guide](../../analysis.md#from-recorded-behavior-to-an-assessment) | `from-recorded-behavior-to-an-assessment.mmd` |
| [Historical calibration study flow](historical-calibration-study-flow.svg) | [Guide](../../calibration.md#historical-calibration-study-flow) | `historical-calibration-study-flow.mmd` |
| [Choose checks for the change](choose-checks-for-the-change.svg) | [Guide](../../development.md#choose-checks-for-the-change) | `choose-checks-for-the-change.mmd` |
| [Rollout record lifecycle](rollout-record-lifecycle.svg) | [Guide](../../releases/README.md#rollout-record-lifecycle) | `rollout-record-lifecycle.mmd` |

| [Top-k observation pipeline](top-k-observation-pipeline.svg) | [Guide](../../observations.md#visible-target-slots) | `top-k-observation-pipeline.mmd` |
| [Inspector: current and sampled state](inspector-current-and-sampled.svg) | [Guide](../../recording.md#visual-agent-inspector) | `inspector-current-and-sampled.mmd` |
| [Shared champion weight storage](shared-champion-weights.svg) | [Guide](../../binary-storage.md#champion-storage-measurement) | `shared-champion-weights.mmd` |
| [Early-stopping patience](early-stopping-patience.svg) | [Guide](../../running.md#stop-on-performance-plateaus) | `early-stopping-patience.mmd` |

## Regenerate and verify

From the repository root, with Graphviz (`dot`) installed:

```sh
.venv/bin/python -m work.build_doc_diagrams
.venv/bin/python -m work.build_doc_diagrams --check
```

Viewing the SVGs needs no Graphviz installation. Regeneration requires only
Graphviz and Python's standard library; no runtime project dependencies changed.
For the original flows, keep authoring sources locally; fresh clones contain
SVGs and the manifest. Regenerating those flows requires the local sources. Edit sources and the
manifest together when adding/removing nodes or edges. With sources present,
`--check` compares regenerated exports with tracked SVGs; without them, it checks
SVG structure and topology counts only and reports that reduced scope. Layout details may
change across Graphviz versions, so review generated images before committing.
Sources use quoted rectangular/decision nodes, single directed arrows with
optional labels, and non-nested subgraphs. Keep more involved diagrams in the
existing physics asset workflow rather than relying on unsupported syntax.

These are original explanatory diagrams based on repository code and guide
semantics. They contain no model arrays, experiment snapshots or personal paths.
PNG previews used for visual review stay under ignored `.cache/docs-viz/`.

## Recent feature diagrams

The four recent flows above and the revised champion-chaining flow have portable
authoring definitions in [build_recent_doc_diagrams.py](../../../work/build_recent_doc_diagrams.py).
They can be regenerated from a fresh clone without retained `.mmd` files:

```sh
.venv/bin/python -m work.build_recent_doc_diagrams
.venv/bin/python -m work.build_recent_doc_diagrams --check
```

This also builds the [controlled inspector illustration](../inspector/README.md).
Generated `.mmd` files and PNG previews remain ignored local caches.
