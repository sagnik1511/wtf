# Visual guide to the documentation

[Project overview](../README.md) · [Project context](project-context.md) · [Release history](releases/README.md)

Use the diagrams in the guide that matches your question. They describe the
implemented data flow and ownership, not measured performance or proof of
learning. The guides embed SVG images, so GitHub and local Markdown readers need no
Mermaid extension to display them. Mermaid authoring sources are local-only and ignored by Git; see the
[diagram catalog](assets/diagrams/README.md) for export validation and optional
local regeneration. Physics SVGs and recorded GIFs remain separate, ordinary images.

| Question | Guide and visual |
| --- | --- |
| Where do Python and JavaScript meet? | [Runtime boundaries](architecture.md#runtime-boundaries) |
| What happens during a step? | [Simulation order](architecture.md#one-simulation-step) |
| How does an animal learn? | [Private learning loop](algorithms.md#private-learning-loop) |
| How do GRPO trials avoid changing the real world? | [Trial isolation](algorithms.md#grpo-trial-isolation) |
| What does recurrent memory do? | [Recurrent state](advanced-algorithms.md#recurrent-decision-state) |
| How does QR-DQN use return distributions? | [Distributional values](advanced-algorithms.md#distributional-action-values) |
| What does action masking change? | [Masked decisions](advanced-algorithms.md#masked-decisions) |
| Which MAPPO inputs are local/global? | [Information boundaries](advanced-algorithms.md#mappo-information-boundaries) |
| How do I start, resume or view? | [Run entry points](running.md#run-continue-or-view) |
| How do JSON settings become a run? | [Configuration flow](configuration.md#from-configuration-to-experiment) |
| Why did reproduction fail? | [Physical and decision gates](configuration.md#reproduction-gates) |
| Do species share a brain? | [Example private controllers](species.md#names-roles-and-private-brains) |
| What is restored by a champion versus a checkpoint? | [Initialization paths](brain-library.md#weights-versus-exact-state) |
| How are champions carried into later runs? | [Species/sex chaining](brain-library.md#champion-chaining-between-runs) |
| Which animal does mutation affect? | [Child-only path](mutation.md#child-only-mutation-path) |
| How do physical costs reach an optimizer? | [Reward feedback](rewards.md#physical-costs-and-learning-feedback) |
| Why does seeking not load every run? | [Playback and chart data](recording.md#seeking-and-charts-use-separate-data) |
| What is compressed and what is restored? | [Replay and checkpoint payloads](binary-storage.md#two-payloads-one-bounded-file-at-a-time) |
| When is an observation cache invalidated? | [Cache dependencies](performance.md#observation-cache-dependencies) |
| Can a behavior report prove learning? | [Evidence path](analysis.md#from-recorded-behavior-to-an-assessment) |
| How were the calibration comparisons organized? | [Historical experiment flow](calibration.md#historical-calibration-study-flow) |
| Which checks fit a change? | [Verification paths](development.md#choose-checks-for-the-change) |
| When is a rollout committed? | [Record lifecycle](releases/README.md#rollout-record-lifecycle) |
| What do rivers, cliffs, visibility and births look like? | [Physics SVG gallery and interactive explorer](physics.md) |
| What do the actual maps look like during playback? | [Full-viewer terrain GIFs](geography.md#recorded-terrain-previews) |

README and the contributor context already have overview/navigation tables;
asset catalogs already map filenames to cases. They link these visuals rather
than duplicating them. Historical release notes retain their original evidence;
the release hub explains their lifecycle. New illustrations belong alongside
the behavior they explain, with source/provenance for any measured values.
