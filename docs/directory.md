# Complete documentation directory

[Documentation home](index.md) · [Visual guide](visual-guide.md)

Use this directory to find every maintained guide without guessing a page name.
The navigation uses the same subject groups. Topic sections below also link
straight to frequently inspected visuals. This site documents the package;
training and the replay viewer run locally.

## Jump to a feature or visual

| Looking for | Open |
| --- | --- |
| Animal pointers, sex colors, food and birth/kill/death symbols | [Board-piece legend](viewer-visual-reference.md#animals-resources-and-event-markers) |
| Movement, water, current, height and visible area | [Physics explorer](physics.md), [terrain/sight specimens](viewer-visual-reference.md#terrain-and-current-sight) |
| Environment maps in motion | [Terrain playback gallery](geography.md#recorded-terrain-previews) |
| Sunlight, time dial and interface themes | [Day–night rules](day-night.md), [theme comparison](viewer-visual-reference.md#interface-themes) |
| Selected animal's observation circle and predictions | [Agent inspector](recording.md#visual-agent-inspector), [brain activity](explainability.md#recorded-animal-activations) |
| Population, health, births/deaths and action trends over time | [Metrics dashboard](recording.md#per-step-species-dashboard) |
| Archived weights, inputs, activations and explanation routes | [Saved-brain comparison](explainability.md#archived-comparison) |
| Replay files, model weights, optimizer, RNG and buffers | [Replay/artifact map](replay-data-structures.md), [checkpoint structures](checkpoint-data-structures.md) |
| Any configurable field or supported command | [Field definitions](configuration-reference.md), [CLI reference](command-reference.md) |
| Add a policy, species, map or action | [Customization handbook](customization.md), [contributor context](project-context.md) |

## Get started

| Guide | What it covers |
| --- | --- |
| [Installation](installation.md) | Dependencies, installation commands and the first viewer launch. |
| [Linux and Ubuntu](linux.md) | Ubuntu/Linux requirements, native builds and platform checks. |
| [Docker](docker.md) | Container build and launch instructions, ports and persistent output. |
| [Run and resume](running.md) | CLI runs, seeds, recording, resuming and early stopping. |
| [Complete tutorial](tutorial.md) | Entry point for the complete Python experiment walkthrough. |
| [Tutorial 1: Configure](tutorials/01-configuration.md) | Choose and validate an environment before running. |
| [Tutorial 2: Run and analyze](tutorials/02-run-record-analyze.md) | Run a complete experiment and inspect saved metrics. |
| [Tutorial 3: Restore and view](tutorials/03-restore-and-view.md) | Restore checkpoints, initialize champions and open playback. |
| [Python example](examples.md) | Runnable Python example and its output contract. |

## Worlds and species

| Guide | What it covers |
| --- | --- |
| [Customization handbook](customization.md) | Choose JSON, CLI, viewer preferences or source extensions for a change. |
| [Environment configuration](configuration.md) | World/species settings, action sets and reproduction controls. |
| [Species](species.md) | Add named species and configure body, brain and sex colors. |
| [Geography and maps](geography.md) | Map presets, editable geography, currents and terrain playback clips. |
| [Movement and visibility](physics.md) | Interactive explorer and illustrated movement, swimming, sight and mating cases. |
| [Day and night](day-night.md) | Optional sunlight, sight/cost modifiers, dial and playback lighting. |
| [Current calibration](calibration.md) | Current supported tuning guidance, objectives and tradeoffs. |

## Learning and behavior

| Guide | What it covers |
| --- | --- |
| [Algorithms and networks](algorithms.md) | Available learning rules, private controllers and network depth. |
| [Advanced algorithms](advanced-algorithms.md) | Recurrent PPO, QR-DQN, masked PPO and multi-agent PPO contracts. |
| [Brain inputs and observations](observations.md) | Input ordering, top-k target slots and visibility versus observations. |
| [Rewards and costs](rewards.md) | Physical energy costs, successful actions and failed-action feedback. |
| [Feedback contracts](feedback-review.md) | Generic reward contracts and algorithm/environment boundaries. |
| [Mating observations](mating-observations.md) | Maturity/readiness inputs, eligibility and model compatibility. |
| [Mating intent](mating-intent.md) | Mutual consent windows and failed mating reasons. |
| [Escape and consent](escape-and-consent.md) | Learned escape actions and observable nearby mating intent. |
| [Mutation](mutation.md) | Rare newborn-only weight perturbations and inheritance. |
| [Champions and inheritance](brain-library.md) | Champion criteria, species/sex archives, lineage and warm starts. |
| [Behavior analysis](analysis.md) | Behavior metrics, outcomes and limits of learning claims. |
| [Evaluation](evaluation.md) | Matched-seed comparisons and evaluation methodology. |

## Viewer and storage

| Guide | What it covers |
| --- | --- |
| [Playback controls and metrics](recording.md) | Play/seek controls, selected animals, metrics dashboard and recording options. |
| [Board objects and viewer guide](viewer-visual-reference.md) | Animal pointers, food, obstacles, events, sight overlays, themes and chart specimens. |
| [Agent inspector and brain explanations](explainability.md) | Selected-animal inputs/outputs/activations and saved-brain comparisons. |
| [Replay files and data structures](replay-data-structures.md) | Artifact tree, chunk records, frame deltas and champion weight records. |
| [Checkpoint state and restoration](checkpoint-data-structures.md) | World, optimizer, RNG and replay-buffer state with restore/access diagrams. |
| [Binary formats and compression](binary-storage.md) | Typed binary layouts, compression, deduplication and legacy readers. |

## Development

| Guide | What it covers |
| --- | --- |
| [Architecture](architecture.md) | Python/JavaScript ownership, simulation phases and runtime data flow. |
| [Development and testing](development.md) | Development setup, test suites and mandatory source formatting. |
| [Add policy models](extending-policies.md) | Implement and integrate a new learning algorithm. |
| [Add geography and actions](extending-worlds-and-actions.md) | Add geography, physical rules, actions and observation contracts. |
| [Package builds](packaging.md) | Wheel/sdist resources, isolated installation and release checks. |
| [Performance](performance.md) | Shared work, bounded caches, allocation reductions and invalidation. |
| [Native kernels](native-kernels.md) | Optional native learning kernels and portable fallback behavior. |
| [Extended native kernels](native-kernels-v2.md) | Additional compiled paths and equivalence boundaries. |
| [Native performance review](native-performance-review.md) | Kernel coverage, numeric types and measured performance tradeoffs. |
| [Contributor context](project-context.md) | Contributor/agent context and implementation boundaries. |
| [Documentation website](documentation-site.md) | MkDocs publication, link/theme handling and Pages maintenance. |

## Reference

| Guide | What it covers |
| --- | --- |
| [Visual assets and sources](assets/README.md) | Catalog of tracked visual assets, source provenance and regeneration guides. |
| [Configuration fields](configuration-reference.md) | Complete generated schema definitions, types, ranges and defaults. |
| [Command reference](command-reference.md) | Simulator, viewer/server and analyzer flags. |
| [Compatibility policy](compatibility.md) | Replay, brain and checkpoint version/runtime compatibility. |
| [Visual guide](visual-guide.md) | Question-to-figure index and visual/theme authoring rules. |
| [Project identity](branding.md) | SVG/PNG logos, placement, light/dark variants and motion policy. |
| [Release history](releases/README.md) | Implementation history and versioned release records. |

## Supporting catalogs and history

These pages are maintained alongside the guides they support; each catalog
links its figures and source/export instructions. Historical release entries
are indexed by the release hub rather than repeated in the sidebar.

| Catalog | Contents |
| --- | --- |
| [Viewer specimens](assets/viewer/README.md) | Board objects, terrain, lighting, themes, workspace and brain views |
| [Inspector illustration](assets/inspector/README.md) | Current sight versus sampled observations/predictions |
| [Physics assets](assets/physics/README.md) | Interactive explorer and controlled physics SVG cases |
| [Map diagrams](assets/geography/README.md) | Vector geography from maintained map definitions |
| [Terrain playback exports](assets/terrains/README.md) | Full-viewer clips and refresh instructions |
| [Flow diagrams](assets/diagrams/README.md) | Architecture, learning, inheritance, recording and maintenance flows |
| [Runtime structures](assets/structures/README.md) | Input vectors, lineage, caches, activations and explanations |
| [Storage structures](assets/storage/README.md) | Replay, model-weight and checkpoint blocks |
| [Release history](releases/README.md) | Existing versioned implementation notes and the authoring template |

Research observations, local experiments, results and the paper workspace are
excluded from the published documentation.
