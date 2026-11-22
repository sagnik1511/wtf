# Visual guide to the documentation

[All visual files](visuals.md) · [Verified homepage click paths](access-map.md)

[Documentation directory](directory.md) · [Project overview](../README.md) · [Project context](project-context.md) · [Release history](releases/README.md)

Use the diagrams in the guide that matches your question. They describe the
implemented data flow and ownership, not measured performance or proof of
learning. The guides embed SVG images, so GitHub and local Markdown readers need no
Mermaid extension to display them. Mermaid authoring sources are local-only and ignored by Git; see the
[diagram catalog](assets/diagrams/README.md) for export validation and optional
local regeneration. Physics SVGs and recorded GIFs remain separate, ordinary images.

| Question | Guide and visual |
| --- | --- |
| What do animal pointers, food and birth/death markers mean? | [Board-piece legend](viewer-visual-reference.md#animals-resources-and-event-markers) |
| What do viewer overlays and brain graphs look like? | [Board objects and viewer guide](viewer-visual-reference.md) |
| How does dawn/day/dusk/night change the board and animals? | [Day–night appearance and rules](day-night.md) |
| Where do Python and JavaScript meet? | [Runtime boundaries](architecture.md#runtime-boundaries) |
| What happens during a step? | [Simulation order](architecture.md#one-simulation-step) |
| How are input fields packed into the vector? | [Contiguous blocks and repeated slots](observations.md#input-layout) |
| Which visible targets enter the brain? | [Top-k input pipeline](observations.md#visible-target-slots) |
| How does an animal learn? | [Private learning loop](algorithms.md#private-learning-loop) |
| How do GRPO trials avoid changing the real world? | [Trial isolation](algorithms.md#grpo-trial-isolation) |
| What does recurrent memory do? | [Recurrent state](advanced-algorithms.md#recurrent-decision-state) |
| How does QR-DQN use return distributions? | [Distributional values](advanced-algorithms.md#distributional-action-values) |
| What does action masking change? | [Masked decisions](advanced-algorithms.md#masked-decisions) |
| Which MAPPO inputs are local/global? | [Information boundaries](advanced-algorithms.md#mappo-information-boundaries) |
| How do I start, resume or view? | [Run entry points](running.md#run-continue-or-view) |
| When does training stop improving? | [Completed-run patience](running.md#stop-on-performance-plateaus) |
| How do JSON settings become a run? | [Configuration flow](configuration.md#from-configuration-to-experiment) |
| Why did reproduction fail? | [Physical and decision gates](configuration.md#reproduction-gates) |
| Do species share a brain? | [Example private controllers](species.md#names-roles-and-private-brains) |
| What is restored by a champion versus a checkpoint? | [Initialization paths](brain-library.md#weights-versus-exact-state) |
| How does repeated ancestry affect family credit? | [Founder sets and overlapping ledgers](brain-library.md#founder-membership-and-family-credit) |
| How are champions carried into later runs? | [Species/sex chaining](brain-library.md#champion-chaining-between-runs) |
| Which animal does mutation affect? | [Child-only path](mutation.md#child-only-mutation-path) |
| How do physical costs reach an optimizer? | [Reward feedback](rewards.md#physical-costs-and-learning-feedback) |
| Why does seeking not load every run? | [Playback and chart data](recording.md#seeking-and-charts-use-separate-data) |
| Why do inspector inputs lag the board? | [Current sight, sampled inputs and prediction graphs](recording.md#visual-agent-inspector) |
| How does the selected animal’s brain change during replay? | [Actual activation sampling and playback](explainability.md#recorded-animal-activations) |
| How does a saved brain respond to an animal’s observation? | [Selected inputs, archived weights and explanation views](explainability.md#what-is-being-explained) |
| When is a whole network shown versus a summary? | [Input/output first and hidden-rank threshold](explainability.md#inputoutput-first-and-the-rank-threshold) |
| What do weighted links, activations and local input influence mean? | [Worked forward and reverse pass](explainability.md#worked-activation-example) |
| How do checkpoints restore optimizer, RNG and buffers? | [File records, private controller state and access paths](checkpoint-data-structures.md) |
| What is inside a stored model? | [References, binary model record and layer arrays](replay-data-structures.md#champion-weight-records) |
| How do archives avoid repeating model weights? | [Shared champion blobs](binary-storage.md#champion-storage-measurement) |
| How are replay files and frame deltas structured? | [Artifact map, binary records and field masks](replay-data-structures.md) |
| What is compressed and what is restored? | [Replay and checkpoint payloads](binary-storage.md#two-payloads-one-bounded-file-at-a-time) |
| Where would compiled kernels and smaller dtypes help? | [Native performance review](native-performance-review.md) (priority and dtype tables) |
| What is shared between observation windows? | [Bucket objects, references and revisions](performance.md#spatial-window-structure) |
| When is an observation cache invalidated? | [Cache dependencies](performance.md#observation-cache-dependencies) |
| Can a behavior report prove learning? | [Evidence path](analysis.md#from-recorded-behavior-to-an-assessment) |
| Which checks fit a change? | [Verification paths](development.md#choose-checks-for-the-change) |
| When is a rollout committed? | [Record lifecycle](releases/README.md#rollout-record-lifecycle) |
| What do rivers, cliffs, visibility and births look like? | [Physics SVG gallery and interactive explorer](physics.md) |
| What do the actual maps look like during playback? | [Full-viewer terrain GIFs](geography.md#recorded-terrain-previews) |

README and the contributor context already have overview/navigation tables;
asset catalogs already map filenames to cases. They link these visuals rather
than duplicating them. Historical release notes retain their original evidence;
the release hub explains their lifecycle. New illustrations belong alongside
the behavior they explain, with source/provenance for any measured values.

## When a visual helps

Add a visual when seeing containment, references, ordering, spatial overlap or
state ownership makes the explanation faster to understand. Use connected or
nested blocks to reveal those relationships; example values should support the
structure. A paragraph split among boxes is not automatically a useful diagram.
Reuse an existing figure by linking it when the same relationship applies.

Keep tables for comparisons, field definitions and benchmark results. Keep prose
and commands for setup, flags and short rules. A document needs no new figure
merely because it currently has none. Preserve these choices in future updates.

The latest review added three structural views: input-vector packing, founder
family membership and spatial-window references. It also reused the reward
figure from the feedback review and linked mating inputs to the vector layout.

| Reviewed area | Visual choice | Reason |
| --- | --- | --- |
| Observation fields / mating extension | Added one shared vector layout | Section offsets and repeated slots are easier to see spatially |
| Lineage curation | Added a pedigree with separate family ledgers | Explains overlapping ancestry and strongest-family credit |
| Spatial observation cache | Added window/bucket/reference structure | Separates shared objects from invalidation dependencies |
| Feedback audit | Link existing reward figure | Same accounting boundary; algorithm differences already fit a table |
| Native kernels, dtypes and performance reports | Keep coverage/benchmark tables and architecture links | Comparisons and measured values are already clear |
| Physics, maps, inheritance and algorithm information boundaries | Keep existing figures/explorer and add links where needed | Existing visuals cover these relationships |
| Configuration flags, setup commands, calibration records and release history | Keep prose/tables and existing navigation | No new structural question warrants another diagram |

## Light and dark appearance

Open **Home → All visuals → the visual name** for a preview with the site's
light/dark toggle. The adjacent original-file link opens the exact saved asset;
a standalone SVG follows the browser/OS preference and a raster file keeps its
saved pixels. The preview keeps theme selection inside Zensical, pairs animation
exports and logo variants, and embeds the full physics explorer.


Structural flow, storage, checkpoint and runtime diagrams use self-contained SVG
palettes for both light and dark surfaces. Physics diagrams adapt in the same
way. On GitHub Pages, embedded SVGs follow the site's sun/moon toggle through
CSS `color-scheme`, even when the operating-system theme differs. Opened on
their own, these SVGs follow the browser's system preference. Logo variants are
selected explicitly by the site theme.

Playback previews select an actual light or dark export: the website follows
its palette toggle and GitHub uses picture media selection. The light animation
uses compact WebP; the original GIF remains the dark/legacy fallback. No CSS
color-inversion filter changes recorded animals, lighting or chart values.
Standalone GIF URLs still show their saved pixels; theme selection belongs to
the page embedding them.

Geography and inspector/viewer SVG specimens have light/dark presentation
palettes too. Terrain values, node activities, action scores and sign/category
meanings are preserved. Daylight-phase boards render the selected viewer palette as well as the
corresponding sunlight. The interface comparison puts the selected palette
first and labels the alternate; both retain their actual viewer colors.
Outer labels, card surfaces and backgrounds adapt with them. The standalone physics explorer
adapts to system light/dark preference. The live viewer has its own theme toggle,
separate from simulated sunlight.

The project symbol stays to the left of the shared documentation header title
on every page, including narrow layouts. It is a home link and uses the matching
light/dark logo; only the home-page hero has the ribbon animation.
