# Visual movement memory and scent trails

[Brain inputs](observations.md) · [Inspector appearances](viewer-visual-reference.md#movement-memory-and-scent) · [Configuration fields](configuration-reference.md)

Both senses are opt-in and work with all nine controller algorithms. Visual
memory belongs to an observer; scent belongs to the world. Neither sense tells
an animal which action to choose, guarantees improved hunting, or relaxes mating
radius, readiness, consent or challenge eligibility. No sniff action is required.

![Ownership and data flow for personal sightings and anonymous scent](assets/structures/tracking-data-flow.svg)

## Enable and configure

The maintained [tracking preset](../environments/tracking.json) extends masked
PPO with three previous steps of visual memory and automatic ground-scent
sampling. Existing presets retain their current settings.

```sh
.venv/bin/python -m wtf run tracking --runs 1 --seed 100 --progress
.venv/bin/python -m wtf view --results results
```

At the environment root, enable trail deposition and its lifetime:

```json
"scent_trails": {
  "enabled": true,
  "lifetime_steps": 20
}
```

On each brain template that should use these senses:

```json
"movement_history_steps": 3,
"scent_observation": {
  "mode": "blended",
  "radius": 2
}
```

`movement_history_steps` accepts 0–16 and defaults to zero. It counts previous
simulation steps, independently of the number of targets in
`observation_top_k`. Omit `scent_observation` to disable that brain's scent
inputs. Blended mode senses all configured species and sexes through one
anonymous field; omit `species` and `separate_sexes`. Radius accepts 1–8 and
defaults to two orthogonal dry-cell path steps.

For old saved brains, omission of `mode` retains `separated` behavior. In that
mode, the ordered `species` list accepts 1–8 exact configured names, and
`separate_sexes` defaults to true; false merges sexes within each species.
Explicit `"mode": "separated"` is equivalent to the old omitted-mode contract.
Scent input configuration requires `scent_trails.enabled: true`. To disable the
field entirely, remove those brain input blocks before disabling the root flag.
Trail lifetime accepts 2–256 steps. These are simulation controls, not calibrated
biological timescales.

Different brain templates may use different history lengths, scent modes and
smell radii; separated mode also chooses species and sex channels. Animals of every configured species leave traces
while the world field is enabled, including species that do not sense scent.

## Personal visual history

Only animals selected into the observer's visible top-k resource-eater and
prey-eater slots contribute sightings. Food has no movement history. The
existing visibility rules still apply: sight radius, sunlight modifiers, water
extensions and wall occlusion. A newly detected animal has no backfilled past.
Internal animal IDs match sightings across distance-slot changes; IDs are never
passed into brain inputs or the inspector's scent plots.

For each diet category, current visible targets receive tracking slots first.
Remaining slots can hold the most recently seen targets still within the history
window. A hidden target retains its **last observed** position and velocity
estimate, without updates from its actual position, health, reproduction or
hidden death. Confidence decreases with time since sighting and the memory
expires after the configured window. Missing sightings remain explicitly absent;
reappearance does not fill gaps. A velocity estimate can use two sightings
separated by a gap, but describes average observed displacement rather than a
known route through the gap.

Historical positions are relative to the observer's current position, using the
same current sight normalization as ordinary target slots. This compensates for
observer movement: a stationary target does not gain fictitious velocity when
the observer moves. Last-seen positions may lie outside the current sight circle.
A newborn starts with no personal past; it can acquire its first sightings at
birth. It does not inherit either parent's movement memories.

## Anonymous ground scent

Each occupied dry cell records the most recent deposit time for each species/sex
group. Multiple animals in the same group refresh one merged trace; they do not
create identifiable trails or per-animal concentrations. Standing still also
refreshes a trace. Scent can persist after its maker moves away or dies.
An animal's own scent is included in its group: the field does not secretly
subtract it using individual identity.

Freshness decreases linearly from one to zero over `lifetime_steps`. Detection
samples reachable dry cells within the configured path radius. For each sensed
group in legacy separated mode, the brain receives the strongest remaining
freshness in **each** of nine sectors: here, N, NE, E, SE, S, SW, W, NW. Two intersecting trails can therefore
produce several directional signals; there is no chosen prey or destination.
These are sector summaries, not a full local image or a complete historical path.

In blended mode, freshness from all species/sex groups is summed **at each
reachable cell**, capped at one. Each sector reports its strongest blended
cell, producing nine values total. This reuses the same bounded sparse deposits
without creating another world map. Repeated deposits within one group refresh
its timestamp, so strength is a bounded trace signal, not an animal count or a
physical concentration. The brain cannot identify a species, sex or individual
from a channel label. Own scent remains included.

Sealed obstacles and wet cells stop ground-scent sampling; no traces are
deposited in water. This is a deliberately simple ground-trace model, without
wind, airborne diffusion, transported river scent or terrain-specific chemistry.
Scent range is separate from sight and does not change with daylight. Species and
sex channels do not reveal individual identity, age, health or mating consent.

## Input contract and model compatibility

The existing observation-version 1–4 blocks stay in their current order. Optional
movement and scent blocks are appended afterwards. For `K` visible slots per
category and `H > 0` history steps, movement adds `2 * K * (7 + 3 * H)` values:

| Per tracking slot | Meaning |
| --- | --- |
| 7 summary values | Last-seen dx/dy, estimated velocity x/y per step, normalized age, confidence, currently-visible flag |
| 3 values per previous step | Historical dx/dy and observed flag, newest lag first; zeros mark unknown steps |

Blended scent adds nine values total. Separated scent adds nine values per
channel: one channel per sensed species, or female then male channels per species
with sex separation enabled. Legacy channel order follows `scent_observation.species`. Version 1, K=1, H=3 and two sex-separated species has
`20 + 32 + 36 = 88` inputs. The tracking preset's version-4 prey K=3 and predator
K=10 now have 174 and 482 inputs respectively with blended scent.

Model signatures include history length and the ordered scent contract. Changing
those fields, including scent mode, requires fresh compatible weights; equal input width alone is not
sufficient. Explicit history zero retains the legacy signature. Old recordings
and configurations remain readable and keep their original input layouts.

## Inspector and replay

![Actual inspector SVG components with controlled movement and scent samples](assets/viewer/tracking.svg)

Select an animal during saved playback, then click **Tracks & scent charts** in
the inspector to jump directly to **Movement memory & scent**. This section is
below **Sampled observation** in the independently scrollable sidebar and shows
its recorded sensory inputs. Movement
dots are personal sightings; missing steps break the connecting line. Hollow
markers and fading opacity identify remembered targets. Dashed line segments in the inspector show the
last velocity estimate, not confirmed future movement. Prey/predator colors
refer to the existing diet categories. The scent panel shows the local
blended strength pattern in a single heatmap (or the original species/sex grids
for separated recordings), with north upwards and
the observer's current cell in the center. Compact headings keep the diagrams
stable; hover a movement marker for its slot, visibility, age and confidence
instead of repeating those details in paragraphs below the chart.

Panels update at diagnostic samples as playback advances, rather than inventing
intermediate data. Use a nonzero `recording.diagnostics_every_steps`; lower
intervals give smoother updates and larger recordings. The existing diagnostic
step label distinguishes the sampled observation from the current replay frame.
Old recordings without these inputs do not acquire retrospective tracks.

## Selected-agent trails on the board

**Show sampled tracks on board** displays the selected animal's observed target
histories as simple connected dots, without projected arrows. Nothing is drawn without an agent
selection or supported visual-history sample. Scent remains in the inspector;
there is no world scent/hot-path overlay on the board. The origin is the selected
observer's recorded position at the diagnostic tick, not its later body position.
Unknown gaps remain disconnected. The inspector status names both the sample tick
and current body tick. At step zero there are no previous sightings yet; advance
the replay to see movement histories develop.

<a id="whole-run-movement-heatmap"></a>

## Replay-time movement heatmap

**Run movement heatmap** jumps to **Behavior & learning**, where an all-animal
heatmap accumulates arrivals **only through the current replay frame**. It updates
while playing and rewinds when seeking backward. Seek to the end to see the
entire run. Other behavior/learning tables still summarize the full saved run;
this heatmap is explicitly labeled with its current ending step.

Heat counts changed-position arrivals per destination cell across all species
and sexes, including drift and displacement when they change the final position.
Stationary frames and newborn placements add no heat. Only consecutive recorded
frames are compared; missing intervals and unknown terminal routes are not
interpolated. Known terminal destination positions can contribute arrivals.
A combined movement/drift step contributes its known final cell, not an invented
path through intermediate cells. The logarithmic color scale preserves lightly
visited areas; dark mode uses increasing-brightness blue/teal/yellow colors.
The numeric color key shows arrival counts, and gray cells are obstacles.

The browser retains one reusable unsigned-32-bit count grid and previous animal
positions, reusing its already loaded replay chunk for forward playback and
rewinds within that chunk. At 40 × 28 cells, the count buffer is 4,480 bytes,
plus small metadata and the previous-position map. There is no per-frame grid
history, grid copy on each replay step or extra decoded run in RAM.

For a jump outside the available frames, the read-only movement-prefix endpoint
streams saved chunks through the requested frame and returns one count grid.
The server retains one chunk and prior positions while rebuilding, and discards
them afterwards. The browser keeps its single current chunk; it never downloads
the entire run for this rebuild. Large jumps trade recomputation for bounded
memory. Full-run behavior responses omit their unused heatmap grid; the CLI
analysis report may still include a full-run grid by default.

No heatmap file or new replay field is written. Existing recordings work without
regeneration, even when the optional senses were disabled. Restart an already
running playback server after updating Python reader code. Future frames never
contribute to the displayed prefix. This is analyst information and is not sent
to animal brains or drawn as a global hot-path layer on the main board.

## Memory and storage boundaries

Personal storage keeps at most H+1 frames per living observer, each containing
only its selected animal sightings. Dead observers are removed and newborns get
new empty histories. The sparse scent field keeps one integer deposit time per
cell/species/sex group; refreshing a cell replaces its record. An ordered map
expires oldest records without scanning a full map, and a bounded 16-entry plan
cache shares static scent-channel lookups between observers.

Exact checkpoints store a versioned `world.sensory_state`: personal sighting
records and sparse deposit times. Restore and GRPO trials receive independent
mutable copies. Weight-only champions contain neither movement memory nor scent
state; a fresh world begins with new founder traces. Missing/unsupported sensory
state is rejected when restoring a checkpoint that enables these features.

Replays store **no full scent map or separate trajectory log**. The inspector
reads the extra values already carried by sampled observation vectors through
the existing lossless binary/compressed chunk codec. Enabling senses still grows
input layers, training buffers, checkpoints and sampled diagnostics; cost depends
on H, K, sensed channels, population and sampling interval. Whole-experiment
decoding is not required, and read-only playback does not advance either sense.

## Biological interpretation

The visual buffer approximates short-term movement estimation, not access to an
animal's true route. Predictive interception has been demonstrated in
[dragonflies](https://www.nature.com/articles/nature14045); ground-odour following
has been studied in [rats](https://www.nature.com/articles/ncomms1712) and
[dogs](https://pubmed.ncbi.nlm.nih.gov/8371085/). These motivate separate visual
memory and environmental traces. Fixed windows, linear fading and grouped sector
strengths are engineering abstractions, not universal biological laws.
