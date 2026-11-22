# GRPO trials and independent actor batches

[Performance](performance.md) · [Algorithms](algorithms.md) · [CPU kernels](native-kernels.md)

These optimizations remove trial construction and inference-dispatch overhead.
They preserve independent policies, ordered sampling, learning cadence and world
physics. Batching means several independent forward passes in one C call; it
neither merges weights nor changes the arithmetic to a shared matrix policy.

![Group-private GRPO resets and independent model batches](assets/diagrams/trial-actor-batching.svg)

## Group-private GRPO state

One world is restored for a collected group, with a private configuration copy.
Later trials reset that object to the same starting state. Animals, resources,
obstacles, counters and events are reconstructed independently; world RNG state
is reset exactly, and all derived per-trial caches are discarded. Current
in-memory records skip legacy/default-field adaptation and existing RNG tuples
skip recursive conversion. Ordinary animal construction and authoritative world
attribute layout are retained for efficient hot-path reads. Public/legacy
checkpoint restoration keeps its compatibility/default-field handling.

The initial observation dictionary contains immutable tuples and is reused
within the group because all trials start from exactly the same world state.
Action masks and physical mating contexts are rebuilt from each trial's current
bodies. The group retains that initial dictionary until collection returns;
this is an explicit memory tradeoff. Trial policies stay frozen except for
independent newborn inheritance/mutation. Recurrent memory and action RNGs
remain trial-local; no real controller's `choose` or `learn` runs in a trial.

The group-private configuration and shared read-only geography are not reset
through copying each trial. Physics treats configuration as immutable, as in
ordinary runs. There is no new cross-group/world cache or stored checkpoint field.
The retained world, initial observations and query scratch end with collection.

## Independent actor inference

The optional `mlp_predict_many` kernel accepts independent dense models and
observations. Weights are borrowed read-only, with no packed weight copies,
shared optimizer state or model-sized persistent allocation. One query-local
activation workspace grows to the largest model required in that call and is
reused across jobs; all layers retain the scalar kernel's ordered reductions.
Output vectors and job descriptors are bounded by the current decision phase.

PPO, A2C, REINFORCE, GRPO and masked PPO can use the batch path. MAPPO training
batches its separate local actor and centralized critic; greedy execution still
skips the critic. Mask packing, sampling, pending experience and every animal's
RNG remain Python-owned and ordered as before. Learning occurs afterward in its
existing order. Frozen compatible policies inside GRPO also use batched forward
passes, followed by the existing per-trial distributions and samplers. Their
model-qualification plan belongs to the collected group: it records eligible
frozen network objects once, copies its small mapping per trial and updates that
mapping for births/deaths. It caches no predictions or live training state,
never crosses groups and is discarded with collection.

Q-network, QR-DQN and recurrent controllers retain scalar decisions. Random
baselines, legacy list/FP64 models, custom/subclassed policies, overridden
predictors, single eligible actors and missing/older extensions use their
reference paths. Rebuild explicitly to enable batching; imports never compile:

```sh
.venv/bin/python -m work.build_native
```

No observations, actions, model signatures, recording cadence or artifact formats
change. Playback and diagnostics still evaluate the correct saved/sample state;
pre-action logits are not reused as post-learning recording diagnostics.

## Reproduce the comparison

```sh
WTF_KERNELS=native .venv/bin/python -m work.performance.brain_workflow \
  --environment environments/grpo.json --case train --steps 360 \
  --samples 5 --components --output .cache/grpo-workflow-benchmark.json
```

The reference mode uses fresh worlds/initial observations per trial and scalar
policy inference. Other kernels and prior optimizations stay enabled. Optional
`--components` separates trial reuse, batching and their combination. Each mode
uses five fresh processes in alternating order, seed 100 and one fixed-seed run.
All state and replay hashes must agree. Temporary recordings are cleaned after
each sample; measured artifacts remain ignored.

`--case choose` measures repeated decisions on unchanged observations;
`--case groups` measures repeated collection from one unchanged world. They
exclude the main experiment's initialization. `train` and `record` include
initialization and the real learning loop. Imports and final digest construction
are excluded; peak RSS includes process initialization and is read before hashing.

## Measurements and validation

Local macOS / CPython 3.12.9 medians use five fresh processes per mode,
seed 100 and the preset configurations named below. These compare scalar/fresh
trial reference paths against the current workflow, with all previous native
math, perception and champion optimizations enabled on both sides.

| Workload | Reference | Both optimizations | Observed result |
| --- | ---: | ---: | --- |
| 16 GRPO group collections, `grpo.json` | 0.2939 s | 0.2629 s | 1.12×; 10.6% less time |
| 1,000 MAPPO decision phases, `mappo.json` | 0.3544 s | 0.3314 s | 1.07×; 6.5% less time |
| 360 training steps, `grpo.json` | 0.7804 s | 0.7424 s | 1.05×; 4.9% less time |
| 240 training steps, `competition.json` | 1.2010 s | 1.1927 s | Approximately unchanged; 0.7% less time |
| 120 steps plus recording, `meandering_river.json` | 4.1455 s | 4.1574 s | Approximately unchanged; 0.3% more time |

The default GRPO cadence is 128 steps. Its 360-step probe collected **23 groups**;
a 120-step probe would not exercise that learning schedule. The mixed river
recording collected **36 groups** under its own maintained settings. No cadence,
population, policy architecture or reproduction setting was changed for the
measurement. The group-only probe uses four trials with four steps each; the
MAPPO decision probe does not advance world time or train the models.

Component controls show which change accounts for the measured difference:

| Workload | Reference | Trial reuse only | Batching only | Both |
| --- | ---: | ---: | ---: | ---: |
| GRPO collections | 0.2939 s | 0.2662 s | 0.2879 s | 0.2629 s |
| GRPO training | 0.7804 s | 0.7393 s | 0.7701 s | 0.7424 s |
| Mixed river recording | 4.1455 s | 4.1880 s | 4.1223 s | 4.1574 s |

Trial reuse accounts for most of the GRPO improvement in these workloads.
Batching does not uniformly accelerate whole training, and the component gains
are not additive. Small timing differences may be noise; these measurements do
not establish a dramatic or universal speedup.

| Peak process RSS | Reference | Both |
| --- | ---: | ---: |
| GRPO collections | 33.92 MiB | 33.86 MiB |
| MAPPO decisions | 33.50 MiB | 33.55 MiB |
| GRPO training | 41.11 MiB | 41.17 MiB |
| Competition training | 43.88 MiB | 44.05 MiB |
| Mixed river recording | 44.00 MiB | 43.94 MiB |

There is no meaningful peak-RAM reduction in these probes. Initial-observation
retention, compact policy plans and batched output vectors offset allocation
savings; workspace reuse is not a claim of lower RSS on every setup.

For a four-trial group, world restoration/configuration copies and initial
observation construction fall from four to one. All four trajectories still
execute their full simulator steps. Eligible actors' dense jobs enter native
code together (one entry rather than per-model entries), while sampling remains
per animal. No learning or diagnostics work is silently skipped.

Every compared state and replay hash matched, including **202,676 compressed
replay bytes** for the mixed river recording. Thus this optimization adds no
recording data and does not change the saved viewer inputs. Regression tests
also compare all nine algorithms across training, evaluation, random decisions,
run chaining and exact resume, with chunk-boundary replay/activation checks.
Birth/mutation, recurrent/masked/centralized trial policies, terrain and day/night
phases are included. A separate comparison against the pre-change source snapshot
also matched per-step state and canonical compressed replay hashes in 12 cases
(all nine algorithms plus three maintained environment presets). See [rollout 0.0.2.4](releases/0.0.2.4.md) for final validation.

