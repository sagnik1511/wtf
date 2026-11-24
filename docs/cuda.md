# Experimental CUDA training

[Performance](performance.md) · [Installation](installation.md) · [Running experiments](running.md)

The CPU backend remains the default. The optional CUDA backend batches independent
animals' dense networks on Linux/NVIDIA without sharing their weights or adding a
mandatory ML framework. This is a source-branch prototype, not an established GPU
speedup. It has not yet been compiled or executed on NVIDIA hardware by its author.
Use the real-device tests and full-loop benchmark below before adopting it.

## What moves to the GPU

| Learner | GPU work | Work that remains on CPU |
| --- | --- | --- |
| DQN / DDQN | Dense action evaluation; sampled TD targets, Huber loss and backpropagation | Replay sampling, exploration, gradient clipping, Adam, target synchronization |
| PPO / masked PPO / A2C | Dense actor/value evaluation during actual world steps | Rollout preparation, losses, backpropagation and optimizer |
| REINFORCE / GRPO | Dense action evaluation during actual world steps | Policy gradients, optimizer and GRPO trial worlds/evaluation |
| MAPPO | Dense local actor and centralized critic evaluation | Context preparation, gradients and optimizer |
| TRPO / discrete SAC | Dense action-selection actor; TRPO value evaluation | Trust-region/replay work, gradients and optimizers |
| QR-DQN | Dense action evaluation | Quantile targets, gradients and optimizer |
| Recurrent PPO / LSTM PPO / legacy `q_network` | None in this prototype | Entire learner |

This does **not** make PPO training fully GPU-resident. Policy inference may be a
small fraction of its runtime; a faster matrix evaluation alone is insufficient
evidence of faster training. Unsupported or customized brains retain their CPU
path in a mixed experiment. An entirely unsupported population rejects CUDA
selection. Legacy non-FP32 parameter containers retain their CPU path.

World physics, observations, champion selection, births, recording and the viewer
remain on their existing paths. The saved artifacts have the same formats.

## Build on Linux or Colab

Use a checkout containing this implementation (including `tests/` and `work/` for
validation), Python 3.12+, an NVIDIA GPU/driver and the
[NVIDIA CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
Select a GPU runtime in Colab first. The published package may predate this branch;
install your checkout rather than assuming the released wheel contains the prototype.

```sh
python -m pip install -e .
nvidia-smi
nvcc --version
python -m wtf.native.cuda_build
WTF_TEST_CUDA=1 python -m unittest tests.test_cuda -v
```

The explicit build uses NVCC, C++17 and the CUDA runtime through a small `ctypes`
ABI; it does not require PyTorch, CuPy or NumPy. Imports never launch a compiler.
The default architecture is `native`. If the toolkit requires an explicit target,
use `--arch sm_75` for a T4, or the architecture of your actual GPU. Build locally
for the destination GPU; do not copy a Mac/another GPU's binary into a wheel.

`--output /path/to/_cuda_dqn.so` changes the build destination. Set
`WTF_CUDA_LIBRARY=/path/to/_cuda_dqn.so` when loading that location. The portable
wheel and source archive ship CUDA source, not a precompiled CUDA binary. Rebuild
after changing kernel source. Driver/toolkit errors stop the run visibly.

## Run and resume

```sh
python -m wtf run ppo --runs 5 --seed 100 --progress --device cuda
python -m wtf run maskable_ppo --runs 5 --seed 100 --progress --device cuda
python -m wtf run grpo --runs 5 --seed 100 --progress --device cuda
```

The historical `dqn` preset uses the legacy `q_network` controller. For the dense
DQN/DDQN gradient kernel, use a JSON whose brain `type` is `dqn` or `ddqn`, as
explained in [policies](additional-policies.md); the benchmark constructs both from the
maintained QR-DQN preset with its quantile setting removed.

`--cuda-batch-agents 64` caps jobs per batch. `--cuda-workspace-mb 256` caps cached
device allocations in MiB, excluding CUDA context/driver memory and host replay
buffers. Reduce either for smaller GPUs. A single model/sample batch exceeding
the budget fails explicitly. Workspaces are reused by architecture, population
count, sample count and prediction/training mode, with least-recently-used eviction.
Prediction buffers do not allocate replay, target or per-sample gradient arrays.

For Python callers, use `experiment.set_training_device('cuda')` after creating or
restoring an `Experiment`. `set_training_device('cpu')` releases its device cache.
Checkpoints retain weights, optimizer state, replay and RNGs on the CPU. Device
handles and backend choices are transient: restore defaults to CPU; explicitly
pass `--device cuda` again to accelerate a resumed CLI run. Evaluation mode is
CPU-only in this prototype.

## Benchmark all learners, including transfers

From the same source checkout on the GPU machine:

```sh
python -m work.benchmark_cuda --steps 256 --warmup 16 --repeats 3 \
  --output .cache/cuda-comparison.json
```

The default list covers all eleven supported dense learners. For deeper models:

```sh
python -m work.benchmark_cuda --algorithms ppo maskable_ppo a2c reinforce grpo \
  dqn ddqn mappo qrdqn trpo discrete_sac --hidden 64 64 32 \
  --steps 512 --warmup 32 --repeats 3 --output .cache/cuda-deep.json
```

Both species use the named learner; nonmatching legacy controllers in a mixed
preset are replaced with that preset's named learner. Each CPU/GPU case uses a fresh subprocess with matched seed/configuration. The
report includes startup, complete `Experiment.step` time, warmup/steady timing,
step/update counts, process peak RSS, device workspace use and actual kernel-job
counts. No recording files are written. Packing, transfers, CPU optimizers, world
steps and episode finalization are timed. Do not count a zero-update test as a
training benchmark; increase the horizon if the configured update schedule has
not started. REINFORCE may update only at episode end. The script suppresses the
speedup ratio when update/step counts differ, no updates occur, or no GPU jobs run. Matching counts
alone do not establish identical work or equivalent learned behavior.

For CPU native baselines, build the existing CPU extension first using the
[development workflow](development.md). Without it the baseline is portable Python,
which must be identified when reporting numbers. `--devices cpu` is a local harness
smoke check and provides no CUDA performance evidence.

## Numerical and ownership boundaries

CUDA uses FP32 stored parameters with FP64 arithmetic/activations/gradients to
follow the existing arithmetic contract closely. It disables fused multiply-add
contraction. GPU reductions and `tanh` still differ from CPython's compensated sums
and platform math: **GPU trajectories are not guaranteed bit-for-bit identical**.
The default CPU path is unchanged. This prototype does not silently reduce precision
for speed; modest GPUs can be particularly slow at FP64.

Each animal retains its own parameters, optimizer, replay and RNG. The scheduler
collects ready DQN jobs, computes independent gradients, then applies the existing
CPU update before champion selection and child inheritance. No device work is left
in flight at the checkpoint boundary. Sampling and action probabilities remain
with the owning brain; masks still apply. Failed device work raises an error,
rather than silently switching algorithms. It is not a transactional retry of an
already partially advanced world: resume a previously completed checkpoint.

CPU-reference tests verify packing/scheduling against existing learners, including
mixed recurrent populations and multiple species. Only `WTF_TEST_CUDA=1` exercises
the actual shared library; a missing compiler/device must not be reported as a
passed GPU test. Follow [NVIDIA's best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
by comparing complete workloads including transfers. Keep CPU when it is faster.
