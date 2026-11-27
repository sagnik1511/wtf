# Development

[Documentation home](../index.md) · [Complete directory](../directory.md)

Use the implementation boundaries and extension guides before changing shared
contracts. Validate the behavior affected by a change and format all project
source using the maintained development workflow.

| Task | Guide |
| --- | --- |
| [Navigate the Python package](../package-layout.md) | `wtf` namespace, module ownership and import migration. |
| [Follow calls and callers](../source-graphs.md) | Searchable source, outgoing call graphs and incoming caller graphs. |
| [Understand the architecture](../architecture.md) | Simulation phases, data ownership and Python/viewer separation. |
| [Set up and test development](../development.md) | Checks, formatter commands and contribution workflow. |
| [Add a policy](../extending-policies.md) | Dispatch, signatures, persistence, inheritance and diagnostics. |
| [Add geography or actions](../extending-worlds-and-actions.md) | Physics, masks, observations, feedback and replay contracts. |
| [Build installable packages](../packaging.md) | Wheel/sdist resources and isolated installation checks. |
| [Review shared performance paths](../performance.md) | Caches, repeated work and bounded memory. |
| [Run native PPO episodes](../native-engine.md) | C++ execution with detached Python queries and existing artifacts. |
| [Use optional native kernels](../native-kernels.md) | Compiled paths and portable fallbacks. |
| [Review native top-k perception](../native-top-k.md) | Selection boundary, parity checks and reproducible benchmarks. |
| [Review extended native kernels](../native-kernels-v2.md) | Additional native coverage and equivalence boundaries. |
| [Review native performance](../native-performance-review.md) | Coverage, dtypes and benchmark evidence. |
| [Read contributor context](../project-context.md) | Project rules and generic extension contracts. |
| [Maintain this documentation site](../documentation-site.md) | Zensical structure, link/theme checks and Pages deployment. |
