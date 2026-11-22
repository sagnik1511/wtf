# Learning and behavior

[Documentation home](../index.md) · [Complete directory](../directory.md)

Each animal owns its controller. Start with the observation and reward contracts,
then choose a learner and decide how descendants and champions inherit knowledge.

| Task | Guide |
| --- | --- |
| [Choose algorithms and network depth](../algorithms.md) | Private learning, architectures and supported algorithms. |
| [Use advanced algorithms](../advanced-algorithms.md) | Recurrent, distributional, masked and multi-agent variants. |
| [Track observed movement and nearby scent](../tracking.md) | Optional personal memory, blended or legacy separated scent and inspector plots. |
| [Understand brain inputs](../observations.md) | Top-k target slots, input layouts and visibility. |
| [Understand feedback](../rewards.md) | Action outcomes, effort and failure penalties. |
| [Review feedback contracts](../feedback-review.md) | Shared accounting across environments and policies. |
| [Understand mating inputs](../mating-observations.md) | Readiness and eligibility signals. |
| [Understand mating intent](../mating-intent.md) | Consent windows and failed mating reasons. |
| [Understand escape and visible consent](../escape-and-consent.md) | Relative movement and observable intent. |
| [Understand mutation](../mutation.md) | Rare changes to newborn weights. |
| [Choose champions and inheritance](../brain-library.md) | Species/sex archives, lineage selection and warm starts. |
| [Analyze behavior](../analysis.md) | Saved metrics and limits of learning claims. |
| [Design an evaluation](../evaluation.md) | Matched seeds and controlled comparisons. |

For larger campaigns, use [training-only capture followed by recorded frozen inference](../running.md#train-without-replay-then-record-inference).
