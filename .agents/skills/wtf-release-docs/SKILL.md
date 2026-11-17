---
name: wtf-release-docs
description: Maintain Wildlife Trial Framework four-component release records and the rollout hub for implementation or documentation changes, using verified Git history.
---

# Maintain WTF rollout records

Read [hub conventions](../../../docs/releases/README.md) and use the existing
[template](../../../docs/releases/TEMPLATE.md). Inspect `git status`, recent log,
relevant diffs and existing pending records before choosing the next version.

Each rollout has a child `docs/releases/X.X.X.X.md`, a hub row and valid navigation.
Use the next unused four-component version; extend an existing pending rollout
when it is the same scope. Lead with the problem and resulting behavior. Record
affected configuration/formats, migration or restart needs, validation actually
performed and practical rollback boundaries. Keep operating instructions in the
relevant guide and link them instead of expanding the README.

Uncommitted work is Pending with its actual committed base. Only mark Committed
after inspecting a real implementation hash/date. If the user committed an older
pending rollout, reconcile it with that hash as part of maintenance; one commit
can cover several records. A hash-reconciliation edit does not itself require
another application release. Keep latest committed runtime separate from latest
documentation rollout when applicable. Do not invent tags, commit IDs or pass
counts, change package versions, or create commits/tags unless requested.

Check new/changed relative links, previous/next links, version uniqueness and
that every claimed commit exists. For documentation-only work, validate links,
assets or skill metadata as relevant; avoid unrelated training/test runs. Describe
unverified behavior plainly. Record rollout status accurately in the final report.
