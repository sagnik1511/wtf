# X.X.X.X Descriptive rollout title

[Release hub](README.md)

| Field | Value |
| --- | --- |
| Version | `X.X.X.X` |
| Status | Pending |
| Prepared | YYYY-MM-DD |
| Implementation commit or range | Not committed |
| Base version and commit | Fill in the committed base |

## Overview

Describe the concrete problem and the resulting behavior for a reader who has
not seen the development conversation. Describe an implementation or adopted
configuration change; do not create a record for executing experiments.

## Changes

- Describe meaningful changes and the affected components.
- Link maintained configuration or operational guides when relevant.

## Compatibility and migration

State default/configuration changes, saved-model/checkpoint/replay compatibility,
requirements and any user migration steps. For documentation-only work, say so.

## Fixes and optimizations

The `wtf-reviewer` fills this section after reviewing the stable release diff.
Record verified fixes with their trigger, impact and regression check. Record
adopted optimizations with preserved contracts, measured workload/baseline and
results; label unmeasured changes explicitly. Link detailed maintained guides
rather than duplicating benchmark reports. State "None verified in this review"
for an empty category, and distinguish pending review from a completed clean pass.
Include unresolved findings or owner decisions with their practical impact.
Do not invent fixes or measurements to fill this section. This section does not
authorize new experiments, optimizations, or rewriting historical release records.

## Validation

Record checks actually performed, their result, and the scope they cover.
Identify any material unverified behavior. Give benchmark workload and baseline
with performance numbers; do not turn an unmeasured expectation into a result.
Exclude individual experiment IDs, ecological observations and campaign history.

## Rollout and rollback notes

Describe adoption steps, restart requirements, data to preserve and the reversal
boundary. After the implementation commit exists, add its real hash/date and
change the status to Committed. Link a reversal if one is later applied.
