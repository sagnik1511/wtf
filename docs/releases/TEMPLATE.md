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
not seen the development conversation.

## Changes

- Describe meaningful changes and the affected components.
- Link maintained configuration or operational guides when relevant.

## Compatibility and migration

State default/configuration changes, saved-model/checkpoint/replay compatibility,
requirements and any user migration steps. For documentation-only work, say so.

## Validation

Record checks actually performed, their result, and the scope they cover.
Identify any material unverified behavior. Give benchmark workload and baseline
with performance numbers; do not turn an unmeasured expectation into a result.

## Rollout and rollback notes

Describe adoption steps, restart requirements, data to preserve and the reversal
boundary. After the implementation commit exists, add its real hash/date and
change the status to Committed. Link a reversal if one is later applied.
