# Artifact delivery status

## Current state

- Branch: `codex/paper-mvp`
- Runtime: Python 3.10+, standard library only
- Public data: one sanitized synthetic file at `data/sample.json`
- Paper v4 mechanisms: implemented and mapped in `docs/PAPER_IMPLEMENTATION.md`
- Architecture: seven focused package modules with a stable compatibility surface
- Automated checks: eighteen focused tests pass
- Production data and experiment artifacts: intentionally excluded

## Verified behavior

The bundled sample exercises all three paper routes:

| Query | Route | Output |
| --- | --- | --- |
| `query-common-001` | common | `fiber` |
| `query-ambiguous-001` | ambiguous | `L1` |
| `query-novel-001` | novel | `L2` |

Low-information inputs use `request_evidence` or `human_review` rather than a
forced final class. Knowledge changes require an explicit operator identity and
confirmed label and produce a new content-hashed repository version. Periodic
consolidation removes exact duplicates while retaining mixed-label patterns for
audit. Monthly heuristic refinement remains validation-pending, and deprecated
knowledge is retained for rollback but excluded from online retrieval.

## Scope note

This is a readable, modular implementation artifact. The private production corpus,
full evaluation pipeline, and generated experimental outputs remain outside the
branch and are not represented by the sample.
