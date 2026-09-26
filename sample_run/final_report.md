# NADI-9 Final Report

## Run Summary
- **Run ID**: `run-49c4d148`
- **Episode ID**: `EP001`
- **Timestamp**: `2026-09-26 10:56:10.031155+00:00`
- **Total Subtitles Processed**: 3
- **Accepted Count**: 1
- **Human Review Required Count**: 2
- **Abstained Count**: 1

## Final Release Recommendation
**HOLD - HUMAN REVIEW REQUIRED**

## Decision Summary

| Subtitle ID | Source Text | Nadi-9 Result | Status | Confidence | Human Review |
|---|---|---|---|---|---|
| `EP001-L001` | formal greeting to elders | Namaskar. | `accepted` | `high` | No |
| `EP001-L002` | informal farewell to friends | Alvida. | `human_review` | `medium` | Yes |
| `EP001-L003` | rare archaic idiom | Translation unavailable: insufficient supporting evidence. | `human_review` | `low` | Yes |

## Evidence & Verification
- **Total Evidence Items Cited**: 3
- **Source Precedence**: Gold approved examples prioritized over secondary dictionaries and crowdsourced feedback.
- **Independent Verification**: Deterministic checks enforced for proper name preservation, number/quantity preservation, and subtitle reading speed (max 25 cps).
- **Verification Failures**: 1
- **Explicit Abstentions**: 1

## Learned Rules
None generated during this run.

## Human Review Queue
- **Pending Review Items**: 2
- **REV-EP001-L002-1** (`EP001-L002`): Please resolve the conflicting evidence and confirm which interpretation should be used for this subtitle. Conflict IDs: CONFLICT-E-002-E-003.
- **REV-EP001-L003-1** (`EP001-L003`): Please review the proposed translation because independent verification failed: No hypothesis or evidence available..

## Audit / Traceability
- **Evidence-Grounded**: Every candidate translation cites verified evidence records.
- **Independent Verification**: Deterministic verification checks ran prior to decision assembly.
- **Deterministic Source Precedence**: High-authority sources overrule lower-ranking dictionary entries.
- **Human-in-the-Loop Interrupts**: Suspended execution state saved in SQLite checkpointer when review is required.
- **Learned Rules**: Learned rules record superseding pointers and affected subtitle IDs for targeted replanning.
