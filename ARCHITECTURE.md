# NADI-9 Architecture & System Overview

Nadi-9 is an evidence-grounded agentic subtitle decision system built for the STAGE Nadi-9 assignment. It combines LangGraph workflow orchestration, Pydantic domain models, LLM provider abstractions, deterministic evidence ranking & verification, stateful HITL interrupts, durable SQLite checkpointing, database persistence, FastAPI REST services, Typer CLI, and production Docker containerization.

---

## 1. End-to-End System Pipeline

```text
                  ┌──────────────────────┐         ┌──────────────────────┐
                  │    Typer CLI App     │         │  FastAPI Web Service │
                  │  (`src/nadi9/cli.py`)│         │  (`src/nadi9/api/`)  │
                  └──────────┬───────────┘         └──────────┬───────────┘
                             │                                │
                             └────────────────┬───────────────┘
                                              ▼
                                   ┌────────────────────┐
                                   │  EpisodeProcessor  │ (`src/nadi9/processor.py`)
                                   └──────────┬─────────┘
                                              │
             ┌────────────────────────────────┴────────────────────────────────┐
             ▼                                                                 ▼
      JSONL Ingestion                                                 SQLAlchemy Database
(`src/nadi9/ingestion/jsonl.py`)                                 (`src/nadi9/storage/`)
             │                                                                 │
             ▼                                                                 │
      Retrieval & Ranking                                                      │
(`EvidenceRetriever` & `EvidenceRanker`)                                       │
             │                                                                 │
             ▼                                                                 │
      LangGraph Stateful Workflow Graph (`src/nadi9/graph/workflow.py`)        │
             │                                                                 │
┌────────────┼────────────────────────────────────────┐                        │
▼            ▼                                        ▼                        │
plan_episode generate_hypothesis                detect_conflicts               │
│            │                                        │                        │
└────────────┼────────────────────────────────────────┘                        │
             ▼                                                                 │
      verify_decision                                                          │
             │                                                                 │
             ▼                                                                 │
      generate_decision                                                        │
       /            \                                                          │
      ▼              ▼                                                         │
  ACCEPTED       REVIEW REQUIRED                                               │
      │              │                                                         │
      │              ▼                                                         │
      │       Stateful Interrupt (`interrupt()`)                               │
      │              │                                                         │
      │              ▼  (Persisted to `SqliteCheckpointSaver`)                 │
      │        PROCESS RESTART RECOVERY SAFE BOUNDARY                          │
      │              │                                                         │
      │              ▼                                                         │
      │        HUMAN ACTION (Approve / Reject / Correct)                       │
      │              │                                                         │
      │              ▼                                                         │
      │       Workflow Resume (`Command(resume=action)`)                       │
      │              │                                                         │
      ▼              ▼                                                         │
       \            /                                                          │
        ▼          ▼                                                           │
        finalize Node                                                          │
             │                                                                 │
             ▼                                                                 │
  Artifact Exporter (`src/nadi9/artifacts/exporter.py`) ─────────────────────────┘
      │
      ▼
  Standard Package Artifacts (`subtitles.srt`, `subtitle_decisions.jsonl`, `learned_rules.json`, `review_queue.json`, `final_report.md`)
```

---

## 2. Component Categorization

| Component Category | Subsystems / Modules | Execution Nature | Description |
|---|---|---|---|
| **Deterministic Components** | `EpisodePlanner`, `EvidenceRanker`, `HypothesisVerifier`, `ConflictDetector`, `ArtifactExporter` | Strict Logic / Zero Randomness | Analyzes risk factors, ranks evidence by authority precedence, verifies proper names/numbers/timing, detects conflicts, and exports artifacts. |
| **LLM Provider Components** | `OpenAIProvider`, `MockLLMProvider`, `BudgetedLLMProvider`, `generate_hypothesis()` | LLM / Model Dependent | Generates evidence-grounded candidate hypotheses and translations. Governed by budget limits and provider retries. |
| **Human-in-the-Loop Components** | `Nadi9Workflow` Interrupts, `ReviewManager`, CLI `review`, REST `/reviews/` | Human Interactive | Suspends graph execution via `interrupt()` when evidence is missing/conflicting or verification fails; resumes via human action (`approve`, `reject`, `correct`). |

---

## 3. Core Architecture Details

### A. Structured Episode Planning (`EpisodePlanner`)
- Operates prior to line processing to evaluate risk factors for episode lines (`RELATIONSHIP_KEYWORDS`, `TONE_KEYWORDS`, `CULTURAL_RARE_KEYWORDS`, Proper Names, Numbers).
- Produces an `EpisodePlan` containing sorted `SubtitlePriority` records.
- Supports targeted replanning (`plan_subset()`) for re-processing only affected subtitle subsets.

### B. Source Precedence Ranking (`EvidenceRanker`)
- Ranks retrieved evidence by authority hierarchy:
  `APPROVED_EXAMPLE` (`+0.50`) > `EXPERT_NOTE` (`+0.40`) > `DICTIONARY_A` (`+0.30`) > `DICTIONARY_B` (`+0.25`) > `GRAMMAR` (`+0.20`) > `AUDIO_INTERVIEW` (`+0.15`) > `EPISODE` (`+0.10`) > `VIEWER_FEEDBACK` (`+0.05`).
- Produces auditable breakdown explanations (`explain_ranking()`).

### C. Structured Learned Rule Memory & Targeted Replanning (`LearnedRule`)
- Strongly typed domain model representing rules learned/confirmed from evidence or human review.
- `AgentState` contains a `learned_rules` channel managed by `merge_learned_rules` reducer.
- Supports rule superseding (`supersedes="R-OLD"`) while preserving audit history.
- `EpisodeProcessor.apply_rule_correction()` selectively invalidates and replans only subtitles in `affected_subtitle_ids` without mutating unaffected decisions.

### D. Explicit Abstention & Precise Escalation
- When evidence is missing, insufficient, or conflicting, NADI-9 explicitly abstains with:
  `"Translation unavailable: insufficient supporting evidence."`
- Generates precise review questions tailored to failure categories (insufficient evidence, conflicting evidence, proper name failure, number failure, or timing failure).

### E. Enhanced Independent Verification (`HypothesisVerifier`)
- **Proper Name Preservation Check**: Extracts capitalized proper nouns and verifies presence in translation.
- **Number / Quantity Preservation Check**: Extracts integers, decimals, percentages, and currencies, verifying numerical equivalence.
- **Subtitle Timing & Reading Speed Check**: Validates duration (`0.5s`–`12.0s`) and reading speed (`max 25 cps`).

### F. Durable LangGraph Checkpointing (`SqliteCheckpointSaver`)
- Business DB (`nadi9.db`) stores domain records; checkpoint DB (`nadi9_checkpoints.db`) stores graph channel state and thread snapshots.
- Thread ID format: `run_id:subtitle_id`. Process restarts safely reload state snapshots from SQLite and resume without re-running LLM calls.

---

## 4. Artifact Exporter Package (`sample_run/`)

The exporter (`src/nadi9/artifacts/exporter.py`) generates the standardized assignment package:
- `subtitles.srt`: Standard SRT subtitle file formatted with timestamps.
- `subtitle_decisions.jsonl`: One JSON decision object per line.
- `learned_rules.json`: Serialized `LearnedRule` collection.
- `review_queue.json`: Unresolved human review items.
- `final_report.md`: Markdown report detailing run metrics, decision tables, verification, and auditability.
