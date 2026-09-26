# AI Collaboration & Engineering Transparency

This document provides a transparent, factual record of how AI-assisted development was utilized during the construction of the NADI-9 Agentic AI Subtitle Decision System.

---

## 1. Purpose & Development Workflow

NADI-9 was developed using a pair-programming workflow between a human lead engineer and AI agentic coding assistants. The goal was to build a production-grade, evidence-grounded agentic subtitle processing system adhering to strict architectural invariants, high reliability, and zero-regression testing standards across sequential development phases.

The overall engineering approach combined:
1. Human domain design, architectural requirements specification, and strict rule enforcement.
2. AI-assisted initial boilerplate generation, state machine graph creation, and test case expansion.
3. Rigorous automated verification (`pytest` suite) after every single phase to validate implementation integrity.

---

## 2. Key Areas of AI Assistance

AI assistance was leveraged across all 8 feature phases of the project:

1. **Structured Episode Planning (Phase 2)**
   - AI helped generate deterministic risk scoring keywords (`RELATIONSHIP_KEYWORDS`, `TONE_KEYWORDS`, `CULTURAL_RARE_KEYWORDS`) and construct `EpisodePlanner.plan_episode()` & `plan_subset()`.

2. **Deterministic Source Precedence (Phase 3)**
   - AI assisted in building `EvidenceRankingPolicy` weight tables and auditable explanation breakdown methods (`score_breakdown()`, `explain_ranking()`) mapping evidence authority (`APPROVED_EXAMPLE` > `EXPERT_NOTE` > `DICTIONARY_A` > `DICTIONARY_B` > `GRAMMAR` > `AUDIO_INTERVIEW` > `EPISODE` > `VIEWER_FEEDBACK`).

3. **Structured Learned Rule Memory (Phase 4)**
   - AI assisted in drafting the strongly typed Pydantic `LearnedRule` domain model and the `merge_learned_rules` reducer for LangGraph `AgentState`.

4. **Targeted Rule Correction & Selective Replanning (Phase 5)**
   - AI assisted in structuring `EpisodeProcessor.apply_rule_correction()`, ensuring affected subtitles were selectively invalidated while preserving unaffected subtitle decisions and audit history.

5. **Explicit Abstention & Precise Human Escalation (Phase 6)**
   - AI assisted in integrating canonical abstention text (`"Translation unavailable: insufficient supporting evidence."`) and generating actionable human-review questions based on failure types.

6. **Enhanced Independent Verification (Phase 7)**
   - AI assisted in writing regex-based deterministic functions for proper name extraction, numeric normalization, and subtitle timing/reading-speed calculations (`MAX_CHARS_PER_SECOND = 25.0`).

7. **Standardized Artifact Exporter Package (Phase 8)**
   - AI assisted in building `src/nadi9/artifacts/exporter.py` to produce `subtitles.srt`, `subtitle_decisions.jsonl`, `learned_rules.json`, `review_queue.json`, and `final_report.md`.

8. **Testing & Debugging Support**
   - AI generated unit tests, state concurrency checks, and durable checkpoint recovery tests, bringing total test coverage to 194 passing tests.

---

## 3. Human Engineering Responsibilities & Decision Making

While AI assisted in writing code and tests, all critical architectural decisions, security constraints, and design validations were governed by human engineering principles:

- **Architectural Preservation**: Rejection of broad refactors; insistence on preserving existing public API contracts and LangGraph state channels.
- **Python 3.11 Runtime Compatibility**: When AI initially introduced Python 3.12+ generic syntax (`def func[T](...)`), human validation identified Docker build runtime crashes on `python:3.11-slim` and mandated standard `TypeVar` syntax.
- **Hypothesis Preservation Logic**: When explicit abstention was added in Phase 6, AI logic initially overwrote valid candidate hypotheses on verification failure. Human review corrected `workflow.py` so valid AI hypotheses were preserved for human review while empty-evidence cases explicitly abstained.
- **Zero-Regression Policy**: AI tool executions were strictly verified against the complete `pytest` test suite before accepting any code edits.

---

## 4. Architectural Retention & Stability

Key architectural decisions intentionally retained throughout development:
- **Separation of Concerns**: SQLite business persistence (`nadi9.db`) remains separate from durable execution checkpointing (`nadi9_checkpoints.db`).
- **Provider Abstraction**: Provider interface (`LLMProvider`) allows seamless swapping between `MockLLMProvider`, `OpenAILLMProvider`, and `BudgetedLLMProvider`.
- **Deterministic Independence**: Verification (`HypothesisVerifier`) and ranking (`EvidenceRanker`) run strictly deterministically without asking the LLM to grade its own work.

---

## 5. Summary Statement

AI assistance accelerated implementation velocity, test generation, and documentation drafting. However, every line of generated code was systematically reviewed, tested, and validated against empirical test results and production requirements. AI was employed as an intelligent tool under strict human engineering guidance.
