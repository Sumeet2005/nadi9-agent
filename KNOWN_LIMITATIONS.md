# NADI-9: Known Limitations & Technical Trade-Offs

This document outlines the architectural trade-offs, known limitations, and future operational enhancements for the **NADI-9** agentic subtitle decision system.

---

## 1. Evidence Retrieval Engine

* **Current Implementation**: Keyword matching and exact term matching against structured evidence sources (term base, context files).
* **Limitation**: Semantically ambiguous terms or non-exact phrasing in sub-dialect subtitle sources may fail to match unless explicitly listed in evidence records.
* **Production Recommendation**: Replace keyword searching with a hybrid dense/sparse vector retrieval engine (e.g., Qdrant, PGVector) powered by domain-specific subtitle & term embeddings.

---

## 2. LLM Provider Fallback & Mock Execution

* **Current Implementation**: Built-in `MockLLMProvider` fallback that deterministically resolves subtitle translation choices using evidence rule heuristics when `OPENAI_API_KEY` is not present or API limits are reached.
* **Limitation**: While offline mock mode guarantees 100% test reproducibility and zero setup friction for external evaluation, complex unscripted human escalation questions require a live OpenAI key for open-ended response generation.
* **Production Recommendation**: Deploy with multi-provider failover (e.g., Azure OpenAI + Anthropic Claude fallback) and explicit schema enforcement.

---

## 3. Subtitle Timing & Reading Speed Constraints

* **Current Implementation**: Strict max reading-speed limit enforced at **25 characters per second (CPS)**. Subtitles exceeding this constraint trigger an automatic split suggestion or human review escalation.
* **Limitation**: Timing adjustments are computed per-subtitle line without full shot-change / scene-boundary alignment metadata.
* **Production Recommendation**: Integrate scene-boundary detection (EDL / video shot boundary keyframes) to align subtitle splits precisely with video cut points.

---

## 4. State Persistence & Checkpointing

* **Current Implementation**: Embedded SQLite durable checkpointer for state management and workflow resumes.
* **Limitation**: Local SQLite databases are scoped per instance/run and require a shared network filesystem or cloud DB for multi-worker distributed clusters.
* **Production Recommendation**: Use a centralized PostgreSQL instance with `pg_vector` for multi-tenant enterprise deployments.

---

## 5. Learned Rule Memory Scope

* **Current Implementation**: Rules learned from human review feedback are saved in a structured JSON/SQLite memory store indexed by context key.
* **Limitation**: Memory updates are episode and context-bound; global cross-project generalization requires periodic offline rule aggregation.
* **Production Recommendation**: Implement an offline asynchronous rule consolidation job that merges local learned rules into global enterprise term bases.
