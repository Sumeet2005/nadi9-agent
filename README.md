# NADI-9 Agent

NADI-9 is an evidence-grounded, agentic subtitle decision system built for the **STAGE NADI-9 assignment**.

The system processes subtitle lines using evidence retrieval, deterministic source-precedence ranking, hypothesis generation, conflict detection, independent verification, human-in-the-loop (HITL) review, durable workflow checkpointing, learned-rule memory, targeted replanning, explicit abstention, auditability, and standardized submission artifacts.

The implementation uses **Python, LangGraph, Pydantic, SQLAlchemy, SQLite, FastAPI, Typer, and configurable LLM providers**.

---

## Key Capabilities

NADI-9 provides:

* Evidence-grounded subtitle decision making
* Deterministic evidence retrieval and source-precedence ranking
* LLM provider abstraction
* Offline deterministic mock-provider execution
* Structured episode and run state
* LangGraph-based workflow orchestration
* Conflict detection between evidence sources
* Independent deterministic verification
* Explicit abstention when evidence is insufficient
* Human-in-the-loop review workflows
* Durable SQLite checkpointing
* Learned-rule memory
* Rule correction with targeted selective replanning
* Budget enforcement for model usage
* Audit and traceability records
* FastAPI REST API
* Typer CLI
* Docker and Docker Compose support
* Standardized sample-run artifacts
* Automated test coverage for required failure and correction scenarios

---

# Quick Start

## 1. Requirements

* Python **3.11+**
* Git
* Docker / Docker Compose *(optional, only required for containerized execution)*

Create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install the package

```bash
pip install -e .
```

### Install development dependencies

```bash
pip install -e ".[dev]"
```

---

## 2. Environment and LLM Configuration

Copy the example environment file:

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

### macOS / Linux

```bash
cp .env.example .env
```

The main configuration variables are:

| Variable                   | Description                           | Default                |
| -------------------------- | ------------------------------------- | ---------------------- |
| `OPENAI_API_KEY`           | API key used by the OpenAI provider   | Empty                  |
| `OPENAI_MODEL`             | OpenAI model name                     | `gpt-4o-mini`          |
| `NADI9_LLM_PROVIDER`       | LLM provider selection                | Configurable           |
| `NADI9_DATABASE_URL`       | SQLite application database URL       | `sqlite:///nadi9.db`   |
| `NADI9_CHECKPOINT_DB_PATH` | Durable LangGraph checkpoint database | `nadi9_checkpoints.db` |
| `NADI9_MAX_BUDGET`         | Maximum execution budget in USD       | `10.0`                 |

### Offline / deterministic mode

NADI-9 supports deterministic execution without an OpenAI API key.

For local testing:

```powershell
$env:OPENAI_API_KEY=""
$env:NADI9_LLM_PROVIDER="mock"
```

This allows the complete test suite and sample workflow to execute without external LLM calls.

---

# Running the Sample Episode

NADI-9 includes a sample episode and evidence dataset for end-to-end verification.

Run:

```bash
python -m nadi9 run \
    --episodes data/episodes/sample_episode.jsonl \
    --evidence data/evidence/sample_evidence.jsonl \
    --output sample_run \
    --force
```

The command processes the sample episode and exports the standardized assignment artifacts.

A verified sample execution produces:

```text
Total processed: 3
Accepted: 1
Review required: 2
```

The sample demonstrates both successful processing and safety-oriented escalation to human review.

---

# Sample Run Artifacts

The generated `sample_run/` directory contains:

```text
sample_run/
├── subtitles.srt
├── subtitle_decisions.jsonl
├── learned_rules.json
├── review_queue.json
├── final_report.md
├── report.json
├── decisions.json
└── review_items.json
```

### Artifact descriptions

| Artifact                   | Purpose                                          |
| -------------------------- | ------------------------------------------------ |
| `subtitles.srt`            | Generated subtitle output in SRT format          |
| `subtitle_decisions.jsonl` | One structured decision record per subtitle      |
| `learned_rules.json`       | Learned-rule memory generated during corrections |
| `review_queue.json`        | Pending human review items                       |
| `final_report.md`          | Human-readable final audit/report                |
| `report.json`              | Structured audit report                          |
| `decisions.json`           | Backward-compatible decision export              |
| `review_items.json`        | Backward-compatible review-item export           |

---

# Human-in-the-Loop Review

NADI-9 does not force unsupported or conflicting subtitle decisions through the pipeline.

When evidence is conflicting, insufficient, or independent verification fails, the subtitle can be routed to human review.

## List pending reviews

```bash
python -m nadi9 review list --input sample_run
```

## Approve an AI candidate

```bash
python -m nadi9 review approve \
    --input sample_run \
    --review-id REV-EP001-L002-1
```

## Reject an AI candidate

```bash
python -m nadi9 review reject \
    --input sample_run \
    --review-id REV-EP001-L003-1 \
    --reason "Terminology mismatch"
```

## Provide a human correction

```bash
python -m nadi9 review correct \
    --input sample_run \
    --review-id REV-EP001-L002-1 \
    --text "Alvida, dosto!"
```

Human corrections create learned rules that can be used for targeted replanning of affected subtitle decisions.

---

# Workflow and Decision Safety

The processing workflow is designed around evidence grounding and explicit uncertainty.

A simplified flow is:

```text
Episode Input
     │
     ▼
Evidence Ingestion
     │
     ▼
Evidence Retrieval
     │
     ▼
Source-Precedence Ranking
     │
     ▼
Planning
     │
     ▼
Hypothesis Generation
     │
     ▼
Conflict Detection
     │
     ▼
Independent Verification
     │
     ├───────────────┐
     │               │
     ▼               ▼
Verified         Conflict /
Decision         Verification Failure
     │               │
     │               ▼
     │          Human Review
     │               │
     │               ▼
     │        Human Correction
     │               │
     │               ▼
     │        Learned Rule
     │               │
     │               ▼
     │       Targeted Replanning
     │
     ▼
Decision Assembly
     │
     ▼
Audit + Artifacts
```

This design ensures that unsupported evidence does not automatically become a confident final translation.

---

# Evidence Source Precedence

NADI-9 applies deterministic source-precedence ranking.

The implemented precedence policy prioritizes:

```text
Gold Standard
      >
Termbase
      >
Dictionary
      >
Crowdsourced
```

Higher-authority evidence can therefore take precedence over lower-authority evidence when determining the candidate interpretation.

Evidence citations are preserved for auditability.

---

# Independent Verification

Verification is performed independently from the model-generated hypothesis.

The verification layer includes deterministic checks for areas such as:

* Proper-name preservation
* Number and quantity preservation
* Subtitle reading speed
* Evidence availability
* Candidate consistency
* Verification failures requiring human review

This prevents the LLM from being the sole authority for final validation.

---

# Explicit Abstention

When the system does not have sufficient supporting evidence, NADI-9 can explicitly abstain instead of inventing a translation.

For example:

```text
Translation unavailable: insufficient supporting evidence.
```

Abstention is treated as a valid safety outcome and can result in human review.

---

# Conflict Detection

NADI-9 detects contradictory evidence before final decision assembly.

For example, the sample workflow includes a conflict between evidence records:

```text
CONFLICT-E-002-E-003
```

The affected subtitle is routed to human review rather than silently selecting an unsupported interpretation.

---

# Learned Rules and Targeted Replanning

Human corrections are represented as first-class learned rules.

A learned rule can contain information such as:

* Rule category
* Rule statement
* Version
* Superseded rule
* Affected subtitle IDs
* Correction information

When a correction is supplied, NADI-9 performs **targeted selective replanning** rather than unnecessarily reprocessing the entire episode.

This preserves historical audit records while allowing the corrected knowledge to influence affected decisions.

---

# Budget Enforcement

NADI-9 includes execution budget controls.

The budget layer can enforce limits on:

* Model calls
* Tool calls
* Total execution budget

If the configured execution budget is exhausted, processing is safely routed toward review instead of continuing without control.

---

# Durable Checkpointing

The LangGraph workflow uses durable SQLite checkpointing.

This provides persistence across workflow interruptions and supports recovery of stateful execution.

The implementation includes tests covering:

* Durable checkpoints
* Concurrent checkpoint access
* Process restart recovery
* Run persistence
* HITL workflow state

---

# FastAPI REST API

NADI-9 provides a FastAPI-based REST service.

Start the API with:

```bash
uvicorn nadi9.api.app:app --host 0.0.0.0 --port 8000
```

## Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

## Core API Operations

The API provides operations for:

* Health checks
* Starting processing runs
* Listing runs
* Retrieving run details
* Retrieving decisions
* Retrieving review items
* Approving review items
* Rejecting review items
* Providing human corrections

The implementation exposes these through the FastAPI route modules under:

```text
src/nadi9/api/routes/
```

---

# Docker

NADI-9 includes a production-oriented Docker configuration.

## Build the image

```bash
docker build -t nadi9-agent .
```

## Run the container

```bash
docker run --rm -p 8000:8000 nadi9-agent
```

## Verify the health endpoint

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

---

# Docker Compose

The project also includes Docker Compose configuration.

Start the service:

```bash
docker compose up -d
```

View running containers:

```bash
docker compose ps
```

View logs:

```bash
docker compose logs
```

Stop the service:

```bash
docker compose down
```

Persistent database/checkpoint locations are configured through the Compose volumes and application configuration.

---

# Testing

NADI-9 has a comprehensive automated test suite covering the core assignment requirements.

Run all tests:

```bash
pytest -q
```

### Verified local test result

```text
194 passed
0 failed
```

The latest verified local execution completed successfully with only a third-party LangGraph deprecation warning.

The warning originates from an external dependency and does not represent a project test failure.

---

# Additional Verification

Before submission, the project was also checked using the following validation steps:

### Python compilation

```bash
python -m compileall -q src tests
```

### Test collection

```bash
pytest --collect-only -q
```

### Package installation

```bash
pip install -e .
```

### Formatting / whitespace validation

```bash
git diff --check
```

### Docker image build

```bash
docker build -t nadi9-agent .
```

### Container health verification

The Docker container was started locally and the `/health` endpoint was successfully verified before cleanup.

---

# Test Coverage Areas

The test suite covers areas including:

```text
tests/
├── test_abstention.py
├── test_api.py
├── test_artifacts.py
├── test_audit.py
├── test_budget.py
├── test_budgeted_provider.py
├── test_checkpoint_concurrency.py
├── test_cli.py
├── test_config.py
├── test_conflict.py
├── test_decision.py
├── test_durable_checkpoint.py
├── test_e2e_process_restart_recovery.py
├── test_end_to_end.py
├── test_hitl_workflow.py
├── test_hypothesis.py
├── test_ingestion.py
├── test_learned_rules.py
├── test_models.py
├── test_observability.py
├── test_openai_provider.py
├── test_planner.py
├── test_providers.py
├── test_ranking.py
├── test_retriever.py
├── test_review.py
├── test_rule_correction.py
├── test_run_persistence.py
├── test_state.py
├── test_storage.py
├── test_verification.py
├── test_verification_enhancements.py
└── test_workflow.py
```

The tests specifically exercise:

* Evidence retrieval
* Evidence ranking
* Conflict detection
* Hypothesis generation
* Verification
* Abstention
* Budget enforcement
* Provider abstraction
* HITL review
* Learned-rule correction
* Targeted replanning
* Durable checkpointing
* Process restart recovery
* Storage
* API behavior
* CLI behavior
* Artifact generation
* End-to-end processing

---

# Project Structure

```text
E:\nadi9-agent\
│
├── data/
│   ├── episodes/
│   │   └── sample_episode.jsonl
│   └── evidence/
│       └── sample_evidence.jsonl
│
├── sample_run/
│   ├── subtitles.srt
│   ├── subtitle_decisions.jsonl
│   ├── learned_rules.json
│   ├── review_queue.json
│   ├── final_report.md
│   ├── report.json
│   ├── decisions.json
│   └── review_items.json
│
├── src/
│   └── nadi9/
│       ├── api/
│       │   ├── routes/
│       │   ├── app.py
│       │   ├── dependencies.py
│       │   └── schemas.py
│       │
│       ├── artifacts/
│       │   └── exporter.py
│       │
│       ├── audit/
│       │   ├── report.py
│       │   └── review_service.py
│       │
│       ├── budget/
│       │   └── manager.py
│       │
│       ├── domain/
│       │   ├── enums.py
│       │   ├── errors.py
│       │   └── models.py
│       │
│       ├── graph/
│       │   ├── conflict.py
│       │   ├── decision.py
│       │   ├── hypothesis.py
│       │   ├── planner.py
│       │   ├── state.py
│       │   ├── verification.py
│       │   └── workflow.py
│       │
│       ├── ingestion/
│       │   ├── jsonl.py
│       │   ├── ranking.py
│       │   └── retriever.py
│       │
│       ├── observability/
│       │   ├── logging.py
│       │   └── metrics.py
│       │
│       ├── providers/
│       │   ├── base.py
│       │   ├── budgeted.py
│       │   ├── mock.py
│       │   └── openai.py
│       │
│       ├── storage/
│       │   ├── checkpointer.py
│       │   ├── database.py
│       │   ├── migrations.py
│       │   ├── models.py
│       │   └── repositories.py
│       │
│       ├── cli.py
│       ├── config.py
│       └── processor.py
│
├── tests/
│   └── 194 automated tests
│
├── AI_COLLABORATION.md
├── ARCHITECTURE.md
├── KNOWN_LIMITATIONS.md
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── .gitignore
```

---

# Architecture Documentation

Additional technical documentation is available in:

### `ARCHITECTURE.md`

Describes the system architecture, workflow components, data flow, persistence, provider abstraction, verification, and HITL design.

### `AI_COLLABORATION.md`

Documents how AI-assisted development was used during implementation.

### `KNOWN_LIMITATIONS.md`

Documents known limitations and boundaries of the current implementation.

---

# Security and Repository Hygiene

The repository is configured to avoid committing local runtime and secret files.

Ignored files include development/runtime artifacts such as:

```text
.env
.venv/
__pycache__/
.pytest_cache/
*.db
*.sqlite
```

The repository uses `.env.example` for configuration documentation rather than committing actual credentials.

**Never commit a real `OPENAI_API_KEY` or other secret credentials.**

---

# Submission Verification Checklist

Before submitting the assignment, verify:

```text
[ ] README.md is present and up to date
[ ] ARCHITECTURE.md is present
[ ] AI_COLLABORATION.md is present
[ ] KNOWN_LIMITATIONS.md is present
[ ] .env.example is present
[ ] No .env file is committed
[ ] No API keys or secrets are committed
[ ] pytest -q passes
[ ] Python compilation succeeds
[ ] Sample CLI run succeeds
[ ] sample_run artifacts are present
[ ] Docker image builds successfully
[ ] /health endpoint responds successfully
[ ] git diff --check passes
[ ] git status is clean
[ ] Changes are pushed to GitHub
```

---

# Current Verification Status

The current repository has been locally verified with:

```text
Python compilation       PASS
Package installation     PASS
Automated test suite     PASS
194 tests                PASS
Test failures            0
Sample CLI execution     PASS
Docker build             PASS
Docker health check      PASS
Git whitespace check     PASS
Repository hygiene      PASS
```

The sample workflow demonstrates:

```text
3 subtitles processed
1 accepted
2 requiring human review
1 explicit abstention
```

The final release recommendation for the sample run is:

```text
HOLD - HUMAN REVIEW REQUIRED
```

This is intentional: the system is designed to avoid releasing subtitle decisions when evidence conflicts or verification is insufficient.

---

# License

This repository was created as part of the **STAGE NADI-9 assignment**.
