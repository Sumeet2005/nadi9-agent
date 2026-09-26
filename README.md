# NADI-9 Agent

Nadi-9 is an evidence-grounded agentic subtitle decision system built for the STAGE Nadi-9 assignment. It orchestrates evidence retrieval, source-precedence ranking, hypothesis generation, conflict detection, enhanced independent verification, stateful LangGraph human-in-the-loop (HITL) workflow routing with durable SQLite checkpointing, learned-rule memory, targeted replanning, explicit abstention, and standardized artifact exporting using LangGraph, Pydantic, SQLAlchemy, and FastAPI.

---

## Quick Start

### 1. Local Installation

Ensure Python 3.11+ is installed. Install package dependencies in editable mode:

```bash
pip install -e .
```

To install dev dependencies (pytest, ruff):

```bash
pip install -e .[dev]
```

### 2. Environment & LLM Provider Configuration

Copy `.env.example` to `.env` and configure environment variables as needed:

```bash
cp .env.example .env
```

Key environment variables:
- `OPENAI_API_KEY`: API key for OpenAI provider. **Note**: If `OPENAI_API_KEY` is not set or empty, NADI-9 automatically falls back to the deterministic `MockLLMProvider` for offline processing and testing without raising runtime errors.
- `OPENAI_MODEL`: OpenAI model name (default: `gpt-4o-mini`).
- `NADI9_DATABASE_URL`: SQLite database connection URL (default: `sqlite:///nadi9.db`).
- `NADI9_CHECKPOINT_DB_PATH`: Durable LangGraph checkpoint SQLite database path (default: `nadi9_checkpoints.db`).
- `NADI9_MAX_BUDGET`: Maximum allowable execution budget in USD (default: `10.0`).

---

## Running Sample Episode Processing & Generating Artifacts

Execute end-to-end subtitle decision processing over the sample dataset using the verified CLI command:

```bash
python -m nadi9 run \
    --episodes data/episodes/sample_episode.jsonl \
    --evidence data/evidence/sample_evidence.jsonl \
    --output sample_run \
    --force
```

This command generates both the standard assignment package files and backward-compatible JSON reports in `sample_run/`:

```text
sample_run/
├── subtitles.srt              # Formatted SRT subtitle file
├── subtitle_decisions.jsonl   # One JSON decision object per line
├── learned_rules.json         # Serialized LearnedRule domain objects
├── review_queue.json          # Unresolved human review items
├── final_report.md            # Comprehensive Markdown report
├── report.json                # Structured audit report
├── decisions.json             # Backward-compatible JSON decisions list
└── review_items.json          # Backward-compatible JSON review items list
```

---

## Human-in-the-Loop (HITL) Review Management via CLI

When evidence is missing/conflicting or verification fails, Nadi-9 escalates to human review:

```bash
# List pending reviews
python -m nadi9 review list --input sample_run

# Approve AI candidate subtitle
python -m nadi9 review approve --input sample_run --review-id REV-EP001-L002-1

# Reject AI candidate subtitle with reason
python -m nadi9 review reject --input sample_run --review-id REV-EP001-L003-1 --reason "Terminology mismatch"

# Supply human correction (creates a LearnedRule and updates learned_rules.json)
python -m nadi9 review correct --input sample_run --review-id REV-EP001-L002-1 --text "Alvida, dosto!"
```

---

## Docker & Docker Compose Deployment

NADI-9 includes production Docker deployment files supporting persistent volume mounts:

### 1. Build Docker Image
```bash
docker build -t nadi9-agent .
```

### 2. Single-Command Startup via Docker Compose
```bash
docker compose up -d
```

### 3. Verify Health Endpoint
```bash
curl http://localhost:8000/health
# Response: {"status":"ok","version":"0.1.0"}
```

---

## FastAPI REST Web Service

Start the production REST API service:

```bash
uvicorn nadi9.api.app:app --host 0.0.0.0 --port 8000
```

### Key API Endpoints
- `GET /health` - Health check.
- `POST /runs` - Execute subtitle processing run.
- `GET /runs` - List execution runs.
- `GET /runs/{run_id}` - Retrieve run details.
- `GET /runs/{run_id}/decisions` - Get run decisions.
- `GET /runs/{run_id}/reviews` - Get run review items.
- `POST /reviews/{review_id}/approve` - Approve review item.
- `POST /reviews/{review_id}/reject` - Reject review item.
- `POST /reviews/{review_id}/correct` - Supply human correction.

---

## Running Tests & Test Baseline Status

Run the full pytest test suite:

```bash
pytest
```

**Verified Test Baseline**:
- **197 passed, 0 failed**
- **2 third-party deprecation warnings** (Starlette `testclient` and LangGraph `allowed_objects`; these originate from external packages and are not test failures).

---

## Project Structure

```text
.
├── data/                       # Sample episodes and evidence JSONL datasets
│   ├── episodes/
│   └── evidence/
├── src/nadi9/                  # Core package codebase
│   ├── api/                    # FastAPI web routes and request schemas
│   ├── artifacts/              # Standardized sample_run package exporter
│   ├── audit/                  # Audit report generators and review service
│   ├── domain/                 # Strongly typed Pydantic models and StrEnums
│   ├── graph/                  # LangGraph workflow, planner, conflict, verifier, decision
│   ├── ingestion/              # JSONL readers, retriever, and ranking policy
│   ├── observability/          # Structured JSON logging and latency metrics
│   ├── providers/              # LLM provider abstraction (OpenAI, Mock, Budgeted)
│   ├── storage/                # SQLite business persistence and durable checkpointer
│   ├── cli.py                  # Typer CLI application
│   ├── config.py               # Pydantic Settings configuration management
│   └── processor.py            # End-to-end episode pipeline processor
├── tests/                      # Pytest suite
├── sample_run/                 # Generated sample run output package
├── ARCHITECTURE.md             # System architecture documentation
├── AI_COLLABORATION.md         # AI collaboration transparency report
├── KNOWN_LIMITATIONS.md        # Technical trade-offs & limitations
├── Dockerfile                  # Production container Dockerfile
├── docker-compose.yml          # Docker compose orchestration
└── pyproject.toml              # Build dependencies and project metadata
```
