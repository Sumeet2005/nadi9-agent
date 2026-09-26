# NADI-9 Agent

NADI-9 is an evidence-grounded, agentic subtitle decision system built for the **STAGE NADI-9 assignment**.

The system processes subtitle lines using evidence retrieval, deterministic source-precedence ranking, hypothesis generation, conflict detection, independent verification, human-in-the-loop (HITL) review, durable workflow checkpointing, learned-rule memory, targeted replanning, explicit abstention, auditability, and standardized submission artifacts.

The implementation uses **Python, LangGraph, Pydantic, SQLAlchemy, SQLite, FastAPI, Typer, and configurable LLM providers**.

---

## Key Capabilities

NADI-9 provides:

- Evidence-grounded subtitle decision making
- Deterministic evidence retrieval and source-precedence ranking
- LLM provider abstraction
- Offline deterministic mock-provider execution
- Structured episode/run state
- LangGraph-based workflow orchestration
- Conflict detection between evidence sources
- Independent deterministic verification
- Explicit abstention when evidence is insufficient
- Human-in-the-loop review workflows
- Durable SQLite checkpointing
- Learned-rule memory
- Rule correction with targeted selective replanning
- Budget enforcement for model usage
- Audit and traceability records
- FastAPI REST API
- Typer CLI
- Docker and Docker Compose support
- Standardized sample-run artifacts
- Automated test coverage for required failure and correction scenarios

---

# Quick Start

## 1. Requirements

- Python **3.11+**
- Git
- Docker / Docker Compose (optional, only required for containerized execution)

Create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
