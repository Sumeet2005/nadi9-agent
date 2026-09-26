import pytest
from fastapi.testclient import TestClient

from nadi9.api.app import app
from nadi9.storage import create_db_engine, create_session_factory, init_db

from nadi9.api.dependencies import get_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    db_file = tmp_path / "test_api_nadi9.db"
    engine = create_db_engine(f"sqlite:///{db_file}")
    init_db(engine)
    session_factory = create_session_factory(engine)

    def _override_get_db():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.clear()


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"


def test_create_run_and_get_endpoints(tmp_path):
    episode_data = [
        {
            "subtitle_id": "SUB-API-01",
            "source_text": "formal greeting",
            "speaker": "Speaker 1",
            "scene_id": "SCENE-01",
            "start_time": 0.0,
            "end_time": 2.0,
        }
    ]
    evidence_data = [
        {
            "evidence_id": "E-001",
            "source_id": "SRC-1",
            "evidence_type": "dictionary_a",
            "content": "formal greeting used by elders",
            "provenance": {"original_file": "dict.json", "extraction_method": "manual"},
        }
    ]

    payload = {
        "run_id": "run-api-101",
        "episode_id": "ep-api-101",
        "episode_lines": episode_data,
        "evidence_records": evidence_data,
    }

    resp = client.post("/runs", json=payload)
    assert resp.status_code == 201
    run_report = resp.json()
    assert run_report["run_id"] == "run-api-101"
    assert run_report["total_subtitles_processed"] == 1
    assert len(run_report["decisions"]) == 1

    # List runs
    list_resp = client.get("/runs")
    assert list_resp.status_code == 200
    runs = list_resp.json()
    assert any(r["run_id"] == "run-api-101" for r in runs)

    # Get single run
    get_resp = client.get("/runs/run-api-101")
    assert get_resp.status_code == 200
    assert get_resp.json()["run_id"] == "run-api-101"

    # Get decisions
    dec_resp = client.get("/runs/run-api-101/decisions")
    assert dec_resp.status_code == 200
    decisions = dec_resp.json()
    assert len(decisions) == 1
    assert decisions[0]["subtitle_id"] == "SUB-API-01"


def test_get_nonexistent_run_404():
    resp = client.get("/runs/non_existent_run_999")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]
