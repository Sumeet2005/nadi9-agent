import json
import pytest

from nadi9.budget import BudgetManager
from nadi9.domain.enums import DecisionStatus, VerificationStatus
from nadi9.domain.models import EpisodeLine, EvidenceRecord, Provenance
from nadi9.processor import EpisodeProcessor
from nadi9.providers import BudgetedLLMProvider, MockLLMProvider


def make_evidence_dict(
    evidence_id: str,
    content: str,
    conflicts_with: list[str] | None = None,
) -> dict:
    metadata = {}
    if conflicts_with:
        metadata["conflicts_with"] = conflicts_with

    return {
        "evidence_id": evidence_id,
        "source_id": "SRC-001",
        "evidence_type": "dictionary_a",
        "content": content,
        "metadata": metadata,
        "provenance": {
            "original_file": "dictionary.json",
            "location": "entry-1",
            "extraction_method": "manual",
        },
    }


def make_episode_dict(
    subtitle_id: str = "SUB-001",
    source_text: str = "formal greeting",
) -> dict:
    return {
        "subtitle_id": subtitle_id,
        "source_text": source_text,
        "speaker": "Speaker 1",
        "scene_id": "SCENE-01",
        "start_time": 0.0,
        "end_time": 2.0,
    }


def test_end_to_end_happy_path(tmp_path):
    episode_file = tmp_path / "episodes.jsonl"
    evidence_file = tmp_path / "evidence.jsonl"

    episode_file.write_text(
        json.dumps(make_episode_dict("SUB-001", "formal greeting")) + "\n",
        encoding="utf-8",
    )
    evidence_file.write_text(
        json.dumps(make_evidence_dict("E-001", "formal greeting used by elders")) + "\n",
        encoding="utf-8",
    )

    provider = MockLLMProvider(
        responses=[
            {
                "hypothesis_id": "H-001",
                "category": "greeting",
                "statement": "Namaskar.",
                "supporting_evidence": ["E-001"],
                "counterexamples": [],
                "status": "proposed",
                "confidence": "high",
            }
        ]
    )

    processor = EpisodeProcessor(provider=provider)
    report = processor.process_files(
        episode_path=episode_file,
        evidence_path=evidence_file,
        run_id="run-e2e-1",
        episode_id="ep-e2e-1",
    )

    assert report.run_id == "run-e2e-1"
    assert report.total_subtitles_processed == 1
    assert report.accepted_count == 1
    assert report.human_review_count == 0
    assert len(report.decisions) == 1
    assert report.decisions[0].status == DecisionStatus.ACCEPTED
    assert report.decisions[0].verification.status == VerificationStatus.PASSED


def test_end_to_end_verification_failure(tmp_path):
    episode_file = tmp_path / "episodes.jsonl"
    evidence_file = tmp_path / "evidence.jsonl"

    episode_file.write_text(
        json.dumps(make_episode_dict("SUB-001", "formal greeting")) + "\n",
        encoding="utf-8",
    )
    evidence_file.write_text(
        json.dumps(make_evidence_dict("E-001", "formal greeting used by elders")) + "\n",
        encoding="utf-8",
    )

    # Provider cites non-existent evidence ID E-999
    provider = MockLLMProvider(
        responses=[
            {
                "hypothesis_id": "H-001",
                "category": "greeting",
                "statement": "Namaskar.",
                "supporting_evidence": ["E-999"],
                "counterexamples": [],
                "status": "proposed",
                "confidence": "medium",
            }
        ]
    )

    processor = EpisodeProcessor(provider=provider)
    report = processor.process_files(
        episode_path=episode_file,
        evidence_path=evidence_file,
    )

    assert report.total_subtitles_processed == 1
    assert report.accepted_count == 0
    assert report.human_review_count == 1
    assert len(report.errors) > 0


def test_end_to_end_evidence_conflict(tmp_path):
    episode_file = tmp_path / "episodes.jsonl"
    evidence_file = tmp_path / "evidence.jsonl"

    episode_file.write_text(
        json.dumps(make_episode_dict("SUB-001", "formal greeting")) + "\n",
        encoding="utf-8",
    )

    ev1 = make_evidence_dict("E-001", "formal greeting", conflicts_with=["E-002"])
    ev2 = make_evidence_dict("E-002", "informal greeting")

    evidence_file.write_text(
        json.dumps(ev1) + "\n" + json.dumps(ev2) + "\n",
        encoding="utf-8",
    )

    provider = MockLLMProvider(
        responses=[
            {
                "hypothesis_id": "H-001",
                "category": "greeting",
                "statement": "Namaskar.",
                "supporting_evidence": ["E-001", "E-002"],
                "counterexamples": [],
                "status": "proposed",
                "confidence": "medium",
            }
        ]
    )

    processor = EpisodeProcessor(provider=provider)
    report = processor.process_files(
        episode_path=episode_file,
        evidence_path=evidence_file,
    )

    assert report.human_review_count == 1
    assert len(report.review_items) == 1
    assert report.decisions[0].status == DecisionStatus.REVIEW_REQUIRED
    assert len(report.decisions[0].conflicts) == 1


def test_end_to_end_budget_exhaustion(tmp_path):
    episode_file = tmp_path / "episodes.jsonl"
    evidence_file = tmp_path / "evidence.jsonl"

    episode_file.write_text(
        json.dumps(make_episode_dict("SUB-001", "formal greeting")) + "\n",
        encoding="utf-8",
    )
    evidence_file.write_text(
        json.dumps(make_evidence_dict("E-001", "formal greeting used by elders")) + "\n",
        encoding="utf-8",
    )

    mock = MockLLMProvider(
        responses=[
            {
                "hypothesis_id": "H-001",
                "category": "greeting",
                "statement": "Namaskar.",
                "supporting_evidence": ["E-001"],
                "counterexamples": [],
                "status": "proposed",
                "confidence": "high",
            }
        ]
    )

    budget_mgr = BudgetManager(max_model_calls=1)
    budget_mgr.consume_model_call()  # Exhaust budget before pipeline run

    budgeted_provider = BudgetedLLMProvider(mock, budget_mgr)
    processor = EpisodeProcessor(provider=budgeted_provider)

    report = processor.process_files(
        episode_path=episode_file,
        evidence_path=evidence_file,
    )

    assert report.human_review_count == 1
    assert len(report.errors) > 0
    assert report.budget_summary["model_calls_used"] == 1


def test_end_to_end_multiple_episode_lines(tmp_path):
    episode_file = tmp_path / "episodes.jsonl"
    evidence_file = tmp_path / "evidence.jsonl"

    lines = [
        make_episode_dict("SUB-001", "formal greeting"),
        make_episode_dict("SUB-002", "welcome home"),
    ]
    episode_file.write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n",
        encoding="utf-8",
    )

    ev_records = [
        make_evidence_dict("E-001", "formal greeting used by elders"),
        make_evidence_dict("E-002", "welcome home expression"),
    ]
    evidence_file.write_text(
        "\n".join(json.dumps(ev) for ev in ev_records) + "\n",
        encoding="utf-8",
    )

    provider = MockLLMProvider(
        responses=[
            {
                "hypothesis_id": "H-001",
                "category": "greeting",
                "statement": "Namaskar.",
                "supporting_evidence": ["E-001"],
                "counterexamples": [],
                "status": "proposed",
                "confidence": "high",
            },
            {
                "hypothesis_id": "H-002",
                "category": "welcome",
                "statement": "Swagatam.",
                "supporting_evidence": ["E-002"],
                "counterexamples": [],
                "status": "proposed",
                "confidence": "high",
            },
        ]
    )

    processor = EpisodeProcessor(provider=provider)
    report = processor.process_files(
        episode_path=episode_file,
        evidence_path=evidence_file,
    )

    assert report.total_subtitles_processed == 2
    assert report.accepted_count == 2
    assert len(report.decisions) == 2
    assert report.decisions[0].subtitle_id == "SUB-001"
    assert report.decisions[0].nadi9_text == "Namaskar."
    assert report.decisions[1].subtitle_id == "SUB-002"
    assert report.decisions[1].nadi9_text == "Swagatam."
