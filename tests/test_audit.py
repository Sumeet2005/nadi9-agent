from nadi9.domain.enums import DecisionStatus, VerificationStatus
from nadi9.domain.models import EpisodeLine, EvidenceRecord, Provenance
from nadi9.graph.state import create_initial_state
from nadi9.graph.workflow import build_nadi9_workflow
from nadi9.providers import MockLLMProvider
from nadi9.audit import build_episode_audit_report, AuditReport, DecisionAuditDetail


def make_evidence(
    evidence_id: str,
    content: str,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_id="SRC-001",
        evidence_type="dictionary_a",
        content=content,
        provenance=Provenance(
            original_file="test.json",
            location="entry-1",
            extraction_method="fixture",
        ),
    )


def make_episode() -> EpisodeLine:
    return EpisodeLine(
        subtitle_id="SUB-001",
        source_text="formal greeting",
        speaker="Speaker A",
        scene_id="SCENE-001",
        start_time=0.0,
        end_time=2.0,
    )


def test_build_episode_audit_report_serializes_cleanly():
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

    workflow = build_nadi9_workflow(provider)
    state = create_initial_state("run-101", "ep-50", episode_lines=[make_episode()])
    state["evidence"] = [make_evidence("E-001", "formal greeting used by elders")]

    final_state = workflow.run(state)
    report = build_episode_audit_report(final_state)

    assert isinstance(report, AuditReport)
    assert report.run_id == "run-101"
    assert report.episode_id == "ep-50"
    assert report.total_subtitles_processed == 1
    assert report.accepted_count == 1
    assert report.human_review_count == 0

    assert len(report.decisions) == 1
    decision_audit = report.decisions[0]
    assert isinstance(decision_audit, DecisionAuditDetail)
    assert decision_audit.subtitle_id == "SUB-001"
    assert decision_audit.status == DecisionStatus.ACCEPTED
    assert len(decision_audit.evidence) == 1
    assert decision_audit.evidence[0].evidence_id == "E-001"

    raw_json = report.to_dict()
    assert raw_json["run_id"] == "run-101"
    assert raw_json["total_subtitles_processed"] == 1
    assert raw_json["decisions"][0]["status"] == "accepted"


def test_audit_report_includes_human_review_and_conflicts():
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

    workflow = build_nadi9_workflow(provider)
    state = create_initial_state("run-102", "ep-50", episode_lines=[make_episode()])
    ev1 = make_evidence("E-001", "formal greeting")
    ev1.metadata["conflicts_with"] = ["E-002"]
    ev2 = make_evidence("E-002", "informal greeting")
    state["evidence"] = [ev1, ev2]

    final_state = workflow.run(state)
    report = build_episode_audit_report(final_state)

    assert report.human_review_count == 1
    assert len(report.review_items) == 1
    assert len(report.decisions[0].conflicts) == 1
    assert report.decisions[0].status == DecisionStatus.REVIEW_REQUIRED
