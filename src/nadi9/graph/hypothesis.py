from typing import Any

from nadi9.domain.models import EpisodeLine, Hypothesis
from nadi9.ingestion import EvidenceRanker, EvidenceRetriever
from nadi9.providers import LLMProvider
from nadi9.graph.state import AgentState


class HypothesisGenerationError(ValueError):
    """Raised when a hypothesis cannot be generated safely."""


def build_hypothesis_prompt(
    episode: EpisodeLine,
    evidence_text: list[dict[str, Any]],
) -> str:
    """Build a deterministic prompt from the episode and retrieved evidence."""

    evidence_section = "\n".join(
        (
            f"- Evidence ID: {item['evidence_id']}\n"
            f"  Type: {item['evidence_type']}\n"
            f"  Content: {item['content']}\n"
        )
        for item in evidence_text
    )

    return f"""
You are generating a linguistic hypothesis for a subtitle decision.

Episode line:
Subtitle ID: {episode.subtitle_id}
Source text: {episode.source_text}
Speaker: {episode.speaker}
Scene: {episode.scene_id}

Retrieved evidence:
{evidence_section}

Return a JSON object with exactly these fields:
- hypothesis_id
- category
- statement
- supporting_evidence
- counterexamples
- status
- confidence

Rules:
1. Only cite evidence IDs present in the retrieved evidence.
2. Do not invent evidence.
3. The hypothesis must be supported by the provided evidence.
4. If the evidence is insufficient, say so in the statement.
5. Do not make unsupported linguistic claims.
""".strip()


def generate_hypothesis(
    *,
    state: AgentState,
    provider: LLMProvider,
    top_k: int = 5,
) -> AgentState:
    """Generate and validate an evidence-grounded hypothesis."""

    episode = state.get("episode")

    if episode is None:
        raise HypothesisGenerationError(
            "Cannot generate a hypothesis without an episode line."
        )

    evidence = state.get("evidence", [])

    retriever = EvidenceRetriever(evidence)

    retrieved = retriever.search(
        episode.source_text,
        top_k=top_k,
    )

    ranker = EvidenceRanker()
    ranked = ranker.rank(retrieved)

    if not ranked and evidence:
        evidence_payload = [
            {
                "evidence_id": item.evidence_id,
                "evidence_type": item.evidence_type.value
                if hasattr(item.evidence_type, "value")
                else str(item.evidence_type),
                "content": item.content,
            }
            for item in evidence[:top_k]
        ]
    else:
        evidence_payload = [
            {
                "evidence_id": item.evidence.evidence_id,
                "evidence_type": item.evidence.evidence_type.value,
                "content": item.evidence.content,
            }
            for item in ranked
        ]

    if not evidence_payload:
        raise HypothesisGenerationError(
            "No relevant evidence was found for the episode line."
        )

    prompt = build_hypothesis_prompt(
        episode,
        evidence_payload,
    )

    response = provider.generate(
        prompt,
        system_prompt=(
            "You are an evidence-grounded subtitle reasoning agent. "
            "Never invent evidence IDs."
        ),
        response_schema=Hypothesis,
    )

    if not isinstance(response, Hypothesis):
        raise HypothesisGenerationError(
            "LLM provider did not return a Hypothesis."
        )

    available_ids = {
        item["evidence_id"]
        for item in evidence_payload
    }

    invalid_supporting_ids = (
        set(response.supporting_evidence) - available_ids
    )

    if invalid_supporting_ids:
        raise HypothesisGenerationError(
            "Hypothesis references unavailable evidence IDs: "
            f"{sorted(invalid_supporting_ids)}"
        )

    response.supporting_evidence = list(
        dict.fromkeys(response.supporting_evidence)
    )

    hypotheses = list(state.get("hypotheses", []))
    hypotheses.append(response)

    state["hypotheses"] = hypotheses

    return state