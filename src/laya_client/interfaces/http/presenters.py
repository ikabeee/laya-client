"""Domain entities -> JSON payloads.

The full payload is Laya's (Jev's contract plus ``routing``, ``answer_confidence`` and the usage
diagnostics). With ``strict=True`` it is projected onto the Jev contract alone, for clients that
validate responses with no extra fields allowed.
"""

from __future__ import annotations

from typing import Any

from ...domain.entities import (
    Answer,
    BatchDecision,
    Decision,
    EngineStatus,
    ModelInfo,
    Preset,
    QuestionType,
    Usage,
)


def answer_payload(answer: Answer, *, strict: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": answer.type.value}
    if answer.type is QuestionType.CHOICE:
        payload.update(choice=answer.choice, confidence=answer.confidence, probabilities=answer.probabilities)
    elif answer.type is QuestionType.SCORE:
        payload.update(
            score=answer.score,
            confidence=answer.confidence,
            probabilities=answer.probabilities,
            legend=answer.legend,
        )
    else:
        payload["noul"] = answer.noul
        if strict:
            return payload
        if answer.confidence is not None:
            payload["confidence"] = answer.confidence
    if strict:
        return payload
    if answer.answer_confidence is not None:
        payload["answer_confidence"] = answer.answer_confidence
    payload.update(answer.extras)
    return payload


def usage_payload(usage: Usage, *, strict: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}
    if not strict:
        payload.update(usage.extras)
    return payload


def decision_payload(decision: Decision, *, strict: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": decision.model,
        "answers": {qid: answer_payload(a, strict=strict) for qid, a in decision.answers.items()},
        "usage": usage_payload(decision.usage, strict=strict),
    }
    if not strict and decision.routing is not None:
        payload["routing"] = dict(decision.routing)
    return payload


def batch_payload(batch: BatchDecision, *, strict: bool = False) -> dict[str, Any]:
    return {
        "results": [decision_payload(d, strict=strict) for d in batch.results],
        "total_usage": usage_payload(batch.total_usage, strict=True),
    }


def model_payload(model: ModelInfo) -> dict[str, Any]:
    return {
        "name": model.name,
        "repo": model.repo,
        "description": model.description,
        "aliases": list(model.aliases),
        "loaded": model.loaded,
    }


def preset_summary_payload(preset: Preset) -> dict[str, Any]:
    return {
        "name": preset.name,
        "title": preset.title,
        "description": preset.description,
        "state_field": preset.state_field,
        "questions": [q.id for q in preset.questions],
    }


def preset_detail_payload(preset: Preset) -> dict[str, Any]:
    return {
        "name": preset.name,
        "title": preset.title,
        "description": preset.description,
        "state_field": preset.state_field,
        "questions": {q.id: q.to_definition() for q in preset.questions},
    }


def readiness_payload(status: EngineStatus) -> dict[str, Any]:
    return {
        "ready": status.ready,
        "engine": status.engine,
        "loaded": list(status.loaded),
        "device": status.device,
        "details": dict(status.details),
    }
