"""Translate between Laya's JSON-shaped results and domain entities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...domain.entities import Answer, Decision, PredictionControls, QuestionType, Usage

_ANSWER_FIELDS = {"type", "choice", "score", "noul", "confidence", "answer_confidence", "probabilities", "legend"}


def answer_from_dict(payload: Mapping[str, Any]) -> Answer:
    kind = payload.get("type")
    try:
        qtype = QuestionType(kind)
    except ValueError:
        # An answer of a type this service does not know yet: keep all of it rather than fail.
        qtype = QuestionType.NOUL if "noul" in payload else QuestionType.CHOICE
    return Answer(
        type=qtype,
        choice=payload.get("choice"),
        score=payload.get("score"),
        noul=payload.get("noul"),
        confidence=payload.get("confidence"),
        answer_confidence=payload.get("answer_confidence"),
        probabilities=payload.get("probabilities"),
        legend=payload.get("legend"),
        extras={k: v for k, v in payload.items() if k not in _ANSWER_FIELDS},
    )


def usage_from_dict(payload: Mapping[str, Any] | None) -> Usage:
    payload = payload or {}
    return Usage(
        input_tokens=int(payload.get("input_tokens", 0) or 0),
        output_tokens=int(payload.get("output_tokens", 0) or 0),
        extras={k: v for k, v in payload.items() if k not in ("input_tokens", "output_tokens")},
    )


def decision_from_dict(payload: Mapping[str, Any], inference_ms: float | None = None) -> Decision:
    return Decision(
        model=str(payload.get("model", "")),
        answers={qid: answer_from_dict(answer) for qid, answer in (payload.get("answers") or {}).items()},
        usage=usage_from_dict(payload.get("usage")),
        routing=payload.get("routing"),
        inference_ms=inference_ms,
    )


def controls_to_kwargs(controls: PredictionControls, *, include_model: bool = True) -> dict[str, Any]:
    """Only the controls the caller set: an absent keyword means "inherit the Router's default"."""
    kwargs: dict[str, Any] = {}
    for key in ("task", "lang", "lang_guess", "max_len", "head_max_len", "min_confidence"):
        value = getattr(controls, key)
        if value is not None:
            kwargs[key] = dict(value) if isinstance(value, Mapping) else value
    if include_model:
        kwargs["model"] = controls.model
    return kwargs
