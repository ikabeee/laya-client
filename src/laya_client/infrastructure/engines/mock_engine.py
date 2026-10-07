"""A deterministic stand-in for Laya, for development, CI and contract tests.

It answers every request instantly and without a model: probabilities are derived from a hash of
the state and the question, so the same request always gets the same answer. The answers have the
exact shape Laya returns, which makes the mock useful for building and testing clients, and useless
for real decisions. ``/health`` and every routing report say ``mock`` so nobody mistakes one for
the other.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from ...domain.entities import (
    Answer,
    BatchDecision,
    BatchDecisionRequest,
    Decision,
    DecisionRequest,
    EngineStatus,
    MinConfidence,
    PredictionControls,
    Question,
    QuestionType,
    Usage,
)
from ...domain.ports import DecisionEngine

_DEFAULT_MODEL = "english"


def _unit_floats(*parts: str, count: int) -> list[float]:
    """``count`` floats in [0, 1) derived deterministically from ``parts``."""
    values: list[float] = []
    counter = 0
    while len(values) < count:
        digest = hashlib.sha256(("\x1f".join(parts) + "\x1e%d" % counter).encode("utf-8")).digest()
        values.extend(int.from_bytes(digest[i : i + 4], "big") / 2**32 for i in range(0, 32, 4))
        counter += 1
    return values[:count]


def _softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    exps = [math.exp(x - peak) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _normalised_entropy_confidence(probs: list[float]) -> float:
    if len(probs) < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return round(1.0 - entropy / math.log(len(probs)), 6)


def _threshold_for(min_confidence: MinConfidence | None, question: Question) -> float | None:
    if min_confidence is None:
        return None
    if isinstance(min_confidence, Mapping):
        bucket = "%s:%d" % (question.type.value, question.option_count())
        value = min_confidence.get(bucket, min_confidence.get("default"))
        return None if value is None else float(value)
    return float(min_confidence)


class MockDecisionEngine(DecisionEngine):
    """Implements the ``DecisionEngine`` port without loading any model."""

    def status(self) -> EngineStatus:
        return EngineStatus(
            engine="mock",
            ready=True,
            loaded=(_DEFAULT_MODEL,),
            device="cpu",
            details={"note": "deterministic mock answers; no model is loaded"},
        )

    def _answer(self, state_key: str, question: Question, min_confidence: MinConfidence | None) -> Answer:
        if question.type is QuestionType.NOUL:
            (u,) = _unit_floats(state_key, question.id, count=1)
            p_true = round(0.02 + 0.96 * u, 6)
            answer = Answer(
                type=QuestionType.NOUL,
                noul=p_true,
                confidence=round(max(p_true, 1 - p_true), 6),
                answer_confidence=round(max(p_true, 1 - p_true), 6),
            )
        else:
            criteria = question.criteria
            labels = list(criteria) if isinstance(criteria, (list, tuple, dict)) else []
            logits = [4.0 * u for u in _unit_floats(state_key, question.id, count=len(labels))]
            probs = [round(p, 6) for p in _softmax(logits)]
            if question.type is QuestionType.CHOICE:
                keys = [str(label) for label in labels]
                best = max(range(len(keys)), key=probs.__getitem__)
                answer = Answer(
                    type=QuestionType.CHOICE,
                    choice=keys[best],
                    confidence=_normalised_entropy_confidence(probs),
                    answer_confidence=max(probs),
                    probabilities=dict(zip(keys, probs, strict=True)),
                )
            else:
                keys = [str(i) for i in range(len(labels))]
                answer = Answer(
                    type=QuestionType.SCORE,
                    score=round(sum(i * p for i, p in enumerate(probs)), 6),
                    confidence=_normalised_entropy_confidence(probs),
                    answer_confidence=max(probs),
                    probabilities=dict(zip(keys, probs, strict=True)),
                    legend=dict(zip(keys, labels, strict=True)),
                )

        threshold = _threshold_for(min_confidence, question)
        if threshold is None:
            return answer
        extras: dict[str, Any] = {"abstention_threshold": threshold}
        if (answer.answer_confidence or 0.0) < threshold:
            extras.update(abstention="abstained", low_confidence=True)
        else:
            extras["abstention"] = "passed"
        return replace(answer, extras=extras)

    def _decide(self, state: Any, questions: tuple[Question, ...], request_controls: PredictionControls) -> Decision:
        state_key = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)
        model = request_controls.model or _DEFAULT_MODEL
        answers = {q.id: self._answer(state_key, q, request_controls.min_confidence) for q in questions}
        state_tokens = max(1, len(state_key.split()))
        return Decision(
            model="mock/%s" % model,
            answers=answers,
            usage=Usage(input_tokens=state_tokens * len(questions), output_tokens=0),
            routing={
                "model": model,
                "repo": "mock",
                "reason": "mock engine: deterministic answers, no model loaded",
                "detection": None,
                "workflow": None,
            },
        )

    def predict(self, request: DecisionRequest) -> Decision:
        started = time.perf_counter()
        decision = self._decide(request.state, request.questions, request.controls)
        return replace(decision, inference_ms=(time.perf_counter() - started) * 1000.0)

    def predict_batch(self, request: BatchDecisionRequest) -> BatchDecision:
        started = time.perf_counter()
        results = tuple(self._decide(state, request.questions, request.controls) for state in request.states)
        total = sum((d.usage for d in results), start=Usage())
        return BatchDecision(results=results, total_usage=total, inference_ms=(time.perf_counter() - started) * 1000.0)
