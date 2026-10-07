from __future__ import annotations

import math

import pytest

from laya_client.domain.entities import (
    BatchDecisionRequest,
    DecisionRequest,
    PredictionControls,
    QuestionType,
)
from laya_client.domain.errors import EngineUnavailableError, InferenceFailedError, InvalidRequestError
from laya_client.infrastructure.engines import LayaEngineConfig, LayaRouterEngine, MockDecisionEngine


def test_mock_is_deterministic_and_well_formed(questions):
    engine = MockDecisionEngine()
    request = DecisionRequest({"body": "refund please"}, questions)
    first, second = engine.predict(request), engine.predict(request)
    assert first.answers == second.answers

    choice = first.answers["department"]
    assert choice.type is QuestionType.CHOICE
    assert choice.choice in {"billing", "tech"}
    assert math.isclose(sum(choice.probabilities.values()), 1.0, abs_tol=1e-4)

    score = first.answers["urgency"]
    assert 0.0 <= score.score <= 2.0
    assert score.legend == {"0": "low", "1": "medium", "2": "high"}

    noul = first.answers["churn"]
    assert 0.0 < noul.noul < 1.0


def test_mock_abstention(questions):
    decision = MockDecisionEngine().predict(DecisionRequest("x", questions, PredictionControls(min_confidence=1.0)))
    for answer in decision.answers.values():
        assert answer.extras["abstention"] == "abstained"
        assert answer.extras["low_confidence"] is True


class FakeRouter:
    """Stands in for ``laya.Router``: records calls and returns Laya-shaped payloads."""

    loaded = ["english"]
    loaded_revisions = {"english": "abc123"}

    def __init__(self, fail: Exception | None = None) -> None:
        self.calls: list = []
        self.fail = fail

    def _result(self, questions):
        return {
            "model": "laya-rl-agent",
            "answers": {
                qid: {
                    "type": "noul",
                    "noul": 0.7,
                    "confidence": 0.7,
                    "answer_confidence": 0.7,
                    "action": {"act_probability": 1.0},
                }
                for qid in questions
            },
            "usage": {"input_tokens": 10, "output_tokens": 0, "truncated": False},
            "routing": {"model": "english", "reason": "English Latin text"},
        }

    def predict(self, state, questions, **kwargs):
        self.calls.append(("predict", state, kwargs))
        if self.fail:
            raise self.fail
        return self._result(questions)

    def predict_batch(self, requests, **kwargs):
        self.calls.append(("predict_batch", requests, kwargs))
        return [self._result(r["questions"]) for r in requests]


def test_laya_engine_forwards_only_controls_that_were_set(questions):
    router = FakeRouter()
    engine = LayaRouterEngine(LayaEngineConfig(), router=router)
    decision = engine.predict(DecisionRequest("hi", questions, PredictionControls(model="english", max_len=256)))
    _, _, kwargs = router.calls[0]
    assert kwargs == {"model": "english", "max_len": 256}
    assert decision.answers["churn"].extras == {"action": {"act_probability": 1.0}}
    assert decision.usage.extras == {"truncated": False}
    assert decision.inference_ms is not None
    assert engine.status().ready and engine.status().loaded == ("english",)


def test_laya_engine_batch_splits_controls(questions):
    router = FakeRouter()
    engine = LayaRouterEngine(LayaEngineConfig(), router=router)
    result = engine.predict_batch(
        BatchDecisionRequest(
            ("a", "b"),
            questions,
            PredictionControls(lang="de", min_confidence=0.5),
            sort_by_length=True,
        )
    )
    _, requests, kwargs = router.calls[0]
    assert kwargs == {"min_confidence": 0.5, "sort_by_length": True}
    assert all(r["lang"] == "de" and "min_confidence" not in r for r in requests)
    assert result.total_usage.input_tokens == 20


def test_laya_engine_chunks_large_batches(questions):
    router = FakeRouter()
    engine = LayaRouterEngine(LayaEngineConfig(max_batch_tokens=3 * 512 * 2), router=router)
    engine.predict_batch(BatchDecisionRequest(tuple("abcde"), questions))
    assert router.calls[0][2]["batch_size"] == 2


def test_laya_engine_maps_errors(questions):
    engine = LayaRouterEngine(LayaEngineConfig(), router=FakeRouter(fail=ValueError("unknown task 'x'")))
    with pytest.raises(InvalidRequestError, match="unknown task"):
        engine.predict(DecisionRequest("hi", questions))
    engine = LayaRouterEngine(LayaEngineConfig(), router=FakeRouter(fail=RuntimeError("CUDA OOM /secret/path")))
    with pytest.raises(InferenceFailedError) as info:
        engine.predict(DecisionRequest("hi", questions))
    assert "secret" not in info.value.message


def test_laya_engine_without_laya_installed(monkeypatch, questions):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "laya":
            raise ImportError("no laya")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    engine = LayaRouterEngine(LayaEngineConfig())
    with pytest.raises(EngineUnavailableError, match="LAYA_ENGINE=mock"):
        engine.predict(DecisionRequest("hi", questions))
    assert engine.status().ready is False
