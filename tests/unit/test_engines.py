from __future__ import annotations

import pytest

from laya_client.domain.entities import BatchDecisionRequest, DecisionRequest, PredictionControls
from laya_client.domain.errors import EngineUnavailableError, InferenceFailedError, InvalidRequestError
from laya_client.infrastructure.engines import LayaEngineConfig, LayaRouterEngine, RuntimeInfo


class FakeRouter:
    """Stands in for ``laya.Router``: records calls and returns Laya-shaped payloads."""

    loaded_revisions = {"english": "abc123"}

    def __init__(self, fail: Exception | None = None, loaded=("english",), preload_error=None, p_true=0.7):
        self.calls: list = []
        self.fail = fail
        self.loaded = list(loaded)
        self.preload_error = preload_error
        self.p_true = p_true
        self.preloaded = None

    def preload(self, names=None):
        self.preloaded = names
        if self.preload_error:
            raise self.preload_error

    def _result(self, questions):
        return {
            "model": "laya-rl-agent",
            "answers": {
                qid: {
                    "type": "noul",
                    "noul": self.p_true,
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


CPU = RuntimeInfo(device="cpu", torch_version="2.8.0", laya_version="0.3.28")
GPU = RuntimeInfo(
    device="cuda", torch_version="2.8.0+cu128", torch_cuda="12.8", accelerator="NVIDIA GeForce RTX 5080 (sm_120)"
)


def _engine(router, runtime=CPU, config=None, device_of=lambda router, name: "cpu"):
    options = []

    def factory(**kwargs):
        options.append(kwargs)
        return router

    engine = LayaRouterEngine(
        config or LayaEngineConfig(models=("english",)),
        runtime_check=lambda device, require_gpu: runtime,
        router_factory=factory,
        device_of=device_of,
    )
    return engine, options


# -- start-up -----------------------------------------------------------------------------------


def test_start_loads_verifies_and_smoke_tests():
    router = FakeRouter()
    engine, options = _engine(router, config=LayaEngineConfig(models=("english",), default_model="multilingual"))
    engine.start()
    assert options == [{"device": "cpu", "auto_task_detection": False, "default": "multilingual"}]
    assert router.preloaded == ["english"]
    assert router.calls[0][0] == "predict" and router.calls[0][2] == {"model": "english"}
    status = engine.status()
    assert status.ready and status.loaded == ("english",) and status.device == "cpu"
    assert status.details["runtime"]["torch"] == "2.8.0"


def test_start_passes_the_resolved_gpu_device_to_laya():
    router = FakeRouter()
    engine, options = _engine(router, runtime=GPU, device_of=lambda router, name: "cuda:0")
    engine.start()
    assert options[0]["device"] == "cuda"


def test_start_fails_when_the_runtime_check_fails():
    def broken(device, require_gpu):
        raise EngineUnavailableError("the 'laya' package is not installed")

    engine = LayaRouterEngine(LayaEngineConfig(), runtime_check=broken)
    with pytest.raises(EngineUnavailableError, match="not installed"):
        engine.start()
    assert engine.status().ready is False


def test_start_fails_when_the_download_fails():
    engine, _ = _engine(FakeRouter(preload_error=OSError("couldn't connect to huggingface.co")))
    with pytest.raises(EngineUnavailableError, match="huggingface.co"):
        engine.start()
    assert engine.status().ready is False


def test_start_fails_when_nothing_loaded():
    engine, _ = _engine(FakeRouter(loaded=()))
    with pytest.raises(EngineUnavailableError, match="no checkpoint"):
        engine.start()


def test_start_fails_when_laya_fell_back_to_cpu():
    engine, _ = _engine(FakeRouter(), runtime=GPU, device_of=lambda router, name: "cpu")
    with pytest.raises(EngineUnavailableError, match="fell back from the GPU"):
        engine.start()


def test_start_fails_when_the_test_prediction_fails():
    engine, _ = _engine(FakeRouter(fail=RuntimeError("CUDA error: no kernel image is available")))
    with pytest.raises(EngineUnavailableError, match="no kernel image"):
        engine.start()
    assert engine.status().ready is False


def test_start_fails_on_an_invalid_probability():
    engine, _ = _engine(FakeRouter(p_true=float("nan")))
    with pytest.raises(EngineUnavailableError, match="invalid probability"):
        engine.start()


def test_predict_before_start_is_refused(questions):
    engine = LayaRouterEngine(LayaEngineConfig())
    with pytest.raises(EngineUnavailableError, match="not running"):
        engine.predict(DecisionRequest("hi", questions))


# -- inference ----------------------------------------------------------------------------------


def test_forwards_only_controls_that_were_set(questions):
    router = FakeRouter()
    engine = LayaRouterEngine(LayaEngineConfig(), router=router)
    decision = engine.predict(DecisionRequest("hi", questions, PredictionControls(model="english", max_len=256)))
    _, _, kwargs = router.calls[0]
    assert kwargs == {"model": "english", "max_len": 256}
    assert decision.answers["churn"].extras == {"action": {"act_probability": 1.0}}
    assert decision.usage.extras == {"truncated": False}
    assert decision.inference_ms is not None
    assert engine.status().ready and engine.status().loaded == ("english",)


def test_batch_splits_controls(questions):
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


def test_chunks_large_batches(questions):
    router = FakeRouter()
    engine = LayaRouterEngine(LayaEngineConfig(max_batch_tokens=3 * 512 * 2), router=router)
    engine.predict_batch(BatchDecisionRequest(tuple("abcde"), questions))
    assert router.calls[0][2]["batch_size"] == 2


def test_maps_errors(questions):
    engine = LayaRouterEngine(LayaEngineConfig(), router=FakeRouter(fail=ValueError("unknown task 'x'")))
    with pytest.raises(InvalidRequestError, match="unknown task"):
        engine.predict(DecisionRequest("hi", questions))
    engine = LayaRouterEngine(LayaEngineConfig(), router=FakeRouter(fail=RuntimeError("CUDA OOM /secret/path")))
    with pytest.raises(InferenceFailedError) as info:
        engine.predict(DecisionRequest("hi", questions))
    assert "secret" not in info.value.message
