from __future__ import annotations

from laya_client.domain.entities import Answer, Decision, QuestionType, Usage
from laya_client.interfaces.http.presenters import decision_payload


def _decision() -> Decision:
    return Decision(
        model="laya-rl-agent",
        answers={
            "c": Answer(
                QuestionType.CHOICE,
                choice="a",
                confidence=0.5,
                answer_confidence=0.8,
                probabilities={"a": 0.8, "b": 0.2},
                extras={"action": {"act_probability": 1.0}},
            ),
            "n": Answer(QuestionType.NOUL, noul=0.9, confidence=0.9, answer_confidence=0.9),
        },
        usage=Usage(10, 0, extras={"truncated": False}),
        routing={"model": "english"},
    )


def test_full_payload_keeps_laya_extras():
    payload = decision_payload(_decision())
    assert payload["routing"] == {"model": "english"}
    assert payload["answers"]["c"]["action"] == {"act_probability": 1.0}
    assert payload["answers"]["c"]["answer_confidence"] == 0.8
    assert payload["usage"]["truncated"] is False


def test_strict_payload_is_the_jev_contract():
    payload = decision_payload(_decision(), strict=True)
    assert set(payload) == {"model", "answers", "usage"}
    assert payload["answers"]["c"] == {
        "type": "choice",
        "choice": "a",
        "confidence": 0.5,
        "probabilities": {"a": 0.8, "b": 0.2},
    }
    assert payload["answers"]["n"] == {"type": "noul", "noul": 0.9}
    assert payload["usage"] == {"input_tokens": 10, "output_tokens": 0}
