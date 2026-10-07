from __future__ import annotations

import pytest

from laya_client.domain.entities import (
    BatchDecisionRequest,
    DecisionRequest,
    PredictionControls,
    Question,
    QuestionType,
)
from laya_client.domain.errors import InvalidRequestError, MissingStateError, PayloadTooLargeError
from laya_client.domain.policies import (
    RequestLimits,
    state_length,
    validate_batch_request,
    validate_decision_request,
)

LIMITS = RequestLimits()


def _request(state="hello", questions=None, **controls) -> DecisionRequest:
    questions = questions or (Question("q", QuestionType.NOUL, "Is it?"),)
    return DecisionRequest(state=state, questions=questions, controls=PredictionControls(**controls))


def test_valid_request_passes(questions):
    validate_decision_request(_request(questions=questions), LIMITS)


def test_missing_state_is_reported_before_anything_else():
    with pytest.raises(MissingStateError):
        validate_decision_request(DecisionRequest(state=None, questions=()), LIMITS)


def test_blank_string_state_is_invalid():
    with pytest.raises(InvalidRequestError):
        validate_decision_request(_request(state="   "), LIMITS)


def test_state_length_measures_serialized_json():
    assert state_length("abc") == 3
    # A quote is escaped by JSON, so the tokenized text is longer than the raw characters.
    assert state_length({"b": '"'}) == len('{"b": "\\""}')


def test_oversized_state_is_refused():
    with pytest.raises(PayloadTooLargeError):
        validate_decision_request(_request(state="x" * 11), RequestLimits(max_state_chars=10))


def test_no_questions_is_invalid():
    with pytest.raises(InvalidRequestError):
        validate_decision_request(DecisionRequest(state="x", questions=()), LIMITS)


def test_too_many_questions():
    questions = tuple(Question("q%d" % i, QuestionType.NOUL, "?") for i in range(3))
    with pytest.raises(PayloadTooLargeError):
        validate_decision_request(_request(questions=questions), RequestLimits(max_questions=2))


def test_choice_needs_two_options():
    q = Question("c", QuestionType.CHOICE, "pick", ["only"])
    with pytest.raises(InvalidRequestError, match="at least two"):
        validate_decision_request(_request(questions=(q,)), LIMITS)


def test_choice_option_cap():
    q = Question("c", QuestionType.CHOICE, "pick", ["a", "b", "c"])
    with pytest.raises(PayloadTooLargeError):
        validate_decision_request(_request(questions=(q,)), RequestLimits(max_choice_options=2))


def test_score_null_level_is_rejected():
    q = Question("s", QuestionType.SCORE, "how much", ["low", None, "high"])
    with pytest.raises(InvalidRequestError, match="null level at index 1"):
        validate_decision_request(_request(questions=(q,)), LIMITS)


def test_noul_criteria_keys():
    ok = Question("n", QuestionType.NOUL, "?", {"true": "yes", "false": "no"})
    validate_decision_request(_request(questions=(ok,)), LIMITS)
    bad = Question("n", QuestionType.NOUL, "?", {"maybe": "?"})
    with pytest.raises(InvalidRequestError):
        validate_decision_request(_request(questions=(bad,)), LIMITS)


def test_duplicate_question_ids():
    q = Question("dup", QuestionType.NOUL, "?")
    with pytest.raises(InvalidRequestError, match="duplicate"):
        validate_decision_request(_request(questions=(q, q)), LIMITS)


@pytest.mark.parametrize("value", [-0.1, 1.5, float("nan"), True])
def test_min_confidence_range(value):
    with pytest.raises(InvalidRequestError):
        validate_decision_request(_request(min_confidence=value), LIMITS)


def test_min_confidence_map():
    validate_decision_request(_request(min_confidence={"choice:2": 0.6, "default": 0.5}), LIMITS)
    with pytest.raises(InvalidRequestError):
        validate_decision_request(_request(min_confidence={"default": 2}), LIMITS)


def test_token_budget_cap():
    with pytest.raises(InvalidRequestError, match="exceeds server limit"):
        validate_decision_request(_request(max_len=9000), LIMITS)


def test_batch_limits(questions):
    validate_batch_request(BatchDecisionRequest(states=("a", "b"), questions=questions), LIMITS)
    with pytest.raises(InvalidRequestError):
        validate_batch_request(BatchDecisionRequest(states=(), questions=questions), LIMITS)
    with pytest.raises(PayloadTooLargeError):
        validate_batch_request(
            BatchDecisionRequest(states=("a", "b", "c"), questions=questions), RequestLimits(max_batch_states=2)
        )
    with pytest.raises(MissingStateError):
        validate_batch_request(BatchDecisionRequest(states=("a", None), questions=questions), LIMITS)
