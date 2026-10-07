"""Business rules every decision request must satisfy before it reaches an engine.

The limits guard a self-hosted deployment against requests that are valid JSON but would tie up the
inference worker: the state is tokenized once per question and collated into one tensor, so an
unbounded request can exhaust memory. The defaults match ``laya-serve`` so a client written against
either server sees the same refusals.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .entities import (
    BatchDecisionRequest,
    DecisionRequest,
    JsonValue,
    PredictionControls,
    Question,
    QuestionType,
)
from .errors import InvalidRequestError, MissingStateError, PayloadTooLargeError


@dataclass(frozen=True)
class RequestLimits:
    """Size limits for one request. Every limit is enforced before inference starts."""

    max_questions: int = 64
    max_state_chars: int = 50_000
    max_batch_states: int = 64
    max_choice_options: int = 100
    max_score_levels: int = 32
    max_total_options: int = 512
    max_token_budget: int = 8192


def state_length(state: JsonValue) -> int:
    """Length of the text the engine will tokenize for ``state``.

    A string is read as-is; anything else is serialized as JSON with non-ASCII kept, which is what
    Laya tokenizes. Measuring ``str(state)`` instead would disagree with the tokenizer in both
    directions (quote escaping, ``\\u`` escapes).
    """
    if isinstance(state, str):
        return len(state)
    try:
        return len(json.dumps(state, ensure_ascii=False))
    except (TypeError, ValueError, RecursionError) as error:
        raise InvalidRequestError("'state' must be JSON-serializable") from error


def _validate_state(state: JsonValue, limits: RequestLimits, label: str = "state") -> None:
    if state is None:
        raise MissingStateError("'%s' is required" % label)
    if isinstance(state, str) and not state.strip():
        raise InvalidRequestError("'%s' must not be empty" % label)
    length = state_length(state)
    if length > limits.max_state_chars:
        raise PayloadTooLargeError("%s too large (%d > %d chars)" % (label, length, limits.max_state_chars))


def _validate_question(question: Question, limits: RequestLimits) -> None:
    qid = question.id
    if not qid:
        raise InvalidRequestError("question ids must be non-empty strings")
    if question.instructions is None or (isinstance(question.instructions, str) and not question.instructions.strip()):
        raise InvalidRequestError("question %r needs non-empty 'instructions'" % qid)

    criteria = question.criteria
    if question.type is QuestionType.CHOICE:
        if not isinstance(criteria, (list, tuple, dict)) or len(criteria) < 2:
            raise InvalidRequestError(
                "choice question %r needs 'criteria' with at least two options "
                "(a list of labels or a label -> description object)" % qid
            )
        if isinstance(criteria, (list, tuple)) and not all(isinstance(c, str) and c for c in criteria):
            raise InvalidRequestError("choice question %r: every label must be a non-empty string" % qid)
        if len(criteria) > limits.max_choice_options:
            raise PayloadTooLargeError(
                "too many choice options for %r (%d > %d)" % (qid, len(criteria), limits.max_choice_options)
            )
    elif question.type is QuestionType.SCORE:
        if not isinstance(criteria, (list, tuple)) or len(criteria) < 2:
            raise InvalidRequestError(
                "score question %r needs 'criteria' as a list of at least two rubric levels" % qid
            )
        if len(criteria) > limits.max_score_levels:
            raise PayloadTooLargeError(
                "too many score levels for %r (%d > %d)" % (qid, len(criteria), limits.max_score_levels)
            )
        for index, level in enumerate(criteria):
            if level is None:
                raise InvalidRequestError(
                    "score question %r has a null level at index %d; give every level a description" % (qid, index)
                )
    elif question.type is QuestionType.NOUL:
        if criteria is not None and (not isinstance(criteria, Mapping) or not set(criteria) <= {"true", "false"}):
            raise InvalidRequestError(
                "noul question %r: 'criteria' is optional and may only name 'true' and 'false'" % qid
            )


def _validate_questions(questions: Iterable[Question], limits: RequestLimits) -> None:
    questions = tuple(questions)
    if not questions:
        raise InvalidRequestError("'questions' must name at least one question")
    if len(questions) > limits.max_questions:
        raise PayloadTooLargeError("too many questions (%d > %d)" % (len(questions), limits.max_questions))
    seen: set[str] = set()
    total_options = 0
    for question in questions:
        if question.id in seen:
            raise InvalidRequestError("duplicate question id %r" % question.id)
        seen.add(question.id)
        _validate_question(question, limits)
        if question.type is not QuestionType.NOUL:
            total_options += question.option_count()
    if total_options > limits.max_total_options:
        raise PayloadTooLargeError(
            "too many answer options across questions (%d > %d)" % (total_options, limits.max_total_options)
        )


def _validate_threshold(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InvalidRequestError("%s must be a number in [0, 1]" % label)
    if not 0.0 <= float(value) <= 1.0:
        raise InvalidRequestError("%s must be in [0, 1], got %r" % (label, value))


def _validate_controls(controls: PredictionControls, limits: RequestLimits) -> None:
    for key in ("max_len", "head_max_len"):
        value = getattr(controls, key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise InvalidRequestError("%s must be a positive integer" % key)
        if value > limits.max_token_budget:
            raise InvalidRequestError("%s exceeds server limit (%d > %d)" % (key, value, limits.max_token_budget))
    threshold = controls.min_confidence
    if threshold is None:
        return
    if isinstance(threshold, Mapping):
        if not threshold:
            raise InvalidRequestError("min_confidence map must name at least one bucket")
        for bucket, value in threshold.items():
            _validate_threshold(value, "min_confidence[%r]" % bucket)
    else:
        _validate_threshold(threshold, "min_confidence")


def validate_decision_request(request: DecisionRequest, limits: RequestLimits) -> None:
    """Raise a domain error if ``request`` breaks a rule; return ``None`` when it may run."""
    if request.state is None:
        # Checked first: Jev answers a missing state with 400 whatever else the body holds.
        raise MissingStateError("'state' is required")
    _validate_questions(request.questions, limits)
    _validate_state(request.state, limits)
    _validate_controls(request.controls, limits)


def validate_batch_request(request: BatchDecisionRequest, limits: RequestLimits) -> None:
    """Batch version of :func:`validate_decision_request`."""
    if not request.states:
        raise InvalidRequestError("'states' must be a non-empty list")
    if len(request.states) > limits.max_batch_states:
        raise PayloadTooLargeError("too many states (%d > %d)" % (len(request.states), limits.max_batch_states))
    _validate_questions(request.questions, limits)
    for index, state in enumerate(request.states):
        _validate_state(state, limits, label="states[%d]" % index)
    _validate_controls(request.controls, limits)
    if request.batch_size is not None and (
        isinstance(request.batch_size, bool) or not isinstance(request.batch_size, int) or request.batch_size < 1
    ):
        raise InvalidRequestError("batch_size must be a positive integer")
