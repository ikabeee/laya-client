"""Request schemas -> domain entities."""

from __future__ import annotations

from ...domain.entities import (
    BatchDecisionRequest,
    DecisionRequest,
    PredictionControls,
    Question,
    QuestionType,
)
from .schemas.requests import (
    NoulQuestion,
    PredictionControlsSchema,
    SystemOneBatchRequest,
    SystemOneRequest,
)


def to_controls(body: PredictionControlsSchema) -> PredictionControls:
    return PredictionControls(
        model=body.model,
        task=body.task,
        lang=body.lang,
        lang_guess=body.lang_guess,
        max_len=body.max_len,
        head_max_len=body.head_max_len,
        min_confidence=body.min_confidence,
    )


def to_questions(questions: dict) -> tuple[Question, ...]:
    result = []
    for qid, question in questions.items():
        criteria = question.criteria
        if isinstance(question, NoulQuestion):
            criteria = criteria.model_dump(exclude_none=True) if criteria is not None else None
            criteria = criteria or None
        result.append(
            Question(id=qid, type=QuestionType(question.type), instructions=question.instructions, criteria=criteria)
        )
    return tuple(result)


def to_decision_request(body: SystemOneRequest) -> DecisionRequest:
    return DecisionRequest(state=body.state, questions=to_questions(body.questions), controls=to_controls(body))


def to_batch_request(body: SystemOneBatchRequest) -> BatchDecisionRequest:
    return BatchDecisionRequest(
        states=tuple(body.states),
        questions=to_questions(body.questions),
        controls=to_controls(body),
        batch_size=body.batch_size,
        sort_by_length=body.sort_by_length,
    )
