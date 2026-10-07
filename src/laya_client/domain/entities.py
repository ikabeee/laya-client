"""Domain entities and value objects for System-1 decisions.

A *decision request* is a state (the text or JSON the model reads) plus a set of typed questions.
A *decision* is one answer per question, the token usage, and how the request was routed. These
mirror the Jev ``/v1/systemone`` contract that Laya answers, but carry no transport concerns.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

#: Anything that survives a JSON round trip: the shape a state may take.
JsonValue = Any


class QuestionType(str, Enum):
    """The three question kinds Laya answers in one forward pass."""

    CHOICE = "choice"  # pick one label out of N
    SCORE = "score"  # expected position on an ordered rubric
    NOUL = "noul"  # probability that a yes/no statement is true


@dataclass(frozen=True)
class Question:
    """One typed question asked about a state."""

    id: str
    type: QuestionType
    instructions: JsonValue
    criteria: JsonValue = None

    def option_count(self) -> int:
        """How many answer options the question defines (2 for a yes/no question)."""
        if self.type is QuestionType.NOUL:
            return 2
        if isinstance(self.criteria, (list, tuple, dict)):
            return len(self.criteria)
        return 0

    def to_definition(self) -> dict[str, Any]:
        """The question as the engine expects it: ``{type, instructions, criteria?}``."""
        definition: dict[str, Any] = {"type": self.type.value, "instructions": self.instructions}
        if self.criteria is not None:
            definition["criteria"] = self.criteria
        return definition


#: Abstention threshold: one number for every answer, or a per-bucket map (``"choice:3-5" -> 0.6``).
MinConfidence = float | Mapping[str, float]


@dataclass(frozen=True)
class PredictionControls:
    """Per-request knobs a caller may set. ``None`` always means "inherit the deployment default"."""

    model: str | None = None
    task: str | None = None
    lang: str | None = None
    lang_guess: str | None = None
    max_len: int | None = None
    head_max_len: int | None = None
    min_confidence: MinConfidence | None = None

    def with_model(self, model: str | None) -> PredictionControls:
        return PredictionControls(
            model=model,
            task=self.task,
            lang=self.lang,
            lang_guess=self.lang_guess,
            max_len=self.max_len,
            head_max_len=self.head_max_len,
            min_confidence=self.min_confidence,
        )


@dataclass(frozen=True)
class DecisionRequest:
    """A single state and the questions to answer about it."""

    state: JsonValue
    questions: tuple[Question, ...]
    controls: PredictionControls = field(default_factory=PredictionControls)

    def question_definitions(self) -> dict[str, dict[str, Any]]:
        return {q.id: q.to_definition() for q in self.questions}


@dataclass(frozen=True)
class BatchDecisionRequest:
    """Several states answered against the same questions, sharing forward passes."""

    states: tuple[JsonValue, ...]
    questions: tuple[Question, ...]
    controls: PredictionControls = field(default_factory=PredictionControls)
    batch_size: int | None = None
    sort_by_length: bool = False

    def question_definitions(self) -> dict[str, dict[str, Any]]:
        return {q.id: q.to_definition() for q in self.questions}


@dataclass(frozen=True)
class Answer:
    """The answer to one question.

    Only the fields that belong to the answer's type are set: ``choice`` for a choice question,
    ``score`` + ``legend`` for a score question, ``noul`` for a yes/no question. ``extras`` keeps
    anything else the engine reported (``action``, ``abstention``, ``low_confidence`` ...) so a
    newer engine never has its output silently dropped.
    """

    type: QuestionType
    choice: str | None = None
    score: float | None = None
    noul: float | None = None
    confidence: float | None = None
    answer_confidence: float | None = None
    probabilities: Mapping[str, float] | None = None
    legend: Mapping[str, JsonValue] | None = None
    extras: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class Usage:
    """Token accounting. ``output_tokens`` is always 0: the head answers in one pass."""

    input_tokens: int = 0
    output_tokens: int = 0
    extras: Mapping[str, JsonValue] = field(default_factory=dict)

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True)
class Decision:
    """Every answer for one state, plus usage and the routing report."""

    model: str
    answers: Mapping[str, Answer]
    usage: Usage
    routing: Mapping[str, JsonValue] | None = None
    inference_ms: float | None = None


@dataclass(frozen=True)
class BatchDecision:
    """Decisions in the order the states were sent, and their summed usage."""

    results: tuple[Decision, ...]
    total_usage: Usage
    inference_ms: float | None = None


@dataclass(frozen=True)
class ModelInfo:
    """A checkpoint the deployment can answer with."""

    name: str
    repo: str
    description: str
    aliases: tuple[str, ...] = ()
    loaded: bool = False


@dataclass(frozen=True)
class Preset:
    """A ready-made question set for a common workflow."""

    name: str
    title: str
    description: str
    questions: tuple[Question, ...]
    state_field: str | None = None


@dataclass(frozen=True)
class EngineStatus:
    """What the inference engine reports about itself."""

    engine: str
    ready: bool
    loaded: tuple[str, ...] = ()
    device: str | None = None
    details: Mapping[str, JsonValue] = field(default_factory=dict)
