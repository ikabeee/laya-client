"""Response bodies, for the OpenAPI document.

Every model allows extra fields: Laya adds diagnostics over time (abstention, option collapse,
truncation) and a client must see them without this service being upgraded first.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class ChoiceAnswer(_Open):
    type: Literal["choice"]
    choice: str = Field(description="The most probable label.")
    confidence: float = Field(description="1 - normalised entropy of the distribution.")
    answer_confidence: float | None = Field(None, description="Probability of the reported label.")
    probabilities: dict[str, float]


class ScoreAnswer(_Open):
    type: Literal["score"]
    score: float = Field(description="Expected zero-based rubric index.")
    confidence: float
    answer_confidence: float | None = None
    probabilities: dict[str, float]
    legend: dict[str, Any] = Field(description="Rubric index -> level description.")


class NoulAnswer(_Open):
    type: Literal["noul"]
    noul: float = Field(description="P(true), between 0 and 1.")
    confidence: float | None = None
    answer_confidence: float | None = None


class UsageSchema(_Open):
    input_tokens: int
    output_tokens: int = Field(description="Always 0: the head answers in one pass and generates nothing.")


class DecisionResponse(_Open):
    model: str = Field(description="The checkpoint that answered.")
    answers: dict[str, ChoiceAnswer | ScoreAnswer | NoulAnswer]
    usage: UsageSchema
    routing: dict[str, Any] | None = Field(None, description="Which checkpoint the router chose and why.")


class BatchDecisionResponse(_Open):
    results: list[DecisionResponse]
    total_usage: UsageSchema


class ModelSchema(BaseModel):
    name: str
    repo: str
    description: str
    aliases: list[str]
    loaded: bool


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelSchema]


class PresetSummary(BaseModel):
    name: str
    title: str
    description: str
    state_field: str | None = Field(description="The state field the questions read.")
    questions: list[str] = Field(description="Question ids.")


class PresetListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[PresetSummary]


class PresetDetail(BaseModel):
    name: str
    title: str
    description: str
    state_field: str | None
    questions: dict[str, dict[str, Any]] = Field(description="Ready to send as `questions` to /v1/systemone.")


class HealthResponse(_Open):
    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    ready: bool
    engine: str
    loaded: list[str]
    device: str | None = None
    details: dict[str, Any] = {}


class ErrorResponse(BaseModel):
    detail: str
    error: str = Field(description="Stable machine-readable error code.")
