"""Request bodies. Field names and meanings follow Jev's ``/v1/systemone`` and ``laya-serve``."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

JsonState = Annotated[
    str | dict[str, Any] | list[Any],
    Field(description="What the model reads: plain text, or any JSON object/array (serialized as JSON)."),
]
Instructions = Annotated[
    str | dict[str, Any] | list[Any],
    Field(description="The question, in natural language. Name the state field it reads in backticks."),
]

#: ``laya.Router.predict`` arguments that run code inside the server process. A JSON body cannot
#: carry them, so a request naming one is refused (422) instead of having it silently dropped.
HOOK_FIELDS = ("hooks", "on_predict_start", "on_predict_end", "hooks_raise", "hooks_timeout")

_SYSTEMONE_EXAMPLE: dict[str, Any] = {
    "state": {"body": "Hi, we were billed twice for March. Please refund it today or we cancel."},
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle `body`?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs, outages, system errors",
                "other": "everything else",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is `body`?",
            "criteria": ["not urgent", "soon", "blocking"],
        },
        "churn_risk": {"type": "noul", "instructions": "Does `body` threaten to cancel or leave?"},
    },
}

_BATCH_EXAMPLE: dict[str, Any] = {
    "states": [{"body": "billed twice, refund please"}, {"body": "cannot login, getting a 500 error"}],
    "questions": {
        "dept": {
            "type": "choice",
            "instructions": "Which team handles `body`?",
            "criteria": {"billing": "refunds, invoices", "tech": "bugs, outages"},
        }
    },
}


class _QuestionBase(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ChoiceQuestion(_QuestionBase):
    """Pick one label out of N."""

    type: Literal["choice"]
    instructions: Instructions
    criteria: list[str] | dict[str, str | None] = Field(
        description="Labels, or label -> short description. At least two options.",
        examples=[{"billing": "invoices, payments, refunds", "technical": "bugs, outages"}],
    )


class ScoreQuestion(_QuestionBase):
    """Expected position on an ordered rubric (0 = first level)."""

    type: Literal["score"]
    instructions: Instructions
    criteria: list[Any] = Field(
        description="Rubric levels from lowest to highest. Every level needs a description.",
        examples=[["not urgent", "soon", "blocking"]],
    )


class NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true: Any = Field(None, description="What a yes looks like.")
    false: Any = Field(None, description="What a no looks like.")


class NoulQuestion(_QuestionBase):
    """Probability that a yes/no statement is true."""

    type: Literal["noul"]
    instructions: Instructions
    criteria: NoulCriteria | None = None


Question = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


class PredictionControlsSchema(BaseModel):
    """Optional per-request controls. Omit a field to inherit the deployment default."""

    model_config = ConfigDict(extra="ignore")

    model: str | None = Field(
        None,
        description=(
            "Pin a checkpoint: `english`, `multilingual`, `typed-decisions`, an alias, or a published "
            "Hub id. Omit it (or send a Jev id such as `jev-1`) to let the router choose."
        ),
        examples=["multilingual"],
    )
    task: str | None = Field(None, description="Force a typed-decisions workflow by name.")
    lang: str | None = Field(None, description="ISO language code that skips language detection.", examples=["es"])
    lang_guess: str | None = Field(None, description="Soft language hint that still participates in routing.")
    max_len: int | None = Field(None, gt=0, description="Token window for this request (capped by the server).")
    head_max_len: int | None = Field(None, gt=0, description="Token window shared by the option prompt.")
    min_confidence: float | dict[str, float] | None = Field(
        None,
        description=(
            'Abstention gate in [0, 1], or a per-bucket map such as `{"choice:3-5": 0.6, "default": 0.5}`. '
            "Answers below it come back with `low_confidence: true`."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _refuse_hooks(cls, data: Any) -> Any:
        if isinstance(data, dict):
            given = sorted(key for key in HOOK_FIELDS if data.get(key) is not None)
            if given:
                raise ValueError(
                    "%s run inside the server process and cannot be sent over HTTP; drop them" % ", ".join(given)
                )
        return data


def _state_required(schema: dict[str, Any]) -> None:
    """Document ``state`` as required while the model itself accepts its absence.

    Jev answers a missing or ``null`` state with 400, not with a validation error, so the model lets
    it through and the domain rule (``MissingStateError``) produces that status.
    """
    required = schema.setdefault("required", [])
    if "state" not in required:
        required.insert(0, "state")
    schema["examples"] = [_SYSTEMONE_EXAMPLE]


class SystemOneRequest(PredictionControlsSchema):
    """One state, every question answered in a single forward pass."""

    model_config = ConfigDict(extra="ignore", json_schema_extra=_state_required)

    state: JsonState | None = None
    questions: dict[str, Question] = Field(description="Question id -> question definition.")


class SystemOneBatchRequest(PredictionControlsSchema):
    """Several states answered against the same questions, sharing forward passes."""

    model_config = ConfigDict(extra="ignore", json_schema_extra={"examples": [_BATCH_EXAMPLE]})

    states: list[JsonState] = Field(min_length=1, description="States, answered in this order.")
    questions: dict[str, Question]
    batch_size: int | None = Field(None, ge=1, description="States per forward pass. Omit for one pass.")
    sort_by_length: bool = Field(
        False, description="Group similar-length states to reduce padding. Needs a `batch_size`."
    )


class PresetPredictRequest(PredictionControlsSchema):
    """A state to answer with a preset's questions."""

    state: JsonState = Field(
        description=(
            "Plain text is placed in the field the preset reads (`message`, `body`, `prompt` ...). "
            "Send an object to control the fields yourself."
        ),
        examples=["I was charged twice this month, refund me today or I'm switching providers."],
    )
