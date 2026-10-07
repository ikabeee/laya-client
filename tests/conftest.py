from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from laya_client.container import build_container
from laya_client.domain.entities import Question, QuestionType
from laya_client.infrastructure.config import Settings
from laya_client.interfaces.http import create_app
from tests.fakes import FakeDecisionEngine

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle `body`?",
        "criteria": {"billing": "invoices, refunds", "technical": "bugs", "other": "everything else"},
    },
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
    "churn_risk": {"type": "noul", "instructions": "Does `body` threaten to cancel?"},
}


def make_settings(**overrides) -> Settings:
    # _env_file=None: tests must not pick up a developer's local .env.
    return Settings(_env_file=None, **overrides)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings, build_container(settings, engine=FakeDecisionEngine()))) as test_client:
        yield test_client


@pytest.fixture
def make_client():
    clients = []

    def _make(engine=None, **overrides) -> TestClient:
        settings = make_settings(**overrides)
        app = create_app(settings, build_container(settings, engine=engine or FakeDecisionEngine()))
        test_client = TestClient(app)
        test_client.__enter__()
        clients.append(test_client)
        return test_client

    yield _make
    for test_client in clients:
        test_client.__exit__(None, None, None)


@pytest.fixture
def questions() -> tuple[Question, ...]:
    return (
        Question("department", QuestionType.CHOICE, "Which team?", {"billing": "refunds", "tech": "bugs"}),
        Question("urgency", QuestionType.SCORE, "How urgent?", ["low", "medium", "high"]),
        Question("churn", QuestionType.NOUL, "Will they cancel?"),
    )
