from __future__ import annotations

import pytest

from laya_client.domain.errors import ModelNotFoundError, PresetNotFoundError
from laya_client.infrastructure.catalog import StaticModelCatalog
from laya_client.infrastructure.presets import InMemoryPresetRepository


@pytest.mark.parametrize(
    "name, expected",
    [
        (None, None),
        ("", None),
        ("jev-1", None),  # a Jev model id means "let the router choose"
        ("jev-1.13.0", None),
        ("convaiinnovations/laya", None),  # the bundle id's documented meaning
        ("english", "english"),
        ("ML", "multilingual"),
        ("typed", "typed-decisions"),
        ("convaiinnovations/laya-multilingual", "multilingual"),
    ],
)
def test_resolve(name, expected):
    assert StaticModelCatalog().resolve(name) == expected


@pytest.mark.parametrize("name", ["/models/ckpt", "./ckpt", "~/ckpt", "someone/private-model", "C:\\ckpt"])
def test_resolve_refuses_paths_and_unpublished_ids(name):
    with pytest.raises(ModelNotFoundError):
        StaticModelCatalog().resolve(name)


def test_presets_are_listed_and_retrievable():
    repo = InMemoryPresetRepository()
    names = [p.name for p in repo.list()]
    assert names == ["triage", "email", "guard", "moderation", "router"]
    triage = repo.get("Triage")
    assert triage.state_field == "message"
    assert {q.id for q in triage.questions} >= {"intent", "churn_risk"}


def test_unknown_preset():
    with pytest.raises(PresetNotFoundError):
        InMemoryPresetRepository().get("nope")
