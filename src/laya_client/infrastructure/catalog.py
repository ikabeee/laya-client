"""The checkpoints Laya ships and the names a caller may use for them."""

from __future__ import annotations

from ..domain.entities import ModelInfo
from ..domain.errors import ModelNotFoundError
from ..domain.ports import ModelCatalog

BUNDLE_REPO = "convaiinnovations/laya"

_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        name="english",
        repo="convaiinnovations/laya",
        description="ModernBERT-large checkpoint for English states. Default for Latin-script English text.",
        aliases=("en", "laya", "default"),
    ),
    ModelInfo(
        name="multilingual",
        repo="convaiinnovations/laya-multilingual",
        description="mmBERT-base checkpoint for 100+ languages; reads up to 8,192 tokens with max_len.",
        aliases=("multi", "ml", "laya-multilingual"),
    ),
    ModelInfo(
        name="typed-decisions",
        repo="convaiinnovations/laya-typed-decisions",
        description="Fine-tuned for customer service, invoices, security incidents and agent traces.",
        aliases=("typed", "typed_decisions", "laya-typed-decisions", "decisions"),
    ),
)


class StaticModelCatalog(ModelCatalog):
    """Resolves model names the way ``laya-serve`` does, so a Jev client keeps working unchanged.

    * nothing, ``jev-1`` or any other unknown plain name   -> ``None`` (the router chooses)
    * the bundle id ``convaiinnovations/laya``             -> ``None`` (its documented meaning)
    * a checkpoint name, alias or published Hub id         -> that checkpoint
    * a path or an unpublished Hub id                      -> ``ModelNotFoundError``
    """

    def __init__(self, models: tuple[ModelInfo, ...] = _MODELS) -> None:
        self._models = models
        self._by_name: dict[str, str] = {}
        for model in models:
            self._by_name[model.name] = model.name
            for alias in model.aliases:
                self._by_name[alias] = model.name
            if model.repo != BUNDLE_REPO:
                self._by_name[model.repo.lower()] = model.name

    def list(self) -> list[ModelInfo]:
        return list(self._models)

    def resolve(self, name: str | None) -> str | None:
        if name is None or not str(name).strip():
            return None
        key = str(name).strip().lower()
        if key == BUNDLE_REPO:
            return None
        if key in self._by_name:
            return self._by_name[key]
        if key[0] in ".~" or "/" in key or "\\" in key:
            known = ", ".join(m.name for m in self._models)
            raise ModelNotFoundError(
                "unknown model %r: pick one of %s, or omit model to let the router choose" % (name, known)
            )
        return None
