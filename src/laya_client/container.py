"""Composition root: the one place that knows every concrete class and wires them together.

Use cases receive their ports through their constructors. Swapping the engine (real Laya, the mock,
a test double) or the preset store is a change here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

from .application.use_cases import (
    GetHealth,
    GetPreset,
    ListModels,
    ListPresets,
    PredictBatchDecision,
    PredictDecision,
    PredictWithPreset,
)
from .domain.ports import DecisionEngine, ModelCatalog, PresetRepository
from .infrastructure.catalog import StaticModelCatalog
from .infrastructure.config import Settings
from .infrastructure.engines import LayaEngineConfig, LayaRouterEngine, MockDecisionEngine
from .infrastructure.presets import InMemoryPresetRepository


@dataclass(frozen=True)
class Container:
    settings: Settings
    engine: DecisionEngine
    catalog: ModelCatalog
    presets: PresetRepository
    predict_decision: PredictDecision
    predict_batch: PredictBatchDecision
    predict_with_preset: PredictWithPreset
    list_models: ListModels
    list_presets: ListPresets
    get_preset: GetPreset
    get_health: GetHealth


def build_engine(settings: Settings) -> DecisionEngine:
    if settings.engine == "mock":
        return MockDecisionEngine()
    return LayaRouterEngine(
        LayaEngineConfig(
            device=settings.device,
            preload=settings.preload,
            models=tuple(settings.preload_models),
            threads=settings.threads,
            auto_task=settings.auto_task,
            default_model=settings.default_model,
            max_loaded=settings.max_loaded,
            max_batch_tokens=settings.max_batch_tokens,
        )
    )


def build_container(settings: Settings, engine: DecisionEngine | None = None) -> Container:
    engine = engine or build_engine(settings)
    catalog = StaticModelCatalog()
    presets = InMemoryPresetRepository()
    limits = settings.limits
    predict_decision = PredictDecision(engine, catalog, limits)
    return Container(
        settings=settings,
        engine=engine,
        catalog=catalog,
        presets=presets,
        predict_decision=predict_decision,
        predict_batch=PredictBatchDecision(engine, catalog, limits),
        predict_with_preset=PredictWithPreset(presets, predict_decision),
        list_models=ListModels(catalog, engine),
        list_presets=ListPresets(presets),
        get_preset=GetPreset(presets),
        get_health=GetHealth(engine),
    )
