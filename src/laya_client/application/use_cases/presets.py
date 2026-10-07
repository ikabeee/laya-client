"""Ready-made question sets, and answering a state with one of them."""

from __future__ import annotations

from ...domain.entities import Decision, DecisionRequest, JsonValue, PredictionControls, Preset
from ...domain.ports import PresetRepository
from .decisions import PredictDecision


class ListPresets:
    def __init__(self, presets: PresetRepository) -> None:
        self._presets = presets

    def execute(self) -> list[Preset]:
        return self._presets.list()


class GetPreset:
    def __init__(self, presets: PresetRepository) -> None:
        self._presets = presets

    def execute(self, name: str) -> Preset:
        return self._presets.get(name)


class PredictWithPreset:
    """Answer a preset's questions about ``state``.

    Every preset reads one named field of the state (``message``, ``body``, ``prompt`` ...). A plain
    string state is wrapped into that field, so ``"refund me"`` becomes ``{"message": "refund me"}``
    for the triage preset and the model reads the text where the questions say it is.
    """

    def __init__(self, presets: PresetRepository, predict: PredictDecision) -> None:
        self._presets = presets
        self._predict = predict

    def execute(self, name: str, state: JsonValue, controls: PredictionControls) -> Decision:
        preset = self._presets.get(name)
        if isinstance(state, str) and preset.state_field:
            state = {preset.state_field: state}
        return self._predict.execute(DecisionRequest(state=state, questions=preset.questions, controls=controls))
