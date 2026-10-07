"""Runtime configuration, read from environment variables (prefix ``LAYA_``) or a ``.env`` file.

Variable names follow ``laya-serve`` wherever the two overlap, so a deployment can switch between
them without renaming its configuration.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ...domain.policies import RequestLimits


def _split_csv(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LAYA_",
        env_file=".env",
        env_file_encoding="utf-8",
        # `LAYA_THREADS=` in a .env means "not set", not "the empty string".
        env_ignore_empty=True,
        extra="ignore",
    )

    # --- server -------------------------------------------------------------------------------
    host: str = Field("0.0.0.0", description="Bind address.")
    port: int = Field(8000, ge=1, le=65535, description="Bind port.")
    log_level: Literal["critical", "error", "warning", "info", "debug", "trace"] = "info"
    root_path: str = Field("", description="Public URL prefix when served behind a reverse proxy.")
    cors_origins: str = Field("", description="Comma-separated origins allowed by CORS; empty disables CORS.")
    docs_enabled: bool = Field(True, description="Serve the Scalar reference at /docs and /openapi.json.")

    # --- security ------------------------------------------------------------------------------
    api_key: SecretStr | None = Field(
        None,
        description="Comma-separated bearer tokens. When set, clients must send `Authorization: Bearer <key>`.",
    )

    # --- engine --------------------------------------------------------------------------------
    engine: Literal["laya", "mock"] = Field(
        "laya", description="`laya` runs the real model; `mock` answers deterministically without one."
    )
    device: str | None = Field(None, description="Torch device (cpu, cuda, mps). Empty lets Laya choose.")
    preload: bool = Field(True, description="Load checkpoints at startup instead of on the first request.")
    models: str = Field("", description="Comma-separated checkpoints to preload; empty preloads all of them.")
    threads: int | None = Field(None, ge=1, description="Cap torch intra-op threads for CPU inference.")
    auto_task: bool = Field(False, description="Auto-route typed-decisions workflows to their checkpoint.")
    default_model: str | None = Field(None, description="Checkpoint for states with no language evidence.")
    max_loaded: int | None = Field(None, ge=1, description="Checkpoints kept resident at once.")
    jev_strict: bool = Field(False, description="Reply with the strict Jev contract only (no Laya extras).")

    # --- limits --------------------------------------------------------------------------------
    max_concurrent: int = Field(16, ge=1, description="Inference requests in flight; excess gets 503.")
    max_body_bytes: int = Field(2 * 1024 * 1024, ge=1024, description="Largest request body accepted.")
    max_questions: int = Field(64, ge=1)
    max_state_chars: int = Field(50_000, ge=1)
    max_batch_states: int = Field(64, ge=1)
    max_choice_options: int = Field(100, ge=2)
    max_score_levels: int = Field(32, ge=2)
    max_total_options: int = Field(512, ge=2)
    max_token_budget: int = Field(8192, ge=1)
    max_batch_tokens: int = Field(131_072, ge=1, description="Tokens one batch forward pass may collate.")

    @field_validator("device", "default_model", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value.strip() if isinstance(value, str) else value

    @property
    def api_keys(self) -> list[str]:
        return _split_csv(self.api_key.get_secret_value() if self.api_key else None)

    @property
    def preload_models(self) -> list[str]:
        return _split_csv(self.models)

    @property
    def cors_origin_list(self) -> list[str]:
        return _split_csv(self.cors_origins)

    @property
    def limits(self) -> RequestLimits:
        return RequestLimits(
            max_questions=self.max_questions,
            max_state_chars=self.max_state_chars,
            max_batch_states=self.max_batch_states,
            max_choice_options=self.max_choice_options,
            max_score_levels=self.max_score_levels,
            max_total_options=self.max_total_options,
            max_token_budget=self.max_token_budget,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
