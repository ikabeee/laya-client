# Architecture

`laya-client` follows Robert C. Martin's **Clean Architecture**. The dependency rule is the one rule
that is not negotiable: **code only depends on layers further in**. The domain does not know that
FastAPI, torch or Laya exist.

```
            ┌──────────────────────────────────────────────────────────┐
            │  interfaces/http  (FastAPI, Pydantic, Scalar)            │
            │   ┌──────────────────────────────────────────────────┐   │
            │   │  application  (use cases)                        │   │
            │   │   ┌──────────────────────────────────────────┐   │   │
            │   │   │  domain  (entities, policies, ports)     │   │   │
            │   │   └──────────────────────────────────────────┘   │   │
            │   └──────────────────────────────────────────────────┘   │
            │  infrastructure  (Laya Router, runtime checks, presets)  │
            └──────────────────────────────────────────────────────────┘
                     container.py  = composition root (wires it all)
```

`tests/unit/test_architecture.py` enforces the rule: it fails if `domain` imports anything outside the
standard library, if `application` imports anything but `domain`, or if any layer other than
`infrastructure` imports `laya`, `torch` or `transformers`.

## Layers

### `domain/` — Enterprise Business Rules

| Module | Contents |
|---|---|
| `entities.py` | `Question`, `DecisionRequest`, `BatchDecisionRequest`, `Answer`, `Decision`, `Usage`, `ModelInfo`, `Preset`, `EngineStatus` (immutable dataclasses) |
| `policies.py` | `RequestLimits` and the rules every request must satisfy before inference (state size, number of questions/options, token budget, `min_confidence`) |
| `errors.py` | Business errors (`MissingStateError`, `PayloadTooLargeError`, `ModelNotFoundError`, `EngineBusyError` ...) |
| `ports.py` | Interfaces infrastructure implements: `DecisionEngine`, `ModelCatalog`, `PresetRepository` |

### `application/` — Application Business Rules

One use case per thing a caller can ask the service to do. Each receives its ports through its
constructor (dependency injection) and knows nothing about HTTP:

- `PredictDecision` / `PredictBatchDecision`: validate (policies) → resolve the model (catalog) → engine
- `PredictWithPreset`: places plain text in the field the preset reads and delegates to `PredictDecision`
- `ListModels`, `ListPresets`, `GetPreset`, `GetHealth`

### `infrastructure/` — Frameworks & Drivers

- `engines/runtime.py` — `check_runtime`: resolves `LAYA_DEVICE` and proves the runtime can use it
  (laya and torch installed, a CUDA kernel actually runs on the GPU, no silent CPU fallback when an
  NVIDIA GPU is present).
- `engines/laya_engine.py` — `LayaRouterEngine`: adapter from the `DecisionEngine` port to `laya.Router`.
  `start()` runs the runtime check, loads the checkpoints, verifies their device and runs a test
  prediction, raising `EngineUnavailableError` on any failure. Inference serializes forward passes with a
  lock (one at a time is what a CPU/GPU wants), turns Laya's `ValueError` into `InvalidRequestError` and
  hides internal failures.
- `catalog.py` — model-name resolution identical to `laya-serve` (a Jev model id auto-routes).
- `presets.py` — Laya's five presets, as data.
- `config/settings.py` — `pydantic-settings`, `LAYA_` prefix.

### `interfaces/http/` — Interface Adapters

- `schemas/` — Pydantic models: the wire contract and the source of the OpenAPI document.
- `mappers.py` — schema → entity. `presenters.py` — entity → JSON (full or strict Jev).
- `errors.py` — the **only** place a domain error becomes an HTTP status code.
- `security.py` — optional bearer auth, constant-time comparison.
- `dependencies.py` — container access and admission control (503 instead of queueing).
- `middleware.py` — request body size limit and `X-Request-ID`.
- `docs.py` — Scalar reference at `/docs` (telemetry disabled).

## Start-up

```
laya-client  →  uvicorn  →  app lifespan
  → DecisionEngine.start()            (LayaRouterEngine)
       → check_runtime(LAYA_DEVICE)   laya + torch installed, GPU kernel probe
       → Router(...).preload(models)  download + load the checkpoints
       → device check                 each checkpoint on the requested device
       → test prediction              one per checkpoint
  → any failure: CRITICAL log line, uvicorn exits with status 3 — the server never listens
```

There is no mock engine and no lazy loading: a running server is a server that can answer with the real
model. `laya-client doctor [--load]` runs the same steps without starting the server.

## Request flow

```
POST /v1/systemone
  → BodySizeLimitMiddleware / RequestIdMiddleware
  → require_api_key → admit (503 when saturated)
  → SystemOneRequest (Pydantic)  → mappers.to_decision_request
  → PredictDecision.execute      (thread pool: never blocks the event loop)
       → policies.validate_decision_request
       → ModelCatalog.resolve
       → DecisionEngine.predict  (LayaRouterEngine)
  → presenters.decision_payload  (strict when LAYA_JEV_STRICT)
  → JSONResponse + X-Inference-Time-Ms / Server-Timing
```

## Extending

- **Another engine** (e.g. ONNX, or a remote Laya): implement `DecisionEngine` (including a strict
  `start()`) in `infrastructure/engines/` and select it in `container.build_engine`. Nothing else changes.
- **Custom presets** (e.g. from a database): implement `PresetRepository` and swap it in `container.py`.
- **Another interface** (gRPC, CLI, a queue consumer): add `interfaces/<name>/` calling the same use cases.

## Tests

| Path | What it covers |
|---|---|
| `tests/unit/test_architecture.py` | The dependency rule |
| `tests/unit/test_policies.py` | Domain rules |
| `tests/unit/test_use_cases.py` | Use cases against a test double (`RecordingEngine`) |
| `tests/unit/test_runtime.py` | Device resolution and GPU checks against stand-ins for `torch` |
| `tests/unit/test_engines.py` | The Laya adapter's start-up and inference against a fake `Router` |
| `tests/unit/test_presenters.py` | Full payload vs. the strict Jev contract |
| `tests/integration/test_api.py` | The whole HTTP app with a test-double engine (`tests/fakes.py`): contract, auth, limits, 503, start-up failure, OpenAPI, Scalar |
| CI `e2e` job | The production Docker image with the real model: download, load, API calls, and a start-up failure when a missing GPU is requested |
