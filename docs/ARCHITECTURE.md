# Arquitectura

`laya-client` sigue la **Clean Architecture** de Robert C. Martin. La regla de dependencias es la única
que no se negocia: **el código sólo depende de capas más internas**. El dominio no sabe que existe
FastAPI, ni torch, ni Laya.

```
            ┌──────────────────────────────────────────────────────────┐
            │  interfaces/http  (FastAPI, Pydantic, Scalar)            │
            │   ┌──────────────────────────────────────────────────┐   │
            │   │  application  (casos de uso)                     │   │
            │   │   ┌──────────────────────────────────────────┐   │   │
            │   │   │  domain  (entidades, políticas, puertos) │   │   │
            │   │   └──────────────────────────────────────────┘   │   │
            │   └──────────────────────────────────────────────────┘   │
            │  infrastructure  (Laya Router, mock, settings, presets)  │
            └──────────────────────────────────────────────────────────┘
                     container.py  = composition root (cablea todo)
```

## Capas

### `domain/` — Enterprise Business Rules

| Módulo | Contenido |
|---|---|
| `entities.py` | `Question`, `DecisionRequest`, `BatchDecisionRequest`, `Answer`, `Decision`, `Usage`, `ModelInfo`, `Preset`, `EngineStatus` (dataclasses inmutables) |
| `policies.py` | `RequestLimits` y las reglas que toda petición cumple antes de inferir (tamaño del state, nº de preguntas/opciones, presupuesto de tokens, `min_confidence`) |
| `errors.py` | Errores de negocio (`MissingStateError`, `PayloadTooLargeError`, `ModelNotFoundError`, `EngineBusyError`…) |
| `ports.py` | Interfaces que implementa infraestructura: `DecisionEngine`, `ModelCatalog`, `PresetRepository` |

### `application/` — Application Business Rules

Un caso de uso por cosa que se le puede pedir al servicio. Reciben sus puertos por constructor
(inyección de dependencias) y no conocen HTTP:

- `PredictDecision` / `PredictBatchDecision`: valida (políticas) → resuelve el modelo (catálogo) → motor
- `PredictWithPreset`: coloca el texto en el campo que lee el preset y delega en `PredictDecision`
- `ListModels`, `ListPresets`, `GetPreset`, `GetHealth`

### `infrastructure/` — Frameworks & Drivers

- `engines/laya_engine.py` — `LayaRouterEngine`: adaptador del puerto `DecisionEngine` a `laya.Router`.
  Importa `laya`/torch de forma perezosa, serializa las pasadas con un lock (una a la vez es lo que una
  CPU/GPU quiere), traduce `ValueError` de Laya a `InvalidRequestError` y oculta los fallos internos.
- `engines/mock_engine.py` — `MockDecisionEngine`: respuestas deterministas con la forma exacta de Laya.
- `catalog.py` — resolución de nombres de modelo idéntica a `laya-serve` (un id de Jev auto-enruta).
- `presets.py` — los cinco presets de Laya como datos.
- `config/settings.py` — `pydantic-settings`, prefijo `LAYA_`.

### `interfaces/http/` — Interface Adapters

- `schemas/` — modelos Pydantic: el contrato wire y la fuente del documento OpenAPI.
- `mappers.py` — schema → entidad. `presenters.py` — entidad → JSON (completo o Jev estricto).
- `errors.py` — **único** lugar donde un error de dominio se convierte en código HTTP.
- `security.py` — Bearer opcional, comparación en tiempo constante.
- `dependencies.py` — acceso al contenedor y control de admisión (503 sin encolar).
- `middleware.py` — límite de tamaño del cuerpo y `X-Request-ID`.
- `docs.py` — referencia Scalar en `/docs` (telemetría desactivada).

## Flujo de una petición

```
POST /v1/systemone
  → BodySizeLimitMiddleware / RequestIdMiddleware
  → require_api_key → admit (503 si hay saturación)
  → SystemOneRequest (Pydantic)  → mappers.to_decision_request
  → PredictDecision.execute      (thread pool: no bloquea el event loop)
       → policies.validate_decision_request
       → ModelCatalog.resolve
       → DecisionEngine.predict  (LayaRouterEngine | MockDecisionEngine)
  → presenters.decision_payload  (strict si LAYA_JEV_STRICT)
  → JSONResponse + X-Inference-Time-Ms / Server-Timing
```

## Extender

- **Otro motor** (p. ej. ONNX o un Laya remoto): implementa `DecisionEngine` en `infrastructure/engines/`
  y elígelo en `container.build_engine`. Nada más cambia.
- **Presets propios** (p. ej. en base de datos): implementa `PresetRepository` y cámbialo en `container.py`.
- **Otra interfaz** (gRPC, CLI, cola): crea `interfaces/<nombre>/` que llame a los mismos casos de uso.

## Tests

| Carpeta | Qué prueba |
|---|---|
| `tests/unit/test_policies.py` | Reglas de dominio |
| `tests/unit/test_use_cases.py` | Casos de uso con un motor doble (`RecordingEngine`) |
| `tests/unit/test_engines.py` | Mock y adaptador de Laya contra un `Router` falso |
| `tests/unit/test_presenters.py` | Payload completo vs. contrato Jev estricto |
| `tests/integration/test_api.py` | HTTP de punta a punta: contrato, auth, límites, 503, OpenAPI, Scalar |
