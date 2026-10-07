# Contribuir

## Git flow

| Rama | Propósito | Sale de | Se fusiona en |
|---|---|---|---|
| `main` | Lo que está en producción. Cada merge lleva un tag `vX.Y.Z` | — | — |
| `develop` | Integración de la próxima versión | `main` | `release/*` |
| `feature/<nombre>` | Una funcionalidad | `develop` | `develop` (PR) |
| `release/<X.Y.Z>` | Estabilización: versión, changelog, fixes menores | `develop` | `main` + `develop` |
| `hotfix/<X.Y.Z>` | Corrección urgente de producción | `main` | `main` + `develop` |

```bash
git checkout develop && git pull
git checkout -b feature/mi-cambio
# ... commits ...
git push -u origin feature/mi-cambio   # abrir PR contra develop
```

Para publicar: `release/X.Y.Z` desde `develop` → subir `version` en `pyproject.toml` y
`src/laya_client/__init__.py` → PR a `main` → tag `vX.Y.Z` → merge de vuelta a `develop`.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/) con la capa como scope:

```
feat(api): add /v1/presets endpoints
fix(domain): reject null score levels
refactor(infrastructure): lazy-load torch in the Laya engine
test(api): cover the admission limit
docs: deployment guide for VPS
chore(deploy): add Caddy reverse proxy
```

## Antes de abrir un PR

```bash
make check      # ruff (lint + formato) y pytest
```

- Respeta la regla de dependencias (ver `docs/ARCHITECTURE.md`): `domain` no importa nada externo,
  `application` sólo importa `domain`, y sólo `infrastructure` importa `laya`/torch.
- Toda regla nueva lleva su test; todo endpoint nuevo, su test de integración.
- Si cambias el contrato HTTP, revisa que `/docs` (Scalar) lo describa bien.
