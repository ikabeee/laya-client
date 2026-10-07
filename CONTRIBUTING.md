# Contributing

## Issues

Open an issue with one of the [templates](.github/ISSUE_TEMPLATE): **bug report**, **feature request**,
**documentation** or **question**. Report security vulnerabilities privately as described in the
[security policy](.github/SECURITY.md), never in a public issue.

## Git flow

| Branch | Purpose | Branches from | Merges into |
|---|---|---|---|
| `main` | What runs in production. Every merge is tagged `vX.Y.Z` | — | — |
| `develop` | Integration branch for the next release | `main` | `release/*` |
| `feature/<name>` | One feature | `develop` | `develop` (via PR) |
| `release/<X.Y.Z>` | Stabilization: version bump, changelog, small fixes | `develop` | `main` + `develop` |
| `hotfix/<X.Y.Z>` | Urgent production fix | `main` | `main` + `develop` |

```bash
git checkout develop && git pull
git checkout -b feature/my-change
# ... commits ...
git push -u origin feature/my-change   # open a PR against develop
```

To release: branch `release/X.Y.Z` from `develop` → bump `version` in `pyproject.toml` and
`src/laya_client/__init__.py` → PR into `main` → tag `vX.Y.Z` → merge back into `develop`.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/), with the layer as the scope:

```
feat(api): add /v1/presets endpoints
fix(domain): reject null score levels
refactor(infrastructure): lazy-load torch in the Laya engine
test(api): cover the admission limit
docs: deployment guide for a VPS
chore(deploy): add Caddy reverse proxy
```

## Before opening a PR

```bash
make check      # ruff (lint + format) and pytest
```

- Respect the dependency rule (see `docs/ARCHITECTURE.md`): `domain` imports nothing external,
  `application` imports only `domain`, and only `infrastructure` imports `laya`/torch.
  `tests/unit/test_architecture.py` enforces this.
- Every new rule comes with a test; every new endpoint, with an integration test.
- If you change the HTTP contract, check that `/docs` (Scalar) still describes it accurately.
- Write code, comments, docs, commits and PRs in English.
