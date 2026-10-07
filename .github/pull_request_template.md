## Summary

<!-- What does this PR change, and why? Link the issue it closes. -->

Closes #

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change (HTTP contract or configuration)
- [ ] Refactor / internal
- [ ] Documentation
- [ ] Deployment / CI

## Checklist

- [ ] The PR targets `develop` (or `main` for a `hotfix/*` / `release/*` branch)
- [ ] `make check` passes (ruff lint + format, pytest)
- [ ] New behavior is covered by tests
- [ ] The dependency rule holds (`domain` ← `application` ← `infrastructure` / `interfaces`)
- [ ] HTTP contract changes are reflected in the schemas, so `/docs` (Scalar) stays accurate
- [ ] New settings are documented in `.env.example` and the README
- [ ] Commits follow Conventional Commits, in English

## How was this tested?

<!-- Commands you ran, requests you sent, device used (CPU, CUDA GPU, MPS). -->
