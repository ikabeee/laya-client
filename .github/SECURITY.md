# Security Policy

## Supported versions

Security fixes are released for the latest minor version on `main`.

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Report them privately through
[GitHub Security Advisories](https://github.com/ikabeee/laya-client/security/advisories/new) and include:

- the affected version or commit,
- a description of the issue and its impact,
- steps or a request that reproduces it (without real credentials or private data).

You will get an acknowledgement within a few days. Once a fix is available we will publish an advisory
and credit you, unless you prefer to stay anonymous.

## Deployment hardening

- Always set `LAYA_API_KEY` when the API is reachable from outside the host.
- Terminate TLS in front of the service (the Compose `proxy` profile runs Caddy with automatic HTTPS).
- Keep `LAYA_MAX_*` limits at their defaults unless you have measured your hardware.
- Set `LAYA_DOCS_ENABLED=false` if the API reference should not be public.
