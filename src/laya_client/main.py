"""Command line entry point.

    laya-client                 start the server (same as `laya-client serve`)
    laya-client doctor          check the runtime: laya, torch, the GPU and the device it would use
    laya-client doctor --load   also download, load and test-run the checkpoints (what start-up does)

Every failure exits with a non-zero status and a message that says what to fix.
"""

from __future__ import annotations

import argparse
import logging
import sys

from pydantic import ValidationError

from .domain.errors import LayaClientError
from .infrastructure.config import Settings, get_settings

EXIT_CONFIG = 2
EXIT_RUNTIME = 1


def _load_settings() -> Settings:
    try:
        return get_settings()
    except ValidationError as error:
        lines = ["invalid configuration:"]
        for item in error.errors():
            field = ".".join(str(part) for part in item["loc"]) or "settings"
            lines.append("  LAYA_%s: %s" % (field.upper(), item["msg"]))
        print("\n".join(lines), file=sys.stderr)
        raise SystemExit(EXIT_CONFIG) from None


def _configure_logging(settings: Settings) -> None:
    level = "DEBUG" if settings.log_level == "trace" else settings.log_level.upper()
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def serve() -> None:
    import uvicorn

    settings = _load_settings()
    _configure_logging(settings)
    # One worker on purpose: each worker process would hold its own copy of the model weights.
    # When the engine cannot start, the app's lifespan fails and uvicorn exits with status 3.
    uvicorn.run(
        "laya_client.interfaces.http.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        proxy_headers=True,
        workers=1,
    )


def doctor(load: bool) -> int:
    from .container import build_engine
    from .infrastructure.engines import check_runtime

    settings = _load_settings()
    _configure_logging(settings)
    try:
        runtime = check_runtime(settings.device, require_gpu=settings.require_gpu)
    except LayaClientError as error:
        print("FAIL runtime: %s" % error.message, file=sys.stderr)
        return EXIT_RUNTIME
    print("ok   laya %s" % runtime.laya_version)
    print("ok   torch %s (CUDA build: %s)" % (runtime.torch_version, runtime.torch_cuda or "none"))
    print("ok   device %s%s" % (runtime.device, " - %s" % runtime.accelerator if runtime.accelerator else ""))
    for note in runtime.notes:
        print("note %s" % note)
    if not load:
        print("Runtime looks good. Run `laya-client doctor --load` to also load and test the checkpoints.")
        return 0

    engine = build_engine(settings)
    try:
        engine.start()
    except LayaClientError as error:
        print("FAIL model: %s" % error.message, file=sys.stderr)
        return EXIT_RUNTIME
    status = engine.status()
    engine.shutdown()
    print("ok   checkpoints loaded and answering on %s: %s" % (status.device, ", ".join(status.loaded)))
    print("Everything works. Start the server with `laya-client` (or `make start`).")
    return 0


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="laya-client", description="Self-hosted Laya System-1 decision API.")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("serve", help="start the HTTP server (default)")
    check = commands.add_parser("doctor", help="check that the runtime and the model work")
    check.add_argument("--load", action="store_true", help="also load and test-run the checkpoints")
    args = parser.parse_args(argv)

    if args.command == "doctor":
        raise SystemExit(doctor(load=args.load))
    serve()


if __name__ == "__main__":
    run()
