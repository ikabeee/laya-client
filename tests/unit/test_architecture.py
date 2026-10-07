"""Enforce the dependency rule: inner layers never import outer layers or frameworks."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "laya_client"
STDLIB = set(sys.stdlib_module_names) | {"__future__"}

# layer -> laya_client sub-packages it may import
ALLOWED_INTERNAL = {
    "domain": {"domain"},
    "application": {"domain", "application"},
}


def _imports(path: Path) -> list[tuple[str, int]]:
    """(absolute module name, relative level) for every import in ``path``."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.extend((alias.name, 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.module or "", node.level))
    return found


def _resolve(path: Path, module: str, level: int) -> str:
    if level == 0:
        return module
    parts = list(path.relative_to(PACKAGE.parent).with_suffix("").parts)[:-level]
    return ".".join(parts + ([module] if module else []))


@pytest.mark.parametrize("layer", sorted(ALLOWED_INTERNAL))
def test_layer_only_depends_inward(layer):
    violations = []
    for path in (PACKAGE / layer).rglob("*.py"):
        for module, level in _imports(path):
            name = _resolve(path, module, level)
            top = name.split(".")[0]
            if top == "laya_client":
                inner = name.split(".")[1] if "." in name else ""
                if inner not in ALLOWED_INTERNAL[layer]:
                    violations.append("%s imports %s" % (path.name, name))
            elif top not in STDLIB:
                violations.append("%s imports third-party %s" % (path.name, name))
    assert not violations, violations


def test_only_infrastructure_touches_laya():
    for path in PACKAGE.rglob("*.py"):
        if "infrastructure" in path.parts:
            continue
        for module, level in _imports(path):
            if level == 0:
                assert module.split(".")[0] not in {"laya", "torch", "transformers"}, path
