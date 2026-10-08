"""``providers._declared_python_dependencies``: manifest-based optional-dependency pre-skip.

The loader must be able to skip a model-provider plugin whose optional distribution is
absent *before* importing it, so a minimal runtime (PM runtime venv) does not execute
plugin code that can only fail. The declaration is read from ``plugin.yaml`` and must
survive an unavailable YAML parser, because the skip decision itself must never blow up.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import providers
import utils


def _write(directory: Path, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "plugin.yaml").write_text(text, encoding="utf-8")
    return directory


@pytest.mark.parametrize(
    "manifest,expected",
    [
        ("name: p\npython_dependencies:\n  - httpx\n  - 'ruamel.yaml'\nversion: 1\n", ["httpx", "ruamel.yaml"]),
        ("name: p\npython_dependencies: [httpx, json]\n", ["httpx", "json"]),
        ('name: p\npython_dependencies: "httpx"\n', ["httpx"]),
        ("name: p\npython_dependencies:\n  - httpx\n# comment\nversion: 1\n", ["httpx"]),
        ("name: p\nversion: 1\n", None),
    ],
)
def test_reads_declared_dependencies(tmp_path, manifest, expected):
    assert providers._declared_python_dependencies(_write(tmp_path / "p", manifest)) == expected


def test_missing_manifest_is_not_an_error(tmp_path):
    assert providers._declared_python_dependencies(tmp_path / "absent") is None


@pytest.mark.parametrize(
    "manifest,expected",
    [
        ("name: p\npython_dependencies:\n  - httpx\n  - 'ruamel.yaml'\nversion: 1\n", ["httpx", "ruamel.yaml"]),
        ("name: p\npython_dependencies: [httpx, json]\n", ["httpx", "json"]),
        ("name: p\nversion: 1\n", None),
    ],
)
def test_falls_back_to_manual_parsing_when_yaml_is_unavailable(tmp_path, monkeypatch, manifest, expected):
    def boom(*args, **kwargs):
        raise RuntimeError("yaml unavailable")

    monkeypatch.setattr(utils, "fast_safe_load", boom)
    assert providers._declared_python_dependencies(_write(tmp_path / "p", manifest)) == expected


def test_bundled_solstice_declares_its_optional_dependency():
    """solstice is the plugin whose noisy failure motivated the declaration path."""
    solstice = Path(providers.__file__).resolve().parent.parent / "plugins" / "model-providers" / "solstice"
    assert "httpx" in (providers._declared_python_dependencies(solstice) or [])


def test_unparsable_manifest_does_not_raise(tmp_path):
    assert providers._declared_python_dependencies(_write(tmp_path / "p", "python_dependencies: [\n  - broken\n")) is None
