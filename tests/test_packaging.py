from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from unittest.mock import patch

import setuptools

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
PINNED_REF = "82b83b5898b702f3d861805d9fa61cbe1ba90b2f"


def test_cpu_metadata_and_native_pin() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["name"] == "realtimetts-qwen-native"
    assert project["project"]["version"] == "0.2.0+cpu2"
    assert "cuda12" not in project["project"].get("optional-dependencies", {})
    assert project["tool"]["qwentts-cpp-python"] == {
        "qwentts-ref": PINNED_REF,
        "qwentts-abi": 5,
    }


def test_all_workflows_use_release_pin_and_windows_builds() -> None:
    workflows = {
        path.name: path.read_text(encoding="utf-8")
        for path in (ROOT / ".github" / "workflows").glob("*.yml")
    }

    for name in ("publish.yml",):
        text = workflows[name]
        refs = set(
            re.findall(r"(?:default:\s+|QWENTTS_REF:\s+)([0-9a-f]{40})", text)
        )
        assert refs == {PINNED_REF}, f"unexpected qwentts.cpp pin in {name}: {refs}"
        assert "runs-on: windows-2022" in text
        assert "delvewheel repair" in text
        assert "auditwheel repair" in text

    assert "manylinux_2_35_x86_64" in workflows["publish.yml"]
    linux_release_architectures = "75-virtual;86-real;90-real;120-real;120-virtual"
    windows_release_architectures = "75-real;75-virtual"
    assert workflows["publish.yml"].count(linux_release_architectures) == 2
    assert workflows["publish.yml"].count(windows_release_architectures) == 1
    assert "limit=104857600" in workflows["publish.yml"]


def test_setup_emits_one_python_abi_independent_platform_wheel(
    monkeypatch,
) -> None:
    setup_path = ROOT / "setup.py"
    captured: dict[str, object] = {}

    def capture_setup(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.delenv("QWENTTS_CPP_WHEEL_BUILD_TAG", raising=False)
    spec = importlib.util.spec_from_file_location("qwentts_setup_test", setup_path)
    assert spec and spec.loader
    with patch.object(setuptools, "setup", capture_setup):
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    distribution = captured["distclass"]({"name": "realtimetts-qwen-native", "version": "0.2.0"})
    command = captured["cmdclass"]["bdist_wheel"](distribution)
    command.ensure_finalized()
    python_tag, abi_tag, platform_tag = command.get_tag()

    assert (python_tag, abi_tag) == ("py3", "none")
    assert platform_tag not in {"any", ""}
