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
PINNED_REF = "b47728bd6cb60331bd02afacb390e533479329b5"


def test_cpu_metadata_and_native_pin() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["name"] == "realtimetts-qwen-native-cpu"
    assert project["project"]["version"] == "0.3.0rc1"
    assert "cuda12" not in project["project"].get("optional-dependencies", {})
    assert project["tool"]["qwentts-cpp-python"] == {
        "qwentts-ref": PINNED_REF,
        "qwentts-abi": 5,
    }


def test_cpu_workflow_has_four_platforms_exact_pin_and_no_publication() -> None:
    text = (ROOT / ".github/workflows/cpu-wheels.yml").read_text(encoding="utf-8")
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert set(re.findall(r"ref:\s+([0-9a-f]{40})", text)) == {PINNED_REF}
    for runner in ("ubuntu-22.04", "windows-2022", "macos-15-intel", "macos-15"):
        assert f"runner: {runner}\n" in text
    for repair in ("delvewheel repair", "auditwheel repair --only-plat", "delocate-wheel"):
        assert repair in text
    assert "manylinux_2_35_x86_64" in text
    assert "ci_cpu_synthesis.py" in text
    assert "python -m venv .ci-venv" in text
    assert "twine upload" not in text
    assert "--backend cpu" in text
    assert "CMAKE_CUDA_ARCHITECTURES" not in text
    assert "architecture: ${{ matrix.python_arch }}" in text
    assert "macos_arch: arm64" in text
    assert "wheel_platform: macosx_11_0_arm64" in text
    assert "wheel_platform: macosx_13_0_x86_64" in text
    assert "-DCMAKE_OSX_ARCHITECTURES=$MACOS_ARCH" in text
    assert 'delocate-wheel --require-archs "$MACOS_ARCH"' in text
    assert "uname -m" not in text
    assert "include scripts/build_native.py" in manifest


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

    distribution = captured["distclass"]({"name": "realtimetts-qwen-native-cpu", "version": "0.2.0"})
    command = captured["cmdclass"]["bdist_wheel"](distribution)
    command.ensure_finalized()
    python_tag, abi_tag, platform_tag = command.get_tag()

    assert (python_tag, abi_tag) == ("py3", "none")
    assert platform_tag not in {"any", ""}


def test_setup_uses_explicit_macos_wheel_platform(monkeypatch) -> None:
    monkeypatch.setenv("QWENTTS_CPP_WHEEL_PLATFORM", "macosx_11_0_arm64")
    captured: dict[str, object] = {}

    def capture_setup(**kwargs) -> None:
        captured.update(kwargs)

    spec = importlib.util.spec_from_file_location("qwentts_setup_test", ROOT / "setup.py")
    assert spec and spec.loader
    with patch.object(setuptools, "setup", capture_setup):
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    distribution = captured["distclass"]({"name": "realtimetts-qwen-native-cpu", "version": "0.2.0"})
    command = captured["cmdclass"]["bdist_wheel"](distribution)
    command.ensure_finalized()

    assert command.get_tag() == ("py3", "none", "macosx_11_0_arm64")
    assert module.PlatformWheel is command.__class__


def test_setup_accepts_windows_native_payload(monkeypatch, tmp_path: Path) -> None:
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "qwen.dll").write_bytes(b"native")
    (lib_dir / "ggml-cpu.dll").write_bytes(b"native")

    captured: dict[str, object] = {}

    def capture_setup(**kwargs) -> None:
        captured.update(kwargs)

    spec = importlib.util.spec_from_file_location("qwentts_setup_test", ROOT / "setup.py")
    assert spec and spec.loader
    with patch.object(setuptools, "setup", capture_setup):
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_native_payload_dir", lambda: lib_dir)

    module._validate_native_payload()
