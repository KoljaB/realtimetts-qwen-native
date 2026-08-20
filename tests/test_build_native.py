from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_native.py"
SPEC = importlib.util.spec_from_file_location("qwentts_build_native", SCRIPT)
assert SPEC and SPEC.loader
build_native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_native)


def _nvcc(tmp_path: Path) -> Path:
    path = tmp_path / ("nvcc.exe" if os.name == "nt" else "nvcc")
    path.write_bytes(b"")
    return path


def test_cuda_compiler_prefers_cudacxx(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cudacxx = _nvcc(tmp_path)
    monkeypatch.setenv("CUDACXX", str(cudacxx))
    monkeypatch.setenv("CUDA_PATH", str(tmp_path / "not-used"))

    assert build_native.find_cuda_compiler() == cudacxx.resolve()


def test_cuda_compiler_uses_cuda_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("CUDACXX", "CMAKE_CUDA_COMPILER", "CUDA_HOME"):
        monkeypatch.delenv(name, raising=False)
    cuda_root = tmp_path / "cuda"
    (cuda_root / "bin").mkdir(parents=True)
    compiler = _nvcc(cuda_root / "bin")
    monkeypatch.setenv("CUDA_PATH", str(cuda_root))

    assert build_native.find_cuda_compiler() == compiler.resolve()


def test_cuda_compiler_rejects_missing_explicit_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in ("CUDACXX", "CMAKE_CUDA_COMPILER", "CUDA_PATH", "CUDA_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(build_native.shutil, "which", lambda _name: None)

    with pytest.raises(SystemExit, match="Could not find the CUDA compiler"):
        build_native.find_cuda_compiler(tmp_path / "missing-nvcc")


def test_cuda_wheel_configuration_disables_unused_nccl(tmp_path: Path) -> None:
    cuda_bin = tmp_path / "cuda" / "bin"
    cuda_bin.mkdir(parents=True)
    compiler = _nvcc(cuda_bin)

    args = build_native.cuda_cmake_arguments(compiler)

    assert "-DGGML_CUDA=ON" in args
    assert "-DGGML_CUDA_NCCL=OFF" in args
    assert f"-DCUDAToolkit_ROOT={compiler.parent.parent.as_posix()}" in args


def test_qwentts_abi_must_match(tmp_path: Path) -> None:
    source = tmp_path / "qwentts.cpp"
    header = source / "src" / "qwen.h"
    header.parent.mkdir(parents=True)
    header.write_text("#define QT_ABI_VERSION 4\n", encoding="utf-8")

    assert build_native.qwentts_abi_version(source) == 4
    build_native.verify_qwentts_abi(source, 4)
    with pytest.raises(SystemExit, match="expects ABI 3"):
        build_native.verify_qwentts_abi(source, 3)


def test_release_pin_is_full_sha_and_abi_four() -> None:
    assert build_native.PINNED_QWENTTS_REF == "7b6ed4f6db964c14fd3ac36c1ca13f1ce6150f4e"
    assert build_native.PINNED_QWENTTS_ABI == 4
    assert build_native.PORTABLE_CMAKE_ARGUMENTS == ["-DGGML_NATIVE=OFF"]


def test_skip_build_copy_does_not_destroy_package_on_failure(tmp_path: Path) -> None:
    build_dir = tmp_path / "empty-build"
    build_dir.mkdir()
    package_lib = tmp_path / "package-lib"
    package_lib.mkdir()
    existing = package_lib / "qwen.dll"
    existing.write_bytes(b"keep-me")

    with pytest.raises(SystemExit, match="No qwentts shared library"):
        build_native.copy_shared_libraries(
            build_dir,
            package_lib,
            "cuda",
            clean_package_lib=False,
        )

    assert existing.read_bytes() == b"keep-me"
