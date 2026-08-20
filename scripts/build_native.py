#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import shlex
import subprocess
import sys
from pathlib import Path


PINNED_QWENTTS_REF = "7b6ed4f6db964c14fd3ac36c1ca13f1ce6150f4e"
PINNED_QWENTTS_ABI = 4
PORTABLE_CMAKE_ARGUMENTS = [
    # Binary wheels must run on CPUs other than the build host. ggml otherwise
    # defaults to -march=native, which can produce illegal instructions on an
    # older x86_64 machine even when inference primarily uses CUDA.
    "-DGGML_NATIVE=OFF",
]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def split_env_args(value: str | None) -> list[str]:
    return shlex.split(value or "")


def _resolve_executable(value: str | os.PathLike[str] | None) -> Path | None:
    if not value:
        return None
    expanded = os.path.expandvars(os.path.expanduser(os.fspath(value)))
    candidate = Path(expanded)
    if candidate.is_file():
        return candidate.resolve()
    found = shutil.which(expanded)
    return Path(found).resolve() if found else None


def find_cuda_compiler(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve nvcc without assuming a Unix CUDA installation layout."""

    named_candidates = [
        ("--cuda-compiler", explicit),
        ("CUDACXX", os.environ.get("CUDACXX")),
        ("CMAKE_CUDA_COMPILER", os.environ.get("CMAKE_CUDA_COMPILER")),
    ]
    for _source, value in named_candidates:
        compiler = _resolve_executable(value)
        if compiler:
            return compiler

    nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"
    for env_name in ("CUDA_PATH", "CUDA_HOME"):
        cuda_root = os.environ.get(env_name)
        if not cuda_root:
            continue
        compiler = _resolve_executable(Path(cuda_root) / "bin" / nvcc_name)
        if compiler:
            return compiler

    compiler = _resolve_executable(nvcc_name)
    if compiler:
        return compiler

    requested = next((f"{name}={value!r}" for name, value in named_candidates if value), None)
    detail = f" The requested value ({requested}) was not executable." if requested else ""
    raise SystemExit(
        "Could not find the CUDA compiler (nvcc). Set CUDACXX, CUDA_PATH, "
        "CMAKE_CUDA_COMPILER, or put nvcc on PATH. A CUDA toolkit is needed "
        f"to build a wheel, but not to install a released wheel.{detail}"
    )


def cuda_cmake_arguments(cuda_compiler: Path) -> list[str]:
    """Return the release CUDA configuration for a single-process TTS wheel."""

    cuda_root = cuda_compiler.parent.parent
    return [
        "-DGGML_CUDA=ON",
        # NCCL is only useful for multi-GPU collective operations.  qwentts
        # uses one CUDA device per context, and linking NCCL would make
        # auditwheel bundle roughly 380 MB of unused runtime libraries.
        "-DGGML_CUDA_NCCL=OFF",
        f"-DCMAKE_CUDA_COMPILER={cuda_compiler.as_posix()}",
        f"-DCUDAToolkit_ROOT={cuda_root.as_posix()}",
    ]


def qwentts_abi_version(source: Path) -> int:
    header = source / "src" / "qwen.h"
    if not header.is_file():
        raise SystemExit(f"Could not verify qwentts.cpp ABI; header not found: {header}")
    match = re.search(r"^\s*#define\s+QT_ABI_VERSION\s+(\d+)\s*$", header.read_text(encoding="utf-8"), re.MULTILINE)
    if not match:
        raise SystemExit(f"Could not find QT_ABI_VERSION in {header}")
    return int(match.group(1))


def verify_qwentts_abi(source: Path, expected: int = PINNED_QWENTTS_ABI) -> None:
    actual = qwentts_abi_version(source)
    if actual != expected:
        raise SystemExit(
            f"Incompatible qwentts.cpp ABI {actual}; this wrapper expects ABI {expected}. "
            f"The release-tested native revision is {PINNED_QWENTTS_REF}."
        )


def find_first(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        for match in matches:
            if match.is_file() or match.is_symlink():
                return match
    return None


def strip_shared_library(path: Path) -> None:
    if not sys.platform.startswith("linux"):
        return
    strip = shutil.which("strip")
    if not strip:
        return
    try:
        run([strip, "--strip-unneeded", str(path)])
    except subprocess.CalledProcessError:
        pass


def copy_shared_libraries(
    build_dir: Path,
    package_lib_dir: Path,
    backend: str,
    *,
    clean_package_lib: bool = True,
) -> None:
    package_lib_dir.mkdir(parents=True, exist_ok=True)
    if clean_package_lib:
        for path in package_lib_dir.iterdir():
            if path.name == ".gitkeep":
                continue
            if path.is_file() or path.is_symlink():
                path.unlink()

    if sys.platform.startswith("linux"):
        libraries = [
            ("libqwen.so", ["libqwen.so", "libqwen.so.*"]),
            ("libggml-base.so.0", ["libggml-base.so.0", "libggml-base.so.*"]),
            ("libggml-cpu.so.0", ["libggml-cpu.so.0", "libggml-cpu.so.*"]),
            ("libggml-cuda.so.0", ["libggml-cuda.so.0", "libggml-cuda.so.*"]),
            ("libggml-vulkan.so.0", ["libggml-vulkan.so.0", "libggml-vulkan.so.*"]),
            ("libggml-sycl.so.0", ["libggml-sycl.so.0", "libggml-sycl.so.*"]),
            ("libggml.so.0", ["libggml.so.0", "libggml.so.*"]),
        ]
    elif sys.platform == "darwin":
        libraries = [
            ("libqwen.dylib", ["libqwen.dylib"]),
            ("libggml-base.dylib", ["libggml-base.dylib"]),
            ("libggml-cpu.dylib", ["libggml-cpu.dylib"]),
            ("libggml-cuda.dylib", ["libggml-cuda.dylib"]),
            ("libggml-vulkan.dylib", ["libggml-vulkan.dylib"]),
            ("libggml-sycl.dylib", ["libggml-sycl.dylib"]),
            ("libggml.dylib", ["libggml.dylib"]),
        ]
    else:
        libraries = [
            ("qwen.dll", ["qwen.dll"]),
            ("libqwen.dll", ["libqwen.dll"]),
            ("ggml-base.dll", ["ggml-base.dll"]),
            ("ggml-cpu.dll", ["ggml-cpu.dll"]),
            ("ggml-cuda.dll", ["ggml-cuda.dll"]),
            ("ggml-vulkan.dll", ["ggml-vulkan.dll"]),
            ("ggml-sycl.dll", ["ggml-sycl.dll"]),
            ("ggml.dll", ["ggml.dll"]),
        ]

    copied = []
    seen_targets = set()
    for dest_name, patterns in libraries:
        path = find_first(build_dir, patterns)
        if path is None:
            continue
        dest = package_lib_dir / dest_name
        resolved = path.resolve()
        target_key = (dest_name, resolved)
        if target_key in seen_targets:
            continue
        seen_targets.add(target_key)
        shutil.copy2(resolved, dest)
        copied.append(dest)

    if not any(p.name.startswith(("libqwen", "qwen")) for p in copied):
        raise SystemExit(f"No qwentts shared library found in {build_dir}")

    backend_stems = {
        "cpu": ("ggml-cpu", "libggml-cpu"),
        "cuda": ("ggml-cuda", "libggml-cuda"),
        "vulkan": ("ggml-vulkan", "libggml-vulkan"),
        "sycl": ("ggml-sycl", "libggml-sycl"),
    }
    if not any(path.name.startswith(backend_stems[backend]) for path in copied):
        raise SystemExit(f"No {backend} backend shared library found in {build_dir}")

    patchelf = shutil.which("patchelf")
    if patchelf and sys.platform.startswith("linux"):
        for path in copied:
            if path.is_file() and ".so" in path.name:
                try:
                    run([patchelf, "--set-rpath", "$ORIGIN", str(path)])
                except subprocess.CalledProcessError:
                    pass

    if os.environ.get("QWENTTS_CPP_NO_STRIP") != "1":
        for path in copied:
            strip_shared_library(path)

    print("Copied:")
    for path in copied:
        print(f"  {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=os.environ.get("QWENTTS_CPP_SOURCE", "third_party/qwentts.cpp"))
    parser.add_argument("--build-dir", default=os.environ.get("QWENTTS_CPP_BUILD_DIR", "build/qwentts-cpp"))
    parser.add_argument(
        "--backend",
        choices=["cpu", "cuda", "vulkan", "sycl"],
        default=os.environ.get("QWENTTS_CPP_BACKEND", "cuda"),
    )
    parser.add_argument("--cuda-compiler", default=None)
    parser.add_argument(
        "--expected-abi",
        type=int,
        default=int(os.environ.get("QWENTTS_CPP_EXPECTED_ABI", PINNED_QWENTTS_ABI)),
        help="Required qwentts.cpp C ABI version",
    )
    parser.add_argument("--cmake-arg", action="append", default=[], help="Extra CMake configure argument; repeatable")
    parser.add_argument("--target", default=os.environ.get("QWENTTS_CPP_CMAKE_TARGET", "qwen"))
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("QWENTTS_CPP_BUILD_JOBS", os.cpu_count() or 2)))
    parser.add_argument("--skip-build", action="store_true", help="Only copy shared libraries from --build-dir")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source = Path(args.source).resolve()
    build_dir = Path(args.build_dir).resolve()
    package_lib_dir = root / "src" / "qwentts_cpp" / "lib"

    if not args.skip_build and not source.is_dir():
        raise SystemExit(
            f"qwentts.cpp source checkout not found: {source}\n"
            "Clone it with --recurse-submodules or pass --source /path/to/qwentts.cpp."
        )

    if not args.skip_build:
        verify_qwentts_abi(source, args.expected_abi)

    if args.clean and build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        cmake_args = [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build_dir),
            "-DQWEN_SHARED=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON",
            "-DCMAKE_INSTALL_RPATH=$ORIGIN",
            *PORTABLE_CMAKE_ARGUMENTS,
        ]
        if args.backend == "cpu":
            cmake_args.append("-DGGML_BLAS=OFF")
        elif args.backend == "cuda":
            cuda_compiler = find_cuda_compiler(args.cuda_compiler)
            cmake_args.extend(cuda_cmake_arguments(cuda_compiler))
        elif args.backend == "vulkan":
            cmake_args.append("-DGGML_VULKAN=ON")
        elif args.backend == "sycl":
            cmake_args.append("-DGGML_SYCL=ON")
        cmake_args.extend(split_env_args(os.environ.get("QWENTTS_CPP_CMAKE_ARGS")))
        cmake_args.extend(args.cmake_arg)

        run(cmake_args)
        run(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--config",
                "Release",
                "--target",
                args.target,
                "--parallel",
                str(args.jobs),
            ]
        )
    copy_shared_libraries(
        build_dir,
        package_lib_dir,
        args.backend,
        clean_package_lib=not args.skip_build,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
