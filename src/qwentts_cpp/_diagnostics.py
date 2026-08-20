from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from . import QT_ABI_VERSION, __version__
from ._binding import QwenLibrary, QwenTTSError, discover_runtime_library_dirs, find_library
from .models import GGUF_REPO, resolve_gguf_paths

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_QUANT = "Q8_0"


@dataclass(frozen=True)
class GPUInfo:
    name: str
    driver_version: str
    compute_capability: str | None = None


def query_nvidia_gpus() -> tuple[list[GPUInfo], str | None]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return [], "nvidia-smi was not found on PATH"

    commands = (
        ([executable, "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader,nounits"], True),
        ([executable, "--query-gpu=name,driver_version", "--format=csv,noheader,nounits"], False),
    )
    last_error: str | None = None
    for command, has_capability in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            last_error = str(exc)
            continue
        if result.returncode != 0:
            last_error = result.stderr.strip() or f"nvidia-smi exited with {result.returncode}"
            continue
        gpus: list[GPUInfo] = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            capability = parts[2] if has_capability and len(parts) > 2 else None
            gpus.append(GPUInfo(parts[0], parts[1], capability))
        return gpus, None if gpus else "nvidia-smi returned no GPUs"
    return [], last_error or "nvidia-smi could not query the NVIDIA driver"


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in value.split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _hardware_errors(gpus: Sequence[GPUInfo]) -> list[str]:
    errors: list[str] = []
    minimum_driver = (527, 41) if sys.platform == "win32" else (525, 60, 13)
    for gpu in gpus:
        driver = _version_tuple(gpu.driver_version)
        if not driver:
            errors.append(f"{gpu.name}: could not parse NVIDIA driver version {gpu.driver_version!r}")
        elif driver < minimum_driver:
            errors.append(
                f"{gpu.name}: driver {gpu.driver_version} is older than the CUDA 12 compatibility "
                f"baseline {'.'.join(map(str, minimum_driver))}; update the NVIDIA driver"
            )
        if gpu.compute_capability:
            try:
                capability = float(gpu.compute_capability)
            except ValueError:
                capability = None
            if capability is None:
                errors.append(f"{gpu.name}: could not parse compute capability {gpu.compute_capability!r}")
            elif capability < 7.5:
                errors.append(
                    f"{gpu.name}: compute capability {gpu.compute_capability} is below the supported minimum 7.5"
                )
        else:
            errors.append(f"{gpu.name}: nvidia-smi did not report compute capability; cannot verify the 7.5 minimum")
    return errors


def _platform_errors() -> list[str]:
    errors: list[str] = []
    machine = platform.machine().lower()
    if machine not in {"amd64", "x86_64"}:
        errors.append(f"Unsupported machine {platform.machine()}; the MVP supports x86_64 only")
    if sys.platform == "win32":
        release = platform.release()
        if release not in {"10", "11"}:
            errors.append(f"Unsupported Windows release {release}; the MVP supports Windows 10/11")
    elif sys.platform.startswith("linux"):
        libc_name, libc_version = platform.libc_ver()
        if libc_name.lower() not in {"glibc", "gnu libc"}:
            try:
                confstr = os.confstr("CS_GNU_LIBC_VERSION")
            except (AttributeError, OSError, ValueError):
                confstr = None
            if confstr:
                parts = confstr.rsplit(" ", 1)
                if len(parts) == 2:
                    libc_name, libc_version = parts
        if libc_name.lower() not in {"glibc", "gnu libc"}:
            errors.append(
                f"Unsupported Linux C library {libc_name or 'unknown'} {libc_version or ''}; glibc >= 2.35 is required"
            )
        elif _version_tuple(libc_version) < (2, 35):
            errors.append(f"glibc {libc_version} is older than the supported minimum 2.35")
    else:
        errors.append(f"Unsupported platform {sys.platform}; the MVP supports Windows and glibc Linux")
    if not ((3, 10) <= sys.version_info[:2] <= (3, 14)):
        errors.append(
            f"Unsupported Python {platform.python_version()}; use Python 3.10 through 3.14"
        )
    return errors


def build_doctor_report(
    *,
    library_path: str | None = None,
    model: str = DEFAULT_MODEL,
    quant: str = DEFAULT_QUANT,
    cache_dir: str | None = None,
    check_model: bool = True,
) -> dict[str, object]:
    report: dict[str, object] = {
        "package_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "expected_native_abi": QT_ABI_VERSION,
        "runtime_library_dirs": [
            {"path": str(item.path), "source": item.source}
            for item in discover_runtime_library_dirs()
        ],
        "errors": [],
        "warnings": [],
        "solutions": [],
    }
    errors = report["errors"]
    warnings = report["warnings"]
    solutions = report["solutions"]
    assert isinstance(errors, list) and isinstance(warnings, list) and isinstance(solutions, list)

    errors.extend(_platform_errors())

    gpus, gpu_error = query_nvidia_gpus()
    report["gpus"] = [asdict(gpu) for gpu in gpus]
    if gpu_error:
        errors.append(gpu_error)
        solutions.append("Install/update the NVIDIA driver and ensure nvidia-smi is available on PATH")
    errors.extend(_hardware_errors(gpus))

    try:
        native_path = find_library(library_path)
        report["native_library"] = str(native_path)
        library = QwenLibrary(native_path)
        report["native_version"] = library.version()
        report["native_abi"] = library.native_abi
        report["native_cuda_major"] = library.cuda_major
        report["preloaded_libraries"] = [
            {
                "path": str(path),
                "source": library.preloaded_library_sources.get(path, "unknown"),
            }
            for path in library.preloaded_libraries
        ]
    except QwenTTSError as exc:
        report["native_error"] = str(exc)
        errors.append(str(exc))
        solutions.append(
            "Install `realtimetts-qwen-native` from the matching CUDA wheel and run this doctor command again"
        )

    if check_model:
        report["model"] = model
        report["quant"] = quant
        try:
            talker, codec = resolve_gguf_paths(
                model,
                quant=quant,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            report["model_cache"] = {
                "ready": True,
                "talker": str(talker),
                "codec": str(codec),
            }
        except Exception as exc:
            report["model_cache"] = {"ready": False, "detail": str(exc)}
            warnings.append(f"The {model} {quant} model is not fully present in the local cache")
            solutions.append(
                f"Run `python -m qwentts_cpp prefetch --model {model} --quant {quant}` while online"
            )
    return report


def _print_human_report(report: dict[str, object]) -> None:
    print(f"realtimetts-qwen-native {report['package_version']}")
    print(f"Python: {report['python']} ({report['machine']})")
    print(f"Platform: {report['platform']}")
    print(f"Expected native ABI: {report['expected_native_abi']}")
    print(f"Native library: {report.get('native_library', 'not found')}")
    if "native_version" in report:
        print(
            f"Native version: {report['native_version']} (ABI {report['native_abi']}, "
            f"CUDA {report['native_cuda_major']})"
        )
    preloaded = report.get("preloaded_libraries", [])
    if preloaded:
        print("Preloaded CUDA libraries:")
        for item in preloaded:  # type: ignore[assignment]
            print(f"  - {item['path']} [{item['source']}]")

    gpus = report.get("gpus", [])
    if gpus:
        for gpu in gpus:  # type: ignore[assignment]
            capability = gpu.get("compute_capability") or "unknown"
            print(f"GPU: {gpu['name']} (driver {gpu['driver_version']}, compute capability {capability})")
    else:
        print("GPU: not detected")

    directories = report.get("runtime_library_dirs", [])
    print("CUDA library directories:")
    if directories:
        for item in directories:  # type: ignore[assignment]
            print(f"  - {item['path']} [{item['source']}]")
    else:
        print("  - none found")

    cache = report.get("model_cache")
    if isinstance(cache, dict):
        if cache.get("ready"):
            print(f"Model cache: ready\n  talker: {cache['talker']}\n  codec: {cache['codec']}")
        else:
            print("Model cache: not ready (the first normal load can download it)")

    for label in ("warnings", "errors", "solutions"):
        values = report.get(label, [])
        if values:
            print(f"{label.capitalize()}:")
            for value in values:  # type: ignore[assignment]
                print(f"  - {value}")
    if not report.get("errors"):
        print("Result: native runtime is loadable")


def run_doctor(args) -> int:
    report = build_doctor_report(
        library_path=args.library,
        model=args.model,
        quant=args.quant,
        cache_dir=args.cache_dir,
        check_model=not args.no_model_check,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human_report(report)
    if report.get("errors"):
        return 1
    if args.strict and report.get("warnings"):
        return 1
    return 0


def run_prefetch(args) -> int:
    try:
        talker, codec = resolve_gguf_paths(
            args.model,
            quant=args.quant,
            repo_id=args.repo_id,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        print(f"Could not prefetch {args.model} {args.quant}: {exc}", file=sys.stderr)
        return 1
    print(f"Talker: {talker}")
    print(f"Codec: {codec}")
    print("The model is ready for offline use with the same model, quant, repo, and cache directory.")
    return 0
