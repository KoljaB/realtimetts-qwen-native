from __future__ import annotations

import json
from pathlib import Path

import qwentts_cpp.__main__ as cli
import qwentts_cpp._diagnostics as diagnostics


def test_doctor_json_exit_status_reflects_errors(monkeypatch, capsys):
    report = {
        "package_version": "test",
        "errors": ["native library missing"],
        "warnings": [],
    }
    monkeypatch.setattr(diagnostics, "build_doctor_report", lambda **_kwargs: report)

    # run_doctor resolves its module global, while argparse wiring stays public.
    assert cli.main(["doctor", "--json", "--no-model-check"]) == 1
    assert json.loads(capsys.readouterr().out)["errors"] == ["native library missing"]


def test_prefetch_and_download_alias_forward_model_options(tmp_path, monkeypatch, capsys):
    calls = []
    talker = tmp_path / "talker.gguf"
    codec = tmp_path / "codec.gguf"

    def resolve(model, **kwargs):
        calls.append((model, kwargs))
        return talker, codec

    monkeypatch.setattr(diagnostics, "resolve_gguf_paths", resolve)

    for command in ("prefetch", "download"):
        assert cli.main([command, "--model", "model-id", "--quant", "Q8_0", "--cache-dir", str(tmp_path)]) == 0

    assert len(calls) == 2
    assert all(call[0] == "model-id" for call in calls)
    assert all(call[1]["quant"] == "Q8_0" for call in calls)
    assert all(call[1]["cache_dir"] == str(tmp_path) for call in calls)
    assert str(talker) in capsys.readouterr().out


def test_prefetch_reports_download_error_without_traceback(monkeypatch, capsys):
    def fail(*_args, **_kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(diagnostics, "resolve_gguf_paths", fail)

    assert cli.main(["prefetch"]) == 1
    assert "Could not prefetch" in capsys.readouterr().err


def test_build_doctor_report_checks_native_abi_gpu_and_cached_model(tmp_path, monkeypatch):
    library_path = tmp_path / "qwen.dll"
    library_path.touch()
    talker = tmp_path / "talker.gguf"
    codec = tmp_path / "codec.gguf"

    class Library:
        native_abi = 4
        cuda_major = 12
        preloaded_libraries = [tmp_path / "cublas.dll"]
        preloaded_library_sources = {tmp_path / "cublas.dll": "nvidia-cublas-cu12"}

        def __init__(self, path):
            assert path == library_path

        def version(self):
            return "7b6ed4f (2026-08-05)"

    monkeypatch.setattr(diagnostics, "find_library", lambda _path: library_path)
    monkeypatch.setattr(diagnostics, "QwenLibrary", Library)
    monkeypatch.setattr(
        diagnostics,
        "query_nvidia_gpus",
        lambda: ([diagnostics.GPUInfo("RTX", "555.1", "7.5")], None),
    )
    monkeypatch.setattr(diagnostics, "resolve_gguf_paths", lambda *_args, **_kwargs: (talker, codec))

    report = diagnostics.build_doctor_report()

    assert report["errors"] == []
    assert report["native_abi"] == 4
    assert report["gpus"][0]["compute_capability"] == "7.5"
    assert report["model_cache"]["ready"] is True
    assert report["preloaded_libraries"] == [
        {"path": str(tmp_path / "cublas.dll"), "source": "nvidia-cublas-cu12"}
    ]


def test_doctor_hardware_contract_rejects_missing_gpu_old_driver_and_old_gpu(monkeypatch):
    monkeypatch.setattr(diagnostics, "query_nvidia_gpus", lambda: ([], "nvidia-smi missing"))
    report = diagnostics.build_doctor_report(check_model=False)
    assert "nvidia-smi missing" in report["errors"]

    assert diagnostics._hardware_errors([diagnostics.GPUInfo("old", "524.0", "7.0")]) == [
        "old: driver 524.0 is older than the CUDA 12 compatibility baseline 527.41; update the NVIDIA driver",
        "old: compute capability 7.0 is below the supported minimum 7.5",
    ]


def test_platform_contract_requires_glibc_235_on_linux(monkeypatch):
    monkeypatch.setattr(diagnostics.sys, "platform", "linux")
    monkeypatch.setattr(diagnostics.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(diagnostics.platform, "libc_ver", lambda: ("glibc", "2.34"))

    assert diagnostics._platform_errors() == [
        "glibc 2.34 is older than the supported minimum 2.35"
    ]


def test_platform_contract_requires_windows_10_or_11(monkeypatch):
    monkeypatch.setattr(diagnostics.sys, "platform", "win32")
    monkeypatch.setattr(diagnostics.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(diagnostics.platform, "release", lambda: "8.1")

    assert diagnostics._platform_errors() == [
        "Unsupported Windows release 8.1; the MVP supports Windows 10/11"
    ]


def test_platform_contract_uses_confstr_when_libc_probe_is_empty(monkeypatch):
    monkeypatch.setattr(diagnostics.sys, "platform", "linux")
    monkeypatch.setattr(diagnostics.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(diagnostics.platform, "libc_ver", lambda: ("", ""))
    monkeypatch.setattr(diagnostics.os, "confstr", lambda _name: "glibc 2.35", raising=False)

    assert diagnostics._platform_errors() == []
