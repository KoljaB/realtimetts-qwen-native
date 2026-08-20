from __future__ import annotations

import ctypes
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import qwentts_cpp._binding as binding
from qwentts_cpp import (
    ABIMismatchError,
    OutOfMemoryError,
    QT_ABI_VERSION,
    QwenLibrary,
    QwenStatus,
    QwenTTS,
)
from qwentts_cpp._binding import QtAudio, QtInitParams, QtTTSParams


def test_abi4_ctypes_structs_match_qwen_header_field_order_and_offsets():
    assert QT_ABI_VERSION == 4
    assert [name for name, _ in QtInitParams._fields_] == [
        "abi_version",
        "talker_path",
        "codec_path",
        "use_fa",
        "clamp_fp16",
        "max_batch",
        "codec_chunk_sec",
    ]
    assert [name for name, _ in QtTTSParams._fields_] == [
        "abi_version",
        "text",
        "lang",
        "instruct",
        "speaker",
        "ref_audio_24k",
        "ref_n_samples",
        "ref_text",
        "seed",
        "max_new_tokens",
        "do_sample",
        "temperature",
        "top_k",
        "top_p",
        "repetition_penalty",
        "subtalker_do_sample",
        "subtalker_temperature",
        "subtalker_top_k",
        "subtalker_top_p",
        "dump_dir",
        "cancel",
        "cancel_user_data",
        "on_chunk",
        "on_chunk_user_data",
        "ref_spk_emb",
        "ref_spk_dim",
        "ref_codes",
        "ref_T",
    ]
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        assert ctypes.sizeof(QtInitParams) == 40
        assert ctypes.sizeof(QtTTSParams) == 184
        assert QtInitParams.max_batch.offset == 28
        assert QtInitParams.codec_chunk_sec.offset == 32
        assert QtTTSParams.ref_spk_emb.offset == 152
        assert QtTTSParams.ref_T.offset == 176


def test_default_initializer_rejects_mismatched_abi_without_small_struct_write():
    callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(QtInitParams))

    def initialize(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = QT_ABI_VERSION + 1

    with pytest.raises(ABIMismatchError, match="reports ABI 5"):
        QwenLibrary._read_default_params(callback_type(initialize), QtInitParams, "test initializer")


def test_runtime_discovery_prioritizes_nvidia_packages_before_torch_and_cuda_path(tmp_path, monkeypatch):
    nvidia_runtime = tmp_path / "nvidia" / "cuda_runtime"
    nvidia_cublas = tmp_path / "nvidia" / "cublas"
    torch = tmp_path / "torch"
    cuda = tmp_path / "cuda"
    for path in (
        nvidia_runtime / "bin",
        nvidia_cublas / "bin",
        torch / "lib",
        cuda / "bin",
        cuda / "bin" / "x64",
    ):
        path.mkdir(parents=True)
    (nvidia_runtime / "bin" / "cudart64_12.dll").touch()
    (nvidia_cublas / "bin" / "cublas64_12.dll").touch()
    (torch / "lib" / "cudart64_12.dll").touch()
    (cuda / "bin" / "cudart64_12.dll").touch()
    (cuda / "bin" / "x64" / "cublas64_12.dll").touch()

    roots = {
        "nvidia.cuda_runtime": nvidia_runtime,
        "nvidia.cublas": nvidia_cublas,
        "torch": torch,
    }
    monkeypatch.setattr(binding, "_package_root", roots.get)
    monkeypatch.setattr(binding.sys, "platform", "win32")
    monkeypatch.setenv("CUDA_PATH", str(cuda))
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("QWENTTS_CPP_CUDA_LIBRARY_PATH", raising=False)

    discovered = binding.discover_runtime_library_dirs()

    assert [(item.path, item.source) for item in discovered] == [
        (nvidia_runtime / "bin", "nvidia-cuda-runtime-cu12"),
        (nvidia_cublas / "bin", "nvidia-cublas-cu12"),
        (torch / "lib", "torch fallback"),
        (cuda / "bin", "CUDA_PATH fallback"),
        (cuda / "bin" / "x64", "CUDA_PATH fallback"),
    ]


def test_explicit_runtime_override_precedes_nvidia_packages(tmp_path, monkeypatch):
    override = tmp_path / "override"
    nvidia = tmp_path / "nvidia" / "cuda_runtime"
    override.mkdir()
    (nvidia / "lib").mkdir(parents=True)
    (nvidia / "lib" / "libcudart.so.12").touch()
    monkeypatch.setenv("QWENTTS_CPP_CUDA_LIBRARY_PATH", str(override))
    monkeypatch.setattr(
        binding,
        "_package_root",
        lambda name: nvidia if name == "nvidia.cuda_runtime" else None,
    )
    monkeypatch.setattr(binding.sys, "platform", "linux")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)

    discovered = binding.discover_runtime_library_dirs()

    assert discovered[0].path == override
    assert discovered[0].source == "QWENTTS_CPP_CUDA_LIBRARY_PATH"
    assert discovered[1].source == "nvidia-cuda-runtime-cu12"


def test_native_error_translation_identifies_oom_and_abi():
    oom = binding._native_exception("CUDA out of memory", status=int(QwenStatus.OOM))
    abi = binding._native_exception("ABI version 2 outside supported range")

    assert isinstance(oom, OutOfMemoryError)
    assert "Q8_0" in str(oom)
    assert isinstance(abi, ABIMismatchError)
    assert "ABI 4" in str(abi)


def test_cuda_major_is_inferred_from_windows_backend_binary(tmp_path, monkeypatch):
    backend = tmp_path / "ggml-cuda.dll"
    backend.write_bytes(b"prefix cublas64_13.dll suffix")
    monkeypatch.setattr(binding.sys, "platform", "win32")

    assert binding._detect_cuda_major(tmp_path) == 13


class _FakeStreamingNative:
    def __init__(self):
        self.calls = 0

    def qt_synthesize(self, _ctx, params_pointer, _audio_pointer):
        self.calls += 1
        params = params_pointer._obj
        samples = (ctypes.c_float * 4)(0.0, 0.25, -0.25, 0.0)
        while not params.cancel(None):
            if not params.on_chunk(samples, len(samples), None):
                return int(QwenStatus.CANCELLED)
            time.sleep(0.002)
        return int(QwenStatus.CANCELLED)

    def qt_audio_free(self, _audio_pointer):
        return None


class _FakeStreamingLibrary:
    def __init__(self):
        self._lib = _FakeStreamingNative()
        self._has_qt_num_codebooks = False

    def default_tts_params(self):
        params = QtTTSParams()
        params.abi_version = QT_ABI_VERSION
        params.seed = -1
        return params

    def last_error(self):
        return ""


def _fake_tts() -> QwenTTS:
    tts = QwenTTS.__new__(QwenTTS)
    tts.library = _FakeStreamingLibrary()
    tts._ctx = 123
    tts._lock = threading.Lock()
    tts.last_stream_profile = None
    return tts


def test_stream_external_cancel_event_stops_native_and_context_is_reusable():
    tts = _fake_tts()
    external = threading.Event()
    first_stream = tts.stream(text="first", cancel_event=external)

    first_chunk, sample_rate = next(first_stream)
    external.set()
    assert list(first_stream) == []

    assert sample_rate == 24000
    np.testing.assert_array_equal(first_chunk, np.array([0.0, 0.25, -0.25, 0.0], dtype=np.float32))
    second_external = threading.Event()
    second_stream = tts.stream(text="second", cancel_event=second_external)
    next(second_stream)
    second_stream.close()
    assert tts.library._lib.calls == 2
    assert not second_external.is_set(), "the generator must not take ownership of the caller's Event"


def test_stream_profile_exposes_absolute_callback_timestamp():
    tts = _fake_tts()
    stream = tts.stream(text="profile")
    next(stream)
    stream.close()

    profile = tts.last_stream_profile
    assert isinstance(profile["first_callback_perf_counter_ns"], int)
    assert isinstance(profile["first_yield_perf_counter_ns"], int)
    assert profile["first_yield_perf_counter_ns"] >= profile["first_callback_perf_counter_ns"]


def test_stream_cancellation_after_last_queued_chunk_does_not_drop_it():
    class OneChunkNative(_FakeStreamingNative):
        def qt_synthesize(self, _ctx, params_pointer, _audio_pointer):
            params = params_pointer._obj
            samples = (ctypes.c_float * 2)(0.5, -0.5)
            assert params.on_chunk(samples, len(samples), None)
            return int(QwenStatus.OK)

    tts = _fake_tts()
    tts.library._lib = OneChunkNative()

    chunks = list(tts.stream(text="short"))

    assert len(chunks) == 1
    np.testing.assert_array_equal(chunks[0][0], np.array([0.5, -0.5], dtype=np.float32))


def test_close_waits_for_active_native_operation_before_freeing_context():
    freed = threading.Event()

    class Native:
        def qt_free(self, context):
            assert context == 123
            freed.set()

    tts = QwenTTS.__new__(QwenTTS)
    tts.library = type("Library", (), {"_lib": Native()})()
    tts._ctx = 123
    tts._lock = threading.Lock()
    tts._lock.acquire()
    closer = threading.Thread(target=tts.close)
    closer.start()
    try:
        assert not freed.wait(0.05)
    finally:
        tts._lock.release()
    closer.join(timeout=1.0)

    assert freed.is_set()
    assert tts._ctx is None


def test_bundled_windows_library_reports_pinned_abi4():
    if os.name != "nt":
        pytest.skip("Windows ABI smoke")
    library_path = Path(binding.__file__).resolve().parent / "lib" / "qwen.dll"
    if not library_path.is_file():
        pytest.skip("Bundled qwen.dll not present in this checkout")

    library = QwenLibrary(library_path)
    init_defaults = library.default_init_params()
    tts_defaults = library.default_tts_params()

    assert library.native_abi == 4
    assert library.version().startswith("7b6ed4f")
    assert init_defaults.abi_version == 4
    assert init_defaults.max_batch == 1
    assert init_defaults.codec_chunk_sec == pytest.approx(24.0)
    assert tts_defaults.abi_version == 4
    assert tts_defaults.max_new_tokens == 2048
