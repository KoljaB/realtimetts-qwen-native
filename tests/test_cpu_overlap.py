import ctypes
import threading

import pytest

from qwentts_cpp_cpu import _binding as binding
from test_runtime_loader import _fake_tts


class FakeLibrary:
    def __init__(self, _path):
        self._has_qt_cpu_only = True
        self._has_qt_init_cpu = True
        self._has_qt_init_cpu_ex = True
        self._lib = self
        self.calls = []

    def default_init_params(self):
        params = binding.QtInitParams()
        params.abi_version = binding.QT_ABI_VERSION
        return params

    def cpu_only(self):
        return True

    def qt_init_cpu(self, params, threads):
        self.calls.append(("legacy", threads.value))
        return 123

    def qt_init_cpu_ex(self, params, threads, options):
        opts = options._obj
        self.calls.append(("extended", threads, opts.version, opts.codec_threads,
                           opts.stream_frames, opts.worker_mask, opts.codec_mask))
        return 123

    def qt_free(self, ctx):
        pass


def test_cpu_extended_initializer_preserves_legacy_struct(monkeypatch):
    monkeypatch.setattr(binding, "QwenLibrary", FakeLibrary)
    monkeypatch.setattr(binding.os, "cpu_count", lambda: 24)
    model = binding.QwenTTS("talker.gguf", "codec.gguf", cpu_threads=6,
                           cpu_codec_threads=6, cpu_stream_frames=2,
                           cpu_affinity=0x555, cpu_codec_affinity=0x555000)
    assert model.library.calls == [("extended", 6, 1, 6, 2, 0x555, 0x555000)]
    assert ctypes.sizeof(binding.QtInitParams) == 40
    assert ctypes.sizeof(binding.QtCpuOptions) == 32
    model.close()


def test_default_cpu_options_keep_legacy_initializer(monkeypatch):
    monkeypatch.setattr(binding, "QwenLibrary", FakeLibrary)
    model = binding.QwenTTS("talker.gguf", "codec.gguf", cpu_threads=1)
    assert model.library.calls == [("legacy", 1)]
    model.close()


def test_requested_overlap_cannot_silently_use_old_library(monkeypatch):
    class OldLibrary(FakeLibrary):
        def __init__(self, path):
            super().__init__(path)
            self._has_qt_init_cpu_ex = False

    monkeypatch.setattr(binding, "QwenLibrary", OldLibrary)
    with pytest.raises(binding.ABIMismatchError, match="qt_init_cpu_ex"):
        binding.QwenTTS("talker.gguf", "codec.gguf", cpu_threads=1, cpu_codec_threads=1)


def test_overlap_rejects_batched_context(monkeypatch):
    monkeypatch.setattr(binding, "QwenLibrary", FakeLibrary)
    with pytest.raises(ValueError, match="max_batch"):
        binding.QwenTTS("talker.gguf", "codec.gguf", cpu_threads=1,
                       cpu_codec_threads=1, max_batch=2)


def test_stream_consumer_never_acknowledges_native_pause():
    checkpoint_threads = []
    consumer_polls = []

    class Control:
        def is_set(self):
            checkpoint_threads.append(threading.get_ident())
            return False

        def cancelled(self):
            consumer_polls.append(threading.get_ident())
            return False

    tts = _fake_tts()
    stream = tts.stream(text="pause checkpoint", cancel_event=Control())
    next(stream)
    stream.close()
    consumer_thread = threading.get_ident()
    assert checkpoint_threads and consumer_thread not in checkpoint_threads
    assert consumer_polls == [consumer_thread]
    assert tts.last_stream_profile["producer_alive_after_close"] is False
