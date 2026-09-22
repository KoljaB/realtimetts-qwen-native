"""Actual CPU speech from a fresh installed wheel, never a source import."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import wave
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
import qwentts_cpp_cpu as native


MODEL_REVISION = "e0f336a048a3de02b29b8ad92969217d9ecffe3e"
MODELS = {
    "qwen-talker-0.6b-base-Q8_0.gguf": "d54dbaf10591421fa764ed630d764efa717ae40cd959bd48c66d4eb1af226426",
    "qwen-tokenizer-12hz-Q8_0.gguf": "1883beeed99348fc35e23dd225e9082f93f6f8c109330a33d935baa8acdbfd94",
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path, help="Existing pinned GGUF files")
    args = parser.parse_args()
    module_path = Path(native.__file__).resolve()
    assert module_path.is_relative_to(Path(sys.prefix).resolve()), module_path
    assert native.CPU_ONLY is True and native.QT_ABI_VERSION == 5
    library = native.QwenLibrary()
    assert library.cpu_only()
    print("Installed CPU wheel loaded:", module_path, flush=True)
    models = []
    for filename, expected in MODELS.items():
        print("Resolving checkpoint:", filename, flush=True)
        path = args.model_dir / filename if args.model_dir else Path(hf_hub_download(
            repo_id="Serveurperso/Qwen3-TTS-GGUF", revision=MODEL_REVISION,
            filename=filename, cache_dir=str(args.cache_dir),
        ))
        assert digest(path) == expected, filename
        models.append(path)
    with wave.open(str(args.sample), "rb") as sample:
        assert (sample.getnchannels(), sample.getsampwidth(), sample.getframerate()) == (1, 2, 24000)
        reference = np.frombuffer(sample.readframes(sample.getnframes()), dtype="<i2").astype(np.float32) / 32768
    assert reference.size and np.isfinite(reference).all()
    print("Loading the qualified 0.6B Base Q8 model ...", flush=True)
    started = time.perf_counter()
    outputs = {}
    original_priority = os.environ.get("QWENTTS_CPU_STARTUP_PRIORITY")
    try:
        for mode, codec_threads, priority in (
            ("serial", 0, "off"),
            ("overlap", 2, "off"),
            ("startup_priority", 2, "second_chunk"),
        ):
            os.environ["QWENTTS_CPU_STARTUP_PRIORITY"] = priority
            with native.QwenTTS(*models, use_fa=False, cpu_threads=2,
                               cpu_codec_threads=codec_threads, cpu_stream_frames=2) as tts:
                print("Synthesizing CPU speech:", mode, flush=True)
                chunks = list(tts.stream(
                    text="Portable CPU wheel smoke.", lang="english",
                    ref_audio_24k=np.ascontiguousarray(reference),
                    onset_silence_profile=native.QWEN3_TTS_12HZ_0_6B_BASE_Q8_ONSET_PROFILE,
                    seed=42, max_new_tokens=20, do_sample=False,
                ))
            assert chunks and all(rate == 24000 for _, rate in chunks), mode
            audio = np.concatenate([chunk for chunk, _ in chunks])
            assert np.isfinite(audio).all() and np.max(np.abs(audio)) > 1e-6, mode
            outputs[mode] = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        assert outputs["serial"] == outputs["overlap"] == outputs["startup_priority"]
    finally:
        if original_priority is None:
            os.environ.pop("QWENTTS_CPU_STARTUP_PRIORITY", None)
        else:
            os.environ["QWENTTS_CPU_STARTUP_PRIORITY"] = original_priority
    sample_rate = 24000
    audio = np.asarray(audio)
    assert sample_rate == 24000 and audio.size and np.isfinite(audio).all()
    assert float(np.max(np.abs(audio))) > 1e-6, "silent synthesis"
    with wave.open("cpu-smoke.wav", "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    result = {
        "python": sys.version, "platform": platform.platform(),
        "module": str(module_path), "version": native.__version__,
        "native_version": library.version(), "cpu_only": True, "abi": 5,
        "model_sha256": MODELS, "sample_rate": int(sample_rate),
        "samples": int(audio.size), "peak": float(np.max(np.abs(audio))),
        "elapsed_seconds": time.perf_counter() - started,
        "pcm_sha256": {mode: hashlib.sha256(pcm).hexdigest() for mode, pcm in outputs.items()},
        "serial_overlap_startup_pcm_equal": True,
    }
    Path("cpu-smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
