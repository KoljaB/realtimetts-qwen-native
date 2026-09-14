# realtimetts-qwen-native

RealtimeTTS-maintained Python `ctypes` bindings and binary-wheel packaging for
the `qwentts.cpp` C ABI. The distribution name is
`realtimetts-qwen-native`; the Python import remains `qwentts_cpp`.

This package is based on the MIT-licensed
[`qwentts-cpp-python`](https://github.com/andimarafioti/qwentts-cpp-python) by
Andres Marafioti and [`qwentts.cpp`](https://github.com/ServeurpersoCom/qwentts.cpp), with the release patch maintained in [`KoljaB/qwentts.cpp`](https://github.com/KoljaB/qwentts.cpp).
It uses a distinct distribution name and does not claim to be an official
release of either upstream project.

The package exposes buffered and streaming synthesis and can bundle the native
qwen/GGML libraries. It deliberately does **not** bundle GGUF model weights.

## Release status

Version 0.2.0+cpu2 is the local CPU-only package variant coordinated with RealtimeTTS 0.8.4.

The release-tested native source is pinned to qwentts.cpp commit
`82b83b5898b702f3d861805d9fa61cbe1ba90b2f`, which uses C ABI v5. The build
script verifies `QT_ABI_VERSION == 5` before compiling, and all release workflows
fetch that exact commit by default. The native `qt_version()` result remains the
authoritative runtime build identity.

This variant bundles only CPU native libraries, publishes the local `py3-none-linux_x86_64` wheel, and intentionally has no CUDA optional dependency.

### Optional x-vector onset profile

ABI v5 adds request-local suppression of checkpoint-specific leading-silence
codec tokens. It is **off by default**, applies only to Base x-vector-only
voice cloning, and rejects ICL or mismatched model assets instead of silently
changing them.

The first validated profile targets the exact public 0.6B Base Q8 talker and
12 Hz Q8 codec pair:

```python
from qwentts_cpp import (
    QWEN3_TTS_12HZ_0_6B_BASE_Q8_ONSET_PROFILE,
    QwenTTS,
)

for audio, sample_rate in tts.stream(
    text="Hello.",
    ref_spk_emb=speaker_embedding,
    onset_silence_profile=QWEN3_TTS_12HZ_0_6B_BASE_Q8_ONSET_PROFILE,
):
    consume(audio, sample_rate)
```

The profile validates both GGUF SHA-256 hashes once, then suppresses eight
tokenizer-derived c0 silence IDs during frames 0–2. It is checkpoint-specific,
not speaker-specific: any x-vector voice on the validated model pair can use it.

The CPU wheel also exports the separate experimental recovery profile
`qwen3_tts_12hz_0_6b_base_q8_cpu_recovery_v2`. It uses the same model hashes
and three-frame window, adds c0 ID `1221` to the v1 set, and is accepted only
when the loaded native library reports a CPU-only runtime. Use it only for one
bounded retry after an initial attempt has produced at least 240 ms of raw PCM
that is still silent. Do not apply it to every initial request: blanket use can
add an extra syllable. It carries no general quality or latency claim; `off`
and v1 remain unchanged.

## Supported binary targets

| Target | Wheel tag | Minimum runtime | Status |
| --- | --- | --- | --- |
| Windows 10/11 x64, NVIDIA | `py3-none-win_amd64` | AVX2/FMA/F16C/BMI2 CPU, CUDA-12-compatible driver | 0.2.0 release target |
| Linux x86_64, NVIDIA | `py3-none-manylinux_2_35_x86_64` | AVX2/FMA/F16C/BMI2 CPU, glibc 2.35, CUDA-12-compatible driver | 0.2.0 release target |
| Linux AArch64, NVIDIA | `py3-none-manylinux_*_aarch64` | target-dependent | retained secondary target |
| Linux CPU | `py3-none-linux_x86_64` | no CUDA, glibc 2.39 host | local CPU package variant |

The Python wrapper has no CPython extension, so one `py3-none-<platform>` wheel
supports Python 3.10 through 3.14. CUDA GPU builds target compute capability 7.5
and newer. Alpine/musl, Windows ARM64, and AMD GPU runtimes are not release
targets yet.

## Installation target

Install the local CPU wheel directly:

```bash
pip install realtimetts_qwen_native-0.2.0+cpu2-py3-none-linux_x86_64.whl
```

The wheel contains no CUDA binaries or CUDA package dependencies. Model weights
remain separate and must be supplied from the Hugging Face cache or a local
model directory.

Backend-specific development wheels can also be installed from a local wheelhouse:

```bash
python -m pip install --find-links /path/to/wheelhouse \
  "realtimetts-qwen-native[cuda12]==0.2.0"
```

Optional Hugging Face wheel indexes may carry backend-specific local versions.
Those local-version variants must not be uploaded to PyPI.

## Maintainer builds

Building a native wheel requires a compiler toolchain; installing a repaired
wheel does not. The build script resolves NVCC in this order:

1. `--cuda-compiler`
2. `CUDACXX`
3. `CMAKE_CUDA_COMPILER`
4. `CUDA_PATH/bin` or `CUDA_HOME/bin`
5. `PATH`

It supports both Ninja and Visual Studio generators and always builds the
`Release` configuration. The selected NVCC also fixes `CUDAToolkit_ROOT`, so a
different Toolkit earlier on `PATH` cannot leak headers or libraries into the
build. CUDA wheels explicitly set `GGML_CUDA_NCCL=OFF`: qwentts uses one device
per context, and accidentally linking unused NCCL would add roughly 380 MB to a
repaired Linux wheel. An incompatible qwentts.cpp ABI is rejected before CMake
is invoked.

CUDA development build with an existing qwentts.cpp checkout:

```bash
python scripts/build_native.py \
  --source /path/to/qwentts.cpp \
  --backend cuda \
  --clean \
  --cmake-arg=-G \
  --cmake-arg=Ninja
QWENTTS_CPP_WHEEL_BUILD_TAG=1cu128 python -m build --wheel
```

PowerShell with a Visual Studio generator can omit `-G Ninja`:

```powershell
$env:CUDACXX = "$env:CUDA_PATH\bin\nvcc.exe"
python scripts/build_native.py --source D:\src\qwentts.cpp --backend cuda --clean
$env:QWENTTS_CPP_WHEEL_BUILD_TAG = "1cu128"
python -m build --wheel
python -m delvewheel repair --analyze-existing `
  --ignore-existing `
  --add-path "$env:CUDA_PATH\bin;$env:CUDA_PATH\bin\x64;src\qwentts_cpp\lib" `
  --exclude "cudart64_12.dll;cublas64_12.dll;cublasLt64_12.dll;nvcuda.dll" `
  --wheel-dir D:\wheelhouse dist\*.whl
```

Windows release builds include both a native sm_75 cubin and sm_75 PTX. That
keeps the RTX 2080 SUPER latency target free of first-use JIT while retaining
forward compatibility for newer NVIDIA GPUs. Linux release builds use sm_75
PTX plus native cubins for newer architectures to stay below PyPI's default
per-file upload limit; on CC 7.5, the one-time PTX compilation happens while
the native model context is initialized.

CUDA runtime and cuBLAS are intentionally excluded from `delvewheel` and
`auditwheel` repair because the `cuda12` extra supplies them. qwen/GGML and any
other redistributable native dependencies remain inside the platform wheel.
The build helper also forces `GGML_NATIVE=OFF`; release wheels must not inherit
`-march=native` from the build host.

CPU development build:

```bash
python scripts/build_native.py \
  --source /path/to/qwentts.cpp \
  --backend cpu \
  --clean
QWENTTS_CPP_WHEEL_BUILD_TAG=1cpu python -m build --wheel
```

For a reproducible Linux x86_64 CUDA 12.8 release candidate, run in an Ubuntu
22.04/glibc-2.35 build environment with the CUDA 12.8 development toolkit:

```bash
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
python scripts/build_native.py \
  --source /path/to/qwentts.cpp \
  --build-dir /artifacts/qwentts-build \
  --backend cuda \
  --clean \
  --cmake-arg=-G \
  --cmake-arg=Ninja \
  --cmake-arg='-DCMAKE_CUDA_ARCHITECTURES=75-virtual;86-real;90-real;120-real;120-virtual'
QWENTTS_CPP_WHEEL_BUILD_TAG=1cu128 \
  python -m build --wheel --outdir /artifacts/raw
python -m auditwheel repair \
  --plat manylinux_2_35_x86_64 \
  --exclude libcudart.so.12 \
  --exclude libcublas.so.12 \
  --exclude libcublasLt.so.12 \
  --exclude libcuda.so.1 \
  --wheel-dir /artifacts/repaired \
  /artifacts/raw/*.whl
```

The expected repaired artifact is
`realtimetts_qwen_native-0.2.0-1cu128-py3-none-manylinux_2_35_x86_64.whl`.
The validated portable Linux candidate was 94,601,334 bytes, below PyPI's
104,857,600-byte per-file limit.
The earlier Linux validation build with an additional native sm_75 cubin was
111,869,499 bytes (about 112 MB), which is too large for PyPI's default file
limit. Linux release candidates must use the architecture set above and verify
the repaired wheel is at most 104,857,600 bytes. The portable Windows
validation candidate with native sm_75 plus PTX was 68,782,667 bytes. It also
uses `GGML_NATIVE=OFF`; a seven-run warm A/B on an RTX 2080 SUPER differed from
the build-host-tuned candidate by less than one percent in median RTF for both
X-Vector and ICL. Architecture sets and toolchain versions can change these
sizes.
Builds made with another CUDA minor must use the matching build tag (for
example `1cu125`) rather than claiming `1cu128`.

`QWENTTS_CPP_WHEEL_BUILD_TAG` distinguishes artifacts in a local wheelhouse.
Public indexes must contain only one backend flavor for each package version
and platform compatibility tag; otherwise pip cannot choose the intended
runtime.

## CI and release process

- `.github/workflows/tests.yml` runs model-free unit tests on Windows and Linux
  with the oldest and newest supported Python versions.
- `.github/workflows/publish.yml` builds CUDA 12.8 Windows and
  `manylinux_2_35` wheels plus the source archive from the pinned source. It
  never uploads them; publication happens only after installed-artifact
  acceptance and signed release-guard verification.
- Windows repair uses `delvewheel --analyze-existing`; Linux repair uses
  `auditwheel`. CUDA runtime, cuBLAS, and the driver library are external by
  design.
- Release builds run `twine check --strict` and are retained as immutable CI
  artifacts for installed-artifact acceptance and guarded publication.

Do not publish, tag, or upload a wheel simply because a local build succeeded.
Before release, validate an installed artifact on real Windows and Linux NVIDIA
hosts without a repository checkout or system CUDA Toolkit.
Build every public wheel from the tagged source; never rename a development
wheel into a release artifact.

## Fresh-wheel validation

Build artifacts should live outside the source checkout. Then create a clean
environment and install only from that wheelhouse:

```bash
python -m venv /path/to/fresh-venv
/path/to/fresh-venv/bin/python -m pip install --find-links /path/to/wheelhouse \
  "realtimetts-qwen-native[cuda12]==0.2.0"
/path/to/fresh-venv/bin/python -c \
  "from qwentts_cpp import QwenLibrary; print(QwenLibrary().version())"
```

On Windows use `fresh-venv\Scripts\python.exe`. Release validation must also
load the 0.6B Q8 model and exercise streaming, cancellation, X-Vector cloning,
and ICL cloning on a real GPU; a DLL-load-only check is not sufficient.

Model files are resolved through `huggingface-hub` by
`QwenTTS.from_pretrained(...)` or passed directly to `QwenTTS(...)` as GGUF
paths.

## Cached voice references

The pinned qwentts.cpp ABI v5 can skip reference WAV encoding for Base voice cloning by
passing precomputed latents:

- `.spk`: raw float32 speaker embedding from `qwen-codec --talker`
- `.rvq`: packed 11-bit reference codec stream from `qwen-codec`

The wrapper can create those files in-process from decoded mono float32 audio at
24 kHz:

```python
from qwentts_cpp import QwenTTS

tts = QwenTTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", quant="Q4_K_M")

# ref_audio_24k is a 1-D numpy float32 array, already resampled to 24 kHz.
voice_ref = tts.extract_voice_ref(ref_audio_24k)
voice_ref.save("reference.spk", "reference.rvq")
```

```python
from qwentts_cpp import QwenTTS, load_speaker_embedding

tts = QwenTTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", quant="Q4_K_M")

spk = load_speaker_embedding("reference.spk")
audio, sr = tts.synthesize(
    text="The sky is blue today.",
    lang="english",
    ref_spk_emb=spk,
    max_new_tokens=128,
)
```

For ICL clone mode, load the RVQ matrix with the model's codebook count and
also pass the reference transcript:

```python
from qwentts_cpp import load_rvq_codes

rvq = load_rvq_codes("reference.rvq", tts.num_codebooks())
audio, sr = tts.synthesize(
    text="The sky is blue today.",
    lang="english",
    ref_spk_emb=spk,
    ref_codes=rvq,
    ref_text="Transcript of the reference audio.",
)
```

## Private CPU pool build (2026-09-14)

The `2pool1` build of 0.2.0+cpu2 uses one independently owned persistent
GGML pthread pool per CPU context. Completed requests park their workers;
cancellation, pause/resume and context teardown retain their contracts.
Build for this i9-13900KF deployment with `GGML_OPENMP=OFF`,
`GGML_LLAMAFILE=OFF`, `GGML_NATIVE=ON`, and `QWEN_CPU_ONLY=ON`.
The model, quantization, sampling and codec ramp remain unchanged.
This local host-specific wheel is not a public portable binary.
