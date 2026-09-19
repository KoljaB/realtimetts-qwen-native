# realtimetts-qwen-native-cpu

`realtimetts-qwen-native-cpu` is the CPU-only native binding used by the
RealtimeTTS Qwen CPU server. The distribution name is
`realtimetts-qwen-native-cpu`; the Python import is `qwentts_cpp_cpu`, so it
can be installed beside the CUDA package without an import collision.

Version 0.3.0 bundles the qualified `qwentts.cpp` CPU implementation at
commit `b47728bd6cb60331bd02afacb390e533479329b5` (C ABI v5), including the
parked worker pool, strict affinity handling, onset-silence profiles, and CPU
recovery path. Model weights and voice references are not included.

## Install

In a fresh virtual environment, install the matching wheel from PyPI:

```bash
python -m pip install --upgrade pip
python -m pip install "realtimetts-qwen-native-cpu==0.3.0"
python -c "import qwentts_cpp_cpu as q; print(q.__version__, q.QT_ABI_VERSION, q.CPU_ONLY)"
```

The release provides these CPU wheels:

| Platform | Wheel baseline |
| --- | --- |
| Linux x86_64 | `manylinux_2_35_x86_64` (glibc 2.35 or newer) |
| Windows x86_64 | `win_amd64` |
| macOS Intel | `macosx_13_0_x86_64` |
| macOS Apple Silicon | `macosx_11_0_arm64` |

The native payload has no CUDA, Metal, Vulkan, OpenMP, or BLAS dependency.
The x86_64 builds use `GGML_NATIVE=OFF` with a fixed AVX2/FMA/F16C/BMI2
kernel baseline; they require a CPU with those features. The macOS arm64
wheel is built separately for Apple Silicon. There is no runtime CPU
dispatcher or generic x86 fallback.

Only the four platform families above have bundled native wheels. On another
platform, do not rely on an automatic source fallback: the sdist intentionally
refuses to create a wheel without the pinned native library and a qualified
`qwentts.cpp` checkout.

## Use

```python
from qwentts_cpp_cpu import QwenLibrary, QwenTTS

library = QwenLibrary()
assert library.cpu_only()
print(library.version())
```

Model files can be resolved from the configured Hugging Face repository:

```python
from qwentts_cpp_cpu import resolve_gguf_paths

talker, codec = resolve_gguf_paths(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base", quant="Q8_0"
)
```

The full RealtimeTTS CPU server supplies HTTP streaming, segmentation,
language detection, and recovery behavior on top of this binding.

## Build from source

Clone this repository and the pinned native source, then build the wheel and
sdist:

```bash
git clone https://github.com/KoljaB/realtimetts-qwen-native.git
git clone https://github.com/KoljaB/qwentts.cpp.git third_party/qwentts.cpp
git -C third_party/qwentts.cpp checkout b47728bd6cb60331bd02afacb390e533479329b5
python -m pip install --upgrade build
python scripts/build_native.py --source third_party/qwentts.cpp --backend cpu --clean
python -m build --sdist --wheel
```

The build helper verifies the native revision and ABI before configuring CMake.
The wheel bundles only `qwen`, `ggml`, `ggml-base`, and `ggml-cpu` shared
libraries with an origin-relative runtime path; it refuses a build with no
native payload.
