# realtimetts-qwen-native-cpu

Portable CPU-only native wheels for the RealtimeTTS Qwen server. The package
distribution is realtimetts-qwen-native-cpu; its Python import is
qwentts_cpp_cpu, so it can be installed beside the CUDA package without an
import collision.

This candidate bundles the qualified qwentts.cpp CPU implementation at commit
b47728bd6cb60331bd02afacb390e533479329b5 (C ABI v5), including the parked
worker pool, strict affinity handling, onset-silence profiles, and CPU
recovery path. It does not bundle model weights.

INSTALL
Install the wheel in a fresh virtual environment:

  python -m pip install realtimetts_qwen_native_cpu-0.3.0-py3-none-linux_x86_64.whl
  python -c "from qwentts_cpp_cpu import QwenLibrary; print(QwenLibrary().version())"

The wheel contains no CUDA, Metal, Vulkan, OpenMP, or BLAS dependency. It is
compiled with GGML_NATIVE=OFF and therefore does not encode this build host's
CPU instruction set. The current x86_64 build uses the fixed AVX2/FMA/F16C/BMI2
kernel set and requires a CPU with those features; it does not claim to run on
every x86_64 machine. It also requires a compatible glibc runtime. Windows and
macOS CPU wheels require their own platform builds.

BUILD
The source checkout must contain qwentts.cpp at the pinned revision:

  python scripts/build_native.py --source /path/to/qwentts.cpp --backend cpu --clean
  python -m build --sdist --wheel

The build helper verifies the qwentts.cpp revision and ABI before configuring
CMake. The resulting wheel bundles only libqwen, libggml, libggml-base, and
libggml-cpu shared libraries with an ORIGIN runtime path.

RUNTIME
  from qwentts_cpp_cpu import QwenLibrary, QwenTTS
  library = QwenLibrary()
  assert library.cpu_only()
  print(library.version())

Model files and voice references are separate from the wheel. The full
RealtimeTTS CPU server supplies the HTTP streaming, segmentation, language
detection, and recovery behavior on top of this binding.

SCOPE
This is a local release candidate, not an uploaded PyPI artifact. It has been
built and exercised on Linux x86_64 with the existing Q8 model pair. A public
release still requires clean-host acceptance on supported Linux, Windows, and
macOS CPU environments and platform-specific wheels.
