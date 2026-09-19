import os
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # pragma: no cover - older setuptools fallback
    from wheel.bdist_wheel import bdist_wheel


class BinaryDistribution(Distribution):
    """Force a platform wheel even though the Python wrapper is pure Python."""

    def has_ext_modules(self):
        return True


def _native_payload_dir() -> Path:
    return Path(__file__).resolve().parent / "src" / "qwentts_cpp_cpu" / "lib"


def _validate_native_payload() -> None:
    lib_dir = _native_payload_dir()
    required = (
        any(path.name.startswith("libqwen.") for path in lib_dir.iterdir() if path.is_file())
        if lib_dir.is_dir()
        else False
    )
    cpu_backend = (
        any(path.name.startswith("libggml-cpu.") for path in lib_dir.iterdir() if path.is_file())
        if lib_dir.is_dir()
        else False
    )
    if not (required and cpu_backend):
        raise RuntimeError(
            "The CPU wheel requires bundled native libraries in "
            f"{lib_dir}. Run scripts/build_native.py --backend cpu --clean "
            "from the pinned qwentts.cpp checkout before building a wheel. "
            "Building a wheel from the sdist without native payload is refused."
        )


class NativeBuildPy(build_py):
    """Refuse a wheel that would install Python code without native CPU libraries."""

    def run(self):
        _validate_native_payload()
        super().run()


class PlatformWheel(bdist_wheel):
    """Build a py3-none-<platform> wheel for ctypes-bundled shared libraries."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False
        build_tag = os.environ.get("QWENTTS_CPP_WHEEL_BUILD_TAG")
        if build_tag and not self.build_number:
            self.build_number = build_tag

    def get_tag(self):
        _python, _abi, platform = super().get_tag()
        return "py3", "none", platform


setup(
    distclass=BinaryDistribution,
    cmdclass={"bdist_wheel": PlatformWheel, "build_py": NativeBuildPy},
)
