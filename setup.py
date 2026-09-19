import os
import re
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


def _has_native_library(lib_dir: Path, base_name: str) -> bool:
    """Accept the platform-specific names emitted by qwentts.cpp builds."""
    if not lib_dir.is_dir():
        return False
    exact_names = {
        f"{base_name}.dll",
        f"lib{base_name}.dll",
        f"lib{base_name}.so",
        f"lib{base_name}.dylib",
    }
    versioned_linux_prefix = f"lib{base_name}.so."
    return any(
        path.is_file()
        and (path.name in exact_names or path.name.startswith(versioned_linux_prefix))
        for path in lib_dir.iterdir()
    )


def _validate_native_payload() -> None:
    lib_dir = _native_payload_dir()
    required = _has_native_library(lib_dir, "qwen")
    cpu_backend = _has_native_library(lib_dir, "ggml-cpu")
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
        explicit_platform = os.environ.get("QWENTTS_CPP_WHEEL_PLATFORM")
        if explicit_platform:
            if not re.fullmatch(r"macosx_\d+_\d+_(?:x86_64|arm64)", explicit_platform):
                raise ValueError(
                    "QWENTTS_CPP_WHEEL_PLATFORM must be a macOS x86_64 or arm64 wheel tag"
                )
            platform = explicit_platform
        return "py3", "none", platform


setup(
    distclass=BinaryDistribution,
    cmdclass={"bdist_wheel": PlatformWheel, "build_py": NativeBuildPy},
)
