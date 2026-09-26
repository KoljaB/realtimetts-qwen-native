"""Download only the exact release packages from TestPyPI; normal dependencies from PyPI."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
EXPECTED = {
"realtimetts-0.8.9-py3-none-any.whl":"6690f6652d69fa82b24a4b72f787ae970b62d05fdfed1c826ab9c30374ad73cf",
"realtimetts_qwen_native_cpu-0.4.1-py3-none-manylinux_2_35_x86_64.whl":"d6f431393ab2fe2fda2d074a2752ca72e89f25e8392df0c3825b89c7ada29978",
"realtimetts_qwen_native_cpu-0.4.1-py3-none-win_amd64.whl":"5db18d6e2052ae3fd1a300c549eb9af4ce3ef2a0eb16555bf71d927302957ff3",
"realtimetts_qwen_native_cpu-0.4.1-py3-none-macosx_13_0_x86_64.whl":"6f355005a955f83e49472983c07a9f7aa8e4e8879637f6e524ba8831a23a3036",
"realtimetts_qwen_native_cpu-0.4.1-py3-none-macosx_11_0_arm64.whl":"b7474a4a8e6789cb26c67f50e59ff994ff088cc5a62f773443df5f46d858f57c",
}
wheelhouse = Path("testpypi-wheelhouse")
subprocess.run([sys.executable,"-m","pip","download","--no-deps","--only-binary=:all:",
                "--index-url","https://test.pypi.org/simple","--dest",str(wheelhouse),
                "realtimetts==0.8.9","realtimetts-qwen-native-cpu==0.4.1"],check=True)
wheels = sorted(wheelhouse.glob("*.whl"))
assert len(wheels) == 2, wheels
hashes = {}
for wheel in wheels:
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert digest == EXPECTED[wheel.name], wheel
    hashes[wheel.name] = digest
framework = next(p for p in wheels if p.name.startswith("realtimetts-"))
native = next(p for p in wheels if p.name.startswith("realtimetts_qwen_native_cpu-"))
subprocess.run([sys.executable,"-m","pip","install",str(native),
                str(framework)+"[qwen-cpu-server]","httpx"],check=True)
subprocess.run([sys.executable,"-m","pip","check"],check=True)
Path("downloaded-artifact-hashes.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
