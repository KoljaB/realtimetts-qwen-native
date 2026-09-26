"""Download only the exact release packages from TestPyPI; normal dependencies from PyPI."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
EXPECTED = {
"realtimetts-0.8.8-py3-none-any.whl":"2dd5bbc76f6d6a4333670626b40fda71c6327159999da7071b60d78f5757160f",
"realtimetts_qwen_native_cpu-0.4.0-py3-none-manylinux_2_35_x86_64.whl":"9ab601df762767f48ea157d1def60847c8f9ec3503c0593b4801d75e9c47af15",
"realtimetts_qwen_native_cpu-0.4.0-py3-none-win_amd64.whl":"9aedc3ca0a0714ef9d88f65e1c65380db6a352765950416a00f19f27bcdccca7",
"realtimetts_qwen_native_cpu-0.4.0-py3-none-macosx_13_0_x86_64.whl":"94a6338e7bd5343a204566a51a9851ebf708072bccfc512fc43077a942568705",
"realtimetts_qwen_native_cpu-0.4.0-py3-none-macosx_11_0_arm64.whl":"84e94ddb78981165ca15bd225c5bc37d426253063246b9225a7de5d56c86dc14",
}
wheelhouse = Path("testpypi-wheelhouse")
subprocess.run([sys.executable,"-m","pip","download","--no-deps","--only-binary=:all:",
                "--index-url","https://test.pypi.org/simple","--dest",str(wheelhouse),
                "realtimetts==0.8.8","realtimetts-qwen-native-cpu==0.4.0"],check=True)
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
