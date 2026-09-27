"""Download only the exact release packages from TestPyPI; normal dependencies from PyPI."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
EXPECTED = {
    "realtimetts-0.8.10-py3-none-any.whl": "1359aba65bc2389b2d1ccd2b3a04330af4d5032b64b3463d5f0cda35a9968676",
    "realtimetts_qwen_native_cpu-0.4.2-py3-none-win_amd64.whl": "93e47ad4deca65ac78bcf96dea2a22315d89503c4383a46609cb40487dfd1e6f",
    "realtimetts_qwen_native_cpu-0.4.2-py3-none-manylinux_2_35_x86_64.whl": "eb04e35d2961a585abdadf9dc3d164a8f265d32bd49ccd38d46b4c07ca294e84",
    "realtimetts_qwen_native_cpu-0.4.2-py3-none-macosx_13_0_x86_64.whl": "0964058281c50531a78023441eddfcda3fd6c679b327f210b2b47c83e3fc399b",
    "realtimetts_qwen_native_cpu-0.4.2-py3-none-macosx_11_0_arm64.whl": "4aef0209b79b46d40202d196f920bcede0c020f609d9df38b1abf6d706f7ef5a"
}
wheelhouse = Path("testpypi-wheelhouse")
subprocess.run([sys.executable,"-m","pip","download","--no-deps","--only-binary=:all:",
                "--index-url","https://test.pypi.org/simple","--dest",str(wheelhouse),
                "realtimetts==0.8.10","realtimetts-qwen-native-cpu==0.4.2"],check=True)
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
