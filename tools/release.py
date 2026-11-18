#!/usr/bin/env python3
"""Check and publish RealtimeTTS release artifacts.

The same file is used by RealtimeTTS and its native worker repositories.

    python tools/release.py check   --repository testpypi dist/*
    python tools/release.py publish --repository testpypi dist/*
    python tools/release.py verify-index --repository testpypi dist/*

``check`` refuses to continue unless:

- the working tree is clean and HEAD carries the tag ``v<version>``;
- that tag exists on the release remote and HEAD is on its release branch;
- every artifact belongs to one distribution and version;
- platform wheels were built by a successful CI run of HEAD (``--ci-run``);
- native wheels carry the build-info contract pinned in build_package.py;
- ``twine check --strict`` passes.

``publish`` runs ``check``, uploads the unchanged files with Twine and then
compares the package index's SHA-256 values with the local files.
Engine smoke tests on an installed artifact remain a manual release step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Release remote and branch per distribution and package index.
PROJECTS = {
    "realtimetts": {
        "testpypi": ("development", "master"),
        "pypi": ("origin", "master"),
    },
    "realtimetts-breeze-native": {
        "testpypi": ("origin", "main"),
        "pypi": ("origin", "main"),
        "native_backend": "vulkan",
    },
    "realtimetts-breeze-native-cuda12": {
        "testpypi": ("origin", "main"),
        "pypi": ("origin", "main"),
        "native_backend": "cuda12",
    },
    # realtimetts-qwen-native: one repository, the CUDA line on gpu, the CPU line on main.
    "realtimetts-qwen-native": {
        "testpypi": ("origin", "gpu"),
        "pypi": ("origin", "gpu"),
    },
    "realtimetts-qwen-native-cpu": {
        "testpypi": ("origin", "main"),
        "pypi": ("origin", "main"),
    },
}
INDEX_JSON = {
    "testpypi": "https://test.pypi.org/pypi/{name}/{version}/json",
    "pypi": "https://pypi.org/pypi/{name}/{version}/json",
}


class ReleaseError(RuntimeError):
    pass


def git(*args: str) -> str:
    result = subprocess.run(["git", "-C", str(ROOT), *args],
                            capture_output=True, text=True)
    if result.returncode:
        raise ReleaseError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def artifact_identity(path: Path) -> tuple[str, str, bool]:
    """Return (distribution, version, is_platform_wheel) from a file name."""
    name = path.name
    if name.endswith(".whl"):
        parts = name[:-4].split("-")
        if len(parts) not in (5, 6):
            raise ReleaseError(f"not a wheel file name: {name}")
        return normalize(parts[0]), parts[1], parts[-1] != "any"
    if name.endswith(".tar.gz"):
        stem = name[:-7]
        distribution, _, version = stem.rpartition("-")
        if not distribution or not version:
            raise ReleaseError(f"not an sdist file name: {name}")
        return normalize(distribution), version, False
    raise ReleaseError(f"unsupported artifact: {name}")


def check_artifacts(paths: list[Path]) -> tuple[str, str, bool]:
    if not paths:
        raise ReleaseError("no artifacts given")
    identities = {artifact_identity(path)[:2] for path in paths}
    if len(identities) != 1:
        raise ReleaseError(f"artifacts mix distributions or versions: {sorted(identities)}")
    distribution, version = identities.pop()
    if distribution not in PROJECTS:
        raise ReleaseError(f"no release configuration for {distribution}")
    if not any(path.name.endswith(".whl") for path in paths):
        raise ReleaseError("at least one wheel is required")
    platform = any(artifact_identity(path)[2] for path in paths)
    return distribution, version, platform


def check_source(version: str, remote: str, branch: str) -> str:
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ReleaseError("working tree is not clean")
    head = git("rev-parse", "HEAD")
    tag = f"v{version}"
    if git("rev-parse", f"{tag}^{{commit}}") != head:
        raise ReleaseError(f"tag {tag} does not point to HEAD")
    remote_refs = {}
    for line in git("ls-remote", remote, f"refs/heads/{branch}", f"refs/tags/{tag}",
                    f"refs/tags/{tag}^{{}}").splitlines():
        commit, ref = line.split("\t")
        remote_refs[ref] = commit
    remote_tag = remote_refs.get(f"refs/tags/{tag}^{{}}") or remote_refs.get(f"refs/tags/{tag}")
    if remote_tag != head:
        raise ReleaseError(f"tag {tag} is not pushed to {remote} at HEAD")
    remote_branch = remote_refs.get(f"refs/heads/{branch}")
    if not remote_branch:
        raise ReleaseError(f"{remote} has no branch {branch}")
    git("fetch", "--quiet", remote, branch)
    result = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                             head, remote_branch])
    if result.returncode:
        raise ReleaseError(f"HEAD is not on {remote}/{branch}; merge it first")
    return head


def check_ci_run(run_id: str, head: str) -> None:
    result = subprocess.run(
        ["gh", "run", "view", run_id, "--json", "headSha,status,conclusion"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode:
        raise ReleaseError(f"cannot read CI run {run_id}: {result.stderr.strip()}")
    run = json.loads(result.stdout)
    if run.get("headSha") != head:
        raise ReleaseError(f"CI run {run_id} built {run.get('headSha')}, not HEAD {head}")
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise ReleaseError(f"CI run {run_id} did not succeed")


def check_native_build_info(paths: list[Path], backend: str) -> None:
    sys.path.insert(0, str(ROOT))
    try:
        import build_package
    finally:
        sys.path.pop(0)
    profile = build_package.backend_profile(backend)
    expected = {
        "distribution": profile["package"],
        "source_revision": build_package.UPSTREAM_REVISION,
        "ggml_revision": build_package.GGML_REVISION,
        "control_patch_sha256": build_package.CONTROL_PATCH_SHA256,
    }
    for path in paths:
        if not path.name.endswith(".whl"):
            continue
        with zipfile.ZipFile(path) as archive:
            name = f"{profile['package']}/bin/build-info.json"
            try:
                info = json.loads(archive.read(name))
            except KeyError:
                raise ReleaseError(f"{path.name} has no {name}") from None
            for key, value in expected.items():
                if info.get(key) != value:
                    raise ReleaseError(f"{path.name}: build-info {key} differs from build_package.py")
            if info.get("cmake", {}).get("cache_flags") != profile["cache_flags"]:
                raise ReleaseError(f"{path.name}: build-info CMake flags differ from build_package.py")
            worker = f"{profile['package']}/bin/{info['native_binary']['name']}"
            if hashlib.sha256(archive.read(worker)).hexdigest() != info["native_binary"]["sha256"]:
                raise ReleaseError(f"{path.name}: native worker hash differs from build-info")


def twine(*args: str) -> None:
    if subprocess.run([sys.executable, "-m", "twine", *args]).returncode:
        raise ReleaseError(f"twine {args[0]} failed")


def check(paths: list[Path], repository: str, ci_run: str | None) -> tuple[str, str]:
    distribution, version, platform = check_artifacts(paths)
    remote, branch = PROJECTS[distribution][repository]
    head = check_source(version, remote, branch)
    if platform and not ci_run:
        raise ReleaseError("platform wheels need --ci-run with the CI run that built them")
    if ci_run:
        check_ci_run(ci_run, head)
    backend = PROJECTS[distribution].get("native_backend")
    if backend:
        check_native_build_info(paths, backend)
    twine("check", "--strict", *map(str, paths))
    for path in paths:
        print(f"{sha256(path)}  {path.name}")
    print(f"OK: {distribution} {version} at {head[:12]} on {remote}/{branch}")
    return distribution, version


def verify_index(paths: list[Path], repository: str, distribution: str, version: str,
                 attempts: int = 10) -> None:
    url = INDEX_JSON[repository].format(name=distribution, version=version)
    local = {path.name: sha256(path) for path in paths}
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                published = {item["filename"]: item["digests"]["sha256"]
                             for item in json.load(response)["urls"]}
        except (urllib.error.URLError, KeyError, ValueError):
            published = {}
        if all(name in published for name in local):
            break
        time.sleep(min(30, 2 ** attempt))
    for name, digest in local.items():
        if published.get(name) != digest:
            raise ReleaseError(f"{name}: index SHA-256 {published.get(name)} differs from local {digest}")
        print(f"index OK  {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("check", "publish", "verify-index"))
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--repository", choices=sorted(INDEX_JSON), required=True)
    parser.add_argument("--ci-run", help="GitHub Actions run ID that built platform wheels")
    args = parser.parse_args(argv)
    paths = [path.resolve() for path in args.artifacts]
    try:
        if args.command == "verify-index":
            distribution, version, _ = check_artifacts(paths)
        else:
            distribution, version = check(paths, args.repository, args.ci_run)
        if args.command == "publish":
            twine("upload", "--non-interactive", "--repository", args.repository, *map(str, paths))
        if args.command in ("publish", "verify-index"):
            verify_index(paths, args.repository, distribution, version)
    except ReleaseError as exc:
        print(f"release refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
