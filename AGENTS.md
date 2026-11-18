# RealtimeTTS Qwen native packages

One repository, two distributions:

| Distribution | Import | Release branch | Wheels built by |
| --- | --- | --- | --- |
| `realtimetts-qwen-native-cpu` | `qwentts_cpp_cpu` | `main` | `.github/workflows/cpu-wheels.yml` (Linux, Windows, macOS x86_64 and arm64; fresh install and real speech per platform) |
| `realtimetts-qwen-native` (CUDA 12) | `qwentts_cpp` | `gpu` | `.github/workflows/publish.yml` on the `gpu` branch (Linux x86_64 and aarch64, Windows x64) |

Both pin one qwentts.cpp revision of `KoljaB/qwentts.cpp` (`PINNED_QWENTTS_REF` in
`scripts/build_native.py`); that commit must be pushed there before a release build.

## Releases

Release wheels come from GitHub CI, never from a local build.

1. Bring the release state onto the release branch (fast-forward), tag it `v<version>`, and push both.
2. Run the wheel workflow for that tag and wait for it to succeed; download its wheels and sdist unchanged.
   The workflow records a GitHub build provenance attestation for each of these files.
3. Install the CI wheel and run a short synthesis on the target device.
4. Publish only with `python tools/release.py publish --repository testpypi --ci-run <run id> <artifacts>`,
   then the same files with `--repository pypi`. It refuses a dirty tree, an untagged or unpushed commit,
   a commit not on the release branch, a CI run that did not build HEAD, files without an attestation
   from that run and a failed `twine check`; after uploading it compares the index SHA-256 values with
   the local files.

Anyone can check that a released file was built by this repository's CI:
`gh attestation verify <file> --repo KoljaB/realtimetts-qwen-native` (GitHub CLI, signed in).

Never upload directly with Twine, never replace files under an existing version, and keep
`tools/release.py` identical to the copy in the RealtimeTTS repository.
