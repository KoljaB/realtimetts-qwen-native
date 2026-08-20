from __future__ import annotations

import argparse

from ._diagnostics import (
    DEFAULT_MODEL,
    DEFAULT_QUANT,
    GGUF_REPO,
    run_doctor,
    run_prefetch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m qwentts_cpp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Diagnose native, CUDA, GPU, ABI, and model-cache setup")
    doctor.add_argument("--library", help="Explicit qwen.dll/libqwen.so path")
    doctor.add_argument("--model", default=DEFAULT_MODEL)
    doctor.add_argument("--quant", default=DEFAULT_QUANT)
    doctor.add_argument("--cache-dir")
    doctor.add_argument("--no-model-check", action="store_true")
    doctor.add_argument("--strict", action="store_true", help="Treat warnings as a failing exit status")
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    doctor.set_defaults(handler=run_doctor)

    prefetch = subparsers.add_parser(
        "prefetch",
        aliases=["download"],
        help="Download the talker and codec GGUF files for later offline use",
    )
    prefetch.add_argument("--model", default=DEFAULT_MODEL)
    prefetch.add_argument("--quant", default=DEFAULT_QUANT)
    prefetch.add_argument("--repo-id", default=GGUF_REPO)
    prefetch.add_argument("--cache-dir")
    prefetch.add_argument("--local-files-only", action="store_true")
    prefetch.set_defaults(handler=run_prefetch)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
