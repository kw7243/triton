"""Assemble or audit the exact cached Llama-3 packed-W4A4 preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime import prepare_model, sha256_file, verify_cached_snapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--extension-sha256", required=True)
    parser.add_argument("--quarot-root", type=Path, required=True)
    parser.add_argument("--layer", action="append", type=int, default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hash-weight-blobs", action="store_true")
    parser.add_argument("--load-model", action="store_true")
    return parser


def build_assembly(args: argparse.Namespace) -> dict[str, object]:
    extension = args.extension.resolve(strict=True)
    extension_sha256 = sha256_file(extension)
    if extension_sha256 != args.extension_sha256:
        raise ValueError(f"extension digest mismatch: {extension_sha256}")
    layers = tuple(args.layer) if args.layer else (0,)
    result = {
        "schema_version": "phase-a-llama3-w4a4-assembly-v1",
        "snapshot": verify_cached_snapshot(args.snapshot, hash_weight_blobs=args.hash_weight_blobs),
        "extension": {"path": str(extension), "sha256": extension_sha256},
        "quarot_root": str(args.quarot_root.resolve(strict=True)),
        "layers": list(layers),
        "replacement_site": "model.layers[*].mlp.down_proj",
        "attention": "transformers-standard-eager",
        "device": args.device,
        "model_loaded": False,
        "inference_run": False,
    }
    if args.load_model:
        prepare_model(
            args.snapshot, extension, extension_sha256, args.quarot_root,
            layers=layers, device=args.device,
        )
        result["model_loaded"] = True
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(build_assembly(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
