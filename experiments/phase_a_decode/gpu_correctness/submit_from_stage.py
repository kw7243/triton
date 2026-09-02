#!/usr/bin/env python3
"""Consume a frozen request exactly once; never prepare, select, or retry it."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from prepare_clean_rerun import submit_once


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    print(submit_once(args.manifest, args.request, args.ledger))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
