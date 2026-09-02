#!/usr/bin/env python3
"""Independently verify a copied Phase B scientific result directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.structured_hadamard.phase_b import analysis, schema


class VerificationError(ValueError):
    """Copied Phase B evidence is incomplete or internally inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def verify(root: Path, *, cache_root: Path | None = None) -> dict[str, object]:
    scientific = root / "scientific-run"
    run_state = root / "run-state"
    results = _json(scientific / "results.json")
    rows = _jsonl(scientific / "map-rows.jsonl")
    raw_timings = _jsonl(scientific / "raw-timings.jsonl")
    validations = _jsonl(scientific / "six-site-validation.jsonl")
    selection = _json(scientific / "selection-freeze.json")
    ranking = _json(scientific / "ranking-stability-analysis.json")

    schema.validate_rows(rows)
    _require(len(raw_timings) == 32 * 4 * 2 * 100, "raw timing count differs")
    raw_keys = {
        (row["layer"], row["site"], row["transform"], row["metric"], row["repetition"])
        for row in raw_timings
    }
    _require(len(raw_keys) == len(raw_timings), "raw timing keys are not unique")
    _require(
        {row["metric"] for row in raw_timings}
        == {"transform", "affected_packed_w4a4_layer"},
        "raw timing metric set differs",
    )

    for name, expected in results["artifact_hashes"].items():
        _require(_sha256(scientific / name) == expected, f"artifact hash differs: {name}")

    frozen = analysis.freeze_selection(rows)
    _require(selection == frozen, "selection freeze does not reproduce from map rows")
    _require(selection["frozen_before_ppl"] is True, "selection was not frozen before PPL")
    primary = analysis.proxy_agreement(selection, validations)
    proxy_validation = ranking["proxy_validation"]
    _require(proxy_validation["primary"] == primary, "primary proxy agreement differs")
    _require(proxy_validation["active_proxy"] == analysis.PRIMARY_PROXY,
             "this result did not retain the primary proxy")
    _require(proxy_validation["stronger"] is None and proxy_validation["switch_count"] == 0,
             "unexpected stronger-proxy state")
    for row in validations:
        _require(row["replacement"] == "Hfull -> H32 at this site only",
                 "six-site replacement differs")
        _require(row["all_other_down_projection_choices"] == "Hfull",
                 "six-site control choices differ")

    scores = analysis.primary_scores(rows)["mean"]
    decision = analysis.literal_decision(
        rows,
        scores,
        stability_rho=float(selection["stability"]["spearman_rho"]),
        proxy_validated=bool(primary["agrees"]),
    )
    _require(ranking["active_scores"] == {str(key): value for key, value in scores.items()},
             "active proxy scores differ")
    _require(ranking["decision"] == decision, "ranking decision does not reproduce")
    _require(results["literal_decision_B"] == decision, "result Decision B differs")
    _require(results["phase_c_entered"] is False, "Phase C was entered")

    for record in results["correctness"].values():
        _require(float(record["inverse_max_abs"]) <= 1.0e-12,
                 "inverse transform tolerance failed")
        _require(float(record["equivalence_max_abs"]) <= 1.0e-12,
                 "folding equivalence tolerance failed")

    expected_ledger = {
        "owner_attempts": 1,
        "salloc_attempts": 1,
        "srun_attempts": 1,
        "driver_attempts": 1,
        "terminal_events_fired": 1,
    }
    ledger = _json(run_state / "owner-ledger.json")
    terminal = _json(run_state / "remote-terminal.json")
    _require(ledger == expected_ledger, "final one-shot ledger differs")
    _require(terminal["owner_exit"] == 0 and terminal["failure"] is None,
             "remote terminal was not successful")
    _require(terminal["ledger"] == expected_ledger, "terminal ledger differs")

    cache_manifest = _json(scientific / "site-cache-manifest.json")
    entries = cache_manifest["entries"]
    _require(len(entries) == 32, "site-cache manifest must contain 32 entries")
    _require(sum(int(entry["bytes"]) for entry in entries) == cache_manifest["total_bytes"],
             "site-cache byte total differs")
    cache_hashes_verified = False
    if cache_root is not None:
        for entry in entries:
            local_path = cache_root / Path(entry["path"]).name
            _require(local_path.stat().st_size == entry["bytes"],
                     f"site-cache byte size differs: {local_path.name}")
            _require(_sha256(local_path) == entry["sha256"],
                     f"site-cache hash differs: {local_path.name}")
        cache_hashes_verified = True

    return {
        "schema_version": "phase-b-result-verification-v1",
        "status": "passed",
        "root": str(root.resolve()),
        "source_commit": results["source_commit"],
        "map_rows": len(rows),
        "raw_timing_rows": len(raw_timings),
        "six_site_rows": len(validations),
        "site_cache_entries": len(entries),
        "site_cache_bytes": cache_manifest["total_bytes"],
        "site_cache_hashes_verified": cache_hashes_verified,
        "ranking_stability_spearman_rho": selection["stability"]["spearman_rho"],
        "six_site_spearman_rho": primary["six_site_spearman_rho"],
        "proxy_switch_count": proxy_validation["switch_count"],
        "decision": decision["conclusion"],
        "ledger": ledger,
        "input_hashes": {
            "results.json": _sha256(scientific / "results.json"),
            "map-rows.jsonl": _sha256(scientific / "map-rows.jsonl"),
            "selection-freeze.json": _sha256(scientific / "selection-freeze.json"),
            "six-site-validation.jsonl": _sha256(scientific / "six-site-validation.jsonl"),
            "ranking-stability-analysis.json": _sha256(
                scientific / "ranking-stability-analysis.json"
            ),
            "owner-ledger.json": _sha256(run_state / "owner-ledger.json"),
            "remote-terminal.json": _sha256(run_state / "remote-terminal.json"),
        },
    }


def _write_exclusive(path: Path, value: Mapping[str, object]) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(args.root.resolve(strict=True), cache_root=args.cache_root)
    if args.output:
        _write_exclusive(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
