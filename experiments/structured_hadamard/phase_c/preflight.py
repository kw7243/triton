#!/usr/bin/env python3
"""Fail-closed staged preflight for the frozen Phase C selector run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
from typing import Mapping

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.structured_hadamard.phase_a.real_model import preflight_remote_audit as base
from experiments.structured_hadamard.phase_c import selector


SCHEMA_VERSION = "phase-c-selector-remote-preflight-v1"
RESULT_SCHEMA_VERSION = "phase-c-selector-remote-preflight-result-v1"
TOP_LEVEL_KEYS = {
    "schema_version", "owner", "project", "accepted_project", "dependency", "extension",
    "model", "data", "environment", "helpers", "execution", "ledger", "terminal_path",
    "audit_output", "plan", "phase_b", "policy",
}


class AuditError(RuntimeError):
    """The remote state differs from the exact Phase C clearance."""


def _bound_file(record: Mapping[str, object], *, owner_uid: int) -> dict[str, object]:
    path = base.require_regular(
        Path(str(record["path"])), owner_uid=owner_uid, mode=int(record["mode"]),
    )
    if path.stat().st_size != record["bytes"] or base.sha256_file(path) != record["sha256"]:
        raise AuditError(f"bound file identity differs: {path}")
    return {
        "path": str(path), "bytes": path.stat().st_size,
        "mode": stat.S_IMODE(path.stat().st_mode), "sha256": record["sha256"],
    }


def audit(clearance_path: Path, *, expected_clearance_sha256: str,
          expected_helper_sha256: str, allow_existing_output: bool = False) -> dict[str, object]:
    helper = Path(__file__).resolve(strict=True)
    if base.sha256_file(helper) != expected_helper_sha256:
        raise AuditError("executed Phase C preflight helper digest differs")
    clearance_path = base.require_regular(clearance_path, owner_uid=os.getuid(), mode=0o600)
    if base.sha256_file(clearance_path) != expected_clearance_sha256:
        raise AuditError("Phase C clearance digest differs")
    value = json.loads(clearance_path.read_text(encoding="utf-8"))
    if set(value) != TOP_LEVEL_KEYS or value.get("schema_version") != SCHEMA_VERSION:
        raise AuditError("Phase C clearance schema or top-level keys differ")
    owner = value["owner"]
    if owner != {"name": os.environ.get("USER"), "uid": os.getuid()}:
        raise AuditError("clearance owner differs")
    owner_uid = int(owner["uid"])
    ledger_path = base.require_regular(Path(value["ledger"]["path"]), owner_uid=owner_uid, mode=0o600)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger != value["ledger"]["expected_zero"]:
        raise AuditError(f"one-shot ledger is not zeroed: {ledger!r}")
    status_path = base.require_regular(Path(value["ledger"]["status_path"]), owner_uid=owner_uid, mode=0o600)
    if base.sha256_file(status_path) != value["ledger"]["status_sha256"]:
        raise AuditError("remote task-status bytes changed before launch")
    terminal_path = Path(value["terminal_path"])
    if terminal_path.exists():
        raise AuditError("remote terminal event already exists")
    audit_output = Path(value["audit_output"]).resolve()
    if audit_output.exists() and not allow_existing_output:
        raise AuditError("preflight output already exists")

    plan = _bound_file(value["plan"], owner_uid=owner_uid)
    if plan["sha256"] != "4c0f16b28a8c92aa2a70e163df36d2e5a6e98bdcd9c4aac77a0969491f307d45":
        raise AuditError("authoritative plan digest differs")
    phase_b = {name: _bound_file(record, owner_uid=owner_uid)
               for name, record in value["phase_b"].items()}
    if phase_b["map"]["sha256"] != selector.PHASE_B_MAP_SHA256:
        raise AuditError("accepted Phase B map digest differs")
    if phase_b["report"]["sha256"] != selector.PHASE_B_REPORT_SHA256:
        raise AuditError("accepted Phase B report digest differs")
    policy = _bound_file(value["policy"]["freeze"], owner_uid=owner_uid)
    freeze = json.loads(Path(policy["path"]).read_text(encoding="utf-8"))
    selector.validate_freeze(
        freeze, map_path=Path(phase_b["map"]["path"]),
        report_path=Path(phase_b["report"]["path"]),
    )
    if value["policy"]["freeze_sha256"] != policy["sha256"]:
        raise AuditError("policy freeze digest binding differs")

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "passed",
        "helper": {"path": str(helper), "sha256": expected_helper_sha256},
        "clearance": {"path": str(clearance_path), "sha256": expected_clearance_sha256},
        "owner": owner,
        "project": base.audit_git_stage(value["project"], owner_uid=owner_uid),
        "accepted_project": base.audit_git_stage(value["accepted_project"], owner_uid=owner_uid),
        "dependency": base.audit_dependency(value["dependency"], owner_uid=owner_uid),
        "extension": base.audit_extension(value["extension"], owner_uid=owner_uid),
        "model": base.audit_cache(value["model"], owner_uid=owner_uid, kind="model"),
        "data": base.audit_cache(value["data"], owner_uid=owner_uid, kind="data"),
        "environment": base.audit_environment(value["environment"], owner_uid=owner_uid),
        "helpers": base.audit_helpers(value["helpers"], owner_uid=owner_uid),
        "execution": base.audit_execution(value["execution"], owner_uid=owner_uid),
        "plan": plan,
        "phase_b": phase_b,
        "policy": {
            "freeze": policy,
            "source_rows_verified": sum(len(row["source_rows"])
                                        for row in freeze["measurements"].values()),
            "identity_to_measurement": freeze["identity_to_measurement"],
            "measurement_order": freeze["measurement_order"],
        },
        "ledger": {"path": str(ledger_path), "value": ledger,
                   "status_path": str(status_path), "status_sha256": value["ledger"]["status_sha256"]},
        "terminal_path": str(terminal_path),
    }
    return result


def _exclusive_json(path: Path, value: object) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clearance", type=Path, required=True)
    parser.add_argument("--expected-clearance-sha256", required=True)
    parser.add_argument("--expected-helper-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit(
            args.clearance, expected_clearance_sha256=args.expected_clearance_sha256,
            expected_helper_sha256=args.expected_helper_sha256,
        )
        expected = Path(json.loads(args.clearance.read_text(encoding="utf-8"))["audit_output"])
        if args.output.resolve() != expected.resolve():
            raise AuditError("preflight output argv differs from clearance")
        _exclusive_json(args.output, result)
    except (AuditError, base.AuditError, json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
