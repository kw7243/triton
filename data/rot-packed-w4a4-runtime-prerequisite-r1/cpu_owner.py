#!/usr/bin/env python3
"""Exactly-once salloc/srun owner for the non-GPU packed-W4A4 build."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

from runtime_clearance import audit, sha256_file


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _exclusive_json(path: Path, value: object) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _advance(path: Path, key: str) -> dict[str, int]:
    value = _load(path)
    if value.get(key) != 0:
        raise RuntimeError(f"one-shot ledger refusal: {key}={value.get(key)!r}")
    value[key] = 1
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _exclusive_json(temporary, value)
    os.replace(temporary, path)
    return value


def _run_logged(argv: list[str], log: Path, *, cwd: Path, env: dict[str, str]) -> None:
    with log.open("xb") as stream:
        completed = subprocess.run(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {argv!r}; log={log}")


def _tool(clearance: dict[str, object], name: str) -> str:
    matches = [record["path"] for record in clearance["toolchain"] if record["name"] == name]
    if len(matches) != 1:
        raise RuntimeError(f"clearance must bind exactly one {name}")
    return matches[0]


def _payload(clearance_path: Path) -> int:
    clearance = _load(clearance_path)
    ledger_path = Path(clearance["ledger_path"])
    _advance(ledger_path, "payload_attempts")
    host = os.uname().nodename
    if "login" in host.lower():
        raise RuntimeError(f"build payload must not run on a login host: {host}")
    allocation = {
        "schema_version": "packed-w4a4-cpu-allocation-v1",
        "recorded_at_utc": _utc(),
        "host": host,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if not allocation["job_id"] or not allocation["partition"]:
        raise RuntimeError("payload lacks Slurm allocation identity")
    allocation_path = Path(clearance["build"]["allocation_json"])
    _exclusive_json(allocation_path, allocation)
    print("ALLOCATION " + json.dumps(allocation, sort_keys=True), flush=True)

    source_root = Path(clearance["source"]["root"])
    build = clearance["build"]
    environment = {**os.environ, **build["environment"], "PYTHONDONTWRITEBYTECODE": "1"}
    validation_log = Path(build["validation_log"])
    _run_logged(build["validation_argv"], validation_log, cwd=source_root, env=environment)

    started_ns = datetime.now().timestamp() * 1_000_000_000
    build_log = Path(build["build_log"])
    _run_logged(build["argv"], build_log, cwd=source_root, env=environment)
    result_lines = build_log.read_text(encoding="utf-8", errors="replace").splitlines()
    build_result = None
    for line in reversed(result_lines):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if candidate.get("schema_version") == "phase-a-w4a4-build-v1":
            build_result = candidate
            break
    if build_result is None:
        raise RuntimeError("builder log lacks the accepted JSON result")
    extension = Path(build_result["extension_path"]).resolve(strict=True)
    status = extension.stat()
    if not stat.S_ISREG(status.st_mode) or status.st_uid != os.getuid():
        raise RuntimeError("extension is not an owner-controlled regular file")
    if sha256_file(extension) != build_result["extension_sha256"]:
        raise RuntimeError("extension digest changed after the builder")
    build_directory = Path(build["directory"]).resolve(strict=True)
    if extension.parent != build_directory:
        raise RuntimeError("extension escaped the fresh accepted build directory")
    for path in build_directory.iterdir():
        if path.stat().st_ctime_ns < started_ns:
            raise RuntimeError(f"accepted build directory contains a pre-attempt file: {path}")
    ninja = build_directory / "build.ninja"
    ninja_text = ninja.read_text(encoding="utf-8")
    for forbidden in ("/tmp/quarot-phasea-build", "minimal-build-v2"):
        if forbidden in ninja_text:
            raise RuntimeError(f"accepted Ninja input references failed-login build material: {forbidden}")

    inspections = {}
    commands = {
        "ldd": [_tool(clearance, "ldd"), str(extension)],
        "readelf": [_tool(clearance, "readelf"), "-d", str(extension)],
        "nm": [_tool(clearance, "nm"), "-D", str(extension)],
        "cuobjdump_elf": [_tool(clearance, "cuobjdump"), "--list-elf", str(extension)],
        "cuobjdump_ptx": [_tool(clearance, "cuobjdump"), "--dump-ptx", str(extension)],
    }
    for name, argv in commands.items():
        completed = subprocess.run(
            argv, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=source_root, env=environment,
        )
        if completed.returncode:
            raise RuntimeError(f"artifact inspection failed ({name}): {completed.stdout}")
        inspections[name] = completed.stdout
        Path(build[f"{name}_log"]).write_text(completed.stdout, encoding="utf-8")
    if "not found" in inspections["ldd"]:
        raise RuntimeError("extension has an unresolved linked library")
    if "PyInit_phase_a_w4a4_cuda" not in inspections["nm"]:
        raise RuntimeError("extension lacks its Python initialization symbol")
    if "sm_80" not in inspections["cuobjdump_elf"] or "sm_86" not in inspections["cuobjdump_elf"]:
        raise RuntimeError("extension lacks exact SM80/SM86 native images")
    if ".target sm_80" not in inspections["cuobjdump_ptx"]:
        raise RuntimeError("extension lacks compute-80 PTX")

    import_script = (
        "import importlib.util, json, pathlib; "
        f"p=pathlib.Path({str(extension)!r}); "
        "s=importlib.util.spec_from_file_location('phase_a_w4a4_cuda', p); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "print(json.dumps({'module':m.__name__,'functions':[callable(getattr(m,n,None)) "
        "for n in ('matmul','sym_quant','sym_dequant')]}))"
    )
    import_log = Path(build["import_log"])
    _run_logged(
        [clearance["python"]["path"], "-c", import_script], import_log,
        cwd=source_root, env=environment,
    )
    imported = json.loads(import_log.read_text(encoding="utf-8").splitlines()[-1])
    if imported != {"module": "phase_a_w4a4_cuda", "functions": [True, True, True]}:
        raise RuntimeError(f"CPU-only extension import proof differs: {imported!r}")

    file_records = []
    for path in sorted(build_directory.iterdir()):
        if path.is_file():
            file_records.append({
                "path": str(path), "bytes": path.stat().st_size,
                "mode": stat.S_IMODE(path.stat().st_mode), "sha256": sha256_file(path),
            })
    result = {
        "schema_version": "packed-w4a4-cpu-build-result-v1",
        "status": "success",
        "recorded_at_utc": _utc(),
        "allocation": allocation,
        "build": build_result,
        "build_directory_files": file_records,
        "build_ninja_sha256": sha256_file(ninja),
        "validation_log_sha256": sha256_file(validation_log),
        "build_log_sha256": sha256_file(build_log),
        "import_log_sha256": sha256_file(import_log),
        "failed_login_build_contamination": False,
        "cuda_device_operation": False,
        "model_inference": False,
    }
    _exclusive_json(Path(build["result_json"]), result)
    print("RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return 0


def _allocated(clearance_path: Path) -> int:
    clearance = _load(clearance_path)
    ledger_path = Path(clearance["ledger_path"])
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("salloc owner lacks SLURM_JOB_ID")
    _advance(ledger_path, "srun_attempts")
    argv = [
        *clearance["scheduler"]["srun_argv"],
        clearance["python"]["path"], str(Path(__file__).resolve()), "--payload", str(clearance_path),
    ]
    return subprocess.run(argv, check=False).returncode


def _owner(clearance_path: Path) -> int:
    clearance = _load(clearance_path)
    terminal_path = Path(clearance["terminal_path"])
    ledger_path = Path(clearance["ledger_path"])
    owner_exit = 125
    failure = None
    try:
        audit_result = audit(clearance_path)
        audit_path = Path(clearance["build"]["clearance_audit_json"])
        _exclusive_json(audit_path, audit_result)
        _advance(ledger_path, "owner_attempts")
        _advance(ledger_path, "salloc_attempts")
        argv = [
            *clearance["scheduler"]["salloc_argv"],
            clearance["python"]["path"], str(Path(__file__).resolve()), "--allocated", str(clearance_path),
        ]
        owner_exit = subprocess.run(argv, check=False).returncode
        if owner_exit:
            failure = f"salloc_owner_exit={owner_exit}"
        return owner_exit
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        ledger = _advance(ledger_path, "terminal_events_fired")
        terminal = {
            "schema_version": "packed-w4a4-cpu-terminal-event-v1",
            "recorded_at_utc": _utc(),
            "owner_exit": owner_exit,
            "failure": failure,
            "ledger": ledger,
        }
        _exclusive_json(terminal_path, terminal)
        print("TERMINAL " + json.dumps(terminal, sort_keys=True), flush=True)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 2 or argv[0] not in {"--owner", "--allocated", "--payload"}:
        raise SystemExit("usage: cpu_owner.py (--owner|--allocated|--payload) CLEARANCE.json")
    clearance_path = Path(argv[1]).resolve(strict=True)
    if argv[0] == "--owner":
        return _owner(clearance_path)
    if argv[0] == "--allocated":
        return _allocated(clearance_path)
    return _payload(clearance_path)


if __name__ == "__main__":
    raise SystemExit(main())
