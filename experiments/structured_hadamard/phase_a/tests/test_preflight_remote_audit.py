from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

from experiments.structured_hadamard.phase_a.real_model import preflight_remote_audit as audit
from experiments.structured_hadamard.phase_a.stage_repository import stage_repository


def _run(argv, cwd):
    subprocess.run(argv, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _git_repo(root: Path, name: str = "payload.txt") -> tuple[str, str]:
    root.mkdir()
    _run(("git", "init", "-q"), root)
    _run(("git", "config", "user.name", "Fixture"), root)
    _run(("git", "config", "user.email", "fixture@example.invalid"), root)
    (root / name).write_text("fixture\n", encoding="utf-8")
    _run(("git", "add", name), root)
    _run(("git", "commit", "-qm", "fixture"), root)
    head = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=root, text=True).strip()
    tree = subprocess.check_output(("git", "rev-parse", "HEAD^{tree}"), cwd=root, text=True).strip()
    return head, tree


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_record(stage: Path, immutable: bool) -> dict[str, object]:
    manifest = stage / "REPRODUCIBILITY_MANIFEST.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "manifest_path": str(manifest),
        "manifest_file_sha256": _sha(manifest),
        "entries_sha256": audit.canonical_sha256(payload["entries"]),
        "immutable": immutable,
    }


def _file_record(root: Path, name: str) -> dict[str, object]:
    path = (root / name).resolve(strict=True)
    return {"name": name, "resolved_path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}


class RemotePreflightAuditTest(unittest.TestCase):

    def test_complete_fixture_passes_without_importing_scientific_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            head, tree = _git_repo(source)
            project = root / "project-stage"
            stage_repository(source, project)
            accepted = root / "accepted-stage"
            stage_repository(source, accepted)

            dependency = root / "dependency"
            dep_head, _ = _git_repo(dependency, "dep.txt")
            cutlass = dependency / "cutlass"
            cutlass_head, _ = _git_repo(cutlass, "cutlass.txt")
            dep_entries = {
                "dep.txt": audit._entry_identity(dependency / "dep.txt", owner_uid=os.getuid(), immutable=False),
                "cutlass/cutlass.txt": audit._entry_identity(
                    cutlass / "cutlass.txt", owner_uid=os.getuid(), immutable=False,
                ),
            }
            dep_manifest = root / "dependency-manifest.json"
            dep_manifest.write_text(json.dumps({"entries": dep_entries}, sort_keys=True), encoding="utf-8")

            extension = root / "phase_a_w4a4_cuda.so"
            extension.write_bytes(b"accepted-extension")
            extension.chmod(0o755)
            cuobjdump = root / "cuobjdump"
            cuobjdump.write_text(
                "#!/bin/sh\ncase \"$1\" in --list-elf) echo 'sm_80 sm_86';; --dump-ptx) echo '.target sm_80';; esac\n",
                encoding="utf-8",
            )
            cuobjdump.chmod(0o755)

            model = root / "model-revision"
            model.mkdir()
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            data = root / "data-revision"
            data.mkdir()
            (data / "test.arrow").write_bytes(b"arrow")

            site = root / "site-packages"
            package = site / "fixture_package"
            dist_info = site / "fixture_package-1.0.dist-info"
            package.mkdir(parents=True)
            dist_info.mkdir()
            (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
            (dist_info / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: fixture-package\nVersion: 1.0\n", encoding="utf-8",
            )
            with (dist_info / "RECORD").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["fixture_package/__init__.py", "", ""])
                writer.writerow(["fixture_package-1.0.dist-info/METADATA", "", ""])
                writer.writerow(["fixture_package-1.0.dist-info/RECORD", "", ""])

            output_parent = root / "outputs"
            output_parent.mkdir(mode=0o750)
            ledger = root / "ledger.json"
            zero = {"owner_attempts": 0, "salloc_attempts": 0, "srun_attempts": 0,
                    "driver_attempts": 0, "terminal_events_fired": 0}
            ledger.write_text(json.dumps(zero), encoding="utf-8")
            ledger.chmod(0o600)
            helper = Path(audit.__file__).resolve()
            driver_argv = [sys.executable, str(project / "payload.txt")]
            salloc_argv = ["/usr/bin/salloc", "--account=fixture", "--time=00:01:00"]
            srun_argv = ["/usr/bin/srun", "--pty", "--nodes=1"]

            for stage in (project, accepted):
                for control in ("REPRODUCIBILITY_MANIFEST.json", "REPRODUCIBILITY_METADATA.json", "payload.txt"):
                    (stage / control).chmod(0o444)

            project_manifest = _manifest_record(project, True)
            accepted_manifest = _manifest_record(accepted, True)
            python = root / "python"
            shutil.copy2(Path(sys.executable).resolve(strict=True), python)
            environment_record = {
                "python": {"path": str(python), "sha256": _sha(python),
                           "mode": stat.S_IMODE(python.stat().st_mode)},
                "site_paths": [str(site)],
                "allowed_roots": [str(root)],
                "packages": [{"name": "fixture-package", "version": "1.0", "exact_bytes": True}],
            }
            observed_environment = audit.audit_environment(environment_record, owner_uid=os.getuid())
            environment_record["packages"][0]["byte_manifest_sha256"] = (
                observed_environment["packages"][0]["byte_manifest_sha256"]
            )
            clearance = {
                "schema_version": audit.SCHEMA_VERSION,
                "owner": {"name": os.environ.get("USER"), "uid": os.getuid()},
                "project": {
                    "root": str(project), "head": head, "tree": tree, "required_ancestor": head,
                    "manifest": project_manifest,
                    "metadata_path": str(project / "REPRODUCIBILITY_METADATA.json"),
                    "metadata_sha256": _sha(project / "REPRODUCIBILITY_METADATA.json"),
                },
                "accepted_project": {
                    "root": str(accepted), "head": head, "tree": tree, "required_ancestor": head,
                    "manifest": accepted_manifest,
                    "metadata_path": str(accepted / "REPRODUCIBILITY_METADATA.json"),
                    "metadata_sha256": _sha(accepted / "REPRODUCIBILITY_METADATA.json"),
                },
                "dependency": {
                    "root": str(dependency), "head": dep_head,
                    "cutlass_root": str(cutlass), "cutlass_head": cutlass_head,
                    "manifest": {
                        "manifest_path": str(dep_manifest),
                        "manifest_file_sha256": _sha(dep_manifest),
                        "entries_sha256": audit.canonical_sha256(dep_entries), "immutable": False,
                    },
                },
                "extension": {
                    "path": str(extension), "bytes": extension.stat().st_size, "sha256": _sha(extension),
                    "mode": 0o755, "cuobjdump_path": str(cuobjdump), "cuobjdump_sha256": _sha(cuobjdump),
                    "cuobjdump_mode": 0o755, "sass_targets": ["sm_80", "sm_86"],
                    "ptx_markers": [".target sm_80"], "forbidden_arch_markers": ["sm_90"],
                },
                "model": {"name": "fixture/model", "revision": model.name, "root": str(model),
                          "files": [_file_record(model, "config.json")]},
                "data": {"name": "fixture/data", "configuration": "test", "revision": data.name,
                         "root": str(data), "files": [_file_record(data, "test.arrow")]},
                "environment": environment_record,
                "helpers": [{"name": "preflight", "path": str(helper), "sha256": _sha(helper),
                             "mode": stat.S_IMODE(helper.stat().st_mode)}],
                "execution": {
                    "driver_argv": driver_argv, "salloc_argv": salloc_argv, "srun_argv": srun_argv,
                    "argv_sha256": audit.canonical_sha256({
                        "driver_argv": driver_argv, "salloc_argv": salloc_argv, "srun_argv": srun_argv,
                    }),
                    "duration_estimate": "fixture",
                    "output_parent": str(output_parent), "output_parent_mode": 0o750,
                    "output_directory": str(output_parent / "new"),
                },
                "ledger": {"path": str(ledger), "expected_zero": zero},
                "terminal_path": str(root / "terminal.json"),
                "audit_output": str(root / "audit.json"),
            }
            clearance_path = root / "clearance.json"
            clearance_path.write_text(json.dumps(clearance), encoding="utf-8")
            clearance_path.chmod(0o600)
            before = "datasets" in sys.modules
            result = audit.audit(
                clearance_path, expected_clearance_sha256=_sha(clearance_path),
                expected_helper_sha256=_sha(helper),
            )
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["dependency"]["head"], dep_head)
            self.assertFalse(result["environment"]["scientific_packages_imported"])
            self.assertEqual("datasets" in sys.modules, before)
            for stage in (project, accepted):
                for path in stage.rglob("*"):
                    if not path.is_symlink():
                        path.chmod(path.stat().st_mode | stat.S_IWUSR)

    def test_file_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "input"
            path.write_text("before", encoding="utf-8")
            record = _file_record(root, "input")
            path.write_text("after", encoding="utf-8")
            with self.assertRaisesRegex(audit.AuditError, "identity differs"):
                audit._audit_file_record(root, record, owner_uid=os.getuid())


if __name__ == "__main__":
    unittest.main()
