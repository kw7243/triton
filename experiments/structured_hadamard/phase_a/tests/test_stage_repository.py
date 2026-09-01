from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from experiments.structured_hadamard.phase_a.stage_repository import (DEFAULT_EXCLUDED_ROOT_DIRECTORIES, StageError,
                                                                      STAGE_MANIFEST_NAME,
                                                                      STAGE_MANIFEST_SCHEMA_VERSION,
                                                                      STAGE_SCHEMA_VERSION, stage_repository)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=root, text=True, stderr=subprocess.STDOUT).strip()


def _init_repository(path: Path) -> str:
    path.mkdir()
    _git(path, "init", "--quiet")
    _git(path, "config", "user.name", "Phase A Test")
    _git(path, "config", "user.email", "phase-a-test@example.invalid")
    (path / "tracked.txt").write_text("committed\n", encoding="utf-8")
    (path / "outputs").mkdir()
    (path / "outputs" / "tracked-input.cfg").write_text("committed tracked input\n", encoding="utf-8")
    (path / ".gitignore").write_text("ignored-input.cfg\noutputs/\n__pycache__/\n", encoding="utf-8")
    _git(path, "add", "tracked.txt", ".gitignore")
    _git(path, "add", "--force", "outputs/tracked-input.cfg")
    _git(path, "commit", "--quiet", "-m", "fixture")
    return _git(path, "rev-parse", "HEAD")


class RepositoryStageTest(unittest.TestCase):

    def _assert_complete_stage(self, source: Path, stage: Path, expected_head: str) -> None:
        staged, status = stage_repository(
            source,
            stage,
            command=("git", "cat-file", "-e", "HEAD^{commit}"),
        )
        self.assertEqual(status, 0)
        self.assertEqual(staged, stage)
        self.assertTrue((stage / ".git").is_dir())
        self.assertEqual(_git(stage, "rev-parse", "HEAD"), expected_head)
        self.assertEqual(_git(stage, "rev-parse", "--git-dir"), ".git")
        self.assertEqual(_git(stage, "rev-parse", "--git-common-dir"), ".git")
        self.assertEqual((stage / "tracked.txt").read_text(encoding="utf-8"), "dirty tracked\n")
        self.assertEqual((stage / "untracked.txt").read_text(encoding="utf-8"), "untracked input\n")
        self.assertEqual((stage / "ignored-input.cfg").read_text(encoding="utf-8"), "ignored input\n")
        self.assertEqual((stage / "outputs" / "tracked-input.cfg").read_text(encoding="utf-8"),
                         "dirty tracked excluded-name input\n")
        self.assertFalse((stage / "outputs" / "generated.bin").exists())
        self.assertFalse((stage / "package" / "__pycache__").exists())
        metadata = json.loads((stage / "REPRODUCIBILITY_METADATA.json").read_text(encoding="utf-8"))
        manifest = json.loads((stage / STAGE_MANIFEST_NAME).read_text(encoding="utf-8"))
        self.assertEqual(metadata["schema_version"], STAGE_SCHEMA_VERSION)
        self.assertEqual(manifest["schema_version"], STAGE_MANIFEST_SCHEMA_VERSION)
        self.assertEqual(metadata["source_head"], expected_head)
        self.assertGreaterEqual(metadata["tracked_and_untracked_entries"], 5)
        self.assertEqual(set(manifest["entries"]), {
            ".gitignore", "ignored-input.cfg", "outputs/tracked-input.cfg", "tracked.txt", "untracked.txt"
        })
        self.assertEqual(metadata["default_exclusions"]["root_directories"],
                         list(DEFAULT_EXCLUDED_ROOT_DIRECTORIES))

    def _add_dirty_inputs(self, source: Path) -> None:
        (source / "tracked.txt").write_text("dirty tracked\n", encoding="utf-8")
        (source / "outputs" / "tracked-input.cfg").write_text("dirty tracked excluded-name input\n",
                                                               encoding="utf-8")
        (source / "untracked.txt").write_text("untracked input\n", encoding="utf-8")
        (source / "ignored-input.cfg").write_text("ignored input\n", encoding="utf-8")
        (source / "outputs" / "generated.bin").write_bytes(b"generated output")
        (source / "package" / "__pycache__").mkdir(parents=True)
        (source / "package" / "__pycache__" / "module.pyc").write_bytes(b"generated cache")
        self.assertEqual(_git(source, "check-ignore", "ignored-input.cfg"), "ignored-input.cfg")
        self.assertEqual(_git(source, "check-ignore", "outputs/generated.bin"), "outputs/generated.bin")
        self.assertEqual(_git(source, "check-ignore", "package/__pycache__/module.pyc"),
                         "package/__pycache__/module.pyc")

    def test_stages_ordinary_repository_with_independent_git_and_dirty_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            stage = root / "stage"
            head = _init_repository(source)
            self._add_dirty_inputs(source)

            self._assert_complete_stage(source, stage, head)
            shutil.rmtree(source)
            self.assertEqual(_git(stage, "rev-parse", "HEAD"), head)
            self.assertIn("tracked.txt", _git(stage, "status", "--short"))

    def test_stages_linked_worktree_as_independent_ordinary_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary = root / "primary"
            linked = root / "linked"
            stage = root / "stage"
            head = _init_repository(primary)
            _git(primary, "worktree", "add", "--quiet", "-b", "fixture-linked", str(linked), head)
            self.assertTrue((linked / ".git").is_file())
            self._add_dirty_inputs(linked)

            self._assert_complete_stage(linked, stage, head)
            shutil.rmtree(linked)
            shutil.rmtree(primary)
            self.assertEqual(_git(stage, "cat-file", "-t", head), "commit")
            self.assertIn("untracked.txt", _git(stage, "status", "--short"))

    def test_refuses_destination_inside_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            _init_repository(source)
            with self.assertRaisesRegex(StageError, "outside the source"):
                stage_repository(source, source / "staging" / "bad")

    def test_refuses_tracked_path_beneath_symlink_without_writing_outside_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_parent = root / "source-parent"
            stage_parent = root / "stage-parent"
            source_parent.mkdir()
            stage_parent.mkdir()
            source = source_parent / "source"
            stage = stage_parent / "stage"
            _init_repository(source)
            (source / "nested").mkdir()
            (source / "nested" / "tracked.txt").write_text("committed nested input\n", encoding="utf-8")
            _git(source, "add", "nested/tracked.txt")
            _git(source, "commit", "--quiet", "-m", "nested fixture")
            shutil.rmtree(source / "nested")
            source_target = source_parent / "link-target"
            source_target.mkdir()
            (source_target / "tracked.txt").write_text("dirty linked input\n", encoding="utf-8")
            (source / "nested").symlink_to("../link-target")
            outside_stage = stage_parent / "link-target"
            outside_stage.mkdir()

            with self.assertRaisesRegex(StageError, "symlink ancestor"):
                stage_repository(source, stage)

            self.assertFalse(stage.exists())
            self.assertEqual(list(outside_stage.iterdir()), [])

    def test_stages_relative_symlink_to_included_content_without_source_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            stage = root / "stage"
            _init_repository(source)
            (source / "config-data").write_text("staged config\n", encoding="utf-8")
            (source / "config").symlink_to("config-data")

            stage_repository(source, stage)
            shutil.rmtree(source)

            self.assertTrue((stage / "config").is_symlink())
            self.assertEqual((stage / "config").readlink(), Path("config-data"))
            self.assertEqual((stage / "config").read_text(encoding="utf-8"), "staged config\n")

    def test_refuses_symlink_to_external_or_excluded_content(self):
        for target_kind in ("absolute-external", "relative-external", "excluded"):
            with self.subTest(target_kind=target_kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "source"
                stage = root / "stage"
                _init_repository(source)
                if target_kind == "absolute-external":
                    target = root / "external-config"
                    target.write_text("mutable external config\n", encoding="utf-8")
                elif target_kind == "relative-external":
                    (root / "external-config").write_text("mutable external config\n", encoding="utf-8")
                    target = Path("../external-config")
                else:
                    (source / ".cache").mkdir()
                    target = source / ".cache" / "excluded-config"
                    target.write_text("excluded config\n", encoding="utf-8")
                    target = Path(".cache/excluded-config")
                (source / "config").symlink_to(target)

                with self.assertRaisesRegex(StageError, "relative target|escapes the source|excluded from the stage"):
                    stage_repository(source, stage)

                self.assertFalse(stage.exists())


if __name__ == "__main__":
    unittest.main()
