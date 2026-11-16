from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/lte/build_source_bundle.py"
SPEC = importlib.util.spec_from_file_location("build_source_bundle", SCRIPT)
assert SPEC and SPEC.loader
BUNDLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUNDLE)


class SourceBundleTest(unittest.TestCase):
    def init_repo(self, root: Path) -> None:
        subprocess.run(["git", "init", "-q", root], check=True)
        subprocess.run(["git", "-C", root, "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "-C", root, "config", "user.name", "Test"], check=True)

    def commit_all(self, root: Path) -> None:
        subprocess.run(["git", "-C", root, "add", "."], check=True)
        subprocess.run(["git", "-C", root, "commit", "-qm", "fixture"], check=True)

    def test_builds_nested_bundle_and_excludes_ignored_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            rmbench = base / "RMBench"
            xpolicylab = base / "XPolicyLab"
            output = base / "output"
            rmbench.mkdir()
            xpolicylab.mkdir()
            self.init_repo(rmbench)
            self.init_repo(xpolicylab)

            (rmbench / ".gitignore").write_text("scripts/lte/.env\nlogs/\n", encoding="utf-8")
            recipe = rmbench / "config/lte/example.yaml"
            recipe.parent.mkdir(parents=True)
            recipe.write_text("launcher: {}\nargs: {}\n", encoding="utf-8")
            (rmbench / "scripts/lte").mkdir(parents=True)
            (rmbench / "scripts/lte/.env").write_text("SECRET=value\n", encoding="utf-8")
            (rmbench / "logs").mkdir()
            (rmbench / "logs/run.log").write_text("ignored\n", encoding="utf-8")
            self.commit_all(rmbench)

            executable = xpolicylab / "policy/LTE_SmolVLA/install.sh"
            executable.parent.mkdir(parents=True)
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o755)
            (xpolicylab / "policy/LTE_SmolVLA/new.py").write_text("NEW = True\n", encoding="utf-8")
            (xpolicylab / ".gitignore").write_text("cache/\n", encoding="utf-8")
            self.commit_all(xpolicylab)
            (xpolicylab / "untracked.py").write_text("VALUE = 1\n", encoding="utf-8")
            os.symlink("/external/data", xpolicylab / "unsafe-link")

            result = subprocess.run(
                [
                    "python3",
                    SCRIPT,
                    "--rmbench-root",
                    rmbench,
                    "--xpolicylab-root",
                    xpolicylab,
                    "--recipe",
                    recipe,
                    "--output-dir",
                    output,
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            paths = json.loads(result.stdout)
            archive = Path(paths["archive"])
            checksum = Path(paths["checksum"])
            expected = checksum.read_text(encoding="utf-8").split()[0]
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), expected)

            with tempfile.TemporaryDirectory() as extraction:
                subprocess.run(
                    ["tar", "--zstd", "-xf", archive, "-C", extraction],
                    check=True,
                )
                release = Path(extraction) / "RMBench"
                self.assertTrue((release / "config/lte/example.yaml").is_file())
                self.assertTrue((release / "XPolicyLab/policy/LTE_SmolVLA/new.py").is_file())
                self.assertTrue((release / "XPolicyLab/untracked.py").is_file())
                self.assertFalse((release / "scripts/lte/.env").exists())
                self.assertFalse((release / "logs").exists())
                self.assertFalse((release / "XPolicyLab/unsafe-link").exists())
                mode = (release / "XPolicyLab/policy/LTE_SmolVLA/install.sh").stat().st_mode
                self.assertTrue(mode & 0o100)
                manifest = json.loads((release / "SOURCE_BUNDLE.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["bundle_id"], paths["bundle_id"])
                self.assertEqual(manifest["format"], "rmbench-source-bundle-v1")
                self.assertEqual(manifest["exclusions"][0]["type"], "excluded_symlink")

    def test_rejects_selected_credential_like_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.init_repo(root)
            (root / "service.pem").write_text("credential\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "credential-like"):
                BUNDLE.selected_paths(root)


if __name__ == "__main__":
    unittest.main()
