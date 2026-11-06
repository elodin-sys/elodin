"""Exercise CI cache cleanup with disposable checkouts and a fake disk report."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("prepare_cargo_cache.sh")


class CargoCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        script = self.root / "scripts/ci/prepare_cargo_cache.sh"
        script.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPT, script)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        df = self.bin / "df"
        df.write_text(
            "#!/bin/sh\n"
            'if [ -e target/debug/sentinel ] || [ "${STILL_FULL:-}" = 1 ]; then\n'
            "  free=1\n"
            "else\n"
            "  free=999999999\n"
            "fi\n"
            'echo "Filesystem 1024-blocks Used Available Capacity Mounted"\n'
            'echo "test 1000000000 1 $free 1% /"\n'
        )
        df.chmod(0o755)

    def run_script(self, **env):
        return subprocess.run(
            ["bash", "scripts/ci/prepare_cargo_cache.sh"],
            cwd=self.root,
            env={
                **os.environ,
                "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
                "BUILDKITE": "true",
                "BUILDKITE_BUILD_CHECKOUT_PATH": str(self.root),
                **env,
            },
            capture_output=True,
            text=True,
        )

    def artifact(self, path):
        file = self.root / path / "sentinel"
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
        return file

    def test_low_space_cleans_only_own_build_artifacts(self):
        paths = ["target/debug", "target/release", "fsw/sensor-fw/target/debug"]
        artifacts = [self.artifact(path) for path in paths]
        preserved = self.artifact("target/shells")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(not file.exists() for file in artifacts))
        self.assertTrue(preserved.exists())

    def test_warm_cache_is_kept_when_space_is_available(self):
        artifact = self.artifact("target/release")
        self.assertEqual(self.run_script().returncode, 0)
        self.assertTrue(artifact.exists())

    def test_developer_checkout_is_never_cleaned(self):
        artifact = self.artifact("target/debug")
        self.assertEqual(self.run_script(BUILDKITE="").returncode, 0)
        self.assertTrue(artifact.exists())

    def test_wrong_checkout_is_rejected(self):
        artifact = self.artifact("target/debug")
        result = self.run_script(BUILDKITE_BUILD_CHECKOUT_PATH="/")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(artifact.exists())

    def test_symlink_is_not_followed(self):
        outside = self.root / "outside"
        outside.mkdir()
        artifact = self.artifact("outside/debug")
        (self.root / "target").symlink_to(outside, target_is_directory=True)
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(artifact.exists())

    def test_full_host_fails_before_starting_a_build(self):
        self.artifact("target/debug")
        result = self.run_script(STILL_FULL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("host needs disk maintenance", result.stderr)
