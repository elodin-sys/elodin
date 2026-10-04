#!/usr/bin/env python3
"""Ensure tracked repository paths can be checked out on Windows."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from windows_paths import (
    sanitize_export_filenames,
    windows_path_error,
    windows_safe_name,
)


def tracked_paths() -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files"],
        text=True,
    )
    return output.splitlines()


class WindowsPathTests(unittest.TestCase):
    def test_windows_path_error_detects_invalid_paths(self) -> None:
        self.assertIsNotNone(windows_path_error("scripts/foo_>_bar.csv"))
        self.assertIsNotNone(windows_path_error("scripts/aux/data.csv"))
        self.assertIsNotNone(windows_path_error("scripts/file."))
        self.assertIsNone(windows_path_error("scripts/foo_to_bar.csv"))

    def test_windows_safe_name_maps_edge_arrows_to_to(self) -> None:
        self.assertEqual(windows_safe_name("css_0_>_sat.css_edge.csv"), "css_0_to_sat.css_edge.csv")
        self.assertEqual(windows_safe_name("sat_>_rw_1.rw_edge.csv"), "sat_to_rw_1.rw_edge.csv")
        self.assertEqual(windows_safe_name("ore_sat.world_pos.csv"), "ore_sat.world_pos.csv")

    def test_sanitize_export_filenames_renames_arrow_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "css_0_>_sat.css_edge.csv").write_text("edge\n", encoding="utf-8")
            (root / "ore_sat.world_pos.csv").write_text("pos\n", encoding="utf-8")
            renamed = sanitize_export_filenames(root)
            self.assertEqual(len(renamed), 1)
            self.assertTrue((root / "css_0_to_sat.css_edge.csv").is_file())
            self.assertFalse((root / "css_0_>_sat.css_edge.csv").exists())
            self.assertTrue((root / "ore_sat.world_pos.csv").is_file())

    def test_tracked_paths_are_windows_checkout_safe(self) -> None:
        invalid_paths = [
            error for path in tracked_paths() if (error := windows_path_error(path)) is not None
        ]
        self.assertEqual([], invalid_paths)

    def test_baseline_filenames_are_windows_checkout_safe(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        root = repo / "scripts" / "ci" / "baseline"
        invalid_paths = [
            error
            for path in root.rglob("*")
            if path.is_file()
            and (error := windows_path_error(path.relative_to(repo).as_posix())) is not None
        ]
        self.assertEqual([], invalid_paths)


if __name__ == "__main__":
    unittest.main()
