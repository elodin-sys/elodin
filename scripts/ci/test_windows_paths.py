#!/usr/bin/env python3
"""Ensure tracked repository paths can be checked out on Windows."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import PurePosixPath


WINDOWS_INVALID_CHARS = frozenset('<>:"|?*')
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def windows_path_error(path: str) -> str | None:
    for part in PurePosixPath(path).parts:
        if any(char in WINDOWS_INVALID_CHARS for char in part):
            return f"contains a Windows-invalid character: {path}"

        stem = part.rstrip(" .").split(".", 1)[0].upper()
        if stem in WINDOWS_RESERVED_NAMES:
            return f"uses a Windows-reserved path component: {path}"

        if part.endswith((" ", ".")):
            return f"ends a path component with a space or dot: {path}"

    return None


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

    def test_tracked_paths_are_windows_checkout_safe(self) -> None:
        invalid_paths = [
            error for path in tracked_paths() if (error := windows_path_error(path)) is not None
        ]
        self.assertEqual([], invalid_paths)


if __name__ == "__main__":
    unittest.main()
