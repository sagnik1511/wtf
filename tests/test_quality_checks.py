"""Quality checks select real source safely and parse it without running bodies."""

# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

from pathlib import Path
import subprocess
import tempfile
import unittest

from work.check_quality import source_files, syntax_errors


class QualityChecksTests(unittest.TestCase):
    def test_inventory_preserves_spaces_ignores_generated_research_and_keeps_palette_inputs(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / ".gitignore").write_text(".cache/\nignored.py\n")
            for name in [
                "tracked.py",
                "new file.py",
                "ignored.py",
                ".cache/scratch.py",
                "docs/paper/analysis.py",
                "docs/assets/physics/explorer.html",
                "docs/assets/viewer/theme-palette.css",
                ".codex/agents/reviewer.toml",
            ]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("")
            subprocess.run(["git", "add", "tracked.py"], cwd=root, check=True)
            (root / "tracked.py").unlink()
            self.assertEqual(
                source_files(root),
                [
                    ".codex/agents/reviewer.toml",
                    "docs/assets/viewer/theme-palette.css",
                    "new file.py",
                ],
            )

    def test_source_symlink_cannot_point_outside_the_checkout(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
        ):
            root = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            target = Path(outside) / "source.py"
            target.write_text("pass\n")
            try:
                (root / "link.py").symlink_to(target)
            except OSError:
                self.skipTest("symlinks unavailable on this platform")
            with self.assertRaises(ValueError):
                source_files(root)

    def test_syntax_check_reports_invalid_files_without_executing_valid_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, text in [
                ("valid.py", "raise RuntimeError('must not execute')\n"),
                ("bad.py", "def broken(:\n"),
                ("bad.json", "{"),
                ("bad.toml", "[broken"),
            ]:
                (root / name).write_text(text)
            errors = syntax_errors(["valid.py", "bad.py", "bad.json", "bad.toml"], root)
            self.assertEqual(len(errors), 3)
            self.assertFalse(any("valid.py" in error for error in errors))

    def test_python_compile_checks_scope_errors_without_running_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for source in ("return 1\n", "break\n", "def f(x, x): pass\n"):
                (root / "invalid.py").write_text(source)
                self.assertEqual(len(syntax_errors(["invalid.py"], root)), 1)
