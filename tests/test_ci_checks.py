"""CI gates must reject broken sources and validate fresh artifacts only."""

import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from work import check_ci


class CICheckTests(unittest.TestCase):
    """Behavioral regressions for source gates and isolated artifact validation."""

    # Descriptive test names explain outcomes; repeating them adds no detail.
    # pylint: disable=missing-function-docstring

    @unittest.skipUnless(
        importlib.util.find_spec("black"), "Optional dev-only Black unavailable"
    )
    def test_black_catches_unformatted_untracked_python(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            source = root / "new example.py"
            source.write_text("answer=1\n")
            self.assertNotEqual(check_ci.check_black(root), 0)
            source.write_text("answer = 1\n")
            self.assertEqual(check_ci.check_black(root), 0)

    @unittest.skipUnless(
        importlib.util.find_spec("pylint"), "Optional dev-only Pylint unavailable"
    )
    def test_pylint_catches_errors_and_warnings_in_untracked_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            package = root / "wtf"
            package.mkdir()
            source = package / "__init__.py"
            source.write_text("print(undefined_runtime_name)\n")
            self.assertNotEqual(check_ci.check_pylint(root), 0)
            source.write_text('"""Fixture module."""\n\nVALUE = 1\n')
            self.assertEqual(check_ci.check_pylint(root), 0)
            # Full lint must also reject a warning in a new non-package tool.
            (root / "new_tool.py").write_text('"""Fixture tool."""\n\nimport os\n')
            self.assertNotEqual(check_ci.check_pylint(root), 0)

    def test_build_is_portable_ignores_stale_artifacts_and_checks_both_formats(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stale = root / ".cache/ci-builds/stale.whl"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"preserve")
            built = []

            def run(command, _root, env=None):
                if "build" in command:
                    self.assertEqual(env["WTF_BUILD_NATIVE"], "0")
                    self.assertEqual(env["WTF_BUILD_ENGINE"], "0")
                    output = Path(command[command.index("--outdir") + 1])
                    built.extend([output / "new.whl", output / "new.tar.gz"])
                    for path in built:
                        path.write_bytes(b"fixture")
                else:
                    self.assertIn("--strict", command)
                    self.assertEqual(command[-2:], sorted(map(str, built)))
                return 0

            with (
                patch.dict(
                    os.environ, {"WTF_BUILD_NATIVE": "1", "WTF_BUILD_ENGINE": "1"}
                ),
                patch.object(check_ci, "run_command", side_effect=run),
            ):
                self.assertEqual(check_ci.check_build(root), 0)
            self.assertEqual(stale.read_bytes(), b"preserve")
            self.assertTrue(all(not path.exists() for path in built))

    def test_missing_sdist_fails_before_metadata_validation(self):
        with tempfile.TemporaryDirectory() as folder:

            def run(command, _root, _env=None):
                output = Path(command[command.index("--outdir") + 1])
                (output / "only.whl").touch()
                return 0

            with patch.object(check_ci, "run_command", side_effect=run) as runner:
                self.assertEqual(check_ci.check_build(Path(folder)), 1)
                self.assertEqual(runner.call_count, 1)

    def test_failed_gate_does_not_hide_other_results_or_return_success(self):
        with (
            patch.object(check_ci, "check_black", return_value=1),
            patch.object(check_ci, "check_pylint", return_value=0) as lint,
            patch.object(check_ci, "check_build", return_value=0) as build,
        ):
            self.assertEqual(check_ci.main([]), 1)
            lint.assert_called_once()
            build.assert_called_once()
