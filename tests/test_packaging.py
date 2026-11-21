"""Distribution asset contracts; installed-wheel smoke checks live in work/."""

import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET

from wtf import load_config
from wtf.simulation.geography import resolve_map
from wtf.resources import asset, preset_names
from wtf.server import argument_parser

ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_public_namespace_stays_lazy_and_uses_one_module_identity(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, wtf; "
                "assert 'wtf.experiment.runner' not in sys.modules; "
                "assert 'wtf.simulation.world' not in sys.modules; "
                "from wtf import Experiment; "
                "from wtf.experiment.runner import Experiment as implementation; "
                "assert Experiment is implementation; "
                "assert not any(n == 'ecosystem' or n.startswith('ecosystem.') for n in sys.modules)",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_source_checkout_rejects_older_interpreters_clearly(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.version_info = (3, 11); import wtf",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires Python 3.12+", result.stderr)

    def test_presets_load_outside_checkout_with_packaged_map_lookup(self):
        old = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                self.assertEqual(load_config("baseline")["name"], "baseline")
                self.assertEqual(
                    load_config("environments/baseline.json")["name"], "baseline"
                )
                for name in preset_names():
                    config = load_config(name)
                    self.assertIn("species", config)
                    if "terrain" in config:
                        self.assertIn("rows", resolve_map(config["terrain"]))
                self.assertEqual(
                    argument_parser().parse_args([]).results,
                    Path(temporary).resolve() / "results",
                )
            finally:
                os.chdir(old)

    def test_asset_manifest_covers_presets_viewer_maps_and_schema_only(self):
        paths = json.loads((ROOT / "packaging_assets.json").read_text())
        self.assertEqual(len(paths), len(set(paths)))
        self.assertIn("environment.schema.json", paths)
        self.assertIn("viewer/index.html", paths)
        self.assertIn("viewer/theme.mjs", paths)
        for relative in paths:
            self.assertTrue((ROOT / relative).is_file(), relative)
            self.assertNotIn("local", Path(relative).parts)
            self.assertNotIn("results", Path(relative).parts)
        for path in (ROOT / "viewer").iterdir():
            if path.suffix in (".js", ".mjs", ".css", ".html"):
                self.assertIn(path.relative_to(ROOT).as_posix(), paths)
        for relative in paths:
            if relative.startswith("environments/"):
                config = json.loads((ROOT / relative).read_text())
                if (
                    isinstance(config.get("terrain"), dict)
                    and "map_file" in config["terrain"]
                ):
                    self.assertIn(config["terrain"]["map_file"], paths)

    def test_branding_is_vector_based_and_ships_with_viewer(self):
        paths = json.loads((ROOT / "packaging_assets.json").read_text())
        for source in (ROOT / "viewer/branding").iterdir():
            self.assertIn(source.relative_to(ROOT).as_posix(), paths)
            self.assertEqual(
                source.read_bytes(),
                (ROOT / "docs/assets/branding" / source.name).read_bytes(),
            )
        for form in ("symbol", "lockup"):
            geometry = []
            for theme in ("light", "dark"):
                root = ET.parse(
                    ROOT / "docs/assets/branding" / f"wtf-{form}-{theme}.svg"
                ).getroot()
                self.assertIsNotNone(root.get("viewBox"))
                self.assertFalse(
                    any(
                        node.tag.rsplit("}", 1)[-1]
                        in ("image", "text", "script", "rect")
                        for node in root.iter()
                    )
                )
                geometry.append(
                    [
                        (p.get("d"), p.get("transform"))
                        for p in root.iter("{http://www.w3.org/2000/svg}path")
                    ]
                )
            self.assertTrue(geometry[0])
            self.assertEqual(*geometry)

    def test_linux_build_support_leaves_system_compiler_configuration_unchanged(self):
        from _build_support import configure_compiler

        before = dict(os.environ)
        with (
            patch("sys.platform", "linux"),
            patch(
                "pathlib.Path.is_file",
                side_effect=AssertionError("macOS tool lookup on Linux"),
            ),
        ):
            configure_compiler()
        self.assertEqual(dict(os.environ), before)

    def test_asset_paths_reject_traversal(self):
        for relative in (
            "../README.md",
            "viewer/../../LICENSE",
            "/etc/passwd",
            "viewer\\index.html",
        ):
            with self.subTest(path=relative), self.assertRaises(ValueError):
                asset(relative)
