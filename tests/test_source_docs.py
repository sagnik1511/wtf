"""Source-reference scope, parser coverage, graphs and cache invalidation."""

# Black owns code layout; long literal payloads, URLs and regression names retain
# their exact text.
# pylint: disable=line-too-long


# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from work import build_source_docs as source


class SourceReferenceTests(unittest.TestCase):
    def test_inventory_excludes_ignored_tests_assets_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            names = [
                "wtf/module.py",
                "wtf/native/kernel.c",
                "viewer/module.mjs",
                "work/tool.py",
                "work/private.py",
                "wtf/assets/copy.js",
                "tests/test_module.py",
                "docs/paper/runner.py",
                "results/run.py",
            ]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("")
            (root / "wtf/link.py").symlink_to(root / "work/private.py")
            (root / "wtf/alias").symlink_to(root / "work", target_is_directory=True)
            names += ["wtf/link.py", "wtf/alias/private.py"]
            with patch.object(
                source.subprocess,
                "check_output",
                side_effect=["\0".join(names).encode(), b"work/private.py\0"],
            ):
                selected = source.source_files(root)
            self.assertEqual(
                selected,
                [
                    "viewer/module.mjs",
                    "work/tool.py",
                    "wtf/module.py",
                    "wtf/native/kernel.c",
                ],
            )

    def test_missing_component_is_an_error_even_if_other_files_parse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "file.xml").write_text(
                '<doxygen><compounddef kind="file"><location file="wtf/a.py"/></compounddef></doxygen>'
            )
            with self.assertRaisesRegex(ValueError, "wtf/b.py"):
                source.summarize(root, ["wtf/a.py", "wtf/b.py"])

    def test_missing_system_tool_has_actionable_error(self):
        with patch.object(source.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Install doxygen"):
                source.build()

    def test_every_source_page_is_reachable_with_or_without_internal_crawler(self):

        # Keep imports lazy so optional tools and training dependencies load only on
        # this path.
        # pylint: disable=import-outside-toplevel

        from work.check_docs_links import check_site

        for has_crawler in (False, True):
            with (
                self.subTest(has_crawler=has_crawler),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                api = root / "source-api"
                api.mkdir()
                (root / "index.html").write_text(
                    '<meta name="generator" content="zensical-0.1.3">'
                    '<a href="source-api/index.html">Source browser</a>'
                )
                (root / "source-graphs").mkdir()
                (root / "source-graphs/index.html").write_text("<h1>Source guide</h1>")
                marker = '<meta name="generator" content="Doxygen 1.9.8">'
                (api / "index.html").write_text(
                    marker
                    + '<title>Source home</title><a href="source-index.html">Complete source page index</a>'
                    + ('<a href="doxygen_crawl.html"></a>' if has_crawler else "")
                )
                (api / "class_component.html").write_text(
                    marker + "<title>Component &amp; helpers</title>"
                )
                (api / "component_source.html").write_text(
                    marker + "<title>Source listing</title>"
                )
                (api / "nested").mkdir()
                (api / "nested/members.html").write_text(
                    marker + "<title>Members</title>"
                )
                if has_crawler:
                    (api / "doxygen_crawl.html").write_text(
                        '<a href="missing.html#incorrect">Broken internal link</a>'
                    )
                before = check_site(root, "https://example.com/wtf/")
                self.assertTrue(before["errors"])
                source.write_page_index(api)
                report = check_site(root, "https://example.com/wtf/")
                self.assertEqual(report["errors"], [])
                self.assertEqual(report["reachable_documents"], report["documents"])
                index = (api / "source-index.html").read_text()
                self.assertIn('href="nested/members.html"', index)
                self.assertIn("Component &amp; helpers", index)

    @unittest.skipUnless(
        shutil.which("doxygen") and shutil.which("dot"), "Install Doxygen and Graphviz"
    )
    def test_real_python_javascript_native_graphs_and_cached_rebuild(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "wtf").mkdir()
            (root / "viewer").mkdir()
            shutil.copy2(source.ROOT / "Doxyfile", root / "Doxyfile")
            names = ["wtf/component.py", "viewer/component.mjs", "wtf/kernel.c"]
            (root / "wtf/component.py").write_text(
                "def leaf():\n    return 1\n\ndef caller():\n    return leaf()\n"
            )
            (root / "viewer/component.mjs").write_text(
                "export function jsLeaf() { return 1; }\nexport function jsCaller() { return jsLeaf(); }\n"
            )
            (root / "wtf/kernel.c").write_text(
                "static int c_leaf(void) { return 1; }\nint c_caller(void) { return c_leaf(); }\n"
            )
            # The builder hashes its own maintained implementation too.
            (root / "work").mkdir()
            shutil.copy2(Path(source.__file__), root / "work/build_source_docs.py")
            with patch.object(source, "source_files", return_value=names):
                output = source.build(root)
                manifest = root / ".cache/source-reference/manifest.json"
                initial = json.loads(manifest.read_text())
                self.assertEqual(initial["files"], 3)
                self.assertGreaterEqual(initial["call_graphs"], 3)
                self.assertGreaterEqual(initial["caller_graphs"], 3)
                stamp = manifest.stat().st_mtime_ns
                self.assertEqual(source.build(root), output)
                self.assertEqual(manifest.stat().st_mtime_ns, stamp)
                (root / "wtf/component.py").write_text(
                    "def leaf():\n    return 2\n\ndef caller():\n    return leaf()\n"
                )
                source.build(root)
                self.assertNotEqual(
                    json.loads(manifest.read_text())["fingerprint"],
                    initial["fingerprint"],
                )


if __name__ == "__main__":
    unittest.main()
