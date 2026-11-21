"""Documentation publication boundaries; optional docs tools stay out of runtime."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(
    importlib.util.find_spec("mkdocs"), "Install requirements-docs.txt"
)
class DocumentationSiteTests(unittest.TestCase):
    def setUp(self):
        from work import docs_site

        self.site = docs_site
        self.config = SimpleNamespace(repo_url="https://github.com/example/project/")

    def render(self, text, destination="nested/guide.md"):
        page = SimpleNamespace(file=SimpleNamespace(src_uri=destination))
        return self.site.on_page_markdown(text, page, self.config, None)

    def test_links_keep_documentation_local_and_source_on_github(self):
        self.assertEqual(
            self.render(
                "[Install](../../INSTALL.md#local-python-installation) "
                "[Home](../../README.md) [Code](../../ecosystem/world.py#L12) "
                "![Figure](../assets/viewer/brain-views.svg)"
            ),
            "[Install](../installation.md#local-python-installation) "
            "[Home](../index.md) "
            "[Code](https://github.com/example/project/blob/main/ecosystem/world.py#L12) "
            "![Figure](../assets/viewer/brain-views.svg)",
        )

    def test_generated_installation_resolves_links_from_original_source(self):
        page = SimpleNamespace(file=SimpleNamespace(src_uri="installation.md"))
        self.assertEqual(
            self.site.on_page_markdown(
                "[Tutorial](docs/tutorial.md)", page, self.config, None
            ),
            "[Tutorial](tutorial.md)",
        )
        self.assertEqual(
            page.edit_url, "https://github.com/example/project/edit/main/INSTALL.md"
        )

    def test_fenced_examples_are_not_rewritten(self):
        for fence in ("```", "~~~~"):
            text = f"{fence}md\n[Install](../../INSTALL.md)\n{fence}\n"
            self.assertEqual(self.render(text), text)
        text = "[Web](https://example.com/a?q=1#anchor) [Section](#section)"
        self.assertEqual(self.render(text), text)

    def test_ignored_research_and_symlink_files_are_never_published(self):
        from mkdocs.config.defaults import MkDocsConfig
        from mkdocs.structure.files import File, Files

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            docs = root / "docs"
            docs.mkdir()
            (root / "INSTALL.md").write_text("# Install")
            (root / "examples").mkdir()
            (root / "examples/README.md").write_text("# Example")
            paths = ["guide.md", "ignored.md", "paper/draft.md", "experiments/raw.csv"]
            for name in paths:
                path = docs / name
                path.parent.mkdir(exist_ok=True)
                path.write_text("local or public")
            (docs / "linked.md").symlink_to(root / "INSTALL.md")
            paths.append("linked.md")
            files = Files([File(p, str(docs), str(root / "site"), True) for p in paths])
            files.append(
                File("theme.css", str(root / "theme"), str(root / "site"), True)
            )
            config = MkDocsConfig()
            config.site_dir = str(root / "site")
            config.plugins = SimpleNamespace(_current_plugin=None)
            visible = "\0".join("docs/" + p for p in paths).encode()
            with (
                patch.object(self.site, "ROOT", root),
                patch.object(
                    self.site.subprocess,
                    "check_output",
                    side_effect=[visible, b"docs/ignored.md\0"],
                ),
            ):
                result = self.site.on_files(files, config)
            self.assertEqual(
                {file.src_uri for file in result},
                {"guide.md", "theme.css", "installation.md", "examples.md"},
            )
            self.assertEqual(
                result.get_file_from_path("installation.md").content_string, "# Install"
            )
