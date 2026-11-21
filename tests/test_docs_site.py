"""Documentation publication boundaries; optional docs tools stay out of runtime."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from xml.etree import ElementTree as ET
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

    def test_theme_source_links_do_not_point_into_excluded_site_content(self):
        self.assertEqual(
            self.render("[Partial](../overrides/partials/logo.html)"),
            "[Partial](https://github.com/example/project/blob/main/docs/overrides/partials/logo.html)",
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

    def test_logo_motion_is_limited_to_home_hero(self):
        root = Path(__file__).resolve().parents[1]
        home = (root / "docs/index.md").read_text()
        ns = {"s": "http://www.w3.org/2000/svg"}
        for theme, ink in (("light", "#163b2a"), ("dark", "#def3e5")):
            name = f"wtf-lockup-motion-{theme}.svg"
            self.assertIn(name, home)
            svg = ET.parse(root / "docs/assets/branding" / name).getroot()
            style = svg.find("s:style", ns).text
            self.assertIn(ink, style)
            self.assertIn("prefers-reduced-motion:no-preference", style)
            self.assertNotIn("prefers-color-scheme", style)
            self.assertIsNotNone(svg.find("s:g[@class='ribbon-static']", ns))
            self.assertEqual(len(svg.findall(".//s:use", ns)), 4)
            animation = svg.find(".//s:animate", ns)
            self.assertEqual(animation.get("attributeName"), "d")
            self.assertEqual(animation.get("dur"), "12s")
            frames = animation.get("values").split(";")
            self.assertEqual(frames[0], frames[-1])
            for tag in ("script", "image", "animateTransform", "clipPath", "mask"):
                self.assertEqual(svg.findall(f".//s:{tag}", ns), [])
        for name in (
            "README.md",
            "docs/overrides/partials/logo.html",
            "viewer/index.html",
        ):
            self.assertNotIn("wtf-lockup-motion", (root / name).read_text())

    def test_structural_figures_have_embedded_theme_palettes(self):
        root = Path(__file__).resolve().parents[1]
        folders = ("storage", "structures", "diagrams", "physics")
        ns = {"s": "http://www.w3.org/2000/svg"}
        for folder in folders:
            for path in (root / "docs/assets" / folder).glob("*.svg"):
                svg = ET.parse(path).getroot()
                style = "".join(e.text or "" for e in svg.findall(".//s:style", ns))
                self.assertIn("prefers-color-scheme:", style, str(path))
                self.assertIn("var(--", path.read_text(), str(path))
                self.assertEqual(svg.findall(".//s:script", ns), [])

    def test_legacy_source_urls_redirect_to_directory_pages(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = SimpleNamespace(
                site_dir=temporary, site_url="https://example.com/wtf/"
            )
            cases = (
                (
                    "configuration.md",
                    "configuration/",
                    "configuration/index.html",
                    "docs/configuration.html",
                    "../configuration/",
                ),
                (
                    "tutorials/01-configuration.md",
                    "tutorials/01-configuration/",
                    "tutorials/01-configuration/index.html",
                    "docs/tutorials/01-configuration.html",
                    "../../tutorials/01-configuration/",
                ),
                (
                    "installation.md",
                    "installation/",
                    "installation/index.html",
                    "INSTALL.html",
                    "installation/",
                ),
                ("index.md", "", "index.html", "README.html", "./"),
                (
                    "assets/README.md",
                    "assets/",
                    "assets/index.html",
                    "docs/assets/index.html",
                    "../../assets/",
                ),
            )
            for source, url, destination, alias, target in cases:
                page = SimpleNamespace(
                    file=SimpleNamespace(src_uri=source, dest_uri=destination), url=url
                )
                canonical = Path(temporary) / destination
                canonical.parent.mkdir(parents=True, exist_ok=True)
                canonical.write_text("Canonical page")
                self.assertEqual(
                    self.site.on_post_page("Canonical page", page, config),
                    "Canonical page",
                )
                text = (Path(temporary) / alias).read_text()
                self.assertIn(f'href="{target}"', text)
                self.assertIn("location.search + location.hash", text)
                self.assertIn('name="robots" content="noindex"', text)
                self.assertEqual(canonical.read_text(), "Canonical page")

    def test_built_link_audit_checks_prefix_anchors_and_picture_sources(self):
        from work.check_docs_links import check_site

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "guide").mkdir()
            (root / "guide/index.html").write_text('<h1 id="section">Guide</h1>')
            (root / "logo.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
            (root / "index.html").write_text(
                '<a href="guide/#section">Guide</a><source srcset="logo.svg">'
            )
            report = check_site(root, "https://example.com/wtf/")
            self.assertEqual(report["errors"], [])
            (root / "index.html").write_text(
                '<a href="/guide/">Wrong prefix</a><a href="guide/#missing">Bad section</a><source srcset="missing.svg">'
            )
            report = check_site(root, "https://example.com/wtf/")
            self.assertEqual(len(report["errors"]), 3)
            self.assertTrue(any("escapes site prefix" in e for e in report["errors"]))
            self.assertTrue(any("missing section" in e for e in report["errors"]))
            self.assertTrue(any("missing target" in e for e in report["errors"]))

    def test_navigation_and_directory_cover_all_primary_guides(self):
        from mkdocs.config import load_config

        root = Path(__file__).resolve().parents[1]
        config = load_config(str(root / "mkdocs.yml"))
        self.assertIn("navigation.indexes", config.theme["features"])

        def destinations(entries):
            for entry in entries:
                if isinstance(entry, str):
                    yield entry
                else:
                    for value in entry.values():
                        if isinstance(value, list):
                            self.assertTrue(
                                value, "Navigation groups need a landing page"
                            )
                            self.assertIsInstance(value[0], str)
                            self.assertEqual(Path(value[0]).name, "index.md")
                            yield from destinations(value)
                        else:
                            yield value

        navigation = set(destinations(config.nav))
        directory = (root / "docs/directory.md").read_text()
        directory_targets = {
            match[2].split("#", 1)[0] for match in self.site.LINKS.finditer(directory)
        }
        self.assertFalse(navigation - {"index.md", "directory.md"} - directory_targets)
        for target in navigation:
            if target not in self.site.EXTRA_PAGES:
                self.assertTrue((root / "docs" / target).is_file(), target)
        # Supporting catalogs and individual historical releases have their own
        # linked indexes; current user/contributor guides belong in navigation.
        for source in (root / "docs").glob("*.md"):
            self.assertIn(source.name, navigation, source.name)
        for source in (root / "docs/tutorials").glob("*.md"):
            self.assertIn(source.relative_to(root / "docs").as_posix(), navigation)

    def test_click_routes_exclude_navigation_images_and_proof_page_links(self):
        from work.docs_access import content_routes

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generator = '<meta name="generator" content="mkdocs-1.6.1">'
            for folder in ("guide", "access-map"):
                (root / folder).mkdir()
            (root / "index.html").write_text(
                generator + '<nav><a href="guide/">Hidden in navigation</a></nav>'
                '<article><img src="figure.svg"><a href="access-map/">Proof</a></article>'
            )
            (root / "access-map/index.html").write_text(
                generator
                + '<article><a href="../guide/">Only the proof links here</a></article>'
            )
            (root / "guide/index.html").write_text(
                generator + "<article><h1>Guide</h1></article>"
            )
            with self.assertRaisesRegex(ValueError, "guide/index.html"):
                content_routes(root, "https://example.com/wtf/")
            (root / "index.html").write_text(
                generator + '<article><a href="guide/">Read the guide</a>'
                '<img src="figure.svg"><a href="access-map/">Proof</a></article>'
            )
            pages, parents = content_routes(root, "https://example.com/wtf/")
            self.assertEqual(len(pages), 3)
            self.assertEqual(
                parents["guide/index.html"], ("index.html", "Read the guide")
            )
            self.assertNotIn("figure.svg", parents)

    def test_visual_catalog_covers_variants_and_checks_runtime_copies(self):
        from work.docs_access import visual_catalog, label

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = [
                "docs/assets/branding/favicon.png",
                "docs/assets/branding/favicon.svg",
                "viewer/branding/favicon.svg",
            ]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"same artwork")
            text = visual_catalog(names, root)
            for name in names[:2]:
                self.assertIn(f"]({name.removeprefix('docs/')})", text)
            self.assertIn("viewer/branding/favicon.svg", text)
            self.assertNotEqual(label(names[0]), label(names[1]))
            (root / names[-1]).write_bytes(b"different artwork")
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                visual_catalog(names, root)

    def test_generated_visual_previews_use_pairs_and_real_edit_source(self):
        from work.docs_access import visual_previews, preview_source

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = [
                "docs/assets/example.gif",
                "docs/assets/themed/example-light.webp",
                "docs/assets/branding/wtf-symbol-light.svg",
                "docs/assets/branding/wtf-symbol-dark.svg",
            ]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            previews = visual_previews(names, root)
            gif_source = preview_source(names[0])
            page = SimpleNamespace(
                file=SimpleNamespace(
                    src_uri=gif_source, dest_uri=gif_source[:-3] + "/index.html"
                )
            )
            rendered = self.site.on_page_markdown(
                previews[gif_source], page, self.config, None
            )
            self.assertIn('class="replay-theme-light"', rendered)
            self.assertIn('class="replay-theme-dark"', rendered)
            self.assertTrue(page.edit_url.endswith("/edit/main/work/docs_access.py"))
            logo = previews[preview_source(names[2])]
            self.assertIn("brand-on-light", logo)
            self.assertIn("brand-on-dark", logo)
            self.assertIn("Open original file", logo)

    def test_built_link_audit_rejects_orphan_pages_and_follows_index_hubs(self):
        from work.check_docs_links import check_site

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generator = '<meta name="generator" content="mkdocs-1.6.1">'
            (root / "index.html").write_text(
                generator + '<a href="catalog/">Catalog</a>'
            )
            (root / "catalog").mkdir()
            (root / "catalog/index.html").write_text(generator)
            (root / "figure-guide").mkdir()
            (root / "figure-guide/index.html").write_text(generator)
            (root / "404.html").write_text(generator)
            report = check_site(root, "https://example.com/wtf/")
            self.assertEqual(report["documents"], 3)
            self.assertEqual(report["reachable_documents"], 2)
            self.assertEqual(len(report["errors"]), 1)
            self.assertIn(
                "figure-guide/index.html: documentation page is unreachable",
                report["errors"][0],
            )
            (root / "catalog/index.html").write_text(
                generator + '<a href="../figure-guide/">Figure guide</a>'
            )
            report = check_site(root, "https://example.com/wtf/")
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["reachable_documents"], 3)

    def test_replay_pictures_follow_site_theme_and_nested_paths(self):
        page = SimpleNamespace(
            file=SimpleNamespace(
                src_uri="nested/guide.md", dest_uri="nested/guide/index.html"
            )
        )
        picture = (
            '<picture class="replay-preview">'
            '<source media="(prefers-color-scheme: light)" srcset="../assets/themed/demo-light.webp" />'
            '<source media="(prefers-color-scheme: dark)" srcset="../assets/themed/demo-dark.webp" />'
            '<img src="../assets/demo.gif" alt="Replay &amp; charts" />'
            "</picture>"
        )
        rendered = self.site.on_page_markdown(picture, page, self.config, None)
        self.assertIn('class="replay-theme-light"', rendered)
        self.assertIn('class="replay-theme-dark"', rendered)
        self.assertIn('src="../../assets/themed/demo-light.webp"', rendered)
        self.assertIn('alt="Replay &amp; charts"', rendered)
        self.assertNotIn("prefers-color-scheme", rendered)
        fenced = "```html\n" + picture + "\n```\n"
        self.assertEqual(
            self.site.on_page_markdown(fenced, page, self.config, None), fenced
        )

    def test_inspector_and_map_specimens_have_theme_support(self):
        root = Path(__file__).resolve().parents[1]
        ns = {"s": "http://www.w3.org/2000/svg"}
        for folder in ("inspector", "geography", "viewer"):
            for path in (root / "docs/assets" / folder).glob("*.svg"):
                svg = ET.parse(path).getroot()
                style = "".join(e.text or "" for e in svg.findall(".//s:style", ns))
                self.assertIn("prefers-color-scheme: light", style, str(path))

    def test_theme_styles_are_owned_by_the_outer_svg(self):
        root = Path(__file__).resolve().parents[1]
        ns = "{http://www.w3.org/2000/svg}"
        for path in (root / "docs/assets").rglob("*.svg"):
            document = ET.parse(path).getroot()

            def visit(element, viewport):
                if element.tag == ns + "svg":
                    viewport = element
                if element.tag == ns + "style" and "prefers-color-scheme" in (
                    element.text or ""
                ):
                    self.assertIs(
                        viewport,
                        document,
                        f"{path}: theme stylesheet is trapped inside a nested SVG viewport",
                    )
                for child in element:
                    visit(child, viewport)

            visit(document, document)

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
