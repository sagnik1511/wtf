"""Catalog public visuals and prove click-only routes from the documentation home."""

import argparse
from collections import defaultdict, deque
from html.parser import HTMLParser
import html
import json
from pathlib import Path
import posixpath
import re
import subprocess
import sys
from urllib.parse import unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1]
MEDIA = {
    ".svg",
    ".png",
    ".gif",
    ".webp",
    ".jpg",
    ".jpeg",
    ".avif",
    ".mp4",
    ".webm",
    ".pdf",
}
CATALOG = "visuals.md"
ACCESS_MAP = "access-map.md"
GROUPS = {
    "branding": "Logos and theme variants",
    "diagrams": "Architecture and learning flows",
    "geography": "Map diagrams",
    "inspector": "Agent inspector",
    "physics": "Physics and interactive explorer",
    "storage": "Replay, model and checkpoint structures",
    "structures": "Runtime and brain structures",
    "terrains": "Terrain playback clips",
    "themed": "Light-theme playback variants",
    "viewer": "Viewer objects and appearance",
    "": "Playback clips and GitHub banner",
}


def inventory(root=ROOT):
    """Use the same Git-visible boundary as the site; ignore local research/media."""
    arguments = []
    visible = set(
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                *arguments,
            ],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    ignored = set(
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--ignored",
                "--exclude-standard",
                *arguments,
            ],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    result = []
    for name in sorted(visible - ignored - {""}):
        path = root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or name.startswith(("docs/paper/", "docs/experiments/"))
        ):
            continue
        if path.suffix.lower() in MEDIA or name in {
            "docs/assets/physics/explorer.html",
            "docs/assets/physics/explorer.fragment.html",
        }:
            if not name.startswith(("docs/assets/", "viewer/branding/")):
                raise ValueError(
                    f"Document the publication location for this visual: {name}"
                )
            result.append(name)
    return result


def label(name):
    text = Path(name).stem.replace("-", " ").replace("_", " ").capitalize()
    text = re.sub(
        r"\b(wtf|ppo|grpo|mappo|dqn|svg|rng)\b",
        lambda match: match[0].upper(),
        text,
        flags=re.I,
    )
    if name.endswith("explorer.fragment.html"):
        text = "Explorer editable source fragment"
    return f"{text} ({Path(name).suffix[1:].upper()})"


def preview_source(name):
    relative = name.removeprefix("docs/assets/")
    return "visual-previews/" + relative.replace(".", "-") + ".md"


def visual_previews(names, root=ROOT):
    """Small generated MkDocs wrappers retain the selected site palette."""
    pages = {}
    for name in names:
        if not name.startswith("docs/"):
            continue
        source = preview_source(name)

        def href(asset):
            return posixpath.relpath(
                asset.removeprefix("docs/"), posixpath.dirname(source)
            )

        title = label(name)
        body = [
            f"# {title}",
            "",
            f"[All visuals]({href('docs/visuals.md')}) · [Homepage click paths]({href('docs/access-map.md')})",
            "",
            "This preview retains the documentation light/dark controls. The original",
            "file link below opens the saved asset directly, using its own theme rules.",
            "",
        ]
        original = name
        stem = Path(name).stem.removesuffix("-light")
        if name.endswith(".gif") or (
            name.startswith("docs/assets/themed/") and name.endswith(".webp")
        ):
            light = f"docs/assets/themed/{stem}-light.webp"
            dark = next(
                (
                    f"docs/assets/{prefix}{stem}.gif"
                    for prefix in ("", "terrains/")
                    if (root / f"docs/assets/{prefix}{stem}.gif").is_file()
                ),
                None,
            )
            if dark is None or not (root / light).is_file():
                raise ValueError(
                    f"Playback preview needs matching light/dark assets: {name}"
                )
            body.extend(
                [
                    '<picture class="replay-preview">',
                    f'  <source media="(prefers-color-scheme: light)" srcset="{href(light)}" />',
                    f'  <source media="(prefers-color-scheme: dark)" srcset="{href(dark)}" />',
                    f'  <img src="{href(dark)}" alt="{html.escape(title, quote=True)}" />',
                    "</picture>",
                    "",
                    "The preview selects the matching recorded export; no pixel inversion is used.",
                    "",
                ]
            )
        elif name.startswith("docs/assets/branding/") and any(
            x in name for x in ("-light.", "-dark.")
        ):
            for theme in ("light", "dark"):
                variant = name.replace("-light.", f"-{theme}.").replace(
                    "-dark.", f"-{theme}."
                )
                body.append(
                    f'<img class="brand-on-{theme} visual-brand-preview" src="../{href(variant)}" alt="{html.escape(title, quote=True)}" />'
                )
            body.extend(
                [
                    "",
                    "Logo ink follows the selected surface. The original link keeps the named variant.",
                    "",
                ]
            )
        elif name.endswith(".html"):
            standalone = "docs/assets/physics/explorer.html"
            body.extend(
                [
                    f'<iframe class="visual-explorer" src="../{href(standalone)}" title="Interactive physics explorer" loading="lazy"></iframe>',
                    "",
                    "The full explorer runs inside this preview. The fragment file is editable source.",
                    "",
                ]
            )
        else:
            extra = (
                "{ .brand-sample-light }"
                if name.endswith("branding/favicon.png")
                else ""
            )
            body.extend([f"![{title}]({href(name)}){extra}", ""])
            if name.endswith("github-banner.png"):
                body.extend(
                    [
                        "The GitHub banner is a fixed screenshot; its recorded pixels do not change with the interface palette.",
                        "",
                    ]
                )
            if name.endswith(("viewer/themes.svg", "viewer/day-night.svg")):
                body.extend(
                    [
                        "Daylight boards use the selected viewer palette. Interface comparisons put the selected palette first, with a labeled alternate.",
                        "",
                    ]
                )
        body.extend([f"[Open original file]({href(original)})", ""])
        pages[source] = "\n".join(body)
    return pages


def visual_catalog(names, root=ROOT):
    groups = defaultdict(list)
    duplicates = []
    for name in names:
        if name.startswith("viewer/"):
            master = "docs/assets/branding/" + Path(name).name
            if (
                master not in names
                or (root / name).read_bytes() != (root / master).read_bytes()
            ):
                raise ValueError(
                    f"Runtime branding has no byte-identical documentation master: {name}"
                )
            duplicates.append((name, master))
        else:
            relative = name.removeprefix("docs/assets/")
            groups[posixpath.dirname(relative)].append(name)
    lines = [
        "# All visuals",
        "",
        "[Documentation home](index.md) · [Homepage click paths](access-map.md) · [Visual guide](visual-guide.md)",
        "",
        "Every public visual file has a theme-aware preview and an original-file link,",
        "including variants, fallbacks and the standalone physics explorer. Choose a category",
        "from this page's contents, then click the visual name to retain site theme controls;",
        "the guides in the [asset source catalog](assets/README.md) explain their context.",
        "",
        "These links do not load every animation at once. Fixed screenshots and explicitly",
        "themed exports retain their saved pixels; adaptive SVGs follow the browser theme",
        "when opened directly. The editable explorer fragment is source, not the full app.",
        "",
    ]
    for group, heading in GROUPS.items():
        if group not in groups:
            continue
        lines.extend(
            [
                f"## {heading}",
                "",
                "| Theme-aware preview | Original file / variant |",
                "| --- | --- |",
            ]
        )
        for name in groups.pop(group):
            relative = name.removeprefix("docs/")
            lines.append(
                f"| [{label(name)}]({preview_source(name)}) | [{Path(name).name}]({relative}) |"
            )
        lines.append("")
    if groups:
        raise ValueError(f"Add descriptive visual categories for: {sorted(groups)}")
    lines.extend(
        [
            "## Viewer branding copies",
            "",
            "The installed viewer ships the following byte-identical copies. Each link opens",
            "its documentation master; the runtime copy path is shown for source maintenance.",
            "",
            "| Runtime file | Open identical visual |",
            "| --- | --- |",
        ]
    )
    for name, master in duplicates:
        lines.append(
            f"| `{name}` | [{Path(master).name}]({master.removeprefix('docs/')}) |"
        )
    lines.extend(
        [
            "",
            "Local paper figures, experiment artifacts, ignored authoring files and generated",
            "build/cache files are excluded from the public site.",
            "",
        ]
    )
    return "\n".join(lines)


class ArticleLinks(HTMLParser):
    """Only visible article anchors count as clicks, never search/nav/image loads."""

    def __init__(self):
        super().__init__()
        self.article = False
        self.document = False
        self.heading = False
        self.title = ""
        self.links = []
        self.href = None
        self.text = ""

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "meta" and attrs.get("name") == "generator":
            self.document = attrs.get("content", "").startswith("mkdocs-")
        if tag == "article":
            self.article = True
        if self.article and tag == "h1":
            self.heading = True
        if self.article and tag == "a":
            self.href, self.text = attrs.get("href"), ""

    def handle_data(self, data):
        if self.href is not None:
            self.text += data
        if self.heading:
            self.title += data

    def handle_endtag(self, tag):
        if tag == "article":
            self.article = False
        if tag == "h1":
            self.heading = False
        if tag == "a" and self.href is not None:
            text = " ".join(self.text.split())
            if text and text != "¶":
                self.links.append((self.href, text))
            self.href = None


def destination(origin, href, site_url):
    base = site_url.rstrip("/") + "/"
    location = urlsplit(base)
    target = urlsplit(urljoin(base + origin, href))
    if (
        target.scheme not in {"http", "https"}
        or target.netloc != location.netloc
        or not target.path.startswith(location.path)
    ):
        return None
    path = unquote(target.path[len(location.path) :])
    if not path or path.endswith("/"):
        path += "index.html"
    return path


def content_routes(site_dir, site_url):
    pages = {}
    for path in sorted(Path(site_dir).rglob("*.html")):
        page = ArticleLinks()
        page.feed(path.read_text())
        if page.document and path.name != "404.html":
            pages[path.relative_to(site_dir).as_posix()] = page
    if "index.html" not in pages:
        raise ValueError("Build the MkDocs homepage before auditing click routes")
    parents = {"index.html": None}
    pending = deque(parents)
    while pending:
        origin = pending.popleft()
        # The proof page must not manufacture reachability for its own inventory.
        if origin == "access-map/index.html":
            continue
        for href, text in pages[origin].links:
            target = destination(origin, href, site_url)
            if target is None or target in parents:
                continue
            parents[target] = (origin, text)
            if target in pages:
                pending.append(target)
    missing = sorted(set(pages) - parents.keys())
    if missing:
        raise ValueError(
            f"No content-link route from home to documentation pages: {missing}"
        )
    return pages, parents


def source_page(target, root=ROOT):
    if target == "index.html":
        return "index.md"
    prefix = target.removesuffix("/index.html")
    for name in (prefix + ".md", prefix + "/index.md", prefix + "/README.md"):
        if (root / "docs" / name).is_file() or name in {
            "installation.md",
            "examples.md",
        }:
            return name
    if prefix.startswith("visual-previews/"):
        return prefix + ".md"
    raise ValueError(f"Cannot resolve built page to a maintained source: {target}")


def markdown_link(text, href):
    text = text.replace("|", "&#124;").replace("[", "&#91;").replace("]", "&#93;")
    return f"[{text}]({href})"


def breadcrumb(target, parents, pages, root=ROOT):
    chain = []
    while parents[target] is not None:
        origin, text = parents[target]
        href = source_page(target, root) if target in pages else target
        chain.append(markdown_link(text, href))
        target = origin
    return " → ".join(["[Home](index.md)", *reversed(chain)])


def access_map(names, site_dir, site_url, root=ROOT):
    pages, parents = content_routes(site_dir, site_url)
    media = [name for name in names if name.startswith("docs/")]
    missing = [name for name in media if name[5:] not in parents]
    if missing:
        raise ValueError(f"Visual files lack clickable content routes: {missing}")
    lines = [
        "# Homepage click paths",
        "",
        "[Documentation home](index.md) · [All visuals](visuals.md) · [Complete directory](directory.md)",
        "",
        "Every route below is derived from actual links in rendered page content.",
        "Start at Home and click the named links in order. Search, sidebar links,",
        "automatic image requests and this index's outgoing links are excluded from",
        "the reachability proof. Use All visuals → the preview name for theme controls;",
        "the original-file column opens the exact saved file separately.",
        "",
        f"Coverage: **{len(pages)} public documentation pages**, **{len(media)} visual files**",
        f"and **{len(names)-len(media)} identical viewer branding copies**.",
        "",
        "## Documentation pages",
        "",
        "| Page | Click sequence from Home |",
        "| --- | --- |",
    ]
    page_report = {}
    for target, page in sorted(
        pages.items(),
        key=lambda item: (
            source_page(item[0], root).startswith("releases/"),
            source_page(item[0], root),
        ),
    ):
        title = page.title.replace("¶", "").strip()
        route = breadcrumb(target, parents, pages, root)
        lines.append(f"| {markdown_link(title, source_page(target, root))} | {route} |")
        page_report[target] = route
    visual_report = {}
    for group, heading in GROUPS.items():
        members = [
            name
            for name in media
            if posixpath.dirname(name.removeprefix("docs/assets/")) == group
        ]
        if not members:
            continue
        lines.extend(
            [
                "",
                f"## {heading}",
                "",
                "| Visual file | Click sequence from Home |",
                "| --- | --- |",
            ]
        )
        for name in members:
            target = name[5:]
            route = breadcrumb(target, parents, pages, root)
            lines.append(f"| `{target}` | {route} |")
            visual_report[name] = route
    lines.extend(
        [
            "",
            "## Viewer branding copies",
            "",
            "| Runtime file | Click sequence to identical visual |",
            "| --- | --- |",
        ]
    )
    for name in names:
        if name.startswith("viewer/"):
            master = "assets/branding/" + Path(name).name
            route = breadcrumb(master, parents, pages, root)
            lines.append(f"| `{name}` | {route} |")
            visual_report[name] = route
    lines.extend(
        [
            "",
            "## Maintain this index",
            "",
            "Regenerate with `python -m work.docs_access`. CI runs the checker after",
            "the strict MkDocs build and rejects stale inventories, missing routes or",
            "viewer branding copies that diverge from their documentation masters.",
            "",
        ]
    )
    report = {
        "documentation_pages": len(pages),
        "visual_files": len(media),
        "runtime_copies": len(names) - len(media),
        "page_routes": page_report,
        "visual_routes": visual_report,
    }
    return "\n".join(lines), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    from mkdocs.config import load_config

    config = load_config(str(ROOT / "mkdocs.yml"))
    names = inventory()
    catalog = visual_catalog(names)
    if not args.check:
        (ROOT / "docs" / CATALOG).write_text(catalog)
        placeholder = ROOT / "docs" / ACCESS_MAP
        if not placeholder.exists():
            placeholder.write_text("# Homepage click paths\n\n[Home](index.md)\n")
        subprocess.run(
            [sys.executable, "-m", "mkdocs", "build", "--strict"], cwd=ROOT, check=True
        )
    click_map, report = access_map(names, config.site_dir, config.site_url)
    expected = {CATALOG: catalog, ACCESS_MAP: click_map}
    stale = [
        name
        for name, text in expected.items()
        if (ROOT / "docs" / name).read_text() != text
    ]
    if args.check and stale:
        raise SystemExit(f"Regenerate documentation access catalogs: {stale}")
    if not args.check:
        for name, text in expected.items():
            (ROOT / "docs" / name).write_text(text)
        subprocess.run(
            [sys.executable, "-m", "mkdocs", "build", "--strict"], cwd=ROOT, check=True
        )
        # Recheck against the final output, not the placeholder/proof links.
        final_map, report = access_map(names, config.site_dir, config.site_url)
        if final_map != click_map:
            raise SystemExit("Click paths changed during generation; rerun and inspect")
    report_path = ROOT / ".cache/docs-access-routes.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Proved homepage content-click routes: {report['documentation_pages']} pages, {report['visual_files']} visual files, {report['runtime_copies']} runtime copies."
    )


if __name__ == "__main__":
    main()
