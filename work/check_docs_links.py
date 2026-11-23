"""Check built documentation links, anchors and assets under its deployment prefix."""

# Black owns code layout; long literal payloads, URLs and regression names retain
# their exact text.
# pylint: disable=line-too-long


import argparse
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urljoin, urlsplit


class PageLinks(HTMLParser):
    """Collect page links and anchors for the documentation integrity audit."""

    def __init__(self):
        super().__init__()
        self.links = []
        self.anchors = set()
        self.is_document = False
        self.destinations = []

    def handle_starttag(self, tag, attrs):
        parsed_attrs = dict(attrs)
        if tag == "meta" and parsed_attrs.get("name") == "generator":
            self.is_document = (
                parsed_attrs.get("content", "")
                .lower()
                .startswith(("mkdocs-", "zensical", "doxygen"))
            )
        if tag == "a" and parsed_attrs.get("href"):
            self.destinations.append(parsed_attrs["href"])
        for key in ("id", "name" if tag == "a" else "id"):
            if parsed_attrs.get(key):
                self.anchors.add(parsed_attrs[key])
        for key in ("href", "src", "poster", "xlink:href"):
            if parsed_attrs.get(key):
                self.links.append(parsed_attrs[key])
        srcset = parsed_attrs.get("srcset", "")
        if not srcset.startswith("data:"):
            self.links.extend(
                part.strip().split()[0] for part in srcset.split(",") if part.strip()
            )
        if parsed_attrs.get("http-equiv", "").lower() == "refresh":
            match = re.search(r"url\s*=\s*(.*)", parsed_attrs.get("content", ""), re.I)
            if match:
                self.links.append(match[1].strip("\"'"))
                self.destinations.append(match[1].strip("\"'"))


def check_site(site_dir, site_url):
    """Audit local documentation pages, assets and anchors."""

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements

    root = Path(site_dir).resolve()
    base = site_url.rstrip("/") + "/"
    location = urlsplit(base)
    pages = {}
    references = []
    page_urls = {}
    for path in root.rglob("*.html"):
        page = PageLinks()
        page.feed(path.read_text())
        pages[path] = page
        relative = path.relative_to(root).as_posix()
        url = urljoin(
            base,
            (
                relative[:-10]
                if (relative == "index.html" or relative.endswith("/index.html"))
                else relative
            ),
        )
        page_urls[path] = url
        references.extend((relative, urljoin(url, href)) for href in page.links)
    if not pages:
        raise ValueError(f"No built HTML pages found in {root}")
    # Custom-property URLs resolve in the consuming stylesheet, not necessarily
    # the file declaring the property (Doxygen's search icons use this pattern).
    css = {path: path.read_text() for path in root.rglob("*.css")}
    definitions = re.compile(r"(--[\w-]+)\s*:\s*(url\(.*?\))\s*;", re.S)
    properties = {}
    for text in css.values():
        for match in definitions.finditer(text):
            properties.setdefault(match[1], set()).add(match[2])
    for path, text in css.items():
        relative = path.relative_to(root).as_posix()
        url = urljoin(base, relative)
        text = definitions.sub("", text)
        values = [text]
        for name in re.findall(r"var\(\s*(--[\w-]+)", text):
            values.extend(sorted(properties.get(name, ())))
        for value in values:
            for href in re.findall(r"url\(\s*[\'\"]?([^\'\"\s)]+)", value):
                references.append((relative, urljoin(url, href)))
    errors = []
    checked = 0
    for origin, href in references:
        target = urlsplit(href)
        if target.scheme not in ("http", "https") or target.netloc != location.netloc:
            continue
        checked += 1
        if not target.path.startswith(location.path):
            errors.append(f"{origin}: escapes site prefix: {href}")
            continue
        destination = (root / unquote(target.path[len(location.path) :])).resolve()
        if not destination.is_relative_to(root):
            errors.append(f"{origin}: escapes build directory: {href}")
            continue
        if destination.is_dir():
            destination /= "index.html"
        if not destination.is_file():
            errors.append(f"{origin}: missing target: {href}")
        elif (
            target.fragment
            and destination in pages
            and unquote(target.fragment) not in pages[destination].anchors
        ):
            errors.append(f"{origin}: missing section: {href}")
    # A valid target can still be invisible to readers if nothing links to it.
    # Traverse anchors/redirects from home, excluding image/script references.
    graph = {}
    for path, page in pages.items():
        destinations = set()
        for href in page.destinations:
            target = urlsplit(urljoin(page_urls[path], href))
            if target.netloc != location.netloc or not target.path.startswith(
                location.path
            ):
                continue
            destination = (root / unquote(target.path[len(location.path) :])).resolve()
            if destination.is_dir():
                destination /= "index.html"
            if destination in pages:
                destinations.add(destination)
        graph[path] = destinations
    reachable = set()
    pending = [root / "index.html"]
    while pending:
        path = pending.pop()
        if path in reachable or path not in graph:
            continue
        reachable.add(path)
        pending.extend(graph[path] - reachable)
    documents = {
        path
        for path, page in pages.items()
        if page.is_document and path != root / "404.html"
    }
    for path in sorted(documents - reachable):
        errors.append(
            f"{path.relative_to(root)}: documentation page is unreachable from home"
        )
    return {
        "pages": len(pages),
        "references": checked,
        "errors": errors,
        "documents": len(documents),
        "reachable_documents": len(documents & reachable),
    }


def main():
    """Parse command-line options and run the requested operation."""

    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="zensical.toml")
    args = parser.parse_args()
    from work.build_docs import load_config

    config = load_config(args.config)
    report = check_site(config.site_dir, config.site_url)
    for error in report["errors"]:
        print(error)
    print(
        f'Checked {report["pages"]} HTML pages and {report["references"]} local links/assets/anchors; '
        f'{report["reachable_documents"]}/{report["documents"]} documentation pages reachable from home; {len(report["errors"])} errors.'
    )
    raise SystemExit(bool(report["errors"]))


if __name__ == "__main__":
    main()
