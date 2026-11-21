"""Check built documentation links, anchors and assets under its deployment prefix."""

import argparse
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urljoin, urlsplit


class PageLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.anchors = set()
        self.is_document = False
        self.destinations = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "meta" and attrs.get("name") == "generator":
            self.is_document = attrs.get("content", "").startswith("mkdocs-")
        if tag == "a" and attrs.get("href"):
            self.destinations.append(attrs["href"])
        for key in ("id", "name" if tag == "a" else "id"):
            if attrs.get(key):
                self.anchors.add(attrs[key])
        for key in ("href", "src", "poster", "xlink:href"):
            if attrs.get(key):
                self.links.append(attrs[key])
        srcset = attrs.get("srcset", "")
        if not srcset.startswith("data:"):
            self.links.extend(
                part.strip().split()[0] for part in srcset.split(",") if part.strip()
            )
        if attrs.get("http-equiv", "").lower() == "refresh":
            match = re.search(r"url\s*=\s*(.*)", attrs.get("content", ""), re.I)
            if match:
                self.links.append(match[1].strip("\"'"))
                self.destinations.append(match[1].strip("\"'"))


def check_site(site_dir, site_url):
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
            base, relative[:-10] if relative.endswith("index.html") else relative
        )
        page_urls[path] = url
        references.extend((relative, urljoin(url, href)) for href in page.links)
    if not pages:
        raise ValueError(f"No built HTML pages found in {root}")
    for path in root.rglob("*.css"):
        relative = path.relative_to(root).as_posix()
        url = urljoin(base, relative)
        for href in re.findall(r"url\(\s*[\'\"]?([^\'\"\s)]+)", path.read_text()):
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="mkdocs.yml")
    args = parser.parse_args()
    from mkdocs.config import load_config

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
