"""Build public docs from existing guides without copying local research output."""

# Black owns code layout; long literal payloads, URLs and regression names retain
# their exact text.
# pylint: disable=line-too-long


from pathlib import Path
import html
import json
import posixpath
import re
import subprocess
from xml.etree import ElementTree as ET
from urllib.parse import quote, unquote, urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[1]
EXTRA_PAGES = {"installation.md": "INSTALL.md", "examples.md": "examples/README.md"}
FENCES = re.compile(r"(^(`{3,}|~{3,})[^\n]*\n.*?^\2[^\n]*(?:\n|$))", re.M | re.S)
LINKS = re.compile(r'(!?\[[^\]\n]*\]\()([^\s)]+)((?:\s+"[^"\n]*")?\))')


def public_sources(root=ROOT):
    """Only Git-visible public docs; private research and symlinks stay local."""

    def listed(*options):
        return set(
            subprocess.check_output(
                ["git", "ls-files", "-z", *options, "--exclude-standard", "--", "docs"],
                cwd=root,
            )
            .decode()
            .split("\0")
        )

    visible = listed("--cached", "--others") - listed("--cached", "--ignored")
    docs = (root / "docs").resolve()
    result = {}
    for source in sorted(visible - {""}):
        path = root / source
        relative = path.relative_to(root / "docs").as_posix()
        if (
            relative.startswith(("paper/", "experiments/", "overrides/"))
            or path.is_symlink()
            or any(
                (root / "docs" / parent).is_symlink()
                for parent in Path(relative).parents
            )
            or not path.is_file()
            or not path.resolve().is_relative_to(docs)
        ):
            continue
        result[relative] = path
    return result


def _site_link(href, origin, destination, config):
    parts = urlsplit(href)
    if parts.scheme or parts.netloc or not parts.path or parts.path.startswith("/"):
        return href
    source = posixpath.normpath(
        posixpath.join(posixpath.dirname(origin), unquote(parts.path))
    )
    aliases = {"README.md": "index.md", **{v: k for k, v in EXTRA_PAGES.items()}}
    if source in aliases:
        target = aliases[source]
    elif source.startswith("docs/") and not source.startswith("docs/overrides/"):
        target = source[5:]
    else:
        # Code/config/agent links open their canonical source instead of shipping
        # the source tree, environment files or model artifacts with the website.
        kind = "tree" if (ROOT / source).is_dir() else "blob"
        repository = urlsplit(config.repo_url)
        return urlunsplit(
            (
                repository.scheme,
                repository.netloc,
                f"{repository.path.rstrip('/')}/{kind}/main/{quote(source, safe='/')}",
                parts.query,
                parts.fragment,
            )
        )
    relative = posixpath.relpath(target, posixpath.dirname(destination) or ".")
    return urlunsplit(("", "", relative, parts.query, parts.fragment))


REPLAY_PICTURES = re.compile(r'<picture class="replay-preview">.*?</picture>', re.S)


def _replay_picture(match, origin, destination):
    """GitHub uses picture media; Pages follows its explicit palette toggle."""
    picture = ET.fromstring(match[0])
    fallback = picture.find("img")
    alt = html.escape(fallback.get("alt", "Saved replay preview"), quote=True)
    images = []
    for source in picture.findall("source"):
        media = source.get("media", "")
        theme = "dark" if "dark" in media else "light"
        href = source.get("srcset", "")
        parsed = urlsplit(href)
        asset = posixpath.normpath(
            posixpath.join(posixpath.dirname(origin), unquote(parsed.path))
        )
        if parsed.scheme or parsed.netloc or not asset.startswith("docs/assets/"):
            raise ValueError(
                f"Replay previews must use local documentation assets: {href}"
            )
        relative = posixpath.relpath(asset[5:], posixpath.dirname(destination) or ".")
        images.append(
            f'<img class="replay-theme-{theme}" src="{html.escape(relative, quote=True)}" alt="{alt}" loading="lazy" />'
        )
    if len(images) != 2:
        raise ValueError("Replay previews need light and dark sources")
    return "\n".join(images)


def on_page_markdown(markdown, page, config, files):
    """Preserve checkout links and literal code; resolve website links in memory."""

    # The callback keeps its caller's keyword signature; this branch does not consume
    # every input.
    # pylint: disable=unused-argument

    destination = page.file.src_uri
    origin = EXTRA_PAGES.get(destination, "docs/" + destination)
    sections = FENCES.split(markdown)
    for index in range(0, len(sections), 3):
        if REPLAY_PICTURES.search(sections[index]):
            sections[index] = REPLAY_PICTURES.sub(
                lambda match: _replay_picture(match, origin, destination),
                sections[index],
            )
        sections[index] = LINKS.sub(
            lambda match: match[1]
            + _site_link(match[2], origin, destination, config)
            + match[3],
            sections[index],
        )
    if destination in EXTRA_PAGES:
        page.edit_url = config.repo_url.rstrip("/") + "/edit/main/" + origin
    elif destination.startswith("visual-previews/"):
        page.edit_url = config.repo_url.rstrip("/") + "/edit/main/work/docs_access.py"
    return "".join(part for index, part in enumerate(sections) if index % 3 != 2)


def legacy_page_paths(page):
    """Previous source/Jekyll URLs remain usable after the MkDocs migration."""
    source = page.file.src_uri
    if not source.endswith(".md"):
        return set()
    aliases = {source[:-3] + ".html", "docs/" + source[:-3] + ".html"}
    if source in EXTRA_PAGES:
        aliases.add(EXTRA_PAGES[source][:-3] + ".html")
    if source == "index.md":
        aliases.add("README.html")
    if source.endswith("/README.md"):
        aliases.add("docs/" + source[:-9] + "index.html")
    return aliases - {page.file.dest_uri}


def on_post_page(output, page, config):
    """Write lightweight redirects, preserving query strings and section links."""
    for alias in legacy_page_paths(page):
        target = posixpath.relpath(page.url or ".", posixpath.dirname(alias) or ".")
        if not page.url or page.url.endswith("/"):
            target += "/"
        escaped = html.escape(target, quote=True)
        canonical = config.site_url.rstrip("/") + "/" + page.url
        redirect = (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta name="robots" content="noindex">'
            f'<link rel="canonical" href="{html.escape(canonical, quote=True)}">'
            f'<meta http-equiv="refresh" content="0;url={escaped}">'
            "<title>Documentation moved</title>"
            "<script>location.replace("
            + json.dumps(target)
            + " + location.search + location.hash);</script></head><body>"
            f'<p>This documentation moved. <a href="{escaped}">Open the current page</a>.</p>'
            "</body></html>\n"
        )
        destination = Path(config.site_dir) / alias
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(redirect)
    return output
