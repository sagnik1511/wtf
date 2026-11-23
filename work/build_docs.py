"""Stage Git-visible documentation, build with Zensical, then preserve old URLs."""

import argparse
from contextlib import contextmanager
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from types import SimpleNamespace
from urllib.parse import urlsplit

from work import docs_site

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    """Load the documentation site configuration with resolved paths."""
    path = Path(path or ROOT / "zensical.toml")
    with path.open("rb") as stream:
        values = tomllib.load(stream)["project"]
    for key in ("docs_dir", "site_dir"):
        values[key] = str((path.parent / values[key]).resolve())
    return SimpleNamespace(**values)


def page_for(source):
    """Build the page descriptor used by documentation link rewriting."""
    prefix = source[:-3]
    url = prefix + "/"
    if Path(source).name in {"index.md", "README.md"}:
        url = prefix.rsplit("/", 1)[0] + "/" if "/" in prefix else ""
    return SimpleNamespace(
        file=SimpleNamespace(src_uri=source, dest_uri=url + "index.html"), url=url
    )


def prepare(root=ROOT):
    """Never let the site generator scan the private checkout documentation tree."""

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-locals
    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel
    # Documentation-only dependencies are installed from requirements-docs.txt in the
    # separate docs environment.
    # pylint: disable=import-error

    import tomli_w

    config_path = root / "zensical.toml"
    config = load_config(config_path)
    stage = root / ".cache/docs-build"
    docs = stage / "docs"
    if docs.exists():
        shutil.rmtree(docs)
    docs.mkdir(parents=True)
    sources = docs_site.public_sources(root)
    for name, source in docs_site.EXTRA_PAGES.items():
        sources[name] = root / source
    from work.docs_access import inventory, visual_previews

    generated = visual_previews(inventory(root), root)
    pages = []
    for name in sorted(sources.keys() | generated.keys()):
        target = docs / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".md"):
            page = page_for(name)
            text = generated[name] if name in generated else sources[name].read_text()
            text = docs_site.on_page_markdown(text, page, config, None)
            # Zensical's action partial consumes this explicit edit target.
            if getattr(page, "edit_url", None):
                text = f'---\nwtf_edit_url: "{page.edit_url}"\n---\n\n' + text
            target.write_text(text)
            pages.append(page)
        else:
            shutil.copy2(sources[name], target)
    with config_path.open("rb") as stream:
        data = tomllib.load(stream)
    project = data["project"]
    project["docs_dir"] = os.path.relpath(docs, root / ".cache")
    project["site_dir"] = os.path.relpath(config.site_dir, root / ".cache")
    project["theme"]["custom_dir"] = os.path.relpath(
        root / project["theme"]["custom_dir"], root / ".cache"
    )
    staged_config = root / ".cache/zensical.build.toml"
    staged_config.write_text(tomli_w.dumps(data))
    return staged_config, config, pages


@contextmanager
def build_lock(root):
    """Preview and CLI builds share staging/output; never mutate them together."""

    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel
    # The Windows-only stdlib import is guarded by os.name; it is unavailable on POSIX
    # lint hosts.
    # pylint: disable=import-error

    path = root / ".cache/docs-build.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            if not path.stat().st_size:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def build(root=ROOT):
    """Generate the configured build output and validate required inputs."""
    with build_lock(root):
        return _build(root)


def _build(root):

    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel

    from work.build_source_docs import build as build_source_reference

    source_html = build_source_reference(root)
    staged, config, pages = prepare(root)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "zensical",
            "build",
            "--strict",
            "--config-file",
            str(staged),
        ],
        cwd=root,
        check=True,
    )
    destination = Path(config.site_dir) / "source-api"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source_html, destination)
    for page in pages:
        docs_site.on_post_page("", page, config)
    print(
        f"Built Zensical documentation: {len(pages)} pages in {config.site_dir}",
        flush=True,
    )
    return config


def source_signature(root=ROOT):
    """Watch source changes only, excluding all generated output and private files."""

    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel

    paths = set(docs_site.public_sources(root).values())
    paths.update(root / name for name in docs_site.EXTRA_PAGES.values())
    paths.update(
        (
            root / "zensical.toml",
            root / "requirements-docs.txt",
            root / ".gitignore",
            root / "Doxyfile",
        )
    )
    paths.update((root / "docs/overrides").rglob("*"))
    paths.update((root / "work").glob("*docs*.py"))
    from work.build_source_docs import source_files

    paths.update(root / name for name in source_files(root))
    return tuple(
        sorted(
            (str(p), p.stat().st_mtime_ns, p.stat().st_size)
            for p in paths
            if p.is_file()
        )
    )


def serve(address):
    """Serve a local preview and rebuild when maintained sources change."""
    config = build()
    prefix = urlsplit(config.site_url).path.rstrip("/")

    class Handler(SimpleHTTPRequestHandler):
        """Serve read-only HTTP responses using the configured local services."""

        def translate_path(self, path):
            if path == prefix or path.startswith(prefix + "/"):
                path = path[len(prefix) :] or "/"
            return super().translate_path(path)

    host, port = address.rsplit(":", 1)
    server = ThreadingHTTPServer(
        (host, int(port)), partial(Handler, directory=config.site_dir)
    )
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    print(
        f"Preview: http://{address}{prefix}/ (rebuilds source changes; refresh browser)",
        flush=True,
    )
    signature = source_signature()
    try:
        while True:
            time.sleep(1)
            current = source_signature()
            if current != signature:
                signature = current
                try:
                    build()
                except (subprocess.CalledProcessError, ValueError, OSError) as error:
                    print(
                        f"Build failed; fix sources to rebuild: {error}",
                        file=sys.stderr,
                    )
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def main():
    """Parse command-line options and run the requested operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("build", "serve"), nargs="?", default="build"
    )
    parser.add_argument("--dev-addr", default="127.0.0.1:8002")
    args = parser.parse_args()
    if args.command == "serve":
        serve(args.dev_addr)
    else:
        build()


if __name__ == "__main__":
    main()
