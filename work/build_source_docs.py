"""Generate cached call/caller source documentation without importing project code."""

import argparse
import hashlib
import html
import json
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".py", ".js", ".mjs", ".cjs", ".c", ".h", ".cpp", ".hpp"}


def source_files(root=ROOT):
    """Include maintained implementation/tooling, never dependencies or research."""

    def listed(*options):
        return set(
            subprocess.check_output(
                ["git", "ls-files", "-z", *options, "--exclude-standard"], cwd=root
            )
            .decode()
            .split("\0")
        )

    names = listed("--cached", "--others") - listed("--cached", "--ignored")
    result = []
    for name in sorted(names - {""}):
        path = root / name
        if name.startswith("wtf/assets/"):
            continue
        if (
            Path(name).parts[0] not in {"wtf", "viewer", "work", "examples"}
            and name != "setup.py"
        ):
            continue
        if (
            path.suffix not in EXTENSIONS
            or not path.is_file()
            or path.is_symlink()
            or any((root / parent).is_symlink() for parent in Path(name).parents)
            or not path.resolve().is_relative_to(root.resolve())
        ):
            continue
        result.append(name)
    return result


def summarize(xml_dir, names):
    """Validate file coverage and count parser-recognized function relationships."""
    files = set()
    symbols = set()
    edges = set()
    for path in sorted(Path(xml_dir).glob("*.xml")):
        if path.name in {"Doxyfile.xml", "index.xml"}:
            continue
        document = ET.parse(path)
        for compound in document.findall("compounddef"):
            location = compound.find("location")
            if compound.get("kind") == "file" and location is not None:
                files.add(location.get("file", ""))
            for member in compound.findall(".//memberdef[@kind='function']"):
                symbols.add(member.get("id"))
                for reference in member.findall("references"):
                    edges.add((member.get("id"), reference.get("refid")))
    missing = sorted(set(names) - files)
    if missing:
        raise ValueError(f"Doxygen omitted source components: {missing}")
    return {
        "files": len(names),
        "functions": len(symbols),
        "reference_edges": len(edges),
    }


def build(root=ROOT, force=False):
    for program in ("doxygen", "dot"):
        if not shutil.which(program):
            raise RuntimeError(
                f"Install {program} for source graphs; see docs/source-graphs.md"
            )
    names = source_files(root)
    version = subprocess.check_output(["doxygen", "--version"], text=True).strip()
    parsed = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    if not parsed or tuple(map(int, parsed.groups())) < (1, 9, 8):
        raise RuntimeError("Source graphs require Doxygen 1.9.8 or newer")
    dot_version = subprocess.run(
        ["dot", "-V"], capture_output=True, text=True, check=True
    ).stderr.strip()
    digest = hashlib.sha256((version + dot_version).encode())
    for name in ["Doxyfile", "work/build_source_docs.py", *names]:
        digest.update(name.encode() + b"\0" + (root / name).read_bytes() + b"\0")
    fingerprint = digest.hexdigest()
    output = root / ".cache/source-reference"
    manifest = output / "manifest.json"
    if not force and manifest.is_file() and (output / "html/index.html").is_file():
        saved = json.loads(manifest.read_text())
        if saved.get("fingerprint") == fingerprint:
            print(f"Source reference cached: {saved['files']} files", flush=True)
            return output / "html"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="source-reference-", dir=root / ".cache"
    ) as temporary:
        temporary = Path(temporary)
        overview = temporary / "overview.md"
        overview.write_text(
            """# Source call and caller graphs

[Project documentation](https://sagnik1511.github.io/wtf/source-graphs/)

[All source files](files.html) · [Classes](annotated.html) · [Namespaces](namespaces.html)

Use **Files → File List** for modules, **Classes** for class methods, or search
for a symbol. Open a function's detailed documentation for its **call graph**
(outgoing callees) and **caller graph** (incoming callers). Text references and
source-line links remain available beside diagrams. Files with no recognized
calls may have no graph. Graphs are bounded to two levels / 40 nodes; a truncated
graph is not an exhaustive dependency list.

These are parser-derived, static relationships, not runtime traces or proof that
a branch executes. Dynamic policy dispatch, callbacks, factories, aliases and
Python/native bindings can be missing or ambiguous. Identical method names can
be misattributed. Do not infer that a function is unused from an empty caller
list. See the project guide for scope, regeneration and interpretation.

The index covers maintained Python, JavaScript and native implementations,
developer tooling and Python examples. Tests, data, recordings, models,
dependencies and ignored research scripts are excluded.
"""
        )

        def quoted(value):
            return '"' + str(value).replace("\\", "/").replace('"', '\\"') + '"'

        config = temporary / "Doxyfile"
        config.write_text(
            (root / "Doxyfile").read_text()
            + "\n"
            + "INPUT = "
            + " ".join(quoted(name) for name in names)
            + " "
            + quoted(overview.relative_to(root))
            + "\n"
            + "OUTPUT_DIRECTORY = "
            + quoted(temporary / "output")
            + "\n"
            + "USE_MDFILE_AS_MAINPAGE = "
            + quoted(overview.relative_to(root))
            + "\n"
        )
        with (temporary / "doxygen.log").open("w") as log:
            subprocess.run(
                ["doxygen", str(config)],
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        generated = temporary / "output"
        report = summarize(generated / "xml", names)
        # Doxygen 1.12's hidden crawler index can point Python namespace members
        # at their file pages. Rebuild it from actual HTML destinations instead.
        # The browser's search index is independent and remains untouched.
        crawler = generated / "html/doxygen_crawl.html"
        if crawler.is_file():
            links = [
                f'<a href="{html.escape(path.name, quote=True)}">{html.escape(path.stem)}</a>'
                for path in sorted((generated / "html").glob("*.html"))
                if path != crawler
            ]
            crawler.write_text(
                "<!doctype html><html><head><title>Source reference index</title></head><body>\n"
                + "\n".join(links)
                + "\n</body></html>\n"
            )
        report.update(
            fingerprint=fingerprint,
            doxygen=version,
            graphviz=dot_version,
            sources=names,
            call_graphs=len(list((generated / "html").glob("*_cgraph.svg"))),
            caller_graphs=len(list((generated / "html").glob("*_icgraph.svg"))),
        )
        (generated / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        shutil.copy2(temporary / "doxygen.log", generated / "doxygen.log")
        if output.exists():
            shutil.rmtree(output)
        generated.rename(output)
    print(
        f"Generated source reference: {report['files']} files, {report['functions']} functions",
        flush=True,
    )
    return output / "html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(build(force=args.force))


if __name__ == "__main__":
    main()
