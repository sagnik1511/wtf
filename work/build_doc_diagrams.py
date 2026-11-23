"""Render the documentation's small Mermaid flowchart subset through Graphviz.

Sources support quoted rectangular/decision nodes, labeled arrows and subgraphs.
Unsupported syntax is rejected rather than silently omitted. No Python packages.
"""

# Black owns code layout; long literal payloads, URLs and regression names retain
# their exact text.
# pylint: disable=line-too-long


import argparse
import html
import json
from pathlib import Path
import re
import subprocess
import textwrap
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets/diagrams"
NODE = re.compile(r'([A-Za-z_][A-Za-z_0-9]*)(?:\["([^"\n]*)"\]|\{"([^"\n]*)"\})?')
CSS = """svg{--bg:#ffffff;--fg:#1f2937;--muted:#475569;--line:#64748b;--border:#cbd5e1;--node:#f3f6f8;--cluster:#fafbfc;background:var(--bg)}.graph>polygon{fill:var(--bg);stroke:none}.node polygon,.node path,.node ellipse{fill:var(--node);stroke:var(--line)}.node text,.cluster text{fill:var(--fg)}.edge path{stroke:var(--line)}.edge polygon{fill:var(--line);stroke:var(--line)}.edge text{fill:var(--muted)}.cluster polygon,.cluster path{fill:var(--cluster);stroke:var(--border)}@media(prefers-color-scheme:dark){svg{--bg:#171b22;--fg:#e5eaf1;--muted:#b2bfce;--line:#9aa9ba;--border:#4b596b;--node:#26313e;--cluster:#1c232c}}"""


def parse(source):
    """Parse the supported diagram source into its structured representation."""
    lines = [s.strip() for s in source.splitlines() if s.strip()]
    match = re.fullmatch(r"flowchart (TB|TD|LR)", lines.pop(0))
    if not match:
        raise ValueError("Expected flowchart TB, TD or LR")
    direction = "TB" if match[1] == "TD" else match[1]
    nodes = {}
    edges = []
    clusters = {}
    stack = []

    def node(token):
        m = NODE.fullmatch(token.strip())
        if not m:
            raise ValueError("Unsupported node: " + token)
        name = m[1]
        label = m[2] if m[2] is not None else m[3]
        if name not in nodes:
            nodes[name] = {"label": None, "kind": None, "cluster": None}
        if label is not None:
            kind = "diamond" if m[3] is not None else "box"
            value = nodes[name]
            if value["label"] is not None and (value["label"], value["kind"]) != (
                label,
                kind,
            ):
                raise ValueError("Conflicting node " + name)
            value.update(label=label, kind=kind, cluster=stack[-1] if stack else None)
        return name

    for line in lines:
        if line.startswith("subgraph "):
            m = re.fullmatch(r'subgraph ([A-Za-z_][A-Za-z_0-9]*)\["([^"\n]*)"\]', line)
            if not m or stack:
                raise ValueError("Unsupported subgraph " + line)
            clusters[m[1]] = m[2]
            stack.append(m[1])
            continue
        if line == "end":
            if not stack:
                raise ValueError("Unmatched subgraph end")
            stack.pop()
            continue
        edge = re.fullmatch(r"(.*?)\s+-->\s*(?:\|([^|]+)\|\s*)?(.*)", line)
        if edge:
            edges.append((node(edge[1]), node(edge[3]), edge[2]))
        else:
            node(line)
    if stack:
        raise ValueError("Unclosed subgraph")
    if any(n["label"] is None for n in nodes.values()):
        raise ValueError("Undeclared node label")
    if not edges:
        raise ValueError("Empty flowchart")
    return direction, nodes, edges, clusters


def render(source, title, document):
    """Render a supported structured input into its documentation representation."""

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-locals

    direction, nodes, edges, clusters = parse(source)
    dot = [
        "digraph G {",
        f'graph [rankdir={direction}, bgcolor="#ffffff", pad="0.18", nodesep="0.28", ranksep="0.40", splines=polyline];',
        'node [fontname="Arial", fontsize=16, shape=box, style="rounded,filled", color="#64748b", fillcolor="#f3f6f8", margin="0.16,0.11", penwidth=1.0];',
        'edge [fontname="Arial", fontsize=13, color="#64748b", arrowsize=0.65, penwidth=1.05];',
    ]

    def declaration(name):
        value = nodes[name]
        label = textwrap.fill(
            value["label"], width=27, break_long_words=False, break_on_hyphens=False
        )
        style = "filled" if value["kind"] == "diamond" else "rounded,filled"
        return f'{name} [label={json.dumps(label,ensure_ascii=False)}, shape={value["kind"]}, style="{style}"];'

    for name, value in nodes.items():
        if value["cluster"] is None:
            dot.append(declaration(name))
    for name, label in clusters.items():
        dot.append(
            f'subgraph cluster_{name} {{ label={json.dumps(label)}; fontname="Arial"; fontsize=16; color="#cbd5e1"; style="rounded";'
        )
        dot.extend(declaration(n) for n, v in nodes.items() if v["cluster"] == name)
        dot.append("}")
    for left, right, label in edges:
        attrs = (
            " [label=" + json.dumps(label, ensure_ascii=False) + "]" if label else ""
        )
        dot.append(f"{left} -> {right}{attrs};")
    dot.append("}")
    result = subprocess.run(
        ["dot", "-Tsvg"],
        input="\n".join(dot),
        text=True,
        capture_output=True,
        check=True,
    )
    svg = re.sub(r"<!DOCTYPE.*?>\s*", "", result.stdout, flags=re.S)
    svg = svg.replace(
        "<svg ",
        '<svg role="img" aria-labelledby="diagram-title diagram-description" ',
        1,
    )
    svg = re.sub(
        r"<title>G</title>",
        f'<title id="diagram-title">{html.escape(title)}</title><desc id="diagram-description">{html.escape(title)}. Diagram for {html.escape(document)}.</desc><style>{CSS}</style>',
        svg,
        count=1,
    )
    ET.fromstring(svg)
    return svg, len(nodes), len(edges)


def main():
    """Parse command-line options and run the requested operation."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    entries = json.loads((ASSETS / "manifest.json").read_text())
    missing = [e["source"] for e in entries if not (ASSETS / e["source"]).is_file()]
    if missing and not args.check:
        p.error(
            "Local .mmd sources are not tracked. Provide the missing sources to render, or use --check for SVG-only validation: "
            + ", ".join(missing)
        )
    compared = 0
    for e in entries:
        target = ASSETS / e["asset"]
        source_path = ASSETS / e["source"]
        if source_path.is_file():
            svg, nodes, edges = render(
                source_path.read_text(), e["title"], e["document"]
            )
            if args.check:
                assert target.read_text() == svg, "Stale diagram: " + str(target)
            else:
                target.write_text(svg)
            assert nodes == e["nodes"] and edges == e["edges"], e
            compared += 1
        tree = ET.parse(target).getroot()
        assert len(tree.findall(".//{*}g[@class='node']")) == e["nodes"], target
        assert len(tree.findall(".//{*}g[@class='edge']")) == e["edges"], target
        assert tree.get("role") == "img", target
    if args.check:
        print(
            f"Checked {len(entries)} SVGs: {compared} compared with local sources; {len(missing)} SVG-only structure/count checks."
        )
    else:
        print(
            "Rendered",
            len(entries),
            "SVG flowcharts with matching local source topology.",
        )


if __name__ == "__main__":
    main()
