"""Export every schema field and its definition into the configuration handbook."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render_reference(schema):
    rows = [
        "# Configuration field reference\n",
        "[Customization handbook](customization.md) · [Configuration guide](configuration.md) · [Schema](../environment.schema.json)\n",
        "Generated from schema definitions with `python -m work.build_config_reference`. "
        "Use `--check` to detect missing definitions or stale exports. Every declared "
        "field is listed, including each algorithm-specific variant.\n",
        "A `<name>` component means a user-chosen key. Required status is relative "
        "to the containing object/variant, not a requirement to enable that optional "
        "feature. A schema default describes omission; it does not rewrite your JSON. "
        "A dash means no default is declared here: copy a complete supported preset "
        "and consult its feature guide for runtime behavior.\n",
        "The Python validator also checks cross-field rules (map dimensions, population "
        "capacity, diet/body/actions, reproduction limits and model signatures). "
        "Passing a JSON editor's schema check alone is not sufficient. Physical values "
        "are abstract simulation units; time is measured in world steps.\n",
    ]
    count = 0

    def resolved(node):
        if "$ref" in node:
            current = schema
            for part in node["$ref"].removeprefix("#/").split("/"):
                current = current[part]
            return dict(current, **{k: v for k, v in node.items() if k != "$ref"})
        return node

    def constraints(node):
        node = resolved(node)
        parts = [
            str(
                node.get(
                    "type",
                    (
                        "variant"
                        if any(k in node for k in ("oneOf", "anyOf", "allOf"))
                        else "value"
                    ),
                )
            )
        ]
        for key, label in (
            ("const", "exact"),
            ("enum", "choices"),
            ("minimum", "min"),
            ("exclusiveMinimum", ">"),
            ("maximum", "max"),
            ("minItems", "min items"),
            ("maxItems", "max items"),
            ("minLength", "min length"),
            ("maxLength", "max length"),
            ("pattern", "pattern"),
            ("minProperties", "min properties"),
            ("maxProperties", "max properties"),
            ("uniqueItems", "unique items"),
            ("propertyNames", "key constraints"),
        ):
            if key in node:
                parts.append(f"{label}: `{json.dumps(node[key], ensure_ascii=False)}`")
        if "items" in node and isinstance(node["items"], dict):
            parts.append("items: " + constraints(node["items"]))
        return "; ".join(parts).replace("|", "\\|")

    def visit(original, path="config"):
        nonlocal count
        node = resolved(original)
        props = node.get("properties", {})
        if props:
            rows.extend(
                [
                    f"## `{path}`\n",
                    "| Field | Definition | Type and constraints | Required / schema default |",
                    "| --- | --- | --- | --- |",
                ]
            )
            required = node.get("required", [])
            for key, field in props.items():
                if not field.get("description"):
                    raise ValueError(f"Missing field definition: {path}.{key}")
                presence = "Required" if key in required else "Optional"
                default = (
                    "`" + json.dumps(field["default"]) + "`"
                    if "default" in field
                    else "—"
                )
                meaning = field["description"].replace("|", "\\|").replace("\n", " ")
                rows.append(
                    f"| `{key}` | {meaning} | {constraints(field)} | {presence}; {default} |"
                )
                count += 1
            rows.append("")
            for key, field in props.items():
                visit(field, f"{path}.{key}")
        if isinstance(node.get("additionalProperties"), dict):
            visit(node["additionalProperties"], path + ".<name>")
        if isinstance(node.get("items"), dict):
            visit(node["items"], path + "[]")
        for union in ("oneOf", "anyOf", "allOf"):
            for index, variant in enumerate(node.get(union, []), 1):
                kind = resolved(variant).get("properties", {}).get("type", {})
                name = (
                    kind.get("const")
                    or ", ".join(kind.get("enum", []))
                    or f"{union} variant {index}"
                )
                visit(variant, f"{path} ({name})")

    visit(schema)
    rows.append(f"Field entries documented: **{count}**.\n")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = ROOT / "docs/configuration-reference.md"
    generated = render_reference(
        json.loads((ROOT / "environment.schema.json").read_text())
    )
    if args.check:
        if output.read_text() != generated:
            raise SystemExit("Configuration reference is stale; regenerate it")
    else:
        output.write_text(generated)
    print(
        "Verified configuration reference"
        if args.check
        else "Built configuration reference"
    )


if __name__ == "__main__":
    main()
