# Source call and caller graphs

[Development](development.md) · [Architecture](architecture.md) · [Package layout](package-layout.md)

[Open the generated source browser](https://sagnik1511.github.io/wtf/source-api/index.html)
for searchable source, class/function references and both graph directions.
The source browser is generated from the same checkout as this documentation;
it is separate from the animal-brain explanation interface.

## Find a component

1. Open the source browser, then **Files → File List**. Choose a Python module,
   JavaScript module, native source/header, developer tool or example. The
   **Complete source page index** on the source-browser homepage also links
   every generated page, including member lists and source listings.
2. Follow a class or function to its detailed documentation. Use the source-line
   link to inspect its implementation.
3. Read **call graph** for outgoing callees and **caller graph** for incoming
   callers. Expand the graph section when it is collapsed. SVG graphs support
   navigation to the referenced symbol and, where provided, zoom/pan controls.
4. Follow the textual **References** / **Referenced by** lists as another way to
   navigate. Search and the class index also accept function/class names.

| Area | Source components |
| --- | --- |
| Commands and configuration | `wtf/cli.py`, `commands.py`, `server.py`, `config.py`, resources and progress helpers |
| Physical simulation | `wtf/simulation/`: world, perception, geography, actions, placement and daylight |
| Learning | `wtf/learning/`: controller implementations, dispatch, inheritance, optimization and diagnostics |
| Experiment lifecycle | `wtf/experiment/`: scheduling, lineage and early stopping |
| Saved artifacts | `wtf/storage/`: recordings, binary codecs, checkpoints, models and libraries |
| Analysis and explanation | `wtf/inspection/`: metrics, telemetry and archived-model analysis |
| Native compute | `wtf/native/`: Python dispatch, numeric buffers and C sources/headers |
| Saved-run viewer | `viewer/*.js` and `viewer/*.mjs`: playback, charts, inspector, rendering and API access |
| Development and examples | Maintained source in `work/`, `examples/` and `setup.py` |

All Git-visible source files in these areas enter the inventory, including
private/static helpers. Tests, installed asset copies, dependencies, results,
model archives, caches and ignored research scripts are not separate source
components in this reference. CSS, HTML, schema and environment JSON describe
presentation/data rather than function calls; use [customization](customization.md)
for their definitions and relationships.

## Interpret the graphs

An arrow follows the call direction: **caller → callee**. The call graph starts
at the selected function; the caller graph reverses the exploration direction,
while arrows still describe calls. Self-recursion and mutually recursive
relationships may appear. Diagrams are bounded to **two levels and 40 nodes**
to keep dense kernels and dispatch modules readable. Truncation is a display
limit, not evidence that no other relationships exist.

[Doxygen](https://www.doxygen.nl/manual/diagrams.html) extracts static source
relationships and [Graphviz](https://graphviz.org/) renders the SVGs. These are
not execution traces or a proof that a branch runs. In Python and JavaScript,
callbacks, decorators, factories, aliases, inherited/polymorphic methods and
runtime dispatch may be missing or resolved imperfectly. Some arrow functions
and dynamically created methods may not be recognized as callable symbols.
Python-to-native registration and JavaScript-to-HTTP requests are not inferred
as cross-language call edges. Use [architecture](architecture.md) and
[native boundaries](native-kernels.md) for those contracts.

A graph is shown only when the parser finds a relationship. An empty caller
list does **not** establish dead code: CLI entry points, callbacks and extension
registrations can be reached without a directly resolved source call. Textual
references can include data access as well as calls. The source remains the
final reference; no guessed runtime edges are added to fill gaps.

## Generate locally

Install Doxygen and Graphviz separately from the simulator's Python environment:

```sh
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install doxygen graphviz

# macOS, using Homebrew
brew install doxygen graphviz
```

Use Doxygen **1.9.8 or newer** with JavaScript extension mapping support. The
current generated output records the actual Doxygen/Graphviz versions locally;
CI uses the Ubuntu runner's packages. Installing documentation tools does not
add a simulation runtime dependency.

From the repository root:

```sh
# Generate just the source browser.
.venv/bin/python -m work.build_source_docs

# Force regeneration instead of using the content/version cache.
.venv/bin/python -m work.build_source_docs --force

# Build the complete website using the isolated docs environment.
.venv-docs/bin/python -m work.build_docs
.venv-docs/bin/python -m work.check_docs_links
.venv-docs/bin/python -m work.docs_access --check
.venv-docs/bin/python -m work.build_docs serve --dev-addr 127.0.0.1:8002
```

For local preview, open <http://127.0.0.1:8002/wtf/source-api/index.html>.
The source-browser link in the checked-in Markdown targets the published site;
use this local address when inspecting unpushed changes. The documentation
workflow builds and publishes the reference alongside the normal guides.

## Ownership and maintenance

[`Doxyfile`](../Doxyfile) owns parser mappings, extraction, graph direction/depth
and source-browser appearance. [`work/build_source_docs.py`](../work/build_source_docs.py)
selects Git-visible sources, excludes ignored files/symlinks, generates the
complete page index independently of Doxygen’s optional crawler index, validates that every
selected source file appears in the parsed XML, and caches by source content,
configuration and tool versions. It reads source; it never imports controllers,
runs training or needs a C compiler/native extension to document the kernels.

Generated HTML/SVG/XML and the inventory manifest/log live under ignored
`.cache/source-reference/`. The website copies only the HTML tree into its
`source-api/` subtree; intermediate XML, cache manifests and logs stay local.
Nothing generated is added to Git. Parser warnings are retained in
`.cache/source-reference/doxygen.log` for inspection. Rebuild from the current
checkout after source changes; a build without the required system tools fails
explicitly rather than publishing an empty or stale reference.
