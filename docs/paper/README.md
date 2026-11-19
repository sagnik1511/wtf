# Wildlife Trial Framework paper

[Main LaTeX source](main.tex) · [Future observations](sections/observations.tex) · [Project overview](../../README.md)

This author-prepared **Nature Machine Intelligence Article draft** follows the
journal's section order and uses its accepted standard `article.cls`, Computer
Modern text and numerical citations. It is a manuscript for review, rather than
an imitation of the published journal layout or a claim of submission. The
inspected implementation is `6e354d3` (19 November 2026).

## Journal structure and scope

The [current Article guidance](https://www.nature.com/natmachintell/content)
specifies an unheaded introduction, Results, Discussion and Methods; an
unreferenced abstract of at most 150 words; up to 3,500 main-text words excluding
abstract, Methods, references and legends; and up to six figures/tables.
This draft uses five main figures and one Methods table. Detailed engineering
tables, the future-study registry, outcome template and evidence manifest are
supplementary. The Discussion has no subheadings. Recheck these limits as the
observations grow. Guidance was checked on 19 November 2026.

[Initial submissions](https://www.nature.com/natmachintell/submission-guidelines/initial-formatting)
permit flexible formatting and request a compiled PDF when using LaTeX.
[Accepted-manuscript guidance](https://www.nature.com/natmachintell/submission-guidelines/aip-and-formatting)
accepts standard classes and asks for a single textual TeX file with numerical
references inlined from the bibliography. The editable project remains modular;
the generated source bundle also includes a flattened `submission.tex` with its
bibliography inlined. Before final submission, remove personal macros, resolve
all draft declarations, complete archive links and follow the handling editor's
requirements. This draft is not a completed submission package.

## Figures and their purpose

| Figure | What it makes easier to understand |
| --- | --- |
| [Runtime ownership](figures/architecture.tex) | Independent Python learning and saved-data JavaScript playback |
| [Geography and observations](figures/environment-observation.tex) | Circular sight, water extension, wall occlusion, and the 78-value top-k input layout |
| [Inheritance and ancestry](figures/inheritance.tex) | Two-parent parameter mixing, fresh child training state and overlapping founder families |
| [Artifact access](figures/artifacts.tex) | Replay viewing versus full checkpoint restore versus parameter-only model initialization |
| [Engineering evidence](figures/engineering.tex) | Historical native timings and storage retained across separate fixtures |

All figures are editable vector TikZ/PGFPlots, rendered by LaTeX. Spatial and
inheritance examples are labelled schematics. The engineering plot reads exact
values from [native-timings.csv](figures/native-timings.csv) and
[storage-fixtures.csv](figures/storage-fixtures.csv); it does not invent raw
replicates or uncertainty bars. Each storage fixture has its own baseline.
No ecological trajectories or learning improvements have been fabricated.

## Add experimental observations

Edit [sections/observations.tex](sections/observations.tex). Its study registry,
outcome table and prompts cover survival/hunting, inheritance, geography/sensing
and learning signals. Dashes mean missing measurements, not zero. Record source
runs, config/revision, initialization, seed list, budgets, controls, denominators
and uncertainty. The protocol is in
[sections/experimental-design.tex](sections/experimental-design.tex).

After studies are completed, replace the prospective Results paragraph with the
actual findings and figures, moving only completed evidence into the main text.
Update the abstract, Discussion and display-item allocation at that point.
Historical engineering measurements are separate from ecological outcomes.

[author-info.tex](author-info.tex) retains Sagnik Roy, linked
[GitHub](https://github.com/sagnik1511) and
[LinkedIn](https://www.linkedin.com/in/sagnik1511/) profiles and the supplied
phone number. No affiliation or email was invented. Confirm public-facing author
metadata and fill [declarations.tex](sections/declarations.tex) before submission.
AI drafting assistance is disclosed in Methods.

## Build

Line numbers are disabled by default. For a review copy, change
`\paperlinenumbersfalse` to `\paperlinenumberstrue` near the top of
[main.tex](main.tex). The `lineno` package is loaded only when enabled.

From the repository root, with Tectonic installed:

```sh
mkdir -p docs/paper/build
cd docs/paper
TECTONIC_CACHE_DIR=../../.cache/paper/tectonic tectonic --keep-logs --keep-intermediates --outdir build main.tex
```

Or with TeX Live:

```sh
cd docs/paper
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
bibtex build/main
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
```

Open this directory in the LaTeX editor and compile `main.tex` for an integrated
preview. Generated outputs stay ignored:

- `build/main.pdf`: compiled manuscript and supplementary information.
- `build/wtf-paper-source.tar.gz`: editable source bundle, data tables and generated bibliography.
- `build/submission.tex`: single-text-file export with sections/figure source and references inlined.

The flattened export remains dependent on the two included figure CSV files.
Rebuild and regenerate exports after changing sources; the generated copies are
not independent editing targets. No shell escape, remote images or browser
execution is used. Section boundaries use the built-in `\clearpage` command, so `placeins.sty`
is not required. Standard LaTeX packages are installed by the TeX distribution;
no journal class/style is vendored. See [template provenance](THIRD_PARTY.md).

## Source layout

| Source | Content |
| --- | --- |
| `main.tex`, `author-info.tex` | Manuscript configuration, title, abstract, author and contacts |
| `sections/introduction.tex`, `results-overview.tex` | Motivation and implemented system capabilities |
| `sections/engineering-results.tex`, `discussion.tex` | Historical measured evidence and limitations |
| `sections/environment.tex`, `learning.tex`, `inheritance.tex`, `systems.tex` | Detailed Methods |
| `sections/experimental-design.tex`, `declarations.tex` | Proposed ecological evaluation and author declarations |
| `sections/observations.tex` | Editable future ecological findings |
| `sections/engineering-tables.tex`, `reproducibility.tex` | Supplementary exact values, commands and evidence inventory |
| `figures/*.tex`, `figures/*.csv` | Five vector figures and chart source data |
| `references.bib` | Thirteen primary method/platform references |

## Portable source bundle

Include `main.tex`, `author-info.tex`, `sections/`, `figures/`, `references.bib`,
`THIRD_PARTY.md`, generated `main.bbl` at the bundle root and the flattened
`submission.tex`. Build **one entry point at a time**; `main.tex` is the normal
editable entry point. The supplied archive is independently compiled before
handoff. Exclude caches, logs, raw experiments and preview PDFs. Add any future
PDF/PNG/JPG figure dependencies to the bundle. See also
[arXiv TeX guidance](https://info.arxiv.org/help/submit_tex.html).
