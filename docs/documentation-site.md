# Documentation website

[Documentation home](index.md) · [Development](development.md)

The GitHub Pages site presents the existing Markdown guides with search,
section navigation, a page contents panel, copyable code and light/dark themes.
The expected project URL is <https://sagnik1511.github.io/wtf/> once deployed.
It is a static documentation site; it does not train animals or host the
Python-backed replay server.

## Build or preview locally

From the checkout root, install documentation tools into the development venv:

```sh
.venv/bin/python -m pip install --cache-dir .cache/pip -r requirements-docs.txt
.venv/bin/python -m mkdocs build --strict
.venv/bin/python -m mkdocs serve --dev-addr 127.0.0.1:8002
```

Open <http://127.0.0.1:8002/wtf/>. Stop the preview with Ctrl+C. Documentation tools
are separate from `requirements.txt` and are not simulator dependencies.
The generated site lives under ignored `.cache/docs-site/` and is rebuilt from
source. Do not commit generated HTML, a second copy of the guides or a `gh-pages`
branch. Site builds require a Git checkout to enforce ignored-file exclusions.

## Enable GitHub Pages

1. Push this documentation setup to the repository's `main` branch.
2. Open repository **Settings → Pages → Build and deployment** and select
   **GitHub Actions** as the source.
3. Open **Actions → Documentation** and run the workflow if it has not already
   run for that push. Check the build and deployment jobs.
4. Open the URL shown by the `github-pages` deployment environment.

The workflow builds on documentation pull requests and deploys only from `main`.
It uses GitHub's Pages artifact/deployment actions, with `pages: write` and
`id-token: write` limited to deployment. It needs no personal access token or
repository write permission. Repository/environment protection rules can still
require approval. No repository setting is changed by a local build.

See [GitHub's custom Pages workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
for the repository-side requirements. If the owner/repository name changes,
update `site_url`, `repo_url` and `repo_name` in [mkdocs.yml](../mkdocs.yml).
A default-branch rename also requires updating `edit_uri`, the workflow branch
filters and source-link branch in [the build hook](../work/docs_site.py).

## Content structure

| Section | Owns |
| --- | --- |
| Get started | Installation, Linux, Docker, run/resume and complete Python tutorials |
| Worlds and species | Configuration, maps, physics, species, optional day/night and calibration |
| Learning and behavior | Policies, inputs, feedback, mating, mutation, champions and evaluation |
| Viewer and storage | Replay controls, metrics, inspector, explainability and artifact structures |
| Development | Architecture, tests, extensions, packaging, performance and site upkeep |
| Reference | Complete field/command definitions, compatibility, visual index and existing release history |

Keep the detailed guides in their current locations so GitHub links remain
useful. Edit [the home page](index.md) for reader routes and `nav` in
[mkdocs.yml](../mkdocs.yml) for the website hierarchy. This setup does not create
or require a new software release.

## Reuse sources and exclude local work

[INSTALL.md](../INSTALL.md) and [examples/README.md](../examples/README.md) are
rendered as website pages directly from their original content. They remain the
single maintained copies. The build hook rewrites cross-boundary Markdown links
in memory: documentation stays inside the site, code/config/agent links open
GitHub, and fenced code examples retain their literal text. Existing SVGs, GIFs
and the self-contained physics explorer are served from `docs/assets/`.

Only Git-visible, non-ignored documentation files are considered for local
preview; published Actions checkouts contain committed files. The hook also
blocks `docs/paper/`, `docs/experiments/` and symlinks even if accidentally tracked.
Results, model archives, virtual environments and caches are outside `docs/` and
are never included. Source links do not copy those files into the website.
Asset README pages and historical release details remain reachable through their
indexes without crowding the main sidebar.

## Updating documentation

1. Edit the relevant guide; update existing visual exports only when they clarify
   a changed structure or behavior.
2. For a new guide, add a useful home-page route and a navigation entry. Generated
   field references still come from the schema via `work.build_config_reference`.
3. Build with `--strict`; fix missing targets and anchors instead of hiding them.
4. Check the rendered page, search and links, including under the `/wtf/` project
   prefix. Format source/config changes according to the development guide.
5. Commit the source changes. CI validates pull requests; deployment follows a
   push to `main`. Actual publishing is confirmed by the deployment job.

To check the publication filter and link handling, install the documentation
requirements and run:

```sh
.venv/bin/python -m unittest tests.test_docs_site
```

The site is unversioned and follows `main`. Historical implementation records
remain in the existing release hub; they are separate from experiment history
and documentation deployment status.
