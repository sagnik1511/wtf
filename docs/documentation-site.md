# Documentation website

[Documentation home](index.md) · [Development](development.md)

The GitHub Pages site presents the existing Markdown guides with search,
section navigation, a page contents panel, copyable code and light/dark themes.
The expected project URL is <https://sagnik1511.github.io/wtf/> once deployed.
It is a static documentation site; it does not train animals or host the
Python-backed replay server.

Use the **sun/moon icon at the top right** to switch between light and dark
appearance. The initial palette follows the operating system preference, and a
manual choice is remembered in that browser. Page surfaces, navigation, text and
project logos follow the selected site theme. The project symbol remains visible
beside the title on every page and at mobile widths. Adaptive SVG diagrams follow
the same toggle; [visual theme rules](visual-guide.md#light-and-dark-appearance)
explain how paired replay exports and standalone SVG specimens follow the theme
without recoloring recorded data. The standalone viewer has its own
separate theme control; its simulated day–night lighting is another setting.

## Build or preview locally

From the checkout root, install documentation tools into the development venv:

```sh
.venv/bin/python -m pip install --cache-dir .cache/pip -r requirements-docs.txt
.venv/bin/python -m mkdocs build --strict
.venv/bin/python -m work.check_docs_links
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

Select **GitHub Actions**, not **Deploy from a branch**. A branch-source Jekyll
build can overwrite a successful custom MkDocs deployment at the same URL.
The resulting source-rendered site uses different paths and bypasses the MkDocs
link/visual-theme hooks. A successful Documentation workflow alone does not
prove the live homepage is its output. The deployment checks Pages build type
before publishing and verifies the live homepage identifies itself as MkDocs
after deployment. Check this again when repository Pages settings change.

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

Every navigation group has an `index.md` landing page and Material
`navigation.indexes` is enabled. Section labels are links, not empty headings;
top tabs and the narrow-screen menu open the same landing pages. Keep a group
index first in its navigation list, link to its child guides, and provide return
links from those guides. The tests reject groups without an index.

Keep the detailed guides in their current locations so GitHub links remain
useful. Use the [complete directory](directory.md) for all guides, feature shortcuts and
supporting visual catalogs. Edit [the home page](index.md) for reader routes and `nav` in
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

## Project identity

Use the [branding guide](branding.md) for transparent SVG masters, PNG fallbacks
and symbol/full-lockup placement. Header and home-page assets follow the site
theme toggle. `docs/overrides/` contains the small theme partial overrides and
is excluded from published document content.

## Updating documentation

1. Edit the relevant guide; update existing visual exports only when they clarify
   a changed structure or behavior.
2. For a new guide, update the [complete directory](directory.md), add a useful
   home-page route and a navigation entry. Generated
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

## Legacy documentation addresses

The current site uses directory URLs, such as `/wtf/configuration/`. Older
source/Jekyll addresses such as `/wtf/docs/configuration.html` redirect to the
matching MkDocs page. The build creates these small redirect pages for public
Markdown guides, nested pages, asset catalogs and the original installation/
example entry points. Query strings and section fragments are preserved when
JavaScript is available; a meta refresh and ordinary link provide a fallback.
Ignored research sources and theme partials never receive public redirects.

The Documentation workflow runs `work.check_docs_links` after building. This
checks actual generated HTML links, section anchors, image sources (including
`picture` variants), CSS asset URLs and redirect destinations under the configured
project prefix. It also traverses document links from the homepage and rejects
public MkDocs pages that have become unreachable; the generated 404 page is
excluded. Asset catalogs and release details must remain reachable through their
indexes. The documentation tests also require every primary guide to appear in
navigation and every navigation destination to appear in the complete directory.
It does not fetch third-party websites. A successful local build
only changes the checkout; commit and push the source changes to `main` for the
Pages workflow to deploy the updated site and redirects.

## Verify the deployed site, not only the build

After deployment, start at the public home URL. Confirm its HTML generator is
MkDocs, follow every same-site document/asset destination, and validate section
anchors against the destination HTML. Include picture sources and nested paths.
A local link audit cannot detect another publishing workflow replacing the site.
Keep detailed URL/status audit reports local; report exact checked counts and
any HTTP failures or rate-limited external destinations when sharing results.
