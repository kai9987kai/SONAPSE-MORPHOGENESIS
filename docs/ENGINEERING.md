# Engineering notes (v0.3.0)

Toolchain, local checks, CI, security posture and the additive-compatibility contract for
**Sonapse-LAB / SYNAPSE 0.3.0**. Everything here describes configuration in this repository or a run made
locally on 2026-11-20 (Windows 11 on ARM64, Python 3.12.10, ruff 0.15.16, Node v24). GitHub Actions did not run
as part of writing this page, so no CI result is claimed.

The 0.3.0 engineering changes are additive. They add metadata, tooling, documentation, a theme selector and
accessibility attributes; they do not change simulation, training, checkpoint or HTTP behaviour.

## 1. Toolchain

| Item | Where | Value |
|---|---|---|
| Version | `pyproject.toml`, `synapse/__init__.py` | `0.3.0`; `python -m synapse --version` prints `synapse-connectome-lab 0.3.0` |
| Python | `requires-python` | `>=3.10`; CI matrix `3.10`, `3.12`, `3.13` |
| Build backend | `[build-system]` | `setuptools>=77` (PEP 639 `license = "Apache-2.0"` + `license-files`) |
| Runtime deps | `[project] dependencies` | `torch>=2.4`, `numpy>=1.26`, `scipy>=1.11` (unchanged) |
| Extras | `[project.optional-dependencies]` | `test`, `connectomes`, `yaml` (unchanged) |
| Dev group | `[dependency-groups]` (PEP 735) | `pytest>=8`, `ruff>=0.8`, `pip-audit>=2.7`, `hypothesis>=6` |
| Citation | `CITATION.cff` | CFF 1.2.0; validated locally against the CFF 1.2.0 JSON schema |

Install the `dev` group with any installer that implements PEP 735, or install the four packages directly:

```bash
python -m pip install "pytest>=8" "ruff>=0.8" "pip-audit>=2.7" "hypothesis>=6"
```

Package metadata was built locally with `setuptools.build_meta.prepare_metadata_for_build_wheel` (setuptools 81):
it reports `Metadata-Version: 2.4`, `License-Expression: Apache-2.0`, `License-File: LICENSE`, the keywords,
classifiers and both `Project-URL` entries.

The distribution includes the Tidal checkpoint in `share/synapse-connectome-lab/`. The agent uses the
checkout or configured `SYNAPSE_HOME/artifacts/tidal_core.pt` first. For a wheel install it can also locate
the bundled checkpoint. Both distribution formats exclude generated Python bytecode.

## 2. Local checks

Run tests single-threaded, as CI does:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

Static checks (the same commands as the `lint-and-contracts` CI job):

```bash
ruff check synapse tests scripts
node --check synapse/web/app.js
python scripts/neuroweave_compat_manifest.py --check
python scripts/neuroweave_compat_manifest.py --manifest docs/compatibility/manifest-pre-frontier.json \
    --baseline-tag sonapse-pre-frontier-baseline --check
```

`tests/test_compat_manifest_tool.py` covers the manifest tool, both manifests, version consistency
(`pyproject.toml`, `synapse.__version__`, `CITATION.cff`, `--version`), the action SHA pins and the static
theme/accessibility checks in section 7.

### Pre-commit

`.pre-commit-config.yaml` pins both hook repositories to full commit SHAs:

| Repository | Tag | Commit | Hooks |
|---|---|---|---|
| `astral-sh/ruff-pre-commit` | v0.15.16 | `22f0422809455ec89ffdcf5a00170ba816e42ddb` | `ruff-check` (reads `[tool.ruff]` from `pyproject.toml`) |
| `pre-commit/pre-commit-hooks` | v6.0.0 | `3e8a8703264a2f4a69428a0aa4dcb512790b2c8c` | `check-yaml`, `check-json`, `end-of-file-fixer`, `trailing-whitespace` |

`check-json` excludes `docs/results/` and `synapse/_vendor/`. The whitespace fixers only touch `*.cff`, `*.yml` and
`*.yaml`, so they never rewrite existing Python, Markdown or data files. The hook ids were confirmed in each
repository's `.pre-commit-hooks.yaml` at the pinned commit. The JSON, YAML and whitespace hooks were emulated
locally over all tracked and new files (0 failures); `pre-commit` itself was not installed or run.

## 3. Lint policy

`[tool.ruff]`: `target-version = "py310"`, `line-length = 140`, `extend-exclude = ["synapse/_vendor"]`. The enforced
selection covers correctness rules, including undefined names in runtime code and annotations.

```toml
select = [
    "E9", "F63", "F7", "F50", "F52", "F601", "F602", "F621", "F622", "F821", "F822", "F823", "F901",
    "W605", "PLE",
    "B002", "B003", "B004", "B011", "B012", "B013", "B014", "B015", "B016", "B019", "B020",
    "B021", "B022", "B025", "B026", "B029", "B030", "B031", "B032", "B033", "B034", "B035", "B039",
]
```

`ruff check synapse tests scripts` printed `All checks passed!` on 2026-11-20. With `target-version = "py310"`,
ruff also reports syntax that needs a newer interpreter (for example `except*` or PEP 701 f-strings) under `E9`;
a local probe file confirmed this. That keeps the Python 3.10 CI leg honest.

### Broader rule set (future work, not enforced)

`ruff check --select=F,E,W,B,UP --statistics synapse tests scripts` (same `pyproject.toml` settings), measured
2026-11-20 with ruff 0.15.16 while other work was still landing in the tree: **5,522 findings**, 3,209 of them
auto-fixable. The largest groups:

| Rule | Count | Meaning |
|---|---:|---|
| `UP006` | 1,709 | `typing.List`/`Dict` annotations instead of built-in generics |
| `F401` | 797 | unused import |
| `UP035` | 630 | deprecated `typing` import |
| `E501` | 567 | line longer than 140 |
| `E702` | 494 | several statements on one line (semicolon) |
| `UP045` | 334 | `Optional[X]` instead of `X \| None` |
| `E701` | 289 | several statements on one line (colon) |
| `W293` | 285 | whitespace on a blank line |
| `B905` | 186 | `zip()` without `strict=` |
| `F841` | 55 | unused local variable |
| `B007` | 54 | unused loop variable |
| `UP037` | 45 | quoted annotation |
| `B023` | 15 | closure uses a loop variable |
| `F821` | 5 | undefined name |
| `B006` | 4 | mutable default argument |

The release review resolved all five `F821` findings and enabled the rule in CI. The `B023` and `B006`
findings remain candidates for review.

## 4. Continuous integration (`.github/workflows/regression.yml`)

Triggers, permissions (`contents: read`) and concurrency are unchanged. Every action is pinned to a full commit
SHA, resolved on 2026-11-20 with `gh api repos/<owner>/<repo>/git/ref/tags/<tag>` (all three tags are lightweight
tags that point directly at commits, so no dereferencing was needed):

| Action | Tag | Commit |
|---|---|---|
| `actions/checkout` | v4.4.0 | `11d5960a326750d5838078e36cf38b85af677262` |
| `actions/setup-python` | v5.6.0 | `a26af69be951a213d495a4c3e4e4022e16d87065` |
| `actions/upload-artifact` | v4.6.2 | `ea165f8d65b6e75b540449e92b4886f43607fa02` |

The major versions match the previously used `@v4`/`@v5`/`@v4` references. Checkout uses
`persist-credentials: false`.

Jobs:

1. **`lint-and-contracts`** (Python 3.12, `timeout-minutes: 15`): installs CPU `torch==2.11.0`, the project and
   `ruff==0.15.16`, starts a 16-node-per-source agent from the installed wheel outside the checkout, then runs
   `ruff check synapse tests scripts`, `node --check synapse/web/app.js` (Node from the
   runner image), the pre-NEUROWEAVE manifest check and the pre-Frontier manifest check.
2. **`pytest`** (`timeout-minutes: 60`, `fail-fast: false`, matrix `3.10`, `3.12`, `3.13`): same steps as before
   (CPU `torch==2.11.0`, `pip install -e '.[test,connectomes,yaml]'`, `python -m pytest -q` with one BLAS/OpenMP
   thread), with a per-version JUnit report artifact. Python 3.12 also installs `nir==1.0.8` and
   `h5py==3.14.0` to exercise the HDF5 interchange path. The other legs exercise the JSON fallback.

`torch-2.11.0+cpu` `manylinux_2_28_x86_64` wheels for `cp310`, `cp312` and `cp313` were listed at
<https://download.pytorch.org/whl/cpu/torch/> when checked on 2026-11-20, so no matrix entry was dropped.

`.github/dependabot.yml` checks `pip` and `github-actions` weekly (at most 5 open PRs each). The version comment
after each SHA lets Dependabot propose pin updates.

## 5. Security posture

These controls exist in the code (paths are relative to the repository root):

- **HTTP workbench** (`synapse/server.py`): responses carry `Content-Security-Policy: default-src 'self';
  script-src 'self'; style-src 'self'; style-src-attr 'unsafe-inline'; img-src 'self' data:; connect-src 'self'`,
  `X-Content-Type-Options: nosniff` and `Cache-Control: no-store`. `POST` requires a loopback host and same
  origin, `Content-Type: application/json`, and a body of 1 byte to 1 MiB (`MAX_BODY`); JSON with `NaN` or
  `Infinity` is rejected and responses are serialised with `allow_nan=False`.
- **Checkpoints**: `synapse/stateio.py` loads archives with `torch.load(..., map_location='cpu', weights_only=True)`.
  Weight archives in `synapse/experiments.py` and connectome caches in `synapse/connectome_assets.py` use
  `np.load(..., allow_pickle=False)`.
- **Known exception**: `synapse/capabilities.py` (bundled MaleCNS type table under `synapse/_vendor`) and
  `synapse/connectome_compiler.py` (default bundled dataset, or a caller-supplied `data_path`) call
  `np.load(..., allow_pickle=True)`. Only pass trusted files there; removing the pickle requirement is open work.
- **Read-only package inspection**: `NeuroWeaveWorkbenchAPI.inspect` in `synapse/neuroweave/workbench_api.py` accepts
  at most `MAX_INSPECT_BYTES = 1_000_000` bytes of package text and returns `read_only: true` and
  `live_state_changed: false`.
- **No raw media**: the NEUROWEAVE workbench API and package reader reject raw frame/audio keys
  (`RAW_MEDIA_KEYS` in `synapse/neuroweave/workbench_api.py` and `synapse/neuroweave/package.py`); the sensory
  endpoints take bounded derived summaries.
- **Offline exports**: standalone HTML exports (`synapse/portable_export.py`, `synapse/neuroweave/standalone.py`)
  set a `default-src 'none'` CSP meta tag.
- **Frontier modules**: `docs/frontier/CONTRACT.md` requires `import_state` to validate fully before mutating.

## 6. Additive compatibility contract

Two manifests record the public surface of a baseline tag. A later tree passes `--check` when every recorded
item is still present and every recorded config default is equal; new items may be added freely.

| Manifest | Baseline tag | Commit |
|---|---|---|
| `docs/neuroweave/compatibility-manifest.json` (default) | `sonapse-pre-neuroweave-baseline` | `6f2761b` |
| `docs/compatibility/manifest-pre-frontier.json` | `sonapse-pre-frontier-baseline` | `b9d8270` |

Recorded sections: `config_defaults`, `cli_commands`, `cli_flags`, `http_routes`, `web_dom_ids`,
`checkpoint_namespaces`, `test_modules`, `max_mode_enabled_features`.

`scripts/neuroweave_compat_manifest.py`:

- `--check` / `--write` with no other flags behave as before (default manifest and tag).
- `--manifest PATH` selects another manifest (relative paths resolve against the repository root).
- `--baseline-tag TAG` labels a written manifest; with `--check` the manifest must also record that tag.
- `--root DIR` collects from another tree, such as a worktree of the baseline tag (run in a fresh interpreter).
- Route discovery scans `/api/...` string literals in every `synapse/**/*.py` except `synapse/_vendor`; CLI
  discovery reads `synapse/cli.py` plus `synapse/neuroweave/cli_commands.py` and `synapse/frontier/cli_commands.py`
  when they exist. Discovery only adds items to the current surface, so it cannot make an older manifest fail.

The pre-Frontier manifest was generated from a temporary worktree of `sonapse-pre-frontier-baseline`.
Regenerating it on 2026-11-20 with the current script produced a byte-identical file:

```bash
git worktree add <tmp>/pre-frontier-wt sonapse-pre-frontier-baseline
python scripts/neuroweave_compat_manifest.py --write --root <tmp>/pre-frontier-wt \
    --manifest docs/compatibility/manifest-pre-frontier.json --baseline-tag sonapse-pre-frontier-baseline
git worktree remove --force <tmp>/pre-frontier-wt
```

## 7. Workbench theme and accessibility

- **Theme selector** (sidebar, `#theme-select`): Dark (default), Light, System. Dark leaves `<html>` without a
  `data-theme` attribute, so a first visit with nothing stored renders exactly as before. Light and System set
  `data-theme`; the `prefers-color-scheme` media query applies only under `data-theme="system"`. The choice is
  stored under `synapse_theme_v1` in `localStorage`, and every storage call is wrapped in `try`/`catch`.
  Canvases and the chat console stay dark in every theme because their drawing code assumes a dark background.
- **Contrast**: the test computes WCAG 2.x contrast for every light-theme text colour against every light surface,
  with translucent fills composited over the page and panel colours. The lowest measured pair is 4.66:1 (teal
  `#0b6e5b` on the composited tag fill `#e2dfdb`). White on accent fills is at least 6.18:1, omnibus buttons
  on their inline tints are at least 6.16:1, and the dark chat console's colours are at least 4.80:1.
- **Accessibility**: a skip link to the focusable `<h1 id="title">`, `:focus-visible` outlines for links, `summary`
  and programmatic focus targets (buttons and form fields already had them), `aria-current="page"` on the active
  nav button (set inside the existing tab handler), `aria-label`s that contain the visible text on
  pictograph-prefixed controls, and `role="status"`/`aria-live="polite"` on status lines that lacked a live region.
- **Manual check** (2026-11-20, built-in browser, `python -m synapse --nodes 16 serve`): a first load with
  nothing stored was dark with no `data-theme`; Light persisted across a reload; with OS light emulation, System
  switched to the light palette while Dark stayed dark; choosing Dark cleared the stored value; switching tabs
  moved `aria-current`; the skip link moved focus to `#title`. The console showed no errors.
