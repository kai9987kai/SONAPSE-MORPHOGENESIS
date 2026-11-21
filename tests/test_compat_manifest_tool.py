"""Tests for scripts/neuroweave_compat_manifest.py, both compatibility manifests and release metadata."""
import copy
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRE_NEUROWEAVE = ROOT / 'docs' / 'neuroweave' / 'compatibility-manifest.json'
PRE_FRONTIER = ROOT / 'docs' / 'compatibility' / 'manifest-pre-frontier.json'
SECTIONS = {
    'schema', 'baseline_tag', 'baseline_commit', 'contract',
    'config_defaults', 'cli_commands', 'cli_flags', 'http_routes',
    'web_dom_ids', 'checkpoint_namespaces', 'test_modules',
    'max_mode_enabled_features',
}


def _load_tool():
    path = ROOT / 'scripts' / 'neuroweave_compat_manifest.py'
    spec = importlib.util.spec_from_file_location('neuroweave_compat_manifest', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def tool():
    return _load_tool()


@pytest.fixture(scope='module')
def current(tool):
    return tool.collect()


def test_defaults_are_unchanged(tool):
    assert tool.MANIFEST == PRE_NEUROWEAVE
    assert tool.BASELINE_TAG == 'sonapse-pre-neuroweave-baseline'
    assert tool.BASELINE_COMMIT == '6f2761b'


def test_collect_returns_expected_sections(current):
    assert SECTIONS <= set(current)
    # Route discovery covers synapse/**/*.py, not only server.py.
    assert '/api/neuroweave/catalog' in current['http_routes']
    assert '--version' in current['cli_flags']
    assert current['baseline_tag'] == 'sonapse-pre-neuroweave-baseline'


def test_route_discovery_scans_package_tree_except_vendor(tool, tmp_path):
    pkg = tmp_path / 'synapse'
    (pkg / 'sub').mkdir(parents=True)
    (pkg / '_vendor' / 'lib').mkdir(parents=True)
    (pkg / 'server.py').write_text("ROUTES = ['/api/state']\n", encoding='utf-8')
    (pkg / 'sub' / 'extra.py').write_text('PATH = "/api/sub/extra"\n', encoding='utf-8')
    (pkg / '_vendor' / 'lib' / 'x.py').write_text("URL = '/api/vendored'\n", encoding='utf-8')
    files = [p.relative_to(pkg).as_posix() for p in tool._iter_synapse_py(tmp_path)]
    assert files == ['server.py', 'sub/extra.py']
    assert tool._http_routes(tmp_path) == ['/api/state', '/api/sub/extra']


def test_cli_discovery_includes_neuroweave_cli_commands(tool, tmp_path):
    sources = tool._cli_sources(ROOT)
    assert sources[0] == ROOT / 'synapse' / 'cli.py'
    assert ROOT / 'synapse' / 'neuroweave' / 'cli_commands.py' in sources
    main_cli = tmp_path / 'cli.py'
    extra_cli = tmp_path / 'cli_commands.py'
    main_cli.write_text("sub.add_parser('alpha'); p.add_argument('--beta', type=int)\n", encoding='utf-8')
    extra_cli.write_text("parser.add_argument(\n    '--gamma')\nsub.add_parser( \"delta\")\n", encoding='utf-8')
    assert tool._cli_surfaces([main_cli, extra_cli]) == (['alpha', 'delta'], ['--beta', '--gamma'])


def test_check_flags_removed_route_and_changed_default(tool, current):
    broken = copy.deepcopy(current)
    removed_route = broken['http_routes'].pop(0)
    broken['config_defaults']['seed'] = current['config_defaults']['seed'] + 1
    del broken['config_defaults']['total_neurons']
    problems = tool.check(current, broken)
    assert f'http_routes missing: {removed_route}' in problems
    assert any(p.startswith('config default changed: seed') for p in problems)
    assert 'config field removed: total_neurons' in problems
    assert tool.check(current, current) == []


@pytest.mark.parametrize('path, tag', [
    (PRE_NEUROWEAVE, 'sonapse-pre-neuroweave-baseline'),
    (PRE_FRONTIER, 'sonapse-pre-frontier-baseline'),
])
def test_both_compatibility_manifests_pass_on_current_tree(tool, current, path, tag):
    manifest = json.loads(path.read_text(encoding='utf-8'))
    assert SECTIONS <= set(manifest)
    assert manifest['baseline_tag'] == tag
    assert tool.check(manifest, current) == []


def test_pre_frontier_manifest_records_baseline_commit():
    manifest = json.loads(PRE_FRONTIER.read_text(encoding='utf-8'))
    assert manifest['baseline_commit'] == 'b9d8270'
    assert manifest['baseline_commit_full'].startswith('b9d8270')
    # The pre-Frontier contract is a superset of the pre-NEUROWEAVE contract.
    older = json.loads(PRE_NEUROWEAVE.read_text(encoding='utf-8'))
    for section in ('cli_commands', 'cli_flags', 'web_dom_ids', 'checkpoint_namespaces', 'test_modules'):
        assert set(older[section]) <= set(manifest[section]), section


def test_cli_manifest_and_baseline_tag_flags(tool, tmp_path, capsys):
    assert tool.main(['--check']) == 0
    assert tool.main(['--manifest', str(PRE_FRONTIER), '--baseline-tag', 'sonapse-pre-frontier-baseline',
                      '--check']) == 0
    assert tool.main(['--manifest', 'docs/compatibility/manifest-pre-frontier.json', '--check']) == 0
    # An explicit tag that the manifest does not record is a failure.
    assert tool.main(['--manifest', str(PRE_FRONTIER), '--baseline-tag', 'sonapse-pre-neuroweave-baseline',
                      '--check']) == 1
    assert 'baseline tag mismatch' in capsys.readouterr().out
    written = tmp_path / 'nested' / 'manifest.json'
    assert tool.main(['--write', '--manifest', str(written), '--baseline-tag', 'example-tag']) == 0
    data = json.loads(written.read_text(encoding='utf-8'))
    assert data['baseline_tag'] == 'example-tag'
    assert SECTIONS <= set(data)
    assert tool.main(['--check', '--manifest', str(written), '--baseline-tag', 'example-tag']) == 0


# --- release metadata (version 0.3.0 is declared consistently) -----------------------------------

def test_release_version_is_consistent(capsys):
    import synapse
    from synapse.cli import main

    pyproject = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    citation = (ROOT / 'CITATION.cff').read_text(encoding='utf-8')
    version = re.search(r'^version = "([^"]+)"', pyproject, flags=re.M).group(1)
    assert version == synapse.__version__ == '0.3.0'
    assert re.search(r'^version: "?0\.3\.0"?$', citation, flags=re.M)
    with pytest.raises(SystemExit) as exc:
        main(['--version'])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == 'synapse-connectome-lab 0.3.0'


def test_workflow_actions_are_pinned_to_commit_shas():
    workflow = (ROOT / '.github' / 'workflows' / 'regression.yml').read_text(encoding='utf-8')
    uses = re.findall(r'^\s*(?:-\s+)?uses:\s*(\S+)(.*)$', workflow, flags=re.M)
    assert uses, 'no actions found'
    for ref, comment in uses:
        assert re.fullmatch(r'[\w.-]+/[\w.-]+@[0-9a-f]{40}', ref), ref
        assert re.fullmatch(r'\s+# v\d+\.\d+\.\d+', comment), (ref, comment)


# --- workbench theme layer and accessibility (static checks; no browser) -----------------------

WEB = ROOT / 'synapse' / 'web'
LIGHT_SCOPE = ':where(:root[data-theme="light"])'
SYSTEM_SCOPE = ':where(:root[data-theme="system"])'
THEME_MARKER = 'Accessibility helpers and theme layer (additive).'
THEME_END = '/* End of theme layer. */'
# Elements that stay dark in every theme (canvas drawing code and inline colours assume a dark
# background) and progress-bar fills, which carry no text.
DARK_ISLANDS = ('#chat-stream', '#chat-input', '.canvas-tooltip', '-canvas')
NON_TEXT = ('progress', '.vital-', '.nwl-track')
A11Y_HELPERS = {
    '.visually-hidden:where(:not(:focus, :focus-within, :active))', '.skip-link:focus',
    ':where(a:any-link, summary, [tabindex="-1"]):focus-visible', '.theme-control', '.theme-select',
}


def _web(name):
    return (WEB / name).read_text(encoding='utf-8')


def _rgb(value):
    value = value.strip().lstrip('#')
    if len(value) == 3:
        value = ''.join(ch * 2 for ch in value)
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _luminance(rgb):
    def channel(v):
        v /= 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b):
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _rgba(text):
    r, g, b, a = re.fullmatch(r'rgba\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)', text.strip()).groups()
    return float(r), float(g), float(b), float(a)


def _over(rgba, base):
    *rgb, alpha = rgba
    return tuple(round(alpha * x + (1 - alpha) * y) for x, y in zip(rgb, base))


def _decls(body):
    return {k.strip(): v.replace('!important', '').strip() for k, v in (d.split(':', 1) for d in body.split(';') if ':' in d)}


def _flat_rules(css):
    """(selector, body) for every rule, one @media level flattened, comments removed."""
    css = re.sub(r'/\*.*?\*/', '', css, flags=re.S)
    rules, i = [], 0
    while (j := css.find('{', i)) >= 0:
        depth, k = 1, j + 1
        while depth and k < len(css):
            depth += {'{': 1, '}': -1}.get(css[k], 0)
            k += 1
        selector, body = ' '.join(css[i:j].split()), css[j + 1:k - 1]
        if selector.startswith('@'):
            rules.extend((f'{selector} {inner}', inner_body) for inner, inner_body in _flat_rules(body))
        else:
            rules.append((selector, body))
        i = k
    return rules


def _theme_layer():
    css = _web('style.css')
    start = css.index(THEME_MARKER)
    return css[css.rindex('/*', 0, start):css.index(THEME_END, start)]


def _light_tokens(css):
    block = re.search(r'^:root\[data-theme="light"\] \{(.*?)^\}', css, flags=re.S | re.M).group(1)
    return dict(re.findall(r'(--[\w-]+):\s*(#[0-9a-fA-F]{6})\s*;', block))


def test_theme_selector_defaults_to_dark_and_keeps_first_load_unchanged():
    html = _web('index.html')
    assert re.search(r'<html lang="en">', html), 'the root element must not carry a data-theme attribute'
    select = re.search(r'<select id="theme-select"[^>]*>(.*?)</select>', html, flags=re.S).group(1)
    assert re.findall(r'<option value="(\w+)"', select) == ['dark', 'light', 'system']
    assert re.search(r'<option value="dark" selected>', select)
    assert re.search(r'<label for="theme-select"', html)
    # Every colour override in the appended layer is scoped to an explicit data-theme value, and the
    # prefers-color-scheme query only applies while "system" is chosen.
    for selector, body in _flat_rules(_theme_layer()):
        if selector in A11Y_HELPERS or selector.startswith('@media (max-width'):
            continue
        assert 'data-theme=' in selector, selector
        if 'prefers-color-scheme' in selector:
            assert selector.startswith('@media (prefers-color-scheme: light)'), selector
            assert 'data-theme="system"' in selector and 'data-theme="light"' not in selector, selector
    base = _web('style.css')[:_web('style.css').index(THEME_MARKER)]
    assert 'data-theme' not in base and 'prefers-color-scheme' not in base


def test_system_theme_mirrors_light_theme_exactly():
    layer = _theme_layer()
    light = [(s.replace(LIGHT_SCOPE, '@'), _decls(b)) for s, b in _flat_rules(layer) if LIGHT_SCOPE in s]
    system = [(s.replace('@media (prefers-color-scheme: light) ', '').replace(SYSTEM_SCOPE, '@'), _decls(b))
              for s, b in _flat_rules(layer) if SYSTEM_SCOPE in s]
    assert light and light == system
    media = layer[layer.index('@media (prefers-color-scheme: light)'):]
    sys_tokens = dict(re.findall(r'(--[\w-]+):\s*(#[0-9a-fA-F]{6})\s*;', media.split('}', 1)[0]))
    assert sys_tokens == _light_tokens(layer)


def test_light_theme_text_contrast_meets_wcag_aa():
    css = _web('style.css')
    tokens = _light_tokens(css)

    def resolve(value):
        match = re.fullmatch(r'var\((--[\w-]+)\)', value.strip())
        return tokens[match.group(1)] if match else value.strip()

    texts = {k: tokens[k] for k in ('--text', '--muted', '--teal', '--amber', '--rose', '--purple', '--blue')}
    surfaces = {k: tokens[k] for k in ('--bg', '--panel', '--panel-hover')}
    bases, translucent, accents = [tokens['--bg'], tokens['--panel']], {}, [tokens['--teal']]
    for selector, body in _flat_rules(_theme_layer()):
        if LIGHT_SCOPE not in selector or any(t in selector for t in DARK_ISLANDS + NON_TEXT):
            continue
        name, decls = selector.replace(LIGHT_SCOPE, '').strip(), _decls(body)
        if 'color' in decls and resolve(decls['color']).lower() != '#ffffff':
            texts[name] = resolve(decls['color'])
        background = decls.get('background')
        if not background:
            continue
        if name == '.primary:hover':
            accents.append(resolve(background))
            continue
        if name == 'aside':
            bases.append(resolve(background))
        surfaces.update({f'{name} {stop}': stop for stop in re.findall(r'#[0-9a-fA-F]{6}\b', background)})
        for index, fill in enumerate(re.findall(r'rgba\([^)]*\)', background)):
            translucent[f'{name} #{index}'] = _rgba(fill)
    for name, rgba in translucent.items():
        for base in bases:
            surfaces[f'{name} over {base}'] = '#%02x%02x%02x' % _over(rgba, _rgb(base))
    assert len(texts) >= 30 and len(surfaces) >= 30
    worst = min((_contrast(_rgb(t), _rgb(s)), tn, sn) for tn, t in texts.items() for sn, s in surfaces.items())
    assert worst[0] >= 4.5, worst
    # White text on accent fills (.primary, active view toggle, .primary:hover).
    assert min(_contrast((255, 255, 255), _rgb(fill)) for fill in accents) >= 4.5
    # Omnibus buttons keep inline translucent tints (index.html and app.js) over the white panel.
    html, js = _web('index.html'), _web('app.js')
    omni_text = _rgb(_decls(dict(_flat_rules(_theme_layer()))[f'{LIGHT_SCOPE} .omnibus-button'])['color'])
    tints = re.findall(r'class="[^"]*omnibus-button[^"]*"[^>]*style="[^"]*background:\s*(rgba\([^)]*\))', html)
    omni_ids = set(re.findall(r'id="([^"]+)"[^>]*class="[^"]*omnibus-button', html))
    tints += [t for i, t in re.findall(r"\$\('([\w-]+)'\)\.style\.background\s*=\s*'(rgba\([^)]*\))'", js) if i in omni_ids]
    assert len(tints) >= 10
    assert min(_contrast(omni_text, _over(_rgba(t), _rgb(tokens['--panel']))) for t in tints) >= 4.5
    # The chat console stays a dark island: its own and its script-set colours stay readable.
    island = [('#e6edf4', '#0d1722'), ('#8b9daf', '#0d1722'), ('#8b9daf', '#0b141e')]
    island += [(c, '#0d1722') for c in re.findall(r"Div\.style\.color\s*=\s*'(#[0-9a-fA-F]{6})'", js)]
    assert len(island) >= 6
    assert min(_contrast(_rgb(t), _rgb(s)) for t, s in island) >= 4.5


def test_theme_script_guards_storage_and_tab_handler_sets_aria_current():
    js = _web('app.js')
    start = js.index('Theme selector (additive)')
    block = js[start:js.index('let state = {};', start)]
    calls = [m.start() for m in re.finditer(r'localStorage\.', block)]
    assert len(calls) == 3  # getItem, removeItem, setItem
    for pos in calls:
        opened = block.rfind('try {', 0, pos)
        assert opened >= 0 and block.find('} catch', opened) > pos, block[max(0, pos - 80):pos + 40]
    assert "if (choice === 'dark') root.removeAttribute('data-theme');" in block
    handler = js[js.index("document.querySelectorAll('nav button').forEach(b => b.onclick"):]
    handler = handler[:handler.index("$('title').textContent")]
    assert "n.removeAttribute('aria-current')" in handler
    assert "b.setAttribute('aria-current', 'page');" in handler
    html = _web('index.html')
    nav = re.search(r'<nav aria-label="Primary">(.*?)</nav>', html, flags=re.S).group(1)
    assert re.findall(r'aria-current="page"', nav) == ['aria-current="page"']
    assert re.search(r'<button data-tab="overview" class="active" aria-current="page">', nav)


def test_skip_link_focus_target_and_live_status_lines():
    html = _web('index.html')
    skip = re.search(r'<a href="#(\w+)" class="skip-link visually-hidden">', html)
    assert skip and html.index(skip.group(0)) < html.index('<aside>')
    assert re.search(rf'<h1 id="{skip.group(1)}" tabindex="-1">', html)
    for status_id in ('connection', 'runstatus', 'tandem-status-pill', 'export-status', 'training-job-status',
                      'rsi-status', 'nwl-status', 'audio-status-badge'):
        tag = re.search(rf'<[a-z]+ [^>]*id="{status_id}"[^>]*>', html).group(0)
        assert 'role="status"' in tag or 'aria-live=' in tag, tag
    # Buttons whose visible text carries a pictograph also expose a spoken name that contains it.
    for label, visible in re.findall(r'<button [^>]*aria-label="([^"]+)"[^>]*>([^<]+)</button>', html):
        words = re.findall(r'[A-Za-z]{2,}', visible)
        assert all(w.lower() in label.lower() for w in words), (label, visible)
    css = _web('style.css')
    assert re.search(r'button:focus-visible', css) and re.search(r'\[tabindex="-1"\]\):focus-visible', css)
