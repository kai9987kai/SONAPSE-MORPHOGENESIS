"""NEUROWEAVE Lab workbench API (/api/neuroweave/*) and workbench UI wiring (small and fast)."""
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from synapse.neuroweave.spec import EXPERIMENT_IDS, MECHANISM_IDS, NeuroWeaveSpecV1
from synapse.neuroweave.workbench_api import MAX_SENSORY_BYTES, validate_sensory_payload
from synapse.server import Workbench, create_server

WEB = Path(__file__).resolve().parents[1] / 'synapse' / 'web'


class Runtime:
    """Minimal runtime (as in tests/test_server.py): no NEUROWEAVE channel, so the lab engine is observed."""
    def snapshot(self): return {'step': 0}
    def step(self, steps=1, learn=False): return {'step': steps}


@pytest.fixture(scope='module')
def app(tmp_path_factory):
    wb = Workbench(Runtime(), output_root=tmp_path_factory.mktemp('nwserver'))
    # small lab engine so every endpoint stays fast
    wb.post('/api/neuroweave/configure', {'spec': NeuroWeaveSpecV1.all_mechanisms(
        enabled=True, neurons=24, max_neurons=32, seed=5).to_dict()})
    return wb


def _wait_job(app, ident, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = next(j for j in app.get('/api/jobs')['jobs'] if j['id'] == ident)
        if job['status'] not in ('queued', 'running'):
            return job
        time.sleep(0.2)
    raise AssertionError('job did not finish')


def test_catalog_status_and_lab_observatory(app):
    cat = app.get('/api/neuroweave/catalog')
    assert [m['id'] for m in cat['mechanisms']] == list(MECHANISM_IDS)
    assert [e['id'] for e in cat['experiments']] == list(EXPERIMENT_IDS)
    assert len(cat['topologies']) == 12 and len(cat['sensory_channels']) == 16
    status = app.get('/api/neuroweave/status')
    assert status['observed_source'] == 'lab' and status['live']['enabled'] is False
    assert status['lab']['spec']['neurons'] == 24
    obs = app.get('/api/neuroweave/observatory')
    assert obs['source'] == 'lab' and obs['evidence_tier'] == 'synthetic_abstraction'
    conn = obs['neural']['region_connectivity']
    assert len(conn['counts']) == 10 and sum(map(sum, conn['counts'])) == obs['neural']['synapses']['count']
    for section in ('neural', 'neuromodulation', 'sensory', 'learning', 'memory', 'prediction', 'events'):
        assert section in obs
    json.dumps(obs, allow_nan=False)
    assert app.get('/api/neuroweave/observatory/lab')['source'] == 'lab'
    with pytest.raises(KeyError):
        app.get('/api/neuroweave/nope')


def test_step_ledger_and_ecology(app):
    before = app.get('/api/neuroweave/status')['lab']['telemetry']['step']
    r = app.post('/api/neuroweave/step', {'steps': 3, 'reward': 0.2})
    assert r['step'] == before + 3 and r['source'] == 'lab'
    for bad in ({'steps': 0}, {'steps': 257}, {'steps': True}, {'reward': 5}, {'nope': 1}, {'task_input': [0] * 7}):
        with pytest.raises(ValueError):
            app.post('/api/neuroweave/step', bad)
    led = app.get('/api/neuroweave/ledger')
    assert led['source'] == 'lab' and set(led) >= {'total', 'by_kind', 'events'}
    eco = app.post('/api/neuroweave/ecology', {'config': 'default', 'steps': 3})
    scene = eco['scene']
    assert scene['width'] == 1200 and scene['height'] == 660 and len(scene['agents']) == 1
    assert scene['food'] and scene['raw_media_stored'] is False
    assert app.get('/api/neuroweave/ecology')['attached'] is True
    with pytest.raises(ValueError):
        app.post('/api/neuroweave/ecology', {'config': 'nowhere'})


def test_configure_replaces_only_the_lab_engine_and_rejects_bad_specs(app):
    lab_before = app.get('/api/neuroweave/status')['lab']['spec']
    for bad in ({'spec': {'neurons': 3}}, {'spec': {'mechanisms': ['not_a_mechanism']}}, {'spec': {'bogus': 1}},
                {'spec': {'schema': 'Other'}}, {'spec': {'neurons': 4096, 'max_neurons': 4096}}, {'spec': 'x'},
                {'sensory_modes': {'visual_luminance': 'telepathy'}}, {'sensory_modes': {'not_a_channel': 'live'}},
                {}, {'other': 1}):
        with pytest.raises(ValueError):
            app.post('/api/neuroweave/configure', bad)
    assert app.get('/api/neuroweave/status')['lab']['spec'] == lab_before   # nothing swapped
    r = app.post('/api/neuroweave/configure', {'sensory_modes': {'visual_luminance': 'live', 'thermal': 'external'}})
    assert ['visual_luminance', 'live'] in r['spec']['sensory_modes'] and r['live_runtime_changed'] is False
    assert r['spec']['mechanisms'] == lab_before['mechanisms']


def test_sensory_accepts_scalars_and_rejects_raw_media(app):
    app.post('/api/neuroweave/configure', {'sensory_modes': {'visual_luminance': 'live', 'thermal': 'external'}})
    r = app.post('/api/neuroweave/sensory', {'visual': {'luminance': 0.4, 'motion': 0.1},
                                             'audio': {'low': 0.2, 'amplitude': 0.3},
                                             'external': {'thermal': 0.25}})
    assert r['accepted'] and r['raw_media_stored'] is False
    assert set(r['reaches_engine']) == {'visual_luminance', 'thermal'}
    assert r['applied']['audio_low'] == pytest.approx(0.2)
    rejects = [
        {'visual': {'frame': [0.1] * 10}},                       # raw-media key
        {'audio': {'low': 0.1}, 'samples': [0.0] * 32},          # raw-media key at top level
        {'visual': {'luminance': [0.1] * 65}},                   # array longer than 64
        {'external': {'thermal': 'data:image/png;base64,AAAA'}},  # data URL
        {'visual': {'luminance': 0.1, 'pad': 'x' * 100}},        # unknown feature
        {'image': 'x'}, {'visual': {}}, {}, {'visual': {'luminance': float('nan')}},
        {'external': {'not_a_channel': 0.1}}, {'visual': {'luminance': True}},
    ]
    for bad in rejects:
        with pytest.raises(ValueError):
            app.post('/api/neuroweave/sensory', bad)
    big = {'external': {'thermal': 0.1}, 'visual': {'luminance': 0.1, 'motion': 0.1}}
    big['visual'] = {'luminance': 0.1}
    huge = {'external': {f'k{i}': 0.0 for i in range(6000)}}
    assert len(json.dumps(huge)) > MAX_SENSORY_BYTES
    with pytest.raises(ValueError, match='limit'):
        validate_sensory_payload(huge)
    assert validate_sensory_payload(big)['visual'] == {'luminance': 0.1}


def test_export_and_read_only_inspect(app):
    status = app.get('/api/neuroweave/status')
    pkg = app.post('/api/neuroweave/export', {'kind': 'package'})
    assert pkg['valid'] is True and pkg['mime'] == 'application/json' and pkg['filename'].endswith('.json')
    html = app.post('/api/neuroweave/export', {'kind': 'html'})
    assert html['text'].lstrip().lower().startswith('<!doctype html') and html['not_executed']
    fpga = app.post('/api/neuroweave/export', {'kind': 'fpga'})
    manifest = json.loads(fpga['text'])
    assert fpga['valid'] is True and manifest['rtl_generated'] is False and manifest['hardware_validated'] is False
    with pytest.raises(ValueError):
        app.post('/api/neuroweave/export', {'kind': 'exe'})
    with pytest.raises(ValueError):
        app.post('/api/neuroweave/export', {'kind': 'package', 'source': 'live'})   # no live channel
    report = app.post('/api/neuroweave/inspect', {'text': pkg['text']})
    assert report['read_only'] is True and report['report']['valid'] is True
    assert app.get('/api/neuroweave/status')['lab']['telemetry'] == status['lab']['telemetry']   # untouched
    assert app.post('/api/neuroweave/inspect', {'text': '{not json'})['report']['valid'] is False
    with pytest.raises(ValueError):
        app.post('/api/neuroweave/inspect', {'text': 'x' * 1_000_001})
    with pytest.raises(ValueError):
        app.post('/api/neuroweave/inspect', {'text': 5})


def test_bounded_experiment_topology_and_lesion(app):
    r = app.post('/api/neuroweave/experiment', {'id': 'meta_plasticity', 'seeds': 2, 'steps': 6, 'quick': True,
                                                'mechanisms': ['homeostatic_governor']})
    row = r['rows'][0]
    assert row['experiment_id'] == 'meta_plasticity' and row['verdict'] in ('improved', 'no_detectable_difference', 'worse')
    assert len(row['ci95']) == 2 and r['receipts'][0]['digest'] == row['digest']
    assert r['csv'].startswith('experiment_id,')
    for bad in ({'id': 'nope'}, {'id': 'meta_plasticity', 'steps': 1000}, {'id': 'meta_plasticity', 'seeds': 9},
                {'id': 'all'}, {'id': 'meta_plasticity', 'mechanisms': ['x']}, {'id': 'meta_plasticity', 'zz': 1}):
        with pytest.raises(ValueError):
            app.post('/api/neuroweave/experiment', bad)
    topo = app.post('/api/neuroweave/topology', {'neurons': 16, 'seeds': 1})
    assert len(topo['topologies']) == 12
    with pytest.raises(ValueError):
        app.post('/api/neuroweave/topology', {'neurons': 100000})
    job = app.post('/api/neuroweave/lesion', {'quick': True, 'steps': 8, 'neurons': 16, 'background': True})
    assert job['kind'] == 'neuroweave_lesion'
    done = _wait_job(app, job['id'])
    assert done['status'] == 'completed', done.get('error')
    assert done['result']['result']['originals_untouched'] is True
    assert done['result']['options']['steps'] == 8


def test_http_routes_and_raw_media_rejection(app):
    server = create_server(port=0, workbench=app)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_port}'

    def post(path, payload):
        req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        return urllib.request.urlopen(req)
    try:
        assert json.load(urllib.request.urlopen(base + '/api/neuroweave/catalog'))['schema'] == 'NeuroWeaveWorkbenchApiV1'
        assert json.load(post('/api/neuroweave/step', {'steps': 1}))['steps'] == 1
        for payload in ({'audio': {'low': 0.1}, 'samples': [0.0] * 512}, {'spec': {}}):
            path = '/api/neuroweave/sensory' if 'audio' in payload else '/api/neuroweave/nothing'
            with pytest.raises(urllib.error.HTTPError) as err:
                post(path, payload)
            assert err.value.code == (400 if 'audio' in payload else 404)
        with pytest.raises(urllib.error.HTTPError) as err:
            post('/api/neuroweave/configure', {'spec': {'neurons': 1}})
        assert err.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_live_runtime_engine_is_observed_but_not_stepped(tmp_path):
    from synapse.config import SynapseConfig
    from synapse.runtime import LabRuntime
    spec = NeuroWeaveSpecV1(enabled=True, neurons=16, max_neurons=16, seed=3, gain=0.0,
                            mechanisms=('multimodal_sensory',))
    runtime = LabRuntime(SynapseConfig(nodes_per_source=16, neuroweave=spec), tmp_path)
    wb = Workbench(runtime, output_root=tmp_path)
    with wb.lock:
        runtime.step(2, learn=False)
    engine = runtime.agent.neuroweave_channel.engine
    step = engine.step_count
    status = wb.get('/api/neuroweave/status')
    assert status['observed_source'] == 'live' and status['live']['configured'] is True
    obs = wb.get('/api/neuroweave/observatory')
    assert obs['source'] == 'live' and obs['spec']['neurons'] == 16
    assert wb.get('/api/neuroweave/ledger')['source'] == 'live'
    wb.post('/api/neuroweave/step', {'steps': 2})            # steps the lab engine only
    wb.post('/api/neuroweave/sensory', {'external': {'thermal': 0.3}})
    assert engine.step_count == step and runtime.agent.neuroweave_channel.engine is engine
    assert wb.get('/api/neuroweave/observatory/lab')['source'] == 'lab'
    assert wb.post('/api/neuroweave/export', {'kind': 'fpga', 'source': 'live'})['source'] == 'live'


def test_web_ui_ids_and_handlers():
    html = (WEB / 'index.html').read_text(encoding='utf-8')
    js = (WEB / 'app.js').read_text(encoding='utf-8')
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    assert 'neuroweave' in ids and 'data-tab="neuroweave"' in html
    referenced = set(re.findall(r"\$\('(nwl-[A-Za-z0-9_-]+)'\)", js))
    assert referenced and referenced <= ids, sorted(referenced - ids)
    buttons = re.findall(r'<button[^>]*\bid="(nwl-[^"]+)"', html)
    assert len(buttons) >= 20
    for bid in buttons:
        assert f"$('{bid}').onclick" in js, f'{bid} has no handler'
    for control in ('nwl-auto', 'nwl-import-file'):
        assert f"$('{control}').on" in js
    panels = re.findall(r'data-nwpanel="([a-z]+)"', html)
    assert len(panels) == 7 and all(f'id="nwl-panel-{p}"' in html for p in panels)
    # every NEUROWEAVE endpoint the UI calls exists in the API
    from synapse.neuroweave import workbench_api
    src = Path(workbench_api.__file__).read_text(encoding='utf-8')
    for path in set(re.findall(r"api\('(neuroweave/[a-z/]+)'", js)):
        assert f"'/api/{path}'" in src, path
    # raw media never leaves the browser: sensory posts carry only scalar feature dicts
    assert 'getImageData(0, 0, 32, 24)' in js and "api('audio/push'" in js   # existing mic path untouched
    sensory_calls = re.findall(r'nwlPostSensory\(\{ ?(\w+):', js)
    assert set(sensory_calls) == {'visual', 'audio', 'external'}
