"""Paired real-world planning pilots; receipt verification checks arithmetic/hashes.

This is a development experiment, not confirmation or receipt authentication.
Every arm receives the same supported learner and independent paired query worlds.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from .adaptation_study import (_seed, _json_hash, _file_hash, _source_fingerprints,
                               _environment, _parameters)
from .stateio import digest, _atomic


DEFAULTS_V1 = {'seeds': [23, 29, 31], 'tasks': ['forage', 'push'], 'support_steps': 64,
               'query_steps': 64, 'nodes_per_source': 16, 'horizons': [1, 3]}
DEFAULTS = {'seeds': [23, 29, 31], 'tasks': ['forage', 'delayed_match', 'button_sequence'],
            'support_steps': 64, 'query_steps': 256, 'nodes_per_source': 16,
            'horizons': [1, 3], 'task_horizon': 3}


def validate_config(config=None, version=2):
    from .complex_world import ALL_TASKS
    defaults = DEFAULTS_V1 if version == 1 else DEFAULTS
    if config is None:
        config = {}
    if not isinstance(config, dict) or set(config)-set(defaults):
        raise ValueError('unknown planning-study configuration')
    c = deepcopy({**defaults, **config})
    for key, lo, hi in [('seeds', 0, 2**31-1), ('horizons', 1, 5)]:
        values = c[key]
        limit = 8 if key == 'seeds' else 3
        if (not isinstance(values, list) or not 1 <= len(values) <= limit
                or any(type(v) is not int or not lo <= v <= hi for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f'{key} must contain unique bounded integers')
    tasks = c['tasks']
    if (not isinstance(tasks, list) or not 1 <= len(tasks) <= 4
            or any(t not in ALL_TASKS for t in tasks) or len(set(tasks)) != len(tasks)):
        raise ValueError('tasks must contain 1..4 unique supported world tasks')
    for key, lo, hi in [('support_steps', 8, 512),
                        ('query_steps', 8, 512 if version == 1 else 256),
                        ('nodes_per_source', 16, 200)]:
        if type(c[key]) is not int or not lo <= c[key] <= hi:
            raise ValueError(f'{key} must be an integer in [{lo},{hi}]')
    if version == 1:
        budget = len(c['seeds'])*len(tasks)*(c['support_steps']+(1+len(c['horizons']))*c['query_steps'])
        if budget > 16384:
            raise ValueError('planning study exceeds 16384 executed transitions')
    else:
        task_horizon = c['task_horizon']
        if type(task_horizon) is not int or not 1 <= task_horizon <= 5:
            raise ValueError('task_horizon must be an integer in [1,5]')
        budget = len(c['seeds'])*len(tasks)*(c['support_steps']+(2+len(c['horizons']))*c['query_steps'])
        if budget > 32768:
            raise ValueError('planning study exceeds 32768 executed transitions')
    return c


def _learned_hash(agent):
    return digest({'parameters': _parameters(agent), 'optimizer': agent.opt.state_dict(),
                   'model': agent.experience_model.export_state(),
                   'task_aware_model': agent.task_aware_model.export_state(),
                   'memory': agent.surprise_memory.export_state(),
                   'structural': agent.neurogenesis.export_state(),
                   'pre': agent.graph.graft_pre, 'post': agent.graph.graft_post})


def _arms_v1(config):
    return ['actor']+[f'horizon_{h}' for h in config['horizons']]


def _summarize_v1(records, config):
    from .evidence_engine import bootstrap_confidence_interval
    units = [(s, t) for s in config['seeds'] for t in config['tasks']]
    index = {(r['seed'], r['task'], r['arm']): r for r in records}
    expected = {(s, t, a) for s, t in units for a in _arms_v1(config)}
    if len(records) != len(expected) or set(index) != expected:
        raise ValueError('missing or duplicate paired records')
    if any(type(r['steps']) is not int or r['steps'] != config['query_steps'] for r in records):
        raise ValueError('invalid query step count')
    arms = {}
    for arm in _arms_v1(config):
        rows = [index[s, t, arm] for s, t in units]
        arms[arm] = {'mean_return': float(np.mean([r['return'] for r in rows])),
                     'returns': [r['return'] for r in rows],
                     'planner_use_fraction': float(np.mean([r['planner_used_steps']/r['steps'] for r in rows])),
                     'mean_model_evaluations': float(np.mean([r['model_evaluations']/r['steps'] for r in rows]))}
    comparisons = {}
    for arm in _arms_v1(config)[1:]:
        differences = np.asarray(arms[arm]['returns'])-arms['actor']['returns']
        comparisons[arm+'_minus_actor'] = {
            'paired_differences': differences.tolist(), 'mean_difference': float(differences.mean()),
            'descriptive_bootstrap_95': list(bootstrap_confidence_interval(differences)) if len(units) > 1 else None}
    return {'arms': arms, 'comparisons': comparisons, 'paired_units': len(units),
            'interpretation': 'finite development pilot; shared task types; no automatic promotion; no general performance claim'}


def _arms(config):
    return ['actor']+[f'horizon_{h}' for h in config['horizons']]+[f"task_value_h{config['task_horizon']}"]


def _paired(differences):
    from .evidence_engine import bootstrap_confidence_interval
    values = np.asarray(differences, dtype=np.float64)
    return {'paired_differences': values.tolist(),
            'mean_difference': float(values.mean()),
            'descriptive_bootstrap_95': (list(bootstrap_confidence_interval(values))
                                         if len(values) > 1 else None)}


def _arm_summary(rows):
    outcomes = [row['episode_outcome'] for row in rows]
    total_steps = sum(row['steps'] for row in rows)
    return {'episodes': len(rows),
            'success_rate': float(np.mean([outcome == 'success' for outcome in outcomes])),
            'mean_return': float(np.mean([row['return'] for row in rows])),
            'mean_steps': float(np.mean([row['steps'] for row in rows])),
            'outcome_counts': {name: outcomes.count(name) for name in ('success','fail','timeout','incomplete')},
            'planner_use_fraction': (float(sum(row['planner_used_steps'] for row in rows)/total_steps) if total_steps else 0.0),
            'planner_abstention_fraction': (float(sum(row['planner_abstained_steps'] for row in rows)/total_steps) if total_steps else 0.0),
            'mean_model_evaluations_per_step': (float(sum(row['model_evaluations'] for row in rows)/total_steps) if total_steps else 0.0),
            'mean_member_evaluations_per_episode': float(np.mean([row['member_evaluations'] for row in rows]))}


def _summarize(records, config):
    units = [(seed, task) for seed in config['seeds'] for task in config['tasks']]
    index = {(row['seed'], row['task'], row['arm']): row for row in records}
    expected = {(seed, task, arm) for seed, task in units for arm in _arms(config)}
    if len(records) != len(expected) or set(index) != expected:
        raise ValueError('missing or duplicate paired records')
    arms = {arm: _arm_summary([index[seed, task, arm] for seed, task in units])
            for arm in _arms(config)}
    comparisons = {}
    for arm in _arms(config)[1:]:
        rows = [index[seed, task, arm] for seed, task in units]
        baseline = [index[seed, task, 'actor'] for seed, task in units]
        comparisons[arm+'_minus_actor'] = {
            'completion': _paired([float(a['episode_outcome']=='success')-float(b['episode_outcome']=='success')
                                   for a,b in zip(rows,baseline)]),
            'return': _paired([a['return']-b['return'] for a,b in zip(rows,baseline)])}
    task_summaries = {}
    task_comparisons = {}
    for task in config['tasks']:
        task_units = [(seed, task) for seed in config['seeds']]
        task_summaries[task] = {arm: _arm_summary([index[seed, task, arm] for seed, _ in task_units])
                                for arm in _arms(config)}
        task_comparisons[task] = {}
        for arm in _arms(config)[1:]:
            rows = [index[seed, task, arm] for seed, _ in task_units]
            baseline = [index[seed, task, 'actor'] for seed, _ in task_units]
            task_comparisons[task][arm+'_minus_actor'] = {
                'completion': _paired([float(a['episode_outcome']=='success')-float(b['episode_outcome']=='success')
                                       for a,b in zip(rows,baseline)]),
                'return': _paired([a['return']-b['return'] for a,b in zip(rows,baseline)])}
    task_arm = f"task_value_h{config['task_horizon']}"
    completion = comparisons[task_arm+'_minus_actor']['completion']
    forage = task_comparisons.get('forage',{}).get(task_arm+'_minus_actor',{}).get('completion',{}).get('mean_difference')
    promoted = bool(completion['mean_difference'] >= 0.10
                    and completion['descriptive_bootstrap_95'] is not None
                    and completion['descriptive_bootstrap_95'][0] > 0
                    and forage is not None and forage >= -0.05)
    return {'arms': arms, 'task_summaries': task_summaries,
            'comparisons': comparisons, 'task_comparisons': task_comparisons,
            'paired_units': len(units),
            'promotion': {'rule': 'task-aware pooled completion improves by at least 10 percentage points with a positive paired descriptive 95% bootstrap lower bound; forage completion is no worse than 5 percentage points; executed transitions stay within 32,768',
                          'pooled_completion_improvement': completion['mean_difference'],
                          'pooled_completion_interval': completion['descriptive_bootstrap_95'],
                          'forage_completion_difference': forage,
                          'eligible': promoted},
            'interpretation': 'finite local synthetic-world development study; no real-world reliability or general capability claim'}


def run_planning_study(output_dir=None, config=None):
    from .compute import configure_torch
    thread_configuration=configure_torch(1)
    from .config import SynapseConfig
    from .runtime import LabRuntime
    from .complex_world import ComplexWorld
    c = validate_config(config, version=2)
    started = time.perf_counter()
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=False)
    records, supports, files = [], [], {}
    for seed in c['seeds']:
        for task in c['tasks']:
            agent_seed = _seed('planning-agent', seed, task)
            support_seed = _seed('planning-support', seed, task)
            query_seed = _seed('planning-query', seed, task)
            cfg = SynapseConfig(seed=agent_seed, task=task, nodes_per_source=c['nodes_per_source'])
            support = LabRuntime(cfg, output_root=out)
            support.world = ComplexWorld(seed=support_seed, curriculum=task)
            support.obs = support.world.observe()
            rng = np.random.default_rng(_seed('planning-exploration', seed, task))
            trace = []
            # Manual actions exercise the same validated learning boundary as the UI.
            for _ in range(c['support_steps']):
                previous = support.obs.tolist()
                state = support.step(1, learn=True, action=rng.uniform(-.8, .8, 5))
                row = state['recent'][-1]
                item={'observation': previous, 'action': state['action'],
                      'reward': row['reward'], 'done': row['done']}
                if 'outcome' in row:
                    item['outcome']=row['outcome']
                trace.append(item)
            trained = support.agent.export_state()
            support_digest = _json_hash(trace)
            unit = {'seed': seed, 'task': task, 'agent_seed': agent_seed,
                    'support_seed': support_seed, 'query_seed': query_seed,
                    'support_digest': support_digest, 'trace': trace,
                    'learned_state_digest': _learned_hash(support.agent),
                    'legacy_model_summary': support.agent.experience_model.summary(),
                    'task_aware_model_summary': support.agent.task_aware_model.summary()}
            supports.append(unit)
            for arm in _arms(c):
                query = LabRuntime(cfg, output_root=out)
                query.agent.import_state(trained)
                query.agent.reset_episode()
                is_actor=arm=='actor'
                is_task_value=arm.startswith('task_value_h')
                horizon=(c['task_horizon'] if is_task_value else
                         (1 if is_actor else int(arm.split('_')[1])))
                algorithm='task_value_mpc' if is_task_value else 'experience_mpc'
                query.config = replace(cfg, experience_planning=not is_actor,
                                       experience_horizon=horizon,planner_algorithm=algorithm)
                query.agent.config = query.config
                query.world = ComplexWorld(seed=query_seed, curriculum=task)
                query.obs = query.world.observe()
                before = _learned_hash(query.agent)
                query_trace = []
                episode_outcome=None
                for step in range(c['query_steps']):
                    state = query.step(1, learn=False)
                    row = state['recent'][-1]
                    plan = state['experience']['planner']
                    planner_enabled=bool(plan.get('enabled',False))
                    planner_used=bool(plan.get('used',False))
                    item={'step':step,'reward':row['reward'],'done':row['done'],
                          'action':state['action'],'planner_enabled':planner_enabled,
                          'planner_used':planner_used,
                          'planner_abstained':bool(planner_enabled and not planner_used),
                          'planner_reason':plan.get('reason','actor policy'),
                          'model_family':plan.get('model_family','none'),
                          'model_evaluations':int(plan.get('evaluations',0)),
                          'member_evaluations':int(plan.get('member_evaluations',0))}
                    if 'fallback_reason' in plan:
                        item['fallback_reason']=plan['fallback_reason']
                    if 'outcome' in row:
                        item['outcome']=row['outcome']
                    query_trace.append(item)
                    if row['done']:
                        episode_outcome=row.get('outcome')
                        if episode_outcome not in ('success','fail','timeout'):
                            raise RuntimeError('terminal query transition lacks a valid environment outcome')
                        break
                after = _learned_hash(query.agent)
                if before != after:
                    raise RuntimeError('frozen planning query mutated learned state')
                if episode_outcome is None:
                    episode_outcome='incomplete'
                records.append({'seed': seed, 'task': task, 'arm': arm, 'query_seed': query_seed,
                                'support_digest': support_digest, 'steps': len(query_trace),
                                'episode_outcome':episode_outcome,
                                'return': float(sum(r['reward'] for r in query_trace)),
                                'planner_used_steps': sum(r['planner_used'] for r in query_trace),
                                'planner_abstained_steps':sum(r['planner_abstained'] for r in query_trace),
                                'model_evaluations': sum(r['model_evaluations'] for r in query_trace),
                                'member_evaluations':sum(r['member_evaluations'] for r in query_trace),
                                'learned_before': before, 'learned_after': after,
                                'trace': query_trace, 'trace_digest': _json_hash(query_trace)})
    units = len(supports)
    support_steps=units*c['support_steps']
    query_steps=sum(row['steps'] for row in records)
    costs={'support_world_steps':support_steps,'query_world_steps':query_steps,
           'support_update_calls':support_steps,'query_update_calls':0,
           'model_evaluations':sum(row['model_evaluations'] for row in records),
           'member_evaluations':sum(row['member_evaluations'] for row in records),
           'executed_transitions':support_steps+query_steps,
           'configured_transition_budget':len(c['seeds'])*len(c['tasks'])*
               (c['support_steps']+len(_arms(c))*c['query_steps']),
           'transition_limit':32768}
    summary=_summarize(records,c)
    summary['promotion']['transition_budget_respected']=costs['executed_transitions']<=32768
    summary['promotion']['eligible']=bool(summary['promotion']['eligible']
                                          and summary['promotion']['transition_budget_respected'])
    receipt = {'schema': 'synapse-planning-study-v2', 'status': 'complete', 'config': c,
               'scope': 'actual native and complex-world transitions; shared exploratory support trains both experience learners; one frozen paired query episode per seed/task/arm; heuristic short-horizon planner',
               'supports': supports, 'records': records, 'summary': summary,
               'costs': costs,
               'files': files, 'source_fingerprints': _source_fingerprints(),
               'environment': {**_environment(),
                               'torch_threads_requested':1,
                               'torch_threads_active':thread_configuration['torch_threads'],
                               'torch_thread_configuration_error':thread_configuration['thread_configuration_error']},
               'wall_seconds': time.perf_counter()-started}
    receipt['integrity'] = _json_hash(receipt)
    if out is not None:
        _atomic(out/'receipt.json', json.dumps(receipt, indent=2, allow_nan=False).encode())
        _atomic(out/'report.md', format_planning_report(receipt).encode())
    return receipt


def _verify_sources_and_files(receipt, root, errors, warnings):
    fingerprints=receipt.get('source_fingerprints')
    if not isinstance(fingerprints,dict) or not fingerprints:
        errors.append('missing source fingerprints')
    else:
        for name, sha in fingerprints.items():
            if (not isinstance(name,str) or not isinstance(sha,str) or len(sha)!=64
                    or any(ch not in '0123456789abcdef' for ch in sha)):
                errors.append('invalid source fingerprint')
                break
        current=_source_fingerprints()
        warnings.extend('current source differs: '+name for name,sha in fingerprints.items()
                        if current.get(name)!=sha)
    files=receipt.get('files',{})
    if not isinstance(files,dict):
        errors.append('invalid artifact file map')
        return
    if root is not None:
        for name,sha in files.items():
            if (not isinstance(name,str) or Path(name).name!=name or not isinstance(sha,str)
                    or len(sha)!=64 or _file_hash(root/name)!=sha):
                errors.append('artifact hash mismatch: '+str(name))


def _verify_v1(receipt, root, errors):
    c=validate_config(receipt['config'],version=1)
    expected={(seed,task) for seed in c['seeds'] for task in c['tasks']}
    supports={(unit['seed'],unit['task']):unit for unit in receipt['supports']}
    if set(supports)!=expected or len(supports)!=len(receipt['supports']):
        raise ValueError('support pairing mismatch')
    for unit in supports.values():
        if len(unit['trace'])!=c['support_steps'] or _json_hash(unit['trace'])!=unit['support_digest']:
            errors.append('support trace mismatch')
        if unit['support_seed']==unit['query_seed']:
            errors.append('support/query seed overlap')
    for row in receipt['records']:
        unit=supports[row['seed'],row['task']]
        trace=row['trace']
        if row['steps']!=c['query_steps'] or len(trace)!=row['steps']:
            errors.append('query step count mismatch')
        if row['query_seed']!=unit['query_seed'] or row['support_digest']!=unit['support_digest']:
            errors.append('query pairing mismatch')
        if row['learned_before']!=row['learned_after'] or row['learned_before']!=unit['learned_state_digest']:
            errors.append('query learned-state freeze mismatch')
        if _json_hash(trace)!=row['trace_digest']:
            errors.append('query trace digest mismatch')
        if not np.isfinite(row['return']) or abs(sum(item['reward'] for item in trace)-row['return'])>1e-10:
            errors.append('return arithmetic mismatch')
        if row['planner_used_steps']!=sum(item['planner_used'] for item in trace):
            errors.append('planner count mismatch')
        if row['model_evaluations']!=sum(item['model_evaluations'] for item in trace):
            errors.append('model work count mismatch')
        for step,item in enumerate(trace):
            action=np.asarray(item['action'],dtype=float)
            if (item['step']!=step or action.shape!=(5,) or not np.isfinite(action).all()
                    or np.any(np.abs(action)>1.000001) or not np.isfinite(item['reward'])
                    or type(item['done']) is not bool or type(item['planner_used']) is not bool
                    or type(item['model_evaluations']) is not int
                    or not 0<=item['model_evaluations']<=256):
                errors.append('invalid query transition')
    if _json_hash(_summarize_v1(receipt['records'],c))!=_json_hash(receipt['summary']):
        errors.append('summary recomputation mismatch')
    units=len(expected)
    expected_costs={'support_world_steps':units*c['support_steps'],
                    'query_world_steps':units*len(_arms_v1(c))*c['query_steps'],
                    'support_update_calls':units*c['support_steps'],'query_update_calls':0}
    if receipt['costs']!=expected_costs:
        errors.append('cost recomputation mismatch')


def _verify_v2(receipt, root, errors):
    c=validate_config(receipt['config'],version=2)
    units={(seed,task) for seed in c['seeds'] for task in c['tasks']}
    supports={(unit['seed'],unit['task']):unit for unit in receipt['supports']}
    if set(supports)!=units or len(supports)!=len(receipt['supports']):
        raise ValueError('support pairing mismatch')
    from .experience import NATIVE_TASKS
    for unit in supports.values():
        if len(unit['trace'])!=c['support_steps'] or _json_hash(unit['trace'])!=unit['support_digest']:
            errors.append('support trace mismatch')
        if unit['support_seed']==unit['query_seed']:
            errors.append('support/query seed overlap')
        if unit['legacy_model_summary'].get('samples')!=c['support_steps']:
            errors.append('legacy support model did not learn every support transition')
        expected_task_samples=c['support_steps'] if unit['task'] in NATIVE_TASKS else 0
        if unit['task_aware_model_summary'].get('samples')!=expected_task_samples:
            errors.append('task-aware support model sample count mismatch')
    index={(row['seed'],row['task'],row['arm']):row for row in receipt['records']}
    expected={(seed,task,arm) for seed,task in units for arm in _arms(c)}
    if set(index)!=expected or len(index)!=len(receipt['records']):
        raise ValueError('query arm pairing mismatch')
    query_steps=0
    for row in receipt['records']:
        unit=supports[row['seed'],row['task']]
        trace=row['trace']
        if (type(row['steps']) is not int or not 1<=row['steps']<=c['query_steps']
                or len(trace)!=row['steps']):
            errors.append('query step count mismatch')
        if row['query_seed']!=unit['query_seed'] or row['support_digest']!=unit['support_digest']:
            errors.append('query pairing mismatch')
        if (row['learned_before']!=row['learned_after']
                or row['learned_before']!=unit['learned_state_digest']):
            errors.append('query learned-state freeze mismatch')
        if _json_hash(trace)!=row['trace_digest']:
            errors.append('query trace digest mismatch')
        if not isinstance(row['learned_before'],str) or len(row['learned_before'])!=64:
            errors.append('invalid frozen-state digest')
        if (row['episode_outcome'] not in ('success','fail','timeout','incomplete')
                or not np.isfinite(row['return'])
                or abs(sum(item['reward'] for item in trace)-row['return'])>1e-10):
            errors.append('episode outcome or return mismatch')
        query_steps+=row['steps']
        if row['planner_used_steps']!=sum(item['planner_used'] for item in trace):
            errors.append('planner count mismatch')
        if row['planner_abstained_steps']!=sum(item['planner_abstained'] for item in trace):
            errors.append('planner abstention count mismatch')
        if row['model_evaluations']!=sum(item['model_evaluations'] for item in trace):
            errors.append('model work count mismatch')
        if row['member_evaluations']!=sum(item['member_evaluations'] for item in trace):
            errors.append('member model work count mismatch')
        for step,item in enumerate(trace):
            action=np.asarray(item['action'],dtype=float)
            terminal=bool(item.get('done'))
            if (item['step']!=step or action.shape!=(5,) or not np.isfinite(action).all()
                    or np.any(np.abs(action)>1.000001) or not np.isfinite(item['reward'])
                    or type(item['done']) is not bool or type(item['planner_enabled']) is not bool
                    or type(item['planner_used']) is not bool or type(item['planner_abstained']) is not bool
                    or type(item['model_evaluations']) is not int or not 0<=item['model_evaluations']<=256
                    or type(item['member_evaluations']) is not int
                    or item['planner_abstained']!=(item['planner_enabled'] and not item['planner_used'])
                    or (item['planner_used'] and not item['planner_enabled'])
                    or (item['member_evaluations']!=3*item['model_evaluations'])):
                errors.append('invalid query transition')
            if terminal!=(step==len(trace)-1 and row['episode_outcome']!='incomplete'):
                errors.append('episode terminal position mismatch')
            if terminal:
                if item.get('outcome')!=row['episode_outcome']:
                    errors.append('environment terminal outcome mismatch')
            elif 'outcome' in item:
                errors.append('nonterminal transition contains an outcome')
        if row['episode_outcome']=='incomplete' and row['steps']!=c['query_steps']:
            errors.append('incomplete episode did not reach query limit')
        if row['arm'].startswith('task_value_h'):
            if row['task'] in NATIVE_TASKS:
                if any(item.get('model_family')!='task_aware_40d' or item.get('fallback_reason')
                       for item in trace):
                    errors.append('native task-aware arm did not use its task-aware model')
            elif any(item.get('model_family')!='legacy_16d' or not item.get('fallback_reason')
                     for item in trace):
                errors.append('unsupported task-aware arm fallback mismatch')
    summary=_summarize(receipt['records'],c)
    costs={'support_world_steps':len(units)*c['support_steps'],
           'query_world_steps':query_steps,
           'support_update_calls':len(units)*c['support_steps'],
           'query_update_calls':0,
           'model_evaluations':sum(row['model_evaluations'] for row in receipt['records']),
           'member_evaluations':sum(row['member_evaluations'] for row in receipt['records']),
           'executed_transitions':len(units)*c['support_steps']+query_steps,
           'configured_transition_budget':len(units)*(c['support_steps']+len(_arms(c))*c['query_steps']),
           'transition_limit':32768}
    summary['promotion']['transition_budget_respected']=costs['executed_transitions']<=32768
    summary['promotion']['eligible']=bool(summary['promotion']['eligible']
                                          and summary['promotion']['transition_budget_respected'])
    if _json_hash(summary)!=_json_hash(receipt['summary']):
        errors.append('summary recomputation mismatch')
    if costs!=receipt['costs']:
        errors.append('cost recomputation mismatch')


def verify_planning_study(source):
    errors,warnings=[],[]
    root=None
    try:
        if isinstance(source,dict):
            receipt=deepcopy(source)
        else:
            path=Path(source)
            path=path/'receipt.json' if path.is_dir() else path
            root=path.parent
            receipt=json.loads(path.read_text(encoding='utf-8'))
        integrity=receipt.pop('integrity')
        if _json_hash(receipt)!=integrity:
            errors.append('receipt integrity mismatch')
        if receipt.get('status')!='complete':
            raise ValueError('unsupported receipt status')
        schema=receipt.get('schema')
        if schema=='synapse-planning-study-v1':
            _verify_v1(receipt,root,errors)
        elif schema=='synapse-planning-study-v2':
            _verify_v2(receipt,root,errors)
        else:
            raise ValueError('unsupported receipt schema')
        _verify_sources_and_files(receipt,root,errors,warnings)
    except (ValueError,TypeError,KeyError,OSError,OverflowError) as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
    return {'ok':not errors,'errors':errors,'warnings':warnings,
            'scope':'hash and arithmetic replay; does not re-execute worlds or authenticate consistently rewritten receipts'}


def format_planning_report(receipt):
    config=receipt['config']
    summary=receipt['summary']
    promo=summary['promotion']
    decision='PROMOTE' if promo['eligible'] else 'KEEP OPT-IN'
    lines=['# SYNAPSE task-aware delayed-reward planning study','',
           receipt['scope'],'',
           f"**Decision: {decision}.** The feature remains off by default unless the frozen completion and forage criteria pass.",'',
           '## Frozen protocol','',
           f"- Seeds: `{config['seeds']}`; tasks: `{config['tasks']}`.",
           f"- Shared support: {config['support_steps']} executed transitions per seed/task.",
           f"- Query: one episode per arm, capped at {config['query_steps']} transitions.",
           f"- Arms: `{_arms(config)}`; horizons: `{config['horizons']}`; task-aware horizon: {config['task_horizon']}.",
           f"- Promotion rule: {promo['rule']}.",
           f"- Promotion comparison: {promo['pooled_completion_improvement']:+.3f}; 95% descriptive interval `{promo['pooled_completion_interval']}`; forage difference {promo['forage_completion_difference']!r}.",'',
           '## Aggregate outcomes','',
           '| Arm | Completion | Mean return | Mean steps | Planner use | Abstentions | Model evaluations / step | Incomplete |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,row in summary['arms'].items():
        lines.append(f"| {arm} | {row['success_rate']:.3f} | {row['mean_return']:.4f} | {row['mean_steps']:.1f} | {row['planner_use_fraction']:.3f} | {row['planner_abstention_fraction']:.3f} | {row['mean_model_evaluations_per_step']:.2f} | {row['outcome_counts']['incomplete']} |")
    lines += ['', '## Task-level outcomes','']
    for task,arms in summary['task_summaries'].items():
        lines += [f'### {task}', '',
                  '| Arm | Completion | Mean return | Mean steps | Success / fail / timeout / incomplete |',
                  '|---|---:|---:|---:|---|']
        for arm,row in arms.items():
            counts=row['outcome_counts']
            lines.append(f"| {arm} | {row['success_rate']:.3f} | {row['mean_return']:.4f} | {row['mean_steps']:.1f} | {counts['success']} / {counts['fail']} / {counts['timeout']} / {counts['incomplete']} |")
        lines.append('')
    lines += ['## Paired comparisons with actor','',
              '| Arm | Completion difference | Completion 95% interval | Return difference | Return 95% interval |',
              '|---|---:|---|---:|---|']
    for arm,comparison in summary['comparisons'].items():
        completion,ret=comparison['completion'],comparison['return']
        lines.append(f"| {arm.removesuffix('_minus_actor')} | {completion['mean_difference']:+.3f} | {completion['descriptive_bootstrap_95']} | {ret['mean_difference']:+.4f} | {ret['descriptive_bootstrap_95']} |")
    lines += ['', '## Transition and model costs','',
              '| Cost | Value |','|---|---:|']
    for name,value in receipt['costs'].items():
        lines.append(f'| {name} | {value} |')
    lines += ['', '## Source fingerprints','',
              'The receipt contains the full per-file fingerprint map. This report includes every recorded fingerprint for direct inspection.','',
              '| Source | SHA-256 |','|---|---|']
    for name,sha in sorted(receipt['source_fingerprints'].items()):
        lines.append(f'| `{name}` | `{sha}` |')
    lines += ['', '## Evidence limits','',
              summary['interpretation']+'.',
              'The unit is one paired seed/task episode. Bootstrap intervals are descriptive with a small local sample. Support is shared and fitted; each query begins from the same saved support state and query fitting is disabled. Task-aware planning is valid only for exact native task identities, uses value-guided short MPC with fixed task/graph/donor context, and may abstain. Receipts verify recorded hashes and arithmetic but do not re-execute worlds, authenticate source history, or support real-world reliability claims.','']
    return '\n'.join(lines)
