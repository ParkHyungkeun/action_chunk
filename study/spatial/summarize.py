
import json
from pathlib import Path

METHODS = ('fixed_5', 'fixed_8', 'fixed_10', 'fixed_16', 'aac', 'hmm',
           'threshold', 'emission_only', 'random_matched')
REFERENCE = 'fixed_16'

def summarize(root, group=None):
    root = Path(root)
    if group is None:
        group = sorted(root.glob('test300_*'))[-1]
    group = Path(group)
    data, manifests = {}, {}
    for method in METHODS:
        method_root = group / method
        runs = sorted(method_root.glob('task[0-9][0-9]')) if method_root.is_dir() else []
        if not runs:
            runs = [method_root]
        records, method_manifests, complete_count = [], [], 0
        for run in runs:
            complete = json.loads((run/'COMPLETE.json').read_text())
            manifest = json.loads((run/'manifest.json').read_text())
            shard_records = [json.loads(x) for x in (run/'episodes.jsonl').read_text().splitlines() if x.strip()]
            if complete['episodes'] != len(shard_records):
                raise RuntimeError('Incomplete shard: '+str(run))
            complete_count += complete['episodes']
            records.extend(shard_records)
            method_manifests.append(manifest)
        manifest = method_manifests[0]
        if any(x != manifest for x in method_manifests[1:]):
            raise RuntimeError('Manifest mismatch between task shards: '+method)
        stage = manifest['stage']; cfg = manifest['config']
        n = len(cfg['task_ids'])*len(cfg['initial_states'][stage])*(cfg['test_repeats'] if stage == 'test' else 1)
        if len(records) != n or complete_count != n or any(x['status'] != 'completed' for x in records):
            raise RuntimeError('Incomplete evaluation: '+method)
        if len({x['case'] for x in records}) != n:
            raise RuntimeError('Duplicate cases')
        data[method] = {x['case']: x for x in records}
        manifests[method] = manifest
    for method in METHODS:
        if method == REFERENCE:
            continue
        for key in ('config', 'code', 'checkpoint', 'environment'):
            if manifests[method][key] != manifests[REFERENCE][key]:
                raise RuntimeError('Mismatched '+key)
        if data[method].keys() != data[REFERENCE].keys():
            raise RuntimeError('Mismatched cases')
        for case, ref in data[REFERENCE].items():
            for key in ('seed', 'initial_state_id', 'initial_state_sha256', 'initial_observation_sha256'):
                if ref[key] != data[method][case][key]:
                    raise RuntimeError('Pairing mismatch: '+case+'/'+key)
    summary = {}
    lines = ['# Paired evaluation results', '', '| Method | Success | Rate | Planning seconds/episode | Replans/episode |', '|---|---:|---:|---:|---:|']
    for method, mapping in data.items():
        rows = list(mapping.values()); n = len(rows); wins = sum(r['success'] for r in rows)
        per_task = {}; per_repeat = {}
        for key, target in [('task_id', per_task), ('repeat_id', per_repeat)]:
            for value in sorted({r.get(key, 0) for r in rows}):
                sub = [r for r in rows if r.get(key, 0) == value]
                target[value] = {'n': len(sub), 'successes': sum(r['success'] for r in sub), 'rate_pct': 100*sum(r['success'] for r in sub)/len(sub)}
        summary[method] = {'n': n, 'successes': wins, 'rate_pct': 100*wins/n,
            'planning_s_per_episode': sum(r['planning_s'] for r in rows)/n,
            'policy_compute_s_per_episode': sum(r['planning_s']+r['filter_s'] for r in rows)/n,
            'wall_s_per_episode': sum(r['rollout_wall_s'] for r in rows)/n,
            'replans_per_episode': sum(r['replans'] for r in rows)/n,
            'executed_actions_per_plan': sum(r['steps'] for r in rows)/sum(r['replans'] for r in rows),
            'per_task': per_task, 'per_repeat': per_repeat}
        s = summary[method]
        lines.append(f"| {method} | {wins}/{n} | {s['rate_pct']:.1f}% | {s['planning_s_per_episode']:.3f} | {s['replans_per_episode']:.2f} |")
    for label, key in [('Per task', 'per_task'), ('Per random-seed repetition', 'per_repeat')]:
        lines += ['', '## '+label, '', '| ID | '+' | '.join(METHODS)+' |', '|---|'+'|'.join('---:' for _ in METHODS)+'|']
        for value in summary[REFERENCE][key]:
            cells = [f"{summary[m][key][value]['successes']}/{summary[m][key][value]['n']}" for m in METHODS]
            lines.append('| '+str(value)+' | '+' | '.join(cells)+' |')
    lines += ['', 'Repeated seeds share initial environments; 300 trials are not 300 independent environments.',
        'Descriptive results only; no independence-based significance claim.',
        'AAC uses sequential candidate heads, not official parallel runtime.',
        'Planning time excludes parity checks; wall time includes simulator overhead.']
    (group/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (group/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
