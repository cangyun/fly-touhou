"""NoTorch, read-only within-rollout visual-value / delayed-Miss diagnostic."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.thrl_dodge_diagnostics import _shared_binary
if sys.modules.get('torch') is not None:
    raise RuntimeError('Read-only diagnostic must not import Torch')


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def rank_u(values, labels):
    """P(lower V for a positive than a negative), half credit for ties."""
    scores = -np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    order = np.argsort(scores, kind='stable')
    ranks = np.empty(len(scores), dtype=np.float64)
    left = 0
    while left < len(order):
        right = left + 1
        while right < len(order) and scores[order[right]] == scores[order[left]]:
            right += 1
        ranks[order[left:right]] = (left + 1 + right) / 2
        left = right
    positive = int(labels.sum())
    negative = len(labels) - positive
    return float(ranks[labels].sum() - positive * (positive + 1) / 2), positive * negative


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['rollout']].append(row)
    per_rollout, numerator, pairs, residual_ss, target_ss = [], 0., 0, 0., 0.
    low_y, high_y = [], []
    for rollout, part in sorted(groups.items()):
        v = np.asarray([r['V'] for r in part])
        y = np.asarray([r['Miss'] for r in part], dtype=bool)
        target = np.asarray([r['return16'] for r in part])
        u, n = rank_u(v, y)
        numerator += u
        pairs += n
        residual_ss += len(part) * float(np.var(target - v))
        target_ss += len(part) * float(np.var(target))
        low_y.extend(y[v <= np.quantile(v, .25)].tolist())
        high_y.extend(y[v >= np.quantile(v, .75)].tolist())
        per_rollout.append(dict(rollout=rollout, rows=len(part), positives=int(y.sum()),
            low_V_AUC=u / n if n else None, positive_negative_pairs=n))
    positive = [r['V'] for r in rows if r['Miss']]
    negative = [r['V'] for r in rows if not r['Miss']]
    events = {event for r in rows if r['Miss'] for event in r['events']}
    return dict(rows=len(rows), positive_windows=len(positive), negative_windows=len(negative),
        distinct_positive_Miss_transition_rows=len(events), games=len({r['game'] for r in rows}),
        low_V_within_rollout_pair_weighted_AUC=numerator / pairs if pairs else None,
        positive_negative_pairs=pairs, rollouts_with_both_classes=sum(p['positive_negative_pairs'] > 0 for p in per_rollout),
        positive_V_mean=float(np.mean(positive)) if positive else None,
        negative_V_mean=float(np.mean(negative)) if negative else None,
        low_V_quartile=dict(rows=len(low_y), risk=float(np.mean(low_y)) if low_y else None),
        high_V_quartile=dict(rows=len(high_y), risk=float(np.mean(high_y)) if high_y else None),
        return16_RMSE=float(np.sqrt(np.mean([(r['return16'] - r['V']) ** 2 for r in rows]))) if rows else None,
        return16_bias=float(np.mean([r['return16'] - r['V'] for r in rows])) if rows else None,
        return16_within_rollout_EV=1 - residual_ss / target_ss if target_ss > 0 else None,
        per_rollout=per_rollout)


def analyze(run):
    started, begun, refs, counts = utc(), time.monotonic(), {}, Counter()
    def raw(path):
        if time.monotonic() - begun > 120:
            raise TimeoutError('Read-only diagnostic exceeded120s')
        with _shared_binary(path) as stream:
            data = stream.read()
        refs[str(path.resolve())] = dict(sha256=sha(data), bytes=len(data), read_utc=utc())
        return data
    def read(path):
        return json.loads(raw(path).decode('utf-8-sig'))
    canonical_path = run / ('result.json' if (run / 'result.json').exists() else 'progress.json')
    canonical = read(canonical_path)
    limit = int(canonical['costs']['completed_rollouts'])
    windows, near, episodes = [], [], {}
    for number in range(1, limit + 1):
        update = read(run / 'traces' / f'update-{number:03}.json')
        trace = run / 'traces' / f'rollout-{number-1:03}.json'
        meta = read(trace)
        data = raw(trace.with_suffix('.npz'))
        if sha(data) != meta['trace_sha256'] or update['costs']['completed_rollouts'] != number:
            raise ValueError('Committed ordinal or NPZ identity mismatch')
        version = meta['metadata']['policy_version']
        if version != number - 1 or update['rollout_metadata']['policy_version'] != version:
            raise ValueError('Behavior policy version mismatch')
        with np.load(io.BytesIO(data), allow_pickle=False) as a:
            v, f, reward = [a[k].astype(np.float64) for k in ('values', 'features', 'rewards')]
            game, done, truncated, starts, bootstrap = [a[k] for k in ('game_ids', 'terminated', 'truncated', 'episode_starts', 'bootstrap_values')]
        if v.shape != (32, 30) or f.shape != (32, 30, 25) or not np.isfinite(v).all():
            raise ValueError('Expected actual30-world32-step values/raw25')
        info, components = meta['step_info'], meta['reward_components']
        miss = np.asarray([[-c['death'] / 2 for c in step] for step in components])
        bomb = np.asarray([[c['bomb_used'] < 0 for c in step] for step in components])
        protected = np.asarray([[e['protected_decision'] for e in step] for step in info])
        ticks = np.asarray([[e['elapsed_input_ticks'] for e in step] for step in info])
        if not (np.isfinite(miss).all() and (miss >= 0).all() and (miss == np.rint(miss)).all()
                and np.isin(protected, [0, 1]).all() and (protected == (f[:, :, 5] > 0)).all()
                and np.isin(ticks, [1, 2, 3, 4]).all() and not truncated.any()):
            raise ValueError('Unknown events/protection/timing cannot be labeled negative')
        counts['source_rows'] += v.size
        counts['source_Miss_transition_rows'] += int((miss > 0).sum())
        counts['source_Bomb_transition_rows'] += int(bomb.sum())
        counts['short_native_terminal_or_transition_rows'] += int((ticks != 4).sum())
        counts['right_censored_by_fixed_policy_rollout'] += 15 * 30
        for item in meta['completed_episodes']:
            episodes[(item['worker'], item['game_id'])] = item['summary']['outcome']
        for t in range(17):
            for worker in range(30):
                counts['full16_candidate_starts'] += 1
                sl = slice(t, t + 16)
                if (not np.all(game[sl, worker] == game[t, worker]) or starts[t+1:t+16, worker].any()
                        or done[t:t+15, worker].any()):
                    counts['reset_or_terminal_before_full16_excluded'] += 1
                    continue
                if not np.all(ticks[sl, worker] == 4):
                    counts['shorter_than64_native_frame_window_excluded'] += 1
                    continue
                # GLOBAL_FIELDS invincible5; V5 TIMER_FIELDS normal state1=21.
                if protected[t, worker] != 0 or f[t, worker, 20] > 0:
                    counts['protected_start_excluded'] += 1
                    continue
                if f[t, worker, 21] != 1 or f[t, worker, 19] != 0:
                    counts['non_normal_or_deathbomb_start_excluded'] += 1
                    continue
                terminal = bool(done[t+15, worker])
                if not terminal and t + 16 < 32 and game[t+16, worker] != game[t, worker]:
                    raise ValueError('Nonterminal endpoint unexpectedly changed game')
                endpoint = 0. if terminal else (v[t+16, worker] if t+16 < 32 else float(bootstrap[worker]))
                target = float(np.dot(.999 ** np.arange(16), reward[sl, worker]) + .999 ** 16 * endpoint)
                base = dict(rollout=number, version=version, game=int(game[t, worker]), V=float(v[t, worker]),
                    window_Bomb=bool(bomb[sl, worker].any()), return16=target)
                def events(lo, hi):
                    return [f'{number}:{worker}:{k}' for k in range(t+lo, t+hi) if miss[k, worker] > 0]
                early, delayed = events(0, 8), events(8, 16)
                near.append(dict(base, Miss=bool(early), events=early))
                if early:
                    counts['earlier0to7_Miss_excluded_from_delayed'] += 1
                    continue
                windows.append(dict(base, Miss=bool(delayed), events=delayed))
    summary = {name: summarize(part) for name, part in (
        ('delayed8to15_all', windows), ('delayed8to15_no_window_Bomb', [r for r in windows if not r['window_Bomb']]),
        ('delayed8to15_with_window_Bomb', [r for r in windows if r['window_Bomb']]),
        ('near0to7_all', near), ('near0to7_no_window_Bomb', [r for r in near if not r['window_Bomb']]))}
    sources = [Path(__file__), ROOT / 'src/fly_touhou/th14_completion_v3.py',
        ROOT / 'src/fly_touhou/thrl_geometry_observation.py', ROOT / 'src/fly_touhou/thrl_timer_observation.py',
        ROOT / 'src/fly_touhou/thrl_laser_observation.py', ROOT / 'src/fly_touhou/fly_visual_online_pool.py',
        ROOT / 'src/fly_touhou/fly_visual_actor_critic.py']
    source_hashes = {str(p.resolve()): sha(raw(p)) for p in sources}
    return dict(schema='fly-visual-critic-delayed-Miss-readonly-v1', run=str(run.resolve()),
        started_utc=started, finished_utc=utc(), elapsed_seconds=time.monotonic()-begun,
        canonical=str(canonical_path), canonical_status=canonical.get('status', canonical.get('phase')),
        completed_rollouts=limit, source_costs=canonical['costs'], source_training_outcomes=dict(Counter(episodes.values())),
        counts=dict(counts), metrics=summary, source_sha256=source_hashes, read_files=refs,
        definition=dict(H=16, native_frames=64, gamma=.999, predictor='recorded pre-action visual V, score=-V',
            start='invincible5=0, protection_remaining20=0, state1_21=1, deathbomb19=0; checked against pre-action protected_decision',
            event='actual native Miss transitions from reward_components.death/-2',
            delayed='any Miss at action offsets8..15; no Miss at0..7',
            Bomb='any actual resource consumption in all16 transitions, reward_components.bomb_used<0',
            AUC='within-rollout positive/negative pairs only; pair-count weighted, ties half; never cross policy_version',
            return16='discounted16 real rewards + same behavior critic endpoint (zero at natural terminal)',
            censoring='full16 must fit within rollout and one game; no unknown future is labeled negative'),
        caveats=['Overlapping windows and repeated events are dependent; no iid significance/CI claim.',
            'Conditioning on no early Miss, full horizon and no Bomb selects trajectories; not causal intervention.',
            'V predicts shaped policy return, not calibrated collision probability or safe alternative-action value.',
            'Bootstrapped return reuses the critic itself; EV is not independent Monte Carlo validation or a root-cause test.',
            'All observed changing-policy training episodes, not frozen native development performance.'],
        Torch_imported=sys.modules.get('torch') is not None, GPU=False, model_forwards=0, native_worlds=0, optimizer_updates=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError('Write diagnostic outside the source run')
    if os.name == 'nt':
        import ctypes
        k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
        if not k.SetPriorityClass(k.GetCurrentProcess(), 0x4000):
            raise ctypes.WinError(ctypes.get_last_error())
    result = analyze(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), elapsed_seconds=result['elapsed_seconds'], counts=result['counts'],
        metrics={k:{n:v for n,v in s.items() if n!='per_rollout'} for k,s in result['metrics'].items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
