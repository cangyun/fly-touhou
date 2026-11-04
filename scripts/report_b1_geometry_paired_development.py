"""Seal and compare both complete frozen native cohorts; never score partial runs."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import html
import json
import math
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now
from fly_touhou.thrl_dodge_diagnostics import _shared_binary
from fly_touhou.thrl_durable_records import write_json as write


def wilson(successes, count):
    if count <= 0:
        raise ValueError('No observed outcomes')
    z = 1.959963984540054
    rate = successes / count
    center = (rate + z*z/(2*count)) / (1 + z*z/count)
    radius = z * math.sqrt(rate*(1-rate)/count + z*z/(4*count*count)) / (1 + z*z/count)
    return [center-radius, center+radius]


def cohort_statistics(baseline, candidate):
    import numpy as np
    lookup = []
    for rows in (baseline, candidate):
        if (len(rows) != 20 or len({row['native_seed'] for row in rows}) != 20
                or {row['evaluation_game'] for row in rows} != set(range(1, 21))
                or any(row['outcome'] not in ('clear', 'gameover') for row in rows)):
            raise ValueError('Require both complete, unique, naturally terminated20-game cohorts')
        lookup.append({row['evaluation_game']: row for row in rows})
    pairs = []
    counts = Counter()
    for game in range(1, 21):
        a, b = [cohort[game] for cohort in lookup]
        if a['native_seed'] != b['native_seed']:
            raise ValueError('Initial native seeds do not match by game')
        cleared_a, cleared_b = a['outcome'] == 'clear', b['outcome'] == 'clear'
        category = ('both_clear' if cleared_a and cleared_b else
                    'candidate_only_clear' if cleared_b else
                    'baseline_only_clear' if cleared_a else 'both_gameover')
        counts[category] += 1
        pairs.append(dict(game=game, native_seed=a['native_seed'], category=category,
            clear_gain=int(cleared_b)-int(cleared_a),
            baseline=dict(outcome=a['outcome'], stage=max(a['stage_history']),
                native_frames=a['native_frames'], Miss=a['misses'], Bomb=a['bombs_used']),
            candidate=dict(outcome=b['outcome'], stage=max(b['stage_history']),
                native_frames=b['native_frames'], Miss=b['misses'], Bomb=b['bombs_used']),
            native_frame_gain=b['native_frames']-a['native_frames']))
    x, y = counts['candidate_only_clear'], counts['baseline_only_clear']
    discordant = x+y
    p = min(1., 2*sum(math.comb(discordant, k) for k in range(min(x, y)+1))/2**discordant)
    rng = np.random.default_rng(383021)
    gains = np.array([row['clear_gain'] for row in pairs], dtype=np.float64)
    frames = np.array([row['native_frame_gain'] for row in pairs], dtype=np.float64)
    indices = rng.integers(0, 20, size=(50000, 20))
    return dict(pairs=pairs, paired_outcomes={key: counts[key] for key in (
        'both_clear', 'candidate_only_clear', 'baseline_only_clear', 'both_gameover')},
        candidate_minus_baseline_clear_rate=float(gains.mean()),
        descriptive_paired_clear_gain_percentile95=np.quantile(gains[indices].mean(1), [.025, .975]).tolist(),
        candidate_minus_baseline_mean_native_frames=float(frames.mean()),
        descriptive_paired_native_frame_gain_percentile95=np.quantile(frames[indices].mean(1), [.025, .975]).tolist(),
        McNemar_exact_two_sided_p=p, bootstrap_replicates=50000, bootstrap_seed=383021,
        independent_formal_test=False,
        future_native_hazards_identical=False,
        scope='Matched initial native RNG; development data, complete cohorts, descriptive uncertainty only.')


def aggregate(rows):
    def mean(getter):
        values = [getter(row) for row in rows]
        values = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
        return dict(value=statistics.mean(values) if values else None, observed_games=len(values))
    def diagnostic(row, key):
        return row.get('reward_diagnostics', {}).get(key)
    clears = sum(row['outcome'] == 'clear' for row in rows)
    usable = sum(diagnostic(row, 'deathbomb_usable_decision_count') or 0 for row in rows)
    requests = sum(diagnostic(row, 'deathbomb_usable_bomb_requests') or 0 for row in rows)
    result = dict(episodes=len(rows), clears=clears, clear_rate=clears/len(rows),
        clear_Wilson95=wilson(clears, len(rows)),
        failed_stage_counts=dict(Counter(str(max(row['stage_history'])) for row in rows if row['outcome']=='gameover')),
        GameOver_with_unused_Bomb=sum(row['outcome']=='gameover' and row['bombs_remaining']>0 for row in rows),
        mean_native_frames=mean(lambda row: row['native_frames']),
        mean_max_stage=mean(lambda row: max(row['stage_history'])),
        mean_Miss=mean(lambda row: row['misses']), mean_Bomb=mean(lambda row: row['bombs_used']),
        mean_point_items=mean(lambda row: row['point_items']),
        mean_net_life_increases=mean(lambda row: row['endpoints']['end']['lives']
            -row['endpoints']['start']['lives']+row['misses']),
        mean_first_collision_observed_frames=mean(lambda row: row.get('movement_diagnostics', {}).get('first_collision_observed_frames')),
        mean_observed_collision_events=mean(lambda row: row.get('movement_diagnostics', {}).get('observed_collision_events')),
        mean_undiscounted_shaped_reward=mean(lambda row: sum(row['reward_vector'])),
        usable_deathbomb_preaction_decisions=usable,
        usable_deathbomb_preaction_Bomb_requests=requests,
        usable_deathbomb_decision_request_fraction=requests/usable if usable else None,
        decision_count_is_not_unique_window_count=True,
        collision_and_deathbomb_event_coverage_complete=False)
    for key in ('confirmed_deathbombs', 'ordinary_or_unconfirmed_bombs', 'boss_hp_fraction'):
        result['mean_'+key] = mean(lambda row: diagnostic(row, key))
    return result


def generate(source, output):
    source, output = Path(source), Path(output)
    root = read(source/'report.json')
    if (root.get('schema') != 'B1-point-risk-calibration-paired-development40-v1'
            or root.get('status') != 'completed' or root.get('native_games') != 40
            or root.get('new_optimizer_updates') != 0 or root.get('formal_test_seeds_used') is not False
            or root.get('reward_input_actor_GRU_critic_and_laser_unchanged') is not True
            or root.get('initial_native_seeds_matched') is not True):
        raise ValueError('Only the complete frozen40-game geometry comparison may be sealed')
    plan = read(source/'plan.json')
    if (plan.get('native_frame_limit') is not None or plan.get('continues') != 0
            or plan.get('teacher_actions') != 0 or plan.get('action_selection') != 'greedy'
            or len(root['changed_state_tensors']) != 8
            or any(not key.startswith('risk.network.') for key in root['changed_state_tensors'])):
        raise ValueError('Natural terminal, autonomous19-action policies and isolated point-risk change required')
    reports = root['policies']
    for role in ('baseline', 'candidate'):
        report = reports[role]
        if (report['episodes'] != 20 or report['policy_frozen'] is not True
                or report['PPO_updates'] != 0 or report['action_selection'] != 'greedy'
                or report.get('teacher_actions') != 0 or report.get('safety_takeovers') != 0
                or report.get('continues') != 0
                or report['native_seed_split'] != 'development'
                or report['checkpoint_sha256'] != plan['policies'][role]['sha256']
                or [row['native_seed'] for row in report['episode_results']] != plan['expected_native_seeds']
                or report != read(source/role/'report.json')
                or read(source/role/'games/native_cleanup.json').get('close_errors') != []):
            raise ValueError('Role identity, protocol, complete outcomes or cleanup differs')
    cleanup = read(source/'coordinator_cleanup.json')
    if cleanup.get('owned_processes_only') is not True:
        raise ValueError('Wait for coordinator cleanup')
    statistics_result = cohort_statistics(reports['baseline']['episode_results'], reports['candidate']['episode_results'])
    aggregates = {role: aggregate(reports[role]['episode_results']) for role in reports}
    pins = output/'sealed_inputs'
    if output.exists():
        raise ValueError('Use a fresh review output; do not overwrite prior evidence')
    pins.mkdir(parents=True)
    sources = []
    for relative in ['plan.json', 'report.json', 'launch.json', 'dispatched_children.json', 'coordinator_cleanup.json',
                     'baseline/report.json', 'candidate/report.json',
                     'baseline/games/native_cleanup.json', 'candidate/games/native_cleanup.json']:
        with _shared_binary(source/relative) as stream:
            raw = stream.read()
        target = pins/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            import os
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        sources.append(dict(source=str(source/relative), sealed_copy=str(target), sha256=hashlib.sha256(raw).hexdigest()))
    review = dict(schema='B1-complete-geometry-paired-development-review-v1', utc=utc_now(),
        status='completed', aggregate=aggregates, **statistics_result, sources=sources,
        checkpoints=plan['policies'], changed_state_tensors=root['changed_state_tensors'],
        new_optimizer_updates=0, new_original_worlds=0, reward_modified=False,
        notes=['First collision is endpoint-observed; missed between-snapshot events remain unknown.',
            'Lower Miss/Bomb may reflect shorter games or fewer acquired lives; interpret with clear and resource outcomes.',
            'Net life increases are inventory balance plus Miss, not individual item pickup events.',
            'Cumulative shaped reward is undiscounted; it is not PPO advantage or proof of reward hacking.',
            'Neither20 development games nor geometric proxy scores establish stable independent clear performance.'])
    write(output/'review.json', review)
    rows = [('完整自然通关', 'clears'), ('平均原作帧', 'mean_native_frames'), ('平均推进面数', 'mean_max_stage'),
            ('平均Miss', 'mean_Miss'), ('平均Bomb', 'mean_Bomb'), ('平均净增残机', 'mean_net_life_increases'),
            ('平均点物品', 'mean_point_items'), ('平均观测碰撞次数', 'mean_observed_collision_events'),
            ('首次观测碰撞平均帧', 'mean_first_collision_observed_frames'),
            ('平均确认决死Bomb', 'mean_confirmed_deathbombs'), ('平均普通/未确认Bomb', 'mean_ordinary_or_unconfirmed_bombs'),
            ('Game Over时仍有Bomb', 'GameOver_with_unused_Bomb')]
    table = ''
    for title, key in rows:
        values = [aggregates[role][key] for role in ('baseline', 'candidate')]
        values = [f'{v["value"]:,.2f}（{v["observed_games"]}局）' if isinstance(v, dict) and v['value'] is not None
                  else str(v) for v in values]
        table += '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (title, *values))+'</tr>'
    pair_table = ''
    for row in review['pairs']:
        values = [row['game'], row['native_seed']]
        for role in ('baseline', 'candidate'):
            v = row[role]
            values.append(f'{v["outcome"]} / 第{v["stage"]}面 / {v["native_frames"]:,}帧 / Miss{v["Miss"]} Bomb{v["Bomb"]}')
        pair_table += '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>'
    doc = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>B1尺寸校准完整对照</title>'
        '<style>body{font:16px/1.7 system-ui,"Microsoft YaHei";max-width:1200px;margin:32px auto;padding:24px;color:#172534}'
        'table{width:100%;border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf3fa}</style>'
        '<h1>B1 尺寸校准：完整40局开发对照</h1><p>两组各20局，原作Lunatic六面自然开局至Game Over/通关；'
        '相同初始原作种子、最高概率动作、允许自主Bomb、禁止续关。只有8个点风险张量不同。'
        '</p><p>属于开发比较；后续弹幕会随动作分岔。当前报告不构成稳定通关的独立验收。</p>'
        '<table><tr><th>指标</th><th>原风险分支</th><th>全尺寸校准分支</th></tr>'+table+'</table>'
        '<h2>配对结果与不确定性</h2><pre>'+html.escape(json.dumps({key:review[key] for key in (
            'paired_outcomes', 'candidate_minus_baseline_clear_rate', 'descriptive_paired_clear_gain_percentile95',
            'McNemar_exact_two_sided_p')}, ensure_ascii=False, indent=2))+'</pre>'
        '<p>Miss、Bomb总数受游戏长度和获得残机影响；首次碰撞与决死窗口仅为已观测记录。累计奖励未折扣，单列作诊断。</p>'
        '<h2>固定20个初始种子逐局比较</h2><table><tr><th>局</th><th>种子</th><th>原风险分支</th><th>校准分支</th></tr>'
        +pair_table+'</table></html>')
    (output/'review.html').write_text(doc, encoding='utf-8')
    return dict(output=str(output), aggregate=aggregates, paired_outcomes=review['paired_outcomes'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    print(generate(arguments.source, arguments.output))
