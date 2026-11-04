"""Separate observed collisions, lost lives and deathbomb saves in sealed games.

This consumes complete frozen original-game reports only. It does not infer
unseen collisions, replay rewards, update a model or start original worlds.
"""
from pathlib import Path
import argparse
import hashlib
import json
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now


def summarize(rows):
    if not rows:
        return None
    required_movement=('observed_collision_events','observed_clean_four_tick_movements',
        'observed_collision_action_targets','vulnerable_movement_decisions')
    required_reward=('confirmed_deathbombs','ordinary_or_unconfirmed_bombs',
        'native_misses','vulnerable_frames','deathbomb_observed_window_count')
    for row in rows:
        movement=row['movement_diagnostics'];reward=row['reward_diagnostics']
        for key in required_movement:
            if type(movement.get(key)) is not int or movement[key]<0:
                raise ValueError('Missing measured movement count: '+key)
        for key in required_reward:
            if type(reward.get(key)) is not int or reward[key]<0:
                raise ValueError('Missing measured resource count: '+key)
        if (reward['native_misses']!=row['misses']
                or reward['confirmed_deathbombs']+reward['ordinary_or_unconfirmed_bombs']!=row['bombs_used']):
            raise ValueError('Episode and native cumulative resource counts disagree')
    counts={key:sum(r['movement_diagnostics'][key] for r in rows) for key in required_movement}
    counts.update({key:sum(r['reward_diagnostics'][key] for r in rows) for key in required_reward})
    matched=counts['observed_clean_four_tick_movements']+counts['observed_collision_action_targets']
    first=[r['movement_diagnostics']['first_collision_observed_frames'] for r in rows
           if r['movement_diagnostics'].get('first_collision_observed_frames') is not None]
    components={}
    for row in rows:
        for key,value in row['reward_diagnostics']['reward_component_sums'].items():
            components[key]=components.get(key,0.)+float(value)
    return dict(episodes=len(rows),clears=sum(r['outcome']=='clear' for r in rows),totals=counts,
        mean_misses=statistics.mean(r['misses'] for r in rows),
        mean_observed_collision_events=counts['observed_collision_events']/len(rows),
        mean_confirmed_deathbombs=counts['confirmed_deathbombs']/len(rows),
        mean_Bombs=statistics.mean(r['bombs_used'] for r in rows),
        collision_action_targets_per_1000_matched_four_tick_movements=(
            1000*counts['observed_collision_action_targets']/matched if matched else None),
        matched_four_tick_movements=matched,
        first_collision_observed_games=len(first),
        mean_first_collision_observed_frames=statistics.mean(first) if first else None,
        observed_windows_minus_confirmed_saves_and_misses=(counts['deathbomb_observed_window_count']
            -counts['confirmed_deathbombs']-counts['native_misses']),
        reward_component_sums_undiscounted=components,
        directly_charged_survival_collision_component_present=any(
            'collision' in key or 'contact' in key for key in components))


def audit_report(path):
    report=read(path);rows=report['episode_results']
    if (report.get('episodes')!=20 or len(rows)!=20
            or report.get('policy_frozen') is not True or report.get('PPO_updates')!=0
            or report.get('teacher_actions')!=0 or report.get('safety_takeovers')!=0
            or report.get('continues')!=0 or report.get('training_mode')!='full-story-v1'
            or any(r.get('outcome') not in ('clear','gameover') or r.get('full_story_episode') is not True for r in rows)):
        raise ValueError('Twenty complete autonomous frozen original games required')
    seeds=[r['native_seed'] for r in rows]
    if len(set(seeds))!=20 or any(s%10 not in (8,9) for s in seeds):
        raise ValueError('Twenty unique actual original test seeds required')
    inference=report.get('action_selection')
    if inference is None and report.get('selection_action')=='episode-local PCG64 categorical sampling, temperature1':
        inference='sampled'
    if inference not in ('sampled','greedy'):
        raise ValueError('Measured inference protocol must be explicit')
    details=[]
    for row in rows:
        movement=row['movement_diagnostics'];reward=row['reward_diagnostics']
        details.append(dict(game=row['evaluation_game'],native_seed=row['native_seed'],outcome=row['outcome'],
            stage=row['stage'],native_frames=row['native_frames'],misses=row['misses'],Bombs=row['bombs_used'],
            observed_collision_events=movement['observed_collision_events'],
            confirmed_deathbombs=reward['confirmed_deathbombs'],
            ordinary_or_unconfirmed_Bombs=reward['ordinary_or_unconfirmed_bombs'],
            first_collision_observed_frames=movement.get('first_collision_observed_frames'),
            Bombs_before_first_observed_collision=movement.get('Bombs_used_at_first_collision_observed')))
    return dict(report=str(path.resolve()),report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        inference=inference,inference_description=report.get('selection_action'),
        model_checkpoint_sha256=report['full_game_trace_capture']['model_checkpoint_sha256'],
        actual_original_initial_seeds=seeds,all_games=summarize(rows),
        clear_games=summarize([r for r in rows if r['outcome']=='clear']),games=details)


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    reports={}
    for item in args.report:
        name,path=item.split('=',1)
        if name in reports:raise ValueError('Condition label repeated')
        reports[name]=audit_report(Path(path))
    args.output.mkdir(parents=True)
    result=dict(schema='B1-sealed-full-game-collision-resource-burden-v1',utc=utc_now(),
        reports=reports,new_native_worlds=0,new_optimizer_updates=0,training_modified=False,
        same_future_native_hazards_guaranteed=False,
        scope='Complete-game descriptive evidence;observed endpoints only;Bomb protection and stage exposure differ;not a causal reward or inference ablation',
        limitations=['Unobserved between-snapshot windows are not invented',
            'Observed collision counts and measured four-frame movement targets have different coverage',
            'Full-game reward sums are undiscounted;not PPO advantages or discounted returns',
            'A saved collision is not a successful movement dodge;deathbomb skill remains useful',
            'No change to the currently frozen training method'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(args.output/'report.json',result)
    (args.output/'audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({name:{group:r[group] for group in ('all_games','clear_games')} for name,r in reports.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--report',action='append',required=True,help='condition=sealed report path')
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
