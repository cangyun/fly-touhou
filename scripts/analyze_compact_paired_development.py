"""Compare frozen development runs by the same original-game opening seeds."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json
from fly_touhou.thrl_dodge_diagnostics import _shared_binary


def shared_jsonl_prefix(path: Path) -> list[dict]:
    """Read completed records while the native evaluator keeps appending."""
    if not path.exists():
        return []
    records=[]
    with _shared_binary(path) as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if line.endswith(b'\n'):
                    raise
                break
    return records


def evaluation_snapshot(path: Path) -> tuple[dict, dict, list[dict]]:
    directory=path if path.is_dir() else path.parent
    report_path=directory/'report.json' if path.is_dir() else path
    plan_path=directory/'episode_plan.json'
    plan=read_shared_json(plan_path) if plan_path.exists() else {}
    if report_path.exists():
        return read_shared_json(report_path),plan,[]
    progress_path=directory/'progress.json'
    progress=read_shared_json(progress_path) if progress_path.exists() else {}
    results=shared_jsonl_prefix(directory/'episodes.jsonl')
    finished={row['evaluation_game']-1 for row in results}
    active=[row for row in progress.get('active',[]) if row['world'] not in finished]
    reached=[max(row['stage_history']) for row in results]
    reached.extend(int(row['stage']) for row in active if row.get('stage') is not None)
    return dict(status='in_progress',episodes=progress.get('target',len(plan.get('seeds',[]))),
        completed_episodes=len(results),clears=sum(row['outcome']=='clear' for row in results),
        action_selection=progress.get('action_selection',plan.get('action_selection')),
        evaluation_split=plan.get('evaluation_split'),episode_results=results,
        stage_counts={str(stage):sum(value>=stage for value in reached) for stage in range(1,7)},
        dodge_diagnostics=dict(event_summaries=shared_jsonl_prefix(directory/'diagnostics/events.jsonl'))),plan,active


def summarize(path: Path) -> dict:
    report,plan,active=evaluation_snapshot(path)
    per_world=defaultdict(lambda:defaultdict(Counter))
    per_stage=defaultdict(Counter)
    arrivals=defaultdict(dict)
    for event in report['dodge_diagnostics']['event_summaries']:
        before=event['before'];stage=int(before['stage']);world=int(event['world'])
        kinds=event['kinds'];updates=Counter()
        if 'stage_transition' in kinds:
            after=event['after'];destination=int(after['stage'])
            arrivals[world][str(destination)]={key:after[key] for key in
                ('lives','bombs','misses','bombs_used','episode_frames')}
        if 'life_lost' in kinds:
            updates['lives_lost']+=1
            updates['life_lost_with_no_bombs']+=int(before['bombs']==0)
        if 'contact_window_entered' in kinds:
            updates['contact_windows']+=1
            updates['contact_with_untrusted_4f']+=int(not event['pre_action_trusted'][0])
        if 'Bomb_used' in kinds:
            updates['bombs']+=1
            updates['observed_deathbombs']+=int(event['pre_action_deathbomb_window'])
            trusted=bool(event['pre_action_trusted'][0])
            ordinary=not bool(event['pre_action_deathbomb_window'])
            safe=event.get('geometric_immediate_safe_candidates')
            updates['bombs_with_untrusted_4f']+=int(not trusted)
            updates['ordinary_or_unconfirmed_bombs_with_untrusted_4f']+=int(ordinary and not trusted)
            updates['bombs_with_modeled_safe_move']+=int(trusted and safe is not None and safe>0)
            updates['ordinary_or_unconfirmed_bombs_with_no_modeled_safe_move']+=int(
                ordinary and trusted and safe==0)
        per_world[world][stage].update(updates)
        per_stage[stage].update(updates)
    episodes=[]
    for episode in report['episode_results']:
        world=episode['evaluation_game']-1
        episodes.append(dict(seed=episode['full_story_native_seed'],evaluation_seed=episode.get('evaluation_seed'),
            world=world,outcome=episode['outcome'],
            final_stage=episode['stage'],stage_frame=episode['stage_frame'],
            native_frames=episode['native_frames'],misses=episode['misses'],bombs_used=episode['bombs_used'],
            stage_arrivals=arrivals[world],
            stages={str(k):dict(v) for k,v in sorted(per_world[world].items())}))
    active_worlds=[dict(row,stage_arrivals=arrivals[row['world']],
        stages={str(k):dict(v) for k,v in sorted(per_world[row['world']].items())}) for row in active]
    return dict(source=str(path),complete=report['status']=='completed',
                target_episodes=report['episodes'],completed_episodes=len(episodes),
                planned_opening_seeds=plan.get('seeds'),active_worlds=active_worlds,
                action_selection=report['action_selection'],
                evaluation_split=report['evaluation_split'],clears=report['clears'],
                stage_counts=report['stage_counts'],episodes=episodes,
                stage_totals={str(k):dict(v) for k,v in sorted(per_stage.items())})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',action='append',required=True,help='LABEL=report.json or evaluation directory (live or completed)')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    runs={}
    for item in args.report:
        label,path=item.split('=',1)
        runs[label]=summarize(Path(path))
    reference=next(iter(runs));reference_seeds=runs[reference]['planned_opening_seeds']
    pairing={label:dict(reference=reference,
        same_planned_opening_seeds=(bool(reference_seeds) and run['planned_opening_seeds']==reference_seeds),
        same_action_selection=run['action_selection']==runs[reference]['action_selection'])
        for label,run in runs.items()}
    result=dict(runs=runs,pairing=pairing,scope=('Frozen development observations, not heldout tests. '
        'Only completed native games count as clears or failures; live worlds remain unfinished. '
        'Opening seed agreement is reported explicitly in pairing. '
        'Policies can change subsequent RNG consumption; sensor revisions can differ between runs. '
        'Only observed deathbombs are counted; other Bomb uses are ordinary or unconfirmed. '
        'Modeled safe moves are not factual alternative-action outcomes.'))
    write_json(args.output,result)
    print(json.dumps({k:dict(complete=v['complete'],completed=v['completed_episodes'],target=v['target_episodes'],
                            clears=v['clears'],stage_counts=v['stage_counts'],stage_totals=v['stage_totals'])
                      for k,v in runs.items()},ensure_ascii=False))
