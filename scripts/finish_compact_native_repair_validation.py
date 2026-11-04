"""Finish the already dispatched repair validation and build its teacher bank.

Reads committed records only. Does not launch a game, fit a model or resume PPO.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from fly_touhou.thrl_study_io import read_shared_json as read
from fly_touhou.thrl_durable_records import write_json


def teacher_audit(directory):
    import numpy as np
    directory=Path(directory);index=read(directory/'index.json')
    parts=[]
    for chunk in index['chunks']:
        with np.load(directory/chunk['file'],allow_pickle=False) as data:
            parts.append({key:data[key] for key in ('features','actions','margins','trusted',
                'unprotected_mask','outcome','native_action_frames','stage','partial_point_danger',
                'teacher_guidance_weight','misses_before','bombs_used_before','before_episode_frames')})
    if not parts:return dict(source=str(directory),status=index['status'],rows=0,stages={})
    data={key:np.concatenate([part[key] for part in parts]) for key in parts[0]}
    actions=data['actions'];moving=actions<18;normal=data['unprotected_mask'];dt=data['native_action_frames']
    # The feature retains pre-action trust; the training mask may already have
    # been vetoed by the observed outcome. Do not hide such contradictions.
    before_trusted=data['features'][:,72]>.5
    selected=data['margins'][np.arange(len(actions)),0,np.minimum(actions,17)]
    contact=normal&moving&(data['outcome']==1)&(dt>0)&(dt<=4)
    clear=normal&moving&(data['outcome']==0)&(dt==4)
    stages={}
    for stage in sorted(set(data['stage'].tolist())):
        mask=data['stage']==stage;first=int(np.flatnonzero(mask)[0])
        critical=(data['features'][:,4]>0)&(data['features'][:,12]>0)&(data['features'][:,17]>.5)
        stages[str(stage)]=dict(rows=int(mask.sum()),normal_rows=int((mask&normal).sum()),
            guided_rows=int((mask&(data['teacher_guidance_weight']>0)).sum()),
            trusted4_normal_rows=int((mask&normal&before_trusted).sum()),
            retained_trusted4_rows=int((mask&normal&data['trusted'][:,0]).sum()),
            partial_danger_rows=int((mask&normal&data['partial_point_danger'].any((1,2))).sum()),
            observed_contacts=int((mask&contact).sum()),
            trusted_safe_native_contacts=int((mask&contact&before_trusted&(selected>0)).sum()),
            unknown_forecast_contacts=int((mask&contact&~before_trusted).sum()),
            trusted_danger_native_clears=int((mask&clear&before_trusted&(selected<0)).sum()),
            ordinary_Bomb_requests=int((mask&(actions==18)&~critical).sum()),
            deathbomb_requests=int((mask&(actions==18)&critical).sum()),
            entry_misses=int(data['misses_before'][first]),
            entry_Bombs_used=int(data['bombs_used_before'][first]),
            entry_episode_frames=int(data['before_episode_frames'][first]))
    return dict(source=str(directory),status=index['status'],rows=len(actions),stages=stages,
        episode_summary=index.get('episode_summary'),native_repair_revisions=index.get('native_repair_revisions'),
        scope='Executed native actions only; other directions remain forecasts. Bomb requests are not inventory consumption.')


def frozen_audit(directory):
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    directory=Path(directory);path=directory/'development/diagnostics/events.jsonl'
    events=[]
    if path.exists():
        import json
        with _shared_binary(path) as stream:
            for line in stream:
                if line.endswith(b'\n'):events.append(json.loads(line))
    stages={};contradictions=[]
    for event in events:
        before=event['before'];after=event['after'];stage=str(before['stage'])
        totals=stages.setdefault(stage,Counter())
        contact=bool({'contact_window_entered','life_lost'}&set(event['kinds']))
        normal=event.get('pre_action_normal',False);action=event.get('last_action',18)
        dt=after['episode_frames']-before['episode_frames']
        trusted=event.get('pre_action_trusted',[False])[0]
        if contact and normal and action<18 and 0<dt<=4:
            totals['observed_contacts']+=1
            totals['trusted_contacts' if trusted else 'unknown_forecast_contacts']+=1
            if trusted and event.get('chosen_immediate_margin',0)>0:
                totals['trusted_safe_native_contacts']+=1;contradictions.append(event)
        if 'Bomb_used' in event['kinds']:
            totals['actual_deathbombs' if event.get('pre_action_deathbomb_window') else 'actual_ordinary_Bombs']+=1
    report=directory/'development/report.json'
    return dict(source=str(directory),completed=report.exists(),
        report=read(report) if report.exists() else None,stages=stages,
        trusted_safe_native_contact_events=contradictions,
        scope='Frozen fly decisions; no teacher control or parameter updates. Count contact windows once, not again at the later life decrement.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('jobs','baseline','output','bank-output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    jobs=read(args.jobs)
    try:
        while True:
            status={job['name']:read(Path(job['control'])/'status.json')['status'] for job in jobs}
            write_json(args.output/'status.json',dict(phase='waiting_for_dispatched_native_games',jobs=status,
                production_optimizer_updates=0,training_restart_enabled=False))
            failed={name:value for name,value in status.items() if value in ('failed','launcher-error')}
            if failed:raise RuntimeError('Native validation job failed: '+str(failed))
            if all(value=='finished' for value in status.values()):break
            if (args.output/'STOP').exists():raise InterruptedError('Stopped validation finalizer; producers left alone')
            time.sleep(10)
        teachers=[job for job in jobs if job['name'].startswith('teacher-')]
        audits=[dict(name=job['name'],**teacher_audit(job['output'])) for job in teachers]
        frozen=frozen_audit(next(job['output'] for job in jobs if job['name']=='frozen43'))
        baseline=read(args.baseline)
        train=[job['output'] for job in teachers if job['name'].startswith('teacher-train')]
        held=[job['output'] for job in teachers if job['name']=='teacher-heldout']
        train_stages={int(stage) for audit in audits if audit['name'].startswith('teacher-train') for stage in audit['stages']}
        held_stages={int(stage) for audit in audits if audit['name']=='teacher-heldout' for stage in audit['stages']}
        available=sorted(train_stages&held_stages)
        bank=None
        if {3,4}.issubset(available):
            from prepare_compact_native_rule_multigame import make_multigame_bank
            result=make_multigame_bank(train,held,args.bank_output,stages=available,count=64)
            bank=dict(manifest=str(args.bank_output/'manifest.json'),windows=len(result['windows']),stages=available)
        result=dict(status='native_repair_validation_completed',teachers=audits,frozen=frozen,
            baseline=dict(path=str(args.baseline),clears=baseline['clears'],stage_counts=baseline['stage_counts']),
            teacher_bank=bank,training_restarted=False,production_optimizer_updates=0,
            scope='Engineering and frozen development evidence. A repaired interface does not itself establish improved learned avoidance or stable clears.')
        write_json(args.output/'report.json',result)
        write_json(args.output/'status.json',dict(phase='completed',report=str(args.output/'report.json'),
            teacher_bank=bank,training_restarted=False))
    except BaseException as exc:
        write_json(args.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),training_restarted=False))
        raise


if __name__=='__main__':main()
