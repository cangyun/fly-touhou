"""Compare original-speed and NULLREF uncapped native seeded trajectories.

New owned children only; no desktop input. Strict semantic multisets preserve
duplicate bullets; allocator-dependent raw addresses are excluded, not rounded.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def canonical(snapshot):
    def entities(rows):
        return sorted([{k:v for k,v in asdict(row).items() if k!='id'} for row in rows],
                      key=lambda row:json.dumps(row,sort_keys=True))
    geo=snapshot.extra['native_collision_geometry']
    def records(rows):
        return sorted([{k:v for k,v in row.items() if k!='id'} for row in rows],
                      key=lambda row:json.dumps(row,sort_keys=True))
    return {'stage_frame':snapshot.stage_frame,'stage':snapshot.stage,
        'player':asdict(snapshot.player) if snapshot.player else None,
        'bullets':entities(snapshot.bullets),'lasers':entities(snapshot.lasers),
        'enemies':entities(snapshot.enemies),'lives':snapshot.lives,'bombs':snapshot.bombs,
        'power':snapshot.power,'score':snapshot.score,'graze':snapshot.graze,
        'rng':snapshot.extra['native_rng_state'], 'dialogue':snapshot.dialogue,
        'geometry':{'player':geo['player'],'bullets':records(geo['bullets'])},
        'items':records(snapshot.extra['resources']['items'])}


def run(args,name,headless):
    from fly_touhou.thrl_native_events import HiddenTHRLDialogueGeometryController
    from fly_touhou.background import action_mask
    out=args.output/name;out.mkdir()
    c=HiddenTHRLDialogueGeometryController(args.executable,args.profile,
        private_appdata=out/'userdata',worker_name=name,timeout=25,
        headless=headless,native_seed=args.seed,checkpoint_wait_sleep_ms=10)
    started=time.perf_counter()
    try:
        c.start(); s=c.ensure_run(3)
        while s.stage_frame<64: s=c.advance(1,1)
        if s.stage_frame!=64 or not s.extra['game_rng_seed_applied']:
            raise ValueError('Controlled native start not reached')
        begin=time.perf_counter();trace=[canonical(s)]
        # Deterministic repeated lateral movement, firing held. This is a tape,
        # not a policy, teacher, trained result, or an invulnerability patch.
        for i in range(args.decisions):
            a=(0,1,0,2,0,5,0,8)[(i//32)%8]
            s=c.advance(action_mask(a),4)
            trace.append(canonical(s))
        elapsed=time.perf_counter()-begin
        text=json.dumps(trace,sort_keys=True,separators=(',',':'))
        (out/'trace.json').write_text(text,encoding='utf-8')
        return {'status':'passed','headless':headless,'pid':c.pid,
            'trace_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'elapsed_seconds':elapsed,'decisions_per_second':args.decisions/elapsed,
            'logical_frames_per_second':args.decisions*4/elapsed,
            'seed':s.extra['native_seed'],'execution':s.extra['native_execution'],
            'startup_seconds':begin-started},trace
    finally: c.close()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--executable',required=True,type=Path);p.add_argument('--profile',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--seed',type=int,default=12345)
    p.add_argument('--decisions',type=int,default=256)
    a=p.parse_args()
    if not 1<=a.decisions<=1024:raise ValueError('Bounded comparison only')
    a.output.mkdir(parents=True,exist_ok=False)
    report={'schema':'th14-execution-equivalence-v1','status':'running',
            'scope':'stage1 fixed input tape; full game equivalence not established'}
    try:
        runs={};traces={}
        for name,headless in [('native',False),('headless',True),('headless_repeat',True)]:
            runs[name],traces[name]=run(a,name,headless)
            (a.output/'progress.json').write_text(json.dumps(runs,indent=2),encoding='utf-8')
        differences={}
        for name in ('headless','headless_repeat'):
            mismatch=next((i for i,(x,y) in enumerate(zip(traces['native'],traces[name])) if x!=y),None)
            if mismatch is not None:
                x,y=traces['native'][mismatch],traces[name][mismatch]
                differences[name]={'index':mismatch,'stage_frame':x['stage_frame'],
                    'keys':[k for k in x if x[k]!=y[k]]}
        report.update(status='passed' if not differences else 'mismatch',runs=runs,differences=differences,
            compared_checkpoints=a.decisions+1, no_float_rounding=True,
            native_rng_controlled=True, duplicates_preserved=True,
            trajectory_equivalence_verified_in_tested_scope=not differences)
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        (a.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
