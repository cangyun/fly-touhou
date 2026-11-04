"""Inventory immutable published full19 collision receipts during an evaluation.

This partial event subset is not a completed game comparison or a collision
attribution oracle. Geometry proximity alone does not identify the native hit.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher
from fly_touhou.thrl_laser_risk_labels import route_clearances


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    teacher=MovementSafetyTeacher();rows=[]
    folder=(args.source/'full_game_traces').resolve()
    receipts=sorted(folder.glob('*_first_observed_collision.json'))
    for path in receipts:
        receipt=read(path)
        if (receipt['schema']!='B1-full19-native-observed-event-tail-v1'
                or receipt['checkpoint_sha256']!=args.sha256 or receipt.get('collision_observed') is not True
                or receipt.get('natural_terminal') is not False):
            raise ValueError('Explicit sealed native observed-collision event required')
        arrays=Path(receipt['arrays']).resolve();tape=Path(receipt['action_tape']['path']).resolve()
        if not all(p.is_relative_to(folder) for p in (arrays,tape)):raise ValueError('Trace escaped its authoritative folder')
        if hashlib.sha256(arrays.read_bytes()).hexdigest()!=receipt['sha256']:raise ValueError('Sealed event input hash differs')
        actions=tape.read_bytes()
        if (hashlib.sha256(actions).hexdigest()!=receipt['action_tape']['sha256']
                or len(actions)!=receipt['acknowledged_decisions']):raise ValueError('Sealed source19 action prefix differs')
        with np.load(arrays,allow_pickle=False) as z:
            f,o=z['features'][-1],z['objects'][-1];action=int(z['action'][-1])
            stage=int(z['stage'][-1]);frame=int(z['episode_frame'][-1])
        present=o[:,0]>.5;lasers=present&((o[:,3]>.5)|(o[:,4]>.5))&(o[:,13]>.5)
        curve=lasers&(o[:,4]>.5);known=lasers&(o[:,26]>.5)
        origin=teacher._scene(f,o)[0]
        gap=route_clearances(f,o,origin[None],np.zeros((1,2)),duration=0)[0] if lasers.any() else np.inf
        rows.append(dict(game=receipt['game'],stage=stage,episode_frame=frame,action=action,
            source_receipt=str(path),source_trace_sha256=receipt['sha256'],position=origin.tolist(),
            laser_tokens=int(lasers.sum()),curve_tokens=int(curve.sum()),known_motion_laser_tokens=int(known.sum()),
            unknown_motion_laser_tokens=int((lasers&~known).sum()),
            current_conservative_laser_capsule_gap=float(gap) if np.isfinite(gap) else None,
            nearby_laser_capsule_within18px=bool(gap<=18),
            nonmap_uncertainty_guard=teacher._uncertainty(f,o,None,horizon=4),
            input_table_saturated=int(present.sum())==96,
            exact_native_collision_source_observed=False))
    summary={}
    for stage in sorted({r['stage'] for r in rows}):
        subset=[r for r in rows if r['stage']==stage]
        summary[str(stage)]=dict(sealed_events=len(subset),with_laser_tokens=sum(r['laser_tokens']>0 for r in subset),
            with_curve_tokens=sum(r['curve_tokens']>0 for r in subset),
            with_unknown_motion=sum(r['unknown_motion_laser_tokens']>0 for r in subset),
            nearby_laser_capsule_within18px=sum(r['nearby_laser_capsule_within18px'] for r in subset),
            nonmap_uncertainty_guard=dict(Counter(r['nonmap_uncertainty_guard'] or 'none' for r in subset)))
    args.output.mkdir(parents=True)
    write(args.output/'report.json',dict(schema='B1-sealed-full19-observed-collision-event-inventory-v1',utc=utc_now(),
        source=str(args.source),checkpoint_sha256=args.sha256,source_files_snapshot=[str(p) for p in receipts],
        summary=summary,rows=rows,partial_event_subset_is_not_full_game_score=True,
        exact_native_collision_source_observed=False,native_worlds_started=0,optimizer_updates=0,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    (args.output/'audit_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
