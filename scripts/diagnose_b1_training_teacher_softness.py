"""Measure actual archived training targets before choosing post-BC losses.

No model, fitting, native client or teacher execution. Hypothetical logit
gradients are explicitly a compatibility check, not measured PPO dominance.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import utc_now


def run(args):
    import numpy as np
    from fly_touhou.thrl_teacher_archive import TeacherTrajectoryReader
    if args.output.exists():raise FileExistsError(args.output)
    all_rows=[];sources=[];started=time.monotonic()
    for world in range(5):
        folder=args.folder/'trajectories'/f'world{world:03d}'
        reader=TeacherTrajectoryReader(folder,require_complete=False,cache_bytes=0)
        if reader.index['metadata']['native_seed_split']!='train':raise ValueError('Training data only')
        limit=min(16384,reader.rows);coords=[]
        for start in range(0,limit-31,1024):
            stop=start+32
            data=reader.fetch(start,stop,fields=['features','targets','actions','stage','teacher_guidance_weight'])
            q=data['targets'].astype(np.float64)
            if (np.any(q<0) or not np.allclose(q.sum(-1),1,atol=2e-7,rtol=0)
                    or not np.array_equal(q.argmax(-1),data['actions'])):
                raise ValueError('Actual teacher targets/action identity differs')
            legal=(data['features'][:,4]>0)
            entropy=-(q*np.log(np.maximum(q,1e-300))).sum(-1)
            all_rows.append(np.column_stack((data['stage'],data['actions'],entropy,q.max(-1),
                (q>1e-3).sum(-1),18+legal,data['teacher_guidance_weight'],
                (~legal)&(q[:,18]>1e-8))))
            coords.append([start,stop])
        sources.append(dict(world=world,native_seed=reader.index['metadata']['native_seed'],
            committed_rows_at_read=reader.rows,selected_intervals=coords,
            canonical_index_snapshot_sha256=hashlib.sha256(json.dumps(reader.index,sort_keys=True).encode()).hexdigest()))
    values=np.concatenate(all_rows)
    def summarize(mask):
        v=values[mask]
        if not len(v):return dict(rows=0)
        return dict(rows=len(v),mean_target_entropy_nats=float(v[:,2].mean()),
            mean_entropy_fraction_of_uniform_legal=float((v[:,2]/np.log(v[:,5])).mean()),
            mean_argmax_target_probability=float(v[:,3].mean()),median_argmax_target_probability=float(np.median(v[:,3])),
            mean_actions_with_target_probability_over_point001=float(v[:,4].mean()),
            teacher_Bomb_rows=int((v[:,1]==18).sum()),
            target_has_mass_on_empty_Bomb_rows=int((v[:,7]>.5).sum()),
            hypothetical_probability_point8_on_teacher_action=dict(
                mean_soft_CE_derivative_wrt_teacher_action_logit=float((.8-v[:,3]).mean()),
                hard_CE_derivative_wrt_teacher_action_logit=-.2,
                soft_CE_pushes_teacher_action_down_fraction=float((v[:,3]<.8).mean()),
                not_actual_policy_gradients=True,not_measured_PPO_dominance=True))
    result=dict(schema='B1-current-training-teacher-soft-target-compatibility-v1',status='completed',utc=utc_now(),
        selection='Declared first five training worlds;first32 rows per1024 decisions;at most16384 rows/world',
        sources=sources,all_sampled_training_rows=summarize(np.ones(len(values),bool)),
        usable_online_teacher_rows=summarize(values[:,6]>0),
        by_stage={str(stage):summarize((values[:,0]==stage)&(values[:,6]>0)) for stage in np.unique(values[:,0]).astype(int)},
        actual_policy_performance_not_measured=True,actual_PPO_gradient_balance_not_measured=True,
        main_neural_optimizer_updates=0,neural_model_loaded=False,native_worlds_started=0,
        current_training_method_changed=False,wall_seconds=time.monotonic()-started)
    args.output.mkdir(parents=True);write(args.output/'report.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','by_stage')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
