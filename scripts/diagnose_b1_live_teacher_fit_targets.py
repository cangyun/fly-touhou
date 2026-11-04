"""Read-only teacher-label diagnostic on one committed training prefix.

Replays from the actual opening without resetting the GRU between chunks.
No fitting, native game, geometry takeover, or completed-episode claim.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]


def main(args):
    sys.path.insert(0,str(args.runtime/'src'))
    import numpy as np,torch,psutil
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.thrl_study_io import utc_now
    from fly_touhou.thrl_teacher_archive import TeacherTrajectoryReader
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_policy_actions import policy_logits
    if args.output.exists():raise FileExistsError(args.output)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for the frozen-policy prefix replay')
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=args.checkpoint_sha256:
        raise ValueError('Initializer checkpoint changed')
    reader=TeacherTrajectoryReader(args.world,require_complete=False,cache_bytes=32*1024**2)
    meta=reader.index['metadata']
    if meta.get('native_seed_split')!='train' or not meta.get('use_for_training') or meta.get('qualification_only'):
        raise ValueError('Only current real training-partition teacher prefixes are inspected')
    rows=min(reader.rows,args.max_rows)
    if rows<32:raise ValueError('Wait for committed original observations')
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.25)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=model_from_config(**payload['model_config']).cuda().eval().requires_grad_(False)
    model.gru.flatten_parameters();del payload
    args.output.mkdir(parents=True);started=time.monotonic()
    write(args.output/'launch.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now(),native_worlds=0,optimizer_steps=0))
    write(args.output/'source_index_snapshot.json',reader.index)
    hidden=torch.zeros(1,1,model.gru_hidden_size,device='cuda')
    all_values=[];previous=None
    with torch.no_grad():
        for start in range(0,rows,32):
            stop=min(rows,start+32)
            data=reader.fetch(start,stop)
            if (start==0 and (data['before_episode_frames'][0]!=0 or not data['episode_starts'][0])):
                raise ValueError('Replay must begin at the actual native opening')
            if (previous is not None and data['before_episode_frames'][0]!=previous
                    or np.any(data['episode_starts'][1:]) or np.any(data['dones'][:-1])):
                raise ValueError('Prefix replay must not invent a recurrent reset')
            if np.any(data['before_episode_frames'][1:]!=data['after_episode_frames'][:-1]):
                raise ValueError('Original observations are not contiguous')
            previous=int(data['after_episode_frames'][-1])
            tensors={k:torch.from_numpy(data[k]).to('cuda') for k in ('features','maps','objects')}
            encoded=model._encode(tensors['features'],tensors['maps'],tensors['objects'])
            outputs=[]
            for t in range(stop-start):
                value,hidden=model.gru(encoded[t:t+1].unsqueeze(0),hidden)
                outputs.append(value[0,0])
            raw=model._adjust(model.base._policy_logits(torch.stack(outputs),tensors['features']),
                tensors['features'],tensors['objects'])
            logits=policy_logits(raw);actions=torch.from_numpy(data['actions']).long().to('cuda')
            legal=(actions!=18)|(tensors['features'][:,4]>0)
            target=torch.where(legal,actions,torch.zeros_like(actions))
            losses=torch.nn.functional.cross_entropy(logits,target,reduction='none')
            probability=logits.softmax(-1).gather(1,target[:,None]).squeeze(1)
            target_raw=raw.gather(1,target[:,None]).squeeze(1)
            vals=torch.stack((losses,probability,target_raw,(logits.argmax(-1)==target).float(),
                legal.float()),dim=1).cpu().numpy()
            all_values.append(np.column_stack((data['stage'],data['actions'],vals)))
            if start%1024==0:
                write(args.output/'progress.json',dict(status='read_only_label_replay',completed=stop,target=rows,
                    wall_seconds=time.monotonic()-started,optimizer_steps=0,native_worlds=0))
    values=np.concatenate(all_values)
    def summarize(mask):
        v=values[mask];legal=v[:,6]>.5;v=v[legal]
        if not len(v):return dict(rows=0)
        return dict(rows=len(v),hard_teacher_CE=float(v[:,2].mean()),executed_action_top1_agreement=float(v[:,5].mean()),
            mean_teacher_action_probability=float(v[:,3].mean()),
            teacher_action_outside_policy_logit_bounds_fraction=float((np.abs(v[:,4])>=80).mean()),
            teacher_action_raw_logit_min=float(v[:,4].min()),teacher_action_raw_logit_max=float(v[:,4].max()),
            teacher_action_CE_percentile95=float(np.quantile(v[:,2],.95)),
            teacher_movement_labels=int((v[:,1]<18).sum()),teacher_Bomb_labels=int((v[:,1]==18).sum()))
    result=dict(schema='B1-one-training-teacher-prefix-frozen-label-fit-diagnostic-v1',status='completed',utc=utc_now(),
        world=str(args.world),native_seed=meta['native_seed'],checkpoint=str(args.checkpoint),checkpoint_sha256=args.checkpoint_sha256,
        snapshotted_committed_rows=reader.rows,inspected_contiguous_prefix_rows=rows,
        reached_native_episode_frame=previous,overall=summarize(np.ones(len(values),dtype=bool)),
        by_stage={str(stage):summarize(values[:,0]==stage) for stage in np.unique(values[:,0]).astype(int)},
        invalid_empty_Bomb_labels=int((values[:,6]<.5).sum()),
        source_is_incomplete_prefix=not bool(reader.index['natural_episode_complete']),
        source_prefix_not_reclassified_as_complete_training_data=True,exact_full_prefix_recurrent_replay=True,
        measured_label_fit_not_native_student_performance=True,not_representative_of_all_training_worlds=True,
        CUDA=True,optimizer_steps=0,neural_parameters_modified=False,native_worlds_started=0,
        wall_seconds=time.monotonic()-started)
    write(args.output/'report.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('runtime','checkpoint','world','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--max-rows',type=int,default=16384)
    main(p.parse_args())
