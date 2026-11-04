"""Offline GPU gradient qualification on bounded recorded native observations.

No game, collector, rollout replay training or registered checkpoint writes.
One isolated Adam step for each full-size temporal core checks auxiliary wiring.
"""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def run(source,output):
    import numpy as np
    import torch
    from fly_touhou.atomic_json_io import atomic_write_json
    from fly_touhou.thrl_model import MOActorCritic
    from fly_touhou.thrl_movement_learning import prepare_targets,safe_set_loss,collision_action_loss
    from fly_touhou.thrl_inputs import SEPARATED_OBJECT_PRIORITY_INPUT as INPUT
    if output.exists():raise FileExistsError(output)
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA qualification required')
    torch.set_num_threads(2);torch.manual_seed(241024)
    with np.load(source/'states.npz') as z:
        hit=np.flatnonzero(z['outcome']==1);other=np.flatnonzero(z['outcome']==0)
        ids=np.r_[hit,other[:64-len(hit)]]
        if len(ids)!=64 or len(hit)<20:raise ValueError('Need64 actual observations including20 actual collisions')
        data={k:torch.from_numpy(z[k][ids].copy()) for k in ('features','objects','maps','action','outcome')}
    p=torch.load(source/'frozen.pt',map_location='cpu',weights_only=False)
    records=[]
    for core in ('mlp','gru'):
        config={**p['model_config'],'temporal_core':core}
        model=MOActorCritic(**config).cuda()
        state=model.state_dict();common={k:v for k,v in p['model'].items() if k in state and v.shape==state[k].shape}
        model.load_state_dict({**state,**common},strict=True)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        # Independent observations with explicit per-step hidden resets. This
        # is a wiring check, not fabricated on-policy PPO trajectories.
        features=data['features'].reshape(4,16,25).cuda();maps=data['maps'].reshape(4,16,16,92,96).cuda()
        objects=data['objects'].reshape(4,16,96,21).cuda()
        before={k:v.detach().clone() for k,v in model.named_parameters()}
        t=time.monotonic()
        safe,y,detail=prepare_targets(data['features'].reshape(16,4,25),data['objects'].reshape(16,4,96,21),
            data['maps'].reshape(16,4,16,92,96),data['action'].reshape(16,4),data['outcome'].reshape(16,4),
            input_schema=INPUT,max_labels=128,teacher_enabled=True)
        preparation=time.monotonic()-t
        # Keep the same flat observation order in both target and sequence APIs.
        safe=safe.reshape(64,18).cuda();y=y.reshape(64).cuda()
        model.train();torch.cuda.synchronize();t=time.monotonic()
        logits,values=model.forward_sequence(features,maps,torch.zeros(1,4,model.gru_hidden_size,device='cuda'),
            torch.ones(4,16,dtype=torch.bool,device='cuda'),objects=objects)
        geometric,ns=safe_set_loss(logits,safe)
        factual,nc=collision_action_loss(logits,data['action'].cuda(),y)
        loss=.02*geometric+.02*factual
        optimizer.zero_grad(set_to_none=True);loss.backward()
        gradients=[v.grad for v in model.parameters() if v.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):raise ValueError('Nonfinite CUDA movement gradients')
        torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step();torch.cuda.synchronize()
        changed=sum(not torch.equal(before[k],v) for k,v in model.named_parameters())
        if not changed or not ns or nc!=20:raise ValueError('Movement sources did not reach real full-size model')
        records.append(dict(core=core,device=str(next(model.parameters()).device),parameters=model.parameter_counts()['total'],
            safe_sets=ns,collision_facts=nc,loss=float(loss.detach()),gradient_tensors=len(gradients),changed_tensors=changed,
            teacher_seconds=preparation,Adam_seconds=time.monotonic()-t,all_gradients_finite=True,
            offline_Adam_steps=1,simulation_policy_updates=0))
        del model,optimizer,before,features,maps,objects,logits,values,gradients
        torch.cuda.empty_cache()
    result=dict(status='passed',schema='th14-offline-movement-CUDA-qualification-v1',
        GPU=torch.cuda.get_device_name(),cases=records,source=str(source),
        native_state_bundle_sha256=hashlib.sha256((source/'states.npz').read_bytes()).hexdigest(),
        production_model_files_modified=False,training_jobs_started=0,native_games_started=0,
        limitations='Only isolated auxiliary gradient wiring; no PPO learning improvement claim')
    output.parent.mkdir(parents=True,exist_ok=True);atomic_write_json(output,result)
    return result


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.source,a.output),ensure_ascii=False))
