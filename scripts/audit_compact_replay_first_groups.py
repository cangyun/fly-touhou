"""Replay two recorded PPO minibatches before any update, without game input.

Waits for an existing learner to return to collection so the small GPU replay
does not compete with its optimization phase. The saved checkpoint and tape
are immutable inputs; no policy, original game, or optimizer is modified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

import psutil

from fly_touhou.thrl_study_io import read_shared_json


def wait_for_collection(progress,identity,min_cycle,timeout,max_decisions=4096):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        try:
            owner=psutil.Process(identity['PID'])
            if not owner.is_running() or abs(owner.create_time()-identity['create_time'])>.01:
                return False
            current=read_shared_json(progress)
            if (current.get('phase')=='student_full_story_collection'
                    and current.get('cycle',0)>=min_cycle
                    and (current.get('acquisition') or {}).get('rollout_decisions',0)<max_decisions):
                return True
        except (psutil.NoSuchProcess,FileNotFoundError):
            return False
        time.sleep(5)
    return False


def audit(checkpoint,rollout,connectome,cache,seed,cycle,groups):
    os.environ['FLY_TOUHOU_DETERMINISTIC_SPMM']='1'
    os.environ['FLY_TOUHOU_SPMM_CACHE']=str(cache/'spmm-kernel-cache')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np
    import torch
    from fly_touhou import connectome as graph_module
    from fly_touhou.compact_story_behavior import story_behavior_logits
    from fly_touhou.compact_story_model import from_story_checkpoint,sequence
    from fly_touhou.compact_story_sequence_groups import world_aligned_groups
    from fly_touhou.spiking_engineered_control import SpikingState

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    tape=torch.load(rollout,map_location='cpu',weights_only=False)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    # This is a read-only replay of the currently used graph; skip a redundant
    # full-file digest rather than burdening the running training system.
    original_load=graph_module.load_connectome
    def load_graph(path,*,max_neurons=None,verify=True):
        return original_load(path,max_neurons=max_neurons,verify=False)
    graph_module.load_connectome=load_graph
    try:
        model=from_story_checkpoint(connectome,payload,device='cuda:0').eval()
    finally:
        graph_module.load_connectome=original_load
    length=16
    T,N=tape['actions'].shape
    order=world_aligned_groups(T,N,length,np.random.default_rng(seed+cycle))
    move_temperature=tape.get('metadata',{}).get('movement_sampling_temperature',1.)
    bomb_temperature=tape.get('metadata',{}).get('bomb_sampling_temperature',1.)
    results=[]
    with torch.no_grad():
        for batch_index,chosen in enumerate(order[:groups],start=1):
            times=torch.tensor([t for t,w in chosen])[None]+torch.arange(length)[:,None]
            worlds=torch.tensor([w for t,w in chosen])[None].expand(length,-1)
            features=tape['features'][times,worlds].to(model.device)
            actions=tape['actions'][times,worlds].to(model.device)
            starts=tape['episode_starts'][times,worlds].to(model.device)
            state=SpikingState.pack([tape['block_states'][t].row(w) for t,w in chosen]).to(model.device)
            prediction=sequence(model,features,starts,state,actions,checkpointed=False)
            behavior=story_behavior_logits(prediction['logits'],move_temperature,bomb_temperature)
            new_logp=behavior.log_softmax(-1).gather(-1,actions[...,None])[...,0]
            old_logp=tape['old_log_probs'][times,worlds].to(model.device)
            delta=(new_logp-old_logp).flatten()
            absolute=delta.abs()
            results.append(dict(minibatch=batch_index,rows=delta.numel(),
                max_absolute_logp_error=float(absolute.max()),
                mean_absolute_logp_error=float(absolute.mean()),
                p95_absolute_logp_error=float(torch.quantile(absolute,.95)),
                rows_over_0_01=int((absolute>.01).sum()),
                approximate_KL=float((delta.exp()-1-delta).mean())))
            del prediction,behavior,state,features,actions,starts,delta,absolute
    return dict(scope=('Pre-update full-connectome replay of recorded student states; '
        'no game actions, optimizer updates, or source changes.'),
        checkpoint=str(checkpoint),rollout=str(rollout),checkpoint_costs=payload.get('costs'),
        device=torch.cuda.get_device_name(),groups=results,
        comparison='Small log-probability errors on later groups argue against a replay mismatch; they do not establish why parameter updates change the policy.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--progress',type=Path,required=True)
    parser.add_argument('--pid',type=int,required=True)
    parser.add_argument('--created',type=float,required=True)
    parser.add_argument('--min-cycle',type=int,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--rollout',type=Path,required=True)
    parser.add_argument('--connectome',type=Path,required=True)
    parser.add_argument('--kernel-cache',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--cycle',type=int,required=True)
    parser.add_argument('--groups',type=int,default=2)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    status=args.output.with_suffix('.status.json')
    process=psutil.Process()
    def save_status(phase,**fields):
        status.write_text(json.dumps(dict(phase=phase,PID=process.pid,
            create_time=process.create_time(),**fields)),encoding='utf-8')
    save_status('waiting_for_collection',min_cycle=args.min_cycle)
    try:
        if not wait_for_collection(args.progress,dict(PID=args.pid,create_time=args.created),args.min_cycle,7200):
            save_status('learner_stopped_or_wait_timed_out')
        else:
            save_status('replaying_saved_batches')
            report=audit(args.checkpoint,args.rollout,args.connectome,args.kernel_cache,args.seed,args.cycle,args.groups)
            args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            save_status('completed',report=str(args.output))
            print(json.dumps(report,ensure_ascii=False))
    except Exception:
        save_status('failed',error=traceback.format_exc())
        raise
