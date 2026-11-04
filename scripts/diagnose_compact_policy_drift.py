"""Split saved-rollout policy drift into Bomb and conditional movement KL.

Uses a separate frozen model and the saved pre-sequence brain states. It does
not access the original game, update the model, or change the live trainer.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','rollout','connectome','cache','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--scenes',type=int,default=4)
    args=parser.parse_args()
    os.environ.update(FLY_TOUHOU_DETERMINISTIC_SPMM='1',
                      FLY_TOUHOU_SPMM_CACHE=str(args.cache/'spmm-kernel-cache'),
                      FLY_TOUHOU_EDGE_GRAD_CACHE=str(args.cache/'kernel-cache'),
                      CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    from fly_touhou.runtime_cache_paths import configure_generated_caches
    configure_generated_caches()
    import gc
    import numpy as np
    import torch
    from fly_touhou import connectome as graph_module
    from fly_touhou.compact_story_behavior import story_behavior_logits
    from fly_touhou.compact_policy_kl import action_kl_parts
    from fly_touhou.compact_story_model import from_story_checkpoint,sequence
    from fly_touhou.spiking_engineered_control import SpikingState
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    from fly_touhou.thrl_durable_records import write_json
    torch.set_num_threads(2)
    with _shared_binary(args.checkpoint) as stream:
        checkpoint=torch.load(stream,map_location='cpu',weights_only=False)
    point=checkpoint['model_payload'];costs=dict(point.get('costs',{}))
    completed_cycle=checkpoint.get('completed_cycle')
    del checkpoint
    with _shared_binary(args.rollout) as stream:
        tape=torch.load(stream,map_location='cpu',weights_only=False)
    old_logits=tape.get('recorded_behavior_policy_logits')
    if old_logits is None:
        raise ValueError('The saved rollout does not contain the acting behavior distribution')
    load=graph_module.load_connectome
    graph_module.load_connectome=lambda path,**kw:load(path,**dict(kw,verify=False))
    try:
        model=from_story_checkpoint(args.connectome,point,device='cuda:0').eval()
    finally:
        graph_module.load_connectome=load
    del point
    gc.collect()
    T,N=tape['actions'].shape;length=16
    blocks=np.linspace(0,T//length-1,min(args.scenes,T//length),dtype=int)
    stats=[]
    with torch.no_grad():
        for i,block in enumerate(blocks):
            t=int(block)*length;world=i%N
            features=tape['features'][t:t+length,world:world+1].to(model.device)
            actions=tape['actions'][t:t+length,world:world+1].to(model.device)
            starts=tape['episode_starts'][t:t+length,world:world+1].to(model.device)
            state=SpikingState.pack([tape['block_states'][t].row(world)]).to(model.device)
            output=sequence(model,features,starts,state,actions,checkpointed=False)
            new=story_behavior_logits(output['logits'],tape['metadata']['movement_sampling_temperature'],
                                      tape['metadata']['bomb_sampling_temperature'])
            old=old_logits[t:t+length,world:world+1].to(model.device)
            total,bomb,move=action_kl_parts(old,new)
            stats.append(dict(time_index=t,world=world,stage=int(round(float(features[0,0,111])*6)),
                rows=length,total_kl=float(total.mean()),bomb_kl=float(bomb.mean()),
                movement_contribution_kl=float(move.mean()),
                decomposition_error=float((total-bomb-move).abs().max())))
    result=dict(checkpoint=str(args.checkpoint),checkpoint_costs=costs,completed_cycle=completed_cycle,
        rollout=str(args.rollout),sequences=stats,
        scope=('Separate frozen forward on a few saved student training sequences, not a game or independent evaluation. '
               'Same saved initial neural states and sampling temperatures as PPO. Bomb KL is Bernoulli; '
               'movement KL is conditional and weighted by the old movement probability.'))
    write_json(args.output,result)
    print(result)


if __name__=='__main__':
    main()
