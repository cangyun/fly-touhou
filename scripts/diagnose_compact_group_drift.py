"""Ablate parameter-group changes between two saved models on real PPO states.

One group at a time is restored on a separate frozen copy. This is a nonlinear
intervention, not an additive attribution; no trainer or game is modified.
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
    for name in ('old-training','new-parameters','rollout','connectome','cache','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--groups',nargs='+',default=['new_context_ports','inherited_sensory_ports',
                        'risk','movement','Bomb','brain'])
    parser.add_argument('--all-old',action='store_true')
    parser.add_argument('--only-new',action='store_true',
                        help='Start from the old model and change only one group at a time')
    args=parser.parse_args()
    os.environ.update(FLY_TOUHOU_DETERMINISTIC_SPMM='1',
        FLY_TOUHOU_SPMM_CACHE=str(args.cache/'spmm-kernel-cache'),
        FLY_TOUHOU_EDGE_GRAD_CACHE=str(args.cache/'kernel-cache'),
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    from fly_touhou.runtime_cache_paths import configure_generated_caches
    configure_generated_caches()
    import gc
    import torch
    from fly_touhou import connectome as graph_module
    from fly_touhou.compact_story_behavior import story_behavior_logits
    from fly_touhou.compact_policy_kl import action_kl_parts
    from fly_touhou.compact_story_model import from_story_checkpoint,sequence
    from fly_touhou.compact_story_ppo import optimizer_for
    from fly_touhou.spiking_engineered_control import SpikingState
    from fly_touhou.thrl_durable_records import write_json
    torch.set_num_threads(2)
    old=torch.load(args.old_training,map_location='cpu',weights_only=False)['model_payload']
    new=torch.load(args.new_parameters,map_location='cpu',weights_only=False)
    tape=torch.load(args.rollout,map_location='cpu',weights_only=False)
    original=graph_module.load_connectome
    graph_module.load_connectome=lambda path,**kw:original(path,**dict(kw,verify=False))
    try:model=from_story_checkpoint(args.connectome,new,device='cuda:0').eval()
    finally:graph_module.load_connectome=original
    optimizer=optimizer_for(model)
    names={id(parameter):name for name,parameter in model.named_parameters()}
    groups={group['name']:[(names[id(parameter)],parameter) for parameter in group['params']]
            for group in optimizer.param_groups}
    old_state=old['state_dict'];new_state=new['state_dict'];old_costs=dict(old['costs'])
    del old,new,optimizer
    gc.collect()
    scenes=[(336,1),(672,2)]
    def measure():
        details=[]
        with torch.no_grad():
            for t,world in scenes:
                features=tape['features'][t:t+16,world:world+1].to(model.device)
                actions=tape['actions'][t:t+16,world:world+1].to(model.device)
                starts=tape['episode_starts'][t:t+16,world:world+1].to(model.device)
                state=SpikingState.pack([tape['block_states'][t].row(world)]).to(model.device)
                out=sequence(model,features,starts,state,actions,checkpointed=False)
                logits=story_behavior_logits(out['logits'],tape['metadata']['movement_sampling_temperature'],
                                              tape['metadata']['bomb_sampling_temperature'])
                recorded=tape['recorded_behavior_policy_logits'][t:t+16,world:world+1].to(model.device)
                total,bomb,movement=action_kl_parts(recorded,logits)
                details.append(dict(time_index=t,world=world,total=float(total.mean()),
                    bomb=float(bomb.mean()),weighted_movement=float(movement.mean())))
        return details
    result={'all_new_parameters':measure()}
    if not args.only_new:
        for group in args.groups:
            with torch.no_grad():
                for name,param in groups[group]:param.copy_(old_state[name].to(param.device))
            result['restore_'+group]=measure()
            with torch.no_grad():
                for name,param in groups[group]:param.copy_(new_state[name].to(param.device))
    if args.all_old or args.only_new:
        with torch.no_grad():
            for parameters in groups.values():
                for name,param in parameters:param.copy_(old_state[name].to(param.device))
        result['all_old_parameters']=measure()
    if args.only_new:
        for group in args.groups:
            with torch.no_grad():
                for name,param in groups[group]:param.copy_(new_state[name].to(param.device))
            result['only_new_'+group]=measure()
            with torch.no_grad():
                for name,param in groups[group]:param.copy_(old_state[name].to(param.device))
    output=dict(results=result,old_costs=old_costs,
        new_parameters=str(args.new_parameters),rollout=str(args.rollout),
        scope=('Two saved first-stage student sequences. A single parameter group is swapped '
               'between the old and new models; effects can interact and are not additive. '
               'This does not measure later-stage behavior or native-game outcomes.'))
    write_json(args.output,output)
    print(output)


if __name__=='__main__':main()
