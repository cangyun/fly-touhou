"""Test movement-head learning on cached, unchanged whole-brain DN activity.

Uses real stage3/4 teacher windows, the production movement loss and restored
movement Adam state. Only a separate copy of the movement readout is updated.
This is a representation/optimization probe, not PPO or a native-game result.
"""
from pathlib import Path
import argparse
import copy
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training','manifest','connectome','cache','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--updates',type=int,default=64)
    p.add_argument('--train-windows-per-stage',type=int,default=4)
    p.add_argument('--heldout-windows-per-stage',type=int,default=2)
    p.add_argument('--cache-batch',type=int,default=7)
    a=p.parse_args()
    os.environ.update(FLY_TOUHOU_DETERMINISTIC_SPMM='1',
        FLY_TOUHOU_SPMM_CACHE=str(a.cache/'spmm-kernel-cache'),
        FLY_TOUHOU_EDGE_GRAD_CACHE=str(a.cache/'kernel-cache'),
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    from fly_touhou.runtime_cache_paths import configure_generated_caches
    configure_generated_caches()
    import gc
    import numpy as np
    import torch
    from fly_touhou import connectome as graph_module
    from fly_touhou.compact_story_model import from_story_checkpoint
    from fly_touhou.compact_policy_adam import PolicyMasterAdam
    from fly_touhou.compact_rule_teacher_replay import movement_distribution_loss,mask_rule_guidance_on_contact
    from fly_touhou.spiking_engineered_control import SpikingState
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    from fly_touhou.thrl_durable_records import write_json
    torch.set_num_threads(2)
    with _shared_binary(a.training) as stream:
        point=torch.load(stream,map_location='cpu',weights_only=False)
    payload=point['model_payload'];costs=payload['costs']
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,a.output.with_suffix('.source-parameters.pt'))
    group=next(g for g in point['optimizer']['param_groups'] if g['name']=='movement')
    ids=group['params']
    saved=dict(param_groups=[copy.deepcopy(group)],
        state={i:copy.deepcopy(point['optimizer']['state'][i]) for i in ids
               if i in point['optimizer']['state']},
        policy_masters={i:point['optimizer'].get('policy_masters',{})[i].clone() for i in ids
                        if i in point['optimizer'].get('policy_masters',{})})
    del point
    manifest=json.loads(a.manifest.read_text(encoding='utf-8'));burn=manifest['burn_in']
    fields=('features','actions','episode_starts','targets','teacher_guidance_weight',
            'teacher_behavior_valid','unprotected_mask','outcome','native_action_frames','margins','trusted')
    windows=[];arrays=[];rng=np.random.default_rng(87241)
    for split,count in (('train',a.train_windows_per_stage),('heldout',a.heldout_windows_per_stage)):
        for stage in (3,4):
            candidates=[w for w in manifest['windows'] if w['split']==split and w['stage']==stage]
            chosen=[]
            for index in rng.permutation(len(candidates)):
                row=candidates[int(index)]
                with np.load(a.manifest.parent/row['file'],allow_pickle=False) as z:
                    data={k:z[k].copy() for k in fields}
                f=data['features'][burn:];act=data['actions'][burn:]
                usable=((data['teacher_guidance_weight'][burn:]>0)
                    |data['teacher_behavior_valid'][burn:])&(act<18)&((f[:,14]>.5)|(f[:,17]>.5))
                if not usable.any():continue
                chosen.append(row);windows.append(row);arrays.append(data)
                if len(chosen)==count:break
            if len(chosen)!=count:raise ValueError('Not enough usable distinct teacher windows')
    data={k:torch.from_numpy(np.stack([d[k] for d in arrays],axis=1)) for k in fields}
    load=graph_module.load_connectome
    graph_module.load_connectome=lambda path,**kw:load(path,**dict(kw,verify=False))
    try:model=from_story_checkpoint(a.connectome,payload,device='cuda:0').eval()
    finally:graph_module.load_connectome=load
    del payload,arrays;gc.collect()
    model.requires_grad_(False)
    cached=[]
    with torch.no_grad():
        for first in range(0,len(windows),a.cache_batch):
            count=min(a.cache_batch,len(windows)-first);state=model.initial_state(count);rows=[]
            for t in range(len(data['features'])):
                starts=data['episode_starts'][t,first:first+count].to(model.device)
                state=SpikingState(*(x*~starts[:,None].bool() for x in state.tensors()))
                result,state,_=model.step(data['features'][t,first:first+count].to(model.device),state,
                    actions=data['actions'][t,first:first+count].to(model.device),return_all_DN=t>=burn)
                if t>=burn:rows.append(result['candidate_DN'].cpu())
            cached.append(torch.stack(rows))
            write_json(a.output,dict(status='caching_teacher_DN',windows_done=first+count,total_windows=len(windows),
                live_training_modified=False,native_games_launched=0))
    DN=torch.cat(cached,dim=1);del cached,result,state
    data={k:v[burn:] for k,v in data.items()}
    torch.save(dict(DN=DN,data=data,windows=windows,movement_optimizer=saved,source_costs=costs),
        a.output.with_suffix('.DN.pt'))
    model.cpu();gc.collect();torch.cuda.empty_cache()
    baseline={k:v.clone() for k,v in model.route.state_dict().items()}
    model.route.requires_grad_(True)
    train=[i for i,w in enumerate(windows) if w['split']=='train']
    heldout=[i for i,w in enumerate(windows) if w['split']=='heldout']

    def forward(indices):
        f=data['features'][:,indices];shape=f.shape[:2]
        out=model.heads(DN[:,indices].reshape(-1,18,DN.shape[-1]),f.reshape(-1,f.shape[-1]))
        logits=out['logits'].reshape(*shape,19)
        d={k:v[:,indices] for k,v in data.items()}
        control=(f[...,14]>.5)|(f[...,17]>.5)
        weights,_=mask_rule_guidance_on_contact(d['teacher_guidance_weight'],d['actions'],d['outcome'],
            d['unprotected_mask'],d['native_action_frames'])
        movement,eligible=movement_distribution_loss(logits,d['targets'],weights,d['actions'],control)
        fallback=d['teacher_behavior_valid']&(d['teacher_guidance_weight']<=0)
        target=torch.nn.functional.one_hot(d['actions'].clamp(0,18).long(),19).to(logits.dtype)
        behavior,behavior_rows=movement_distribution_loss(logits,target,fallback.to(logits.dtype),d['actions'],control)
        loss=.125*.1*(movement+.25*behavior)
        with torch.no_grad():
            margin=d['margins'][...,0,:]
            valid=d['trusted'][...,0]&d['unprotected_mask']
            danger=(margin<=0)&valid[...,None];safe=(margin>1)&valid[...,None]
            mixed=danger.any(-1)&safe.any(-1)
            chosen=logits[...,:18].argmax(-1)
            bad=danger.gather(-1,chosen[...,None])[...,0]&mixed
            metrics=dict(objective=float(loss.detach()),teacher_movement_CE=float(movement.detach()),
                teacher_rows=int(eligible.sum()),behavior_CE=float(behavior.detach()),behavior_rows=int(behavior_rows.sum()),
                modeled_mixed_rows=int(mixed.sum()),modeled_danger_choices=int(bad.sum()),
                mean_Bomb_probability=float(logits.softmax(-1)[...,18].mean()))
        return loss,metrics

    rates=[group['lr'],1e-5,3e-5,1e-4];curves=[]
    for rate in rates:
        model.route.load_state_dict(baseline)
        optimizer=PolicyMasterAdam([dict(name='movement',params=model.route.parameters(),lr=rate)])
        optimizer.load_state_dict(copy.deepcopy(saved));optimizer.param_groups[0]['lr']=rate
        batch_rng=np.random.default_rng(91447)
        curve=dict(rate=rate,records=[])
        for step in range(a.updates+1):
            if step in (0,8,32,a.updates):
                with torch.no_grad():
                    measured=dict(step=step,train=forward(train)[1],heldout=forward(heldout)[1])
                    measured['heldout_by_stage']={str(s):forward([i for i in heldout if windows[i]['stage']==s])[1]
                                                   for s in (3,4)}
                curve['records'].append(measured)
            if step==a.updates:break
            optimizer.zero_grad(set_to_none=True)
            batch=[]
            for stage in (3,4):
                available=[i for i in train if windows[i]['stage']==stage]
                batch.extend(batch_rng.choice(available,size=min(4,len(available)),replace=False).tolist())
            loss,_=forward(batch);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.route.parameters(),5.)
            optimizer.step()
        curves.append(curve)
        write_json(a.output,dict(status='head_probe_in_progress',completed_rates=len(curves),curves=curves,
            live_training_modified=False,native_games_launched=0))
    report=dict(status='completed',source_training=str(a.training),source_costs=costs,
        manifest=str(a.manifest),windows=windows,burn_in=burn,updates_per_rate=a.updates,curves=curves,
        source_parameters_snapshot=str(a.output.with_suffix('.source-parameters.pt')),
        cached_DN=str(a.output.with_suffix('.DN.pt')),
        restored_movement_Adam=True,whole_brain_DN_unchanged_between_rates=True,
        live_training_modified=False,native_games_launched=0,production_checkpoint_written=False,
        scope=('Small stage3/4 readout-learning probe on fixed DN from current full SNN. Teacher games are separated '
               'between training and heldout. Zero-state32-step burn-in approximates history. Only the movement '
               'readout learns the existing weighted movement/behavior supervision;no PPO or original-game '
               'outcomes are inferred. Head-only gradient clipping differs from joint training.'))
    write_json(a.output,report)
    print(json.dumps(dict(status='completed',rates=[dict(rate=c['rate'],before=c['records'][0]['heldout'],
        after=c['records'][-1]['heldout']) for c in curves]),ensure_ascii=False))


if __name__=='__main__':main()
