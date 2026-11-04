"""Bounded actor-readout fitting after whole-brain physical-risk learning.

The body first learns on raw maps/objects, then its FINAL parameters replay
the complete recorded histories to build a DN cache. This phase never mixes
old-policy DN with a new body and never supplies raw observations to a head.
"""
from pathlib import Path
import numpy as np
import torch


def build_cache(model,sources,splits,output,*,check_stop=lambda:None,progress=lambda row:None,width=16,length=16):
    from .chronological_avoidance_data import ChronologicalBatches
    from .continuous_full_sensory_ppo import _forward_full_sequence
    from .continuous_object_avoidance_ppo import avoidance_PPO_config
    from .continuous_fullhistory_head_calibration import DN_source_identity
    from .thrl_durable_records import write_json
    output=Path(output);output.mkdir(exist_ok=False)
    model.eval().requires_grad_(False);identity=DN_source_identity(model)
    device=next(model.parameters()).device;versions=tuple(p._version for p in model.parameters())
    result={}
    with torch.no_grad():
        for role in ('train','heldout'):
            pieces=[]
            for kind,source in sources.items():
                stream=ChronologicalBatches(source,splits[kind][role],shuffle=False,width=width,length=length)
                state=model.initial_state(width)
                while not stream.epochs_completed:
                    check_stop();raw,labels,meta=stream.next()
                    data={key:torch.from_numpy(value).to(device) for key,value in raw.items()}
                    forward=_forward_full_sequence(model,data,state,cfg=avoidance_PPO_config(),
                        use_checkpoint=False,counts={},return_DN=True)
                    state=forward['following_state'];valid=raw['valid_mask']
                    part=dict(DN=forward['following_DN'][data['valid_mask']].cpu(),
                        features=torch.from_numpy(raw['features'][valid]),
                        actions=torch.from_numpy(raw['actions'][valid]),
                        episode_starts=torch.from_numpy(raw['episode_starts'][valid]),
                        teacher_source=torch.full((meta['real_rows'],),kind=='teacher',dtype=torch.bool))
                    part.update({key:torch.from_numpy(value[valid]) for key,value in labels.items()})
                    pieces.append(part)
                    if len(pieces)%32==0:progress(dict(role=role,source=kind,windows=len(pieces),rows=sum(len(x['DN']) for x in pieces)))
                    del raw,labels,data,forward
            result[role]={key:torch.cat([part[key] for part in pieces]) for key in pieces[0]}
            torch.save(result[role],output/f'{role}.pt')
    if tuple(p._version for p in model.parameters())!=versions or DN_source_identity(model)!=identity:
        raise RuntimeError('Final-body history replay changed its parameters')
    write_json(output/'COMMITTED.json',dict(schema='current-finalbody-fullhistory-DN-readout-cache-v1',
        body_identity=identity,rows={role:len(data['DN']) for role,data in result.items()},
        raw_source_identities={name:source.identity for name,source in sources.items()},splits=splits,
        cached_DN_is_from_this_final_body=True,history_started_at_actual_openings=True,
        native_evaluation_observations_in_training=False,optimizer_updates=0))
    return result


def fit_actor(model,optimizer,cache,*,updates,seed=1,check_stop=lambda:None,
              progress=lambda row:None,on_checkpoint=lambda step:None):
    from .continuous_object_avoidance import GUIDANCE_CONFIG,no_bomb_policy_logits
    from .fly_gru_aligned_guidance import gru_aligned_guidance_loss
    from .whole_brain_ppo import clip_gradients_fp64
    model.eval().requires_grad_(False);model.actor.train().requires_grad_(True)
    actor_ids={id(p) for p in model.actor.parameters()}
    fixed_versions={name:p._version for name,p in model.named_parameters() if id(p) not in actor_ids}
    device=next(model.parameters()).device;rng=np.random.default_rng(seed)
    datasets={role:{key:value.to(device) for key,value in data.items()} for role,data in cache.items()}
    train=datasets['train'];normal=train['unprotected_mask'];trusted=normal&train['trusted'][:,0]
    mixed=trusted&(train['margins'][:,0]<=0).any(-1)&(train['margins'][:,0]>1).any(-1)
    hit=normal&(train['outcomes']==1)
    informative=trusted|(train['rule_weights']>0)|hit
    pools=[torch.nonzero(mask&informative).flatten().cpu().numpy() for mask in (
        train['teacher_source'],mixed,hit,informative)]
    if not len(pools[-1]):raise ValueError('Informative source directions required for actor fitting')

    def objective(data):
        raw=model.actor(data['DN'])
        with torch.no_grad():cost=model.risk_head(data['DN'])
        combined=model.compose(raw,cost,data['features'],check_finite=False)
        logits=no_bomb_policy_logits(combined,data['features'])
        names=('margins','trusted','unprotected_mask','outcomes','rule_probabilities','rule_weights','deathbomb_mask')
        task={key:data[key].unsqueeze(0) for key in names}
        guide=gru_aligned_guidance_loss(logits.unsqueeze(0),**task,actions=data['actions'].unsqueeze(0),
            valid_mask=torch.ones(1,len(raw),dtype=torch.bool,device=device),config=GUIDANCE_CONFIG)
        return guide['total_loss'],guide,logits

    @torch.no_grad()
    def assess():
        held=datasets['heldout'];sums=dict(loss=0.,known_mixed=0,unsafe_modes=0,rows=0,
            dangerous_queries=0,dangerous_detected=0,false_safe_queries=0,boundary_absolute_error=0.,boundary_queries=0)
        for first in range(0,len(held['DN']),1024):
            check_stop();data={key:value[first:first+1024] for key,value in held.items()}
            loss,_,logits=objective(data);n=len(logits);margin=data['margins'][:,0]
            good=data['trusted'][:,0]&data['unprotected_mask']&(data['features'][:,21]>.5)&(data['features'][:,20]<=0)&(data['features'][:,5]<=0)
            mixed=good&(margin<=0).any(-1)&(margin>1).any(-1)
            chosen=margin.gather(-1,logits[:,:18].argmax(-1)[:,None]).squeeze(-1)
            sums['loss']+=float(loss)*n;sums['rows']+=n;sums['known_mixed']+=int(mixed.sum())
            sums['unsafe_modes']+=int((mixed&(chosen<=0)).sum())
            prediction=1.-.5*model.risk_head.layers(data['DN'])
            bad=(margin<=0)&good[:,None];boundary=(margin.abs()<8)&good[:,None]
            sums['dangerous_queries']+=int(bad.sum());sums['dangerous_detected']+=int((bad&(prediction<=0)).sum())
            sums['false_safe_queries']+=int((bad&(prediction>1)).sum())
            sums['boundary_absolute_error']+=float((prediction-margin).abs()[boundary].sum())
            sums['boundary_queries']+=int(boundary.sum())
        sums['loss']/=max(1,sums['rows'])
        sums['danger_recall']=sums['dangerous_detected']/max(1,sums['dangerous_queries'])
        sums['false_safe_fraction']=sums['false_safe_queries']/max(1,sums['dangerous_queries'])
        sums['boundary_MAE']=sums['boundary_absolute_error']/max(1,sums['boundary_queries'])
        return sums

    held_before=assess()
    for step in range(updates):
        check_stop();ids=[]
        for pool in pools:ids.extend(rng.choice(pool if len(pool) else pools[-1],64,replace=True).tolist())
        index=torch.as_tensor(ids,device=device);data={key:value.index_select(0,index) for key,value in train.items()}
        optimizer.zero_grad(set_to_none=True);loss,guide,_=objective(data)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite actor-only original guidance')
        loss.backward()
        if any(p.grad is not None for p in model.parameters() if id(p) not in actor_ids):
            raise RuntimeError('Bounded actor fitting reached the fixed learned body or risk head')
        norm=clip_gradients_fp64(list(model.actor.parameters()),5.);optimizer.step()
        progress(dict(step=step+1,loss=float(loss.detach()),gradient_norm=norm,
            safe_loss=float(guide['safe_loss'].detach()),rule_loss=float(guide['rule_loss'].detach()),
            future_loss=float(guide['future_loss'].detach()),hit_loss=float(guide['collision_loss'].detach())))
        if (step+1)%250==0:on_checkpoint(step+1)
    if any(p._version!=fixed_versions[name] for name,p in model.named_parameters() if id(p) not in actor_ids):
        raise RuntimeError('The final learned body/risk/three critics changed during actor-only fitting')
    result=dict(updates=updates,held_before=held_before,held_after=assess(),
        body_and_risk_fixed_for_bounded_actor_fitting=True,three_critic_heads_preserved=True,
        raw_feature_risk_or_teacher_logits_bypass=False)
    model.train().requires_grad_(True)
    return result
