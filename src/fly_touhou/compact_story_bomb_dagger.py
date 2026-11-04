"""Teacher-labelled Bomb correction on fresh student deployment trajectories.

The existing full-brain state keeps running; only the temporary CPU Bomb
readout fit is separate. Following PPO resumes with fresh stochastic samples.
"""
import copy
import torch
from .compact_story_bomb_calibration import loss,metrics

FIELDS=('DN','urgency','movement_logmass','teacher','weights','deathbomb','world','Bomb_logit_offset')


@torch.no_grad()
def training_cache(model,rollout,*,normal_limit=512):
    if not rollout['metadata'].get('supervision_only') or rollout['metadata'].get('teacher_actions')!=0:
        raise ValueError('Fresh teacher-labelled student deployment trajectories required')
    f=rollout['features'];T,N=f.shape[:2]
    legal=(f[...,4]>0)&((f[...,14]>.5)|(f[...,17]>.5))
    db=rollout['deathbomb_mask'];teacher=rollout['rule_probabilities'][...,18];weights=rollout['rule_weights']
    ordinary=(legal&~db&(weights>0)&(teacher<=1e-6)&(~rollout['unprotected_mask']|
        (rollout['trusted'][...,0]&(rollout['margins'][...,0,:]>1).any(-1))))
    positive=legal&(db|((weights>0)&(teacher>1e-6)))
    observed=rollout['recorded_pooled_DN_mask']
    probability=rollout['recorded_policy_logits'].softmax(-1)[...,18]
    world=torch.arange(N).repeat(T)
    negative=(ordinary&observed).flatten().nonzero().flatten()
    hard=[]
    for w in range(N):
        idx=negative[world[negative]==w]
        if len(idx):hard.append(idx[probability.flatten()[idx].argsort(descending=True)[:8]])
    hard=torch.cat(hard) if hard else torch.zeros(0,dtype=torch.long)
    rest=negative[~torch.isin(negative,hard)]
    generator=torch.Generator().manual_seed(20261031)
    rest=rest[torch.randperm(len(rest),generator=generator)[:max(0,normal_limit-len(hard))]]
    selected=torch.cat((hard,rest,(positive&observed).flatten().nonzero().flatten())).unique()
    if not len(selected):raise ValueError('No observed stocked Bomb teaching states in this student rollout')
    dn=rollout['recorded_pooled_DN'].reshape(T*N,-1)[selected].clone()
    risk=(8*rollout['recorded_neural_horizon_scores'][...,0]-1).sigmoid().min(-1).values
    urgency=((risk-.2)/.05).sigmoid()
    if model.phase_valid_control:
        urgency*=((f[...,14]>.5)&(f[...,5]<=0)&(f[...,13]<=0))|(f[...,17]>.5)
    urgency=urgency.flatten()[selected].clone()
    head=copy.deepcopy(model.bomb).cpu()
    raw=(head(dn)[:,0]+float(model.bomb_urgency_gain.detach().cpu())*urgency).clamp(-80,80)
    logits=rollout['recorded_policy_logits'].reshape(T*N,19)[selected]
    return dict(DN=dn,urgency=urgency,movement_logmass=logits[:,:18].logsumexp(-1),
        teacher=teacher.flatten()[selected].clone(),weights=weights.flatten()[selected].clone(),
        deathbomb=db.flatten()[selected].clone(),world=world[selected],
        Bomb_logit_offset=logits[:,18]-raw,
        metadata=dict(training_only=True,development_data_used=False,source='fresh deterministic student training rollout',
            action_selection=rollout['metadata']['action_selection'],fresh_student_rows=T*N,
            stocked_teacher_rows=len(selected),normal_hard_rows=len(hard),teacher_actions=0,
            observed_stocked_deathbomb_rows=int((legal&db&observed).sum()),
            observed_normal_teacher_NoBomb_rows=int((ordinary&observed).sum()),
            observed_normal_teacher_Bomb_rows=int((positive&~db&observed).sum()),
            observed_normal_teacher_Bomb_with_safe4_rows=int((positive&~db&observed&rollout['unprotected_mask']
                &rollout['trusted'][...,0]&(rollout['margins'][...,0,:]>1).any(-1)).sum())))


def verify_anchor_model(model,anchor,source_payload):
    if anchor['metadata'].get('development_data_used') is not False or not anchor['metadata'].get('checkpoint'):
        raise ValueError('Bound original training anchor cache required')
    original=source_payload['state_dict']
    for key,value in model.state_dict().items():
        if key.startswith('bomb.') or key=='bomb_urgency_gain':continue
        if key not in original or not torch.equal(value.detach().cpu(),original[key]):
            raise ValueError('Anchor DN belongs to different brain, sensory, risk or movement weights: '+key)


def refit(model,rollout,anchor,*,updates=128,progress=lambda row:None,guard=lambda:None):
    fresh=training_cache(model,rollout)
    for cache in (fresh,anchor):
        if cache['metadata'].get('development_data_used') is not False:raise ValueError('Training data only')
    data={k:torch.cat((fresh[k],anchor[k] if k in anchor else torch.zeros(len(anchor['DN'])))) for k in FIELDS}
    # Preserve a split by original game-world identity across both sources.
    held=data['world']%4==2
    train={k:v[~held] for k,v in data.items()};validation={k:v[held] for k,v in data.items()}
    head=copy.deepcopy(model.bomb).cpu();gain=torch.nn.Parameter(model.bomb_urgency_gain.detach().cpu().clone())
    def odds(d):
        return (head(d['DN'])[:,0]+gain*d['urgency']).clamp(-80,80)+d['Bomb_logit_offset']-d['movement_logmass']
    @torch.no_grad()
    def measure(d):return metrics(odds(d),d['teacher'],d['weights'],d['deathbomb'])
    report=dict(training=measure(train),validation=measure(validation),nodes=[],fresh=fresh['metadata'],
        brain_changed=False,PPO_updates=0,applied=False,paid_readout_updates=updates)
    best=None;best_loss=report['validation']['balanced_BCE']
    optimizer=torch.optim.Adam([*head.parameters(),gain],lr=3e-4)
    for step in range(1,updates+1):
        guard();value=loss(odds(train),train['teacher'],train['weights'],train['deathbomb'])
        optimizer.zero_grad();value.backward();torch.nn.utils.clip_grad_norm_([*head.parameters(),gain],5.);optimizer.step()
        if step%16==0 or step==updates:
            v=measure(validation);row=dict(update=step,loss=float(value.detach()),validation=v)
            report['nodes'].append(row);progress(row)
            if (v['balanced_BCE']<best_loss
                    and v['normal_family_Bomb_choices']<=report['validation']['normal_family_Bomb_choices']
                    and v['deathbomb_family_choices']>=report['validation']['deathbomb_family_choices']
                    and v['normal_mean_Bomb_probability']<report['validation']['normal_mean_Bomb_probability']):
                best_loss=v['balanced_BCE'];best=(copy.deepcopy(head.state_dict()),gain.detach().clone())
                report['selected_update']=step;report['selected_validation']=v
    if best is not None:
        model.bomb.load_state_dict(best[0])
        with torch.no_grad():model.bomb_urgency_gain.copy_(best[1].to(model.device))
        report['applied']=True
    return report,fresh


def reset_live_bomb_moments(optimizer):
    for group in optimizer.param_groups:
        if group.get('name')=='Bomb':
            for parameter in group['params']:
                if hasattr(optimizer,'reset_parameter_state'):
                    optimizer.reset_parameter_state(parameter)
                else:
                    optimizer.state.pop(parameter,None)
