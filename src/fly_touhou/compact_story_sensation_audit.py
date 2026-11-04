"""Controlled target-position sensitivity using real archived brain states.

No optimization and no game actions. This checks whether a task signal can
affect VP drive,DN activity and movement readout;it is not a game score.
"""
import torch
from .compact_story_model import story_candidates
from .spiking_engineered_control import SpikingState


@torch.no_grad()
def audit(model,tape,*,scenes=4,capture=None):
    f=tape['features'];times=torch.arange(0,len(f),16)
    valid=f[times,:,14]>.5
    boss=valid&(f[times,:,113]>.5)&(f[times,:,117]>.5)
    target=valid&(f[times,:,113]>.5)
    eligible=boss if bool(boss.any()) else target if bool(target.any()) else valid
    choices=eligible.nonzero().tolist()
    if not choices:raise ValueError('No native controllable observation in the archived rollout')
    indices=torch.linspace(0,len(choices)-1,min(scenes,len(choices))).round().long().tolist()
    chosen=[(int(times[choices[i][0]]),choices[i][1]) for i in indices]
    features=[];states=[]
    for t,w in chosen:
        for kind in ('actual','left','right'):
            x=f[t,w].clone()
            if kind!='actual':x[114]=-.15 if kind=='left' else .15
            features.append(x);states.append(tape['block_states'][t].row(w).to(model.device))
    features=torch.stack(features).to(model.device)
    versions=[p._version for p in model.parameters()]
    out,_,_=model.step(features,SpikingState.pack(states),return_all_DN=True)
    ports=model.core.input_encoder(story_candidates(features).reshape(-1,47)).reshape(len(features),18,-1)
    from .thrl_movement_teacher import MovementSafetyTeacher
    velocity=torch.as_tensor(MovementSafetyTeacher().velocities,device=model.device)
    left=velocity[:,0]<0;right=velocity[:,0]>0
    route=out['route_scores']/model.temperature
    preference=route[:,right].logsumexp(-1)-route[:,left].logsumexp(-1)
    logits=out['logits'];policy=logits[:,:18]
    policy_preference=policy[:,right].logsumexp(-1)-policy[:,left].logsumexp(-1)
    result=[]
    for j,(t,w) in enumerate(chosen):
        k=3*j;dn=out['candidate_DN'][k:k+3];drive=ports[k:k+3]
        recorded=tape['recorded_policy_logits'][t,w].to(model.device)
        finite=torch.isfinite(recorded)&torch.isfinite(logits[k])
        result.append(dict(time=t,world=w,stage=float(f[t,w,111]*6),actual_target_present=bool(f[t,w,113]>.5),
            actual_boss=bool(f[t,w,117]>.5),actual_target_dx=float(f[t,w,114]),target_dy=float(f[t,w,115]),
            VP_left_right_mean_abs_delta=float((drive[1]-drive[2]).abs().mean()),
            VP_left_right_changed_fraction=float(((drive[1]-drive[2]).abs()>1e-6).float().mean()),
            DN_left_right_mean_abs_delta=float((dn[1]-dn[2]).abs().mean()),
            DN_left_right_relative_L2=float((dn[1]-dn[2]).norm()/dn[0].norm().clamp_min(1e-8)),
            route_right_minus_left_preference_for_actual_left_right=preference[k:k+3].cpu().tolist(),
            final_right_minus_left_preference_for_actual_left_right=policy_preference[k:k+3].cpu().tolist(),
            actor_target_response_in_expected_direction=bool(policy_preference[k+2]>policy_preference[k+1]),
            actual_logits_replay_max_error=float((recorded[finite]-logits[k][finite]).abs().max()) if bool(finite.any()) else None,
            actual_logits_legal_support_equal=bool(torch.equal(torch.isfinite(recorded),torch.isfinite(logits[k])))))
    unchanged=versions==[p._version for p in model.parameters()]
    if not unchanged:raise RuntimeError('Read-only sensation audit changed a parameter')
    if capture is not None:
        capture(dict(features=features.cpu(),candidate_DN=out['candidate_DN'].cpu(),logits=logits.cpu(),
            route_scores=out['route_scores'].cpu(),source_scenes=chosen,variants=['actual','left','right']))
    return dict(schema='compact-SNN-controlled-target-DN-sensitivity-v1',scenes=result,parameters_updated=False,
        native_actions=0,optimizer_updates=0,variants=['recorded scene','same scene with target dx=-.15','same scene with target dx=.15'],
        interpretation='Controlled task-signal sensitivity,not a clear rate or proof of a physically safe counterfactual route')
