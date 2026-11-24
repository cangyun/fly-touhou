"""12/24f risk of the student's actual continuation, separate from route existence.

Four-frame geometry and witnessed teacher routes keep their meaning. Longer
heads receive observed contact/no-contact labels on the executed first action,
and mandatory first-command contacts on other actions. A found alternative
route is not a negative collision label for the student's different future.
No saved state, reset, padding, loading gap or optimizer boundary is bridged.
"""
import torch

REVISION = 'student-native-continuation-risk-v2'


@torch.no_grad()
def classification_weights(batch, weights):
    """Reserve most long-head BCE for real policy futures, balanced by outcome.

Thousands of alternative immediate-contact labels must not drown out the
few actual early approaches to a collision. Preserve each horizon's existing
total BCE budget and leave the4f budget unchanged.
    """
    result=weights.clone();source=batch['risk_continuation_source']
    valid=batch['risk_continuation_valid'] & batch['unprotected_mask'][...,None,None]
    positive=batch['risk_continuation_margins']<=0
    for h,budget in ((1,.3),(2,.1)):
        native=valid[...,h,:] & (source[...,h,:]==5)
        immediate=valid[...,h,:] & (source[...,h,:]==2)
        native_share=.75 if bool(native.any()) and bool(immediate.any()) else 1.
        result[...,h,:]=0
        if bool(native.any()):
            hit=native & positive[...,h,:];clear=native & ~positive[...,h,:]
            classes=int(hit.any())+int(clear.any())
            result[...,h,:]+=budget*native_share*(hit/hit.sum().clamp_min(1)+clear/clear.sum().clamp_min(1))/classes
        if bool(immediate.any()):
            share=1-native_share if bool(native.any()) else 1.
            result[...,h,:]+=budget*share*immediate/immediate.sum()
    return result


@torch.no_grad()
def observed_futures(rollout, horizons=(12, 24), *, same_stage=False):
    actions=rollout['actions'];T,N=actions.shape
    valid=rollout.get('valid_mask',torch.ones_like(actions,dtype=torch.bool))
    duration=rollout['native_action_frames']
    eligible=(valid & rollout['unprotected_mask'] & (actions>=0) & (actions<18)
              & (rollout['outcomes']>=0) & (duration>0) & (duration<=4))
    episodes=rollout['recorded_native_episode'];frames=rollout['recorded_episode_frames']
    known=torch.zeros((T,N,len(horizons)),dtype=torch.bool,device=actions.device)
    hits=torch.zeros_like(known)
    for h,horizon in enumerate(horizons):
        alive=eligible.clone();elapsed=torch.zeros_like(duration)
        # Native duration, not a count of policy rows: loading gaps are censored.
        for offset in range(min(T,horizon)):
            length=T-offset
            good=(alive[:length] & eligible[offset:] & (episodes[:length]==episodes[offset:])
                  & (frames[offset:]==frames[:length]+elapsed[:length]))
            if same_stage:
                stage=rollout['recorded_native_stage']
                good=good & (stage[:length]==stage[offset:])
            elapsed[:length]+=torch.where(good,duration[offset:],0)
            native_hit=good & (rollout['outcomes'][offset:]==1)
            hit=native_hit & (elapsed[:length]<=horizon)
            covered=good & (elapsed[:length]>=horizon) & ~native_hit
            known[:length,:,h] |= hit | covered
            hits[:length,:,h] |= hit
            alive[:length]=good & ~native_hit & ~covered & ~rollout['terminated'][offset:]
            if offset:alive[length:]=False
    return known,hits


@torch.no_grad()
def prepare(rollout):
    """Build fresh full-rollout targets before recurrent minibatch slicing.

The inherited `margins` slots at12/24f hold binary sentinels -1(hit)/2(clear),
NOT clearance measurements. Their regression weight is zero; only BCE uses
them. Unexecuted actions have no invented policy-continuation outcomes.
"""
    known,hits=observed_futures(rollout)
    margins=rollout['risk_continuation_margins'].clone()
    valid=rollout['risk_continuation_valid'].clone()
    source=rollout['risk_continuation_source'].clone()
    witnesses=(source==3)&valid
    immediate=(source[...,1:,:]==2)&valid[...,1:,:]
    valid[...,1:,:]=immediate
    source[...,1:,:]=torch.where(immediate,2,0)
    margins[...,1:,:]=torch.where(immediate,-1.,2.)
    for h in range(2):
        ids=torch.nonzero(known[...,h],as_tuple=True)
        chosen=rollout['actions'][ids]
        # Original-game facts override a contradicted geometric label for the
        # selected action only. Alternative witnesses are still teacher data.
        margins[ids[0],ids[1],h+1,chosen]=torch.where(hits[...,h][ids],-1.,2.)
        valid[ids[0],ids[1],h+1,chosen]=True
        source[ids[0],ids[1],h+1,chosen]=5
    return dict(risk_continuation_margins=margins,risk_continuation_valid=valid,
                risk_continuation_source=source,risk_route_witness_valid=witnesses,
                risk_long_classification_only=torch.ones_like(rollout['actions'],dtype=torch.bool),
                risk_native_future_known=known,risk_native_future_contact=hits)
