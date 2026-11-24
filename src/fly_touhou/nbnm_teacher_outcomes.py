"""Withdraw teacher claims contradicted by the actual native continuation.

This is sealed-tape supervision, never an action override. A student that
ignored the teacher does not invalidate the teacher's unexecuted route.
"""
import torch

LEGACY_REVISION = 'native-outcome-qualified-teacher-v1'
REVISION = 'native-outcome-qualified-teacher-v2'
SUPPORTED_REVISIONS = (LEGACY_REVISION, REVISION)


@torch.no_grad()
def prepare(rollout, horizon=96, *, revision=REVISION):
    if revision not in SUPPORTED_REVISIONS:
        raise ValueError('Unknown teacher outcome revision')
    from .nbnm_policy_continuation_risk import observed_futures
    q=rollout['rule_probabilities'].clone();w=rollout['rule_weights'].clone()
    action=rollout['actions'];valid=rollout.get('valid_mask',torch.ones_like(action,dtype=torch.bool))
    guided=(w>0)&valid
    followed=guided&(action==q[...,:18].argmax(-1))
    # Stop the evidence chain as soon as the student departs from the teacher.
    tape=dict(rollout,valid_mask=valid&followed)
    _,failed=observed_futures(tape,horizons=(horizon,))
    contradicted=failed[...,0]
    # An abstaining terminal teacher used to hide the immediately preceding
    # followed prefix. It is NOT evidence that the student ignored a viable
    # instruction: the teacher itself found no positive-clearance first step.
    # Censor that prefix's positive teaching claim. This is weaker than an
    # observed failed teacher continuation: conservative envelopes can reject
    # real exits, so never turn this finding into counterfactual risk labels.
    censored=torch.zeros_like(contradicted)
    abstaining_dead_end=torch.zeros_like(contradicted)
    if revision == REVISION:
        first=rollout['margins'][...,0,:]
        abstaining_dead_end=(valid&rollout['unprotected_mask']&(rollout['outcomes']==1)
            &(action>=0)&(action<18)&(rollout['native_action_frames']>0)
            &(rollout['native_action_frames']<=4)&(w<=0)&rollout['trusted'][...,0]
            &torch.isfinite(first).all(-1)&(first.amax(-1)<=0))
        _,prefix_hit=observed_futures(dict(rollout,valid_mask=valid&(followed|abstaining_dead_end)),
            horizons=(horizon,))
        censored=prefix_hit[...,0]&followed&~contradicted
    withdrawn=contradicted|censored
    w[withdrawn]=0
    # Even when the sampled action was not the teacher's mode, an observed
    # first-command collision removes that action's positive teaching mass.
    immediate=(valid&rollout['unprotected_mask']&(rollout['outcomes']==1)
        &(action>=0)&(action<18)&(rollout['native_action_frames']>0)
        &(rollout['native_action_frames']<=4))
    ids=torch.nonzero(immediate,as_tuple=True)
    q[ids[0],ids[1],action[ids]]=0
    mass=q.sum(-1,keepdim=True)
    q=q/mass.clamp_min(1e-12)
    w=torch.where(mass[...,0]>0,w,0.)
    trust=rollout['trusted'].clone()
    # Do not silently replace a rejected route with the old straight-line
    # future teacher. Four-frame geometry and native risk targets stay intact.
    trust[...,1:]=trust[...,1:]&~withdrawn[...,None]
    result=dict(rule_probabilities=q,rule_weights=w,trusted=trust,
        teacher_native_contradiction=contradicted,
        teacher_native_contact_removed=immediate)
    if revision == REVISION:
        result.update(teacher_forecast_dead_end_censored=censored,
            teacher_abstaining_native_contact=abstaining_dead_end)
    # A measured native contact also overrides a geometric "safe" label.
    # Otherwise rule CE could reject the action while safe CE/risk regression
    # simultaneously taught it as safe. This is a binary hit, not a distance.
    margins=rollout['margins'].clone()
    selected=action[ids]
    margins[ids[0],ids[1],0,selected]=torch.minimum(margins[ids[0],ids[1],0,selected],margins.new_tensor(-1.))
    result['margins']=margins
    if 'risk_continuation_margins' in rollout:
        rm=rollout['risk_continuation_margins'].clone()
        rv=rollout['risk_continuation_valid'].clone();rs=rollout['risk_continuation_source'].clone()
        rm[ids[0],ids[1],0,selected]=-1.;rv[ids[0],ids[1],0,selected]=True
        rs[ids[0],ids[1],0,selected]=6  # measured native hit; BCE only
        result.update(risk_continuation_margins=rm,risk_continuation_valid=rv,risk_continuation_source=rs)
    return result


def verified_short_mask(record,row,total,first,end,features,geometry):
    """Actual four-frame exposures on a successful native trajectory only.

No label for nearby states, unexecuted directions, the warm-up prefix,
protected ticks, missing measurements, or 12/24f student continuations.
"""
    import numpy as np
    mask=np.zeros(end-first,dtype=bool)
    proof=row.get('native_validation',{})
    if (not row.get('nbnm_success') or row.get('split')!='train'
            or proof.get('contacts')!=0 or proof.get('Bombs')!=0
            or not (proof.get('stage_clear') or proof.get('task_complete'))
            or row.get('native_rule_geometry')!=geometry):return mask
    if not all(k in record for k in ('native_action_frames','observed_contacts','native_frames')):return mask
    duration=record['native_action_frames'];contacts=record['observed_contacts'];frames=record['native_frames']
    if not (len(duration)==len(contacts)==len(frames)):return mask
    offset=total-len(duration)
    for t in range(first,end):
        k=t-offset
        if k<0 or k>=len(duration) or t<int(row.get('training_start',0)):continue
        f=features[t-first]
        mask[t-first]=(duration[k]==4 and not contacts[k] and f[14]>.5 and f[5]<=0 and f[13]<=0)
    return mask


def verified_exit_mask(record, row, features, first, end, geometry):
    """A real 48/96-frame exposure witnesses an exit from the executed first move."""
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    commands = (12, 24)
    last = min(len(features), end + max(commands) - 1)
    eligible = verified_short_mask(record, row, len(features), first, last, features[first:last], geometry)
    targets = np.zeros((end - first, len(commands)), dtype=bool)
    if not eligible.any():
        return targets
    native_frames = record['native_frames']
    offset = len(features) - len(native_frames)
    clock = np.full(last - first, -1, dtype=np.int64)
    lo = max(first, offset)
    clock[lo - first:] = native_frames[lo - offset:last - offset]
    for column, count in enumerate(commands):
        if len(eligible) < count:
            continue
        windows = sliding_window_view(eligible, count)
        times = sliding_window_view(clock, count)
        support = windows.all(-1) & ((times[:, 1:] - times[:, :-1]) == 4).all(-1)
        size = min(len(targets), len(support))
        targets[:size, column] = support[:size]
    return targets


@torch.no_grad()
def matched_native_futures(exposed, native_frames, actions, greedy_actions, commands):
    """A witnessed first action followed by the current greedy student's own decisions."""
    if commands not in (3,6):
        raise ValueError('Expected 12 or 24 original frames in four-frame commands')
    eligible=torch.zeros_like(exposed)
    count=len(actions)-commands+1
    if count<=0:return eligible
    keep=exposed[:count].clone()
    for offset in range(1,commands):
        suffix=slice(offset,offset+count)
        keep&=(exposed[suffix] & (native_frames[suffix]==native_frames[:count]+4*offset)
               & (actions[suffix]==greedy_actions[suffix]))
    eligible[:count]=keep
    return eligible
