"""Dense multiaction movement supervision; execution remains the neural policy.

Targets are conservative current-motion hypotheses. They are not unobserved
native outcomes and do not replace PPO likelihoods, rewards or chosen actions.
"""


def prepare_dense_targets(features, objects, maps, actions, outcomes, *, input_schema, max_labels,horizon_specific=False):
    """Risk-ranked, world-balanced current-motion targets on a frozen rollout.

    The measured outcome can veto a conflicting geometric scene. It never
    proves what any unchosen action would have done. A private RNG keeps
    auxiliary sampling independent of PPO's action and minibatch RNG.
    """
    from collections import Counter
    import numpy as np
    import torch
    from .thrl_inputs import PRIORITY_INPUT_SCHEMAS,object_dimension
    from .thrl_movement_teacher import MovementSafetyTeacher, normal_unprotected, swept_clearances
    if input_schema not in PRIORITY_INPUT_SCHEMAS or objects is None or outcomes is None:
        raise ValueError('Dense online guidance needs repaired objects and native transition evidence')
    if type(max_labels) is not int or not 1 <= max_labels <= 2048:
        raise ValueError('Dense label budget must be 1..2048')
    shape = tuple(features.shape[:2]); size = shape[0] * shape[1]
    def cpu(x): return torch.as_tensor(x).detach().cpu().numpy()
    f = cpu(features).reshape(size, 25); o = cpu(objects).reshape(size, 96, object_dimension(input_schema))
    m = cpu(maps).reshape(size, 16, 92, 96)
    a = cpu(actions).reshape(size); y = cpu(outcomes).reshape(size)
    if not np.isin(y, [-1, 0, 1]).all() or np.any((y == 1) & ((a < 0) | (a >= 18))):
        raise ValueError('Dense targets require valid acknowledged native facts')
    valid = np.zeros(size, bool); margins = np.zeros((size, 3, 18), np.float32)
    horizon_trust=np.zeros((size,3),bool)
    normal = normal_unprotected(f)
    # Closing-time distance ranks imminent hazards before far-away activity.
    relative = o[..., 6:8] * [384, 448]; velocity = o[..., 8:10] * 12
    speed2 = (velocity * velocity).sum(-1)
    tau = np.clip(np.divide(-(relative * velocity).sum(-1), speed2,
        out=np.zeros_like(speed2), where=speed2 > 1e-12), 0, 24)
    distance = np.linalg.norm(relative + tau[..., None] * velocity, axis=-1) - np.linalg.norm(o[..., 10:12] * 128, axis=-1)
    hazard = (o[..., 0] > .5) & ((o[..., 1] > .5) | (o[..., 2] > .5)) & (o[..., 13] > .5)
    risk = np.where(hazard, distance, np.inf).min(-1)
    rng = np.random.default_rng(510241)
    queues = []
    for world in range(shape[1]):
        ids = np.arange(world, size, shape[1]); ids = ids[normal[ids]]
        # Ties are randomized locally, never always the opening of a rollout.
        tie = rng.random(len(ids)); order = np.lexsort((tie, risk[ids], -(y[ids] == 1).astype(int)))
        queues.append(ids[order].tolist())
    teacher = MovementSafetyTeacher(); reasons = Counter(); selected = []; conflicts = []; cursor = [0] * shape[1]
    queried = 0
    while len(selected) < max_labels and any(cursor[w] < len(q) for w, q in enumerate(queues)):
        for world, queue in enumerate(queues):
            if len(selected) >= max_labels: break
            if cursor[world] >= len(queue): continue
            i = queue[cursor[world]]; cursor[world] += 1
            if abs(f[i,6])>1e-6 or abs(f[i,8]-1)>1e-6:
                reasons['unsupported_character_or_difficulty']+=1;continue
            uncertain = teacher._uncertainty(f[i], o[i], m[i],horizon=4 if horizon_specific else None)
            if uncertain: reasons[uncertain] += 1; continue
            origin, *geometry = teacher._scene(f[i], o[i]); queried += 1
            row = np.stack([teacher.clearances(f[i],o[i],np.repeat(origin[None], 18, axis=0),
                teacher.velocities, duration=n) for n in (4, 12, 24)])
            if np.isnan(row).any(): raise FloatingPointError('NaN in dense movement geometry')
            if y[i] >= 0 and 0 <= a[i] < 18:
                gap = row[0, int(a[i])]
                if (y[i] == 1 and gap > 0) or (y[i] == 0 and gap < 0):
                    reasons['native_geometry_contradiction'] += 1; conflicts.append(i); continue
            any_laser=np.any((o[i,:,0]>.5)&((o[i,:,3]>.5)|(o[i,:,4]>.5)))
            if not hazard[i].any() and not any_laser: reasons['no_hazard'] += 1; continue
            margins[i] = np.clip(row, -64, 64); valid[i] = True; selected.append(i)
            horizon_trust[i]=([True,*[teacher._uncertainty(f[i],o[i],m[i],horizon=n) is None for n in (12,24)]]
                if horizon_specific else [True]*3)
    indices = np.asarray(selected, dtype=int)
    evidence = dict(states=int(valid.sum()), geometry_queries=queried,
        protected_or_non_normal=int((~normal).sum()), abstentions=dict(reasons),
        native_geometry_contradictions=len(conflicts), conflict_indices=conflicts,
        worlds_labeled=len(set((indices % shape[1]).tolist())),
        earliest_labeled_time=int((indices // shape[1]).min()) if len(indices) else None,
        latest_labeled_time=int((indices // shape[1]).max()) if len(indices) else None,
        target_is_measured_native_counterfactual=False, policy_actions_changed=False,
        horizons_native_frames=[4, 12, 24], selection='native hit then closing distance; world round robin')
    if horizon_specific:
        evidence.update(horizon_specific_abstention=True,trusted_horizons=horizon_trust.sum(0).tolist())
    trusted=horizon_trust.reshape(*shape,3) if horizon_specific else valid.reshape(shape)
    return torch.from_numpy(margins.reshape(*shape, 3, 18)), torch.from_numpy(trusted), evidence


def dense_safety_loss(logits,margins,trusted):
    import torch
    x=logits.reshape(-1,19)[:,:18]
    c=torch.as_tensor(margins,device=x.device,dtype=x.dtype).detach().reshape(-1,3,18)
    valid=torch.as_tensor(trusted,device=x.device,dtype=torch.bool).reshape(-1)
    if len(c)!=len(x) or len(valid)!=len(x) or not torch.isfinite(c).all():
        raise ValueError('Dense geometry targets must be finite and observation-aligned')
    if not valid.any():return x.sum()*0.,dict(states=0)
    c,x=c[valid],x[valid]
    safe4=torch.sigmoid((c[:,0]-1)/2)
    safe12=torch.sigmoid((c[:,1]-1)/4)
    safe24=torch.sigmoid((c[:,2]-1)/6)
    target=safe4*(.25+.5*safe12+.25*safe24)
    # Retain short-step alternatives when every long straight continuation
    # fails. A policy can turn at the next4-frame decision.
    weight=1+3*(target<.5).to(x.dtype)
    loss=(torch.nn.functional.binary_cross_entropy_with_logits(x,target,reduction='none')*weight).sum()/weight.sum()
    with torch.no_grad():
        probabilities=x.softmax(-1)
        critical4=(c[:,0]<=1).any(-1);critical12=(c[:,1]<=1).any(-1)
        def unsafe_mass(horizon,critical):
            return float((probabilities[critical]*(c[critical,horizon]<=1)).sum(-1).mean()) if critical.any() else None
        detail=dict(states=int(valid.sum()),critical4_states=int(critical4.sum()),critical12_states=int(critical12.sum()),
            predicted_unsafe4_mass=unsafe_mass(0,critical4),predicted_unsafe12_mass=unsafe_mass(1,critical12),
            target_is_measured_native_counterfactual=False,Bomb_logit_supervised=False)
    return loss,detail


def receding_step_safety_loss(logits,margins,trusted):
    """Immediate feasible-set CE, then a weak conditional long-range preference.

    A doomed straight24-frame continuation must not outrank a collision-free
    next4-frame move: the agent can turn again. Uncertain12/24-frame targets
    are absent rather than fabricated. Future preferences act only within the
    first-step feasible set and cannot directly favor an unsafe4-frame action.
    """
    import torch
    x=logits.reshape(-1,19)[:,:18]
    c=torch.as_tensor(margins,device=x.device,dtype=x.dtype).detach().reshape(-1,3,18)
    v=torch.as_tensor(trusted,device=x.device,dtype=torch.bool).reshape(-1,3)
    if len(c)!=len(x) or len(v)!=len(x) or not torch.isfinite(c).all():
        raise ValueError('Per-horizon movement targets must be paired and finite')
    if (v[:,1:]&~v[:,:1]).any():raise ValueError('Long-horizon targets require trusted immediate geometry')
    safe=c[:,0]>1.;size=safe.sum(-1)
    critical=v[:,0]&(size>0)&(size<18)
    immediate=x.sum()*0.
    if critical.any():
        q=x[critical];keep=safe[critical]
        immediate=(torch.logsumexp(q,-1)-torch.logsumexp(q.masked_fill(~keep,-torch.inf),-1)).mean()
    future=v[:,0]&(size>0)&v[:,1:].any(-1)
    continuation=x.sum()*0.
    if future.any():
        q=x[future];keep=safe[future];z=c[future];known=v[future,1:].to(x.dtype)
        scores=(torch.stack((torch.sigmoid((z[:,1]-1)/4),torch.sigmoid((z[:,2]-1)/6)),1)*known[:,:,None]).sum(1)/known.sum(-1)[:,None]
        informative=(scores.masked_fill(~keep,-torch.inf).amax(-1)-scores.masked_fill(~keep,torch.inf).amin(-1))>.05
        if informative.any():
            q,keep,scores=q[informative],keep[informative],scores[informative]
            targets=(scores*2).masked_fill(~keep,-torch.inf).softmax(-1).detach()
            logp=q.masked_fill(~keep,-torch.inf).log_softmax(-1)
            continuation=-(targets*logp.masked_fill(~keep,0)).sum(-1).mean()
    with torch.no_grad():
        p=x.softmax(-1)
        critical12=v[:,1]&(c[:,1]<=1).any(-1)
        d=dict(states=int(v[:,0].sum()),critical4_states=int(critical.sum()),critical12_states=int(critical12.sum()),
            predicted_unsafe4_mass=float((p[critical]*~safe[critical]).sum(-1).mean()) if critical.any() else None,
            predicted_unsafe12_mass=float((p[critical12]*(c[critical12,1]<=1)).sum(-1).mean()) if critical12.any() else None,
            trusted_horizons=v.sum(0).tolist(),immediate_feasible_set_CE=float(immediate),
            future_conditional_CE=float(continuation),future_preference_coefficient=.05,
            target_is_measured_native_counterfactual=False,Bomb_logit_supervised=False,
            schema='receding-step-safety-v2')
    return immediate+.05*continuation,d
