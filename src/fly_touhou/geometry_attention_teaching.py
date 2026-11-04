"""Prepared geometry teaching for the pair VALUE pool; inactive by default.

Labels are training-only current-motion hypotheses. No teacher risk vector is
fed into the policy or brain. Importing this module starts no model or game.
"""
from __future__ import annotations
import numpy as np

SCHEMA='point-object-attention-four-frame-teaching-v1'


def per_object_first4(features,objects,maps,*,teacher=None):
    from .thrl_movement_teacher import MovementSafetyTeacher,normal_unprotected,swept_clearances
    teacher=teacher or MovementSafetyTeacher()
    f,o=np.asarray(features),np.asarray(objects)
    if f.shape!=(25,) or o.shape!=(96,29):raise ValueError('Original matched V5 fields required')
    if not normal_unprotected(f):return None,dict(reason='protected_or_non_normal')
    if abs(f[6])>1e-6 or abs(f[8]-1.)>1e-6:return None,dict(reason='unsupported_character_or_difficulty')
    if np.any((o[:,0]>.5)&((o[:,3]>.5)|(o[:,4]>.5))):
        return None,dict(reason='point_attention_diagnostic_excludes_laser_scenes')
    reason=teacher._uncertainty(f,o,maps,horizon=4)
    if reason:return None,dict(reason=reason)
    valid=(o[:,0]>.5)&((o[:,1]>.5)|(o[:,2]>.5))&(o[:,13]>.5)
    ids=np.flatnonzero(valid)
    if not len(ids):return None,dict(reason='no_point_hazard')
    origin,positions,motions,circle,radii,extents,offset=teacher._scene(f,o)
    starts=np.repeat(origin[None],18,axis=0)
    result=np.full((18,96),64.,np.float32)
    result[:,ids]=np.clip(swept_clearances(starts,teacher.velocities,positions,motions,
        circle,radii,extents,offset,duration=4,per_object=True),-64.,64.)
    return (result,valid),dict(reason='known_four_frame_point_geometry',objects=len(ids),
        native_counterfactual_verified=False,source=SCHEMA)


def attention_teaching_loss(logits,margins,valid_objects,*,temperature=2.,critical_margin=8.):
    """Soft multi-object targets; do not force one unique critical obstacle."""
    import torch
    from torch.nn import functional as F
    target=torch.as_tensor(margins,device=logits.device,dtype=logits.dtype).detach()
    valid=torch.as_tensor(valid_objects,device=logits.device,dtype=torch.bool)
    if logits.shape!=target.shape or logits.shape[-1]!=96 or valid.shape!=logits.shape[:-2]+(96,):
        raise ValueError('Paired per-action object scores/margins and original object mask required')
    real=valid[...,None,:].expand_as(logits)
    eligible=(torch.where(real,target,torch.inf).amin(-1)<=critical_margin)&real.any(-1)
    safe=real.clone();safe[...,0]|=~real.any(-1)
    labels=(-target.clamp(-8.,32.)/temperature).masked_fill(~safe,-torch.inf).softmax(-1)
    logp=logits.masked_fill(~safe,-torch.inf).log_softmax(-1)
    # Subtract target entropy to report KL, not its irreducible entropy floor.
    terms=labels*(labels.clamp_min(1e-30).log()-logp.masked_fill(~safe,0.))
    row=terms.sum(-1)
    loss=torch.where(eligible,row,0.).sum()/eligible.sum().clamp_min(1.)
    with torch.no_grad():
        p=logp.exp();entropy=-(p*logp.masked_fill(~safe,0.)).sum(-1)
        entropy=entropy/valid.sum(-1).clamp_min(2).to(logits.dtype).log()[...,None]
        minimum=torch.where(real,target,torch.inf).amin(-1)
        chosen=target.gather(-1,logp.argmax(-1,keepdim=True)).squeeze(-1)
        top_is_critical=(chosen<=minimum+1.).to(logits.dtype)
        def mean(x):return torch.where(eligible,x,0.).sum()/eligible.sum().clamp_min(1.)
    return loss,dict(eligible_action_queries=eligible.sum().detach(),
        valid_objects=valid.sum().detach(),normalized_entropy=mean(entropy),
        critical_object_top1=mean(top_is_critical),
        multi_object_queries=(eligible&(valid.sum(-1)>1)[...,None]).sum(),
        temperature=temperature,critical_margin=critical_margin)
