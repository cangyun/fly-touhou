"""Captureable forms of the five successful-GRU guidance terms only.

Geometry trust is supplied by the original prepare_dense_targets, including
its native-fact contradiction veto. Bomb/unknown outcomes do not revoke a
trusted normal BEFORE observation. No risk prediction or ranking is added.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F
from .fly_visual_task_supervision import TaskSupervisionLossConfig

SCHEMA='fly-GRU-aligned-five-guidance-terms-v1'
TERM_NAMES=('safe','future','hit','rule','deathbomb')
DEFAULT_CONFIG=TaskSupervisionLossConfig(safe_coefficient=.2,future_coefficient=.05,
    collision_coefficient=.02,rule_coefficient=.1,deathbomb_coefficient=.05,
    risk_coefficient=0.,risk_ranking_coefficient=0.)


def _inputs(logits,margins,trusted,actions,outcomes,valid_mask,unprotected_mask,
        config,rule_probabilities,rule_weights,deathbomb_mask):
    if not isinstance(config,TaskSupervisionLossConfig) or config.risk_coefficient!=0 or config.risk_ranking_coefficient!=0:
        raise ValueError('Original five guidance terms only; risk BCE and ranking must be zero')
    if logits.ndim!=3 or logits.shape[-1]!=19 or not logits.is_floating_point():
        raise ValueError('Time-major source19 logits required')
    leading=logits.shape[:2]
    def tensor(value,shape,kind):
        t=torch.as_tensor(value,device=logits.device).detach()
        if t.shape!=shape or (kind=='bool' and t.dtype!=torch.bool) or (kind=='int' and t.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64)) or (kind=='float' and not t.is_floating_point()):
            raise ValueError('Aligned guidance shape/dtype differs:'+kind)
        return t.long() if kind=='int' else t.to(logits.dtype) if kind=='float' else t
    c=tensor(margins,(*leading,3,18),'float');v=tensor(trusted,(*leading,3),'bool')
    a=tensor(actions,leading,'int');y=tensor(outcomes,leading,'int')
    valid=torch.ones(leading,dtype=torch.bool,device=logits.device) if valid_mask is None else tensor(valid_mask,leading,'bool')
    normal=tensor(unprotected_mask,leading,'bool')
    if (rule_probabilities is None)!=(rule_weights is None):raise ValueError('Paired current-observation rule labels required')
    if config.rule_coefficient>0 and rule_probabilities is None:raise ValueError('Rule guidance needs actual reference labels')
    p=logits.new_zeros((*leading,19)) if rule_probabilities is None else tensor(rule_probabilities,(*leading,19),'float')
    w=logits.new_zeros(leading) if rule_weights is None else tensor(rule_weights,leading,'float')
    if config.deathbomb_coefficient>0 and deathbomb_mask is None:raise ValueError('Measured deathbomb mask required')
    db=torch.zeros(leading,dtype=torch.bool,device=logits.device) if deathbomb_mask is None else tensor(deathbomb_mask,leading,'bool')
    return c,v,a,y,valid,normal,p,w,db


def validate_gru_aligned_guidance_inputs(logits,margins,trusted,actions,outcomes,*,
        valid_mask=None,unprotected_mask,config=DEFAULT_CONFIG,rule_probabilities=None,
        rule_weights=None,deathbomb_mask=None):
    """Explicit host checks on sealed evidence, outside a captured block."""
    c,v,a,y,valid,normal,p,w,db=_inputs(logits,margins,trusted,actions,outcomes,valid_mask,
        unprotected_mask,config,rule_probabilities,rule_weights,deathbomb_mask)
    if (not bool(torch.isfinite(logits[..., :18][valid]).all()) or not bool(torch.isfinite(c).all())
            or bool(((a<0)|(a>18))[valid].any()) or not bool(((y==-1)|(y==0)|(y==1)).all())
            or bool((v[...,1:]&~v[...,:1]).any())
            or bool(((y>=0)&valid&(~normal|(a==18))).any())):
        raise ValueError('Finite paired geometry and acknowledged normal movement facts required')
    if not bool(torch.isfinite(w[valid]).all()) or not bool(((w[valid]==0)|(w[valid]==1)).all()):
        raise ValueError('Original rule reference confidence must be exactly zero or one')
    row=valid&(w>0);legal=torch.isfinite(logits)
    if (not bool(torch.isfinite(p[row]).all()) or bool((p[row]<0).any())
            or not torch.allclose(p[row].sum(-1),torch.ones_like(w[row]),atol=2e-6,rtol=0)
            or bool(((p*legal).sum(-1)[row]<=0).any())):
        raise ValueError('Active rule distributions need positive legal mass before renormalization')
    if not bool(torch.isfinite(logits[...,18][valid&db]).all()):
        raise ValueError('Measured deathbomb labels require a legal stocked Bomb')


def gru_aligned_guidance_loss(logits,margins,trusted,actions,outcomes,*,valid_mask=None,
        unprotected_mask,config=DEFAULT_CONFIG,normalizers=None,rule_probabilities=None,
        rule_weights=None,deathbomb_mask=None,veto_rule_on_contact=False):
    """Local sums / whole256-row counts; sum four microblocks without reweighting.

Call the host validator on sealed CPU evidence first. This tensor path has no
data-dependent Python branches. Receding geometry uses original raw movement
scores; hit/rule/DB use the original T1 policy clamp and inventory support.
"""
    c,v,a,y,valid,normal,p,w,db=_inputs(logits,margins,trusted,actions,outcomes,valid_mask,
        unprotected_mask,config,rule_probabilities,rule_weights,deathbomb_mask)
    eligible=valid&normal
    trust=v&eligible[...,None]
    # Eligibility deliberately does NOT depend on sampled action or outcome.
    move=torch.where(eligible[...,None],logits[...,:18],0.)
    safe=c[...,0,:]>1.;size=safe.sum(-1)
    index=a.clamp(0,17)
    fallback=F.one_hot(torch.zeros_like(index),18).bool()
    keep=safe|((size==0)[...,None]&fallback)
    safe_row=trust[...,0]&(size>0)&(size<18)
    safe_ce=move.logsumexp(-1)-move.masked_fill(~keep,-torch.inf).logsumexp(-1)
    known=trust[...,1:].to(move.dtype)
    scores=(torch.stack((torch.sigmoid((c[...,1,:]-1.)/4.),torch.sigmoid((c[...,2,:]-1.)/6.)),-2)
        *known[...,None]).sum(-2)/known.sum(-1)[...,None].clamp_min(1.)
    best=scores.masked_fill(~keep,-torch.inf).amax(-1)
    worst=scores.masked_fill(~keep,torch.inf).amin(-1)
    future_row=trust[...,0]&(size>0)&trust[...,1:].any(-1)&((best-worst)>.05)
    target=(scores*2.).masked_fill(~keep,-torch.inf).softmax(-1).detach()
    conditional_logp=move.masked_fill(~keep,-torch.inf).log_softmax(-1).masked_fill(~keep,0.)
    future_ce=-(target*conditional_logp).sum(-1)
    hit_row=eligible&(a>=0)&(a<18)&(y==1)
    hit_move=move.clamp(-80.,80.)
    hit_nll=hit_move.logsumexp(-1)-hit_move.masked_fill(F.one_hot(index,18).bool(),-torch.inf).logsumexp(-1)
    if veto_rule_on_contact:
        from .compact_rule_teacher_replay import mask_rule_guidance_on_contact
        w,_=mask_rule_guidance_on_contact(w,a,y,eligible)
    rule_weight=torch.where(valid,w,0.);rule_row=valid&(rule_weight>0);db_row=valid&db
    full=torch.where((rule_row|db_row)[...,None],logits,0.)
    clean=full.clamp(-80.,80.).masked_fill(torch.isneginf(full),-torch.inf)
    legal=torch.isfinite(clean)
    logp=clean.log_softmax(-1)
    selected=torch.where(rule_row[...,None],p,0.)*legal.to(p.dtype)
    mass=selected.sum(-1,keepdim=True)
    selected=selected/torch.where(mass>0,mass,torch.ones_like(mass))
    rule_ce=-(selected*logp.masked_fill(~legal,0.)).sum(-1)
    db_ce=-logp[...,18]
    numerators=dict(safe=torch.where(safe_row,safe_ce,0.).sum(),
        future=torch.where(future_row,future_ce,0.).sum(),hit=torch.where(hit_row,hit_nll,0.).sum(),
        rule=(rule_weight*rule_ce).sum(),deathbomb=torch.where(db_row,db_ce,0.).sum())
    denominators=dict(safe=safe_row.sum(),future=future_row.sum(),hit=hit_row.sum(),
        rule=rule_weight.sum(),deathbomb=db_row.sum())
    denominators={k:value.detach() for k,value in denominators.items()}
    if normalizers is not None and set(normalizers)!=set(TERM_NAMES):
        raise ValueError('Only safe/future/hit/rule/deathbomb whole-minibatch counts are allowed')
    norms=denominators if normalizers is None else normalizers
    norms={k:torch.as_tensor(norms[k],device=logits.device,dtype=logits.dtype).detach() for k in TERM_NAMES}
    if any(value.ndim!=0 for value in norms.values()):raise ValueError('Scalar whole-minibatch counts required')
    losses={k:numerators[k]/norms[k].clamp_min(1.) for k in TERM_NAMES}
    weighted_safe=config.safe_coefficient*(losses['safe']+config.future_coefficient*losses['future'])
    weighted_collision=config.collision_coefficient*losses['hit']
    weighted_rule=config.rule_coefficient*losses['rule']
    weighted_deathbomb=config.deathbomb_coefficient*losses['deathbomb']
    total=weighted_safe+weighted_collision+weighted_rule+weighted_deathbomb
    return dict(total_loss=total,safe_loss=losses['safe'],future_loss=losses['future'],
        collision_loss=losses['hit'],rule_loss=losses['rule'],deathbomb_loss=losses['deathbomb'],
        weighted_safe=weighted_safe,weighted_collision=weighted_collision,
        weighted_rule=weighted_rule,weighted_deathbomb=weighted_deathbomb,
        numerators=numerators,denominators=denominators,normalizers_used=norms,
        trusted_horizon_counts=trust.sum((0,1)).detach())
