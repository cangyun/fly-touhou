"""Explicit GRU-style signed4f risk supervision on following-DN predictions.

The predictor architecture/execution remains cost=softplus(score). Its margin
coordinate is recovered algebraically as1-.5*score,not inverse_softplus(cost).
Pool means are a bounded256-row adaptation of the original4096 query sampler;
the .05 joint coefficient is a NEW coupling choice,not the old independent
GRU risk pretraining coefficient. No ranking or aggregate D95-cost target.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

MODE='signed-clearance4f'
COEFFICIENT=.05
POOL_WEIGHTS=(.25,.5,.25)


def protocol():
    return dict(mode=MODE,label='original trusted4f geometry signed clearance;clip[-64,64]',
        prediction='1-.5*head.layers(following_DN)',execution_cost='softplus(the same score)',
        regression='SmoothL1(pred/8,target/8,beta.1)',
        classification='.2*BCEWithLogits(-(pred-.5),target<=0);known target<=0 or>1',
        query_pools=['target<=0','abs(target)<8','all valid queries'],pool_weights=list(POOL_WEIGHTS),
        overlapping_pools=True,empty_pool_fallback='all valid queries',
        BCE_normalization='known subset of the weighted query mixture,not independent per-pool BCE means',
        coefficient=COEFFICIENT,coefficient_is_new_joint_coupling=True,
        exact_original4096_sampling=False,ranking=False,teacher_aggregate_cost_used=False,
        valid='real row & trusted4f & unprotected & f21>.5 & f20<=0 & f5<=0')


def signed_clearance4f_terms(predicted_margin,task,features,valid_mask):
    """Query-weighted Huber+.2knownBCE,then fixed.05 on one real256-row block."""
    target=task['margins'][...,0,:].detach().to(predicted_margin.dtype).clamp(-64.,64.)
    if predicted_margin.shape!=target.shape or target.shape[-1]!=18:
        raise ValueError('Paired source18 signed4f predictions/targets required')
    normal=(features[...,21]>.5)&(features[...,20]<=0)&(features[...,5]<=0)
    rows=valid_mask&task['trusted'][...,0]&task['unprotected_mask']&normal
    valid=rows[...,None].expand_as(target)
    target=torch.where(valid,target,torch.zeros_like(target))
    pred=torch.where(valid,predicted_margin,torch.zeros_like(predicted_margin))
    huber=F.smooth_l1_loss(pred/8.,target/8.,beta=.1,reduction='none')
    known=(target<=0)|(target>1.)
    bce=F.binary_cross_entropy_with_logits(-(pred-.5),(target<=0).to(pred.dtype),reduction='none')
    pools=(valid&(target<=0),valid&(target.abs()<8.),valid)
    huber_loss=pred.sum()*0.;bce_numerator=pred.sum()*0.;known_fraction=pred.new_zeros(())
    pool_counts={};fallback_counts={}
    for name,weight,mask in zip(('danger','boundary','all'),POOL_WEIGHTS,pools):
        original_count=mask.sum()
        effective=torch.where(original_count>0,mask,valid)
        count=effective.sum().to(pred.dtype);denominator=count.clamp_min(1.)
        selected_known=effective&known
        huber_loss=huber_loss+weight*torch.where(effective,huber,0.).sum()/denominator
        bce_numerator=bce_numerator+weight*torch.where(selected_known,bce,0.).sum()/denominator
        known_fraction=known_fraction+weight*selected_known.sum().to(pred.dtype)/denominator
        pool_counts[name]=original_count.detach()
        fallback_counts[name]=(original_count==0).to(torch.int64).detach()
    bce_loss=bce_numerator/known_fraction.clamp_min(1e-12)
    loss=huber_loss+.2*bce_loss
    return dict(loss=loss,weighted_loss=COEFFICIENT*loss,huber_loss=huber_loss,BCE_loss=bce_loss,
        count=valid.sum().detach(),valid_scenes=rows.sum().detach(),known_queries=(valid&known).sum().detach(),
        danger_pool_queries=pool_counts['danger'],boundary_pool_queries=pool_counts['boundary'],
        all_pool_queries=pool_counts['all'],danger_pool_fallback=fallback_counts['danger'],
        boundary_pool_fallback=fallback_counts['boundary'],known_mixture_fraction=known_fraction.detach(),
        coefficient=pred.new_tensor(COEFFICIENT),teacher_aggregate_cost_used=False)
