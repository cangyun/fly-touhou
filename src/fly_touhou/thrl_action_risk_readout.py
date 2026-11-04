"""Learned immediate-action risk readout; no native access or analytic control."""
import math
import numpy as np
import torch
from torch import nn


class ActionRiskReadout(nn.Module):
    def __init__(self,input_dim=640,hidden_dim=128):
        super().__init__();self.input_dim=input_dim;self.hidden_dim=hidden_dim
        self.network=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,18))
    def forward(self,features):return self.network(features)


def native_risk_targets(margins,trusted,actions,outcomes):
    """Only trusted geometry or the actually acknowledged action is labeled.

    Positive: modeled intersection. Negative: more than one unit clearance.
    The thin positive band is unknown. Native outcomes override the model on
    the executed action only; they never prove alternative native actions.
    """
    c=np.asarray(margins)[:,0];valid=np.asarray(trusted)[:,0]
    if c.shape[1:]!=(18,) or valid.shape!=(len(c),):raise ValueError('Per-state action geometry required')
    target=(c<=0).astype(np.float32);known=valid[:,None]&((c<=0)|(c>1))
    a=np.asarray(actions);y=np.asarray(outcomes)
    if a.shape!=valid.shape or y.shape!=valid.shape or not np.isin(y,[-1,0,1]).all():raise ValueError('Native fact rows differ')
    facts=y>=0
    if ((a[facts]<0)|(a[facts]>=18)).any():raise ValueError('Facts must be measured movements')
    ix=np.flatnonzero(facts);target[ix,a[ix]]=y[ix];known[ix,a[ix]]=True
    return target,known


def descriptor_indices(descriptors,length,worlds):
    result=[]
    for world,start in descriptors:
        if not 0<=world<worlds or not 0<=start<=length-16:raise ValueError('Descriptor outside source')
        result.extend(range(world*length+start,world*length+start+16))
    if len(set(result))!=len(result):raise ValueError('Overlapping native descriptors')
    return np.asarray(result,dtype=np.int64)


def adjust_movement_logits(logits,risk_logits,beta=4.,*,check_finite=True):
    """Soft neural reweighting; preserve the Bomb marginal, never mask a move.

    Used by the versioned global/entity risk policies. Both scores are network
    outputs; no collision computation or teacher runs at execution.
    """
    if not math.isfinite(beta) or beta<0 or logits.shape[-1]!=19 or risk_logits.shape!=(*logits.shape[:-1],18):
        raise ValueError('Finite nonnegative strength and matched19/18 outputs required')
    if type(check_finite) is not bool:raise ValueError('Explicit finite-check mode required')
    if check_finite and (not torch.isfinite(logits[...,:18]).all() or not torch.isfinite(risk_logits).all()):raise ValueError('Nonfinite movement scores')
    if beta==0:return logits
    old=logits[...,:18];corrected=old-beta*torch.nn.functional.softplus(risk_logits)
    corrected=corrected+torch.logsumexp(old,-1,keepdim=True)-torch.logsumexp(corrected,-1,keepdim=True)
    return torch.cat((corrected,logits[...,18:]),-1)
