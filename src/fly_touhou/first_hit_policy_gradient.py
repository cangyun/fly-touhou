"""Finite-horizon first-hit reward and ordinary18 episodic policy gradients.

This is a declared movement curriculum. Its horizon termination is part of
the training task and must not be confused with a completed native campaign.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F


def first_hit_reward(row):
    elapsed=row['elapsed_frames'];action=row['action']
    if (isinstance(elapsed,bool) or not isinstance(elapsed,int) or elapsed<1
            or isinstance(action,bool) or not isinstance(action,int) or not 0<=action<18):
        raise ValueError('Measured positive native frames and ordinary18 action required')
    if row['teacher_action_calls'] or row['safety_takeovers']:
        raise ValueError('This reward requires an action actually executed by the student')
    survival=elapsed/60.;collision=-5.*bool(row['hit']);forecast=wall=0.
    p=row['snapshot']['player'];details=row.get('movement_details') or {}
    c=details.get('clearances')
    if p and p['state']==1 and p['invulnerable_frames']==0:
        wall=-.01*(max(0.,abs(p['x'])-160.)/24.+max(0.,p['y']-416.)/16.)
        if c is not None:
            if len(c)!=18:raise ValueError('Exactly18 current-geometry forecast directions required')
            values=[math.inf if v is None else float(v) for v in c]
            if any(math.isnan(v) for v in values):raise ValueError('Invalid clearance forecast')
            if max(values)>0:
                forecast=-.25*min(1.,max(0.,(5.-values[action])/5.))
    components={'measured_survival_seconds':survival,'actual_first_hit':collision,
        'executed_direction_forecast_penalty':forecast,'lower_side_edge_cost':wall}
    reward=sum(components.values())
    if not math.isfinite(reward):raise ValueError('Nonfinite first-hit reward')
    return reward,components


def finite_lesson_returns(rewards,ended,*,gamma=.99):
    r=np.asarray(rewards,dtype=np.float64);ends=np.asarray(ended,dtype=bool)
    if (r.ndim!=1 or not len(r) or ends.shape!=r.shape or not ends[-1]
            or not np.isfinite(r).all() or not 0<=gamma<=1):
        raise ValueError('Complete finite curriculum episodes and finite rewards required')
    result=np.empty_like(r);tail=0.
    for i in range(len(r)-1,-1,-1):
        if ends[i]:tail=0.
        tail=r[i]+gamma*tail;result[i]=tail
    return result


def ordinary_policy_gradient(logits,actions,advantages,*,entropy_coefficient=.001):
    logp=F.log_softmax(logits[...,:18].double(),dim=-1)
    # This Torch/CUDA runtime's deterministic gather adjoint cannot be
    # captured. Negative unreduced NLL is the already-audited exact selector
    # used by the continuing full-brain PPO implementation.
    chosen=-F.nll_loss(logp,actions,reduction='none')
    entropy=-(logp.exp()*logp).sum(-1)
    return (-advantages.detach()*chosen-entropy_coefficient*entropy).mean()


def pg_tensor_loss(model,observations,actions,advantages,state,entropy_coefficient):
    outputs=[]
    for i in range(len(observations)):
        logits,state=model._forward_validated(observations[i:i+1],state);outputs.append(logits)
    logits=torch.cat(outputs)
    return ordinary_policy_gradient(logits,actions,advantages,entropy_coefficient=entropy_coefficient),logits,state
