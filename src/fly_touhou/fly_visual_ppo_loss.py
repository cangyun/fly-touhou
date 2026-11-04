"""Scalar on-policy PPO + frozen-teacher direction + bounded-gain regularity.

Mathematics only: no optimizer/rollout/native execution. Value MSE is unhalved,
and value_coeff is applied exactly once. Advantages are supplied/detached; any
normalization belongs to the whole rollout, not each four-step replay block.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from .thrl_policy_actions import policy_logits

PILOT_GAMMA=.999
PILOT_GAE_LAMBDA=.995


@dataclass(frozen=True)
class VisualPPOLossConfig:
    clip_epsilon:float=.2
    value_clip_epsilon:float|None=.2
    value_coeff:float=.5
    entropy_coeff:float=.0001
    teacher_coeff:float=1.
    brain_reg_coeff:float=.01
    teacher_objective:str='kl'

    def __post_init__(self):
        if not 0<self.clip_epsilon<1 or self.teacher_objective not in ('kl','ce'):
            raise ValueError('PPO clip in(0,1) and one teacher KL/CE objective required')
        if self.value_clip_epsilon is not None and (not math.isfinite(self.value_clip_epsilon) or self.value_clip_epsilon<0):
            raise ValueError('Finite nonnegative value clipping, or None, required')
        if any(not math.isfinite(v) or v<0 for v in (self.value_coeff,self.entropy_coeff,self.teacher_coeff,self.brain_reg_coeff)):
            raise ValueError('Finite nonnegative loss coefficients required')


def masked_policy_logits(raw_logits,legal_actions):
    if raw_logits.shape[-1]!=19 or legal_actions.shape!=raw_logits.shape or legal_actions.dtype!=torch.bool:
        raise ValueError('Matched source19 logits and recorded inventory legality required')
    if not bool(legal_actions.any(-1).all()):raise ValueError('Every row needs a legal action')
    # Fixed T1, same finite clamp/stock legality as actual rollout sampling.
    clean=policy_logits(raw_logits,action_temperature=1.).masked_fill(~legal_actions,-torch.inf)
    if not bool(torch.isfinite(clean[legal_actions]).all()):raise ValueError('Legal logits must be finite')
    return clean


def categorical_policy(raw_logits,legal_actions):
    return torch.distributions.Categorical(logits=masked_policy_logits(raw_logits,legal_actions))


def visual_ppo_loss(raw_logits,values,actions,old_log_probs,old_values,advantages,returns,
                    teacher_probabilities,legal_actions,gain_logits,*,config=VisualPPOLossConfig(),valid_mask=None):
    """Return separately named terms; do not add another PPO total/value term."""
    leading=raw_logits.shape[:-1]
    if not leading or any(v.shape!=leading for v in (values,actions,old_log_probs,old_values,advantages,returns)):
        raise ValueError('Matched scalar values/actions/advantages/returns on the real rollout rows required')
    valid=torch.ones(leading,dtype=torch.bool,device=raw_logits.device) if valid_mask is None else valid_mask
    if valid.dtype!=torch.bool or valid.shape!=leading or not bool(valid.any()):raise ValueError('Nonempty real-row mask required')
    for value in (values,old_log_probs,old_values,advantages,returns):
        if not bool(torch.isfinite(value[valid]).all()):raise FloatingPointError('Nonfinite real PPO row')
    if actions.dtype!=torch.int64 or bool(((actions<0)|(actions>=19)).any()):raise ValueError('Source19 action indices required')
    if not bool(legal_actions.gather(-1,actions.unsqueeze(-1)).squeeze(-1)[valid].all()):raise ValueError('Recorded action must be legal')
    distribution=categorical_policy(raw_logits,legal_actions)
    new_logp=distribution.log_prob(actions)
    def mean(value):return value[valid].mean()
    log_ratio=(new_logp-old_log_probs.detach()).double();ratio=log_ratio.exp()
    if not bool(torch.isfinite(ratio[valid]).all()):raise FloatingPointError('Nonfinite actual PPO ratio')
    adv=advantages.detach().double()
    policy_loss=-mean(torch.minimum(ratio*adv,ratio.clamp(1-config.clip_epsilon,1+config.clip_epsilon)*adv))
    squared=(values-returns.detach()).square()
    if config.value_clip_epsilon is not None:
        old=old_values.detach();clipped=old+(values-old).clamp(-config.value_clip_epsilon,config.value_clip_epsilon)
        squared=torch.maximum(squared,(clipped-returns.detach()).square())
    value_loss=mean(squared)  # unhalved MSE: value_coeff=.5 is the only factor .5
    entropy=mean(distribution.entropy())
    teacher_ce=raw_logits.new_zeros(());teacher_kl=raw_logits.new_zeros(())
    if teacher_probabilities is not None:
        p=teacher_probabilities.detach()
        if (p.shape!=raw_logits.shape or not bool(torch.isfinite(p[valid]).all()) or bool((p[valid]<0).any())
                or not torch.allclose(p[valid].sum(-1),torch.ones_like(p[valid].sum(-1)),atol=2e-6,rtol=0)):
            raise ValueError('Frozen final19 T1 teacher probabilities required')
        p=p.masked_fill(~legal_actions,0);mass=p.sum(-1,keepdim=True)
        if not bool((mass.squeeze(-1)[valid]>0).all()):raise ValueError('Teacher needs legal probability mass')
        p=p/mass.clamp_min(torch.finfo(p.dtype).tiny)
        log_student=distribution.logits.masked_fill(p==0,0)
        log_teacher=p.clamp_min(torch.finfo(p.dtype).tiny).log().masked_fill(p==0,0)
        teacher_ce=mean(-(p*log_student).sum(-1))
        teacher_kl=mean((p*(log_teacher-log_student)).sum(-1))
    elif config.teacher_coeff:
        raise ValueError('Positive teacher coefficient requires actually recorded D95 final19 labels')
    if gain_logits.ndim!=1 or not gain_logits.numel() or not bool(torch.isfinite(gain_logits).all()):
        raise ValueError('Finite full-neuron bounded-gain logits required')
    brain_regularization=gain_logits.square().mean()
    teacher_loss=teacher_kl if config.teacher_objective=='kl' else teacher_ce
    total=(policy_loss+config.value_coeff*value_loss-config.entropy_coeff*entropy
           +config.teacher_coeff*teacher_loss+config.brain_reg_coeff*brain_regularization)
    return dict(total_loss=total,policy_loss=policy_loss,value_loss=value_loss,entropy=entropy,
        teacher_loss=teacher_loss,teacher_KL=teacher_kl,teacher_CE=teacher_ce,brain_regularization=brain_regularization,
        approximate_KL=mean(torch.expm1(log_ratio)-log_ratio),clip_fraction=mean((ratio-1).abs().gt(config.clip_epsilon).double()),
        ratio_mean=mean(ratio))


@torch.no_grad()
def scalar_gae(rewards,values,bootstrap_values,terminated,*,truncated=None,truncation_values=None,
               episode_starts=None,gamma=PILOT_GAMMA,gae_lambda=PILOT_GAE_LAMBDA):
    """Time-major scalar[T,B]; real terminal zero-bootstrap, rollout cut not terminal."""
    if rewards.ndim!=2 or not rewards.shape[0] or values.shape!=rewards.shape or bootstrap_values.shape!=rewards.shape[1:]:
        raise ValueError('Scalar rewards/values[T,B] and bootstrap[B] required')
    if not 0<=gamma<=1 or not 0<=gae_lambda<=1:raise ValueError('Gamma/GAE lambda must lie in[0,1]')
    if terminated.shape!=rewards.shape or terminated.dtype!=torch.bool:raise ValueError('True terminal[T,B] mask required')
    truncated=torch.zeros_like(terminated) if truncated is None else truncated
    if truncated.shape!=rewards.shape or truncated.dtype!=torch.bool:raise ValueError('Truncation mask shape differs')
    if episode_starts is not None and (episode_starts.shape!=rewards.shape or episode_starts.dtype!=torch.bool
            or bool((episode_starts[1:]&~(terminated|truncated)[:-1]).any())):
        raise ValueError('New episode without true terminal or explicit external truncation')
    if bool((truncated&~terminated).any()) and truncation_values is None:
        raise ValueError('External truncation needs actual final-observation critic value')
    r,v,boot=(x.detach().double() for x in (rewards,values,bootstrap_values))
    if not all(bool(torch.isfinite(x).all()) for x in (r,v,boot)):raise FloatingPointError('Nonfinite scalar GAE input')
    final_values=None if truncation_values is None else truncation_values.detach().double()
    if final_values is not None and (final_values.shape!=v.shape or not bool(torch.isfinite(final_values).all())):
        raise ValueError('Actual truncation values[T,B] required')
    advantages=torch.empty_like(v);tail=torch.zeros_like(boot)
    for t in range(len(r)-1,-1,-1):
        following=boot if t==len(r)-1 else v[t+1]
        if final_values is not None:following=torch.where(truncated[t]&~terminated[t],final_values[t],following)
        delta=r[t]-v[t]+gamma*torch.where(terminated[t],torch.zeros_like(following),following)
        tail=delta+gamma*gae_lambda*(~(terminated[t]|truncated[t])).to(v.dtype)*tail
        advantages[t]=tail
    return advantages,advantages+v
