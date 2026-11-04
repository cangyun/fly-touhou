"""One-epoch, TBPTT4 PPO update on the real collector's complete B4 states.

No collector, game or optimizer is started here. Restored behavior snapshots
are immutable. Training uses detached behavior history at each block boundary;
it does not claim to reconstruct an entire changed-parameter episode history.
"""
from __future__ import annotations

from dataclasses import dataclass,field,fields
import math

import torch
from torch.utils.checkpoint import checkpoint,set_checkpoint_early_stop

from .fly_visual_actor_critic import VisualFlyActorCritic
from .fly_visual_ppo_loss import VisualPPOLossConfig,scalar_gae,categorical_policy,visual_ppo_loss
from .fly_visual_native_batch import pack_whole_brain_states,unpack_whole_brain_state
from .whole_brain import WholeBrainState
from .whole_brain_ppo import clip_gradients_fp64


@dataclass(frozen=True)
class VisualPPOUpdateConfig:
    tbptt:int=4
    epochs:int=1
    gamma:float=.999
    gae_lambda:float=.995
    actor_max_grad_norm:float=.5
    critic_max_grad_norm:float=.5
    verify_initial_log_probs:bool=True
    old_logp_tolerance:float=5e-5
    checkpoint_replay:bool=False
    loss:VisualPPOLossConfig=field(default_factory=VisualPPOLossConfig)

    def __post_init__(self):
        if self.tbptt!=4 or self.epochs!=1:raise ValueError('This finite pilot fixes TBPTT4 and one PPO epoch')
        if (not 0<=self.gamma<=1 or not 0<=self.gae_lambda<=1
                or any(not math.isfinite(x) or x<=0 for x in (self.actor_max_grad_norm,self.critic_max_grad_norm,self.old_logp_tolerance))
                or type(self.checkpoint_replay) is not bool or type(self.verify_initial_log_probs) is not bool):
            raise ValueError('Finite explicit PPO update configuration required')


def _clone_state(state,device):
    if not isinstance(state,WholeBrainState):raise ValueError('Complete behavior WholeBrainState required')
    result={}
    for f in fields(WholeBrainState):
        value=getattr(state,f.name)
        result[f.name]=tuple(x.detach().to(device).clone() for x in value) if f.name=='delay_queue' else value.detach().to(device).clone()
    return WholeBrainState(**result)


def _flatten(state):
    return (state.membrane,state.synaptic,state.spikes,state.refractory,state.rate,*state.delay_queue,
            state.rng_keys,state.noise_counter,state.window_spike_counts)


def _unflatten(parts,delay):
    return WholeBrainState(*parts[:5],tuple(parts[5:5+delay]),*parts[5+delay:])


def _checkpoint_forward(model,encoded,state,record):
    delay=len(state.delay_queue);calls=0
    def invoke(value,*parts):
        nonlocal calls
        calls+=1
        if calls>1:record['checkpoint_recompute_brain_batch_calls']+=1
        logits,values,following=model.forward_encoded(value,_unflatten(parts,delay))
        return (logits,values,*_flatten(following))
    # Nonreentrant handles frozen encoded/state inputs while model Parameters
    # require grad. Counter-based Poisson uses no global random draws.
    with set_checkpoint_early_stop(False):
        output=checkpoint(invoke,encoded,*_flatten(state),use_reentrant=False,preserve_rng_state=False)
    return output[0],output[1],_unflatten(output[2:],delay)


def update(model,optimizer,rollout,cfg=VisualPPOUpdateConfig(),*,check_stop=lambda:None):
    """Update an existing optimizer; actor and critic clip norms stay separate.

    Initial logp identity is checked in the first real training chunk, using
    that same graph for backward. It is a protocol check, not a KL-score gate.
    """
    if not isinstance(model,VisualFlyActorCritic):raise ValueError('Typed whole-brain actor/scalar-critic model required')
    device=model.device;encoded=rollout.encoded_visual
    if encoded.ndim!=3 or encoded.shape[1:]!=(4,384) or not 1<=encoded.shape[0]<=32 or encoded.shape[0]%4:
        raise ValueError('Complete time-major T<=32,B4,384 rollout divisible by4 required')
    steps=encoded.shape[0]
    names=('actions','old_log_probs','values','teacher_probs','legal_actions','rewards','terminated','truncated',
           'episode_starts','brain_rng_keys','noise_counters')
    data={name:getattr(rollout,name).detach().to(device) for name in names}
    for name,value in data.items():
        shape=(steps,4,19) if name in ('teacher_probs','legal_actions') else (steps,4)
        if value.shape!=shape:raise ValueError('Rollout source shape differs:'+name)
    if data['truncated'].any():raise ValueError('Natural pilot has no external environment truncation')
    advantage,returns=scalar_gae(data['rewards'],data['values'],rollout.bootstrap_values.to(device),data['terminated'],
        truncated=data['truncated'],episode_starts=data['episode_starts'],gamma=cfg.gamma,gae_lambda=cfg.gae_lambda)
    std=advantage.std(unbiased=False);normalized=(advantage-advantage.mean())/std.clamp_min(1e-8)
    actor=list(model.student.brain.parameters());critic=list(model.critic.parameters());expected=actor+critic
    actual=[p for group in optimizer.param_groups for p in group['params']]
    if (len(actual)!=len(expected) or len({id(p) for p in actual})!=len(actual)
            or {id(p) for p in actual}!={id(p) for p in expected} or any(not p.requires_grad for p in expected)
            or any(p.requires_grad for p in model.student.sensory_adapter.perception.parameters())):
        raise ValueError('Optimizer owns only input/bias/head/all gains and independent critic, never perception/base graph')
    record=dict(optimizer_updates=0,finite_optimizer_updates=0,optimizer_step_attempts=0,optimizer_transaction_complete=True,
        backward_passes=0,real_rollout_rows=steps*4,teacher_coefficient=cfg.loss.teacher_coeff,
        actor_clip_norm=cfg.actor_max_grad_norm,critic_clip_norm=cfg.critic_max_grad_norm,
        actor_replay_brain_batch_calls=0,checkpoint_recompute_brain_batch_calls=0,
        checkpoint_replay=cfg.checkpoint_replay,old_logp_check_performed=False,
        old_logp_replay_max_abs_error=getattr(model,'_initial_ppo_logp_max_abs_error',None),
        normalized_advantage_mean=float(normalized.mean()),normalized_advantage_std=float(normalized.std(unbiased=False)),
        gamma=cfg.gamma,gae_lambda=cfg.gae_lambda,epoch_count=1,tbptt=4,terms=[],
        history='Recorded complete behavior state per TBPTT block; no full changed-parameter history reconstruction; collector state untouched')
    gradients=[];was_training=model.training;model.train()
    try:
        for start in range(0,steps,4):
            check_stop()
            if start not in rollout.block_states:raise ValueError('Missing actual pre-block full FP32 state snapshot')
            state=_clone_state(rollout.block_states[start],device);logits_rows=[];value_rows=[]
            optimizer.zero_grad(set_to_none=True)
            for t in range(start,start+4):
                reset=data['episode_starts'][t]
                if bool(reset.any()):
                    lanes=list(unpack_whole_brain_state(state))
                    for lane in reset.nonzero().flatten().tolist():
                        lanes[lane]=model.initial_state(1,rng_seed=int(data['brain_rng_keys'][t,lane]))
                    state=pack_whole_brain_states(lanes)
                if (not torch.equal(state.rng_keys,data['brain_rng_keys'][t])
                        or not torch.equal(state.noise_counter,data['noise_counters'][t])):
                    raise ValueError('Replay must preserve per-game key and full before-state counter, no Miss/stage reset')
                z=encoded[t].to(device);record['actor_replay_brain_batch_calls']+=1
                if cfg.checkpoint_replay:logits,value,state=_checkpoint_forward(model,z,state,record)
                else:logits,value,state=model.forward_encoded(z,state)
                if not torch.equal(state.noise_counter,data['noise_counters'][t]+100):raise ValueError('Whole-brain replay must advance100 microsteps')
                logits_rows.append(logits);value_rows.append(value)
            logits=torch.stack(logits_rows);values=torch.stack(value_rows)
            if cfg.verify_initial_log_probs and not getattr(model,'_initial_ppo_logp_verified',False):
                error=float((categorical_policy(logits,data['legal_actions'][start:start+4]).log_prob(data['actions'][start:start+4])
                    -data['old_log_probs'][start:start+4]).detach().abs().max())
                record.update(old_logp_check_performed=True,old_logp_replay_max_abs_error=error)
                if error>cfg.old_logp_tolerance:raise ValueError('Initial unchanged-policy logp replay mismatch:'+str(error))
                model._initial_ppo_logp_verified=True
                model._initial_ppo_logp_max_abs_error=error
            terms=visual_ppo_loss(logits,values,data['actions'][start:start+4],data['old_log_probs'][start:start+4],
                data['values'][start:start+4],normalized[start:start+4],returns[start:start+4],
                data['teacher_probs'][start:start+4],data['legal_actions'][start:start+4],model.brain.recurrent_gain_logits,config=cfg.loss)
            if not bool(torch.isfinite(terms['total_loss'])):raise FloatingPointError('Nonfinite PPO+teacher+gain loss')
            check_stop();terms['total_loss'].backward();record['backward_passes']+=1
            gradient={name:dict(RMS=None if p.grad is None else float(p.grad.detach().double().square().mean().sqrt()),
                nonzero=None if p.grad is None else int(torch.count_nonzero(p.grad))) for name,p in model.student.brain.named_parameters()}
            actor_norm=clip_gradients_fp64(actor,cfg.actor_max_grad_norm);critic_norm=clip_gradients_fp64(critic,cfg.critic_max_grad_norm)
            gradient.update(actor_norm_before_clip=actor_norm,critic_norm_before_clip=critic_norm)
            gradients.append(gradient)
            check_stop();record['optimizer_step_attempts']+=1;record['optimizer_transaction_complete']=False
            optimizer.step();record['optimizer_updates']+=1
            if device.type=='cuda':torch.cuda.synchronize(device)
            if any(not bool(torch.isfinite(p).all()) for p in expected):raise FloatingPointError('Nonfinite parameter after PPO update')
            record['finite_optimizer_updates']+=1;record['optimizer_transaction_complete']=True
            record['terms'].append({k:float(v.detach()) for k,v in terms.items()})
            state=state.detach();del logits_rows,value_rows,logits,values,terms
        record['gradient_diagnostics']=gradients
        record['actor_replay_neural_lane_forwards']=record['actor_replay_brain_batch_calls']*4
        record['checkpoint_recompute_neural_lane_forwards']=record['checkpoint_recompute_brain_batch_calls']*4
        return record
    except BaseException as error:
        record['gradient_diagnostics']=gradients
        record['actor_replay_neural_lane_forwards']=record['actor_replay_brain_batch_calls']*4
        record['checkpoint_recompute_neural_lane_forwards']=record['checkpoint_recompute_brain_batch_calls']*4
        error.ppo_update_diagnostics=record
        raise
    finally:
        optimizer.zero_grad(set_to_none=True);model.train(was_training)
