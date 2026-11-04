"""Future opt-in graph bridge;one pre-Adam attempt,otherwise original pool PPO.

No CUDA/model/native work occurs on import. Numerical backend acceptance is
separate from learning quality. After any Adam, a graph error aborts normally.
"""
from __future__ import annotations

from copy import deepcopy
import gc
import math
import traceback

import torch

from .fly_visual_online_pool import update_pool,WORKER_GROUPS,_pad
from .fly_visual_ppo_loss import scalar_gae
from .fly_visual_ppo_update import VisualPPOUpdateConfig
from .fly_visual_ppo_cuda_graph import block_inputs,make_training_capture
from .whole_brain_ppo import clip_gradients_fp64

LIMITS = dict(loss_atol=2e-7,loss_rtol=2e-6,gradient_atol=2e-7,gradient_RMS_rtol=2e-5,
              logp_atol=2e-6,full_following_state='all fields exact,including floats/delay/RNG')
ZERO_COSTS = dict(setup_neural_lane_forwards=0,setup_backward_attempts=0,setup_backwards_completed=0,
    replay_neural_lane_forwards=0,replay_backward_attempts=0,replay_backwards_completed=0,
    reference_neural_lane_forwards=0,reference_backward_attempts=0,reference_backwards_completed=0)


def numerical_acceptance(proof,loss_values,old_logp_tolerance):
    """Predeclared max-absolute/RMS-scaled contract,not a teacher/PPO score gate."""
    reasons=[]
    if proof.get('optimizer_steps')!=0 or proof.get('collector_state_committed') is not False:
        reasons.append('comparison must precede Adam and cannot commit collector state')
    if proof.get('all_following_state_fields_exact') is not True:
        reasons.append('complete following state differs')
    for key,error in proof.get('loss_abs_error',{}).items():
        value=loss_values.get(key,float('nan'));scale=max(1.,abs(value))
        if not math.isfinite(error) or not math.isfinite(value) or error>LIMITS['loss_atol']+LIMITS['loss_rtol']*scale:
            reasons.append('loss:'+key)
    if not proof.get('loss_abs_error') or not proof.get('gradient_errors'):
        reasons.append('missing actual loss or gradient comparison')
    for item in proof.get('gradient_errors',[]):
        bound=LIMITS['gradient_atol']+LIMITS['gradient_RMS_rtol']*item['reference_RMS']
        if not math.isfinite(item['max_abs']) or not math.isfinite(bound) or item['max_abs']>bound:
            reasons.append('gradient:'+item['name'])
    for key,bound in (('logp_max_abs_error',LIMITS['logp_atol']),
                      ('unchanged_policy_old_logp_max_abs_error',old_logp_tolerance)):
        error=proof.get(key,float('nan'))
        if not math.isfinite(error) or error>bound:reasons.append(key)
    return not reasons,reasons


class GraphPoolUpdater:
    def __init__(self,*,enabled=False):
        if type(enabled) is not bool:raise ValueError('Explicit boolean graph selection required')
        self.requested=enabled;self.capture=None;self.attempted=False;self.new_optimizer_steps=0
        self.status='unattempted' if enabled else 'disabled_by_default'
        self.proof=None;self.reason=None;self.costs=dict(ZERO_COSTS);self.unknown_setup_cost=False

    @staticmethod
    def _guard(check_stop):
        def guard():
            try:check_stop()
            except Exception as error:
                error.training_guard_stop=True
                raise
        return guard

    def _update_costs(self):
        if self.capture is not None:
            self.costs.update({k:self.capture.costs[k] for k in ZERO_COSTS})

    def _attempt(self,model,data,before,cfg,guard):
        """Exactly one same-job,first-real-block attempt;no optimizer available."""
        if self.attempted or self.new_optimizer_steps:
            raise RuntimeError('Training graph may be attempted only before the first new Adam')
        self.attempted=True
        versions=tuple(p._version for p in model.parameters())
        try:
            self.capture=make_training_capture(model,data,before,cfg,enabled=True,check_stop=guard)
            self.proof=self.capture.compare_with_eager(data,before,check_stop=guard)
            if tuple(p._version for p in model.parameters())!=versions:
                error=RuntimeError('Capture/reference changed parameter values before Adam')
                error.training_guard_stop=True
                raise error
            loss_values={k:float(v.detach()) for k,v in self.capture.output[0].items()}
            valid,reasons=numerical_acceptance(self.proof,loss_values,cfg.old_logp_tolerance)
            self._update_costs()
            if valid:
                self.status='enabled_after_first_real_block_comparison'
                return True
            self.reason=dict(kind='numerical_backend_mismatch',reasons=reasons)
        except Exception as error:
            self._update_costs()
            diagnostics=getattr(error,'cuda_graph_setup_diagnostics',None)
            if diagnostics is not None:self.costs.update({k:diagnostics.get(k,0) for k in ZERO_COSTS})
            elif self.capture is None:self.unknown_setup_cost=True
            if getattr(error,'training_guard_stop',False):
                self.status='stopped_by_guard_before_first_Adam'
                error.ppo_update_diagnostics=dict(optimizer_updates=0,backward_passes=0,
                    actor_replay_neural_lane_forwards=0,training_graph=self._metadata(dict(ZERO_COSTS),0))
                raise
            self.reason=dict(kind=type(error).__name__,error=str(error),setup_costs=deepcopy(diagnostics))
            # A failed constructor can be retained by its traceback/partial graph.
            traceback.clear_frames(error.__traceback__);error.__traceback__=None
        self.status='disabled_before_first_Adam_same_rollout_eager'
        if self.capture is not None:
            self.capture.close();self.capture=None
        model.zero_grad(set_to_none=True)
        gc.collect()
        if model.device.type=='cuda':torch.cuda.empty_cache()
        guard()  # Resource/STOP failures remain real stops,never performance fallbacks.
        return False

    def _metadata(self,before_costs,committed_graph_blocks):
        self._update_costs()
        delta={k:self.costs[k]-before_costs[k] for k in ZERO_COSTS}
        discarded=max(0,delta['replay_neural_lane_forwards']-16*committed_graph_blocks)
        extra=delta['setup_neural_lane_forwards']+delta['reference_neural_lane_forwards']+discarded
        extra_backs=(delta['setup_backwards_completed']+delta['reference_backwards_completed']
                     +max(0,delta['replay_backwards_completed']-committed_graph_blocks))
        return dict(requested=self.requested,status=self.status,attempted_once=self.attempted,
            acceptance_limits=deepcopy(LIMITS),first_real_block_proof=deepcopy(self.proof),fallback_reason=deepcopy(self.reason),
            actual_cost_delta=delta,actual_cost_total=deepcopy(self.costs),unknown_partial_setup_cost=self.unknown_setup_cost,
            discarded_replay_neural_lane_forwards=discarded,extra_neural_lane_forwards=extra,
            extra_backwards_completed=extra_backs,first_compared_gradient_reused=committed_graph_blocks>0 and before_costs['replay_neural_lane_forwards']==0,
            no_additional_native_game=True,no_data_or_quality_rejection=True)

    def __call__(self,model,optimizer,rollout,cfg=VisualPPOUpdateConfig(),*,check_stop=lambda:None):
        guard=self._guard(check_stop);before_costs=dict(self.costs)
        if not self.requested or self.status.startswith('disabled'):
            return self._eager(model,optimizer,rollout,cfg,guard,before_costs)
        if cfg.checkpoint_replay:
            raise ValueError('Training graph variant excludes checkpoint replay,auxiliary and accumulation variants')
        from .fly_visual_actor_critic import VisualFlyActorCritic
        if type(model) is not VisualFlyActorCritic:
            raise ValueError('Only the original typed actor+critic is supported')
        steps=len(rollout.actions)
        if steps%4 or rollout.actions.shape!=(steps,30) or rollout.truncated.any():
            raise ValueError('Original complete30-world/TBPTT4 natural rollout required')
        advantage,returns=scalar_gae(rollout.rewards,rollout.values,rollout.bootstrap_values,rollout.terminated,
            truncated=rollout.truncated,episode_starts=rollout.episode_starts,gamma=cfg.gamma,gae_lambda=cfg.gae_lambda)
        normalized=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-8)
        actor=list(model.brain.parameters());critic=list(model.critic.parameters());expected=actor+critic
        actual=[p for g in optimizer.param_groups for p in g['params']]
        if (len(actual)!=len(expected) or len({id(p) for p in actual})!=len(actual)
                or {id(p) for p in actual}!={id(p) for p in expected} or any(not p.requires_grad for p in expected)
                or any(p.requires_grad for p in model.student.sensory_adapter.perception.parameters())):
            raise ValueError('One unchanged actor/critic optimizer required')
        def inputs(start,group):
            micro=rollout.microbatch(start,group);workers=WORKER_GROUPS[group]
            adv=torch.stack([_pad(row) for row in normalized[start:start+4,list(workers)]])
            ret=torch.stack([_pad(row) for row in returns[start:start+4,list(workers)]])
            return block_inputs(micro,normalized_advantages=adv,returns=ret),micro['block_state']
        if not self.attempted:
            data,before=inputs(0,0)
            if not self._attempt(model,data,before,cfg,guard):
                return self._eager(model,optimizer,rollout,cfg,guard,before_costs)
            use_compared_first=True
        else:use_compared_first=False
        record=dict(optimizer_updates=0,finite_optimizer_updates=0,optimizer_step_attempts=0,
            optimizer_transaction_complete=True,backward_passes=0,real_rollout_rows=steps*30,
            trained_real_rows=0,padding_loss_rows=0,actor_replay_brain_batch_calls=0,
            checkpoint_recompute_brain_batch_calls=0,checkpoint_replay=False,
            old_logp_check_performed=use_compared_first,old_logp_replay_max_abs_error=getattr(model,'_initial_ppo_logp_max_abs_error',None),
            normalized_advantage_mean=float(normalized.mean()),normalized_advantage_std=float(normalized.std(unbiased=False)),
            advantage_normalization_real_rows=steps*30,terms=[],gradient_diagnostics=[],epoch_count=1,tbptt=4,
            update_schedule='time block then worker group;one Adam per maskedB4 block',
            history='same recorded behavior snapshots;no collector state committed;no native bitwise resume')
        was_training=model.training;model.train();committed_blocks=0
        try:
            for start in range(0,steps,4):
                for group,workers in enumerate(WORKER_GROUPS):
                    guard();optimizer.zero_grad(set_to_none=True)
                    data,before=inputs(start,group)
                    if use_compared_first:
                        self.capture.assign_last_gradients()
                        terms={k:v.detach().clone() for k,v in self.capture.output[0].items()}
                        error=self.proof['unchanged_policy_old_logp_max_abs_error']
                        model._initial_ppo_logp_verified=True;model._initial_ppo_logp_max_abs_error=error
                        record['old_logp_replay_max_abs_error']=error;use_compared_first=False
                    else:
                        replay=self.capture.replay(data,before,assign_gradients=True,check_stop=guard)
                        terms=replay['terms'];del replay
                    record['actor_replay_brain_batch_calls']+=4;record['backward_passes']+=1
                    gradient={name:dict(RMS=None if p.grad is None else float(p.grad.detach().double().square().mean().sqrt()),
                        nonzero=None if p.grad is None else int(torch.count_nonzero(p.grad))) for name,p in model.brain.named_parameters()}
                    gradient.update(actor_norm_before_clip=clip_gradients_fp64(actor,cfg.actor_max_grad_norm),
                                    critic_norm_before_clip=clip_gradients_fp64(critic,cfg.critic_max_grad_norm))
                    record['gradient_diagnostics'].append(gradient)
                    guard();record['optimizer_step_attempts']+=1;record['optimizer_transaction_complete']=False
                    optimizer.step();record['optimizer_updates']+=1;self.new_optimizer_steps+=1
                    if model.device.type=='cuda':torch.cuda.synchronize(model.device)
                    if any(not bool(torch.isfinite(p).all()) for p in expected):raise FloatingPointError('Nonfinite parameter after graph-backed Adam')
                    record['finite_optimizer_updates']+=1;record['optimizer_transaction_complete']=True
                    committed_blocks+=1;real=int(data['valid_mask'].sum());record['trained_real_rows']+=real
                    record['terms'].append(dict(start=start,group=group,real_rows=real,**{k:float(v) for k,v in terms.items()}))
                    del terms,data,before
            return record
        except BaseException as error:
            error.ppo_update_diagnostics=record
            raise
        finally:
            record['actor_replay_neural_lane_forwards']=record['actor_replay_brain_batch_calls']*4
            record['checkpoint_recompute_neural_lane_forwards']=0
            record['training_graph']=self._metadata(before_costs,committed_blocks)
            optimizer.zero_grad(set_to_none=True);model.train(was_training)

    def _eager(self,model,optimizer,rollout,cfg,guard,before_costs):
        try:
            result=update_pool(model,optimizer,rollout,cfg,check_stop=guard)
            self.new_optimizer_steps+=result['optimizer_updates']
            result['training_graph']=self._metadata(before_costs,0)
            return result
        except BaseException as error:
            detail=getattr(error,'ppo_update_diagnostics',{})
            self.new_optimizer_steps+=detail.get('optimizer_updates',0)
            detail['training_graph']=self._metadata(before_costs,0)
            error.ppo_update_diagnostics=detail
            raise

    def close(self):
        self._update_costs()
        if self.capture is not None:self.capture.close();self.capture=None
