"""Independent30-world B4/B8 collector and real-row accumulated PPO updater.

Nothing starts on import. Old active modules/parameter deployment schemas are
unchanged. Sampling and replay must use the same explicitly selected width;
no B4/B8 numerical or trajectory bitwise equivalence is claimed.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import numpy as np
import torch

from .fly_visual_online_pool import VisualOnlinePool as _OriginalPool, VisualPoolRollout as _OriginalRollout, make_pool_native_vector
from .fly_visual_native_batch import pack_whole_brain_states, unpack_whole_brain_state
from .fly_visual_online_collector import copy_brain_state
from .fly_visual_ppo_loss import categorical_policy, scalar_gae, visual_ppo_loss
from .fly_visual_ppo_update import VisualPPOUpdateConfig, _checkpoint_forward, _clone_state
from .thrl_inputs import SEPARATED_LASER_INPUT
from .thrl_policy_actions import mask_unavailable_bomb, policy_logits
from .whole_brain import WholeBrainState
from .whole_brain_ppo import clip_gradients_fp64

WORLD_COUNT = 30


def validate_neural_batch_size(value):
    if type(value) is not int or value not in (4, 8):
        raise ValueError('Explicit neural batch4 or8 only;default4')
    return value


def worker_groups(neural_batch_size):
    width = validate_neural_batch_size(neural_batch_size)
    return tuple(tuple(range(i,min(i+width,WORLD_COUNT))) for i in range(0,WORLD_COUNT,width))


def _padded_state(model,real_states,neural_batch_size):
    width = validate_neural_batch_size(neural_batch_size)
    lanes = list(real_states)
    if not 1 <= len(lanes) <= width:
        raise ValueError('One to configured-width real complete states required')
    lanes.extend(model.initial_state(1,rng_seed=0) for _ in range(width-len(lanes)))
    return pack_whole_brain_states(lanes)


def _pad(value,neural_batch_size,*,fill=0):
    width = validate_neural_batch_size(neural_batch_size)
    if not 1 <= len(value) <= width:
        raise ValueError('Existing real neural lanes required')
    if len(value)==width:return value
    return torch.cat((value,value.new_full((width-len(value),*value.shape[1:]),fill)),0)


@dataclass
class AccumVisualPoolRollout(_OriginalRollout):
    neural_batch_size:int=4

    def microbatch(self,start,group):
        width=validate_neural_batch_size(self.neural_batch_size)
        groups=worker_groups(width)
        if start%4 or not 0<=start<len(self.actions) or not 0<=group<len(groups):
            raise ValueError('Existing TBPTT4 time block/neural group required')
        workers=groups[group];end=min(start+4,len(self.actions))
        valid=torch.arange(width).lt(len(workers)).expand(end-start,-1).clone()
        result={}
        for name in ('encoded_visual','features','legal_actions','actions','old_log_probs','values',
                     'teacher_probs','episode_starts','brain_rng_keys','noise_counters'):
            value=getattr(self,name)[start:end,list(workers)]
            result[name]=torch.stack([_pad(row,width,fill=(name=='legal_actions')) for row in value])
        result['teacher_probs'][:,len(workers):]=1./19
        result['valid_mask']=valid
        result['block_state']=self.block_states[(start,group)]
        return result


class AccumVisualOnlinePool(_OriginalPool):
    """Same owned native/teacher lifecycle; configurable neural microbatch only."""
    def __init__(self,*args,neural_batch_size=4,**kwargs):
        width=validate_neural_batch_size(neural_batch_size)
        super().__init__(*args,**kwargs)
        capture=self.model.brain._captured_inference
        if capture is not None and capture.batch_size!=width:
            raise ValueError('Existing capture width differs;no silent cross-batch reuse')
        self._neural_batch_size=width
        self.worker_groups=worker_groups(width)

    @property
    def neural_batch_size(self):return self._neural_batch_size

    def enable_sampling_capture(self):
        if self.synthetic_fixture or self.model.device.type!='cuda' or self.sampling_capture is not None:
            raise ValueError('Explicit production CUDA sampling capture only once')
        self.model.brain.enable_cuda_capture(self.neural_batch_size)
        self.sampling_capture=self.model.brain._captured_inference.specification()
        return dict(self.sampling_capture)

    @torch.no_grad()
    def collect(self, *, steps=32, policy_version=0, check_stop=lambda: None):
        if (type(steps) is not int or not 1 <= steps <= 32 or type(policy_version) is not int or policy_version < 0
                or self.observation is None):
            raise ValueError('Started pool, finite1..32 steps and explicit policy version required')
        rows, blocks, components, completed, info_rows = [], {}, [], [], []
        was_training = self.model.training
        self.model.eval()
        stale = self.last_policy_version is not None and self.last_policy_version != policy_version
        try:
            for time in range(steps):
                check_stop()
                f, m, o = self._modalities()
                encoded = self.model.encode_visual(f, m, objects=o)
                if encoded.shape != (30, 384) or encoded.dtype != torch.float32 or not bool(torch.isfinite(encoded).all()):
                    raise ValueError('Batched fixed FP32 visual384 required')
                raw_rows, value_rows, key_rows, counter_rows = [], [], [], []
                following_states = [None] * 30
                for group, workers in enumerate(self.worker_groups):
                    check_stop()
                    before = _padded_state(self.model, [self.states[i] for i in workers], self.neural_batch_size)
                    if time % 4 == 0:
                        blocks[(time, group)] = copy_brain_state(before)
                    raw, value, following = self.model.forward_encoded(_pad(encoded[list(workers)], self.neural_batch_size), before)
                    self.data_neural_lane_forwards += self.neural_batch_size
                    if raw.shape != (self.neural_batch_size, 19) or value.shape != (self.neural_batch_size,) or not bool(torch.isfinite(value).all()):
                        raise ValueError('Configured neural-batch source19 actor/scalar values required')
                    n = len(workers)
                    raw_rows.append(raw[:n]); value_rows.append(value[:n])
                    key_rows.append(before.rng_keys[:n]); counter_rows.append(before.noise_counter[:n])
                    for worker, state in zip(workers, unpack_whole_brain_state(following, count=n)):
                        following_states[worker] = state
                raw, value = torch.cat(raw_rows), torch.cat(value_rows)
                legal_logits = policy_logits(mask_unavailable_bomb(raw, f, SEPARATED_LASER_INPUT))
                distribution = torch.distributions.Categorical(logits=legal_logits)
                # Exactly30 real draws; no dummy generator/draw exists.
                action = torch.stack([torch.multinomial(distribution.probs[i], 1, generator=self.action_generators[i])[0]
                                      for i in range(30)])
                teacher_raw, _, teacher_next = self.teacher.forward_step(f, m, self.teacher_hidden, objects=o)
                self.teacher_query_rows += 30
                teacher_probs = torch.softmax(policy_logits(mask_unavailable_bomb(teacher_raw, f, SEPARATED_LASER_INPUT)), -1)
                if (teacher_probs.shape != (30, 19) or teacher_next.shape != self.teacher_hidden.shape
                        or not bool(torch.isfinite(teacher_probs).all()) or not bool(torch.isfinite(teacher_next).all())):
                    raise ValueError('Independent finite teacher final19 and own continuous GRU required')
                record = dict(encoded_visual=encoded, features=f, legal_actions=torch.isfinite(legal_logits), actions=action,
                    old_log_probs=distribution.log_prob(action), values=value, teacher_probs=teacher_probs,
                    action_probs=distribution.probs, raw_logits=raw,
                    episode_starts=torch.from_numpy(self.starts.copy()), game_ids=torch.tensor(self.game_ids),
                    native_episodes=torch.tensor([i['native_episode'] for i in self.infos]),
                    brain_rng_keys=torch.cat(key_rows), noise_counters=torch.cat(counter_rows))
                record = {k: v.detach().cpu().clone() for k, v in record.items()}
                self.states, self.teacher_hidden = following_states, teacher_next.detach().clone()
                self.pending_actions = 30
                after, reward, terminated, truncated, infos = self.vector.step(action.cpu().tolist())
                self.acknowledged_actions += 30
                self.pending_actions = 0
                reward = np.asarray(reward, dtype=np.float32)
                terminated, truncated = np.asarray(terminated, dtype=bool), np.asarray(truncated, dtype=bool)
                if (reward.shape != (30,) or terminated.shape != (30,) or truncated.shape != (30,)
                        or len(infos) != 30 or not np.isfinite(reward).all() or truncated.any()):
                    raise ValueError('Finite native rewards/terminals only; no external truncation')
                vectors, raw_vectors, row_components, row_infos = [], [], [], []
                for worker, info in enumerate(infos):
                    if info.get('policy_action') != int(action[worker]) or info.get('reward_mapping_complete') is not True:
                        raise ValueError('Native acknowledgement must match the sampled student action')
                    v, r = (np.asarray(info[k], dtype=np.float32) for k in ('reward_vector', 'reward_raw_vector'))
                    c = dict(info['reward_components_raw'])
                    if (v.shape != (3,) or r.shape != (3,) or not np.isfinite(v).all() or not np.isfinite(r).all()
                            or not all(np.isfinite(float(x)) for x in c.values())
                            or not np.isclose(reward[worker], np.sum(v, dtype=np.float32), rtol=1e-6, atol=1e-6)):
                        raise ValueError('Original scalar=sum3 and raw reward evidence required')
                    vectors.append(v.copy()); raw_vectors.append(r.copy()); row_components.append(c)
                    row_infos.append({k: info.get(k) for k in ('stage', 'stage_frame', 'misses', 'bombs_used',
                        'elapsed_input_ticks', 'native_terminal_kind', 'observation_native_frame', 'observation_source', 'native_action')})
                    diagnostics = info.get('reward_diagnostics') or {}
                    row_infos[-1].update(protected_decision=diagnostics.get('protected_decision'),
                                         vulnerable_frames=diagnostics.get('vulnerable_frames'))
                    if terminated[worker]:
                        summary = dict(info['episode_summary'])
                        if summary.get('outcome') not in ('gameover', 'clear') or summary.get('full_story_episode') is not True:
                            raise ValueError('Only natural native GO/clear completes a game')
                        completed.append(dict(worker=worker, game_id=self.game_ids[worker], summary=summary))
                record.update(rewards=torch.from_numpy(reward.copy()), reward_vectors=torch.from_numpy(np.stack(vectors)),
                    reward_raw_vectors=torch.from_numpy(np.stack(raw_vectors)), terminated=torch.from_numpy(terminated.copy()),
                    truncated=torch.from_numpy(truncated.copy()))
                rows.append(record); components.append(tuple(row_components)); info_rows.append(tuple(row_infos))
                self.observation, self.infos, self.starts = after, list(infos), np.zeros(30, dtype=bool)
                done = np.flatnonzero(terminated).tolist()
                if done:
                    reset_infos = self.vector.reset_at(done)
                    self.observation = self.vector.observation
                    self._fresh(done, reset_infos)
                    for worker, info in zip(done, reset_infos):
                        self.infos[worker] = info
            check_stop()
            bootstrap = torch.zeros(30)
            ongoing = np.flatnonzero(~terminated).tolist()
            if ongoing:
                f, m, o = self._modalities(ongoing)
                estimate = self.model.value_encoded(self.model.encode_visual(f, m, objects=o))
                if estimate.shape != (len(ongoing),) or not bool(torch.isfinite(estimate).all()):
                    raise ValueError('Finite critic-only rollout bootstrap required')
                bootstrap[ongoing] = estimate.cpu()
                self.bootstrap_value_rows += len(ongoing)
            snapshot_bytes = sum(t.numel() * t.element_size() for s in blocks.values() for field in fields(WholeBrainState)
                for t in (getattr(s, field.name) if field.name == 'delay_queue' else (getattr(s, field.name),)))
            self.last_policy_version = policy_version
            return AccumVisualPoolRollout(neural_batch_size=self.neural_batch_size, **{k: torch.stack([r[k] for r in rows]) for k in rows[0]},
                reward_components=tuple(components), block_states=blocks, bootstrap_values=bootstrap,
                completed_episodes=tuple(completed), step_info=tuple(info_rows), metadata=dict(
                    schema='fly-visual-source19-online30-accum-B4-B8-pool-v1', worlds=30, steps=steps, policy_version=policy_version,
                    neural_batch_size=self.neural_batch_size, worker_groups=[list(g) for g in self.worker_groups],
                    same_neural_batch_in_collection_and_replay=True, B4_B8_bitwise_equivalence_claimed=False,
                    synthetic_fixture=self.synthetic_fixture, source_teacher_sha256=self.teacher_source_sha256,
                    brain_seed=self.brain_seed, action_seed=self.action_seed, action_temperature=1.,
                    student_actions_executed=steps * 30, teacher_actions_executed=0, teacher_queries=steps * 30,
                    perception_data_batches=steps, teacher_data_batches=steps, brain_data_batches=steps * len(self.worker_groups),
                    action_probs_logged_from_actual_forward=True, extra_brain_forward_for_logging=0,
                    data_neural_lane_forwards=steps * 32, real_neural_lane_forwards=steps * 30, padding_lane_forwards=steps * 2,
                    action_random_draws=steps * 30, bootstrap_neural_lane_forwards=0, teacher_bootstrap_queries=0,
                    bootstrap_value_forward_rows=len(ongoing), state_snapshot_bytes=snapshot_bytes,
                    sampling_capture=self.sampling_capture, state_carry_crossed_parameter_version=stale,
                    state_carry_semantics='own behavior history; changed-parameter retained-state continuation is approximate',
                    reset='new natural game only; no Miss/stage/rollout/update reset', PPO_optimizer_updates=0,
                    teacher_hidden_or_value_in_student=False, padding_outputs_and_states_committed=False))
        finally:
            self.model.train(was_training)


@dataclass(frozen=True)
class AccumVisualPPOUpdateConfig(VisualPPOUpdateConfig):
    neural_batch_size:int=4

    def __post_init__(self):
        super().__post_init__()
        validate_neural_batch_size(self.neural_batch_size)


def update_pool_accum(model,optimizer,rollout,cfg=AccumVisualPPOUpdateConfig(),*,check_stop=lambda:None):
    """Each TBPTT4 time block: sum(real_rows/120 * loss), clip, one Adam.

    Backward is paid per neural microbatch; no graph is retained across groups.
    A partial accumulation never clips/steps. DN-return auxiliary is disabled;
    an optional registered auxiliary head can only have None gradients here.
    """
    from .fly_visual_actor_critic import VisualFlyActorCritic
    width=validate_neural_batch_size(cfg.neural_batch_size)
    groups=worker_groups(width)
    if not isinstance(model,VisualFlyActorCritic) or not isinstance(rollout,AccumVisualPoolRollout):
        raise ValueError('Typed shared actor/critic and independent accumulated-pool rollout required')
    if (rollout.neural_batch_size!=width or rollout.metadata.get('neural_batch_size')!=width
            or rollout.metadata.get('same_neural_batch_in_collection_and_replay') is not True):
        raise ValueError('Collection and replay must use the same declared neural batch width')
    capture=model.brain._captured_inference
    if capture is not None and capture.batch_size!=width:
        raise ValueError('A capture of another neural width cannot be reused')
    encoded=rollout.encoded_visual
    if encoded.ndim!=3 or encoded.shape[1:]!=(30,384) or not 1<=len(encoded)<=32 or len(encoded)%4:
        raise ValueError('Complete T<=32 divisible by4 and30 real visual384 worlds required')
    device,steps=model.device,len(encoded)
    for name in ('actions','old_log_probs','values','rewards','terminated','truncated',
                 'episode_starts','brain_rng_keys','noise_counters','game_ids'):
        if getattr(rollout,name).shape!=(steps,30):raise ValueError('World/time alignment differs:'+name)
    if (rollout.teacher_probs.shape!=(steps,30,19) or rollout.legal_actions.shape!=(steps,30,19)
            or rollout.truncated.any()):
        raise ValueError('Original legal source19 teacher labels and natural terminals required')
    if set(rollout.block_states)!={(start,group) for start in range(0,steps,4) for group in range(len(groups))}:
        raise ValueError('Every recorded own-state TBPTT boundary is required exactly once')
    if any(s.membrane.shape[0]!=width for s in rollout.block_states.values()):
        raise ValueError('Recorded neural state batch differs from collection/replay width')
    advantage,returns=scalar_gae(rollout.rewards,rollout.values,rollout.bootstrap_values,rollout.terminated,
        truncated=rollout.truncated,episode_starts=rollout.episode_starts,gamma=cfg.gamma,gae_lambda=cfg.gae_lambda)
    normalized=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-8)
    actor=list(model.student.brain.parameters());critic=list(model.critic.parameters());expected=actor+critic
    actual=[p for group in optimizer.param_groups for p in group['params']]
    inactive_head=getattr(model,'return_aux_head',None)
    inactive_aux=list(inactive_head.parameters()) if isinstance(inactive_head,torch.nn.Module) else []
    ids={id(p) for p in actual};base_ids={id(p) for p in expected}
    if (len(ids)!=len(actual) or ids not in (base_ids,base_ids|{id(p) for p in inactive_aux})
            or any(not p.requires_grad for p in expected)
            or any(p.requires_grad for p in model.student.sensory_adapter.perception.parameters())):
        raise ValueError('Optimizer owns actor/critic only, plus an optional inactive DN-return head')
    record=dict(schema='fly-visual-PPO30-real120-accum-B4-B8-update-v1',neural_batch_size=width,
        optimizer_updates=0,finite_optimizer_updates=0,optimizer_step_attempts=0,optimizer_transaction_complete=True,
        backward_passes=0,real_rollout_rows=steps*30,trained_real_rows=0,loss_real_rows=0,padding_loss_rows=0,
        actor_replay_brain_batch_calls=0,checkpoint_recompute_brain_batch_calls=0,
        checkpoint_replay=cfg.checkpoint_replay,epoch_count=1,tbptt=4,terms=[],gradient_diagnostics=[],
        teacher_coefficient=cfg.loss.teacher_coeff,return_auxiliary_coefficient=0.,return_auxiliary_head_calls=0,
        old_logp_check_performed=False,old_logp_replay_max_abs_error=getattr(model,'_accum_initial_ppo_logp_error',None),
        normalized_advantage_mean=float(normalized.mean()),normalized_advantage_std=float(normalized.std(unbiased=False)),
        advantage_normalization_real_rows=steps*30,accumulated_real_rows_in_current_timeblock=0,
        history='unchanged recorded complete behavior state per TBPTT4 boundary;no changed-parameter full-history reconstruction',
        update_schedule='all neural groups in each TBPTT4 block,real_rows/120,then one clip/Adam')
    sums={};was_training=model.training;model.train()
    try:
        for start in range(0,steps,4):
            check_stop()
            optimizer.zero_grad(set_to_none=True)
            expected_rows=4*WORLD_COUNT;accumulated=0
            record['accumulated_real_rows_in_current_timeblock']=0
            for group,workers in enumerate(groups):
                check_stop()
                micro=rollout.microbatch(start,group)
                data={k:v.to(device) for k,v in micro.items() if torch.is_tensor(v)}
                state=_clone_state(micro['block_state'],device)
                logits_rows,value_rows=[],[]
                for t in range(4):
                    lanes=list(unpack_whole_brain_state(state))
                    for lane in range(width):
                        if lane>=len(workers):lanes[lane]=model.initial_state(1,rng_seed=0)
                        elif bool(data['episode_starts'][t,lane]):
                            lanes[lane]=model.initial_state(1,rng_seed=int(data['brain_rng_keys'][t,lane]))
                    state=pack_whole_brain_states(lanes);n=len(workers)
                    if (not torch.equal(state.rng_keys[:n],data['brain_rng_keys'][t,:n])
                            or not torch.equal(state.noise_counter[:n],data['noise_counters'][t,:n])):
                        raise ValueError('Own game keys/counters changed during neural replay')
                    record['actor_replay_brain_batch_calls']+=1
                    if cfg.checkpoint_replay:raw,value,state=_checkpoint_forward(model,data['encoded_visual'][t],state,record)
                    else:raw,value,state=model.forward_encoded(data['encoded_visual'][t],state)
                    if not torch.equal(state.noise_counter[:n],data['noise_counters'][t,:n]+100):
                        raise ValueError('Every neural action must advance100 original microsteps')
                    logits_rows.append(raw);value_rows.append(value)
                logits,values=torch.stack(logits_rows),torch.stack(value_rows);valid=data['valid_mask']
                if cfg.verify_initial_log_probs and getattr(model,'_accum_initial_ppo_logp_verified_batch',None)!=width:
                    error=float((categorical_policy(logits,data['legal_actions']).log_prob(data['actions'])
                        -data['old_log_probs'])[valid].detach().abs().max())
                    record.update(old_logp_check_performed=True,old_logp_replay_max_abs_error=error)
                    if error>cfg.old_logp_tolerance:raise ValueError('Initial same-width old-logp replay mismatch:'+str(error))
                    model._accum_initial_ppo_logp_verified_batch=width
                    model._accum_initial_ppo_logp_error=error
                adv=torch.stack([_pad(row,width) for row in normalized[start:start+4,list(workers)]]).to(device)
                ret=torch.stack([_pad(row,width) for row in returns[start:start+4,list(workers)]]).to(device)
                terms=visual_ppo_loss(logits,values,data['actions'],data['old_log_probs'],data['values'],adv,ret,
                    data['teacher_probs'],data['legal_actions'],model.brain.recurrent_gain_logits,
                    config=cfg.loss,valid_mask=valid)
                real_rows=int(valid.sum())
                if real_rows!=4*len(workers):raise ValueError('Only real rows may contribute to accumulation')
                if not bool(torch.isfinite(terms['total_loss'])):raise FloatingPointError('Nonfinite PPO/teacher/gain loss')
                weight=real_rows/expected_rows
                check_stop();(terms['total_loss']*weight).backward();record['backward_passes']+=1
                accumulated+=real_rows;record['loss_real_rows']+=real_rows
                record['accumulated_real_rows_in_current_timeblock']=accumulated
                numbers={k:float(v.detach()) for k,v in terms.items()}
                record['terms'].append(dict(start=start,group=group,real_rows=real_rows,gradient_loss_weight=weight,**numbers))
                for key,value in numbers.items():sums[key]=sums.get(key,0.)+real_rows*value
                del state,logits_rows,value_rows,logits,values,terms
            if accumulated!=expected_rows:raise ValueError('Partial120-row accumulation cannot clip or step')
            if any(p.grad is not None for p in inactive_aux):raise ValueError('DN-return auxiliary is disabled in this updater')
            check_stop()
            gradient={name:dict(RMS=None if p.grad is None else float(p.grad.detach().double().square().mean().sqrt()),
                nonzero=None if p.grad is None else int(torch.count_nonzero(p.grad))) for name,p in model.student.brain.named_parameters()}
            gradient.update(start=start,accumulated_real_rows=accumulated,
                actor_norm_before_clip=clip_gradients_fp64(actor,cfg.actor_max_grad_norm),
                critic_norm_before_clip=clip_gradients_fp64(critic,cfg.critic_max_grad_norm))
            record['gradient_diagnostics'].append(gradient)
            check_stop();record['optimizer_step_attempts']+=1;record['optimizer_transaction_complete']=False
            optimizer.step();record['optimizer_updates']+=1
            if device.type=='cuda':torch.cuda.synchronize(device)
            if any(not bool(torch.isfinite(p).all()) for p in expected):raise FloatingPointError('Nonfinite accumulated update parameter')
            record['finite_optimizer_updates']+=1;record['optimizer_transaction_complete']=True
            record['trained_real_rows']+=accumulated
            optimizer.zero_grad(set_to_none=True)
        record['weighted_terms_mean']={k:v/record['loss_real_rows'] for k,v in sums.items()}
        return record
    except BaseException as error:
        error.ppo_update_diagnostics=record
        raise
    finally:
        record['actor_replay_neural_lane_forwards']=record['actor_replay_brain_batch_calls']*width
        record['checkpoint_recompute_neural_lane_forwards']=record['checkpoint_recompute_brain_batch_calls']*width
        optimizer.zero_grad(set_to_none=True)
        model.train(was_training)

