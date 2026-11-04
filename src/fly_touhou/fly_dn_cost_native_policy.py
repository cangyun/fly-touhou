"""One DN-cost composition contract for collection, PPO and native evaluation.

Definitions only. A native policy must receive the complete new typed model,
not its bare student. Risk reads the same following DN rate; raw features are
used solely for the original normal/unprotected and Bomb inventory predicates.
"""
from __future__ import annotations

import math
import json
import time
import os
from pathlib import Path

import torch

from .fly_dn_continuous_risk_policy import compose_cost_logits
from .fly_visual_native_pool import PoolWideVisualNativePolicy, PoolReadyVector
from .thrl_vector import THRLVectorError, wait
from .fly_visual_native_batch import NativeWorkerBrain, pack_whole_brain_states, unpack_whole_brain_state
from .fly_centered_native_evaluation import native_brain_key
from .thrl_inputs import SEPARATED_LASER_INPUT
from .thrl_policy_actions import mask_unavailable_bomb, policy_logits

SCHEMA = 'fly-DN-cost64-native-policy-and-shared-T1-composition-v3'


def _beta(value):
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or not 0 <= value <= 8):
        raise ValueError('Explicit fixed risk beta in[0,8] required')
    return float(value)


def _features(raw, features):
    if (raw.shape[-1] != 19 or features.shape != (*raw.shape[:-1], 25)
            or features.device != raw.device or features.dtype != raw.dtype):
        raise ValueError('Matched source19 and original source25 features required')


def normal_unprotected_from_features(features):
    """Exact current V5 gate, not teacher-query/confidence/postfact validity."""
    if features.shape[-1] != 25:
        raise ValueError('Original V5 features required')
    return (features[..., 21] > .5) & (features[..., 20] <= 0) & (features[..., 5] <= 0)


def compose_from_active_mask(raw, cost, active_mask, *, beta, check_finite=True):
    """Pure primitive shared with head.adjust; no head or neural call here.

    The existing isolated continuous-cost primitive cancels a common cost
    before reweighting and forms the LSE difference first. A uniform zero-init
    softplus head is exactly raw identity without stopping its risk gradients.
    Cost is never clipped; the single source80 clamp happens AFTER composition.
    """
    return compose_cost_logits(raw, cost, active_mask, beta=_beta(beta), check_finite=check_finite)


def compose_from_features(head, raw, following, features, *, beta=None, check_finite=True):
    """One head call on the already-computed following state; no brain advance.

    beta=None uses head.beta. An explicit frozen-evaluation override changes
    only this operator, never the head's stored beta, parameters or source schema.
    check_finite=False is a tensor-only capture path after setup validation.
    """
    _features(raw, features)
    if type(check_finite) is not bool:
        raise ValueError('Explicit finite-check mode required')
    actual_beta = _beta(head.beta if beta is None else beta)
    if check_finite and not bool(torch.isfinite(features[..., (5,20,21)]).all()):
        raise ValueError('Finite measured normal/unprotected gate fields required')
    cost = head(following)  # ONLY followingDN is passed to the head.
    composed = compose_from_active_mask(raw, cost, normal_unprotected_from_features(features),
        beta=actual_beta, check_finite=check_finite)
    return composed, cost


def source_policy_logits(composed, features, *, check_finite=True):
    """Original stock legality + one clamp80 + T1; supports arbitrary leading dims.

    Sampling and greedy both consume this same distribution. No temperature
    override, cost-derived support, deathbomb rule or second policy is inserted.
    """
    _features(composed, features)
    if type(check_finite) is not bool:
        raise ValueError('Explicit finite-check mode required')
    if check_finite and not bool(torch.isfinite(features[..., 4]).all()):
        raise ValueError('Finite measured Bomb inventory required')
    legal = mask_unavailable_bomb(composed, features, SEPARATED_LASER_INPUT)
    if check_finite:
        return policy_logits(legal, limit=80., action_temperature=1.)
    # The original helper's finite checks are host-side; capture uses the exact
    # same tensor arithmetic after its inputs have passed those setup checks.
    return legal.clamp(-80.,80.).masked_fill(torch.isneginf(legal), -torch.inf)


class DNCostReadyVector(PoolReadyVector):
    """Original ready/ownership/timeout protocol, at most16 consumed replies.

    A minimum batch of16 does NOT cap the old vector's ready list. This static
    derivative adds a round-robin consumption cap, leaving excess commands
    unread in _inflight. No private reply queue can hide a final terminal from
    the evaluator's `while vector._inflight` loop or starve workers16..19.
    """
    def receive_ready(self, *, max_wait_ms=2, min_batch=1, check_stop=lambda: None,
                      copy_observations=True):
        if type(copy_observations) is not bool:
            raise ValueError('Explicit boolean observation-copy mode required')
        # Same fixed2ms coalescing policy as PoolReadyVector, now with max16.
        max_wait_ms=2.
        min_batch=min(16,max(1,len(self._inflight)))
        if not self._inflight or self._closed or self._failed:
            raise RuntimeError('No active ready-driven commands')
        result=[];coalesce_deadline=None;coalesce_wait=0.;window_exhausted=False
        def finish():
            self.last_receive_timing=dict(max_wait_ms=max_wait_ms,min_batch=min_batch,
                ready_count=len(result),coalesce_wait_seconds=coalesce_wait,
                window_exhausted=window_exhausted,copied_observations=copy_observations,
                reset_responses=sum(row['operation']=='reset' for row in result),
                maximum_ready_batch=16,pending_commands=len(self._inflight),
                next_ready_worker=getattr(self,'_ready_cursor',0))
            return result
        while True:
            if check_stop is not None:check_stop()
            if result:
                if len(result)>=min_batch or not self._inflight:return finish()
                if time.monotonic()>=coalesce_deadline:
                    window_exhausted=True;return finish()
                eligible=[i for i,item in self._inflight.items() if item['operation']=='step']
                if not eligible:return finish()
            else:eligible=list(self._inflight)
            earliest=min(item['started']+self.timeout for item in self._inflight.values())
            deadline=min(earliest,coalesce_deadline) if result else earliest
            remaining=max(0.,deadline-time.monotonic())
            if check_stop is not None:remaining=min(remaining,.05)
            wait_started=time.monotonic()
            connections=wait([self._connections[i] for i in eligible],remaining)
            if result:coalesce_wait+=time.monotonic()-wait_started
            if not connections:
                now=time.monotonic()
                if result and now>=coalesce_deadline:
                    window_exhausted=True;return finish()
                if now<earliest:continue
                self._failed=True
                failure={'error':'Ready-driven native timeout','pending_workers':sorted(self._inflight)}
                self.failures.append(failure);self._persist()
                raise THRLVectorError('Ready-driven native timeout',[failure])
            # Only consume at most16. Rotate actual ready workers so a fast
            # resubmitted0..15 cannot repeatedly postpone ready16..19.
            cursor=getattr(self,'_ready_cursor',0)
            connections=sorted(connections,key=lambda c:(self._connections.index(c)-cursor)%self.num_envs)
            for connection in connections[:16-len(result)]:
                index=self._connections.index(connection);expected=self._inflight[index]
                try:
                    message=connection.recv();self._note(index,message)
                    if message.get('kind')=='ownership' and message.get('sequence')==expected['sequence']:continue
                    if message.get('kind')=='error':raise THRLVectorError('Native worker failed',[message])
                    if message.get('sequence')!=expected['sequence'] or message.get('kind')!=expected['operation']:
                        raise ValueError('Ready-driven native response sequence/operation differs')
                    self._needs_reset[index]=(bool(message['terminated'] or message['truncated'])
                        if expected['operation']=='step' else False)
                    if expected['operation']=='reset':self.burn_in[index]=message.get('burn_in')
                    message['observation']={key:(row[index].copy() if copy_observations else row[index].view())
                        for key,row in self._arrays.items()}
                    if not copy_observations:
                        for value in message['observation'].values():value.setflags(write=False)
                    message['operation']=expected['operation']
                    result.append(message);del self._inflight[index]
                    self._ready_cursor=(index+1)%self.num_envs
                    if coalesce_deadline is None:coalesce_deadline=time.monotonic()+max_wait_ms/1000.
                except (EOFError,OSError,ValueError,THRLVectorError) as error:
                    self._failed=True
                    failures=error.failures if isinstance(error,THRLVectorError) else [{'index':index,'error':repr(error)}]
                    self.failures.extend(failures);self._persist()
                    raise THRLVectorError('Ready-driven native receive failed',failures) from error


def make_dn_cost_evaluation_vector(worker_cpus,output):
    """Spread only owned helpers AFTER constructor-ready and BEFORE any reset.

    Caller must save the original allowed CPU set before narrowing the learner.
    None preserves the previous behavior; never infer a pool from its now-two-
    core affinity. The EXE is lazily launched by reset, so it inherits its
    helper's already-assigned pool. No reset, action or timeout is changed.
    """
    if worker_cpus is None:return DNCostReadyVector
    cpus=tuple(worker_cpus)
    if not cpus or any(type(c) is not int or c<0 for c in cpus) or len(set(cpus))!=len(cpus):
        raise ValueError('Explicit saved pre-narrowing worker CPU pool required')
    class ScheduledDNCostReadyVector(DNCostReadyVector):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            try:
                import psutil
                from .owned_worker_affinity import spread_owned_workers
                from .thrl_durable_records import write_json
                parent=psutil.Process(os.getpid())
                receipt=spread_owned_workers(parent.pid,parent.create_time(),cpus)
                write_json(Path(output)/'worker-affinity-before-reset.json',receipt)
            except BaseException as error:
                try:self.close()
                except BaseException as cleanup_error:
                    error.add_note('Owned vector cleanup after affinity failure: '+repr(cleanup_error))
                raise
    return ScheduledDNCostReadyVector


class DNCostNativePolicy(PoolWideVisualNativePolicy):
    """Explicit B16, same original own-game lifecycle, complete typed model/head.

    The forward is a static thin derivative of WideVisualNativePolicy: the one
    brain call is visible, followed immediately by the shared DN-cost operator.
    No hooks, temporary brain replacements or implicit second forward are used.
    """
    def __init__(self, model, *, source_schema, beta=None, neural_batch_size=16,
                 episodes=4, brain_seed=613711, synthetic_fixture=False,trace_observer=None):
        from .fly_dn_cost_model import DNCostActorCritic, DNCostHead, PARAMETER_SCHEMA, HEAD_SCHEMA
        if (not isinstance(model,DNCostActorCritic) or source_schema != PARAMETER_SCHEMA
                or model.source_schema != source_schema or not isinstance(model.risk_head,DNCostHead)
                or model.risk_head.source_schema != HEAD_SCHEMA):
            raise ValueError('Complete typed DN-cost actor-critic/head/schema required;bare student is forbidden')
        if neural_batch_size != 16:
            raise ValueError('This candidate declares identical B16 sampling/replay/deployment width')
        if trace_observer is not None and not callable(getattr(trace_observer,'record_batch',None)):
            raise ValueError('Optional data-only trace observer requires record_batch')
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Independent frozen full typed eval model required;do not adopt the live learner')
        head, brain = model.risk_head, model.brain
        if (head.reference_rate_hz != brain.parameters_source.reference_poisson_rate_hz
                or not torch.equal(head.output_indices.detach().cpu(),brain.output_indices.detach().cpu())
                or any(p.dtype != brain.action_readout.weight.dtype or p.device != brain.action_readout.weight.device
                       for p in head.parameters())):
            raise ValueError('The SAME DN order/reference/device must drive actor and risk')
        super().__init__(model.student,neural_batch_size=16,episodes=episodes,
            brain_seed=brain_seed,synthetic_fixture=synthetic_fixture)
        self.actor_critic = model  # references the same student/head, never a copy.
        self.risk_head = model.risk_head
        self.source_schema = source_schema
        self.source_head_beta = float(head.beta)
        self.beta = _beta(head.beta if beta is None else beta)
        self.beta_overridden = beta is not None
        self.risk_head_batches = self.risk_real_rows = self.risk_dummy_rows = 0
        self.trace_observer=trace_observer
        self.trace_observer_batches=0
        self.last_policy_rows = None

    def set_native_episode_context(self, worker_games):
        context=tuple(worker_games)
        if (self._context is not None or not 1<=len(context)<=16
                or any(not isinstance(pair,(tuple,list)) or len(pair)!=2 for pair in context)):
            raise ValueError('One unconsumed ready batch of at most16 actual worlds required')
        if (any(type(w) is not int or not 0<=w<30 or type(g) is not int
                or not 0<=g<self.episodes for w,g in context)
                or len({w for w,_ in context})!=len(context) or len({g for _,g in context})!=len(context)):
            raise ValueError('Unique bounded actual worker<30/game identities required')
        self._context=context

    @torch.no_grad()
    def forward_step(self, features, maps, cursor, *, objects):
        context,self._context = self._context,None
        if context is None:
            raise ValueError('Actual worker/game context required before every ready batch')
        count=len(context)
        if (tuple(cursor.shape)!=(1,count,1) or tuple(features.shape)!=(count,25)
                or tuple(maps.shape)!=(count,16,92,96) or tuple(objects.shape)!=(count,96,29)
                or not bool(torch.isfinite(cursor).all())):
            raise ValueError('Actual K<=16 V5 rows and own episode cursors required')
        if (self.actor_critic.student is not self.student or self.actor_critic.risk_head is not self.risk_head
                or self.actor_critic.source_schema != self.source_schema
                or float(self.risk_head.beta) != self.source_head_beta):
            raise RuntimeError('Typed model/head/source beta changed during native evaluation')
        tokens=cursor.detach().cpu().reshape(-1).tolist()
        selected,fresh_games=[],[]
        for (worker,game),token in zip(context,tokens):
            previous=self.workers.get(worker)
            if token==0:
                if game in self._seen_games or any(s.game==game for s in self.workers.values()):
                    raise ValueError('Only a genuinely new native game may reset the neural state')
                key=native_brain_key(game,self.brain_seed)
                selected.append(NativeWorkerBrain(game,key,0,self.student.initial_state(1,rng_seed=key)))
                fresh_games.append((worker,game,key))
            elif token!=game+1 or previous is None or previous.game!=game:
                raise ValueError('Stale/mismatched worker/game cursor')
            else:selected.append(previous)
        features,maps,objects=self.student._modalities(features,maps,objects)
        sensory=self.student.sensory_adapter(features,maps,objects)
        if sensory.shape!=(count,384):
            raise ValueError('Original frozen standardized visual384 required')
        states=[item.state for item in selected]
        padded_features=features
        if count<16:
            sensory=torch.cat((sensory,sensory.new_zeros(16-count,384)),0)
            padded_features=torch.cat((features,features.new_zeros(16-count,25)),0)
            states.append(self.student.initial_state(16-count,rng_seed=0x7fff0000))
        raw,following=self.student.brain(sensory,pack_whole_brain_states(states))  # exactly ONE brain advance
        if raw.shape!=(16,19) or not bool(torch.isfinite(raw).all()):
            raise FloatingPointError('One B16 full brain must return finite raw19')
        composed,cost=compose_from_features(self.risk_head,raw,following,padded_features,beta=self.beta)
        actual=source_policy_logits(composed[:count],features)
        if self.trace_observer is not None:
            # Explicit data callback, not a torch forward hook. The caller's
            # observer copies bounded data to CPU, handles its own logging
            # errors, and must not mutate model/state/RNG or advance a brain.
            # Return value is deliberately ignored before actual state commit.
            self.trace_observer.record_batch(model=self.actor_critic,context=context,selected=selected,
                sensory=sensory,following=following,features=features,maps=maps,objects=objects,
                raw=raw,composed=composed,cost=cost,actual_logits=actual)
            self.trace_observer_batches+=1
        real_states=unpack_whole_brain_state(following,count=count)
        for (worker,game),item,state in zip(context,selected,real_states):
            self.workers[worker]=NativeWorkerBrain(game,item.brain_rng_key,item.decisions+1,state)
        for worker,game,key in fresh_games:
            self._seen_games.add(game)
            self.episode_records.append(dict(worker=worker,game=game,brain_rng_key=key))
        self.started_games+=len(fresh_games)
        self.data_forwards+=count;self.brain_batch_calls+=1;self.dummy_lane_forwards+=16-count
        self.risk_head_batches+=1;self.risk_real_rows+=count;self.risk_dummy_rows+=16-count
        for record in self.episode_records:
            live=self.workers.get(record['worker'])
            if live is not None and live.game==record['game']:
                record.update(decisions=live.decisions,noise_counter_after=int(live.state.noise_counter.item()))
        self.last_policy_rows=dict(raw_logits=raw[:count].detach(),composed_logits=composed[:count].detach(),
            predicted_costs=cost[:count].detach(),policy_logits=actual.detach(),
            brain_rng_keys=following.rng_keys[:count].detach(),noise_counters=following.noise_counter[:count].detach())
        return actual,None,cursor.new_tensor([g+1 for _,g in context]).reshape(1,count,1)


def evaluate_dn_cost_native_pool(model,config,output,*,source_schema,checkpoint_sha256,beta=None,
        episodes=4,native_workers=4,neural_batch_size=16,seed=2027431601,brain_seed=613711,
        action_selection='greedy',check_stop=lambda:None,progress=None,worker_cpus=None,trace_observer=None):
    """Explicit caller-owned native invocation, reusing the existing evaluator.

    Default action selection is greedy; sampled uses the same T1 policy, with
    existing episode-seeded action RNG. This function never loads a checkpoint,
    starts a monitor, or changes any teacher/student training controller.
    """
    from .thrl_training_mode import validate_training_mode,episode_protocol
    from .thrl_full_story_evaluation import evaluate
    from .thrl_durable_records import write_json
    import train_thrl_native as native_entry
    if (type(episodes) is not int or not 1<=episodes<=20 or type(native_workers) is not int
            or not 1<=native_workers<=min(30,episodes) or action_selection not in ('greedy','sampled')
            or not isinstance(checkpoint_sha256,str) or len(checkpoint_sha256)!=64
            or any(c not in '0123456789abcdef' for c in checkpoint_sha256)):
        raise ValueError('Bounded native cohort, explicit full typed checkpoint SHA and declared selection required')
    if (not validate_training_mode(config) or config.get('input_schema')!=SEPARATED_LASER_INPUT
            or config.get('headless') is not True or config.get('autonomous_bomb') is not True
            or config.get('development_objective')=='avoidance-first-v1'):
        raise ValueError('Existing private natural V5/source19 autonomous-Bomb full-story config required')
    # Bind the caller's saved pool now; do not rediscover the learner's narrowed
    # affinity after workers have inherited it.
    worker_cpus=None if worker_cpus is None else list(worker_cpus)
    vector_class=make_dn_cost_evaluation_vector(worker_cpus,output)
    policy=DNCostNativePolicy(model,source_schema=source_schema,beta=beta,neural_batch_size=neural_batch_size,
        episodes=episodes,brain_seed=brain_seed,trace_observer=trace_observer).eval()
    kwargs=native_entry.make_env_kwargs(config,Path(output),native_workers,'development')
    if (len(kwargs)!=native_workers or any(not k['headless'] or not k['hide_window']
            or k['policy_action_count']!=19 or k['frame_skip']!=4 or k['wall_time_limit_seconds'] is not None
            or 'online_teacher_revision' in k for k in kwargs)):
        raise ValueError('No teacher action path, truncation or visible native worker allowed')
    versions=tuple(p._version for p in model.parameters())
    def guard():
        check_stop()
        if tuple(p._version for p in model.parameters())!=versions:
            raise RuntimeError('Frozen full typed model parameters changed')
    protocol=dict(episode_protocol(config),neural_reset='actual new game only;no Miss/stage reset',
        brain_batch_size=16,real_native_workers=native_workers,maximum_ready_batch=16,
        excess_ready_responses='round-robin;remain unread in original inflight pipes;no hidden queue',
        worker_cpu_pool=worker_cpus,worker_affinity_before_initial_reset=worker_cpus is not None,
        optional_data_observer=trace_observer is not None,observer_extra_brain_forwards=0,
        dummy_state_or_action_committed=False,
        policy_source_schema=source_schema,head_source_schema=policy.risk_head.source_schema,
        risk_input='same following DNrate/reference only',risk_beta=policy.beta,source_head_beta=policy.source_head_beta,
        beta_override_supplied=policy.beta_overridden,action_temperature=1.,
        composition='same shared helper as collection and PPO;postcomposition clamp80/inventory/T1',
        rawfeatures_role='normal/unprotected and inventory gates only;not head inputs',
        risk_actor_or_teacher_bypass=False,extra_brain_for_risk=0)
    guard()
    try:
        result=evaluate(torch,policy,kwargs,output,episodes=episodes,seed=seed,check_stop=guard,
            progress=progress,native_split='development',episode_protocol=protocol,
            summary=native_entry.development_summary,action_selection=action_selection,
            action_temperature=1.,vector_class=vector_class)
    finally:
        # The existing evaluator closes its vector and writes this in finally.
        # Its constructor also closes a partially-created vector, but an error
        # there leaves the evaluator's local vector=None. Preserve that actual
        # ownership receipt rather than inventing successful cleanup.
        destination=Path(output)
        cleanup=destination/'native_cleanup.json'
        if destination.is_dir() and not cleanup.exists():
            ownership=destination/'workers'/'ownership.json'
            record=dict(workers=native_workers,close_errors=None,cleanup_verified=False,
                status='cleanup_receipt_unavailable',ownership_path=str(ownership))
            try:
                evidence=json.loads(ownership.read_text(encoding='utf-8'))
                if evidence.get('schema')=='thrl-native-vector-ownership-v1':
                    record.update(close_errors=evidence.get('close_errors'),
                        status=evidence.get('status'),cleanup_verified=evidence.get('status')=='closed')
            except (OSError,ValueError) as error:
                record['receipt_error']=repr(error)
            write_json(cleanup,record)
    guard()
    result['DN_cost_policy']=dict(schema=SCHEMA,checkpoint_sha256=checkpoint_sha256,
        source_schema=source_schema,head_source_schema=policy.risk_head.source_schema,
        beta=policy.beta,source_head_beta=policy.source_head_beta,beta_override_supplied=policy.beta_overridden,
        action_temperature=1.,action_selection=action_selection,brain_batch_size=16,native_workers=native_workers,
        maximum_ready_batch=16,maximum_native_workers=30,
        worker_cpu_pool=worker_cpus,worker_affinity_before_initial_reset=worker_cpus is not None,
        optional_data_observer=trace_observer is not None,trace_observer_batches=policy.trace_observer_batches,
        brain_batch_calls=policy.brain_batch_calls,real_data_neural_forwards=policy.data_forwards,
        dummy_lane_forwards=policy.dummy_lane_forwards,risk_head_batches=policy.risk_head_batches,
        risk_real_rows=policy.risk_real_rows,risk_dummy_rows=policy.risk_dummy_rows,
        extra_brain_for_risk=0,episode_records=policy.episode_records,
        stage_or_Miss_resets=0,teacher_queries=0,optimizer_updates=0,
        current_candidate_effectiveness_proven=False,cross_width_bitwise_equivalence_claimed=False)
    write_json(Path(output)/'report.json',result)
    return result


# One implementation, kept for callers prepared before the canonical pool name.
evaluate_dn_cost_native = evaluate_dn_cost_native_pool
