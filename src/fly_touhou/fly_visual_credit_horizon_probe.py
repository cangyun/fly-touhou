"""Optional read-only T4/T16 credit comparison on one already collected B16 group.

Call before ANY update of this rollout. Both arms differentiate only rows12..15,
with the same full960-row GAE/returns, current parameters and recorded game keys.
No optimizer/teacher/native call, global RNG draw, or live state commit occurs.
Failure is diagnostic-only; callers must never reject/recollect the training batch.
"""
from __future__ import annotations

from dataclasses import fields
import time
import traceback

import torch

from .fly_visual_actor_critic import VisualFlyActorCritic
from .fly_visual_ppo_loss import categorical_policy, scalar_gae, visual_ppo_loss
from .fly_visual_ppo_update import VisualPPOUpdateConfig, _clone_state
from .fly_visual_ppo_cuda_graph import tensor_reset_state
from .fly_visual_ppo_cuda_graph_checkpoint import _brain_call

SCHEMA = 'fly-visual-readonly-credit-horizon-T4-T16-v1'


def _state_error(actual, expected, tolerance):
    result = {}
    for field in fields(actual):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        pairs = list(zip(a, b)) if field.name == 'delay_queue' else [(a, b)]
        errors = []
        for left, right in pairs:
            left, right = left.detach().cpu(), right.detach().cpu()
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError('Complete own state shape/dtype differs:' + field.name)
            errors.append(float((left.double()-right.double()).abs().max()))
        maximum = max(errors, default=0.)
        exact = all(torch.equal(x.detach().cpu(), y.detach().cpu()) for x, y in pairs)
        floating = pairs[0][0].is_floating_point() if pairs else False
        result[field.name] = dict(max_abs_error=maximum, exact=exact,verified=maximum <= tolerance if floating else exact)
    return dict(fields=result,verified=all(x['verified'] for x in result.values()),exact=all(x['exact'] for x in result.values()))


def _parameters(model):
    result = [('brain.'+n,p) for n,p in model.brain.named_parameters()]
    result += [('critic.'+n,p) for n,p in model.critic.named_parameters()]
    if len(result) != 8 or any(not p.requires_grad or p.dtype != torch.float32 for _,p in result):
        raise ValueError('Original eight trainable brain/critic FP32 tensors required')
    return result


def _signature(named):
    return tuple((p.data_ptr(),p._version,None if p.grad is None else (p.grad.data_ptr(),p.grad._version)) for _,p in named)


def _arm(model, data, before, cached12, cached16, cfg, horizon, counts, *,
         check_stop, tolerance, use_checkpoint=True):
    """Only the probe uses this graph. Returned gradients/state are CPU detached."""
    state = _clone_state(before, model.device)
    fresh = model.initial_state(16, rng_seed=0)
    raws, values = [], []
    boundary12 = None
    start = 16-horizon
    for t in range(start,16):
        check_stop()
        state = tensor_reset_state(state,fresh,data['episode_starts'][t] | ~data['valid_mask'][t],data['brain_rng_keys'][t])
        real = data['valid_mask'][t]
        if (not torch.equal(state.rng_keys[real],data['brain_rng_keys'][t,real])
                or not torch.equal(state.noise_counter[real],data['noise_counters'][t,real])):
            raise ValueError('Recorded natural-game keys/counters disagree at row'+str(t))
        if t == 12: boundary12 = _state_error(state,cached12,tolerance)
        raw,state = _brain_call(model,data['encoded_visual'][t],state,counts,use_checkpoint)
        if t >= 12:
            raws.append(raw)
            values.append(model.critic(data['encoded_visual'][t]))
    raw, value = torch.stack(raws),torch.stack(values)
    tail = slice(12,16)
    terms = visual_ppo_loss(raw,value,data['actions'][tail],data['old_log_probs'][tail],data['values'][tail],
        data['advantages'][tail],data['returns'][tail],data['teacher_probs'][tail],data['legal_actions'][tail],
        model.brain.recurrent_gain_logits,config=cfg.loss,valid_mask=data['valid_mask'][tail])
    logp = categorical_policy(raw,data['legal_actions'][tail]).log_prob(data['actions'][tail])
    valid = data['valid_mask'][tail]
    logp_error = float((logp-data['old_log_probs'][tail])[valid].detach().abs().max())
    raw_error = float((raw-data['raw_logits'][tail])[valid].detach().abs().max())
    # Stored row16 is pre-forward, so apply ONLY its recorded natural-game reset
    # (and fresh dummy lanes) before comparison. No Miss/stage/update reset.
    aligned = tensor_reset_state(state,fresh,data['episode_starts'][16] | ~data['valid_mask'][16],data['brain_rng_keys'][16])
    boundary16 = _state_error(aligned,cached16,tolerance)
    following = _clone_state(state,'cpu')
    check_stop(); counts['backward_attempts'] += 1
    gradients = torch.autograd.grad(terms['total_loss'],[p for _,p in _parameters(model)])
    counts['backwards_completed'] += 1
    if not bool(torch.isfinite(terms['total_loss'])) or any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError('Nonfinite diagnostic loss/gradient')
    return dict(terms={k:float(v.detach()) for k,v in terms.items()},
        gradients=tuple(g.detach().cpu().clone() for g in gradients), following_state=following,
        raw=raw.detach().cpu(),logp=logp.detach().cpu(),cached_raw_max_abs_error=raw_error,
        cached_old_logp_max_abs_error=logp_error,boundary12=boundary12,boundary16_after_recorded_reset=boundary16,
        within_tolerance=raw_error <= tolerance and logp_error <= cfg.old_logp_tolerance and boundary12['verified'] and boundary16['verified'],
        exact_forward=raw_error == 0 and logp_error == 0 and boundary12['exact'] and boundary16['exact'])


def probe_credit_horizon(model, rollout, cfg=VisualPPOUpdateConfig(), *, enabled=False,
                         group=0, check_stop=lambda:None, atol=5e-5):
    """Return JSON-ready evidence; disabled by default and never a PPO quality gate.

    group0 =16 real lanes; group1 =14+2 padding. Activation checkpoint is always
    used, independent of the updater's setting. Counted forwards are EXTRA neural
    lane work, never PPO decisions, label queries or Adam. Two arms run serially
    and move gradients/state to CPU before releasing each autograd graph.
    """
    report = dict(schema=SCHEMA,status='disabled',enabled=bool(enabled),optimizer_updates=0,
        PPO_decisions=0,native_actions=0,teacher_queries=0,neural_batch_size=16,group=group,
        activation_checkpoint=True,preserve_rng_state=False,actual_cached_samples_reused=True,
        loss_steps=[12,13,14,15],full_rollout_advantage_rows=960,
        interpretation='Gradient difference tests longer surrogate credit only; no performance or causal root-cause claim',
        costs={name:dict(base_neural_lane_forwards=0,checkpoint_recompute_neural_lane_forwards=0,
            backward_attempts=0,backwards_completed=0) for name in ('T4','T16')})
    if not enabled:return report
    begun = time.monotonic(); named = None; signature = None
    try:
        if (not isinstance(model,VisualFlyActorCritic) or getattr(rollout,'neural_batch_size',None) != 16
                or rollout.encoded_visual.shape != (32,30,384) or group not in (0,1)
                or rollout.metadata.get('same_neural_batch_in_collection_and_replay') is not True
                or rollout.truncated.any() or atol < 0 or not torch.isfinite(torch.tensor(atol))):
            raise ValueError('Already collected full960-row same-width16 natural rollout required')
        if model.brain.substeps != 100 or model.brain.dt_ms != .2:
            raise ValueError('Original100*.2ms physiology required')
        named = _parameters(model); signature = _signature(named)
        advantage,returns = scalar_gae(rollout.rewards,rollout.values,rollout.bootstrap_values,rollout.terminated,
            truncated=rollout.truncated,episode_starts=rollout.episode_starts,gamma=cfg.gamma,gae_lambda=cfg.gae_lambda)
        normalized = (advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-8)
        workers = list(range(0,16)) if group == 0 else list(range(16,30))
        count = len(workers)
        def pad(x,fill=0):
            return torch.cat((x,x.new_full((len(x),16-count,*x.shape[2:]),fill)),1) if count < 16 else x
        keys = ('encoded_visual','actions','old_log_probs','values','teacher_probs','legal_actions',
                'episode_starts','brain_rng_keys','noise_counters','raw_logits')
        data = {k:pad(getattr(rollout,k)[:17,workers].detach(),fill=(k=='legal_actions')).to(model.device).clone() for k in keys}
        data['teacher_probs'][:,count:] = 1./19
        data['valid_mask'] = torch.arange(16,device=model.device).lt(count).expand(17,-1)
        data.update(advantages=pad(normalized[:17,workers]).to(model.device),returns=pad(returns[:17,workers]).to(model.device))
        if any(v.requires_grad or (v.is_floating_point() and not bool(torch.isfinite(v).all())) for v in data.values()):
            raise ValueError('Finite detached actual inputs/targets required')
        cached12,cached16 = rollout.block_states[(12,group)],rollout.block_states[(16,group)]
        arms = {}
        for horizon in (4,16):
            tag='T'+str(horizon)
            arms[tag] = _arm(model,data,rollout.block_states[(16-horizon,group)],cached12,cached16,cfg,horizon,
                report['costs'][tag],check_stop=check_stop,tolerance=atol)
        a,b=arms['T4'],arms['T16']; rows={}
        for (name,_),g4,g16 in zip(named,a['gradients'],b['gradients']):
            x,y=g4.double().flatten(),g16.double().flatten();n4=float(x.norm());n16=float(y.norm());difference=y-x
            rows[name]=dict(T4_norm=n4,T16_norm=n16,difference_norm=float(difference.norm()),
                difference_max_abs=float(difference.abs().max()),
                cosine=None if not n4 or not n16 else float(torch.dot(x,y)/(n4*n16)),
                relative_difference_vs_T4=None if not n4 else float(difference.norm())/n4)
        pair_state = _state_error(a['following_state'],b['following_state'],atol)
        loss_error = max(abs(a['terms'][k]-b['terms'][k]) for k in a['terms'])
        raw_error = float((a['raw']-b['raw']).abs().max())
        logp_error = float((a['logp']-b['logp']).abs().max())
        verified = a['exact_forward'] and b['exact_forward'] and pair_state['exact'] and raw_error == logp_error == 0 and loss_error <= atol
        report.update(status='verified_readonly_comparison' if verified else 'unverified',verified=verified,
            parameter_gradients=rows,real_loss_rows=count*4,padding_loss_rows=0,
            arms={tag:{k:v for k,v in arm.items() if k not in ('gradients','following_state','raw','logp')} for tag,arm in arms.items()},
            pair=dict(following_state=pair_state,loss_terms_max_abs_error=loss_error,raw_max_abs_error=raw_error,logp_max_abs_error=logp_error),
            earlier12_input_gain_contribution={n:dict(difference_norm=rows[n]['difference_norm'],
                attribution_verified=verified,nonzero_difference=rows[n]['difference_norm']>0)
                for n in rows if n in ('brain.input_encoder.weight','brain.input_encoder.bias','brain.recurrent_gain_logits')},
            attribution_scope='T16 minus T4 at identical last4 loss; natural resets cut earlier credit; not a performance improvement')
    except Exception as error:
        report.update(status='diagnostic_unavailable',verified=False,error_type=type(error).__name__,error=str(error),
            partial_neural_call_may_be_unmeasured=True,training_batch_must_be_retained=True)
        traceback.clear_frames(error.__traceback__)
    finally:
        if named is not None and _signature(named) != signature:
            report.update(status='unverified',verified=False,parameter_or_existing_gradient_storage_changed=True)
        report['elapsed_seconds']=time.monotonic()-begun
        report['extra_neural_lane_forwards']=sum(v['base_neural_lane_forwards']+v['checkpoint_recompute_neural_lane_forwards'] for v in report['costs'].values())
        report['extra_backwards_completed']=sum(v['backwards_completed'] for v in report['costs'].values())
    return report
