"""Opt-in composition of the existing B16 brain/PPO block and task supervision.

No capture, optimizer, native process or training starts here. The current
production CUDA-graph adapter is not changed or declared compatible. This is
the numerical block for a separately versioned future adapter and runner.
"""
from __future__ import annotations

import math

import torch

from .fly_dn_task_risk import FlyDNTaskRiskReadout
from .fly_visual_ppo_loss import VisualPPOLossConfig
from .fly_visual_ppo_update import VisualPPOUpdateConfig
from .fly_visual_ppo_cuda_graph import tensor_reset_state
from .fly_visual_ppo_cuda_graph_checkpoint import (
    BRAIN_WIDTH, _brain_call, _counts, tensor_ppo_loss_wide, validate_wide_block,
)
from .fly_visual_task_supervision import (
    TaskSupervisionLossConfig, task_supervision_loss, validate_task_supervision_inputs,
)

DEFAULT_PPO_CONFIG = VisualPPOUpdateConfig(checkpoint_replay=True,
    loss=VisualPPOLossConfig(teacher_coeff=0.))
TASK_KEYS = {'margins', 'trusted', 'outcomes', 'unprotected_mask'}
REFERENCE_KEYS = {'rule_probabilities', 'rule_weights', 'deathbomb_mask'}


def validate_task_block_inputs(model, data, before, task_data, *, risk_head=None):
    """Host validation before capture; caller must bind native facts to data."""
    validate_wide_block(model, data, before)
    if not TASK_KEYS.issubset(task_data) or set(task_data)-TASK_KEYS-REFERENCE_KEYS:
        raise ValueError('Only compact paired task evidence, never raw observations, enters this block')
    # The actual rollout actions/valid mask are authoritative for every loss.
    validation_logits=torch.where(data['valid_mask'][...,None],
        torch.zeros_like(data['teacher_probs']).masked_fill(~data['legal_actions'],-torch.inf),
        torch.zeros_like(data['teacher_probs']))
    validate_task_supervision_inputs(validation_logits,
        **task_data, actions=data['actions'], valid_mask=data['valid_mask'])
    if risk_head is not None:
        if not isinstance(risk_head, FlyDNTaskRiskReadout):
            raise ValueError('Explicit same-brain DN risk auxiliary required')
        risk_head.validate_brain(model.brain)


def task_tensor_block(model, data, before, task_data, *, normalizers, ppo_real_row_weight=None,
        ppo_config=DEFAULT_PPO_CONFIG, task_config=TaskSupervisionLossConfig(), risk_head=None,
        use_checkpoint=True, counts=None, fresh=None, allow_action_distillation=False):
    """Four same-width forwards; auxiliary readout reuses each following state.

PPO is weighted by real_rows/120. Task terms use whole120-row label counts and
are summed directly across groups. No sample, rollout action, reward, teacher
history or live neural state is modified, and no action is filtered at inference.
"""
    if (not isinstance(ppo_config, VisualPPOUpdateConfig) or ppo_config.tbptt != 4
            or ppo_config.epochs != 1 or type(use_checkpoint) is not bool
            or type(allow_action_distillation) is not bool):
        raise ValueError('Explicit TBPTT4/one-epoch block configuration required')
    if ppo_config.loss.teacher_coeff != 0 and not allow_action_distillation:
        raise ValueError('Final-action distillation must be explicitly requested as a comparator')
    if ((ppo_real_row_weight is not None and (isinstance(ppo_real_row_weight, bool)
            or not math.isfinite(ppo_real_row_weight) or not 0 < ppo_real_row_weight <= 1))
            or not TASK_KEYS.issubset(task_data) or set(task_data)-TASK_KEYS-REFERENCE_KEYS):
        raise ValueError('Explicit real-row PPO weight and compact task fields required')
    if risk_head is not None and not isinstance(risk_head, FlyDNTaskRiskReadout):
        raise ValueError('DN-only auxiliary head or None required')
    counts = _counts() if counts is None else counts
    state = before
    fresh = model.initial_state(BRAIN_WIDTH, rng_seed=0) if fresh is None else fresh
    raw_rows, values, risk_rows = [], [], []
    for t in range(4):
        state = tensor_reset_state(state, fresh,
            data['episode_starts'][t] | ~data['valid_mask'][t], data['brain_rng_keys'][t])
        raw, state = _brain_call(model, data['encoded_visual'][t], state, counts, use_checkpoint)
        raw_rows.append(raw)
        values.append(model.critic.network(data['encoded_visual'][t].detach()).squeeze(-1))
        if risk_head is not None:
            risk_rows.append(risk_head(state))
    raw, values = torch.stack(raw_rows), torch.stack(values)
    risk = torch.stack(risk_rows) if risk_head is not None else None
    ppo, logp = tensor_ppo_loss_wide(raw, values, data, model.brain.recurrent_gain_logits, ppo_config.loss)
    # Tensor-only dynamic weighting lets one B16 graph serve both16 and14 real
    # lanes. A Python64/120 capture constant would silently overweight the tail.
    weight = (data['valid_mask'].sum().to(ppo['total_loss'].dtype) / 120.
              if ppo_real_row_weight is None else ppo_real_row_weight)
    # New full19 reference/DB terms use the same stock legality as the sampled
    # policy. Keep dummy rows finite; they have no task/real PPO weight.
    task_logits = torch.where(data['valid_mask'][...,None],
        raw.masked_fill(~data['legal_actions'], -torch.inf), torch.zeros_like(raw))
    tasks = task_supervision_loss(task_logits, **task_data, actions=data['actions'],
        valid_mask=data['valid_mask'], config=task_config, risk_logits=risk, normalizers=normalizers)
    return dict(total_loss=weight * ppo['total_loss'] + tasks['total_loss'],
        PPO_terms=ppo, task_terms=tasks, log_probs=logp, following_state=state,
        raw_logits=raw, values=values, risk_logits=risk)
