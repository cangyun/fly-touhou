"""Temporary recurrent/actor fitting with exactly frozen visual encoders.

FP32 encodings and fitted risk scores are valid only while their originating
modules are unchanged. This is not an end-to-end update or a new policy rule.
Full-network optimization must recompute encodings from original observations.
"""
from __future__ import annotations


TRAINABLE_PREFIXES = ('base.gru.', 'base.actor_trunk1.',
                      'base.actor_trunk2.', 'base.actor_head.')


def begin_cached_actor_phase(model):
    if (not getattr(model, 'frozen_readout', False) or not model.is_recurrent
            or not hasattr(model, 'precompute_frozen_risk')):
        raise ValueError('Qualified frozen-risk B1 GRU required')
    model.requires_grad_(False)
    names = []
    for name, parameter in model.named_parameters():
        if name.startswith(TRAINABLE_PREFIXES):
            parameter.requires_grad_(True)
            names.append(name)
    if not names:
        raise ValueError('B1 recurrent and actor parameters were not found')
    model.train()
    model.base.feature_proj.eval()
    model.base.map_encoder.eval()
    model.base.object_attention.eval()
    model.risk.eval()
    if model.laser_risk is not None:
        model.laser_risk.eval()
    return dict(temporary_phase=True, trainable_parameter_names=names,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        original_full_model_parameters=sum(p.numel() for p in model.parameters()),
        encoder_and_fixed_risk_frozen=True, critic_updated=False,
        permanent_encoder_freezing_not_claimed=True)


def forward_cached_actor(model, encoded, features, risk, initial_hidden, dones):
    import torch
    if any(p.requires_grad for name, p in model.named_parameters()
           if not name.startswith(TRAINABLE_PREFIXES)):
        raise ValueError('Frozen encodings cannot train changed input encoders or critics')
    if (encoded.ndim != 3 or features.shape[:2] != encoded.shape[:2]
            or features.shape[-1] != 25 or risk.shape != (*encoded.shape[:2], 18)
            or dones.shape != encoded.shape[:2] or encoded.shape[2] != model.gru.input_size
            or encoded.dtype != torch.float32 or features.dtype != torch.float32
            or risk.dtype != torch.float32 or encoded.requires_grad or risk.requires_grad):
        raise ValueError('Matched original FP32 feature/encoding/frozen-risk sequence required')
    batch, length = encoded.shape[:2]
    if not batch or not length or initial_hidden.shape != (1, batch, model.gru_hidden_size):
        raise ValueError('Matched nonempty sequence and actual recurrent context required')
    if dones[:, :-1].any():
        raise ValueError('Cached recurrent chunks must not cross a natural episode boundary')
    # Keep the original per-timestep GRU calls and multiplication order. A
    # single cuDNN/CPU sequence call is mathematically equivalent but can alter
    # tiny gradients enough for Adam to amplify their rounding differences.
    hidden = initial_hidden.contiguous()
    outputs = []
    for t in range(length):
        if t:
            done = dones[:, t-1].bool()[None, :, None]
            hidden = hidden * (~done).to(hidden.dtype)
        output, hidden = model.gru(encoded[:, t].unsqueeze(0), hidden)
        outputs.append(output.squeeze(0))
    outputs = torch.stack(outputs, dim=1).reshape(batch * length, model.gru_hidden_size)
    flat_features = features.reshape(batch * length, 25)
    logits = model.base._policy_logits(outputs, flat_features)
    # Existing public cache path: the fitted point/laser networks have no
    # dependence on the recurrent/actor weights being optimized here.
    return model._adjust(logits, flat_features, None,
                         precomputed_risk=risk.reshape(batch * length, 18))
