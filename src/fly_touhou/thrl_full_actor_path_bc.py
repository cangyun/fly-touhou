"""Raw V5 actor-path BC helpers; no data loader, optimizer or automatic run.

Encodings are recomputed from raw FP32 observations on every call. Hidden
states belong to the caller's own policy/history, never an old latent bank.
This module does not change the existing PPO, rewards or model classes.
"""
from __future__ import annotations

from .thrl_inputs import SEPARATED_LASER_INPUT

ENCODER_PREFIXES = ('base.feature_proj.', 'base.map_encoder.', 'base.object_attention.')
GRU_ACTOR_PREFIXES = ('base.gru.', 'base.actor_trunk1.',
                      'base.actor_trunk2.', 'base.actor_head.')
FROZEN_PREFIXES = ('base.critic_trunks.', 'base.critic_heads.', 'risk.', 'laser_risk.')
SCOPES = ('full_actor_path', 'gru_actor_control')
SCHEMA = 'B1-raw-V5-full-actor-path-BC-helper-v1'


def _parameters_for_scope(model, scope):
    import torch
    if scope not in SCOPES:
        raise ValueError('BC scope must be full_actor_path or gru_actor_control')
    base = getattr(model, 'base', None)
    if (base is None or getattr(model, 'input_schema', None) != SEPARATED_LASER_INPUT
            or getattr(base, 'input_schema', None) != SEPARATED_LASER_INPUT
            or not getattr(model, 'is_recurrent', False) or not getattr(base, 'is_recurrent', False)
            or getattr(model, 'frozen_readout', None) is not True
            or getattr(base, 'feature_dim', None) != 25 or getattr(base, 'map_channels', None) != 16
            or getattr(base, 'action_dim', None) != 19
            or getattr(base, 'object_attention', None) is None
            or getattr(getattr(base, 'map_encoder', None), 'spatial_attention', None) is None
            or getattr(model, 'laser_risk', None) is None
            or not isinstance(getattr(base, 'gru', None), torch.nn.GRU)
            or base.gru.num_layers != 1 or base.gru.bidirectional
            or not callable(getattr(model, 'precompute_frozen_risk', None))):
        raise ValueError('Require the recurrent V5/source19 policy with both attention branches and frozen point/laser risk')
    parameters = list(model.named_parameters())
    allowed = ENCODER_PREFIXES + GRU_ACTOR_PREFIXES + FROZEN_PREFIXES
    if (not parameters or any(not name.startswith(allowed) for name, _ in parameters)
            or any(not any(name.startswith(prefix) for name, _ in parameters)
                   for prefix in allowed)
            or any(parameter.dtype != torch.float32 for _, parameter in parameters)):
        raise ValueError('Every FP32 policy parameter must belong to an explicit BC whitelist group')
    active = GRU_ACTOR_PREFIXES + (ENCODER_PREFIXES if scope == 'full_actor_path' else ())
    # A future shared-parameter alias must not make a frozen head trainable.
    aliases = list(model.named_parameters(remove_duplicate=False))
    trainable_ids = {id(p) for name, p in aliases if name.startswith(active)}
    frozen_ids = {id(p) for name, p in aliases if not name.startswith(active)}
    if trainable_ids & frozen_ids:
        raise ValueError('BC trainable and frozen whitelist groups cannot share parameters')
    return parameters, active


def begin_actor_path_bc(model, scope='full_actor_path'):
    """Select an explicit whitelist and clear stale gradients; do no update.

    Counts are computed from this model, not assumed from the production D95
    architecture. A new owning loop must create its own BC optimizer.
    """
    parameters, active = _parameters_for_scope(model, scope)
    for name, parameter in parameters:
        parameter.requires_grad_(name.startswith(active))
        parameter.grad = None
    model.train()
    if scope == 'gru_actor_control':
        model.base.feature_proj.eval()
        model.base.map_encoder.eval()
        model.base.object_attention.eval()
    model.base.critic_trunks.eval()
    model.base.critic_heads.eval()
    model.risk.eval()
    model.laser_risk.eval()
    model._actor_path_bc_scope = scope
    return dict(schema=SCHEMA, scope=scope, input_schema=SEPARATED_LASER_INPUT,
        action_count=19, BC_action_temperature=1.,
        trainable_parameter_names=[name for name, p in parameters if p.requires_grad],
        trainable_parameters=sum(p.numel() for _, p in parameters if p.requires_grad),
        frozen_parameter_names=[name for name, p in parameters if not p.requires_grad],
        total_parameters=sum(p.numel() for _, p in parameters),
        encoder_trainable=scope == 'full_actor_path', critic_updated=False,
        point_and_laser_risk_frozen=True, raw_encoder_recomputed_per_call=True,
        encoder_cache_accepted=False, hidden_provenance='caller-owned policy and original-episode history',
        optimizer_created=False, optimizer_steps=0)


def forward_actor_path(model, features, maps, objects, initial_hidden, dones):
    """Return final-wrapper raw logits[B*L,19] and following_hidden[1,B,H].

    Batch-major ordering and GRU multiplication/reset timing match the core:
    done[t] resets before observation[t+1]. Following hidden is the unmasked
    state after the final observation, as in forward_step. The caller zeros
    it for a new natural episode and chooses any TBPTT detach explicitly.
    Stage/Miss fields never trigger a reset. Numeric shape cannot certify a
    hidden state's provenance; the owning loop must track its own history.
    """
    import torch
    scope = getattr(model, '_actor_path_bc_scope', None)
    parameters, active = _parameters_for_scope(model, scope)
    if any(p.requires_grad != name.startswith(active) for name, p in parameters):
        raise ValueError('BC trainability changed after begin_actor_path_bc')
    values = (features, maps, objects, initial_hidden, dones)
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise ValueError('Matched raw observation, hidden and done tensors are required')
    if features.ndim != 3 or features.shape[-1] != 25:
        raise ValueError('Raw V5 features must be [B,L,25]')
    batch, length = features.shape[:2]
    if (batch < 1 or length < 1 or maps.shape != (batch, length, 16, 92, 96)
            or objects.shape != (batch, length, 96, 29)
            or initial_hidden.shape != (1, batch, model.gru_hidden_size)
            or dones.shape != (batch, length) or dones.dtype != torch.bool
            or any(value.device != features.device for value in values)
            or any(p.device != features.device for _, p in parameters)
            or any(value.dtype != torch.float32 for value in values[:-1])):
        raise ValueError('Matched nonempty original FP32 V5/raw shapes, own hidden and boolean dones are required')
    if torch.is_autocast_enabled() or torch.is_autocast_enabled('cpu'):
        raise ValueError('Actor-path BC requires strict FP32 without autocast')
    if features.device.type == 'cuda' and (torch.backends.cuda.matmul.allow_tf32
                                         or torch.backends.cudnn.allow_tf32):
        raise ValueError('Actor-path BC requires both CUDA TF32 flags disabled')
    if any(not bool(torch.isfinite(value).all()) for value in values[:-1]):
        raise ValueError('Raw observations and own initial hidden must be finite')

    flat_features = features.reshape(batch * length, 25)
    flat_objects = objects.reshape(batch * length, 96, 29)
    encoded = model._encode(flat_features, maps.reshape(batch * length, 16, 92, 96), flat_objects)
    encoded = encoded.reshape(batch, length, -1)
    # Only the independent fixed physical readouts are detached, never the
    # mutable visual/object encoder. No cache survives this call.
    with torch.no_grad():
        risk = model.precompute_frozen_risk(flat_features, flat_objects).detach()
    hidden = initial_hidden.contiguous()
    outputs = []
    for t in range(length):
        if t:
            hidden = hidden * (~dones[:, t-1])[None, :, None].to(hidden.dtype)
        output, hidden = model.gru(encoded[:, t].unsqueeze(0), hidden)
        outputs.append(output.squeeze(0))
    outputs = torch.stack(outputs, dim=1).reshape(batch * length, model.gru_hidden_size)
    logits = model.base._policy_logits(outputs, flat_features)
    return model._adjust(logits, flat_features, None, precomputed_risk=risk), hidden


def hard_action_ce(logits, actions, legal=None):
    """Inventory-legal executed-action CE, at the existing T=1 BC definition.

    actions/legal may be [B,L] or flattened, in the same batch-major order.
    Every excluded target is first replaced by an actually legal action, so
    a missing-inventory Bomb cannot produce inf*0/NaN. Zero selected labels
    return a differentiable zero; forward state still belongs to the caller.
    """
    import torch
    from .thrl_policy_actions import policy_logits
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != 19
            or logits.shape[0] < 1 or logits.dtype != torch.float32
            or not isinstance(actions, torch.Tensor) or actions.ndim not in (1, 2)
            or actions.numel() != len(logits) or actions.dtype != torch.int64
            or actions.device != logits.device or bool(((actions < 0) | (actions >= 19)).any())):
        raise ValueError('Matched FP32 source19 logits and integer executed actions are required')
    if legal is not None and (not isinstance(legal, torch.Tensor) or legal.shape != actions.shape
                              or legal.dtype != torch.bool or legal.device != logits.device):
        raise ValueError('Optional legal mask must match the executed-action tensor')
    clean = policy_logits(logits, action_temperature=1.)
    actions = actions.reshape(-1)
    support = torch.isfinite(clean)
    inventory_legal = support.gather(1, actions[:, None]).squeeze(1)
    selected = inventory_legal if legal is None else inventory_legal & legal.reshape(-1)
    fallback = support.to(torch.int64).argmax(-1)
    safe_targets = torch.where(selected, actions, fallback)
    losses = torch.nn.functional.cross_entropy(clean, safe_targets, reduction='none')
    loss = (losses * selected.to(losses.dtype)).sum() / selected.sum().clamp_min(1)
    return loss, dict(rows=len(logits), selected_labels=int(selected.sum()),
        inventory_illegal_labels=int((~inventory_legal).sum()),
        externally_excluded_legal_labels=int((inventory_legal & ~selected).sum()),
        action_temperature=1.)
