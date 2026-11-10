"""Migrate interfaces and objective without changing the connectome."""
import copy
import torch
from torch import nn
from .nbnm_sensation import EXTENSION_WIDTH, SCHEMA


def configure_model(model, config, *, reset_critics=True):
    model.nbnm = copy.deepcopy(config)
    model.input_schema = SCHEMA
    encoder = model.core.input_encoder
    if encoder.extended_context is None:
        encoder.extended_context = nn.Linear(EXTENSION_WIDTH, encoder.residual.out_features,
            bias=False, device=model.device)
        nn.init.zeros_(encoder.extended_context.weight)
    model.core.sensory_width = 47 + EXTENSION_WIDTH
    model.bombs_disabled = True
    model.lower_field_default = False
    model.combat_position_guidance_coefficient = 0.
    model.protected_exit_forecast = False
    model.frozen_decoding = None
    model.full_story_action_selection = 'joint_mode'
    model.pad_candidate_batch = False
    encoder.batch_invariant = True
    model.core.cache_inference_parameters = True
    if reset_critics:
        # The old three critics predicted different returns. Preserve learned
        # brain/risk/movement weights, not numerically incompatible values.
        layers = [m for m in model.critic.modules() if isinstance(m, nn.Linear)]
        if layers:
            nn.init.zeros_(layers[-1].weight)
            if layers[-1].bias is not None: nn.init.zeros_(layers[-1].bias)
    return model
