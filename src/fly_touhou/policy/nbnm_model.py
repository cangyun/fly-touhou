"""Migrate interfaces and objective without changing the connectome."""
import copy
import torch
from torch import nn
from ..nbnm_sensation import EXTENSION_WIDTH, SCHEMA
from ..th08_attack_readout import ATTACK_TASK_CONTEXTS, CRITIC_TASK_CONTEXTS, TH08AttackReadout


def configure_model(model, config, *, reset_critics=True, reset_collection_line=False):
    from ..games.item_collection import validate
    collection = validate(config.get('item_collection_input', False))
    if not collection and getattr(model, 'item_collection_input', False):
        raise ValueError('Use the checkpoint collection observation when loading trained collection ports')
    previous_corridor=(getattr(model,'nbnm',None) or {}).get('corridor_risk') or {}
    corridor=config.get('corridor_risk') or {}
    if previous_corridor and (not corridor or
            (previous_corridor.get('witness_weight') and not corridor.get('witness_weight'))):
        raise ValueError('Use the compatible corridor configuration for this checkpoint')
    if previous_corridor.get('keep_course') and not corridor.get('keep_course'):
        raise ValueError('Use the configured retained-course readout for this checkpoint')
    if previous_corridor.get('endpoint_risk') and not corridor.get('endpoint_risk'):
        raise ValueError('Use the configured endpoint-risk readout for this checkpoint')
    previous_warning = (getattr(model, 'nbnm', None) or {}).get('native_warning48') or {}
    model.nbnm = copy.deepcopy(config)
    observed_attack = config.get("task_context") in ATTACK_TASK_CONTEXTS
    if observed_attack and getattr(model, "native_attack_readout", None) is None:
        model.native_attack_readout = TH08AttackReadout(model.route[1])
    if not observed_attack and getattr(model, "native_attack_readout", None) is not None:
        raise ValueError("Use the saved original TH08 attack context with this checkpoint")
    warning = config.get('native_warning48')
    if warning is not None and (getattr(model, 'warning48_head', None) is None
            or previous_warning.get('revision') != warning['revision']):
        from ..nbnm_native_warning import NativeWarning48
        model.warning48_head = NativeWarning48(warning, model.device, 2*len(model.core.output_indices))
    elif warning is None and hasattr(model, 'warning48_head'):
        del model.warning48_head
    model.input_schema = SCHEMA
    if config.get('native_rule_geometry'):
        from ..nbnm_geometry_revision import schema
        model.input_schema=schema(config['native_rule_geometry'])
    encoder = model.core.input_encoder
    if encoder.extended_context is None:
        encoder.extended_context = nn.Linear(EXTENSION_WIDTH, encoder.residual.out_features,
            bias=False, device=model.device)
        nn.init.zeros_(encoder.extended_context.weight)
    model.core.sensory_width = 47 + EXTENSION_WIDTH
    if config.get('route_prior',{}).get('enabled'):
        from ..nbnm_route_atlas import WIDTH
        if getattr(encoder,'route_prior_context',None) is None:
            encoder.route_prior_context=nn.Linear(WIDTH,encoder.residual.out_features,
                bias=False,device=model.device)
            nn.init.zeros_(encoder.route_prior_context.weight)
        model.core.sensory_width += WIDTH
        model.input_schema = 'connectome-nbnm-route389-atlas-v1'
    if config.get('future_rules',{}).get('enabled'):
        from ..nbnm_future_sensation import WIDTH,SCHEMA as FUTURE_SCHEMA
        if config.get('route_prior',{}).get('enabled'):raise ValueError('Use one future input experiment at a time')
        if getattr(encoder,'future_rule_context',None) is None:
            encoder.future_rule_context=nn.Linear(WIDTH,encoder.residual.out_features,bias=False,device=model.device)
            nn.init.zeros_(encoder.future_rule_context.weight)
        model.core.sensory_width+=WIDTH;model.input_schema=FUTURE_SCHEMA
    from ..nbnm_corridor import validate as validate_corridor
    if validate_corridor(config.get('corridor_risk')):
        from ..nbnm_corridor import WIDTH as CW, SCHEMA as CS
        if getattr(encoder,'future_rule_context',None) or getattr(encoder,'route_prior_context',None):
            raise ValueError('Corridor input requires its dedicated observation')
        if getattr(encoder,'corridor_context',None) is None:
            encoder.corridor_context=nn.Linear(CW,encoder.residual.out_features,bias=False,device=model.device)
            # Existing opponent VP populations already encode danger and clearance.
            # Map the two long horizons there; old 281-D rehearsal has zero extra drive.
            with torch.no_grad():
                weight=encoder.corridor_context.weight
                weight.zero_()
                polarity=encoder.base.polarity.to(weight)
                feature=encoder.base.feature
                for j in range(2):
                    mask=(feature==j+1).to(weight)
                    weight[:,j].copy_(-.5*mask*polarity)
                    weight[:,2+j].copy_(-.25*mask*polarity)
                weight[:,4:6].copy_(encoder.context.weight[:,36:38]*.25)
        if getattr(model,'corridor_head',None) is None:
            model.corridor_head=nn.Linear(2*len(model.core.output_indices),2,device=model.device)
            nn.init.zeros_(model.corridor_head.weight)
            nn.init.zeros_(model.corridor_head.bias)
        if config['corridor_risk'].get('witness_weight') and getattr(model,'corridor_witness_head',None) is None:
            model.corridor_witness_head=nn.Linear(2*len(model.core.output_indices),2,bias=True,device=model.device)
            nn.init.zeros_(model.corridor_witness_head.weight)
            nn.init.zeros_(model.corridor_witness_head.bias)
            with torch.no_grad():
                for j in range(2):
                    opposed=((encoder.base.feature==3+j)*encoder.base.polarity).to(encoder.corridor_context.weight)
                    encoder.corridor_context.weight[:,2+j].add_(opposed)
        if corridor.get('keep_course') and getattr(model,'corridor_course_head',None) is None:
            model.corridor_course_head=nn.Linear(2*len(model.core.output_indices),1,device=model.device)
            nn.init.zeros_(model.corridor_course_head.weight)
            nn.init.zeros_(model.corridor_course_head.bias)
        if corridor.get('endpoint_risk') and getattr(model,'corridor_endpoint_head',None) is None:
            model.corridor_endpoint_head=nn.Linear(2*len(model.core.output_indices),1,device=model.device)
            nn.init.zeros_(model.corridor_endpoint_head.weight)
            nn.init.zeros_(model.corridor_endpoint_head.bias)
        encoder.course_port_gain=corridor.get('course_port_gain',0.)
        encoder.course_port_stages=corridor.get('keep_course_stages')
        if encoder.course_port_gain and not hasattr(encoder,'course_port_population'):
            population=((encoder.base.feature==7)&(encoder.base.polarity>0)).to(encoder.context.weight)
            encoder.register_buffer('course_port_population',population,persistent=False)
        model.core.sensory_width+=CW
        model.input_schema=CS
    if collection:
        model.enable_item_collection(reset_line=reset_collection_line)
    model.bombs_disabled = True
    model.lower_field_default = False
    model.combat_position_guidance_coefficient = 0.
    model.protected_exit_forecast = False
    # Commands last four frames. Legacy long-risk targets assume a constant
    # action. Explicit revisions learn route existence (v1) or observed policy
    # continuations (v2); the latter's long heads use binary native facts.
    # Neither setting weakens the immediate-risk score or bypasses the brain.
    model.movement_risk_horizon_weights=tuple(config.get('movement_risk_horizon_weights',(8.,4.,2.)))
    model.frozen_decoding = None
    model.full_story_action_selection = 'joint_mode'
    model.pad_candidate_batch = False
    encoder.batch_invariant = True
    model.core.cache_inference_parameters = True
    if config.get('task_context') in CRITIC_TASK_CONTEXTS and not hasattr(model, 'task_critic'):
        # Artificial value interface only; no new brain edges or action path.
        model.task_critic = nn.Linear(4, 2*len(model.core.output_indices), bias=False, device=model.device)
        nn.init.zeros_(model.task_critic.weight)
    if reset_critics:
        # The old three critics predicted different returns. Preserve learned
        # brain/risk/movement weights, not numerically incompatible values.
        layers = [m for m in model.critic.modules() if isinstance(m, nn.Linear)]
        if layers:
            nn.init.zeros_(layers[-1].weight)
            if layers[-1].bias is not None: nn.init.zeros_(layers[-1].bias)
    return model
