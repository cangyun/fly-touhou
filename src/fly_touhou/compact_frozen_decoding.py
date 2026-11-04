"""Opt-in frozen neural decoding; no teacher or raw hazard action bypass."""
import torch
from .compact_story_risk_composition import protection_exposure

REVISION='frozen-neural-route-and-shield-suffix-v1'


def reweight(logits,probability,route,features,*,temperature,risk_scale,
             inclusive_expiry,options=None,protected_exit=False,protected_from_stage=1):
    if options is None and not protected_exit:return logits
    options=options or {}
    controllable=features[...,14]>.5
    u=features[...,13];remaining=60*u/(1-u).clamp_min(1e-6)
    stage=features[...,111]*6
    chosen_stage=stage>=float(options.get('from_stage',6))-1e-5
    exposure=protection_exposure(features,inclusive_expiry=inclusive_expiry)
    suffix=protected_exit&controllable&(remaining>0)&(remaining<=24.001)&(stage>=protected_from_stage-1e-5)
    # The sensory margins already ignore protected ticks. Multiplying the
    # resulting neural risk by exposed-duration fractions would discount twice.
    active=(remaining[...,None]<features.new_tensor([4.,12.,24.])+1).to(features.dtype)
    revised_exposure=torch.where(suffix[...,None],active,exposure)
    base_weights=features.new_tensor([8.,4.,2.])
    fraction=float(options.get('long_risk_fraction',1.))
    revised_weights=features.new_tensor([8.,4.*fraction,2.*fraction])
    weights=torch.where(chosen_stage[...,None],revised_weights,base_weights)
    old=-risk_scale*(probability*exposure[...,None,:]*base_weights).sum(-1)
    new=-risk_scale*(probability*revised_exposure[...,None,:]*weights[...,None,:]).sum(-1)
    gain=float(options.get('route_gain',1.))
    delta=new-old+(gain-1.)*route/temperature*chosen_stage[...,None]
    # This is a weak counterfactual positioning preference while protected,
    # not a claim that those bullets can currently hit the player.
    early=(remaining>24.001)&controllable&chosen_stage
    delta-=float(options.get('early_neural_weight',0.))*probability[...,2]*early[...,None]
    original=logits[...,:18];movement=original+delta
    movement=movement-movement.max(-1,keepdim=True).values
    mass=movement.logsumexp(-1,keepdim=True)-original.logsumexp(-1,keepdim=True)
    changed=torch.cat((movement,logits[...,18:19]+mass),-1)
    enabled=controllable&(suffix|chosen_stage)
    return torch.where(enabled[...,None],changed,logits)


def configure(model,config):
    options=config.get('compact_frozen_decoding')
    if options is None:return None
    if not (model.soft_risk_composition and model.phase_valid_control and model.protection_aware_risk):
        raise ValueError('Frozen decoding requires the existing finite protection-aware neural policy')
    options=dict(options)
    if not 1<=int(options.get('from_stage',6))<=6:raise ValueError('Original stage1..6 required')
    model.frozen_decoding=options
    model.protected_exit_forecast=bool(options.get('protected_exit',False))
    model.protected_exit_from_stage=int(options.get('from_stage',6))
    model.protected_exit_preserve_long_shield=True
    config.update(compact_protected_exit_forecast=model.protected_exit_forecast,
        compact_protected_exit_from_stage=model.protected_exit_from_stage,
        compact_protected_exit_preserve_long_shield=True)
    return dict(revision=REVISION,options=options,parameters_changed=False,teacher_actions=False,
        raw_geometry_action_bypass=False,preserves_Bomb_family=True,
        inputs='Original protected inputs above24f;exposed suffix inside24f;original inputs before the selected stage')
