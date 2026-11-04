"""Explicit readout/LR changes only between completed PPO rollouts."""
import math
from .thrl_study_io import read_shared_json


def apply_control(path,model,learning_rate,*,seen_revision=0):
    if not path.exists():return seen_revision,None
    data=read_shared_json(path);revision=data['revision']
    if type(revision) is not int or revision<1:raise ValueError('Positive integer runtime control revision required')
    if revision<=seen_revision:return seen_revision,None
    if set(data)-{'revision','risk_input_soft_clip','policy_lr_scale','context_policy_lr_multiplier','policy_group_multipliers','kl_early_stop','adaptive_policy_lr','target_kl','progress_lr_schedule','neural_risk_scale','bomb_urgency_gain','protection_expiry_inclusive','native_curve_motion_forecast','native_enemy_damage_bound'}:raise ValueError('Unknown runtime control option')
    damage_bound=data.get('native_enemy_damage_bound',getattr(model,'native_enemy_damage_bound',False))
    if type(damage_bound) is not bool:raise ValueError('Boolean native enemy damage bound required')
    curve_motion=data.get('native_curve_motion_forecast',getattr(model,'native_curve_motion_forecast',False))
    if type(curve_motion) is not bool:raise ValueError('Boolean native curve motion mode required')
    if curve_motion and not getattr(model,'native_curve_laser_geometry',False):raise ValueError('Curve motion needs measured native curve geometry')
    inclusive=data.get('protection_expiry_inclusive',getattr(model,'protection_expiry_inclusive',False))
    if type(inclusive) is not bool:raise ValueError('Boolean protection expiry mode required')
    if inclusive and not (model.phase_valid_control and model.protection_aware_risk and model.soft_risk_composition):
        raise ValueError('Inclusive expiry requires finite protection-aware risk')
    from .compact_story_risk_composition import checked_risk_scale
    risk_scale=checked_risk_scale(data.get('neural_risk_scale',getattr(model,'neural_risk_scale',1.)))
    urgency=data.get('bomb_urgency_gain')
    if 'bomb_urgency_gain' in data:
        if (isinstance(urgency,bool) or not isinstance(urgency,(int,float))
                or not math.isfinite(urgency) or urgency<0):raise ValueError('Finite nonnegative Bomb urgency gain required')
        if not hasattr(model,'bomb_urgency_gain'):raise ValueError('Policy has no Bomb urgency parameter')
    clip=data.get('risk_input_soft_clip',model.risk.input_soft_clip)
    if clip is not None and (isinstance(clip,bool) or not isinstance(clip,(int,float)) or not math.isfinite(clip) or not 1<=clip<=1000):
        raise ValueError('Finite readout soft limit[1,1000] or null required')
    scale=data.get('policy_lr_scale',learning_rate.scale)
    if isinstance(scale,bool) or not isinstance(scale,(int,float)) or not math.isfinite(scale) or not learning_rate.min_scale<=scale<=1:
        raise ValueError('Finite policy LR multiplier within the configured minimum and1 required')
    multipliers=data.get('policy_group_multipliers',{})
    if not isinstance(multipliers,dict):raise ValueError('Named policy-group multiplier mapping required')
    multipliers=dict(multipliers)
    if 'context_policy_lr_multiplier' in data:
        context=data['context_policy_lr_multiplier']
        if 'new_context_ports' in multipliers and multipliers['new_context_ports']!=context:
            raise ValueError('Conflicting context-port multiplier settings')
        multipliers['new_context_ports']=context
    for name,multiplier in multipliers.items():learning_rate.validate_group_multiplier(name,multiplier)
    adaptive=data.get('adaptive_policy_lr',learning_rate.adaptive)
    stopping=data.get('kl_early_stop',learning_rate.kl_early_stop)
    target=data.get('target_kl',learning_rate.target)
    if type(adaptive) is not bool or type(stopping) is not bool:raise ValueError('Boolean KL control required')
    if (adaptive or stopping) and (isinstance(target,bool) or not isinstance(target,(int,float)) or not math.isfinite(target) or target<=0):
        raise ValueError('Enabled KL control requires a positive finite target')
    schedule=learning_rate.validate_progress_schedule(data.get('progress_lr_schedule',getattr(learning_rate,'progress_schedule',None)))
    if adaptive and schedule is not None:raise ValueError('Progress schedule cannot use KL-adaptive rates')
    before=dict(risk_input_soft_clip=model.risk.input_soft_clip,policy_lr_scale=learning_rate.scale,
        neural_risk_scale=getattr(model,'neural_risk_scale',1.),
        native_curve_motion_forecast=getattr(model,'native_curve_motion_forecast',False),
        native_enemy_damage_bound=getattr(model,'native_enemy_damage_bound',False),
        protection_expiry_inclusive=getattr(model,'protection_expiry_inclusive',False))
    before.update(kl_early_stop=learning_rate.kl_early_stop,adaptive_policy_lr=learning_rate.adaptive,target_kl=learning_rate.target)
    learning_rate.configure_kl(adaptive=adaptive,kl_early_stop=stopping,target_kl=target)
    model.risk.input_soft_clip=clip
    if 'native_enemy_damage_bound' in data:
        from .thrl_durable_records import write_json
        marker=path.parent/'native-enemy-damage-bound-enabled.json'
        if damage_bound:write_json(marker,dict(enabled=True,control_revision=revision,scope='next student collection'))
        else:marker.unlink(missing_ok=True)
        model.native_enemy_damage_bound=damage_bound
    if 'native_curve_motion_forecast' in data:
        from .thrl_durable_records import write_json
        marker=path.parent/'native-curve-motion-enabled.json'
        if curve_motion:write_json(marker,dict(enabled=True,control_revision=revision,scope='next student collection'))
        else:marker.unlink(missing_ok=True)
        model.native_curve_motion_forecast=curve_motion
    if 'protection_expiry_inclusive' in data:model.protection_expiry_inclusive=inclusive
    if 'neural_risk_scale' in data:model.neural_risk_scale=risk_scale
    if 'policy_lr_scale' in data:learning_rate.scale=scale;learning_rate.apply()
    group_changes=learning_rate.set_group_multipliers(multipliers)
    if 'progress_lr_schedule' in data:learning_rate.configure_progress(schedule)
    bomb_change=None
    if urgency is not None:
        import torch
        parameter=model.bomb_urgency_gain
        previous=float(parameter.detach())
        with torch.no_grad():
            parameter.fill_(urgency)
            # The saved FP64 master must move with this explicit scalar edit,
            # otherwise the next Adam step would restore the old ~12 prior.
            master=getattr(learning_rate.optimizer,'_policy_masters',{}).get(parameter)
            if master is not None:master.copy_(parameter)
        bomb_change=dict(before=previous,after=urgency,Adam_moments_preserved=True,
            master_copy_synchronized=master is not None,learned_Bomb_head_unchanged=True,
            ordinary_Bomb_still_legal=True,parameter_remains_trainable=parameter.requires_grad)
    group_change=next((change for change in group_changes if change['group']=='new_context_ports'),None)
    return revision,dict(revision=revision,before=before,after=dict(risk_input_soft_clip=clip,policy_lr_scale=learning_rate.scale,
        kl_early_stop=learning_rate.kl_early_stop,adaptive_policy_lr=learning_rate.adaptive,target_kl=learning_rate.target,
        progress_lr_schedule=schedule,progress_lr_factor=learning_rate.progress_factor(),neural_risk_scale=risk_scale,
        protection_expiry_inclusive=inclusive,
        native_curve_motion_forecast=curve_motion,
        native_enemy_damage_bound=damage_bound,
        learning_rates={group['name']:group['lr'] for group in learning_rate.optimizer.param_groups}),
        context_policy_lr_change=group_change,
        Bomb_urgency_change=bomb_change,
        policy_group_lr_changes=group_changes,
        boundary='after prior rollout learning,before next student collection',native_worlds_reset=False,brain_state_reset=False,
        previous_PPO_samples_reused_with_changed_policy=False)
