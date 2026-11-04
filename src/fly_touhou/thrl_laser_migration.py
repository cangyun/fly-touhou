"""Explicit V4->V5 model/Adam expansion, without relabeling any old rollout."""
from copy import deepcopy

SCHEMA='B1-observed-laser-segment-PPO-parent-migration-v1'
EXPANDED_PARAMETER='base.object_attention.object_encoder.0.weight'


def laser_parent(torch,payload,laser,*,sensor_evidence,fit_evidence):
    from .thrl_inputs import SEPARATED_CONTACT_INPUT as OLD,SEPARATED_LASER_INPUT as NEW,make_encoder
    from .resource_teacher_v8 import REVISION as OLD_TEACHER
    from .resource_teacher_v9 import REVISION as NEW_TEACHER
    from .thrl_movement_learning import CONTACT_ONLINE_REVISION,LASER_ONLINE_REVISION
    from .thrl_laser_risk import QUERY_SCHEMA
    from .thrl_pair_policy import model_from_config
    if (payload.get('schema')!='th14-thrl-training-checkpoint-v1'
            or payload.get('optimizer_transaction_complete') is not True):
        raise ValueError('A completed sealed original PPO transaction is required')
    cfg=payload['model_config'];wrapper=cfg.get('pair_risk_wrapper',{})
    if (cfg.get('input_schema')!=OLD or wrapper.get('base_model_config',{}).get('input_schema')!=OLD
            or wrapper.get('frozen_readout') is not True or 'laser_width' in wrapper):
        raise ValueError('Only the explicitly frozen-readout V4 B1 policy can be expanded')
    if (payload['config'].get('teacher_revision')!=OLD_TEACHER
            or payload['config'].get('movement_guidance_revision')!=CONTACT_ONLINE_REVISION):
        raise ValueError('Unexpected prior teacher/guidance semantics')
    if (sensor_evidence.get('schema')!='B1-observed-laser-V5-sensor-qualification-v1'
            or sensor_evidence.get('status')!='qualified'
            or sensor_evidence.get('old_V4_all_snapshots_exact') is not True
            or sensor_evidence.get('continuous_motion_pairs')!=890
            or not sensor_evidence.get('qualification_sha256')):
        raise ValueError('Pinned real native sensor transport qualification required')
    if (fit_evidence.get('schema')!='B1-laser-risk-proxy-fit-result-v1'
            or fit_evidence.get('qualified_proxy_only') is not True
            or not fit_evidence.get('fit_report_sha256')
            or laser.get('schema')!='B1-learned-laser-segment-risk-v1' or laser.get('query_schema')!=QUERY_SCHEMA
            or laser.get('Adam_steps')!=fit_evidence.get('selected_Adam_steps')):
        raise ValueError('A sealed qualified CUDA laser proxy fit is required')
    result=deepcopy(payload);new_cfg=result['model_config'];new_cfg['input_schema']=NEW
    w=new_cfg['pair_risk_wrapper'];w['base_model_config']['input_schema']=NEW
    w.update(laser_width=laser['width'],laser_query_schema=QUERY_SCHEMA,laser_buffer=4.,laser_risk_width=1.)
    model=model_from_config(**new_cfg);state=model.state_dict()
    for name,value in payload['model'].items():
        if name==EXPANDED_PARAMETER:
            if value.ndim!=2 or state[name].shape!=(value.shape[0],value.shape[1]+8):
                raise ValueError('Expected exactly8 new independent laser descriptor columns')
            expanded=torch.zeros_like(state[name]);expanded[:,:value.shape[1]]=value;state[name]=expanded
        else:
            if name not in state or state[name].shape!=value.shape:raise ValueError('Unexpected changed shared tensor: '+name)
            state[name]=value.clone()
    if not any(name==EXPANDED_PARAMETER for name in payload['model']):raise ValueError('Missing old object projection')
    if set(laser['model'])!=set(model.laser_risk.state_dict()):raise ValueError('Laser readout state keys differ')
    for name,value in laser['model'].items():
        key='laser_risk.'+name
        if state[key].shape!=value.shape:raise ValueError('Laser fitted tensor shape differs')
        state[key]=value.detach().cpu().clone()
    model.load_state_dict(state,strict=True);result['model']=model.state_dict()
    old_names=payload['optimizer_parameter_names'];names=list(dict(model.named_parameters()))
    if names[:len(old_names)]!=old_names:raise ValueError('Shared optimizer parameter order changed')
    old=payload['optimizer']
    if len(old['param_groups'])!=1 or len(old['param_groups'][0]['params'])!=len(old_names):
        raise ValueError('Expected one complete original Adam group')
    old_ids=dict(zip(old_names,old['param_groups'][0]['params']));new_ids={name:i for i,name in enumerate(names)}
    states={};expanded_moments=0
    for name,old_id in old_ids.items():
        if old_id not in old['state']:continue
        value=deepcopy(old['state'][old_id])
        for moment in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
            if moment not in value:continue
            if name==EXPANDED_PARAMETER:
                pad=torch.zeros_like(result['model'][name]);pad[:,:value[moment].shape[1]]=value[moment]
                value[moment]=pad;expanded_moments+=1
            elif value[moment].shape!=result['model'][name].shape:raise ValueError('Malformed Adam moment')
        states[new_ids[name]]=value
    result['optimizer']=dict(state=states,param_groups=[{**deepcopy(old['param_groups'][0]),'params':list(range(len(names)))}])
    result['optimizer_parameter_names']=names
    c=result['config'];c.update(input_schema=NEW,teacher_revision=NEW_TEACHER,
        movement_guidance_revision=LASER_ONLINE_REVISION,model_config=deepcopy(new_cfg),policy_model_config=deepcopy(new_cfg))
    c['model_overrides']={**c.get('model_overrides',{}),'input_schema':NEW}
    c['observation']=make_encoder(NEW).specification()
    for key in ('sealed_avoidance_baseline','baseline_resume','latest_development','best_development'):
        c.pop(key,None)
    if result.get('input_migration'):
        result.setdefault('prior_input_migrations',[]).append(deepcopy(result['input_migration']))
    result['input_migration']=dict(schema=SCHEMA,old_input=OLD,new_input=NEW,
        old_teacher=OLD_TEACHER,new_teacher=NEW_TEACHER,
        sensor_qualification_sha256=sensor_evidence['qualification_sha256'],
        fitted_laser_checkpoint_sha256=fit_evidence['checkpoint_sha256'],fit_report_sha256=fit_evidence['fit_report_sha256'],
        shared_weights_and_Adam_columns_preserved=True,object_projection_added_zero_columns=8,
        expanded_Adam_moment_arrays=expanded_moments,preserved_Adam_states=len(states),new_laser_readout_frozen=True,
        new_PPO_optimizer_updates=0,sampling_RNG_preserved=True,old_rollout_reused=False,
        old_demonstrations_relabelled=False,old_development_scores_reused=False,requires_new_native_baseline=True,
        laser_margin=dict(buffer=4.,risk_width=1.,basis='baseline1unit +3unit measured endpoint transition allowance;empirical allowance is not a future bound'),
        unverified_native_laser_collision_extent=True,curve_motion_remains_unknown=True,
        migrated_policy_already_evasive=False)
    return result
