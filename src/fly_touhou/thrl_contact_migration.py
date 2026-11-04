"""Explicit same-shape sensor correction; never relabel old collected data."""
from copy import deepcopy
from .thrl_inputs import SEPARATED_OBJECT_PRIORITY_INPUT as OLD,SEPARATED_CONTACT_INPUT as NEW
from .resource_teacher_v7 import REVISION as OLD_TEACHER
from .resource_teacher_v8 import REVISION as NEW_TEACHER
from .thrl_movement_learning import ONLINE_REVISION,CONTACT_ONLINE_REVISION

SCHEMA='B1-HP-independent-contact-PPO-parent-migration-v1'


def contact_parent(payload,*,evidence):
    if (payload.get('schema')!='th14-thrl-training-checkpoint-v1'
            or payload.get('optimizer_transaction_complete') is not True):
        raise ValueError('Contact migration requires a complete real PPO transaction')
    cfg=payload['model_config'];wrapper=cfg.get('pair_risk_wrapper',{})
    if (cfg.get('input_schema')!=OLD or wrapper.get('base_model_config',{}).get('input_schema')!=OLD
            or wrapper.get('frozen_readout') is not True):
        raise ValueError('Only the declared frozen-readout B1 V3 policy can be migrated')
    if (payload['config'].get('teacher_revision')!=OLD_TEACHER
            or payload['config'].get('movement_guidance_revision')!=ONLINE_REVISION):
        raise ValueError('Unexpected prior teacher/guidance semantics')
    if (not isinstance(evidence,dict) or evidence.get('schema')!='B1-HP-independent-contact-sensor-native-qualification-v1'
            or evidence.get('all_historical_V3_inputs_exact') is not True
            or evidence.get('snapshots')!=20 or evidence.get('optimizer_updates')!=0
            or not evidence.get('qualification_sha256') or not evidence.get('source_checkpoint_sha256')):
        raise ValueError('The measured native qualification and source identities are required')
    result=deepcopy(payload)
    new_cfg=result['model_config'];new_cfg['input_schema']=NEW
    new_cfg['pair_risk_wrapper']['base_model_config']['input_schema']=NEW
    c=result['config'];c.update(input_schema=NEW,teacher_revision=NEW_TEACHER,
        movement_guidance_revision=CONTACT_ONLINE_REVISION,
        model_config=deepcopy(new_cfg),policy_model_config=deepcopy(new_cfg))
    c['model_overrides']={**c.get('model_overrides',{}),'input_schema':NEW}
    # Configured observations are metadata; no old arrays or development score
    # may be promoted into the corrected sensor's actual native baseline.
    from .thrl_inputs import make_encoder
    c['observation']=make_encoder(NEW).specification()
    for key in ('sealed_avoidance_baseline','baseline_resume','latest_development','best_development'):
        c.pop(key,None)
    result['input_migration']=dict(schema=SCHEMA,old_input=OLD,new_input=NEW,
        old_teacher=OLD_TEACHER,new_teacher=NEW_TEACHER,
        qualification_sha256=evidence['qualification_sha256'],
        source_checkpoint_sha256=evidence['source_checkpoint_sha256'],
        shapes_and_parameter_names_unchanged=True,all_weights_Adam_and_RNG_preserved=True,
        new_optimizer_updates=0,old_rollout_reused=False,old_demonstrations_relabelled=False,
        old_development_scores_reused=False,requires_new_native_baseline=True,
        measured_fix='retain native contactable enemy bodies independently of HP;teacher uses full enemy dimensions as full dimensions',
        migrated_policy_already_evasive=False)
    return result
