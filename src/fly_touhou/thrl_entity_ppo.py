"""Audited PPO transfer and sealed development evidence for one B1 policy.

No game is created and no optimizer update is performed by these helpers.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_parent(torch,base,entity):
    from .thrl_pair_policy import model_from_config
    if base.get('schema')!='th14-thrl-training-checkpoint-v1' or base.get('optimizer_transaction_complete') is not True:
        raise ValueError('A complete real PPO transaction is required')
    cfg=deepcopy(entity['model_config']);wrapper=cfg['pair_risk_wrapper']
    if wrapper['base_model_config']!=base['model_config']:
        raise ValueError('The fitted readout must wrap exactly the declared PPO body')
    wrapper['frozen_readout']=True
    model=model_from_config(**cfg);model.load_state_dict(entity['model'],strict=True)
    for name,value in base['model'].items():
        if not torch.equal(model.base.state_dict()[name],value):
            raise ValueError('Readout preparation changed the original PPO body: '+name)
    source_names=base['optimizer_parameter_names'];target_names=list(dict(model.named_parameters()))
    if target_names[:len(source_names)]!=['base.'+n for n in source_names]:
        raise ValueError('PPO parameter order is not an exact base prefix')
    old=base['optimizer']
    if len(old['param_groups'])!=1 or len(old['param_groups'][0]['params'])!=len(source_names):
        raise ValueError('Expected one complete original Adam group')
    states={}
    for index,old_id in enumerate(old['param_groups'][0]['params']):
        if old_id in old['state']:states[index]=deepcopy(old['state'][old_id])
    group={**deepcopy(old['param_groups'][0]),'params':list(range(len(target_names)))}
    result=deepcopy(base)
    result.update(model_config=cfg,model=model.state_dict(),
        optimizer=dict(state=states,param_groups=[group]),optimizer_parameter_names=target_names,
        entity_PPO_transfer=dict(schema='exact-PPO-body-and-Adam-plus-frozen-entity-readout-v1',
            source_Adam_states=len(old['state']),preserved_Adam_states=len(states),
            frozen_readout_parameters=sum(p.numel() for p in model.risk.parameters()),
            base_weights_changed=False,new_PPO_updates=0,world_state_restored=False,
            readout_is_not_a_teacher_execution_override=True))
    rollout=base.get('config',{}).get('rollout_total_decisions')
    lifetime_updates=base.get('lifetime_rollout_updates')
    if type(rollout) is int and type(lifetime_updates) is int:
        optimized=rollout*lifetime_updates
        literal=base.get('lifetime_decisions',optimized)
        result.update(lifetime_decisions=optimized,lifetime_optimized_decisions=optimized,
            lifetime_paid_decisions_lower_bound=max(literal,optimized))
        result['entity_PPO_transfer']['counter_provenance']=dict(
            source_lifetime_decisions_literal=literal,lifetime_optimized_decisions=optimized,
            calculation='completed lifetime PPO rollouts * unchanged3840 decisions per rollout',
            source_paid_prefix_retained=True)
    return result


def sealed_avoidance_baseline(torch,model,contract):
    from .thrl_study_io import read_shared_json
    from .thrl_avoidance_development import SCHEMA
    report_path=Path(contract['report']);checkpoint=Path(contract['checkpoint'])
    if sha256(report_path)!=contract['report_sha256'] or sha256(checkpoint)!=contract['checkpoint_sha256']:
        raise ValueError('Sealed baseline identity changed')
    report=read_shared_json(report_path)
    direct=contract.get('format')==SCHEMA
    if direct:
        plan=read_shared_json(report_path.parent/'episode_plan.json')
        if (report.get('schema')!=SCHEMA or plan.get('schema')!=SCHEMA
                or contract.get('input_schema')!=model.input_schema
                or report.get('native_seed_split')!='development'
                or report.get('training_episode_rule_changed') is not False
                or report.get('training_Bomb_rule_changed') is not False):
            raise ValueError('New native baseline must use the same explicit input and movement-only development protocol')
        cap=report.get('diagnostic_native_frame_cap',plan.get('diagnostic_native_frame_cap'))
        legacy=contract.get('legacy_cap_source')
        if legacy:
            path=Path(legacy['config'])
            if sha256(path)!=legacy['sha256']:raise ValueError('Sealed original baseline cap configuration changed')
            source_config=read_shared_json(path)
            source_cap=source_config.get('avoidance_development_max_frames')
            if cap is not None and cap!=source_cap:raise ValueError('Baseline report and original configuration cap differ')
            cap=source_cap
        if cap is not None and cap!=contract['max_frames']:raise ValueError('Baseline declared native cap differs')
    else:
        plan=read_shared_json(report_path.parent/'plan.json')
    if not direct and (report['native_seed_split']!='development' or report.get('training_modified') is not False
            or plan['diagnostic_native_frame_cap']!=contract['max_frames']
            or plan['Bomb_forbidden_in_all_conditions'] is not True):
        raise ValueError('Baseline must be a frozen no-Bomb development comparison')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if direct and payload.get('model_config',{}).get('input_schema')!=contract['input_schema']:
        raise ValueError('The baseline checkpoint sensor semantics differ')
    target=model.state_dict()
    if target.keys()!=payload['model'].keys() or any(not torch.equal(target[k].detach().cpu(),v) for k,v in payload['model'].items()):
        raise ValueError('Reused native baseline did not execute these initial weights')
    rows=sorted((r for r in report['episode_results'] if direct or r['policy']==contract['condition']),key=lambda r:r['game'])
    if len(rows)!=contract['episodes'] or [r['game'] for r in rows]!=list(range(len(rows))):
        raise ValueError('Baseline episodes are missing or duplicated')
    for row in rows:
        env,action=np.random.SeedSequence([contract['seed'],row['game']]).generate_state(2)
        if (row['environment_seed' if direct else 'evaluation_seed']!=int(env) or row['action_seed']!=int(action)
                or row['actual_bombs_used']!=0 or row.get('seed_applied') is not True):
            raise ValueError('Baseline native/action RNG or no-Bomb provenance differs')
        if direct:
            raw=row['native_frames']
            if (row.get('right_censored_at_cap') and raw<contract['max_frames']) or (raw>contract['max_frames'] and cap is None):
                raise ValueError('Baseline cap differs from the declared diagnostic horizon')
    from .thrl_probe_metrics import score_frames,endpoint_fields
    if direct:rows=[{**r,**endpoint_fields(r['native_frames'],contract['max_frames'])} for r in rows]
    values=np.asarray([score_frames(r,contract['max_frames']) for r in rows]);mean=float(values.mean())
    return dict(schema=SCHEMA,episodes=len(rows),score=mean,
        mean_first_collision_capped_frames=mean,median_first_collision_capped_frames=float(np.median(values)),
        observed_collisions=sum(bool(r['first_collision_observed']) for r in rows),
        right_censored_at_cap=sum(bool(r['right_censored_at_cap']) for r in rows),episode_results=rows,
        score_definition='mean capped native frames to first observed collision with Bomb forbidden; diagnostic only',
        diagnostic_native_frame_cap=contract['max_frames'],raw_endpoint_mean_native_frames=float(np.mean([r['native_frames'] for r in rows])),
        full_game_clear_rate_measured=False,formal_test=False,original_game_rng_controlled=True,
        native_seed_split='development',training_episode_rule_changed=False,training_Bomb_rule_changed=False,
        reused_existing_games=True,new_native_games=0,source_contract=deepcopy(contract))
