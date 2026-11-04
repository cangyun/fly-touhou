"""Audited shared-weight / Adam warm starts for controlled architecture branches.

No native game is created here. An attention branch inherits all existing
parameter tensors and their optimizer moments; only genuinely new parameters
start without Adam state. Native worlds and recurrent state start afresh.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import shutil

from .atomic_json_io import atomic_write_json
from .thrl_study_io import read_shared_json
from .thrl_limits import MAX_NATIVE_WORKERS


def warm_start(torch, model, optimizer, checkpoint, expected_sha256, *, exact_optimizer=False,model_factory=None):
    from .thrl_model import MOActorCritic

    checkpoint = Path(checkpoint)
    actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError('Warm-start checkpoint identity changed')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    from .thrl_parameter_transfer import TRANSFER_SCHEMA, apply_parameter_transfer
    if payload.get('schema') == TRANSFER_SCHEMA:
        return apply_parameter_transfer(torch,model,optimizer,payload,checkpoint=checkpoint,
            checkpoint_sha256=actual,exact_optimizer=exact_optimizer)
    if payload.get('schema') != 'th14-thrl-training-checkpoint-v1':
        raise ValueError('An explicit THRL model/Adam/RNG checkpoint is required')
    source = (model_factory or MOActorCritic)(**payload['model_config'])
    source.load_state_dict(payload['model'], strict=True)
    if exact_optimizer:
        source_config, target_config = source.model_config(), model.model_config()
        if source_config.get('attention') == target_config.get('attention') == 'none':
            # This option creates no state or computations in a none model.
            source_config.pop('attention_initialization', None)
            target_config.pop('attention_initialization', None)
        if source_config != target_config:
            raise ValueError('Candidate recovery cannot change the model architecture')
        if payload.get('optimizer_transaction_complete') is False:
            raise ValueError('Cannot resume an interrupted partial optimizer transaction')
    source_names = list(dict(source.named_parameters()))
    recorded_names = payload.get('optimizer_parameter_names', source_names)
    if recorded_names != source_names:
        raise ValueError('Optimizer order does not match the declared source model')
    target_state = model.state_dict()
    for name, value in payload['model'].items():
        if name not in target_state or target_state[name].shape != value.shape:
            raise ValueError('Incompatible shared parameter or buffer: '+name)
        if exact_optimizer and target_state[name].dtype != value.dtype:
            raise ValueError('Candidate recovery cannot silently change tensor dtype: '+name)
    missing = set(target_state)-set(payload['model'])
    if any(not ('.spatial_attention.' in name or name.startswith('object_attention.'))
           for name in missing):
        raise ValueError('Only added attention state may be absent: '+repr(sorted(missing)))
    if missing and model.model_config().get('attention_initialization') != 'identity':
        raise ValueError('Architecture expansion must begin as an identity residual')
    if exact_optimizer and (missing or source_names != list(dict(model.named_parameters()))):
        raise ValueError('Candidate recovery requires exactly the same parameter names/order')
    merged = {**target_state, **payload['model']}
    model.load_state_dict(merged, strict=True)

    old = payload['optimizer']
    new = optimizer.state_dict()
    if len(old['param_groups']) != 1 or len(new['param_groups']) != 1:
        raise ValueError('This migration requires one Adam parameter group')
    old_ids = old['param_groups'][0]['params']
    if len(old_ids) != len(source_names):
        raise ValueError('Source optimizer is missing parameters')
    old_by_name = dict(zip(source_names, old_ids))
    target_names = list(dict(model.named_parameters()))
    new_by_name = dict(zip(target_names, new['param_groups'][0]['params']))
    if not set(source_names).issubset(new_by_name):
        raise ValueError('Target removed source optimizer parameters')
    states = {}
    for name, old_id in old_by_name.items():
        state = old['state'].get(old_id)
        if state is None:
            continue
        for key in ('exp_avg', 'exp_avg_sq'):
            if key not in state or state[key].shape != target_state[name].shape:
                raise ValueError('Malformed Adam moment: '+name+'/'+key)
        states[new_by_name[name]] = copy.deepcopy(state)
    group = {**copy.deepcopy(old['param_groups'][0]),
             'params':new['param_groups'][0]['params'],
             'lr':(old['param_groups'][0]['lr'] if exact_optimizer else new['param_groups'][0]['lr'])}
    optimizer.load_state_dict({'state':states, 'param_groups':[group]})
    # Model construction and attention initialization must not move the
    # inherited sampling RNG forward. Both branches inherit the same state.
    torch.set_rng_state(payload['torch_rng'])
    if next(model.parameters()).is_cuda:
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
    import numpy as np
    import random
    np.random.set_state(payload['numpy_rng'])
    random.setstate(payload['python_rng'])
    audit = {
        'schema':'thrl-shared-checkpoint-migration-v1',
        'checkpoint':str(checkpoint.resolve()), 'sha256':actual,
        'inherited_decisions':int(payload.get('lifetime_decisions',payload['decisions'])),
        'inherited_rollout_updates':int(payload.get('lifetime_rollout_updates',payload['rollout_updates'])),
        'inherited_adam_minibatch_steps':int(payload.get('lifetime_adam_minibatch_steps',payload['adam_minibatch_steps'])),
        'common_parameter_tensors':len(source_names),
        'common_Adam_states':len(states),
        'new_parameter_names':[name for name in target_names if name not in old_by_name],
        'additional_state_keys':sorted(missing),
        'common_weights_exact':all(torch.equal(model.state_dict()[name].cpu(), value)
                                   for name,value in payload['model'].items()),
        'optimizer_moments_preserved':True, 'sampling_RNG_restored':True,
        'source_learning_rate':old['param_groups'][0]['lr'],
        'branch_learning_rate':group['lr'],
        'world_state_restored':False, 'recurrent_state':'zero in fresh worlds',
        'branch_decisions_restart_at_zero':True,
        'exact_optimizer_recovery':exact_optimizer,
        'checkpoint_candidate_optimized_decisions':payload.get('candidate_optimized_decisions'),
        'checkpoint_candidate_sampled_decisions':payload.get('candidate_decisions',payload['decisions']),
        'checkpoint_candidate_paid_decisions_lower_bound':payload.get('candidate_paid_decisions_lower_bound',
                                                    payload.get('candidate_decisions',payload['decisions'])),
        'source_parameters':copy.deepcopy(payload.get('config',{}).get('parameters')),
        'source_checkpoint_phase':payload.get('checkpoint_phase'),
        'source_rollout_total_decisions':payload.get('config',{}).get('rollout_total_decisions'),
        'curriculum_state':copy.deepcopy(payload.get('curriculum_state')),
    }
    del payload, source
    return audit


def validate_parallel_config(config):
    workers = config['workers']
    rollout = config['rollout_total_decisions']
    if type(workers) is not int or not 1 <= workers <= MAX_NATIVE_WORKERS:
        raise ValueError(f'A branch must have one to {MAX_NATIVE_WORKERS} owned native workers')
    if type(rollout) is not int or rollout % workers or (rollout//workers) % 64:
        raise ValueError('Each worker must supply whole 64-decision GAE chunks')
    if rollout < 2048 or rollout > 16384:
        raise ValueError('Explicit rollout memory budget must be 2048..16384 decisions')
    if config.get('training_recipe')=='th14-completion-v3' and rollout//workers<128:
        raise ValueError('Version3 requires at least 128 consecutive decisions per world')
    if config['trial_decisions'] % rollout:
        raise ValueError('Training budget must end at a whole rollout boundary')
    nodes = config['development_decisions']
    if (not nodes or nodes != sorted(set(nodes)) or nodes[0] <= 0
            or nodes[-1] != config['trial_decisions'] or any(n % rollout for n in nodes)):
        raise ValueError('Development nodes must be positive whole-rollout boundaries')
    if config.get('attention') not in ('none','spatial','object','both'):
        raise ValueError('Unknown explicit architecture')
    if config['policy_action_count'] != 19 or config.get('native_frame_horizon') is not None:
        raise ValueError('Use source19 actions and complete finite-life games')
    if not config.get('finite_native_lives') or config.get('continues_allowed'):
        raise ValueError('Finite native lives and no continues are required')
    if not 1 <= config['development_episodes'] <= 20:
        raise ValueError('Finite independent development quota required')
    if not 0 <= config.get('validation_episodes_per_policy',20) <= 20:
        raise ValueError('Finite independent validation quota required')
    minibatch=256 if config.get('training_recipe')=='th14-completion-v3' else 64
    adam=math.ceil(rollout/minibatch)*3
    return {'workers':workers, 'decisions_per_worker':rollout//workers,
            'rollout_map_bytes':rollout*16*92*96*4,
            'expected_Adam_per_rollout':adam,
            'expected_training_Adam':config['trial_decisions']//rollout*adam,
            'Adam_estimate_is_upper_bound':config.get('training_recipe')=='th14-completion-v3'}


def candidate_trial_config(config, number):
    """Resume only the protected architecture trial; HPO candidates stay fresh.

    The completed rollout prefix is distinct from paid, unoptimized actions.
    Development filenames/scores keep the architecture candidate's original
    decision axis, while segment/checkpoint counters remain literal new work.
    """
    result={**config,'protected_architecture_baseline':number==0,
            'candidate_trial_decisions':config['trial_decisions'],
            'candidate_development_decisions':list(config['development_decisions']),
            'candidate_decision_offset':0,'candidate_paid_decision_offset':0,
            'candidate_resume_applied':False}
    resume=config.get('baseline_resume') if number==0 else None
    if resume:
        prefix=resume['optimized_decisions']
        rollout=config['rollout_total_decisions']
        if (type(prefix) is not int or prefix<0 or prefix%rollout
                or prefix>=config['trial_decisions']):
            raise ValueError('Resume prefix must contain whole committed policy rollouts')
        paid=resume.get('paid_decisions',prefix)
        if type(paid) is not int or paid<prefix:
            raise ValueError('Paid candidate sampling lower bound cannot be below optimized progress')
        result.update(warm_start_checkpoint=resume['checkpoint'],
            warm_start_sha256=resume['sha256'],
            trial_decisions=config['trial_decisions']-prefix,
            development_decisions=[node-prefix for node in config['development_decisions'] if node>prefix],
            candidate_decision_offset=prefix,candidate_paid_decision_offset=paid,
            inherited_unoptimized_sampling_lower_bound=paid-prefix,
            unknown_inflight_decisions=bool(resume.get('unknown_inflight_decisions',True)),
            candidate_resume_applied=True)
    validate_parallel_config(result)
    return result


def _file_record(path):
    path=Path(path).resolve()
    return {'path':str(path),'bytes':path.stat().st_size,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def _copy_verified(source, target):
    before=_file_record(source)
    target=Path(target)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    if _file_record(target)['sha256'] != before['sha256']:
        raise ValueError('Inherited file copy changed: '+str(source))
    return {**before,'copy':str(target.resolve())}


def candidate_history_chain(source_run):
    """Follow legacy resumes; v2 initial.pt is not necessarily candidate initial."""
    chain=[]
    seen=set()
    current=Path(source_run).resolve()
    while True:
        if current in seen or len(chain)>=32:
            raise ValueError('Cyclic or excessive candidate resume history')
        seen.add(current)
        config=read_shared_json(current/'config.json')
        chain.append((current,config))
        resume=config.get('baseline_resume')
        resumed=config.get('candidate_resume_applied',bool(config.get('candidate_decision_offset',0)))
        if not resume or not resumed:
            break
        current=Path(resume.get('source_run',Path(resume['checkpoint']).parent)).resolve()
    return chain


def valid_development_report(config,report):
    """A declared first-hit diagnostic is not a native Game Over episode."""
    rows=report.get('episode_results',[]);quota=config['development_episodes']
    if (report.get('episodes')!=quota or not isinstance(rows,list) or len(rows)!=quota
            or not math.isfinite(report.get('score',float('nan')))):
        return False
    if config.get('development_objective')!='avoidance-first-v1':
        return all(row.get('outcome') in ('clear','gameover') for row in rows)
    from .thrl_avoidance_development import SCHEMA
    from .thrl_avoidance_probe import PROBE_SCHEMA
    if (report.get('schema')!=SCHEMA or report.get('original_game_rng_controlled') is not True
            or report.get('native_seed_split')!='development'):
        return False
    limit=config.get('avoidance_development_max_frames',4096)
    if type(limit) is not int or limit<=0 or report.get('diagnostic_native_frame_cap',limit)!=limit:
        return False
    for row in rows:
        hit,cap=row.get('first_collision_observed'),row.get('right_censored_at_cap')
        seed=row.get('native_seed');frames=row.get('native_frames')
        if (row.get('schema')!=PROBE_SCHEMA or type(hit) is not bool or type(cap) is not bool or hit==cap
                or row.get('actual_bombs_used')!=0 or row.get('seed_applied') is not True
                or type(seed) is not int or not 0<=seed<=65535 or seed%10 not in (6,7)
                or type(frames) is not int or frames<=0):return False
        if cap and frames<limit:return False
        if row.get('capped_score_native_frames',min(frames,limit))!=min(frames,limit):return False
    # Native endpoints may overshoot a diagnostic cap by a few frames. Keep
    # the raw endpoint as evidence while checking the actual capped score.
    mean=sum(min(row['native_frames'],limit) for row in rows)/quota
    return (math.isclose(report['score'],mean,rel_tol=1e-12,abs_tol=1e-9)
        and report.get('observed_collisions')==sum(row['first_collision_observed'] for row in rows)
        and report.get('right_censored_at_cap')==sum(row['right_censored_at_cap'] for row in rows))


def inherit_candidate_history(config, output):
    """Copy sealed reports/best/true initial into a NEW run; no game/model calls."""
    output=Path(output)
    resume=config['baseline_resume']
    prefix=config['candidate_decision_offset']
    source=Path(resume.get('source_run',Path(resume['checkpoint']).parent)).resolve()
    chain=candidate_history_chain(source)
    full_story=config.get('training_mode')=='full-story-v1'
    all_chain=chain
    retired=[]
    if full_story:
        from .thrl_learning_revision import FULL_STORY_AUTHORIZATION
        from .thrl_movement_learning import contract as movement_contract,AUTHORIZATION as MOVEMENT_AUTH
        migration=config.get('learning_migration') or {}
        movement=movement_contract(config)
        chain=[(run,cfg) for run,cfg in all_chain if cfg.get('training_mode')=='full-story-v1'
               and movement_contract(cfg)==movement]
        retired=[(run,cfg) for run,cfg in all_chain if (run,cfg) not in chain]
        old_course=[(run,cfg) for run,cfg in retired if cfg.get('training_mode')!='full-story-v1']
        old_movement=[(run,cfg) for run,cfg in retired if cfg.get('training_mode')=='full-story-v1']
        if config.get('development_objective')=='avoidance-first-v1':
            for _,prior in chain:
                for field in ('development_objective','avoidance_development_seed','avoidance_development_max_frames','development_episodes'):
                    if config.get(field)!=prior.get(field):raise ValueError('No-Bomb recovery development protocol differs: '+field)
        if old_movement and migration.get('schema')!=MOVEMENT_AUTH:
            raise ValueError('Movement scores require an explicit objective migration')
        earlier_full_migration=any((cfg.get('learning_migration') or {}).get('schema')==FULL_STORY_AUTHORIZATION
                                   for _,cfg in all_chain if cfg.get('training_mode')=='full-story-v1')
        if old_course and migration.get('schema')!=FULL_STORY_AUTHORIZATION and not earlier_full_migration:
            raise ValueError('Legacy course scores require an explicit full-story migration')
    nodes=config['candidate_development_decisions']
    completed=[]
    pending_at_start=None
    for node in nodes:
        if node>prefix:
            continue
        candidates=[run/f'development_{node:07d}'/'report.json' for run,_ in chain
                    if (run/f'development_{node:07d}'/'report.json').exists()]
        if not candidates:
            if full_story and retired and node<=migration['boundary_optimized_decisions']:
                # No legacy score or pending course gate may select the new policy.
                continue
            if node==prefix:
                pending_at_start=node
                continue
            raise ValueError('Missing sealed development report before recovery prefix: '+str(node))
        path=candidates[0]
        report=read_shared_json(path)
        if not valid_development_report(config,report):
            raise ValueError('Incomplete sealed development report: '+str(path))
        destination=output/f'development_{node:07d}'
        evidence=_copy_verified(path,destination/'report.json')
        for other in candidates[1:]:
            if _file_record(other)['sha256']!=evidence['sha256']:
                raise ValueError('Conflicting sealed reports at candidate node '+str(node))
        descriptor={'inherited':True,'new_evaluation_games':0,'candidate_decisions':node,
                    'report':evidence}
        atomic_write_json(destination/'inheritance.json',descriptor)
        completed.append({'decisions':node,'score':report['score'],
            'report':str((destination/'report.json').resolve()),'source_report':str(path),
            'sha256':evidence['sha256'],'inherited':True,'result':report})
    best=None
    initial_development=None
    # The separately sealed node0 is outside the positive development grid.
    # Losing it during recovery incorrectly promotes any later score to best.
    if config.get('development_objective')=='avoidance-first-v1':
        for run,prior in chain:
            descriptor=run/'best_development.json'
            if not descriptor.exists():continue
            selected=read_shared_json(descriptor)
            if selected.get('decisions')!=0:continue
            path=Path(selected.get('report',run/'initial_avoidance_development/report.json'))
            if not path.is_absolute():path=run/path
            baseline=read_shared_json(path)
            if not valid_development_report(config,baseline) or selected.get('score')!=baseline['score']:
                raise ValueError('Incomplete or mismatched sealed initial development baseline')
            checkpoint=run/'best_development.pt'
            actual=_file_record(checkpoint)
            expected=selected.get('checkpoint_sha256')
            if expected and expected!=actual['sha256']:
                raise ValueError('Sealed node0 selected checkpoint identity changed')
            if initial_development and (actual['sha256']!=initial_development['checkpoint_record']['sha256']
                    or baseline['score']!=initial_development['result']['score']):
                raise ValueError('Conflicting original node0 development selections')
            initial_development=dict(result=baseline,report=path,checkpoint=checkpoint,
                checkpoint_record=actual,source_metadata=_file_record(descriptor))
    if initial_development and (not completed or initial_development['result']['score']>=max(row['score'] for row in completed)):
        report_copy=_copy_verified(initial_development['report'],output/'initial_avoidance_development/report.json')
        best=dict(decisions=0,score=initial_development['result']['score'],checkpoint='best_development.pt',
            report=report_copy['copy'],inherited=True,source_metadata=initial_development['source_metadata'],
            source_checkpoint=_copy_verified(initial_development['checkpoint'],output/'best_development.pt'),
            checkpoint_sha256=initial_development['checkpoint_record']['sha256'],
            selection_note='original sealed node0 remains the best;no new native evaluation games')
        atomic_write_json(output/'best_development.json',best)
    elif completed:
        # Strict > retains the earlier node on an exact tie, as the live trainer.
        winner=max(completed,key=lambda row:(row['score'],-row['decisions']))
        for run,_ in chain:
            descriptor=run/'best_development.json'
            if not descriptor.exists():
                continue
            row=read_shared_json(descriptor)
            if row.get('decisions')==winner['decisions'] and row.get('score')==winner['score']:
                best_source=run/'best_development.pt'
                best={'decisions':winner['decisions'],'score':winner['score'],
                      'checkpoint':'best_development.pt','report':winner['report'],
                      'inherited':True,'source_metadata':_file_record(descriptor),
                      'source_checkpoint':_copy_verified(best_source,output/'best_development.pt')}
                break
        if best is None and winner['decisions']==prefix:
            best={'decisions':prefix,'score':winner['score'],'checkpoint':'best_development.pt',
                  'report':winner['report'],'inherited':True,
                  'source_checkpoint':_copy_verified(resume['checkpoint'],output/'best_development.pt'),
                  'selection_note':'sealed report at current recovered model node, prior selection write interrupted'}
        if best is None:
            raise ValueError('Cannot recover the historical selected best model')
        best['checkpoint_sha256']=_file_record(output/'best_development.pt')['sha256']
        atomic_write_json(output/'best_development.json',best)
    initial_source=None
    for run,prior_config in chain:
        explicit=prior_config.get('candidate_initial_checkpoint')
        if explicit:
            initial_source=Path(explicit)
            if not initial_source.is_absolute():
                initial_source=run/initial_source
            expected=prior_config.get('candidate_initial_sha256')
            if expected and _file_record(initial_source)['sha256']!=expected:
                raise ValueError('Candidate initial checkpoint identity changed')
            break
    if initial_source is None:
        initial_source=(Path(resume['checkpoint']) if full_story and not chain
                        else chain[-1][0]/'initial.pt')
    initial=_copy_verified(initial_source,output/'candidate_initial.pt')
    if best and best['decisions']==0 and best['checkpoint_sha256']!=initial['sha256']:
        raise ValueError('Selected node0 is not the pinned original candidate initial')
    result={'schema':'thrl-candidate-recovery-history-v1','source_run':str(source),
        'source_config_chain':[_file_record(run/'config.json') for run,_ in all_chain],
        'retired_score_runs':[str(run) for run,_ in retired],
        'retired_scores_used_for_selection':False,
        'initial_reference_role':(('movement migration checkpoint' if migration.get('schema')==MOVEMENT_AUTH
                                  else 'full-story migration checkpoint') if full_story and not chain
                                  else 'same protocol candidate initial'),
        'candidate_optimized_decision_offset':prefix,
        'candidate_paid_decision_offset_lower_bound':config['candidate_paid_decision_offset'],
        'inherited_unoptimized_sampling_lower_bound':config['candidate_paid_decision_offset']-prefix,
        'additional_inflight_cost':'unknown' if config.get('unknown_inflight_decisions',True) else 'none reported',
        'completed_development':completed,'best':best,'pending_development_at_start':pending_at_start,
        'inherited_initial_development':None if initial_development is None else dict(
            result=initial_development['result'],source_report=str(initial_development['report']),
            checkpoint=initial_development['checkpoint_record'],new_native_games=0),
        'candidate_initial':initial,'new_development_games_inherited':0,
        'native_world_restored':False,'hidden_state_restored':False}
    atomic_write_json(output/'recovery_inheritance.json',result)
    return result


def inherit_study_lineage(config, output):
    """A new SQLite study; immutable FAIL history is evidence, not completed trials."""
    previous=config.get('previous_study_output')
    if not previous:
        return None
    previous=Path(previous).resolve()
    ancestors=[]
    seen=set()
    current=previous
    while current not in seen:
        if len(seen)>=32:
            raise ValueError('Excessive previous-study lineage')
        seen.add(current)
        prior_status=None
        entries=[]
        effective=None
        for name in ('study_config.json','effective_study_config.json','study_status.json','failure.json',
                     'study_lineage.json'):
            path=current/name
            if path.exists():
                evidence=_copy_verified(path,Path(output)/'previous_study'/str(len(ancestors))/name)
                entries.append(evidence)
                if name=='study_status.json':
                    prior_status=read_shared_json(path)
                if name in ('study_config.json','effective_study_config.json'):
                    effective=read_shared_json(path)
        if prior_status is None:
            raise ValueError('Previous study status is required for an explicit failure lineage')
        ancestors.append({'study_output':str(current),'status':prior_status.get('status'),
                          'trials':prior_status.get('trials',[]),'files':entries})
        prior=(effective or {}).get('previous_study_output')
        resume=(effective or {}).get('baseline_resume')
        if not prior and resume:
            prior=Path(resume.get('source_run',Path(resume['checkpoint']).parent)).parent
        if not prior:
            break
        current=Path(prior).resolve()
    result={'schema':'thrl-failed-study-lineage-v1','previous_study_output':str(previous),
        'status':ancestors[0]['status'],'ancestors':ancestors,
        'trials':[{**row,'source_study':ancestor['study_output']} for ancestor in ancestors
                  for row in ancestor['trials']],
        'old_failed_trials_remain_failed':True,'old_study_or_SQLite_modified':False,
        'new_study_baseline_role':'recover original protected architecture candidate0',
        'new_study_trial_quota':config.get('optimization_trials',2)}
    atomic_write_json(Path(output)/'study_lineage.json',result)
    return result


def reuse_teacher_initialization(config):
    """Every candidate, including HPO trial1, reuses the completed BC common origin."""
    result=config.get('teacher_pretraining_result')
    if result is None:
        return config,False
    result=copy.deepcopy(result)
    resume_audit=None
    path=config.get('teacher_pretraining_result_path')
    if path:
        expected=config.get('teacher_pretraining_result_sha256')
        if not expected or _file_record(path)['sha256']!=expected or read_shared_json(path)!=result:
            raise ValueError('Inherited completed teacher result identity changed')
    if _file_record(result['checkpoint'])['sha256']!=result['checkpoint_sha256']:
        raise ValueError('Inherited completed teacher checkpoint identity changed')
    settings=config.get('teacher_pretraining',{})
    if config.get('training_mode')=='full-story-v1' and (
            not isinstance(settings,dict) or settings.get('validation_enabled') is not True):
        raise ValueError('Full-story teacher reuse requires a sealed validated teacher result')
    if isinstance(settings,dict) and settings:
        if settings.get('validation_enabled',False):
            # A validated fit pays only train-side updates and restores the
            # best epoch; neither count equals the old fixed full-data budget.
            import json
            import torch
            from dataclasses import asdict
            from .thrl_teacher_training import TeacherPretrainingConfig
            cfg=TeacherPretrainingConfig(**settings)
            stored_settings=result.get('teacher_pretraining_parameters')
            normalized_settings=(asdict(TeacherPretrainingConfig(**stored_settings)) if isinstance(stored_settings,dict) else None)
            if (result.get('teacher_validation_enabled') is not True
                    or normalized_settings!=asdict(cfg)):
                raise ValueError('Inherited validated teacher settings differ')
            keys=('training_recipe','reward_schema','input_schema','curriculum_protocol')
            protocol=result.get('teacher_protocol',{})
            if protocol.get('teacher_revision')!=config.get('teacher_revision'):
                raise ValueError('Inherited teacher revision differs; new demonstrations required')
            if 'teacher_revision' in protocol:keys=(*keys,'teacher_revision')
            if 'training_mode' in protocol:keys=(*keys,'training_mode')
            compare_keys=tuple(dict.fromkeys((*keys,'training_mode')))
            observed_protocol={key:protocol.get(key) for key in compare_keys}
            expected_protocol={key:config.get(key) for key in compare_keys}
            historical=False
            if observed_protocol!=expected_protocol:
                from .thrl_learning_revision import FULL_STORY_AUTHORIZATION
                migration=config.get('learning_migration') or {}
                historical=(config.get('training_mode')=='full-story-v1'
                    and migration.get('schema')==FULL_STORY_AUTHORIZATION
                    and bool(migration.get('source_teacher_protocol'))
                    and observed_protocol=={key:migration['source_teacher_protocol'].get(key) for key in compare_keys}
                    and all(protocol.get(key)==config.get(key) for key in
                            ('training_recipe','reward_schema','input_schema')))
            if observed_protocol!=expected_protocol and not historical:
                raise ValueError('Inherited teacher input/reward/training protocol differs')
            records={}
            for label in ('teacher_fit_report','teacher_validation_split'):
                report_path=result.get(label)
                if not report_path or _file_record(report_path)['sha256']!=result.get(label+'_sha256'):
                    raise ValueError('Inherited validated teacher report identity changed: '+label)
                records[label]=read_shared_json(report_path)
            report=records['teacher_fit_report'];split=records['teacher_validation_split']
            canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'))
            if canonical(report.get('split'))!=canonical(split):
                raise ValueError('Inherited teacher holdout split differs from fitting report')
            if (report.get('validation_enabled') is not True or report.get('fit_claimed') is not False
                    or report.get('stopped_reason') not in ('heldout-KL-patience','heldout-objective-patience','maximum-epochs',
                                                           'validation-unavailable-no-BC-updates')):
                raise ValueError('Inherited teacher fit is not a sealed validated result')
            if (result.get('teacher_fit_claimed') is not False
                    or result.get('teacher_fit_status')!=report.get('fit_status')
                    or result.get('teacher_validation_reason')!=split.get('reason')
                    or canonical(result.get('teacher_validation'))!=canonical(report.get('final_heldout'))):
                raise ValueError('Inherited teacher display result differs from its validated report')
            paid_epochs=report.get('epochs');best_epoch=report.get('restored_best_epoch',0)
            if (type(paid_epochs) is not int or not 0<=paid_epochs<=cfg.epochs
                    or type(best_epoch) is not int or not 0<=best_epoch<=paid_epochs):
                raise ValueError('Inherited teacher epoch accounting is invalid')
            if split.get('available'):
                train_samples=len(split['train'])*cfg.seq_len
                batches=math.ceil(len(split['train'])/cfg.seqs_per_batch)
                if (train_samples!=split.get('train_samples') or not split['heldout']
                        or not cfg.minimum_epochs<=paid_epochs<=cfg.epochs
                        or report.get('best_epoch')!=best_epoch
                        or report.get('restored_best_BC_optimizer') is not True):
                    raise ValueError('Inherited teacher train/holdout or best-epoch accounting differs')
            else:
                train_samples=batches=0
                if paid_epochs or best_epoch or report.get('fit_status')!='validation-unavailable':
                    raise ValueError('Unavailable teacher validation cannot contain training updates')
            paid_updates=batches*paid_epochs
            from .thrl_demonstration_sources import verify_supervision_action_accounting
            if (result.get('teacher_samples')!=cfg.samples or not verify_supervision_action_accounting(result,cfg.samples)
                    or report.get('unique_teacher_samples')!=cfg.samples
                    or result.get('BC_adam_minibatch_steps')!=paid_updates
                    or report.get('BC_adam_minibatch_steps')!=paid_updates
                    or report.get('examples_seen')!=train_samples*paid_epochs
                    or result.get('teacher_restored_best_epoch')!=best_epoch):
                raise ValueError('Inherited validated teacher paid-update accounting differs')
            checkpoint=torch.load(result['checkpoint'],map_location='cpu',weights_only=False)
            if ({key:checkpoint.get('config',{}).get(key) for key in keys}!=result['teacher_protocol']
                    or checkpoint.get('model_config',{}).get('input_schema')!=config.get('input_schema')
                    or checkpoint.get('teacher_pretrain_updates')!=paid_updates
                    or canonical(checkpoint.get('teacher_pretraining',{}).get('pretraining'))
                       !=canonical({key:value for key,value in report.items() if key!='epoch_history'})):
                raise ValueError('Inherited teacher checkpoint and sealed fitting report differ')
            steps=[float(state['step']) for state in checkpoint['optimizer_bc']['state'].values()]
            restored_updates=batches*best_epoch
            if (any(not math.isfinite(step) or step!=int(step) or not 0<=step<=restored_updates for step in steps)
                    or (max(steps,default=0.)!=restored_updates)):
                raise ValueError('Inherited teacher restored BC optimizer does not match selected epoch')
            resume_audit={'paid_BC_updates':paid_updates,
                'restored_best_epoch':best_epoch,'restored_BC_updates':restored_updates,
                'recollection_or_refit_required':False,'sealed_reports_verified':True,
                'historical_pre_full_story_teacher':historical,
                'current_training_teacher_actions':0}
        else:
            if result.get('teacher_validation_enabled'):
                raise ValueError('Cannot reuse a validated teacher under legacy fixed-budget settings')
            expected=settings['samples']*settings['epochs']//(settings['seq_len']*settings['seqs_per_batch'])
            if result.get('teacher_samples')!=settings['samples'] or result.get('BC_adam_minibatch_steps')!=expected:
                raise ValueError('Inherited teacher result does not match the declared completed BC budget')
    return {**config,'warm_start_checkpoint':result['checkpoint'],
            'warm_start_sha256':result['checkpoint_sha256'],
            'teacher_pretraining_result':result,'teacher_pretraining_reused':True,
            **({'teacher_resume_audit':resume_audit} if resume_audit is not None else {})},True
