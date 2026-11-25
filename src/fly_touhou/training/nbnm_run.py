"""Run specification, model migration and paired NBNM evaluation helpers."""
from pathlib import Path
import copy
import json
import os
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def task_settings(spec, task):
    """One shared policy can train native cards and real Story failure starts."""
    settings=copy.deepcopy(spec['nbnm']);settings['task']=dict(task)
    for group, candidate in spec.get('tasks',{}).items():
        if candidate==task:
            settings.update(copy.deepcopy(spec.get('task_entries',{}).get(group,{})))
            break
    return settings


def validate_spec(spec):
    from ..nbnm_objective import Objective, qualifies_speed, rule_guidance_coefficient, future_guidance_coefficient
    from ..nbnm_sampling import group_block_budget
    from ..nbnm_task import TaskTracker,entry_matches,fresh_opening
    from ..nbnm_prefix import SCHEMA
    from ..games.item_collection import validate as validate_collection
    validate_collection(spec['nbnm'].get('item_collection_input', False))
    objective=Objective(**spec['nbnm']['objective'])
    for option in ('packed_binary_messages','fused_lif_tick','expanded_worker_cpu_pool','deterministic_learner',
                   'reuse_initial_message'):
        if type(spec.get(option,False)) is not bool:
            raise ValueError(option+' must be a boolean')
    retained=spec.get('retained_message_steps',0)
    if type(retained) is not int or not 0<=retained<=4:
        raise ValueError('retained_message_steps must be an integer between zero and four')
    cadence=spec.get('online_teacher_every',1)
    if type(cadence) is not int or not 0<=cadence<=32:
        raise ValueError('online_teacher_every must be a nonnegative integer at most 32')
    from ..nbnm_wall_exit_guidance import validate as validate_wall_exit_guidance
    validate_wall_exit_guidance(spec['nbnm'].get('wall_exit_guidance'))
    rolling=spec['nbnm'].get('live_route_planner',{})
    from ..nbnm_native_warning import validate as validate_warning48
    validate_warning48(spec['nbnm'].get('native_warning48'))
    from ..nbnm_corridor import validate as validate_corridor
    validate_corridor(spec['nbnm'].get('corridor_risk'))
    emitter_frames = spec['nbnm'].get('native_emitter_forecast_frames', 0)
    if type(emitter_frames) is not int or emitter_frames not in (0, 60, 96, 120):
        raise ValueError('Choose a native emitter forecast of 0, 60, 96 or 120 frames')
    exit_scoring = spec.get('frozen_corridor_exit_scoring', False)
    if type(exit_scoring) is not bool or (exit_scoring and not spec['nbnm'].get('corridor_risk')):
        raise ValueError('Frozen corridor exit scoring needs an enabled corridor')
    exit_limit = spec.get('frozen_corridor_exit_limit', 'auto')
    if exit_limit != 'auto' and (type(exit_limit) is not int or exit_limit not in (4, 8)):
        raise ValueError('Use four, eight, or adaptive first moves for frozen exit probes')
    continuation = spec['nbnm'].get('continuation_supervision', False)
    if type(continuation) is not bool or (continuation and not spec['nbnm'].get('corridor_risk')):
        raise ValueError('Continuation supervision needs an enabled corridor')
    timed = spec['nbnm'].get('timed_corridor_supervision', False)
    if type(timed) is not bool or (timed and not continuation):
        raise ValueError('Timed corridor supervision requires continuation supervision')
    if spec['nbnm'].get('corridor_risk') and any((spec['nbnm'].get(k) or {}).get('enabled') for k in ('route_prior','future_rules')):
        raise ValueError('The 407-column corridor replaces the optional atlas/future extension')
    if rolling.get('supervise_student'):
        from ..nbnm_live_routes import REVISION as ROLLING_REVISION
        if not rolling.get('enabled') or rolling.get('revision')!=ROLLING_REVISION:
            raise ValueError('Online rolling guidance requires the linked-prefix route revision')
        if not rolling.get('stages') or any(type(s) is not int or not 1<=s<=6 for s in rolling['stages']):
            raise ValueError('Declare the native-validated stage scope of rolling guidance')
        if rolling.get('horizon',96) not in (48,96):
            raise ValueError('Unsupported native-frame planning horizon')
        if any(k not in {str(s) for s in range(1,7)} or v not in (48,96)
               for k,v in rolling.get('stage_horizons',{}).items()):
            raise ValueError('Invalid native-stage planning horizon')
        if any(not isinstance(k,str) or not k.startswith(tuple(f's{s}/' for s in range(1,7))) or v not in (48,96)
               for k,v in rolling.get('phase_horizons',{}).items()):
            raise ValueError('Invalid native-phase planning horizon')
        from ..nbnm_live_routes import configured_beam
        for phase in rolling.get('phase_beams',{}):
            if not isinstance(phase,str) or not phase.startswith(tuple(f's{s}/' for s in range(1,7))):
                raise ValueError('A search beam must refer to a native phase')
            configured_beam(int(phase[1]),rolling,phase)
    from ..nbnm_geometry_revision import validate as validate_geometry
    geometry=validate_geometry(spec['nbnm'].get('native_rule_geometry'))
    confirmation=spec['nbnm'].get('event_confirmation_revision')
    if confirmation:
        from ..nbnm_event_progress import CONFIRMATION_REVISION
        profile=read(read(spec['environment_config'])['profile'])
        if (confirmation!=CONFIRMATION_REVISION or
                profile.get('hooks',{}).get('native_attack_events')!=confirmation):
            raise ValueError('Enable the native attack interrupt reader for immediate event confirmation')
    if geometry:
        profile=read(read(spec['environment_config'])['profile'])
        hooks=profile.get('hooks',{})
        if hooks.get('native_ex_program')!=geometry['revision'] or int(hooks.get('native_ex_program_horizon',24))!=geometry['horizon']:
            raise ValueError('Prepare the matching native reader profile before enabling rule geometry')
    from ..nbnm_continuation_risk import enabled as continuation_risk_enabled
    if continuation_risk_enabled(spec['nbnm']) and (
            spec['nbnm'].get('teacher_mode')!='nbnm-route-v1' or spec['nbnm'].get('risk_mode')!='route'
            or spec['nbnm'].get('future_reference')!='teacher-route-v1'):
        raise ValueError('Continuation targets need the existing route inputs and witnessed route teacher')
    rule_guidance_coefficient(spec['nbnm'])
    future_guidance_coefficient(spec['nbnm'])
    if spec['nbnm'].get('teacher_outcome_revision'):
        from ..nbnm_teacher_outcomes import SUPPORTED_REVISIONS
        if spec['nbnm']['teacher_outcome_revision'] not in SUPPORTED_REVISIONS:raise ValueError('Unknown teacher outcome revision')
    if spec['nbnm'].get('future_reference','constant-direction-v1') not in ('constant-direction-v1','teacher-route-v1'):
        raise ValueError('Unknown future guidance target')
    weights=spec['nbnm'].get('movement_risk_horizon_weights',(8.,4.,2.))
    import math
    if len(weights)!=3 or weights[0]!=8 or any(not math.isfinite(w) or w<0 for w in weights):
        raise ValueError('Preserve four-frame risk weight8 and use three finite nonnegative weights')
    demonstration=spec['nbnm'].get('demonstration_route_teacher') or {}
    if demonstration.get('enabled'):
        from ..nbnm_demonstration_routes import DemonstrationRouteGuide
        DemonstrationRouteGuide(demonstration['manifest'])
    if spec['nbnm'].get('route_preference_revision') is not None:
        from ..nbnm_route_preference import REVISION
        if spec['nbnm']['route_preference_revision']!=REVISION:
            raise ValueError('Unknown NBNM route preference revision')
    prior=spec['nbnm'].get('route_prior',{})
    future=spec['nbnm'].get('future_rules',{})
    if future.get('enabled'):
        if prior.get('enabled'):raise ValueError('Use one future-input experiment at a time')
        from ..nbnm_future_sensation import FutureRuleSensor
        FutureRuleSensor(future)
    if prior.get('enabled'):
        from ..nbnm_route_atlas import RouteAtlas
        RouteAtlas(prior['manifest'])
    from ..th08_attack_readout import CRITIC_TASK_CONTEXTS
    if spec['nbnm'].get('task_context','legacy') not in ('legacy', *CRITIC_TASK_CONTEXTS):
        raise ValueError('Unknown NBNM task-context protocol')
    retention=spec.get('rehearsal',{})
    if retention.get('coefficient',0)>0:
        from ..nbnm_rehearsal import SuccessRehearsal
        SuccessRehearsal(retention['manifest'],scope_weights=retention.get('scope_weights'),
            additional_manifests=retention.get('additional_manifests',()),
            attack_spell_fraction=retention.get('attack_spell_fraction'),challenging_fraction=retention.get('challenging_fraction',0.),
            phase_priorities=retention.get('phase_priorities'),priority_fraction=retention.get('priority_fraction',.5),
            native_short_risk_coefficient=retention.get('native_short_risk_coefficient',0.),
            native_correction=retention.get('native_correction'))
        if not 0<retention['coefficient']<=.3 or type(retention.get('every_updates')) is not int or retention['every_updates']<1:
            raise ValueError('Specify a small supervised retention coefficient and positive interval')
    if objective.phase=='speed':
        report=read(spec['speed_qualification_report'])
        if not qualifies_speed(report):raise ValueError('Speed phase requires frozen NBNM qualification')
        if report['task']!=spec['nbnm']['task']:raise ValueError('Qualify the same task before enabling speed')
    workers=spec['workers'];segment=spec['segment_decisions'];budget=spec['decision_budget']
    if not 1<=workers<=100 or segment%256 or budget%segment or budget<=0:
        raise ValueError('Use 1..100 workers and whole 256-row PPO segments')
    if len(spec['world_groups'])!=workers:raise ValueError('One task group per worker required')
    group_block_budget(spec['world_groups'],spec['group_weights'],segment//16)
    if spec.get('coverage_group_weights'):
        group_block_budget(spec['world_groups'],spec['coverage_group_weights'],segment//16)
    for task in spec['tasks'].values():TaskTracker(task)
    pools=spec.get('task_pools')
    if pools:
        if set(spec['world_groups'])!=set(pools):raise ValueError('Task pools must match worker groups')
        for group,keys in pools.items():
            if not keys or len(set(keys))!=len(keys) or any(k not in spec['tasks'] for k in keys):
                raise ValueError('Each pool requires distinct declared task keys')
    adaptive=spec.get('adaptive_course') or {}
    if adaptive.get('enabled'):
        from ..nbnm_adaptive_course import AdaptiveCourse
        if not pools or 'local' not in pools:raise ValueError('Adaptive allocation requires a local task pool')
        if any(spec['tasks'][key]['mode']!='attack' for key in pools['local']):
            raise ValueError('Only local attack tasks use adaptive allocation')
        AdaptiveCourse(pools['local'],spec['world_groups'],window=adaptive.get('window',20),
            uniform_fraction=adaptive.get('uniform_fraction',.25),prior=adaptive.get('prior',2.),
            priority_multipliers=adaptive.get('priority_multipliers'))
        if adaptive.get('stage_chain_enabled'):
            AdaptiveCourse(pools['stage_chain'],spec['world_groups'],group='stage_chain',
                strata={k:spec['tasks'][k]['mode'] for k in pools['stage_chain']},
                stratum_weights=adaptive.get('stage_chain_stratum_weights'),
                priority_multipliers=adaptive.get('stage_chain_priority_multipliers'))
    for group in spec.get('evaluation_tasks',[]):
        if group!='primary' and group not in spec['tasks']:raise ValueError('Unknown final evaluation task '+group)
    from ..nbnm_spell_practice import enabled,validate_task
    from .. import nbnm_stage_practice as stage_practice
    for task in spec['tasks'].values():
        settings=task_settings(spec,task)
        task_objective=Objective(**settings['objective'])
        if task_objective.phase=='speed':
            report=read(settings.get('speed_qualification_report',spec.get('speed_qualification_report')))
            if not qualifies_speed(report) or report['task']!=task:
                raise ValueError('Speed qualification is per task, not transferable to other cards')
        native_stage=stage_practice.enabled(settings)
        if enabled(settings) or native_stage:
            if enabled(settings):validate_task(task)
            if native_stage and (task['mode']=='full_story' or task.get('end_stage',task['stage'])!=task['stage']):
                raise ValueError('Stage Practice cannot substitute for a cross-stage Story chain')
            if not Path(settings['score_template']).is_file():
                raise ValueError('Supply an existing native score record with the requested spells unlocked')
            from ..thrl_training_mode import seed_split
            for split, seeds in settings['practice_evaluation_seeds'].items():
                if not seeds or len(set(seeds)) != len(seeds) or any(seed_split(s)!=split for s in seeds):
                    raise ValueError('Use unique native evaluation seeds belonging to their data split')
            if settings.get('entry_mode')!=stage_practice.PREFIX_MODE:continue
        if fresh_opening(task):continue
        bank=read(settings['start_bank'])
        if bank.get('schema')!=SCHEMA:raise ValueError('Prepare the native attack-start bank first')
        required_splits=(('train',) if native_stage and settings.get('prepare_evaluation_prefixes')
            else ('train','development'))
        for split in required_splits:
            entries=[r for r in bank['entries'] if entry_matches(r['task'],task) and r['split']==split]
            if not entries:raise ValueError(f'Missing {split} task starts: {task}')
        train={r['native_seed'] for r in bank['entries'] if r['split']=='train'}
        dev={r['native_seed'] for r in bank['entries'] if r['split']=='development'}
        if train&dev:raise ValueError('Training and development native seeds overlap')
    return dict(status='ready',network_training=False,workers=workers,
        decision_budget=budget,segment_decisions=segment,optimizer_updates_per_segment=segment//256,
        risk_target_revision=spec['nbnm'].get('risk_target_revision') or 'constant-direction-v1',
        native_rule_geometry=geometry,student_corridor_risk=spec['nbnm'].get('corridor_risk'),
        event_confirmation_revision=confirmation,
        objective=objective.specification(),task_group_fractions=spec['group_weights'],
        route_prior_enabled=bool(prior.get('enabled')),
        future_rule_inputs_enabled=bool(future.get('enabled')),
        prior_is_future_collision_oracle=False)


def runtime(output):
    from ..runtime_cache_paths import configure_generated_caches,generated_kernel_cache_base
    from ..thrl_job_paths import configure_native_temp
    configure_generated_caches()
    cache=generated_kernel_cache_base(Path(output))
    os.environ.setdefault('FLY_NBNM_FORECAST_DIAGNOSTICS',str(Path(output)/'numeric-faults'))
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-nbnm-'+str(os.getpid()),
        FLY_TOUHOU_FUSED_EDGE_GRAD='1',FLY_TOUHOU_EDGE_GRAD_CACHE=str(cache/'kernel-cache'),
        FLY_TOUHOU_DETERMINISTIC_SPMM='1',FLY_TOUHOU_SPMM_CACHE=str(cache/'spmm-kernel-cache'))
    temporary=configure_native_temp(output)
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return temporary


def environment(spec, *, evaluation=False):
    cfg=read(spec['environment_config'])
    # These old options describe resource-play, not an NBNM objective.
    for key in ('compact_lower_field_reward','compact_frozen_prefix_replay','compact_frozen_decoding',
                'compact_capture_stage6_prefix','compact_conditional_wall_guidance_coefficient'):
        cfg.pop(key,None)
    cfg.update(seed=spec['seed'],nbnm=copy.deepcopy(spec['nbnm']),
        compact_full_native_teacher_support=spec['nbnm'].get('teacher_mode') in ('resource-movement-v1','nbnm-route-v1'),
        compact_protected_exit_forecast=False)
    if not evaluation:
        cfg['nbnm']['online_teacher_every']=spec.get('online_teacher_every',1)
        from ..nbnm_course import worker_settings
        rows=worker_settings(spec)
        cfg['nbnm']['world_tasks']=[r[0] for r in rows]
        cfg['nbnm']['world_entries']=[r[1] for r in rows]
    return cfg


def load_cpu_checkpoint(path):
    import torch
    path = Path(path)
    with path.open('rb') as stream:
        zipped = stream.read(4) == b'PK\x03\x04'
    return torch.load(path, map_location='cpu', weights_only=False, mmap=zipped)


def load_model(spec, checkpoint=None, *, payload=None, apply_native_sensors=True,
               preserve_collection_input=False, reset_collection_line=False):
    from ..compact_story_model import from_story_checkpoint
    from ..nbnm_model import configure_model
    point = load_cpu_checkpoint(checkpoint or spec['initial_checkpoint']) if payload is None else payload
    if 'model_payload' in point:point=point['model_payload']
    if preserve_collection_input:
        saved_collection = (point.get('nbnm') or {}).get('item_collection_input', False)
        if saved_collection or 'item_collection_input' in spec['nbnm']:
            spec['nbnm']['item_collection_input'] = saved_collection
    migrated=not bool(point.get('nbnm'))
    if not migrated and point['nbnm']['risk_mode']!=spec['nbnm']['risk_mode']:
        raise ValueError('Use matching route/baseline input checkpoints')
    model=from_story_checkpoint(spec['connectome'],point)
    configure_model(model,spec['nbnm'],reset_critics=migrated,reset_collection_line=reset_collection_line)
    if model.shared_directions is not None and model.directional_cue_width != model.shared_directions.cue_width:
        raise ValueError('Single-state direction checkpoints require their saved candidate input width')
    if apply_native_sensors:
        from ..compact_full_story import apply_sensor_patches
        apply_sensor_patches(model,environment(spec))
    model.core.sparse.deterministic_spmm=True
    model.core.fused_lif_tick=spec.get('fused_lif_tick',False)
    model.reuse_initial_message=spec.get('reuse_initial_message',False)
    model.retained_message_steps=spec.get('retained_message_steps',0)
    return model.eval(),migrated


def atomic_torch_save(value,path):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.writing')
    with temporary.open('wb') as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def archive_torch_parameters(source, target):
    """Archive an immutable parameter generation without serializing it twice."""
    import shutil
    temporary = target.with_suffix(target.suffix + '.writing')
    temporary.unlink(missing_ok=True)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            with source.open('rb') as old, temporary.open('wb') as new:
                shutil.copyfileobj(old, new, length=1024 * 1024)
                new.flush()
                os.fsync(new.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def evaluation_seeds(spec, *, split='development', task=None):
    from ..nbnm_spell_practice import enabled
    task=task or spec['nbnm']['task']
    settings=task_settings(spec,task)
    from ..nbnm_stage_practice import ENTRY_MODE,PREFIX_MODE
    if (enabled(settings) or settings.get('entry_mode')==ENTRY_MODE or
            (settings.get('entry_mode')==PREFIX_MODE and settings.get('prepare_evaluation_prefixes'))):
        return list(settings['practice_evaluation_seeds'][split])
    from ..nbnm_task import fresh_opening
    if fresh_opening(task):
        if split=='train':
            from ..thrl_training_mode import seed_split
            seeds=list(spec.get('full_story_training_seeds',[]))
            if not seeds or any(seed_split(s)!='train' for s in seeds):
                raise ValueError('Explicit training-only opening seeds required for teacher collection')
            return seeds
        return list(spec['full_story_evaluation_seeds'] if split=='development' else spec['full_story_heldout_seeds'])
    bank=read(settings['start_bank'])
    from ..nbnm_task import entry_matches
    return sorted({r['native_seed'] for r in bank['entries'] if r['split']==split and entry_matches(r['task'],task)})


def frozen_story_clears(reports):
    """Original fresh-opening, six-stage NBNM clears, with their evaluation split."""
    evidence=[]
    for report in reports.values():
        if (not report.get('frozen') or report.get('task',{}).get('mode')!='full_story'
                or report.get('frozen_prefix_replay') or report.get('fresh_policy_confirmation') is False):
            continue
        for episode in report.get('episode_results',()):
            if (episode.get('full_game_NBNM') and episode.get('nbnm_success')
                    and episode.get('stage')==6 and episode.get('task_misses')==0
                    and episode.get('task_Bombs')==0 and episode.get('nb_nm_contact') is False
                    and episode.get('full_story_prefix_played_by_current_policy') is not False
                    and episode.get('nb_nm_reason')=='native-six-stage-clear'
                    and episode.get('stage_history')==[1,2,3,4,5,6]):
                evidence.append(dict(native_seed=episode['native_seed'],split=report['evaluation_split'],
                    native_frames=episode['native_frames']))
    return evidence


def full_story_qualification(report,spec):
    """Evaluate the declared stable-clear target on its reserved native seeds."""
    q=spec['qualification']
    episodes=report['episode_results']
    seeds=[e['native_seed'] for e in episodes]
    distinct=len(set(seeds))==len(seeds)
    heldout=(report.get('frozen') and report.get('evaluation_split')=='test'
        and report.get('task',{}).get('mode')=='full_story'
        and set(seeds)==set(spec['full_story_heldout_seeds']) and distinct)
    assisted=any(report.get(key) for key in ('corridor_exit_scoring','frozen_safe_idle_bonus',
        'frozen_safe_heading_bonus','frozen_wall_escape','frozen_route_aware_wait'))
    gate=q.get('stage_gate')
    if gate:
        checked=report.get('verified_movement_decisions',0)
        autonomous=not assisted and report.get('soft_movement_scores') is True and type(checked) is int and checked>0
    else:
        autonomous=not assisted and report.get('soft_movement_scores') is not False
    successes=len(frozen_story_clears({'full_story':report}))
    rate=successes/len(seeds) if seeds else 0.
    result=dict(qualified=bool(heldout and autonomous and len(seeds)>=q['minimum_attempts']
        and successes>=q['minimum_successes'] and rate>=q['minimum_rate']),
        autonomous_movement_scoring=autonomous,
        heldout_native_seeds_matched=bool(heldout),attempts=len(seeds),verified_six_stage_clears=successes,
        success_rate=rate,minimum_attempts=q['minimum_attempts'],minimum_successes=q['minimum_successes'],
        minimum_rate=q['minimum_rate'])
    if gate:
        stage=gate['stage']
        prefixes=[list(e.get('stage_history', ())) for e in episodes
                  if e.get('full_story_prefix_played_by_current_policy') is True]
        entered=sum(h[:stage]==list(range(1,stage+1)) for h in prefixes)
        cleared=sum(h[:stage+1]==list(range(1,stage+2)) for h in prefixes)
        independent=(len(prefixes)==len(episodes) and not report.get('frozen_prefix_replay') and autonomous)
        stage_passed=bool(independent and entered>=gate['minimum_entries'] and cleared>=gate['minimum_clears'])
        result['stage_gate']=dict(gate,entered=entered,cleared=cleared,passed=stage_passed)
        result['qualified'] &= stage_passed
    return result


def evaluate(model,spec,output,*,task=None,split='development',progress=lambda x:None,guard=lambda:None,
             opening_stage=None,through_stage=None,checkpoint=None,resume_bookmark=None):
    """Frozen actual-game task attempts; a prefix is never a six-stage clear."""
    import numpy as np
    import torch
    from .. import thrl_vector
    from ..compact_full_story import factory,story_kwargs
    from ..nbnm_prefix import burn_in
    from ..spiking_engineered_control import SpikingState
    from ..thrl_durable_records import write_json,append_jsonl
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cfg=environment(spec,evaluation=True);task=dict(task or spec['nbnm']['task'])
    cfg['nbnm']=task_settings(spec,task)
    exit_scoring = spec.get('frozen_corridor_exit_scoring', False)
    if type(exit_scoring) is not bool or (exit_scoring and not cfg['nbnm'].get('corridor_risk')):
        raise ValueError('Frozen corridor exit scoring needs an enabled corridor')
    if exit_scoring:
        exit_limit = spec.get('frozen_corridor_exit_limit', 'auto')
        if exit_limit != 'auto' and (type(exit_limit) is not int or exit_limit not in (4, 8)):
            raise ValueError('Use four, eight, or adaptive first moves for frozen exit probes')
        cfg['nbnm']['frozen_corridor_exit_scoring'] = True
        cfg['nbnm']['frozen_corridor_exit_limit'] = exit_limit
    exit_totals = dict(measured_decisions=0, scores_applied=0, moves_changed=0,
                       continuations=0, lost_retreats=0, inspected_corridors=0, emitter_risk_updates=0,
                       emitter_turn_updates=0)
    save_six=task.get('mode')=='full_story' and opening_stage is None and through_stage is None
    if save_six:
        cfg['compact_capture_stage6_prefix']=True
        cfg['compact_frozen_story_bookmarks']=True
    if resume_bookmark and not save_six:
        raise ValueError('A saved Story boundary resumes a complete remaining Story attempt')
    if through_stage is not None and (through_stage not in range(1,6) or opening_stage is not None
            or split!='development' or task.get('mode')!='full_story'):
        raise ValueError('A stage-clear comparison starts at original Story entry and ends on the next stage')
    if opening_stage is not None:
        stage_start = (task.get('mode') == 'stage' and task.get('stage') == opening_stage)
        story_start = task.get('mode') == 'full_story' and opening_stage == 1
        if split != 'development' or not (stage_start or story_start):
            raise ValueError('Opening comparison requires the original development story or stage entry')
    seeds=evaluation_seeds(spec,split=split,task=task)
    if resume_bookmark:
        if resume_bookmark['seed'] not in seeds or resume_bookmark['split'] != split:
            raise ValueError('Saved original seed does not belong to this evaluation split')
        seeds = [resume_bookmark['seed']]
    if cfg['nbnm'].get('entry_mode')=='native_stage_prefix' and cfg['nbnm'].get('prepare_evaluation_prefixes'):
        from ..nbnm_start_cache import prepare_evaluation_bank
        cfg['nbnm']['start_bank']=prepare_evaluation_bank(spec,cfg['nbnm'],task,seeds,split,output,
            progress=progress,guard=guard)
    budget=len(seeds);workers=min(int(spec.get('evaluation_workers',4)),budget)
    if not 1<=workers<=100:raise ValueError('Evaluation needs 1..100 workers and at least one seed')
    results=[];entries={};begun=time.monotonic()
    def evaluation_progress(**fields):
        if through_stage is not None:
            progress(dict(phase='frozen_story_stage_clear',completed=len(results),target=budget,
                through_stage=through_stage,stage_passes=sum(r['stage_passed'] for r in results),**fields))
        elif opening_stage is None:
            progress(dict(phase='frozen_NBNM_evaluation', completed=len(results), target=budget,
                          successes=sum(r['nbnm_success'] for r in results), **fields))
        else:
            progress(dict(phase='frozen_native_opening', completed=len(results), target=budget,
                          opening_stage=opening_stage, opening_passes=sum(r['reached_midboss'] for r in results),
                          **fields))
    diagnostics=spec.get('evaluation_diagnostics',False)
    teacher_diagnostics=diagnostics is True or diagnostics=='teacher'
    check_movement=(split=='test' and task.get('mode')=='full_story'
                    and bool(spec.get('qualification',{}).get('stage_gate')))
    verified_movement_decisions=0
    decisions=0
    old=thrl_vector.worker_factory;thrl_vector.worker_factory=factory;vector=None;graphs=None
    model.eval()
    old_idle_bonus=getattr(model,'frozen_safe_idle_bonus',0.)
    old_heading_bonus=getattr(model,'frozen_safe_heading_bonus',0.)
    old_heading_steps=getattr(model,'frozen_heading_steps',None)
    old_wall_escape=getattr(model,'frozen_wall_escape',False)
    old_route_wait=getattr(model,'frozen_route_aware_wait',False)
    try:
        idle_bonus=float(spec.get('frozen_safe_idle_bonus',0.))
        if not np.isfinite(idle_bonus) or idle_bonus<0:
            raise ValueError('Frozen safe-idle bonus must be finite and nonnegative')
        model.frozen_safe_idle_bonus=idle_bonus
        wall_escape=spec.get('frozen_wall_escape',False)
        if type(wall_escape) is not bool or (wall_escape and getattr(model,'corridor_head',None) is None):
            raise ValueError('Frozen wall escape requires a boolean setting and the original corridor observation')
        model.frozen_wall_escape=wall_escape
        route_wait=spec.get('frozen_route_aware_wait',False)
        if type(route_wait) is not bool or (route_wait and (idle_bonus<=0 or getattr(model,'corridor_head',None) is None)):
            raise ValueError('Route-aware waiting requires a positive idle bonus and the original corridor observation')
        model.frozen_route_aware_wait=route_wait
        heading_bonus=float(spec.get('frozen_safe_heading_bonus',0.))
        if not np.isfinite(heading_bonus) or heading_bonus<0:
            raise ValueError('Frozen safe-heading bonus must be finite and nonnegative')
        previous=None
        if heading_bonus:
            if getattr(model,'corridor_head',None) is None:
                raise ValueError('Safe-heading evaluation requires the 407-column corridor policy and its risk head')
            from ..action_query_residual_objects import source18_action_descriptors
            model.frozen_safe_heading_bonus=heading_bonus
            model.frozen_heading_steps=source18_action_descriptors()[:,2:4].to(model.device)*18
            previous=np.full(workers,-1,dtype=np.int64)
        if spec.get('cuda_graph_inference',False) and model.device.type=='cuda':
            from ..nbnm_cuda import InferenceGraphs
            graphs=InferenceGraphs(model,sample=False,packed_binary_messages=spec.get('packed_binary_messages',False))
        kwargs=story_kwargs(cfg,output,workers,split,teacher=teacher_diagnostics)
        if spec.get('diagnostic_worker_phases'):
            for options in kwargs:options['diagnostic_worker_phases']=True
        if teacher_diagnostics:
            from ..resource_teacher_v10 import REVISION
            for options in kwargs:options['online_teacher_revision']=REVISION
        vector=thrl_vector.THRLNativeVector(kwargs,output/'workers',timeout=600 if resume_bookmark else 180)
        reset_options = [dict(native_seed=s) for s in seeds[:workers]]
        if resume_bookmark:
            reset_options[0]['frozen_bookmark_prefix'] = resume_bookmark['native_input_prefix']
        obs,infos=vector.reset(options=reset_options)
        # Vector observations are immutable shared snapshots. The evaluator
        # keeps its own writable staging batch while workers finish out of order.
        obs={key:np.array(value,copy=True) for key,value in obs.items()}
        def record_entry(index,info):
            entries[str(info['native_seed'])]=dict(features=obs['features'][index].tolist(),
                policy_burn_in=info.get('policy_burn_in'))
            write_json(output/'entry-observations.json',entries)
        for index,info in enumerate(infos):record_entry(index,info)
        if resume_bookmark:
            from ..compact_frozen_prefix_replay import restore_brain
            state = restore_brain(model, resume_bookmark['brain'], obs['features'][0], seeds[0],
                                  receipt_path=output/'frozen-brain-restore.json')
            if (infos[0]['stage'], infos[0]['stage_frame']) != (resume_bookmark['stage'], resume_bookmark['stage_frame']):
                raise ValueError('Restored native Story clock differs from the saved manifest')
            if previous is not None:
                saved_action = resume_bookmark.get('previous_action')
                previous[0] = -1 if saved_action is None else saved_action
            vector.timeout = 180
        else:
            state=SpikingState.pack([burn_in(model,info,model.initial_state(1)) for info in infos])
        pending=list(range(workers));next_seed=workers
        with torch.no_grad():
            while pending or vector._inflight:
                guard()
                if pending:
                    ids=torch.tensor(pending,device=model.device)
                    sub=SpikingState(*(v.index_select(0,ids) for v in state.tensors()))
                    features=np.stack([obs['features'][i] for i in pending])
                    tensor=torch.as_tensor(features,device=model.device)
                    history={'previous_actions':torch.as_tensor(previous[pending],device=model.device)} if previous is not None else {}
                    if exit_scoring:
                        effects = [infos[i].get('nbnm_exit48') for i in pending]
                        offsets = np.stack([e['bias'] if e is not None else np.zeros(18, np.float32)
                                            for e in effects])
                        history['corridor_exit_bias'] = torch.as_tensor(offsets, device=model.device)
                    graph_batch=spec.get('frozen_cuda_batch')
                    if graphs is not None and graph_batch:
                        out,following,_=graphs.frozen_batch(tensor,sub,int(graph_batch),**history)
                    else:
                        predict=graphs if graphs is not None else model.step
                        out,following,_=predict(tensor,sub,**history)
                    if diagnostics:
                        from ..nbnm_diagnostics import decision_record
                        for j,i in enumerate(pending):
                            row=decision_record(infos[i],features[j],out,j,policy='frozen joint_mode')
                            append_jsonl(output/'decisions.jsonl',row)
                    else:
                        from ..nbnm_numerics import validate_network_arrays
                        validate_network_arrays({key:out[key].detach().cpu().numpy() for key in
                            ('logits','route_scores','horizon_scores','values','corridor_logits','corridor_witness_logits','corridor_course_logits','corridor_endpoint_cost')
                            if key in out},features=features)
                    for target,value in zip(state.tensors(),following.tensors()):target.index_copy_(0,ids,value)
                    chosen=out['actions'].cpu().tolist()
                    if check_movement:
                        from ..nbnm_numerics import verified_movement_actions
                        playable=(features[:,14]>.5)|(features[:,17]>.5)
                        if playable.any():
                            scores=out['logits'][...,:18].detach().cpu().numpy()[playable]
                            verified_movement_decisions+=verified_movement_actions(scores,np.asarray(chosen)[playable])
                    if exit_scoring:
                        unadjusted = out['pre_corridor_exit_action'].cpu().tolist()
                        for j, e in enumerate(effects):
                            if e is None:
                                continue
                            exit_totals['measured_decisions'] += bool((e['exits'] >= 0).any())
                            exit_totals['inspected_corridors'] += e['inspected']
                            exit_totals['emitter_risk_updates'] += 'risk24' in e
                            exit_totals['emitter_turn_updates'] += 'turn_cues' in e
                            if not offsets[j].any():
                                continue
                            i = pending[j]
                            changed = chosen[j] != unadjusted[j]
                            exit_totals['scores_applied'] += 1
                            exit_totals['moves_changed'] += changed
                            exit_totals['continuations'] += e['follow'] is not None and offsets[j, e['follow']] >= .25
                            exit_totals['lost_retreats'] += bool(e['lost'].any())
                            append_jsonl(output/'corridor-exits.jsonl', dict(
                                seed=infos[i]['native_seed'], stage=infos[i]['stage'],
                                frame=infos[i]['stage_frame'], base=unadjusted[j], action=chosen[j],
                                exits=e['exits'].tolist(), bias=offsets[j].tolist(),
                                lost=e['lost'].tolist(), follow=e['follow']))
                    vector.submit_at(pending,'step',chosen)
                    if previous is not None:previous[pending]=chosen
                    pending=[]
                messages=vector.receive_ready(max_wait_ms=1,min_batch=4,check_stop=guard)
                for msg in messages:
                    i=msg['index'];info=msg['info']
                    if previous is not None:
                        prior=infos[i]
                        elapsed=(info.get('episode_frames') or 0)-(prior.get('episode_frames') or 0)
                        if (msg['operation']=='reset' or info.get('native_seed')!=prior.get('native_seed')
                                or info.get('stage')!=prior.get('stage') or not 0<elapsed<=4):
                            previous[i]=-1
                    infos[i]=info
                    for k,value in msg['observation'].items():obs[k][i]=value
                    if msg['operation']=='reset':
                        record_entry(i,info)
                        fresh=burn_in(model,info,model.initial_state(1))
                        for target,value in zip(state.tensors(),fresh.tensors()):target[i:i+1].copy_(value)
                        pending.append(i);continue
                    decisions+=1
                    if save_six and not msg['terminated'] and not msg['truncated']:
                        if info.get('frozen_story_bookmark'):
                            from ..nbnm_story_bookmark import save
                            save(output, model, state, i, obs['features'][i], info, spec, checkpoint, split,
                                 previous_action=int(previous[i]) if previous is not None else None)
                        elif info.get('native_stage_record'):
                            from ..nbnm_stage_bookmark import save_live_brain
                            save_live_brain(output,model,state,i,obs['features'][i],info,spec,checkpoint)
                    if diagnostics and 'compact_contact_evidence' in info:
                        append_jsonl(output/'native-contact-evidence.jsonl',dict(native_seed=info.get('native_seed'),
                            transition=info['compact_contact_evidence'],scope='frozen evaluation; not training labels'))
                    if decisions%128==0:
                        evaluation_progress(decisions=decisions, last_stage=info.get('stage'),
                                            last_stage_frame=info.get('stage_frame'))
                    if msg['truncated']:raise RuntimeError('Incomplete evaluation is not an NBNM failure or success')
                    stage_passed=(through_stage is not None and not msg['terminated'] and
                                  info.get('stage')==through_stage+1 and not info.get('nb_nm_contact'))
                    reached = (opening_stage is not None and not msg['terminated']
                               and info.get('stage') == opening_stage
                               and str(info.get('nb_nm_phase')).startswith(f's{opening_stage}/MBoss')
                               and not info.get('nb_nm_contact'))
                    if msg['terminated'] or reached or stage_passed:
                        if through_stage is not None:
                            row=dict(native_seed=info['native_seed'],stage=info['stage'],stage_frame=info['stage_frame'],
                                phase=info.get('nb_nm_phase'),stage_passed=stage_passed,episode_frame=info.get('episode_frames'))
                            append_jsonl(output/'stage-clear-results.jsonl',row)
                        elif opening_stage is None:
                            row=dict(info['episode_summary'],evaluation_split=split)
                            append_jsonl(output/'episodes.jsonl',row)
                        else:
                            row=dict(native_seed=info['native_seed'],stage=info['stage'],
                                     stage_frame=info['stage_frame'],phase=info.get('nb_nm_phase'),
                                     reached_midboss=reached,end_reason=('first-native-midboss' if reached else
                                     info['episode_summary']['end_reason']))
                            append_jsonl(output/'opening-results.jsonl',row)
                        results.append(row)
                        evaluation_progress()
                        if next_seed<budget:
                            vector.submit_at([i],'reset',options=[dict(native_seed=seeds[next_seed])]);next_seed+=1
                    else:pending.append(i)
        if through_stage is not None:
            report=dict(schema='frozen-NBNM-story-stage-clear-v1',frozen=True,task=task,
                through_stage=through_stage,attempts=len(results),stage_passes=sum(r['stage_passed'] for r in results),
                outcomes=sorted(results,key=lambda r:r['native_seed']),decisions=decisions,
                evaluation_split=split,action_selection='joint_mode',elapsed_seconds=time.monotonic()-begun,training_updates=0)
            write_json(output/'report.json',report)
            return report
        if opening_stage is not None:
            report=dict(schema='frozen-NBNM-opening-comparison-v1', frozen=True, task=task,
                opening_stage=opening_stage, entry_mode=cfg['nbnm'].get('entry_mode','story_prefix'),
                action_selection='joint_mode', evaluation_split=split, opening_seeds=seeds,
                attempts=len(results), opening_passes=sum(r['reached_midboss'] for r in results),
                opening_results=sorted(results,key=lambda r:r['native_seed']), decisions=decisions,
                elapsed_seconds=time.monotonic()-begun, training_updates=0)
            write_json(output/'report.json',report)
            return report
        report=dict(schema='frozen-NBNM-task-evaluation-v1',frozen=True,task=task,
            entry_mode=cfg['nbnm'].get('entry_mode','story_prefix'),
            opening_seeds=seeds,action_selection='joint_mode',evaluation_split=split,
            episode_results=sorted(results,key=lambda r:r['native_seed']),
            attempts=len(results),successes=sum(r['nbnm_success'] for r in results),
            spell_captures=sum(r.get('spell_captured',False) for r in results),
            spell_survived_timeouts=sum(r.get('spell_survived_timeout',False) for r in results),
            successful_mean_native_frames=(float(np.mean([r['task_native_frames'] for r in results if r['nbnm_success']]))
                if any(r['nbnm_success'] for r in results) else None),
            elapsed_seconds=time.monotonic()-begun,training_updates=0)
        if resume_bookmark:
            report['frozen_prefix_replay'] = dict(bookmark=resume_bookmark['manifest'],
                stage=resume_bookmark['stage'], stage_frame=resume_bookmark['stage_frame'],
                prior_original_policy_decisions=resume_bookmark['policy_steps'], resumed_policy_decisions=decisions)
        report['cuda_graph_inference']=graphs is not None
        report['soft_movement_scores']=bool(getattr(model,'soft_risk_composition',False))
        if check_movement:
            report['verified_movement_decisions']=verified_movement_decisions
        if exit_scoring:
            report['corridor_exit_scoring'] = dict(exit_totals, probe_limit=cfg['nbnm']['frozen_corridor_exit_limit'])
        if idle_bonus:
            report['frozen_safe_idle_bonus']=idle_bonus
        if heading_bonus:
            report['frozen_safe_heading_bonus']=heading_bonus
        if wall_escape:
            report['frozen_wall_escape']=True
        if route_wait:
            report['frozen_route_aware_wait']=True
        report['graph_capture_seconds']=graphs.capture_seconds if graphs is not None else 0.
        if any('native_event_progress' in r for r in results):
            from ..nbnm_event_progress import summarize_evaluation_events
            report['native_event_outcomes']=summarize_evaluation_events(results)
        write_json(output/'report.json',report)
        return report
    finally:
        model.frozen_safe_idle_bonus=old_idle_bonus
        model.frozen_safe_heading_bonus=old_heading_bonus
        model.frozen_heading_steps=old_heading_steps
        model.frozen_wall_escape=old_wall_escape
        model.frozen_route_aware_wait=old_route_wait
        if graphs is not None:graphs.clear()
        if vector is not None:vector.close()
        thrl_vector.worker_factory=old
