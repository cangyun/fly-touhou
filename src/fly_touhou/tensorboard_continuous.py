"""JSON-first continuous-family exporter for the existing TensorBoard bridge.

Committed behavior NPZ gets one cached CPU NumPy reduction. No Torch,
checkpoint,model,neural replay,game control or independent service is used.
The exporter is passed the existing bridge's shared JSON reader and writer.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

ENTRY_SCHEMA='continuous-connectome-warmup-PPO30-entry-v1'
DEEP_PPO_ENTRY_SCHEMA='continuous-deep128x3-PPO30-cost-routing-entry-v1'
PPO_ENTRY_SCHEMAS=(ENTRY_SCHEMA,DEEP_PPO_ENTRY_SCHEMA)
FROZEN_ENTRY_SCHEMA='continuous-frozen-parameter-development-entry-v1'
RULE_ENTRY_SCHEMA='continuous-complete-rule-episode-pretraining-entry-v1'
DN_CACHE_ENTRY_SCHEMA='continuous-finaltheta-fixed8-DN-cache-entry-v1'
DN_HEAD_ENTRY_SCHEMA='continuous-fullhistory-DN-head-calibration-entry-v1'
DEEP_FROZEN_ENTRY_SCHEMA='continuous-deep-aware-frozen-parameter-development-entry-v1'
DEEP_DN_HEAD_ENTRY_SCHEMA='continuous-fullhistory-DN-deep128x3-head-calibration-entry-v1'
NO_BOMB_ENTRY_SCHEMA='continuous-and-D95-NoBomb-first-contact-entry-v1'
OBJECT_BETA0_SCHEMA='continuous-object-learning-runtime-risk-beta0-NoBomb-first-contact-entry-v1'
NO_BOMB_ENTRY_SCHEMAS=(NO_BOMB_ENTRY_SCHEMA,OBJECT_BETA0_SCHEMA)
NO_BOMB_PROBE_SCHEMA='th14-NoBomb-uncapped-first-observed-contact-v1'
OBJECT_GUIDANCE_SCHEMA='continuous-object-NoBomb-guidance-entry-v1'
OBJECT_AVOIDANCE_PPO_SCHEMA='continuous-object-NoBomb-survival-PPO-entry-v1'
ACTION_QUERY_AVOIDANCE_PPO_SCHEMA='continuous-action-query-object-NoBomb-survival-PPO-entry-v1'
FULL_SENSORY_PPO_SCHEMA='continuous-full-sensory-NoBomb-survival-PPO-entry-v1'
OBJECT_AVOIDANCE_PPO_SCHEMAS=(OBJECT_AVOIDANCE_PPO_SCHEMA,ACTION_QUERY_AVOIDANCE_PPO_SCHEMA,FULL_SENSORY_PPO_SCHEMA)
DN_HEAD_ENTRY_SCHEMAS=(DN_HEAD_ENTRY_SCHEMA,DEEP_DN_HEAD_ENTRY_SCHEMA)
ENTRY_SCHEMAS=(*PPO_ENTRY_SCHEMAS,FROZEN_ENTRY_SCHEMA,RULE_ENTRY_SCHEMA,DN_CACHE_ENTRY_SCHEMA,
    *DN_HEAD_ENTRY_SCHEMAS,DEEP_FROZEN_ENTRY_SCHEMA,*NO_BOMB_ENTRY_SCHEMAS,
    OBJECT_GUIDANCE_SCHEMA,*OBJECT_AVOIDANCE_PPO_SCHEMAS)
CACHED_SCHEMA='continuous-connectome-frozen-cached-game-heldout-evaluation-v1'
ACTIVITY_SCHEMA='continuous-actual-population-activation-trace-v1'
PATCH='continuous-full-sensory-NoBomb-PPO-v13'
VALUE_CHANNELS=('survival','fight','resource')
SCOPE=('Continuous complete-connectome RNN; dimensionless ANN activity, not Hz/mV/spikes. '
    'Warmup supervised losses, PPO learning, frozen cached train/heldout proxies and frozen native games '
    'have separate tags/axes. Cached rows/forwards are not native actions or clears. '
    'D95 costs and geometry are fallible proxy labels. Unknown values are omitted, not zero-filled. '
    'Failure-tail ANN cursors are trace pointers,not complete brain activity checkpoints. '
    'Existing JSON and once-per-paid-update CPU behavior reductions;no checkpoint,Torch,neural-state replay, '
    'game control action or independent service is opened/started.')


def current_algorithm_jobs(exporter_state,sources):
    """Expose only configured active sources; retain historical cursor/event data."""
    jobs=exporter_state.get('fly_online_jobs',{})
    return {label:jobs.get(label,dict(source=str(folder),phase='waiting_for_source',producer_terminal=False))
        for label,folder in sources.items()}


def _number(value):
    return value if type(value) in (int,float) and math.isfinite(value) else None


def _count(value):
    return value if type(value) is int and value>=0 else None


def _signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _read(path,reader):
    if path.suffix!='.json':raise ValueError('Continuous exporter accepts JSON files only')
    value=reader(path)
    if not isinstance(value,dict):raise ValueError('Continuous producer JSON must be an object')
    return value


def _stamp(path):
    stat=path.stat();return [stat.st_mtime_ns,stat.st_size]


def _scalar(exporter,tag,value,step):
    if _number(value) is not None:exporter.scalar(tag,value,step)


def _text(exporter,tag,value,step):
    if hasattr(exporter,'text'):exporter.text(tag,str(value),step)


def _brain_mode(*records):
    for row in records:
        if not isinstance(row,dict):continue
        for name in ('trainable_brain','trainable_internal_brain'):
            if type(row.get(name)) is bool:return row[name]
        if type(row.get('fixed_brain')) is bool:return not row['fixed_brain']
        algorithm=row.get('algorithm')
        if isinstance(algorithm,dict):
            if type(algorithm.get('fixed_brain')) is bool:return not algorithm['fixed_brain']
            core=algorithm.get('core',{})
            if type(core.get('trainable_brain')) is bool:return core['trainable_brain']
    return None


def _flatten_terms(mapping,prefix=''):
    result={}
    if not isinstance(mapping,dict):return result
    for key,value in mapping.items():
        tag=prefix+str(key)
        if _number(value) is not None:result[tag]=value
        elif key=='value_loss' and isinstance(value,list) and len(value)==3:
            result.update({tag+'/'+channel:v for channel,v in zip(VALUE_CHANNELS,value) if _number(v) is not None})
        elif isinstance(value,dict):result.update(_flatten_terms(value,tag+'/'))
    return result


def _mean_terms(terms):
    rows=[(_flatten_terms(t),t.get('real_rows')) for t in terms if isinstance(t,dict)]
    keys=set().union(*(set(r[0]) for r in rows)) if rows else set()
    result={}
    for key in keys:
        selected=[(flat[key],weight) for flat,weight in rows if key in flat]
        weighted=all(_number(w) is not None and w>0 for _,w in selected)
        result[key]=(sum(v*w for v,w in selected)/sum(w for _,w in selected) if weighted
            else sum(v for v,_ in selected)/len(selected))
    return result


def _warmup(exporter,prefix,learning,costs,cursor):
    if not isinstance(learning,dict) or learning.get('phase')!='cached_teacher_warmup':return
    total=_count(costs.get('warmup_Adam'));local=_count(learning.get('optimizer_updates'))
    if total is None or local is None:return
    terms=learning.get('terms') or []
    if not isinstance(terms,list):raise ValueError('Continuous warmup terms must be an ordered list')
    if terms:
        if len(terms)>local:raise ValueError('Warmup terms cannot exceed paid local Adam')
        base=total-len(terms)
        if base<0:raise ValueError('Warmup Adam cannot precede its actual term history')
        candidates=[(base+i+1,t) for i,t in enumerate(terms)]
    else:candidates=[(total,learning.get('last_terms',{}))]
    for step,term in candidates:
        if step<=cursor.get('warmup_step',0) or not isinstance(term,dict):continue
        for key in ('weighted_total_loss','safe','future','hit','rule','deathbomb','cost'):
            _scalar(exporter,prefix+'warmup/loss/'+key,term.get(key),step)
        _scalar(exporter,prefix+'warmup/gradient/preclip_global_norm',term.get('grad_norm'),step)
        for key,value in (term.get('label_counts') or {}).items():
            _scalar(exporter,prefix+'warmup/label_counts/'+key,value,step)
        cursor['warmup_step']=step
    for key in ('optimizer_updates','backward_passes','training_rows','heldout_rows','forward_real_rows',
            'loss_real_rows','forward_lane_decisions','checkpoint_recompute_lane_decisions'):
        _scalar(exporter,prefix+'warmup/counts/'+key,learning.get(key),total)
    scope=dict(heldout_rule=learning.get('heldout_rule'),
        heldout_labels_used=learning.get('heldout_labels_used'),
        heldout_rows_are_forward_exposure_not_optimized_labels=True,
        cached_rows_are_native_actions=False)
    signature=_signature(scope)
    if signature!=cursor.get('warmup_scope_identity'):
        _text(exporter,prefix+'warmup/scope',scope,total);cursor['warmup_scope_identity']=signature


def _training_contacts(exporter,prefix,number,report,cursor):
    """Backfill new tags in the original completed-episode order, once per update."""
    learning=report.get('learning') or {};costs=report.get('costs') or {}
    if not (type(learning.get('optimizer_updates')) is int and learning['optimizer_updates']>0
            and learning.get('optimizer_transaction_complete') is True
            and costs.get('completed_rollouts')==number):return
    base=prefix+'native/changing_policy_training_games/'
    if not cursor.get('training_contact_scope'):
        _text(exporter,base+'observed_contact_scope',
            'Changing-policy training games, not frozen development. First observed contact includes '
            'contacts subsequently rescued by deathbomb; it is not Miss. Usually four-native-tick '
            'decision endpoints, not exact native event frames. Unknown/None is omitted, not zero. '
            'Wall fraction uses unprotected pre-action movement observations within40 native units '
            'of either horizontal wall. Episode ordinal matches the existing native curves.',0)
        cursor['training_contact_scope']=True
    for item in report.get('completed_episodes') or []:
        game=item.get('summary',item)
        cursor['training_contact_episodes']=cursor.get('training_contact_episodes',0)+1
        step=cursor['training_contact_episodes'];diagnostic=game.get('movement_diagnostics') or {}
        for key in ('first_collision_observed_frames','vulnerable_edge_fraction','observed_collision_events',
                'Bombs_used_at_first_collision_observed','vulnerable_decisions_to_first_collision_observed'):
            _scalar(exporter,base+key,diagnostic.get(key),step)


def _updates(exporter,prefix,root,reader,cursor):
    files=[]
    for path in (root/'traces').glob('update-*.json'):
        match=re.fullmatch(r'update-(\d+)',path.stem)
        if match:files.append((int(match[1]),path))
    for number,path in sorted(files):
        update_pending=number>cursor.get('update_number',0)
        dodge_pending=number>cursor.get('own_policy_dodge_number',0)
        contact_pending=number>cursor.get('training_contact_update_number',0)
        if not update_pending and not dodge_pending and not contact_pending:continue
        report=_read(path,reader);learning=report.get('learning',{});costs=report.get('costs',{})
        if contact_pending:
            _training_contacts(exporter,prefix,number,report,cursor)
            cursor['training_contact_update_number']=number
        if dodge_pending:
            from .tensorboard_own_policy_dodge import export_paid_dodge
            export_paid_dodge(exporter,prefix,root,number,report,read_json=reader,cursor=cursor)
            cursor['own_policy_dodge_number']=number
        if not update_pending:continue
        step=_count(costs.get('PPO_Adam'))
        if step is None:continue
        for key,value in _mean_terms(learning.get('terms') or []).items():
            if key=='preclip_grad_norm':tag='PPO/gradient/preclip_global_norm'
            else:tag='PPO/terms/'+key
            _scalar(exporter,prefix+tag,value,step)
        for key in ('optimizer_updates','backward_passes','trained_real_rows','loss_real_rows','epochs_completed',
                'forward_lane_decisions','checkpoint_recompute_lane_decisions','initial_old_logp_max_abs_error'):
            _scalar(exporter,prefix+'PPO/update/'+key,learning.get(key),step)
        post=learning.get('post_epoch') or []
        for epoch,diagnostic in enumerate(post):
            if not isinstance(diagnostic,dict):continue
            q=prefix+'PPO/post_epoch/'+str(epoch+1)+'/'
            for key in ('approx_kl','samples','continuous_lane_decisions'):_scalar(exporter,q+key,diagnostic.get(key),step)
            EV=diagnostic.get('explained_variance')
            if isinstance(EV,list) and len(EV)==3:
                for channel,value in zip(VALUE_CHANNELS,EV):_scalar(exporter,q+'value_EV/'+channel,value,step)
        activity=report.get('activity') or {}
        if activity.get('schema')==ACTIVITY_SCHEMA:
            for key in ('rows','candidate_rows','selected_critical','selected_uniform','payload_bytes',
                    'extra_core_forwards','extra_perception_forwards','extra_teacher_queries'):
                _scalar(exporter,prefix+'activity_unitless/trace_metadata/'+key,activity.get(key),number)
        for item in report.get('completed_episodes') or []:
            game=item.get('summary',item);cursor['episodes']=cursor.get('episodes',0)+1
            game_step=cursor['episodes'];base=prefix+'native/changing_policy_training_games/'
            for key in ('native_frames','misses','bombs_used','stage'):_scalar(exporter,base+key,game.get(key),game_step)
            if game.get('outcome') in ('clear','gameover'):_scalar(exporter,base+'clear',int(game['outcome']=='clear'),game_step)
            _scalar(exporter,base+'first_observed_collision_frames',
                (game.get('movement_diagnostics') or {}).get('first_collision_observed_frames'),game_step)
        cursor['update_number']=number


def _cached(exporter,prefix,root,reader,cursor,step):
    for phase in ('initial','final'):
        # Entry copies the actual evaluator result; the canonical folder report
        # also supports a late-arriving standalone evaluation with the same schema.
        paths=[root/f'cached-evaluation-{phase}.json',root/f'cached-evaluation-{phase}'/'report.json']
        path=next((p for p in paths if p.is_file()),None)
        if path is None:continue
        stamp=_stamp(path);cursor_key='cached_'+phase
        if cursor.get(cursor_key)==stamp:continue
        report=_read(path,reader)
        if (report.get('schema')!=CACHED_SCHEMA
                or report.get('status')!='completed_frozen_offline_history_evaluation'):
            raise ValueError('Unknown/incomplete continuous cached evaluation')
        p=prefix+'cache_frozen/'+phase+'/'
        for role in ('train','heldout'):
            values=(report.get('results') or {}).get(role,{})
            for name in ('rows','queried_cost_rows','valid_cost_rows','trusted_mixed_geometry_hard_rows'):
                _scalar(exporter,p+role+'/counts/'+name,values.get(name),step)
            if isinstance(values.get('games'),list):_scalar(exporter,p+role+'/counts/games',len(values['games']),step)
            for model in ('learned_DN_cost','train_mean_direction_constant'):
                for domain in ('valid','queried_including_revoked','hard_valid'):
                    metrics=(values.get('cost') or {}).get(model,{}).get(domain,{})
                    for name in ('rows','min_centered_MAE','mean_teacher_cost_regret','min_cost_action_agreement',
                            'cost_mixed_scenes','scene_cost_pair_rank','direction_pairs','pooled_cost_pair_rank'):
                        _scalar(exporter,p+role+'/proxy_cost/'+model+'/'+domain+'/'+name,metrics.get(name),step)
            for policy,metrics in (values.get('hard_policy') or {}).items():
                if policy not in ('raw_actor','composed_DN_cost','train_constant_cost','teacher_cost_label_counterfactual'):continue
                for name in ('rows','danger_probability','conditional_movement_danger_probability','Bomb_probability',
                        'greedy_Bomb_count','greedy_movement_rows','greedy19_movement_danger_count','forced_movement_greedy_danger_count'):
                    _scalar(exporter,p+role+'/proxy_hard_policy/'+policy+'/'+name,metrics.get(name),step)
            population=(report.get('population') or {}).get(role,{})
            for group in ('input_ports','middle_remaining','DN_outputs'):
                metrics=population.get(group,{})
                for name in ('neurons','sampled_scenes','mean_activation','mean_absolute_activation','mean_within_scene_SD',
                        'SD_across_scene_means','absolute_activity_gt_095_fraction','absolute_activity_gt_099_fraction'):
                    _scalar(exporter,p+role+'/activity_unitless/'+group+'/'+name,metrics.get(name),step)
        for key,value in (report.get('paid') or {}).items():_scalar(exporter,p+'offline_evaluation_cost/'+key,value,step)
        _scalar(exporter,p+'offline_evaluation_cost/elapsed_seconds',report.get('elapsed_seconds'),step)
        for key in ('optimizer_updates','teacher_queries','native_actions'):_scalar(exporter,p+'offline_evaluation_cost/'+key,report.get(key),step)
        _text(exporter,p+'scope',dict(status=report['status'],heldout_definition=report.get('heldout_definition'),
            first_state=report.get('first_state'),interpretation=report.get('interpretation'),
            frozen_parameters_unchanged=report.get('frozen_parameters_unchanged'),
            native_gameplay=False,activity_units='dimensionless ANN activation'),step)
        cursor[cursor_key]=stamp;cursor.setdefault('cached_evaluations',{})[phase]=dict(source=str(path),
            status=report['status'],heldout_definition=report.get('heldout_definition'))


def _native(exporter,prefix,row,terminal):
    evaluation=row.get('development') if terminal=='result.json' else row.get('evaluation')
    if not isinstance(evaluation,dict) or not evaluation:return
    games=evaluation.get('episode_results') or []
    complete=(terminal=='result.json' and row.get('status')=='completed' and len(games)==20
        and all(g.get('outcome') in ('clear','gameover') and g.get('full_story_episode') is True for g in games))
    if complete:
        p=prefix+'native/frozen_development/final_by_completed_games/';step=20
    else:
        decisions=next((_count(evaluation.get(k)) for k in ('policy_decisions','newly_executed_decisions','decisions')
            if _count(evaluation.get(k)) is not None),None)
        step=decisions if decisions is not None else _count(evaluation.get('completed'))
        if step is None:return
        p=prefix+'native/frozen_development/progress_by_'+('native_decisions/' if decisions is not None else 'completed_games/')
    for name in ('completed','episodes','target','clears','clear_rate','mean_native_frames','mean_max_stage','mean_misses','mean_bombs_used',
            'workers','policy_decisions','wall_seconds'):
        _scalar(exporter,p+name,evaluation.get(name),step)
    current=evaluation.get('current_environments') or []
    stages=[g.get('stage') for g in current if _number(g.get('stage')) is not None]
    if current:_scalar(exporter,p+'active_worlds',len(current),step)
    if stages:_scalar(exporter,p+'current_max_stage',max(stages),step)
    hits=[(g.get('movement_diagnostics') or {}).get('first_collision_observed_frames') for g in games]
    hits=[x for x in hits if _number(x) is not None]
    if hits:_scalar(exporter,p+'first_observed_collision_mean_frames',sum(hits)/len(hits),step)


def _no_bomb_native(exporter,prefix,row,launch,terminal,cursor):
    evaluation=row.get('evaluation') if terminal=='result.json' else row
    if not isinstance(evaluation,dict):return {}
    complete=(terminal=='result.json' and row.get('status')=='completed')
    games=evaluation.get('episode_results') or []
    step=_count(evaluation.get('episodes') if complete else evaluation.get('completed'))
    if step is None:return {}
    p=prefix+'native/frozen_NoBomb_first_contact/'
    override=row.get('runtime_policy_override') or launch.get('runtime_policy_override') or {}
    source_beta=row.get('source_beta',override.get('source_beta',launch.get('source_beta_expected')))
    effective_beta=row.get('effective_risk_beta',override.get('effective_beta',launch.get('effective_risk_beta')))
    _scalar(exporter,p+'runtime_policy/source_beta',source_beta,step)
    _scalar(exporter,p+'runtime_policy/effective_beta',effective_beta,step)
    if override and not cursor.get('NoBomb_runtime_override'):
        _text(exporter,p+'runtime_policy/scope',override,0);cursor['NoBomb_runtime_override']=True
    _scalar(exporter,p+'progress/completed',step,step)
    _scalar(exporter,p+'progress/target',evaluation.get('episodes') if complete else evaluation.get('target'),step)
    _scalar(exporter,p+'progress/terminal',int(complete),step)
    _scalar(exporter,p+'progress/student_decisions',evaluation.get('student_decisions'),step)
    _scalar(exporter,p+'progress/elapsed_seconds',row.get('elapsed_seconds'),step)
    for name in ('first_observed_contacts','natural_clear_right_censored','actual_Bombs',
            'first_stage_without_observed_contact_count','first_stage_without_observed_contact_fraction'):
        _scalar(exporter,p+'summary/'+name,evaluation.get(name),step)
    # Contact distribution is conditional on observed contacts. Censored natural
    # clears remain separate and never acquire a synthetic contact time of zero.
    for domain in ('observed_contact_frames','observed_endpoint_frames'):
        for name in ('count','mean','median','q25','min','max'):
            _scalar(exporter,p+domain+'/'+name,(evaluation.get(domain) or {}).get(name),step)
    for stage,value in (evaluation.get('contact_stage_counts') or {}).items():
        _scalar(exporter,p+'contact_stage_counts/'+stage,value,step)
    for game in games:
        ordinal=_count(game.get('evaluation_game'))
        if ordinal is None or ordinal<=cursor.get('NoBomb_game',0):continue
        for name in ('first_contact_frames','native_frames','actual_bombs_used','stage'):
            _scalar(exporter,p+'by_paired_game/'+name,game.get(name),ordinal)
        for name in ('first_collision_observed','right_censored_at_natural_clear','first_stage_without_observed_contact'):
            if type(game.get(name)) is bool:_scalar(exporter,p+'by_paired_game/'+name,int(game[name]),ordinal)
        cursor['NoBomb_game']=ordinal
    if not cursor.get('NoBomb_scope'):
        _text(exporter,p+'scope',dict(protocol='Frozen greedy T1;source19 Bomb18 masked for every policy;original uncapped opening',
            endpoint='First native accepted state4/Miss at acknowledged endpoint;usually up to4ticks,not exact event frame',
            natural_clear='Right-censored separately;contact-only statistics are not overall survival statistics',
            not_historical_Bomb_allowed_75percent=True,changing_policy_training=False,
            optimizer_updates=0,teacher_takeovers=0),0)
        cursor['NoBomb_scope']=True
    return dict(completed=step,target=evaluation.get('episodes') if complete else evaluation.get('target'),
        terminal=complete,first_observed_contacts=evaluation.get('first_observed_contacts'),
        natural_clear_right_censored=evaluation.get('natural_clear_right_censored'),
        first_stage_without_observed_contact_fraction=evaluation.get('first_stage_without_observed_contact_fraction'),
        family=launch.get('family'),source_beta=source_beta,effective_beta=effective_beta)


def _object_guidance(exporter,prefix,root,row,launch,reader,cursor):
    """Committed supervision updates only;never route this task through PPO."""
    costs=row.get('costs') or {};step=_count(costs.get('supervision_Adam')) or 0
    p=prefix+'NoBomb_object_supervision/'
    for key in ('supervision_Adam','student_actions','completed_rollouts'):
        _scalar(exporter,p+'counts/'+key,costs.get(key),step)
    _scalar(exporter,p+'target/student_decisions',launch.get('student_decisions'),step)
    _scalar(exporter,p+'target/supervision_Adam_max',launch.get('supervised_Adam_max'),step)
    if _count(launch.get('student_decisions')) is not None:
        _scalar(exporter,p+'target/completed_rollouts',launch['student_decisions']//7680,step)
    _scalar(exporter,p+'target/real_rows_per_rollout',7680,step)
    for key in ('PPO_updates','value_updates'):_scalar(exporter,p+'not_PPO_or_value/'+key,row.get(key),step)
    for path in sorted((root/'traces').glob('update-*.json')):
        match=re.fullmatch(r'update-(\d+)',path.stem)
        if not match:continue
        number=int(match[1])
        if number<=cursor.get('object_guidance_update',0):continue
        update=_read(path,reader);learning=update.get('learning') or {};paid=update.get('costs') or {}
        adam=_count(paid.get('supervision_Adam'))
        if (adam is not None and (_count(learning.get('optimizer_updates')) or 0)>0
                and learning.get('optimizer_transaction_complete') is True):
            for name,value in _mean_terms(learning.get('terms') or []).items():
                _scalar(exporter,p+'terms/'+name,value,adam)
            for name in ('optimizer_updates','backwards','epochs_completed','first_global_preclip_grad_norm',
                    'core_lane_forwards','checkpoint_recompute_lane_forwards','trained_rows','forward_rows',
                    'object_attention_forward_rows','initial_NoBomb_logp_max_abs_error'):
                _scalar(exporter,p+'update/'+name,learning.get(name),adam)
            for name,value in _flatten_terms(learning.get('first_gradient_groups') or {}).items():
                _scalar(exporter,p+'first_preclip_gradient_groups/'+name,value,adam)
        cursor['object_guidance_update']=number
    if not cursor.get('object_guidance_scope'):
        _text(exporter,p+'scope','Student-controlled NoBomb first-native-contact supervision;NOT PPO. '
            'Rewards are recorded only;no GAE/returns/value learning. Three critic parameters/Adam retained. '
            'Collection counts publish after a full7680-row rollout,not per-action. '
            'Frozen NoBomb20 is separate;natural clear right-censored,not historical Bomb-allowed75percent.',0)
        cursor['object_guidance_scope']=True
    return dict(supervision_Adam=_count(costs.get('supervision_Adam')),
        student_actions=_count(costs.get('student_actions')),student_decisions_target=launch.get('student_decisions'),
        completed_rollouts=_count(costs.get('completed_rollouts')),supervised_Adam_max=launch.get('supervised_Adam_max'),
        PPO_updates=row.get('PPO_updates'),value_updates=row.get('value_updates'),
        actual_executor=dict(pid=launch.get('PID'),create_time=launch.get('create_time')))


def _avoidance_ppo(exporter,prefix,root,row,launch,reader,cursor):
    """NoBomb task PPO has its own axes;its contacts are not native Game Overs."""
    costs=row.get('costs') or {};step=_count(costs.get('PPO_Adam')) or 0
    p=prefix+'NoBomb_survival_PPO/';settings=launch.get('PPO') or {}
    for key in ('PPO_Adam','student_actions','completed_rollouts'):
        _scalar(exporter,p+'counts/'+key,costs.get(key),step)
    for key,value in dict(student_decisions=launch.get('student_decisions'),
            PPO_Adam_max=settings.get('Adam_max'),rollouts=launch.get('rollouts')).items():
        _scalar(exporter,p+'target/'+key,value,step)
    for channel,value in zip(VALUE_CHANNELS,settings.get('objective_weights') or []):
        _scalar(exporter,p+'actor_objective_weights/'+channel,value,step)
    declared=launch.get('perception_branches')
    if isinstance(declared,dict):
        perception={key:value for key,value in declared.items() if type(value) is bool}
        perception_source='explicit launch.perception_branches'
    elif launch.get('schema') in (OBJECT_AVOIDANCE_PPO_SCHEMA,ACTION_QUERY_AVOIDANCE_PPO_SCHEMA):
        # These two typed legacy implementations explicitly freeze map/numeric
        # and spatial attention;this is a source contract,not model inspection.
        perception=dict(map=False,spatial=False,numeric=False,object=True,
            query=launch.get('schema')==ACTION_QUERY_AVOIDANCE_PPO_SCHEMA)
        perception_source='legacy typed frontend source contract'
    else:perception={};perception_source='not recorded;unknown,not zero-filled'
    for key,value in perception.items():_scalar(exporter,p+'perception_trainable/'+key,int(value),step)
    for path in sorted((root/'traces').glob('update-*.json')):
        match=re.fullmatch(r'update-(\d+)',path.stem)
        if not match:continue
        number=int(match[1])
        if number<=cursor.get('avoidance_PPO_update',0):continue
        update=_read(path,reader);learning=update.get('learning') or {};paid=update.get('costs') or {}
        adam=_count(paid.get('PPO_Adam'))
        if (adam is not None and (_count(learning.get('optimizer_updates')) or 0)>0
                and learning.get('optimizer_transaction_complete') is True):
            for name,value in _mean_terms(learning.get('terms') or []).items():
                _scalar(exporter,p+'terms/'+name,value,adam)
            for name in ('optimizer_updates','backward_passes','backwards','epochs_completed','trained_rows',
                    'trained_real_rows','loss_real_rows','first_global_preclip_grad_norm',
                    'initial_old_logp_max_abs_error','initial_NoBomb_logp_max_abs_error'):
                _scalar(exporter,p+'update/'+name,learning.get(name),adam)
            for field in ('first_gradient_groups','first_perception_gradient_branches'):
                for name,value in _flatten_terms(learning.get(field) or {}).items():
                    _scalar(exporter,p+field+'/'+name,value,adam)
            for epoch,diagnostic in enumerate(learning.get('post_epoch') or [],1):
                q=p+'post_epoch/'+str(epoch)+'/'
                for name,value in _flatten_terms(diagnostic).items():_scalar(exporter,q+name,value,adam)
                ev=diagnostic.get('explained_variance')
                if isinstance(ev,list) and len(ev)==3:
                    for channel,value in zip(VALUE_CHANNELS,ev):_scalar(exporter,q+'value_EV/'+channel,value,adam)
        cursor['avoidance_PPO_update']=number
    if not cursor.get('avoidance_PPO_scope'):
        _text(exporter,p+'scope',dict(task='NoBomb first-native-contact survival task;not full-story Bomb PPO',
            actor_objective_weights=settings.get('objective_weights'),three_value_heads=launch.get('all_three_value_heads_retained_and_updated'),
            task_end=launch.get('task_end'),task_reward=launch.get('task_reward'),
            frozen_NoBomb20_separate=True,full_story_training_resumed=False),0)
        cursor['avoidance_PPO_scope']=True
    return dict(PPO_Adam=_count(costs.get('PPO_Adam')),student_actions=_count(costs.get('student_actions')),
        completed_rollouts=_count(costs.get('completed_rollouts')),PPO_Adam_max=settings.get('Adam_max'),
        rollouts_target=launch.get('rollouts'),student_decisions_target=launch.get('student_decisions'),
        actor_objective_weights=settings.get('objective_weights'),full_story_PPO=False,
        entry_schema=launch.get('schema'),sensory_change=launch.get('sensory_change'),
        new_action_query_parameters=launch.get('new_action_query_parameters'),
        perception_branches=perception,perception_branches_source=perception_source,
        actual_executor=dict(pid=launch.get('PID'),create_time=launch.get('create_time')))


def _complete_rule(exporter,prefix,row,costs,cursor):
    """Offline supervised Adam and fit-held history have their own axes."""
    step=_count(costs.get('optimizer_updates'))
    if step is None:return
    p=prefix+'complete_rule_supervised/'
    for name,value in costs.items():_scalar(exporter,p+'counts/'+name,value,step)
    learning=row.get('learning') or {}
    if isinstance(learning,dict):
        for name,value in _flatten_terms(learning).items():_scalar(exporter,p+'terms/'+name,value,step)
    for i,value in enumerate(row.get('cursors') or []):_scalar(exporter,p+'source_cursor/lane'+str(i),value,step)
    held=row.get('fit_held') or row.get('held')
    if isinstance(held,dict):
        for name,value in (held.get('counts') or {}).items():_scalar(exporter,p+'offline_fit_held/counts/'+name,value,step)
        if held.get('status')=='completed_frozen_offline_fit_held':
            cursor['fit_held_complete']=True
    _text(exporter,p+'scope',dict(native_actions=0,PPO_updates=0,
        heldout='readout-fit held histories;not independent native games',
        complete_epoch=row.get('complete_epoch'),fit_held_complete=cursor.get('fit_held_complete',False)),step)


def _finaltheta_DN_cache(exporter,prefix,row,launch,cursor):
    """Read-only neural replay, not PPO or optimizer progress."""
    cache=row.get('cache') or {};counts=row.get('counts') or cache.get('counts') or {}
    step=_count(counts.get('real_rows'))
    if step is None:return {}
    p=prefix+'finaltheta_DN_cache/by_real_source_rows/'
    for key,value in counts.items():_scalar(exporter,p+'counts/'+key,value,step)
    _scalar(exporter,p+'elapsed_seconds',row.get('elapsed_seconds'),step)
    target=launch.get('rows')
    _scalar(exporter,p+'target_source_rows',target,step)
    if _number(target) is not None and target>0:_scalar(exporter,p+'source_row_fraction',step/target,step)
    cursors=row.get('cursors')
    if cursors is None and cache.get('status')=='complete_finaltheta_all8_DN':
        cursors=[e['rows'] for e in cache.get('episode_offsets',[])]
    if isinstance(cursors,list):
        for i,value in enumerate(cursors):_scalar(exporter,p+'source_cursor/lane'+str(i),value,step)
    scope=dict(kind='frozen-theta history-to-DN cache;no learning or native game',
        DN_output=launch.get('DN_output'),optimizer_updates=row.get('optimizer_updates',launch.get('optimizer_updates')),
        source_checkpoint_sha256=launch.get('source_sha256'),real_lanes=launch.get('real_lanes'))
    if cursor.get('DN_cache_scope')!=_signature(scope):
        _text(exporter,p+'scope',scope,step);cursor['DN_cache_scope']=_signature(scope)
    return dict(counts=counts,cursors=cursors,target_source_rows=target,real_source_rows=step,
        elapsed_seconds=row.get('elapsed_seconds'))


def _DN_head_calibration(exporter,prefix,root,row,costs,reader,cursor):
    """Separate risk/actor Adam axes;fit-held proxies never become native games."""
    total=sum(_count(costs.get(k)) or 0 for k in ('risk_Adam','actor_Adam'))
    p=prefix+'cached_DN_head_calibration/'
    for key,value in costs.items():_scalar(exporter,p+'counts/'+key,value,total)
    _scalar(exporter,p+'elapsed_seconds',row.get('elapsed_seconds'),total)
    for role,phases in (('risk',('risk_training','risk_completed')),('actor',('actor_BC',))):
        learning=row.get(role+'_learning') or (row.get('learning') if row.get('phase') in phases else {}) or {}
        committed=role+'_learning' in row
        history=learning.get('history') or []
        for term in history if committed else history[-1:]:
            step=term.get('updates',term.get('update'))
            cursor_key='DN_head_'+role+('_final_step' if committed else '_live_step')
            if _count(step) is None or step<=cursor.get(cursor_key,0):continue
            for key,value in term.items():
                if key not in ('updates','update'):_scalar(exporter,p+role+('/terms/' if committed else '/latest/')+key,value,step)
            cursor[cursor_key]=step
    for phase in ('before','after'):
        path=root/f'held-{phase}.json';stamp=_stamp(path) if path.is_file() else None
        held=_read(path,reader) if stamp and cursor.get('DN_head_held_'+phase)!=stamp else row.get('held_'+phase)
        if not isinstance(held,dict):continue
        held_identity=_signature(held)
        if cursor.get('DN_head_held_identity_'+phase)==held_identity:continue
        if held.get('held_only') is not True or held.get('fit_held_not_independent_native') is not True:
            raise ValueError('Head calibration requires its explicit fit-held proxy report')
        for episode,values in (held.get('by_episode_index') or {}).items():
            held_step=0 if phase=='before' else total
            q=p+'fit_held_'+phase+'/episode_index'+str(episode)+'/'
            for key in ('rows','teacher_rows','teacher_weight','danger_queries','safe_queries','eligible_geometry_rows',
                    'risk_best_geometry_danger'):_scalar(exporter,q+'counts/'+key,values.get(key),held_step)
            for tag,num,den in (('teacher_KL','teacher_KL_sum','teacher_weight'),
                    ('teacher_CE','teacher_CE_sum','teacher_weight'),
                    ('teacher_top1_weighted_agreement','teacher_top1_weighted_correct','teacher_weight'),
                    ('danger_recall','danger_recall_hits','danger_queries'),
                    ('safe_false_positive_rate','safe_false_positive','safe_queries')):
                if _number(values.get(num)) is not None and _number(values.get(den)) is not None and values[den]>0:
                    _scalar(exporter,q+tag,values[num]/values[den],held_step)
            for domain,metric in (values.get('domains') or {}).items():
                d=q+'proxy_geometry/'+domain+'/'
                for key in ('rows','greedy_moves','greedy_danger','greedy_Bomb'):_scalar(exporter,d+key,metric.get(key),held_step)
                n=metric.get('rows')
                if _number(n) is not None and n>0:
                    for tag,key in (('unconditional_danger','unconditional_danger_sum'),
                            ('conditional_movement_danger','conditional_movement_danger_sum'),('Bomb_probability','Bomb_sum')):
                        if _number(metric.get(key)) is not None:_scalar(exporter,d+tag,metric[key]/n,held_step)
        if stamp:cursor['DN_head_held_'+phase]=stamp
        cursor['DN_head_held_identity_'+phase]=held_identity
    if not cursor.get('DN_head_scope'):
        _text(exporter,p+'scope','Cached final-theta DN;only risk/actor heads learn. '
            'No PPO,brain/critic updates,neural forward,native game or teacher query;fit-held is a proxy.',0)
        cursor['DN_head_scope']=True
    return dict(costs=costs,phase=row.get('phase') or row.get('status'),elapsed_seconds=row.get('elapsed_seconds'))


def export_continuous_connectome(exporter,label,root,*,read_json):
    """Return True only for this exact family; leave old reader paths untouched."""
    root=Path(root);launch_path=root/'launch.json'
    launch=_read(launch_path,read_json) if launch_path.is_file() else {}
    candidates=[root/name for name in ('failure.json','result.json','progress.json') if (root/name).is_file()]
    if not candidates and launch.get('schema') not in ENTRY_SCHEMAS:return False
    source=candidates[0] if candidates else launch_path
    row=_read(source,read_json) if candidates else launch
    schema=launch.get('schema') if launch.get('schema') in ENTRY_SCHEMAS else row.get('schema')
    if schema not in ENTRY_SCHEMAS:return False
    if row.get('schema') not in (None,schema):
        if not (schema in NO_BOMB_ENTRY_SCHEMAS and row.get('schema')==NO_BOMB_PROBE_SCHEMA
                and row.get('entry_schema')==schema):raise ValueError('Continuous entry schema conflict')
    terminal=source.name if source.name in ('failure.json','result.json') else None
    costs=row.get('costs') or {};warm=_count(costs.get('warmup_Adam'));ppo=_count(costs.get('PPO_Adam'))
    if schema==OBJECT_GUIDANCE_SCHEMA:
        total=_count(costs.get('supervision_Adam')) or 0;warm=ppo=None
    elif schema==DN_CACHE_ENTRY_SCHEMA:
        total=_count((row.get('counts') or (row.get('cache') or {}).get('counts') or {}).get('real_rows')) or 0
    elif schema in DN_HEAD_ENTRY_SCHEMAS:total=sum(_count(costs.get(k)) or 0 for k in ('risk_Adam','actor_Adam'))
    else:total=(_count(costs.get('optimizer_updates')) or 0) if schema==RULE_ENTRY_SCHEMA else (warm or 0)+(ppo or 0)
    prefix='fly_iterations/'+label+'/continuous_connectome/'
    cursor=exporter.state.setdefault('continuous_connectome_cursors',{}).setdefault(label,{})
    if cursor.get('scope_patch')!=PATCH:
        scope=('Frozen original NoBomb first-contact comparison;no optimizer or training. '
            'Natural clear is right-censored;not the historical Bomb-allowed15/20 score.'
            if schema in NO_BOMB_ENTRY_SCHEMAS else 'Student-operated NoBomb guidance supervision,not PPO or value learning.'
            if schema==OBJECT_GUIDANCE_SCHEMA else 'NoBomb first-contact survival PPO;three value heads,actor weights(1,0,0);not full-story Bomb PPO.'
            if schema in OBJECT_AVOIDANCE_PPO_SCHEMAS else SCOPE)
        _text(exporter,prefix+'about/scope',scope,0);cursor['scope_written']=True;cursor['scope_patch']=PATCH
    if cursor.get('source') not in (None,str(root.resolve())):raise ValueError('Continuous cursor source changed under same label')
    cursor['source']=str(root.resolve())
    mode=None if schema==DN_CACHE_ENTRY_SCHEMA or schema in DN_HEAD_ENTRY_SCHEMAS else _brain_mode(row,launch)
    if mode is not None:
        exporter.scalar(prefix+'settings/trainable_brain',int(mode),total)
        exporter.scalar(prefix+'settings/fixed_internal_brain',int(not mode),total)
    for key in (('student_actions','completed_rollouts') if schema==OBJECT_GUIDANCE_SCHEMA
            else ('warmup_Adam','PPO_Adam','student_actions','completed_rollouts')):
        _scalar(exporter,prefix+'counts/'+key,costs.get(key),total)
    _scalar(exporter,prefix+'time/elapsed_seconds',row.get('elapsed_seconds'),total)
    phase=(row.get('phase') or ('failed_or_stopped' if terminal=='failure.json' else
        'awaiting_first_progress' if source.name=='launch.json' else row.get('status')))
    identity=_signature(dict(terminal=terminal,phase=phase,trainable_brain=mode,error=row.get('error')))
    if identity!=cursor.get('phase_identity'):
        _text(exporter,prefix+'status/phase',dict(phase=phase,terminal=terminal,trainable_brain=mode,
            error=row.get('error'),checkpoint_error=row.get('checkpoint_error')),total)
        cursor['phase_identity']=identity
    learning=row.get('learning') or {}
    DN_cache_progress=_finaltheta_DN_cache(exporter,prefix,row,launch,cursor) if schema==DN_CACHE_ENTRY_SCHEMA else None
    DN_head_progress=_DN_head_calibration(exporter,prefix,root,row,costs,read_json,cursor) if schema in DN_HEAD_ENTRY_SCHEMAS else None
    if schema==RULE_ENTRY_SCHEMA:_complete_rule(exporter,prefix,row,costs,cursor)
    if schema in PPO_ENTRY_SCHEMAS:_warmup(exporter,prefix,learning,costs,cursor)
    warm_path=root/'warmup-result.json'
    if schema in PPO_ENTRY_SCHEMAS and warm_path.is_file():
        stamp=_stamp(warm_path)
        if cursor.get('warmup_result')!=stamp:
            _warmup(exporter,prefix,_read(warm_path,read_json),costs,cursor);cursor['warmup_result']=stamp
    if schema in PPO_ENTRY_SCHEMAS:
        _updates(exporter,prefix,root,read_json,cursor)
        _cached(exporter,prefix,root,read_json,cursor,total)
    object_guidance=(_object_guidance(exporter,prefix,root,row,launch,read_json,cursor)
        if schema==OBJECT_GUIDANCE_SCHEMA else None)
    avoidance_PPO=(_avoidance_ppo(exporter,prefix,root,row,launch,read_json,cursor)
        if schema in OBJECT_AVOIDANCE_PPO_SCHEMAS else None)
    NoBomb_progress=_no_bomb_native(exporter,prefix,row,launch,terminal,cursor) if schema in NO_BOMB_ENTRY_SCHEMAS else None
    if schema in (OBJECT_GUIDANCE_SCHEMA,*OBJECT_AVOIDANCE_PPO_SCHEMAS):
        frozen=dict(row,evaluation=row.get('development') if terminal=='result.json' else row.get('evaluation'))
        if terminal=='result.json' or frozen['evaluation']:
            # The existing frozen-NoBomb reader owns the endpoint/censor scope.
            if terminal!='result.json':frozen=frozen['evaluation']
            NoBomb_progress=_no_bomb_native(exporter,prefix,frozen,launch,terminal,cursor)
    if schema not in (*NO_BOMB_ENTRY_SCHEMAS,OBJECT_GUIDANCE_SCHEMA,*OBJECT_AVOIDANCE_PPO_SCHEMAS):_native(exporter,prefix,row,terminal)
    exporter.state.setdefault('fly_online_jobs',{})[label]=dict(source=str(root),family='NoBomb_survival_PPO' if schema in OBJECT_AVOIDANCE_PPO_SCHEMAS else 'NoBomb_guidance_supervision' if schema==OBJECT_GUIDANCE_SCHEMA else 'frozen_NoBomb_first_contact' if schema in NO_BOMB_ENTRY_SCHEMAS else 'continuous_connectome',
        activity_units=None if schema in NO_BOMB_ENTRY_SCHEMAS else 'dimensionless ANN activation',phase=phase,producer_file=source.name,
        producer_terminal=terminal is not None,trainable_brain=mode,warmup_Adam=warm,PPO_Adam=ppo,
        student_decisions=_count(costs.get('student_actions')),
        completed_training_games=cursor.get('episodes',0) if schema in PPO_ENTRY_SCHEMAS else None,
        supervised_Adam=(_count(costs.get('supervision_Adam')) if schema==OBJECT_GUIDANCE_SCHEMA else
            _count(costs.get('optimizer_updates')) if schema==RULE_ENTRY_SCHEMA else None),
        producer_schema=schema,native_progress={k:v for k,v in
            (row.get('development' if terminal=='result.json' else 'evaluation') or {}).items()
            if k in ('completed','target','episodes','policy_decisions','workers','wall_seconds','clears','clear_rate')},
        DN_cache_progress=DN_cache_progress,
        DN_head_progress=DN_head_progress,
        NoBomb_progress=NoBomb_progress,
        object_guidance=object_guidance,
        avoidance_PPO=avoidance_PPO,
        own_policy_dodge=cursor.get('own_policy_dodge_status'),
        training_contact_games=cursor.get('training_contact_episodes'),
        cached_evaluations=cursor.get('cached_evaluations',{}),read_error=None,parser_patch=PATCH)
    return True
