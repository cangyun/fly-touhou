"""Prepare or explicitly execute the shared-connectome NBNM experiment.

No training runs without --execute. --evaluate-only uses a frozen checkpoint.
STOP finishes the current acknowledged segment/update and saves its cursor.
"""
from pathlib import Path
import argparse,copy,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',type=Path)
    p.add_argument('--evaluate-only',type=Path)
    p.add_argument('--evaluation-output',type=Path)
    p.add_argument('--evaluation-task',help='One declared task key, such as s6_BossCard5 or full_story')
    p.add_argument('--evaluation-split',choices=('development','test'),default='development')
    p.add_argument('--log-file',type=Path,help='Capture startup/import errors as well as training output')
    a=p.parse_args()
    if a.log_file:
        a.log_file.parent.mkdir(parents=True,exist_ok=True)
        sys.stdout=sys.stderr=a.log_file.open('a',encoding='utf-8',buffering=1)
        import faulthandler
        faulthandler.enable(file=sys.stderr)
        print('NBNM process started: PID='+str(os.getpid()),flush=True)
    if not a.execute and a.evaluate_only is None:
        return run(a, p)
    from fly_touhou.nbnm_job_phase import gpu_phase
    with gpu_phase('evaluation' if a.evaluate_only is not None else 'training'):
        return run(a, p)


def run(a, p):
    from fly_touhou.nbnm_run import (read, validate_spec, runtime, environment, load_model,
        load_cpu_checkpoint, atomic_torch_save, archive_torch_parameters, evaluate)
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    from fly_touhou.atomic_json_io import atomic_write_json
    from fly_touhou.nbnm_resume import (pending_objective, with_pending_objective,
        objective_for_saved_rollout, reset_event_critic, record_objective_activation)
    pending_patch=pending_objective(a.config) if a.resume else None
    from fly_touhou import nbnm_pending_guidance as guidance_resume
    pending_guidance=guidance_resume.pending(a.config) if a.resume else None
    spec=guidance_resume.apply(with_pending_objective(read(a.config),pending_patch),pending_guidance)
    from fly_touhou.training.run_contract import mark_parameters, restore_contract

    identity = spec.get('run_contract')
    groups = identity['optimizer_groups'] if identity is not None else None
    validation=validate_spec(spec)
    if not a.execute and a.evaluate_only is None:
        from torch.utils.tensorboard import SummaryWriter
        if a.resume is not None:
            from fly_touhou.nbnm_resume import preflight
            validation['resume'] = preflight(a.resume, spec, pending_patch, pending_guidance)
        print(json.dumps(validation,ensure_ascii=False,indent=2));return
    if a.evaluate_only:
        if a.evaluation_output is None:p.error('--evaluation-output required')
        if a.evaluation_task and a.evaluation_task not in spec['tasks']:p.error('Unknown --evaluation-task')
        runtime(a.evaluation_output)
        model,_=load_model(spec,a.evaluate_only,preserve_collection_input=True)
        evaluate(model,spec,a.evaluation_output,task=spec['tasks'].get(a.evaluation_task),split=a.evaluation_split)
        return
    output=Path(spec['output']);output.mkdir(parents=True,exist_ok=a.resume is not None)
    temporary=runtime(output)
    import torch
    import psutil
    from torch.utils.tensorboard import SummaryWriter
    from fly_touhou.compact_story_ppo import optimizer_for,update,restore_optimizer
    from fly_touhou.compact_story_ready import ReadyCollector
    from fly_touhou.owned_worker_affinity import worker_cpu_pool
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    proc=psutil.Process()
    cpus=worker_cpu_pool(proc.cpu_affinity(),expanded=spec.get('expanded_worker_cpu_pool',False))
    proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    begun=time.monotonic();collector=None;writer=SummaryWriter(spec.get('tensorboard_directory',str(output/'tensorboard')))
    completed=0;updates=0;cycle=0;in_progress=None;model=None;optimizer=None
    last_status={};stopped=False;episode_offset=0;episode_index=0
    status_api=None;task_cursors=None;continuation=None;resume_info=None
    last_evaluation_decisions=0;last_evaluation_time=time.monotonic();deferred_objective=False
    deferred_guidance=False
    if spec.get('status_port') is not None:
        from fly_touhou.thrl_study_io import StatusAPI
        from summarize_nbnm_trial import summarize
        status_api=StatusAPI(output,port=int(spec['status_port']),snapshot_reader=lambda:summarize(output))
        write_json(output/'status-api.json',dict(url=status_api.url,read_only=True))

    def progress(data):
        last_status.update(data)
        last_status.update(PID=proc.pid,create_time=proc.create_time(),elapsed_seconds=time.monotonic()-begun,
            completed_decisions=completed,optimizer_updates=updates,cycle=cycle,
            budget=spec['decision_budget'],nbnm=True,formal_training=True)
        atomic_write_json(output/'progress.json',last_status)

    checkpoint_seconds = 0.

    def save(*, archive=False):
        nonlocal task_cursors, checkpoint_seconds
        started = time.monotonic()
        if collector is not None:task_cursors=[r.get('nb_nm_task_cursor') for r in collector.infos]
        payload=model.parameter_payload(dict(NBNM_decisions=completed,PPO_updates=updates))
        mark_parameters(payload, identity)
        parameter_file = output/'latest-parameters.pt'
        atomic_torch_save(payload, parameter_file)
        atomic_torch_save(dict(model_payload=payload,optimizer=optimizer.state_dict(),
            completed_decisions=completed,optimizer_updates=updates,cycle=cycle,in_progress=in_progress,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),spec=spec,
            worker_task_cursors=task_cursors,
            native_continuation=continuation,last_evaluation_decisions=last_evaluation_decisions,
            episode_index=episode_index,
            restart_note=('Learner, native input histories and carried SNN state preserved' if continuation
                else 'Legacy checkpoint has no native world capsule; first restart begins fresh games')),
            output/'latest-training.pt')
        if archive:
            archive_torch_parameters(parameter_file, output/f'cycle{cycle:03}-parameters.pt')
        checkpoint_seconds += time.monotonic() - started

    class StopAfterUpdate(Exception):pass
    try:
        torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed'])
        progress(dict(phase='loading_connectome_checkpoint'))
        resumed = load_cpu_checkpoint(a.resume) if a.resume else None
        if resumed is not None:
            restore_contract(resumed, identity)
        model,migrated=load_model(spec,a.resume,payload=resumed['model_payload'] if resumed is not None else None)
        optimizer=optimizer_for(model, trainable_groups=groups)
        rehearsal=None;retention=spec.get('rehearsal',{})
        route_refresh=None
        if spec.get('route_bank_refresh'):
            from fly_touhou.nbnm_demonstration_routes import RouteBankRefresh
            route_refresh=RouteBankRefresh(spec['route_bank_refresh'])
        if retention.get('coefficient',0)>0:
            from fly_touhou.nbnm_rehearsal import SuccessRehearsal
            rehearsal=SuccessRehearsal(retention['manifest'],scope_weights=retention.get('scope_weights'),
                additional_manifests=retention.get('additional_manifests',()),
                attack_spell_fraction=retention.get('attack_spell_fraction'),challenging_fraction=retention.get('challenging_fraction',0.),
                phase_priorities=retention.get('phase_priorities'),priority_fraction=retention.get('priority_fraction',.5),
                native_short_risk_coefficient=retention.get('native_short_risk_coefficient',0.),
                native_correction=retention.get('native_correction'))
            write_json(output/'rehearsal.json',rehearsal.summary())
        warmstart_state=spec.get('warmstart_optimizer_checkpoint')
        if not a.resume and warmstart_state:
            # New on-policy tasks start fresh games, but retain learned Adam
            # moments. Never replay the previous task's stored rollout here.
            previous_state=load_cpu_checkpoint(warmstart_state)
            restore_optimizer(optimizer,previous_state['optimizer'])
            old_objective = previous_state.get('spec', {}).get('nbnm', {}).get('objective', {})
            old_pace = old_objective.get('completion_pace_weight', 0.)
            new_pace = spec['nbnm']['objective'].get('completion_pace_weight', 0.)
            if bool(old_pace) != bool(new_pace):
                from fly_touhou.nbnm_resume import reset_critic_output
                reset_critic_output(model, optimizer, 1)
            warmstart_updates=int(previous_state['optimizer_updates'])
            del previous_state
        else:warmstart_updates=0
        if a.resume:
            state=resumed
            del resumed
            from fly_touhou.nbnm_resume import validate_resume
            if pending_patch:
                spec,deferred_objective=objective_for_saved_rollout(spec,state)
                model.nbnm=copy.deepcopy(spec['nbnm'])
                if deferred_objective:
                    from fly_touhou.nbnm_objective import Objective
                    validation['objective']=Objective(**spec['nbnm']['objective']).specification()
            if pending_guidance:
                spec,deferred_guidance=guidance_resume.for_saved_rollout(spec,state)
                if deferred_guidance:
                    from fly_touhou.nbnm_model import configure_model
                    configure_model(model,spec['nbnm'],reset_critics=False)
                    optimizer=optimizer_for(model, trainable_groups=groups)
                    retention=spec.get('rehearsal',{});rehearsal=guidance_resume.rehearsal_for(spec)
            migration=validate_resume(state,spec)
            optimizer_migration=restore_optimizer(optimizer,state['optimizer'])
            if migration.get('event_progress_added'):
                migration['critic_migration']=reset_event_critic(model,optimizer)
            if migration.get('completion_pace_target_changed'):
                from fly_touhou.nbnm_resume import reset_critic_output
                migration['pace_critic_migration']=reset_critic_output(model,optimizer,1)
            write_json(output/'resume-migration.json',dict(migration,optimizer=optimizer_migration))
            completed=state['completed_decisions'];updates=state['optimizer_updates'];cycle=state['cycle']
            in_progress=state['in_progress'];torch.set_rng_state(state['torch_rng'])
            task_cursors=state.get('worker_task_cursors')
            from fly_touhou.nbnm_continuation import checkpoint_continuation
            continuation=checkpoint_continuation(state)
            last_evaluation_decisions=state.get('last_evaluation_decisions',completed)
            if spec.get('periodic_evaluation') and not state['spec'].get('periodic_evaluation'):
                # A newly enabled interval starts at the resumed checkpoint,
                # not at decision zero of a much older run.
                last_evaluation_decisions=completed
            episode_index=state.get('episode_index',0)
            torch.cuda.set_rng_state_all(state['cuda_rng']);del state
            if pending_patch and not deferred_objective:
                record_objective_activation(a.config,spec,pending_patch,output,completed,updates,migration)
            if pending_guidance and not deferred_guidance:
                guidance_resume.record(a.config,spec,pending_guidance,output,completed,updates,migration)
        write_json(output/'configuration.json',spec)
        from fly_touhou.nbnm_numerics import REVISION as numerics_revision
        write_json(output/'launch.json',dict(validation,PID=proc.pid,create_time=proc.create_time(),
            numerics_revision=numerics_revision,
            device=torch.cuda.get_device_name(),critic_heads_reinitialized=migrated,
            training_updates=updates,network_training=True,topology_changed=False,
            formal_objective=spec['nbnm']['objective'],old_resource_reward_active=False,
            event_objective_deferred_until_old_rollout_complete=deferred_objective,
            native_guidance_deferred_until_old_rollout_complete=deferred_guidance,
            task_only_success_is_full_clear=False,resume=str(a.resume) if a.resume else None))
        if a.resume:
            resume_info=dict(source=str(a.resume),decisions=completed,
                optimizer_updates=updates,native_continuation_available=continuation is not None,
                live_worlds_restored=False,restored_world_count=0)
            write_json(output/'resume.json',resume_info)
        if warmstart_state and not a.resume:
            write_json(output/'warmstart.json',dict(parameter_checkpoint=spec['initial_checkpoint'],
                optimizer_checkpoint=warmstart_state,previous_optimizer_updates=warmstart_updates,
                previous_rollout_reused=False,new_task_sampling=True))
        save()
        # Native collection is delayed until a resumed learner segment finishes.
        while completed<spec['decision_budget'] or in_progress is not None:
            if (output/'STOP').exists():stopped=True;break
            if deferred_objective and in_progress is None:
                # Resumed old tapes finish first. No native collector has been
                # created yet, so all worlds start the new objective together.
                requested=with_pending_objective(spec,pending_patch)
                migration=validate_resume(dict(spec=spec,in_progress=None),requested)
                migration['critic_migration']=reset_event_critic(model,optimizer)
                spec=requested;model.nbnm=copy.deepcopy(spec['nbnm'])
                record_objective_activation(a.config,spec,pending_patch,output,completed,updates,migration)
                deferred_objective=False;save()
            if deferred_guidance and in_progress is None:
                requested=guidance_resume.apply(spec,pending_guidance)
                migration=validate_resume(dict(spec=spec,in_progress=None),requested)
                from fly_touhou.nbnm_model import configure_model
                spec=requested;configure_model(model,spec['nbnm'],reset_critics=False)
                old_optimizer=optimizer
                optimizer=optimizer_for(model, trainable_groups=groups)
                restore_optimizer(optimizer,old_optimizer.state_dict())
                del old_optimizer
                retention=spec.get('rehearsal',{});rehearsal=guidance_resume.rehearsal_for(spec)
                guidance_resume.record(a.config,spec,pending_guidance,output,completed,updates,migration)
                deferred_guidance=False;save()
            batch_started = time.monotonic()
            checkpoint_before_batch = checkpoint_seconds
            restore_seconds = tape_read_seconds = tape_write_seconds = export_seconds = 0.
            sampled_here = in_progress is None
            if sampled_here:
                if collector is None:
                    restore_started = time.monotonic()
                    folder=output/f'native-training-{int(time.time())}'
                    progress(dict(phase='initializing_native_workers'))
                    native_configuration=environment(spec)
                    if task_cursors:
                        for entry,cursor in zip(native_configuration['nbnm']['world_entries'],task_cursors):
                            if cursor is not None:entry['task_pool_offset']=cursor
                    collector=ReadyCollector(model,native_configuration,folder,workers=spec['workers'],
                        worker_cpus=cpus,ready_batch_size=7,ready_wait_ms=1.,per_world_quota=False,
                        movement_sampling_temperature=spec.get('movement_sampling_temperature',1.),
                        continuation=continuation)
                    if resume_info is not None and continuation is not None:
                        resume_info.update(live_worlds_restored=True,restored_world_count=collector.workers)
                        write_json(output/'resume.json',resume_info)
                    collector.worker_groups=spec['world_groups'];collector.group_weights=spec['group_weights']
                    from fly_touhou.nbnm_adaptive_course import attach
                    attach(collector,spec,output,continuation)
                    if getattr(collector,'adaptive_course',None):
                        write_json(output/'adaptive-course.json',collector.adaptive_course.summary())
                    episode_offset=0
                    restore_seconds = time.monotonic() - restore_started
                cycle+=1;progress(dict(phase='sampling'))
                if spec.get('cuda_graph_inference',False):
                    from fly_touhou.nbnm_cuda import InferenceGraphs
                    collector.inference_graphs=InferenceGraphs(model,
                        movement_temperature=spec.get('movement_sampling_temperature',1.),
                        packed_binary_messages=spec.get('packed_binary_messages',False))
                before=time.monotonic()
                rollout=collector.rollout(model,total_decisions=spec['segment_decisions'],progress=progress)
                if getattr(collector,'inference_graphs',None) is not None:
                    rollout['metadata']['graph_capture_seconds']=collector.inference_graphs.capture_seconds
                    collector.inference_graphs.clear();collector.inference_graphs=None
                sample_seconds=time.monotonic()-before
                completed+=rollout['metadata']['student_actions']
                from fly_touhou.nbnm_continuation import export_collector
                export_started = time.monotonic()
                continuation=export_collector(collector)
                export_seconds = time.monotonic() - export_started
                if getattr(collector,'adaptive_course',None):
                    course_summary=collector.adaptive_course.summary()
                    write_json(output/'adaptive-course.json',course_summary)
                    for task_key,detail in course_summary['tasks'].items():
                        for metric in ('smoothed_success_rate','target_local_decision_fraction','actual_decisions','forgetting'):
                            writer.add_scalar('curriculum/'+task_key+'/'+metric,detail[metric],completed)
                if getattr(collector,'adaptive_stage_chain',None):
                    write_json(output/'adaptive-stage-chain.json',collector.adaptive_stage_chain.summary())
                episode_path=collector.output/'training-episodes.jsonl'
                if episode_path.exists():
                    with episode_path.open(encoding='utf-8') as stream:
                        stream.seek(episode_offset)
                        for line in stream:
                            record=json.loads(line);episode_index+=1
                            kind=record['task']['mode']
                            for key in ('nbnm_success','task_native_frames','task_misses','task_Bombs'):
                                writer.add_scalar('train/'+kind+'/'+key,record[key],episode_index)
                            task=record['task']
                            task_name=('stage'+str(task['stage'])+'_'+task.get('attack',task['mode'])) if task['mode']!='full_story' else 'full_story'
                            for key in ('nbnm_success','task_native_frames','spell_captured','spell_survived_timeout'):
                                if key in record:writer.add_scalar('task/'+task_name+'/'+key,record[key],episode_index)
                        episode_offset=stream.tell()
                # Rolling on-policy tape enables a stopped learner to finish
                # this exact data; older tape arrays are replaced, logs remain.
                tape_started = time.monotonic()
                atomic_torch_save(rollout,output/'latest-rollout.pt')
                tape_write_seconds = time.monotonic() - tape_started
                in_progress=dict(rollout=str(output/'latest-rollout.pt'),logs=[],sample_seconds=sample_seconds,
                    collection=rollout['metadata'])
                save()
            else:
                tape_started = time.monotonic()
                rollout=load_cpu_checkpoint(in_progress['rollout'])
                tape_read_seconds = time.monotonic() - tape_started
            if rehearsal is not None:
                rehearsal.refresh()
                write_json(output/'rehearsal.json',rehearsal.summary())
                write_json(output/f'rehearsal-cycle{cycle:03}.json',rehearsal.manifest)
            if route_refresh is not None:
                updated_bank=route_refresh.refresh()
                if updated_bank:append_jsonl(output/'route-bank-refresh.jsonl',dict(cycle=cycle,**updated_bank))
            previous=len(in_progress['logs']);update_origin=updates-previous
            trained_batch_updates = in_progress['collection']['student_actions'] // 256
            before=time.monotonic()
            checkpoint_before_learning = checkpoint_seconds

            def learning_progress(row):
                nonlocal updates
                # update() also emits heartbeats; only completed Adam rows
                # have trained_student_rows and an integer update index.
                if 'trained_student_rows' in row:
                    updates=update_origin+row['update']
                    in_progress['logs'].append(row)
                    append_jsonl(output/'updates.jsonl',dict(row,cycle=cycle,total_optimizer_updates=updates))
                    for k,v in row.items():
                        if isinstance(v,(int,float)) and not isinstance(v,bool):writer.add_scalar('optimizer/'+k,v,updates)
                    if (output/'STOP').exists():raise StopAfterUpdate()
                    if updates%8==0 and row['update'] < trained_batch_updates:save()
                progress(dict(row,phase='learning'))

            deterministic_learner=spec.get('deterministic_learner',False)
            if deterministic_learner:
                old_determinism=torch.are_deterministic_algorithms_enabled()
                old_warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
                torch.use_deterministic_algorithms(True)
            try:
                result=update(model,optimizer,rollout,epochs=1,micro_scenes=16,seed=spec['seed']+cycle,
                    checkpoint_chunks=True,target_kl=None,max_updates=0,sample_decisions=0,
                    rule_archive_replay=rehearsal,rule_archive_coefficient=retention.get('coefficient',0.),
                    rule_archive_every=retention.get('every_updates',8),
                    rule_archive_task=retention.get('reserved_task'),
                    completed_update_prefix=tuple(in_progress['logs']),progress=learning_progress)
            finally:
                if deterministic_learner:
                    torch.use_deterministic_algorithms(old_determinism,warn_only=old_warn_only)
            learn_seconds=time.monotonic()-before
            row_checkpoint_seconds = checkpoint_seconds - checkpoint_before_learning
            row=dict(cycle=cycle,completed_decisions=completed,optimizer_updates=updates,
                sample_seconds=in_progress['sample_seconds'],learn_seconds=learn_seconds,
                collection=in_progress['collection'],learning=result)
            in_progress=None
            save(archive=True)
            row.update(pipeline_seconds=dict(world_restore=restore_seconds, native_export=export_seconds,
                rollout_read=tape_read_seconds, rollout_write=tape_write_seconds,
                checkpoint=checkpoint_seconds-checkpoint_before_batch, observed_wall=time.monotonic()-batch_started,
                sample_in_current_process=sampled_here),
                learn_without_checkpoint_seconds=learn_seconds - row_checkpoint_seconds)
            append_jsonl(output/'cycles.jsonl',row)
            for key in ('sample_seconds','learn_seconds'):
                writer.add_scalar('time/'+key,row[key],completed)
            writer.add_scalar('sampling/decisions_per_second',spec['segment_decisions']/row['sample_seconds'],completed)
            for key,value in row['collection'].get('task_group_decisions',{}).items():
                writer.add_scalar('sampling/task_'+key,value,completed)
            for key,value in row['collection'].get('task_decisions',{}).items():
                writer.add_scalar('exposure/'+key,value,completed)
            guidance=row['collection'].get('nbnm_route_guidance',{})
            event_progress=row['collection'].get('nbnm_event_progress',{})
            for key in ('events','unweighted_progress','weighted_reward'):
                if key in event_progress:writer.add_scalar('native_event_progress/'+key,event_progress[key],completed)
            for key,value in event_progress.get('by_kind',{}).items():
                writer.add_scalar('native_event_progress/events_'+key,value,completed)
            for key,value in guidance.items():
                if isinstance(value,(int,float)):writer.add_scalar('route_guidance/'+key,value,completed)
            for name,numerator,denominator in (
                ('teacher_top_agreement','teacher_top_matches_policy_top','guided_choice_rows'),
                ('teacher_top_student_probability','teacher_top_probability_sum','guided_choice_rows'),
                ('teacher_turn_mass','teacher_turn_mass_sum','teacher_turn_rows'),
                ('student_turn_mass','student_turn_mass_sum','teacher_turn_rows'),
                ('teacher_inward_mass','border_teacher_inward_mass_sum','border_exit_rows'),
                ('student_inward_mass','border_student_inward_mass_sum','border_exit_rows')):
                if guidance.get(denominator,0):
                    writer.add_scalar('route_guidance/'+name,guidance[numerator]/guidance[denominator],completed)
            writer.flush();del rollout;model.eval()
            from summarize_nbnm_trial import summarize
            write_json(output/'capability-progress.json',summarize(output))
            if (output/'STOP').exists():stopped=True;break
            periodic=spec.get('periodic_evaluation',{})
            due=bool(periodic) and completed<spec['decision_budget'] and (completed-last_evaluation_decisions>=periodic.get('decisions',65536)
                or time.monotonic()-last_evaluation_time>=periodic.get('seconds',7200))
            if due:
                # Workers and brain memories remain parked at their current
                # positions. Frozen games never become training samples.
                rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
                for key in periodic.get('tasks',['full_story']):
                    destination=output/f'development-{completed:09d}-{key}'
                    cursor=dict(decisions=completed,optimizer_updates=updates)
                    cached=read(destination/'report.json') if (destination/'report.json').exists() else None
                    if cached and cached.get('learner_cursor')==cursor:report=cached
                    else:
                        if destination.exists():destination.rename(output/(destination.name+'-interrupted-'+str(time.time_ns())))
                        report=evaluate(model,spec,destination,task=spec['tasks'][key],progress=progress)
                    report['learner_cursor']=dict(decisions=completed,optimizer_updates=updates)
                    write_json(destination/'report.json',report)
                    writer.add_scalar('frozen/'+key+'/success_rate',report['successes']/report['attempts'],completed)
                torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
                last_evaluation_decisions=completed;last_evaluation_time=time.monotonic()
                save();writer.flush();write_json(output/'capability-progress.json',summarize(output))
        if collector is not None:collector.close();collector=None
        progress(dict(phase='paused' if stopped else 'training_complete'))
        if not stopped and spec.get('evaluate_at_end',True):
            reports={}
            for group in spec.get('evaluation_tasks', ['primary']):
                task=spec['tasks'][group] if group!='primary' else spec['nbnm']['task']
                destination=output/('frozen-card-evaluation' if group=='primary' else 'frozen-'+group)
                # An interrupted final evaluation need not repeat already
                # completed tasks. Never reuse a changing-policy train report.
                cached=read(destination/'report.json') if (destination/'report.json').exists() else None
                same=cached and cached.get('learner_cursor')==dict(decisions=completed,optimizer_updates=updates)
                if same:report=cached
                else:
                    if destination.exists():
                        archived=output/(destination.name+'-interrupted-'+str(time.time_ns()))
                        if destination.resolve().parent!=output.resolve() or archived.resolve().parent!=output.resolve():
                            raise ValueError('Evaluation archive must remain inside its run directory')
                        destination.rename(archived)
                    report=evaluate(model,spec,destination,task=task,progress=progress)
                    report['learner_cursor']=dict(decisions=completed,optimizer_updates=updates)
                    write_json(destination/'report.json',report)
                reports[group]=report
                writer.add_scalar('frozen/'+group+'/success_rate',report['successes']/report['attempts'],completed)
                if report['successful_mean_native_frames'] is not None:
                    writer.add_scalar('frozen/'+group+'/successful_native_frames',report['successful_mean_native_frames'],completed)
            from fly_touhou.nbnm_run import frozen_story_clears
            full_clears=frozen_story_clears(reports)
            write_json(output/'final-report.json',dict(training_complete=True,report=report,reports=reports,
                full_game_NBNM_proven=bool(full_clears),frozen_full_game_NBNM_clears=full_clears))
            progress(dict(phase='complete',frozen_tasks={g:dict(successes=r['successes'],attempts=r['attempts']) for g,r in reports.items()}))
    except StopAfterUpdate:
        save();progress(dict(phase='paused',reason='operator STOP after committed optimizer update'))
    except BaseException:
        write_json(output/'failure.json',dict(error=traceback.format_exc(),cycle=cycle,
            completed_decisions=completed,optimizer_updates=updates))
        progress(dict(phase='failed'))
        raise
    finally:
        if collector is not None:collector.close()
        if status_api is not None:status_api.close()
        writer.close();cleanup_native_temp(temporary)


if __name__=='__main__':main()
