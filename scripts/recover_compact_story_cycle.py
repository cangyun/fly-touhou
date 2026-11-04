"""Finish a committed on-policy tape after a checkpoint-write failure.

Restore the preceding complete model+Adam, reconstruct the original critical
replay pool and minibatch order, then run the scheduled frozen development.
No new training actions are collected and no brain topology is changed.
"""
from pathlib import Path
import argparse, gc, hashlib, json, os, sys, time, traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-job', 'resume-training', 'connectome', 'output', 'reference-report'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cycle', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(dict(status='prepared',new_training_actions=0,cycle=args.cycle))
        return
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json, append_jsonl
    from fly_touhou.owned_worker_affinity import worker_cpu_pool
    source = read(args.source_job / 'launch.json')
    try:
        old = psutil.Process(source['PID'])
        if abs(old.create_time() - source['create_time']) < .01:
            raise RuntimeError('Source executor is still alive')
    except psutil.NoSuchProcess:
        pass
    cfg = read(args.source_job / 'environment-config.json')
    if any((p.info['name'] or '').lower() == Path(cfg['game_exe']).name.lower()
           for p in psutil.process_iter(['name'])):
        raise RuntimeError('Another original game is active')
    process = psutil.Process()
    cpus = worker_cpu_pool(process.cpu_affinity())
    process.cpu_affinity(process.cpu_affinity()[-2:])
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        FLY_TOUHOU_FUSED_EDGE_GRAD='1',
        FLY_TOUHOU_EDGE_GRAD_CACHE=str(args.source_job / 'kernel-cache'),
        FLY_TOUHOU_DETERMINISTIC_SPMM='1',
        FLY_TOUHOU_SPMM_CACHE=str(args.source_job / 'spmm-kernel-cache'),
        FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-recovered-cycle-' + str(process.pid))
    args.output.mkdir(parents=True, exist_ok=False)
    launch = dict(schema='candidate-SNN-full-story-PPO-recovery-v1', PID=process.pid,
        create_time=process.create_time(), started_UTC=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv, source_job=str(args.source_job), resume_training=str(args.resume_training),
        recovery_cycle=args.cycle, worlds=0, development_worlds=4, new_training_actions=0,
        CUDA_started=False, teacher_actions=0, full_story=True,
        native_training_worlds_lost_on_source_exit=True, topology_changed=False)
    write_json(args.output / 'launch.json', launch)
    write_json(args.output / 'environment-config.json', cfg)
    write_json(args.output / 'algorithm.json', dict(read(args.source_job / 'algorithm.json'),
        recovery='Original committed tape and complete model+Adam;no new training collection'))
    begun = time.monotonic()
    updates = 0
    origin = 0
    decisions = 0

    def progress(phase, **more):
        write_json(args.output / 'progress.json', dict(phase=phase,cycle=args.cycle,
            PPO_updates=updates, joint_updates=updates, new_student_actions=decisions,
            student_actions_this_job=0, PPO_updates_this_job=updates-origin,
            elapsed_seconds=time.monotonic()-begun, **more))

    def guard():
        if (args.output / 'STOP').exists():
            raise RuntimeError('Operator STOP')
        if time.monotonic()-begun > 6*3600:
            raise TimeoutError('Recovery six-hour wall budget')

    try:
        import torch
        from fly_touhou.compact_story_model import from_story_checkpoint
        from fly_touhou.compact_story_ppo import optimizer_for, restore_optimizer, update
        from fly_touhou.compact_story_critical_replay import CriticalTeacherReplay
        from fly_touhou.compact_story_learning_rate import PolicyLearningRate
        from fly_touhou.compact_full_story import evaluate
        from fly_touhou.compact_native_budget import compare
        from fly_touhou.thrl_job_paths import configure_native_temp
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        configure_native_temp(args.output)
        restored = torch.load(args.resume_training,map_location='cpu',weights_only=False)
        costs = dict(restored['model_payload']['costs'])
        origin = updates = int(costs['PPO_updates'])
        model = from_story_checkpoint(args.connectome,restored['model_payload'])
        optimizer = optimizer_for(model)
        launch['optimizer_restore'] = restore_optimizer(optimizer,restored['optimizer'])
        torch.set_rng_state(restored['torch_RNG'])
        torch.cuda.set_rng_state_all(restored['CUDA_RNG'])
        movement_temperature = restored.get('movement_sampling_temperature',1.)
        bomb_temperature = restored.get('bomb_sampling_temperature',1.)
        del restored
        # The rolling checkpoint is written before the end-of-cycle scheduler.
        # Reapply its committed result so the recovered tape uses the same LR.
        lr_rows=[json.loads(line) for line in (args.source_job/'learning-rate.jsonl').read_text().splitlines() if line.strip()]
        preceding=[row for row in lr_rows if row['cycle']<args.cycle]
        scale=preceding[-1]['policy_lr_scale_after'] if preceding else None
        learning_rate = PolicyLearningRate(optimizer,scale=scale,adaptive=source['adaptive_policy_lr'])
        partial_updates=int(read(args.source_job/'failure.json')['PPO_updates'])-origin
        critical = CriticalTeacherReplay()
        replay_source = read(args.source_job / 'critical-teacher-replay.json')
        progress('reconstructing_committed_replay')
        for path in replay_source['source_tapes']:
            guard()
            tape = torch.load(path,map_location='cpu',weights_only=False)
            critical.add(tape,source=path)
            del tape
            gc.collect()
        if critical.summary() != replay_source:
            raise ValueError('Reconstructed critical replay differs from original cycle')
        write_json(args.output / 'critical-teacher-replay.json',critical.summary())
        tape_path = Path(read(args.source_job / f'rollout-{args.cycle:03}-path.json')['path'])
        rollout = torch.load(tape_path,map_location='cpu',weights_only=False)
        reused_decisions = int(rollout['actions'].numel())
        decisions = int(costs['new_student_actions']) + reused_decisions
        write_json(args.output / 'recovery-evidence.json',dict(source_model_costs=costs,
            committed_tape=str(tape_path),reused_decisions=reused_decisions,new_training_actions=0,
            recovered_cycle=args.cycle,source_partial_updates=partial_updates,partial_updates_replayed_not_added_to_lineage=True,
            brain_topology_changed=False,optimizer_and_model_restored_together=True))
        launch.update(CUDA_started=True,CUDA_device=torch.cuda.get_device_name(),
            policy_lr_scale=learning_rate.scale,source_PPO_updates=origin,
            source_student_actions=costs['new_student_actions'])
        write_json(args.output / 'launch.json',launch)

        def save(tag):
            payload = model.parameter_payload(dict(costs,PPO_updates=updates,new_student_actions=decisions))
            model_path = args.output / (tag+'-parameters.pt')
            training_path = args.output / (tag+'-training.pt')
            for path, data in ((model_path,payload),(training_path,dict(model_payload=payload,
                    optimizer=optimizer.state_dict(),torch_RNG=torch.get_rng_state(),
                    CUDA_RNG=torch.cuda.get_rng_state_all(),movement_sampling_temperature=movement_temperature,
                    bomb_sampling_temperature=bomb_temperature))):
                temp = path.with_suffix('.pt.tmp')
                with temp.open('wb') as stream:
                    torch.save(data,stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp,path)
            return model_path,training_path

        def learned(info):
            nonlocal updates
            updates = origin + info['update']
            row = dict(info,step=updates,cycle=args.cycle,new_student_actions=decisions,
                student_actions_this_job=0,PPO_updates_this_job=info['update'])
            append_jsonl(args.output / 'learning.jsonl',row)
            progress('full_story_PPO_update',learning=row)
            if info['update']==partial_updates:
                expected=torch.load(args.source_job/'latest-parameters.pt',map_location='cpu',weights_only=False)
                errors={}
                for k,v in model.state_dict().items():
                    if not v.numel():continue
                    actual=v.detach().cpu();reference=expected['state_dict'][k]
                    errors[k]=0. if torch.equal(actual,reference) else float((actual.float()-reference.float()).abs().max())
                max_error=max(errors.values(),default=0.)
                write_json(args.output/'interrupted-prefix-reproduction.json',dict(max_parameter_error=max_error,
                    tensor_errors=errors,expected_costs=expected['costs'],replayed_updates=partial_updates))
                del expected
                if max_error>1e-6:
                    raise ValueError('Interrupted update prefix did not reproduce: '+str(max_error))
            if info['update']%4==0:
                save('latest')

        result=update(model,optimizer,rollout,epochs=1,micro_scenes=16,
            seed=int(cfg['seed'])+args.cycle,guard=guard,progress=learned,critical_replay=critical,
            world_aligned=True,checkpoint_chunks=True)
        schedule=learning_rate.after_rollout(result['logs'])
        append_jsonl(args.output/'learning-rate.jsonl',dict(schedule,cycle=args.cycle,PPO_updates=updates))
        parameters,training=save(f'cycle{args.cycle:03}')
        del rollout,critical
        gc.collect()
        torch.cuda.empty_cache()
        model.eval()
        report=evaluate(model,cfg,args.output/f'development-cycle{args.cycle:03}',episodes=4,
            worker_cpus=cpus,guard=guard,progress=lambda x:progress('frozen_full_story_development',evaluation=x),
            action_selection='family_mode')
        comparison=compare(report,read(args.reference_report))
        write_json(args.output/'comparison.json',dict(comparison,reference_report=str(args.reference_report),
            candidate_parameters=str(parameters),independent_formal_test=False))
        write_json(args.output/'result.json',dict(status='completed_recovered_cycle_and_frozen_development',
            parameters=str(parameters),training=str(training),PPO_updates=updates,PPO_updates_this_job=updates-origin,
            student_actions_this_job=0,reused_student_decisions=reused_decisions,new_student_actions=decisions,
            clears=report['clears'],comparison=comparison,goal_complete_claimed=False))
        progress('completed_recovered_cycle_and_frozen_development')
    except BaseException as error:
        write_json(args.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            PPO_updates=updates,new_training_actions=0,no_automatic_retry=True))
        raise


if __name__=='__main__':
    main()
