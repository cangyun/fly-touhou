"""Finish a saved partial PPO cycle, then hand off to the 16-world trainer."""
from pathlib import Path
import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def restore_late_replay(payload,model,revisions):
    if payload is None:return None
    from fly_touhou.compact_late_stage_replay import LateStageReplay,SENSOR_FLAGS
    replay=LateStageReplay(sensor_flags={key:getattr(model,key,False) for key in SENSOR_FLAGS},
        native_repair_revisions=revisions)
    replay.merge_payload(payload)
    return replay


def committed_prefix(records,cycle,origin,completed_cycle):
    prefix=[row for row in records if row['cycle']==cycle and row['step']<=origin]
    if prefix:
        if prefix[-1]['step']!=origin:raise ValueError('Checkpoint must match its committed learning prefix')
    elif completed_cycle!=cycle-1:
        raise ValueError('An empty prefix requires the preceding cycle to be complete')
    return prefix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-control','output','continuation-spec'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--training-checkpoint',type=Path)
    p.add_argument('--batch-invariant-sensory',action='store_true')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if not args.execute:return
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    spec=read(args.source_control/'job-spec.json');argv=spec['arguments']
    def option(name,default=None):
        return argv[argv.index(name)+1] if name in argv else default
    source_job=Path(option('--output'));source=read(source_job/'launch.json')
    old_checkpoint=args.training_checkpoint or Path(read(args.source_control/'next-resume-settings.json')['training_checkpoint'])
    cycle=int(read(source_job/'progress.json')['cycle'])
    process=psutil.Process()
    continuation_cpu_affinity=list(spec.get('cpu_affinity',process.cpu_affinity()))
    process.cpu_affinity(process.cpu_affinity()[-2:])
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    kernel_cache=Path(source.get('reuse_kernel_cache_from') or source_job)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_FUSED_EDGE_GRAD='1',
        FLY_TOUHOU_EDGE_GRAD_CACHE=str(kernel_cache/'kernel-cache'),
        FLY_TOUHOU_DETERMINISTIC_SPMM='1',FLY_TOUHOU_SPMM_CACHE=str(kernel_cache/'spmm-kernel-cache'))
    args.output.mkdir(parents=True,exist_ok=False)
    launch=dict(source,PID=process.pid,create_time=process.create_time(),
        started_UTC=datetime.now(timezone.utc).isoformat(),argv=sys.argv,worlds=0,development_worlds=0,
        source_job=str(source_job),resume_training=str(old_checkpoint),CUDA_started=False,
        continuation_cpu_affinity=continuation_cpu_affinity,
        execution_mode='resume_committed_partial_cycle',teacher_actions=0,
        new_training_actions=0,native_world_positions_lost_on_source_exit=True)
    write_json(args.output/'launch.json',launch)
    write_json(args.output/'environment-config.json',read(source_job/'environment-config.json'))
    write_json(args.output/'algorithm.json',dict(read(source_job/'algorithm.json'),
        recovery='Retain saved model and Adam; skip committed minibatches; finish original tape',KL_early_stopping_enabled=False))
    from train_compact_story_ppo import register_tensorboard_run
    register_tensorboard_run(args.output)
    started=time.monotonic();updates=0;origin=0;decisions=0;performed=0
    def progress(phase,**extra):
        write_json(args.output/'progress.json',dict(phase=phase,cycle=cycle,PPO_updates=updates,joint_updates=updates,
            new_student_actions=decisions,student_actions_this_job=0,PPO_updates_this_job=updates-origin,
            elapsed_seconds=time.monotonic()-started,**extra))
    def guard():
        if (args.output/'STOP').exists():raise RuntimeError('Operator STOP')
    save=None
    try:
        import torch
        from fly_touhou.compact_story_model import from_story_checkpoint
        from fly_touhou.compact_story_ppo import optimizer_for,restore_optimizer,update
        from fly_touhou.compact_story_learning_rate import PolicyLearningRate
        from fly_touhou.compact_native_rule_replay import NativeRuleTeacherReplay
        from fly_touhou.compact_story_tape_retention import save_replays
        from train_compact_story_ppo import native_teacher_stage_weight_map
        torch.set_num_threads(2)
        torch.manual_seed(int(option('--seed')));torch.cuda.manual_seed_all(int(option('--seed')))
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        progress('loading_saved_partial_update')
        restored=torch.load(old_checkpoint,map_location='cpu',weights_only=False)
        costs=dict(restored['model_payload']['costs']);updates=origin=int(costs['PPO_updates'])
        decisions=int(costs['new_student_actions'])
        learning_file=source_job/'learning.jsonl'
        records=[json.loads(line) for line in learning_file.read_text(encoding='utf-8').splitlines() if line.strip()] if learning_file.exists() else []
        prefix=committed_prefix(records,cycle,origin,restored['completed_cycle'])
        base_updates=origin-len(prefix)
        model=from_story_checkpoint(Path(option('--connectome')),restored['model_payload'])
        if source.get('unpad_candidates'):model.pad_candidate_batch=False
        if source.get('deterministic_spmm'):model.core.sparse.deterministic_spmm=True
        if args.batch_invariant_sensory:model.core.input_encoder.batch_invariant=True
        optimizer=optimizer_for(model);launch['optimizer_restore']=restore_optimizer(optimizer,restored['optimizer'])
        torch.set_rng_state(restored['torch_RNG']);torch.cuda.set_rng_state_all(restored['CUDA_RNG'])
        learning_rate=PolicyLearningRate(optimizer,adaptive=False,kl_early_stop=False,target_kl=None)
        archive=torch.load(restored['student_replay_archive'],map_location='cpu',weights_only=False)
        critical=archive.get('critical_replay')
        late=restore_late_replay(archive.get('late_replay'),model,restored['native_repair_revisions'])
        weights=native_teacher_stage_weight_map(restored['native_rule_teacher_manifest'],restored.get('native_rule_teacher_stage_weights'))
        native_rule=NativeRuleTeacherReplay(restored['native_rule_teacher_manifest'],stage_weights=weights)
        # The per-stage teacher quotas persisted across completed cycles in
        # the original process. Restore those counts before burning this prefix.
        for row in records:
            if row['cycle']>=cycle:continue
            for stage,count in row.get('native_rule_stage_windows',{}).items():
                native_rule.stage_draws[int(stage)]+=int(count)
        launch.update(CUDA_started=True,CUDA_device=torch.cuda.get_device_name(),
            batch_invariant_sensory=model.core.input_encoder.batch_invariant,
            committed_updates_in_cycle=len(prefix),source_PPO_updates=origin,
            movement_sampling_temperature=restored['movement_sampling_temperature'],
            bomb_sampling_temperature=restored['bomb_sampling_temperature'])
        write_json(args.output/'launch.json',launch)
        progress('loading_committed_rollout',resumed_completed_updates=len(prefix))
        tape=Path(read(source_job/f'rollout-{cycle:03}-path.json')['path'])
        rollout=torch.load(tape,map_location='cpu',weights_only=False)
        write_json(args.output/f'rollout-{cycle:03}-path.json',dict(path=str(tape)))
        write_json(args.output/'recovery-evidence.json',dict(source_checkpoint=str(old_checkpoint),
            committed_tape=str(tape),preserved_PPO_updates=origin,committed_updates_in_cycle=len(prefix),
            original_collection_decisions=rollout['metadata']['student_actions'],new_training_actions=0,
            completed_gradients_applied_twice=False,brain_topology_changed=False,
            model_and_Adam_restored_together=True,KL_stopping=False,KL_threshold=None))
        inherited={k:v for k,v in restored.items() if k not in ('model_payload','optimizer','torch_RNG','CUDA_RNG')}
        del restored;gc.collect()
        archive_path=str(args.output/f'student-supervision-{cycle:03}.pt')
        def save(tag,completed=False):
            payload=model.parameter_payload(dict(costs,PPO_updates=updates,new_student_actions=decisions))
            state=dict(inherited,model_payload=payload,optimizer=optimizer.state_dict(),
                torch_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state_all())
            if completed:state.update(completed_cycle=cycle,student_replay_archive=archive_path)
            paths=[]
            for suffix,value in (('parameters',payload),('training',state)):
                path=args.output/f'{tag}-{suffix}.pt';temporary=path.with_suffix('.pt.tmp')
                with temporary.open('wb') as stream:
                    torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
                os.replace(temporary,path);paths.append(path)
            return paths
        for row in prefix:append_jsonl(args.output/'learning.jsonl',row)
        def learned(info):
            nonlocal updates,performed
            updates=base_updates+info['update'];performed+=1
            row=dict(info,step=updates,cycle=cycle,new_student_actions=decisions,
                student_actions_this_job=0,PPO_updates_this_job=updates-origin,resumed_completed_updates=len(prefix))
            append_jsonl(args.output/'learning.jsonl',row)
            progress('full_story_PPO_update',learning=row)
            old=next((x for x in records if x['step']==updates),None)
            if old is not None:
                fields=('loss','gradient_norm','PPO','guidance','risk','exact_action_KL')
                append_jsonl(args.output/'replayed-unsaved-updates.jsonl',dict(step=updates,
                    metric_absolute_errors={k:abs(row[k]-old[k]) for k in fields}))
            if updates%4==0:save('latest')
        selection=source['PPO_sample_selection']
        progress('resuming_committed_PPO_updates',resumed_completed_updates=len(prefix))
        result=update(model,optimizer,rollout,epochs=int(option('--epochs','1')),
            micro_scenes=int(option('--micro-scenes','16')),seed=int(option('--seed'))+cycle,
            guard=guard,progress=learned,critical_replay=critical,world_aligned=source['world_aligned_updates'],
            checkpoint_chunks=source['checkpoint_candidate_chunks'],
            boss_attack_coefficient=inherited.get('boss_attack_guidance_coefficient',0.),
            conditional_wall_coefficient=inherited.get('conditional_wall_guidance_coefficient',0.),
            guided_wall_coefficient=inherited.get('guided_wall_guidance_coefficient',0.),
            selective_attack_coefficient=inherited.get('selective_attack_guidance_coefficient',0.),
            late_replay=late,late_replay_coefficient=inherited.get('late_stage_replay_coefficient',0.),
            native_rule_replay=native_rule,native_rule_coefficient=inherited['native_rule_teacher_coefficient'],
            native_rule_every=inherited['native_rule_teacher_every'],native_rule_windows=inherited['native_rule_teacher_windows'],
            max_updates=int(option('--max-adam-updates','0')),sample_decisions=selection['rows'],
            stage_weights={int(k):v for k,v in selection['stage_weights'].items()},target_kl=None,
            completed_update_prefix=prefix)
        schedule=learning_rate.after_rollout(result['logs'])
        append_jsonl(args.output/'learning-rate.jsonl',dict(schedule,cycle=cycle,PPO_updates=updates))
        save_replays(archive_path,critical,late,inherited['native_repair_revisions'])
        parameters,training=save(f'cycle{cycle:03}',completed=True)
        write_json(args.output/f'update-{cycle:03}.json',{k:v for k,v in result.items() if k!='logs'})
        write_json(args.output/'training_result.json',dict(status='recovered_original_cycle',cycle=cycle,
            parameters=str(parameters),training=str(training),PPO_updates=updates,
            preserved_updates=origin,additional_updates=performed,new_training_actions=0,
            source_decisions_preserved=decisions,continuation_spec=str(args.continuation_spec)))
        progress('completed_recovered_cycle')
        del model,optimizer,learning_rate,critical,late,native_rule,rollout,archive
        gc.collect();torch.cuda.empty_cache()
        guard()
        # Existing detached launcher keeps the next job in the same hidden
        # native execution mode; no desktop input or foreground activation.
        continuation_spec=read(args.continuation_spec)
        continuation_spec.setdefault('cpu_affinity',continuation_cpu_affinity)
        write_json(args.continuation_spec,continuation_spec)
        launched=subprocess.run([sys.executable,str(ROOT/'scripts/background_job.py'),
            '--spec',str(args.continuation_spec)],cwd=ROOT,capture_output=True,text=True,check=True)
        write_json(args.output/'continuation-launch.json',json.loads(launched.stdout))
    except BaseException as error:
        emergency=None
        if save is not None:
            try:emergency=[str(v) for v in save('interrupted')]
            except Exception:pass
        write_json(args.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            PPO_updates=updates,additional_updates=performed,new_training_actions=0,
            interrupted_checkpoint=emergency,no_automatic_retry=True))
        raise


if __name__=='__main__':main()
