"""Prepare or explicitly execute the shared-connectome NBNM experiment.

No training runs without --execute. --evaluate-only uses a frozen checkpoint.
STOP finishes the current acknowledged segment/update and saves its cursor.
"""
from pathlib import Path
import argparse,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',type=Path)
    p.add_argument('--evaluate-only',type=Path)
    p.add_argument('--evaluation-output',type=Path)
    p.add_argument('--log-file',type=Path,help='Capture startup/import errors as well as training output')
    a=p.parse_args()
    if a.log_file:
        a.log_file.parent.mkdir(parents=True,exist_ok=True)
        sys.stdout=sys.stderr=a.log_file.open('a',encoding='utf-8',buffering=1)
        import faulthandler
        faulthandler.enable(file=sys.stderr)
        print('NBNM process started: PID='+str(os.getpid()),flush=True)
    from fly_touhou.nbnm_run import read,validate_spec,runtime,environment,load_model,atomic_torch_save,evaluate
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    spec=read(a.config);validation=validate_spec(spec)
    if not a.execute and a.evaluate_only is None:
        from torch.utils.tensorboard import SummaryWriter
        print(json.dumps(validation,ensure_ascii=False,indent=2));return
    if a.evaluate_only:
        if a.evaluation_output is None:p.error('--evaluation-output required')
        runtime(a.evaluation_output)
        model,_=load_model(spec,a.evaluate_only)
        evaluate(model,spec,a.evaluation_output)
        return
    output=Path(spec['output']);output.mkdir(parents=True,exist_ok=a.resume is not None)
    temporary=runtime(output)
    import torch
    import psutil
    from torch.utils.tensorboard import SummaryWriter
    from fly_touhou.compact_story_ppo import optimizer_for,update
    from fly_touhou.compact_story_ready import ReadyCollector
    from fly_touhou.owned_worker_affinity import worker_cpu_pool
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    proc=psutil.Process();cpus=worker_cpu_pool(proc.cpu_affinity())
    proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    begun=time.monotonic();collector=None;writer=SummaryWriter(spec.get('tensorboard_directory',str(output/'tensorboard')))
    completed=0;updates=0;cycle=0;in_progress=None;model=None;optimizer=None
    last_status={};stopped=False;episode_offset=0;episode_index=0

    def progress(data):
        last_status.update(data)
        last_status.update(PID=proc.pid,create_time=proc.create_time(),elapsed_seconds=time.monotonic()-begun,
            completed_decisions=completed,optimizer_updates=updates,cycle=cycle,
            budget=spec['decision_budget'],nbnm=True,formal_training=True)
        write_json(output/'progress.json',last_status)

    def save():
        payload=model.parameter_payload(dict(NBNM_decisions=completed,PPO_updates=updates))
        atomic_torch_save(payload,output/'latest-parameters.pt')
        atomic_torch_save(dict(model_payload=payload,optimizer=optimizer.state_dict(),
            completed_decisions=completed,optimizer_updates=updates,cycle=cycle,in_progress=in_progress,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),spec=spec,
            restart_note='Learner cursor preserved; live native worlds restart only after process restart'),
            output/'latest-training.pt')

    class StopAfterUpdate(Exception):pass
    try:
        torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed'])
        model,migrated=load_model(spec,a.resume)
        optimizer=optimizer_for(model)
        if a.resume:
            state=torch.load(a.resume,map_location='cpu',weights_only=False)
            for field in ('nbnm','tasks','world_groups','segment_decisions'):
                if state['spec'][field]!=spec[field]:raise ValueError('Resume requires the same '+field)
            optimizer.load_state_dict(state['optimizer'])
            completed=state['completed_decisions'];updates=state['optimizer_updates'];cycle=state['cycle']
            in_progress=state['in_progress'];torch.set_rng_state(state['torch_rng'])
            torch.cuda.set_rng_state_all(state['cuda_rng']);del state
        write_json(output/'configuration.json',spec)
        write_json(output/'launch.json',dict(validation,PID=proc.pid,create_time=proc.create_time(),
            device=torch.cuda.get_device_name(),critic_heads_reinitialized=migrated,
            training_updates=updates,network_training=True,topology_changed=False,
            formal_objective=spec['nbnm']['objective'],old_resource_reward_active=False,
            task_only_success_is_full_clear=False,resume=str(a.resume) if a.resume else None))
        save()
        # Native collection is delayed until a resumed learner segment finishes.
        while completed<spec['decision_budget'] or in_progress is not None:
            if (output/'STOP').exists():stopped=True;break
            if in_progress is None:
                if collector is None:
                    folder=output/f'native-training-{int(time.time())}'
                    progress(dict(phase='initializing_native_workers'))
                    collector=ReadyCollector(model,environment(spec),folder,workers=spec['workers'],
                        worker_cpus=cpus,ready_batch_size=7,ready_wait_ms=1.,per_world_quota=False,
                        movement_sampling_temperature=spec.get('movement_sampling_temperature',1.))
                    collector.worker_groups=spec['world_groups'];collector.group_weights=spec['group_weights']
                    episode_offset=0
                cycle+=1;progress(dict(phase='sampling'))
                if spec.get('cuda_graph_inference',False):
                    from fly_touhou.nbnm_cuda import InferenceGraphs
                    collector.inference_graphs=InferenceGraphs(model,
                        movement_temperature=spec.get('movement_sampling_temperature',1.))
                before=time.monotonic()
                rollout=collector.rollout(model,total_decisions=spec['segment_decisions'],progress=progress)
                if getattr(collector,'inference_graphs',None) is not None:
                    rollout['metadata']['graph_capture_seconds']=collector.inference_graphs.capture_seconds
                    collector.inference_graphs.clear();collector.inference_graphs=None
                sample_seconds=time.monotonic()-before
                completed+=rollout['metadata']['student_actions']
                episode_path=collector.output/'training-episodes.jsonl'
                if episode_path.exists():
                    with episode_path.open(encoding='utf-8') as stream:
                        stream.seek(episode_offset)
                        for line in stream:
                            record=json.loads(line);episode_index+=1
                            kind=record['task']['mode']
                            for key in ('nbnm_success','task_native_frames','task_misses','task_Bombs'):
                                writer.add_scalar('train/'+kind+'/'+key,record[key],episode_index)
                        episode_offset=stream.tell()
                # Rolling on-policy tape enables a stopped learner to finish
                # this exact data; older tape arrays are replaced, logs remain.
                atomic_torch_save(rollout,output/'latest-rollout.pt')
                in_progress=dict(rollout=str(output/'latest-rollout.pt'),logs=[],sample_seconds=sample_seconds,
                    collection=rollout['metadata'])
                save()
            else:
                rollout=torch.load(in_progress['rollout'],map_location='cpu',weights_only=False)
            previous=len(in_progress['logs']);update_origin=updates-previous
            before=time.monotonic()

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
                    if updates%8==0 or (output/'STOP').exists():save()
                    if (output/'STOP').exists():raise StopAfterUpdate()
                progress(dict(row,phase='learning'))

            result=update(model,optimizer,rollout,epochs=1,micro_scenes=16,seed=spec['seed']+cycle,
                checkpoint_chunks=True,target_kl=None,max_updates=0,sample_decisions=0,
                completed_update_prefix=tuple(in_progress['logs']),progress=learning_progress)
            learn_seconds=time.monotonic()-before
            row=dict(cycle=cycle,completed_decisions=completed,optimizer_updates=updates,
                sample_seconds=in_progress['sample_seconds'],learn_seconds=learn_seconds,
                collection=in_progress['collection'],learning=result)
            append_jsonl(output/'cycles.jsonl',row)
            for key in ('sample_seconds','learn_seconds'):
                writer.add_scalar('time/'+key,row[key],completed)
            writer.add_scalar('sampling/decisions_per_second',spec['segment_decisions']/row['sample_seconds'],completed)
            for key,value in row['collection'].get('task_group_decisions',{}).items():
                writer.add_scalar('sampling/task_'+key,value,completed)
            in_progress=None;save()
            atomic_torch_save(model.parameter_payload(dict(NBNM_decisions=completed,PPO_updates=updates)),
                output/f'cycle{cycle:03}-parameters.pt')
            writer.flush();del rollout;model.eval()
            if (output/'STOP').exists():stopped=True;break
        if collector is not None:collector.close();collector=None
        progress(dict(phase='paused' if stopped else 'training_complete'))
        if not stopped and spec.get('evaluate_at_end',True):
            report=evaluate(model,spec,output/'frozen-card-evaluation',progress=progress)
            writer.add_scalar('frozen/task_success_rate',report['successes']/report['attempts'],completed)
            if report['successful_mean_native_frames'] is not None:
                writer.add_scalar('frozen/successful_mean_native_frames',report['successful_mean_native_frames'],completed)
            write_json(output/'final-report.json',dict(training_complete=True,report=report,
                full_game_NBNM_proven=False))
            progress(dict(phase='complete',frozen_successes=report['successes'],frozen_attempts=report['attempts']))
    except StopAfterUpdate:
        save();progress(dict(phase='paused',reason='operator STOP after committed optimizer update'))
    except BaseException:
        write_json(output/'failure.json',dict(error=traceback.format_exc(),cycle=cycle,
            completed_decisions=completed,optimizer_updates=updates))
        progress(dict(phase='failed'))
        raise
    finally:
        if collector is not None:collector.close()
        writer.close();cleanup_native_temp(temporary)


if __name__=='__main__':main()
