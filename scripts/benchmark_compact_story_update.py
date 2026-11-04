"""One isolated16-world update: inference/replay fidelity,VRAM and wall time.

Uses fixed historical native inputs/actions for computation diagnostics only.
No games run,no new on-policy samples,no resulting weights are deployed.
"""
from pathlib import Path
import argparse,os,sys,time,traceback,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','rollout','critical-rollout','connectome','previous-job','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--padded-candidates',action='store_true')
    p.add_argument('--phase-valid-control',action='store_true')
    p.add_argument('--deterministic-spmm',action='store_true')
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:print(dict(status='prepared',CUDA_started=False));return
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json
    old=read(a.previous_job/'launch.json')
    try:
        q=psutil.Process(old['PID'])
        if abs(q.create_time()-old['create_time'])<.001:raise RuntimeError('Previous CUDA job still active')
    except psutil.NoSuchProcess:pass
    q=psutil.Process();q.cpu_affinity(q.cpu_affinity()[-2:]);q.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        FLY_TOUHOU_FUSED_EDGE_GRAD='1',FLY_TOUHOU_EDGE_GRAD_CACHE=str(a.output/'kernel-cache'))
    if a.deterministic_spmm:
        os.environ['FLY_TOUHOU_DETERMINISTIC_SPMM']='1'
        os.environ['FLY_TOUHOU_SPMM_CACHE']=str(a.output/'spmm-kernel-cache')
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'launch.json',dict(PID=q.pid,create_time=q.create_time(),argv=sys.argv,schema='isolated-world-aligned-SNN-update-compute-v1'))
    try:
        import torch
        from fly_touhou.compact_story_model import from_story_checkpoint,prime_critical_event_ports
        from fly_touhou.compact_story_ppo import optimizer_for,update
        from fly_touhou.compact_story_critical_replay import CriticalTeacherReplay
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=from_story_checkpoint(a.connectome,torch.load(a.checkpoint,map_location='cpu',weights_only=False)).eval()
        model.pad_candidate_batch=a.padded_candidates;event=prime_critical_event_ports(model,amplitude=.25)
        if a.deterministic_spmm:model.core.sparse.deterministic_spmm=True
        model.phase_valid_control=a.phase_valid_control
        original=torch.load(a.rollout,map_location='cpu',weights_only=False,mmap=True)
        T=16;N=original['features'].shape[1]
        if N!=16:raise ValueError('16 real original world slots required')
        data={k:v[:T] for k,v in original.items() if torch.is_tensor(v) and k!='bootstrap_values'}
        canonicalized=0
        if model.phase_valid_control:
            control=(data['features'][...,14]>.5)|(data['features'][...,17]>.5)
            canonicalized=int((~control&(data['actions']!=0)).sum())
            data['actions']=torch.where(control,data['actions'],torch.zeros_like(data['actions']))
            data['rule_weights']=data['rule_weights']*control.to(data['rule_weights'].dtype)
        data['block_states']={0:original['block_states'][0]}
        state=original['block_states'][0].to(model.device);logp=[];values=[]
        with torch.no_grad():
            for t in range(T):
                state=state.reset(data['episode_starts'][t].to(model.device))
                actions=data['actions'][t].to(model.device)
                out,state,_=model.step(data['features'][t].to(model.device),state,actions=actions)
                logp.append(torch.distributions.Categorical(logits=out['logits'],validate_args=False).log_prob(actions).cpu())
                values.append(out['values'].cpu())
            after,_,_=model.step(original['features'][T].to(model.device),state)
        data['old_log_probs']=torch.stack(logp);data['old_values']=torch.stack(values)
        data['bootstrap_values']=after['values'].cpu()
        critical=CriticalTeacherReplay()
        source=torch.load(a.critical_rollout,map_location='cpu',weights_only=False,mmap=True)
        critical.add(source,source=str(a.critical_rollout));del source,original,state,out,after
        optimizer=optimizer_for(model);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        result=update(model,optimizer,data,epochs=1,micro_scenes=16,world_aligned=True,
            checkpoint_chunks=True,critical_replay=critical)
        torch.cuda.synchronize()
        report=dict(status='passed',initial_inference_to_replay_logp_max_error=result['initial_replay_logp_max_error'],
            optimizer_updates=result['optimizer_updates'],logs=result['logs'],wall_seconds=time.monotonic()-started,
            max_memory_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            max_memory_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
            actual_device_memory_GiB=torch.cuda.get_device_properties(0).total_memory/2**30,
            full_neurons=138639,full_signed_edges=15091983,neural_ticks=16,BPTT=16,
            real_world_slots=16,sequence_order='shuffle time blocks,retain original world order',
            fused_edge_grad=True,pad_candidate_batch=a.padded_candidates,checkpoint_candidate_chunks=True,
            deterministic_spmm=model.core.sparse.deterministic_spmm,
            phase_valid_control=model.phase_valid_control,historical_ignored_input_actions_canonicalized=canonicalized,
            critical_event_initialization=event,critical_replay=critical.summary(),
            new_native_actions=0,new_on_policy_samples=0,output_policy_saved=False,
            scope='Compute/memory/fidelity diagnostic on historical fixed inputs/actions;not a new on-policy training or game-performance result')
        write_json(a.output/'report.json',report);print(json.dumps(report))
    except BaseException as error:
        write_json(a.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),no_policy_deployed=True));raise


if __name__=='__main__':main()
