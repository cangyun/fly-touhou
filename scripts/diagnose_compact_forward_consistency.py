"""Localize cached/training CSR and grad-mode differences on identical inputs."""
from pathlib import Path
import argparse,os,sys,json,time,types,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','rollout','connectome','previous-job','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--deterministic-spmm',action='store_true')
    p.add_argument('--unpad-candidates',action='store_true')
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:print(dict(status='prepared',CUDA_started=False));return
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json
    previous=read(a.previous_job/'launch.json')
    try:
        q=psutil.Process(previous['PID'])
        if abs(q.create_time()-previous['create_time'])<.001:raise RuntimeError('Previous CUDA job live')
    except psutil.NoSuchProcess:pass
    q=psutil.Process();q.cpu_affinity(q.cpu_affinity()[-2:]);q.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        FLY_TOUHOU_FUSED_EDGE_GRAD='1',FLY_TOUHOU_EDGE_GRAD_CACHE=str(a.output/'kernel-cache'))
    if a.deterministic_spmm:
        os.environ['FLY_TOUHOU_DETERMINISTIC_SPMM']='1'
        os.environ['FLY_TOUHOU_SPMM_CACHE']=str(a.output/'spmm-kernel-cache')
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'launch.json',dict(PID=q.pid,create_time=q.create_time(),argv=sys.argv,schema='fixed-input-SNN-forward-consistency-v1'))
    try:
        import torch
        from fly_touhou.compact_story_model import from_story_checkpoint,prime_critical_event_ports,sequence
        model=from_story_checkpoint(a.connectome,torch.load(a.checkpoint,map_location='cpu',weights_only=False))
        model.phase_valid_control=True;model.pad_candidate_batch=not a.unpad_candidates;prime_critical_event_ports(model)
        if a.deterministic_spmm:model.core.sparse.deterministic_spmm=True
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
        original=torch.load(a.rollout,map_location='cpu',weights_only=False,mmap=True)
        features=original['features'][:16].to(model.device);starts=original['episode_starts'][:16].to(model.device)
        actions=original['actions'][:16].to(model.device)
        control=(features[...,14]>.5)|(features[...,17]>.5)
        actions=torch.where(control,actions,torch.zeros_like(actions))
        def initial():return original['block_states'][0].to(model.device)
        @torch.no_grad()
        def collect():
            model.eval();state=initial();rows=[]
            for t in range(16):
                state=state.reset(starts[t]);out,state,_=model.step(features[t],state,actions=actions[t])
                rows.append(out['logits'].log_softmax(-1).gather(-1,actions[t,:,None])[...,0])
            return torch.stack(rows),state
        original_message=model.core.message;cases=[]
        for deterministic in ((True,) if a.deterministic_spmm else (False,True)):
            torch.use_deterministic_algorithms(deterministic)
            for backend in (('fixed_order_CSR',) if a.deterministic_spmm else ('cached_inference','same_sparse_function')):
                model.core.message=(original_message if backend=='cached_inference' else
                    types.MethodType(lambda self,spikes,values:self.sparse(spikes,values),model.core))
                row=dict(deterministic=deterministic,backend=backend)
                try:
                    reference,reference_state=collect();repeat,repeat_state=collect()
                    row['same_collection_repeat_logp_max_error']=float((repeat-reference).abs().max())
                    row['same_collection_repeat_spike_changes']=int((repeat_state.spikes!=reference_state.spikes).sum())
                    for grad in (False,True):
                        model.train()
                        with torch.set_grad_enabled(grad):
                            result=sequence(model,features,starts,initial(),actions,checkpointed=grad,checkpoint_chunks=grad)
                            selected=result['logits'].log_softmax(-1).gather(-1,actions[...,None])[...,0]
                        difference=(selected.detach()-reference).abs()
                        peak=difference.argmax();t=int(peak//difference.shape[1]);w=int(peak%difference.shape[1])
                        row['grad' if grad else 'no_grad']=dict(max_logp_error=float(difference.max()),
                            per_step_max=difference.amax(-1).cpu().tolist(),largest_error_step=t,largest_error_world=w,
                            largest_error_features={str(i):float(features[t,w,i]) for i in (4,5,12,13,14,15,16,17)},
                            final_spike_changes=int((result['state'].spikes.detach()!=reference_state.spikes).sum()))
                        del result,selected,difference
                except BaseException as error:row.update(error=repr(error),traceback=traceback.format_exc())
                cases.append(row);write_json(a.output/'progress.json',dict(cases=cases));torch.cuda.empty_cache()
        report=dict(status='completed_forward_diagnosis',cases=cases,weights_changed=False,
            deterministic_spmm=model.core.sparse.deterministic_spmm,pad_candidate_batch=model.pad_candidate_batch,
            new_on_policy_samples=0,native_actions=0,full_nodes=138639,full_edges=15091983,neural_ticks=16,sequence_steps=16)
        write_json(a.output/'report.json',report);print(json.dumps(report))
    except BaseException as error:
        write_json(a.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
