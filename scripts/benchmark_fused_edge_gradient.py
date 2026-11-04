"""Numerical and speed comparison of full-graph edge gradients; opt-in only."""
from pathlib import Path
import argparse,sys,time,json,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--connectome',type=Path,required=True);p.add_argument('--previous-job',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--rollout',type=Path)
    p.add_argument('--reference-checkpoint',type=Path)
    p.add_argument('--event-rollout',type=Path)
    p.add_argument('--event-checkpoint',type=Path)
    p.add_argument('--event-port-amplitude',type=float,default=.25)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:print(dict(status='prepared',CUDA_started=False));return
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json
    previous=read(a.previous_job/'launch.json')
    try:
        q=psutil.Process(previous['PID'])
        if abs(q.create_time()-previous['create_time'])<.001:raise RuntimeError('The exact previous CUDA/native job is still active')
    except psutil.NoSuchProcess:pass
    q=psutil.Process();q.cpu_affinity(q.cpu_affinity()[-2:]);q.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'launch.json',dict(PID=q.pid,create_time=q.create_time(),
        schema='full-connectome-same-computation-CUDA-benchmark-v1',argv=sys.argv))
    os.environ['FLY_TOUHOU_EDGE_GRAD_CACHE']=str(a.output/'kernel-cache')
    import torch
    from fly_touhou.fused_edge_gradient import edge_gradient
    from fly_touhou.connectome import load_connectome
    from fly_touhou.continuous_sparse import TrainableDirectedSparse
    torch.set_num_threads(2);torch.manual_seed(202610330);torch.cuda.manual_seed_all(202610330)
    torch.backends.cuda.matmul.allow_tf32=False
    def reference(x,g,pre,post):
        out=x.new_empty(len(pre))
        for first in range(0,len(pre),262144):
            last=first+262144
            out[first:last]=(x.index_select(1,pre[first:last])*g.index_select(1,post[first:last])).sum(0)
        return out
    results=[]
    for B,E in ((1,7),(16,10003),(32,8192),(128,131072)):
        x=torch.randn(B,512,device='cuda');g=torch.randn_like(x)
        pre=torch.randint(512,(E,),device='cuda');post=torch.randint(512,(E,),device='cuda')
        actual=edge_gradient(x,g,pre,post);expected=reference(x,g,pre,post)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)
        results.append(dict(batch=B,edges=E,max_error=float((actual-expected).abs().max())))
    # Check the actual custom-autograd integration, not just the standalone dot.
    pre=torch.randint(512,(8192,));post=torch.randint(512,(8192,))
    op=TrainableDirectedSparse(512,pre.numpy(),post.numpy()).cuda()
    x=torch.randn(16,512,device='cuda',requires_grad=True);w=torch.randn(8192,device='cuda',requires_grad=True)
    loss_grad=torch.randn_like(x)
    op.fused_edge_grad=False;old=torch.autograd.grad((op(x,w)*loss_grad).sum(),(x,w))
    op.fused_edge_grad=True;new=torch.autograd.grad((op(x,w)*loss_grad).sum(),(x,w))
    for actual,expected in zip(new,old):torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)
    graph=load_connectome(a.connectome,max_neurons=None,verify=True)
    pre=torch.as_tensor(graph.pre,device='cuda',dtype=torch.long)
    post=torch.as_tensor(graph.post,device='cuda',dtype=torch.long)
    x=(torch.rand(128,graph.n_neurons,device='cuda')>.9).float();g=torch.randn_like(x)
    with torch.no_grad():
        expected=reference(x,g,pre,post);actual=edge_gradient(x,g,pre,post)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)
        max_error=float((actual-expected).abs().max())
        def timing(fn,count):
            fn();torch.cuda.synchronize();rows=[]
            for _ in range(count):
                torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();value=fn();end.record();torch.cuda.synchronize()
                rows.append(dict(milliseconds=start.elapsed_time(end),extra_peak_bytes=torch.cuda.max_memory_allocated()-base));del value
            return rows
        old=timing(lambda:reference(x,g,pre,post),3);new=timing(lambda:edge_gradient(x,g,pre,post),5)
    report=dict(status='passed',synthetic_numerics=results,autograd_integration_passed=True,
        full_graph=dict(neurons=graph.n_neurons,edges=graph.n_edges,batch=128,max_error=max_error),
        reference=old,fused=new,speedup=(sum(v['milliseconds'] for v in old)/len(old))/(sum(v['milliseconds'] for v in new)/len(new)),
        neural_topology_or_loss_changed=False,GPU=torch.cuda.get_device_name())
    if a.checkpoint and a.rollout:
        del graph,x,g,expected,actual,pre,post,op,w,loss_grad
        import gc
        gc.collect();torch.cuda.empty_cache()
        report['candidate_batching']=candidate_batching(a.checkpoint,a.rollout,a.connectome)
        if a.reference_checkpoint:
            import gc
            gc.collect();torch.cuda.empty_cache()
            report['reference_candidate_batching']=candidate_batching(a.reference_checkpoint,a.rollout,a.connectome)
            gc.collect();torch.cuda.empty_cache()
            report['policy_shift_sensitivity']=policy_shift_sensitivity(a.reference_checkpoint,a.checkpoint,a.rollout,a.connectome)
        if a.event_rollout:
            import gc
            gc.collect();torch.cuda.empty_cache()
            report['critical_event_propagation']=critical_event_propagation(a.event_checkpoint or a.checkpoint,a.event_rollout,a.connectome,a.event_port_amplitude)
    write_json(a.output/'report.json',report);print(json.dumps(report))


def candidate_batching(checkpoint_path,rollout_path,connectome):
    """Actual carried brain states; accept only equal actions/spikes and close values."""
    import torch
    from fly_touhou.compact_story_model import from_story_checkpoint
    from fly_touhou.spiking_engineered_control import SpikingState
    payload=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    model=from_story_checkpoint(connectome,payload).eval();del payload
    data=torch.load(rollout_path,map_location='cpu',weights_only=False);results=[]
    with torch.no_grad():
        for time_index in (0,128,384):
            if time_index>=len(data['features']):continue
            for B in (1,2,4,8,16):
                if B>data['features'].shape[1]:continue
                f=data['features'][time_index,:B].to(model.device)
                before=SpikingState.pack([data['block_states'][time_index].row(w) for w in range(B)]).to(model.device)
                model.pad_candidate_batch=True;old,old_state,_=model.step(f,before,return_all_DN=True)
                model.pad_candidate_batch=False;new,new_state,_=model.step(f,before,return_all_DN=True)
                row=dict(rollout_step=time_index,worlds=B,passed=True)
                try:
                    for key in ('logits','horizon_scores','values','candidate_DN'):
                        torch.testing.assert_close(new[key],old[key],rtol=2e-5,atol=2e-5)
                    for a,b in zip(new_state.tensors(),old_state.tensors()):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-5)
                    assert torch.equal(new['actions'],old['actions'])
                    assert torch.equal(new_state.spikes,old_state.spikes)
                except (AssertionError,RuntimeError) as error:row.update(passed=False,error=str(error)[:1000])
                row['risk_score_max_error']=float((old['horizon_scores']-new['horizon_scores']).abs().max())
                def timing(pad):
                    model.pad_candidate_batch=pad;model.step(f,before);torch.cuda.synchronize();times=[]
                    for _ in range(3):
                        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        start.record();value=model.step(f,before);end.record();torch.cuda.synchronize()
                        times.append(start.elapsed_time(end));del value
                    return sum(times)/len(times)
                row['padded_ms']=timing(True);row['unpadded_ms']=timing(False)
                row['speedup']=row['padded_ms']/row['unpadded_ms'];results.append(row)
    return dict(passed=all(r['passed'] for r in results),cases=results,
        checkpoint=str(checkpoint_path),student_rollout=str(rollout_path),
        loss_graph_and_neural_ticks_unchanged=True,original_padded_batch=128)


def policy_shift_sensitivity(reference_path,current_path,rollout_path,connectome):
    """Fixed real observations/states; block substitution is nonadditive."""
    import torch
    from fly_touhou.compact_story_model import from_story_checkpoint
    from fly_touhou.spiking_engineered_control import SpikingState
    old=from_story_checkpoint(connectome,torch.load(reference_path,map_location='cpu',weights_only=False)).eval()
    new=from_story_checkpoint(connectome,torch.load(current_path,map_location='cpu',weights_only=False)).eval()
    data=torch.load(rollout_path,map_location='cpu',weights_only=False);rows=[]
    def substitute(DN,f,**parts):
        previous={k:getattr(old,k) for k in parts}
        try:
            for k,v in parts.items():setattr(old,k,v)
            return old.heads(DN.reshape(-1,DN.shape[-1]),f)
        finally:
            for k,v in previous.items():setattr(old,k,v)
    with torch.no_grad():
        for t in (0,128,384):
            if t>=len(data['features']):continue
            B=min(8,data['features'].shape[1]);f=data['features'][t,:B].to(old.device)
            before=SpikingState.pack([data['block_states'][t].row(w) for w in range(B)]).to(old.device)
            baseline,_,_=old.step(f,before,return_all_DN=True);full,_,_=new.step(f,before,return_all_DN=True)
            DN=baseline['candidate_DN']
            variants=dict(full_new=full,
                brain_and_ports_only=substitute(full['candidate_DN'],f),
                readouts_only=substitute(DN,f,risk=new.risk,route=new.route,bomb=new.bomb,bomb_urgency_gain=new.bomb_urgency_gain),
                risk_readout_only=substitute(DN,f,risk=new.risk),
                movement_readout_only=substitute(DN,f,route=new.route),
                Bomb_readout_only=substitute(DN,f,bomb=new.bomb,bomb_urgency_gain=new.bomb_urgency_gain))
            ref=torch.distributions.Categorical(logits=baseline['logits'],validate_args=False)
            def guard(out):
                safe=(8*out['horizon_scores'][...,0]-1).sigmoid()<.2
                return safe.any(-1,keepdim=True)&~safe
            old_guard=guard(baseline)
            margins=data['margins'][t,:B,0].to(old.device)
            valid=data['trusted'][t,:B,0].to(old.device)&data['unprotected_mask'][t,:B].to(old.device)
            for name,out in variants.items():
                dist=torch.distributions.Categorical(logits=out['logits'],validate_args=False)
                kl=torch.distributions.kl_divergence(ref,dist);chosen=out['logits'].argmax(-1)
                gap=margins.gather(-1,chosen.clamp_max(17)[:,None])[:,0]
                rows.append(dict(rollout_step=t,worlds=B,variant=name,exact_categorical_KL_mean=float(kl.mean()),
                    exact_categorical_KL_max=float(kl.max()),action_changes=int((chosen!=baseline['actions']).sum()),
                    neural_guard_direction_changes=int((guard(out)!=old_guard).sum()),
                    eligible_physical_rows=int(valid.sum()),geometric_unsafe_chosen_moves=int((valid&(chosen<18)&(gap<=0)).sum())))
    return dict(reference=str(reference_path),current=str(current_path),student_rollout=str(rollout_path),cases=rows,
        scope='Fixed observations and real carried brain states,one-decision sensitivity only;nonadditive parameter block substitutions;not an independent native-game performance result',
        parameters_saved_or_game_actions_sent=False)


def critical_event_propagation(checkpoint_path,rollout_path,connectome,amplitude):
    """Actual teacher-labeled event histories;stimulus changes,no native actions."""
    import torch
    from fly_touhou.compact_story_model import from_story_checkpoint,prime_critical_event_ports
    model=from_story_checkpoint(connectome,torch.load(checkpoint_path,map_location='cpu',weights_only=False)).eval()
    data=torch.load(rollout_path,map_location='cpu',weights_only=False,mmap=True)
    blocks=sorted({(t//16*16,w) for t,w in data['deathbomb_mask'].nonzero().tolist()})[:4]
    middle=torch.ones(model.core.n_neurons,dtype=torch.bool,device=model.device)
    middle[model.core.input_indices]=False;middle[model.core.output_indices]=False
    reference={};rows=[]
    with torch.no_grad():
        for t,w in blocks:
            state=data['block_states'][t].row(w).to(model.device)
            for j in range(t,min(t+16,len(data['features']))):
                state=state.reset(data['episode_starts'][j,w:w+1].to(model.device))
                out,state,_=model.step(data['features'][j,w:w+1].to(model.device),state,
                    actions=data['actions'][j,w:w+1].to(model.device))
                reference[j,w]=(out,state)
        receipt=prime_critical_event_ports(model,amplitude=amplitude)
        for t,w in blocks:
            state=data['block_states'][t].row(w).to(model.device)
            for j in range(t,min(t+16,len(data['features']))):
                state=state.reset(data['episode_starts'][j,w:w+1].to(model.device))
                f=data['features'][j,w:w+1].to(model.device)
                out,state,_=model.step(f,state,actions=data['actions'][j,w:w+1].to(model.device))
                old,previous=reference[j,w];delta=state.rate-previous.rate
                def stats(q):return dict(filtered_spike_delta_RMS=float(delta[:,q].square().mean().sqrt()),
                    binary_spike_changes=int((state.spikes[:,q]!=previous.spikes[:,q]).sum()))
                rows.append(dict(rollout_step=j,world=w,state4=bool(f[0,17]>.5),
                    observed_usable_deathbomb=bool(data['deathbomb_mask'][j,w]),
                    VP=stats(model.core.input_indices),intermediate=stats(middle),DN=stats(model.core.output_indices),
                    DN_pooled_readout_delta_RMS=float((out['DN_mean']-old['DN_mean']).square().mean().sqrt()),
                    baseline_action_probabilities=old['logits'].softmax(-1)[0].cpu().tolist(),
                    primed_action_probabilities=out['logits'].softmax(-1)[0].cpu().tolist(),
                    teacher_Bomb_probability=float(data['rule_probabilities'][j,w,18])))
    return dict(checkpoint=str(checkpoint_path),source_actual_student_rollout=str(rollout_path),
        initialization=receipt,observed_blocks=len(blocks),steps=rows,
        interpretation='Sensory initialization intervention on fixed real histories;no Bomb head training here. Detects VP/intermediate/DN signal passage,not generalized critical-state recognition or native game success.',
        new_PPO_samples=0,native_actions_executed=0,checkpoint_modified=False)


if __name__=='__main__':main()
