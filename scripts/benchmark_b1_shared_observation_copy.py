"""Original-FP32 copy benchmark and real-weight CUDA PPO gradient parity.

This is a numerical diagnostic, not native training, an episode, or a skill
test. Original frames repeat solely to measure parent-side memory work.
"""
from pathlib import Path
import argparse
from collections import deque
import gc
import hashlib
import io
import json
import os
import statistics
import sys
import time

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(a):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    sys.path.insert(0,str(a.runtime/'src'))
    import numpy as np,psutil,torch
    from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_ppo import PPOConfig,ppo_loss_terms
    from fly_touhou.thrl_policy_actions import policy_logits
    if a.output.exists():raise FileExistsError(a.output)
    if psutil.virtual_memory().available<3*2**30:
        raise RuntimeError('Defer diagnostic:less than3GiB available host memory')
    if not torch.cuda.is_available():raise RuntimeError('Real CUDA qualification required')
    trainer_identity=read(a.training/'launch.json')
    trainer=psutil.Process(trainer_identity['pid'])
    if abs(trainer.create_time()-trainer_identity['create_time'])>=.01:
        raise RuntimeError('Actual trainer identity changed')
    trace=read(a.trace)
    if (trace['policy']!='trained_greedy' or trace['game']!=0
            or trace.get('earlier_history_full_map_coverage_known') is not True
            or sha(trace['arrays_file'])!=trace['sha256']):
        raise ValueError('The fixed first BC development trace is required')
    with np.load(trace['arrays_file'],allow_pickle=False) as archive:
        source={k:archive[k] for k in ('features','maps','objects','hidden')}
    shapes=dict(features=(25,),maps=(16,92,96),objects=(96,29))
    if any(source[k].shape!=(128,*shape) or source[k].dtype!=np.float32 for k,shape in shapes.items()):
        raise ValueError('Exact original FP32 modalities required')
    count,steps=4,64
    pool={k:source[k][-count:].copy() for k in shapes}
    current={k:v.copy() for k,v in pool.items()}
    storage={k:torch.zeros((steps,count,*shape),dtype=torch.float32) for k,shape in shapes.items()}
    torch.set_num_threads(2)
    def copy_trial(shared):
        histories=[deque(maxlen=128) for _ in range(count)]
        gc.collect();started=time.perf_counter()
        for t in range(steps):
            for i in range(count):
                if shared:
                    message={k:v[i].view() for k,v in pool.items()}
                    for value in message.values():value.setflags(write=False)
                    for key,value in message.items():np.copyto(storage[key][t,i].numpy(),value,casting='no')
                    histories[i].append({k:v.copy() for k,v in message.items()})
                else:
                    message={k:v[i].copy() for k,v in pool.items()}
                    for key,value in message.items():current[key][i]=value
                    for key,value in current.items():storage[key][t,i].copy_(torch.from_numpy(value[i]))
                    histories[i].append({k:v[i].copy() for k,v in current.items()})
        # One owned boundary snapshot is retained in both modes.
        boundary={k:v.copy() for k,v in (pool if shared else current).items()}
        elapsed=time.perf_counter()-started
        digest=hashlib.sha256()
        for key,value in storage.items():
            array=value.numpy()
            for i in range(count):np.testing.assert_array_equal(array[:,i],np.broadcast_to(pool[key][i],(steps,*shapes[key])))
            digest.update(memoryview(array))
        for i,history in enumerate(histories):
            for row in history:
                for key,value in row.items():np.testing.assert_array_equal(value,pool[key][i])
        for key,value in boundary.items():np.testing.assert_array_equal(value,pool[key])
        return dict(shared=shared,seconds=elapsed,destination_sha256=digest.hexdigest(),
            history_rows=sum(len(h) for h in histories),sample_rows=steps*count)
    timings=[copy_trial(mode) for mode in (False,True,True,False)]
    if len({r['destination_sha256'] for r in timings})!=1:raise ValueError('FP32 rollout copy changed bits')
    # Independently construct the two input paths for the actual policy.
    legacy={k:torch.from_numpy(np.stack([pool[k].copy() for _ in range(16)])).transpose(0,1).contiguous() for k in shapes}
    borrowed={k:torch.empty_like(v) for k,v in legacy.items()}
    for key,value in pool.items():
        for i in range(count):
            for t in range(16):np.copyto(borrowed[key][i,t].numpy(),value[i],casting='no')
        if not torch.equal(legacy[key],borrowed[key]):raise ValueError('Actual CUDA inputs differ')
    del storage,current;gc.collect()
    with _shared_binary(a.training/'trial_000_both/latest.pt') as stream:raw=stream.read()
    checkpoint=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    if checkpoint.get('optimizer_transaction_complete') is not True:
        raise ValueError('Only a committed actual PPO policy may be copied')
    torch.cuda.set_per_process_memory_fraction(.06)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model=model_from_config(**checkpoint['model_config']).cuda().train()
    model.load_state_dict(checkpoint['model'],strict=True)
    if any(not torch.equal(v.detach().cpu(),checkpoint['model'][k]) for k,v in model.state_dict().items()):
        raise ValueError('The qualification model did not load its actual checkpoint')
    cfg=PPOConfig(**checkpoint['config']['parameters'])
    h=torch.from_numpy(source['hidden'][-count:])[None].cuda()
    dones=torch.zeros((count,16),dtype=torch.bool,device='cuda')
    torch.manual_seed(823711);torch.cuda.manual_seed_all(823711)
    cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all()
    reference=None;results=[];old=None
    for label,modalities in [('legacy',legacy),('borrowed',borrowed)]:
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(gpu_rng)
        model.zero_grad(set_to_none=True)
        batch={k:v.cuda() for k,v in modalities.items()}
        logits,values=model.forward_sequence(batch['features'],batch['maps'],h.clone(),dones,
            cfg.critic_shared_grad_scale,objects=batch['objects'])
        flat_logits=logits.reshape(-1,19);flat_values=values.reshape(-1,3)
        if old is None:
            actions=flat_logits.detach().argmax(-1)
            distribution=torch.distributions.Categorical(logits=policy_logits(flat_logits.detach(),cfg.logit_clamp))
            old=(actions,distribution.log_prob(actions),flat_values.detach().clone())
        advantages=torch.linspace(-1,1,count*16,device='cuda')
        returns=old[2]+advantages[:,None]*torch.tensor([.01,-.02,.03],device='cuda')
        terms=ppo_loss_terms(flat_logits,flat_values,old[0],old[1],old[2],advantages,returns,cfg,.0001)
        terms['total_loss'].backward();torch.cuda.synchronize()
        with torch.no_grad():
            _,_,step_hidden=model.forward_step(batch['features'][:,0],batch['maps'][:,0],h.clone(),
                objects=batch['objects'][:,0])
        result=dict(logits=logits.detach().cpu(),values=values.detach().cpu(),hidden=step_hidden.detach().cpu(),
            loss=float(terms['total_loss'].detach()),gradients={k:p.grad.detach().cpu().clone() for k,p in model.named_parameters() if p.grad is not None})
        if reference is None:reference=result
        else:
            errors={key:float((result[key]-reference[key]).abs().max()) for key in ('logits','values','hidden')}
            if result['gradients'].keys()!=reference['gradients'].keys():raise ValueError('Gradient coverage changed')
            gradient_errors={k:float((v-reference['gradients'][k]).abs().max()) for k,v in result['gradients'].items()}
            if max(errors.values())>3e-5 or max(gradient_errors.values())>3e-5 or abs(result['loss']-reference['loss'])>3e-5:
                raise ValueError('Real CUDA forward/PPO gradient parity failed')
            results.append(dict(output_max_errors=errors,gradient_max_error=max(gradient_errors.values()),
                gradient_parameter_tensors=len(gradient_errors),loss_difference=abs(result['loss']-reference['loss']),
                greedy_actions_equal=torch.equal(result['logits'].argmax(-1),reference['logits'].argmax(-1))))
        del batch,logits,values,step_hidden,flat_logits,flat_values,terms
    if any(not torch.equal(v.detach().cpu(),checkpoint['model'][k]) for k,v in model.state_dict().items()):
        raise ValueError('Diagnostic changed policy weights')
    median_legacy=statistics.median(r['seconds'] for r in timings if not r['shared'])
    median_shared=statistics.median(r['seconds'] for r in timings if r['shared'])
    report=dict(schema='B1-original-FP32-map-copy-real-CUDA-PPO-parity-v1',status='qualified-numerically-only',
        utc=utc_now(),runtime=str(a.runtime),source_manifest_sha256=sha(a.runtime/'source_manifest.json'),
        trace=str(a.trace),trace_sha256=trace['sha256'],unique_original_input_frames=4,
        timing_repetitions=timings,median_legacy_copy_seconds=median_legacy,
        median_borrowed_copy_seconds=median_shared,isolated_copy_speed_ratio=median_legacy/median_shared,
        speed_scope='Parent-side copy choreography only;four original frames repeat;not native or neural end-to-end throughput',
        actual_checkpoint_sha256=hashlib.sha256(raw).hexdigest(),actual_checkpoint_optimized_decisions=checkpoint['candidate_optimized_decisions'],
        actual_checkpoint_weights_verified_loaded=True,CUDA=True,real_policy_PPO_parity=results,
        hidden_comparison_scope='One real-policy forward_step from the same fixed input/context;forward_sequence returns actor and critic outputs only',
        PPO_loss_scope='Actor surrogate,clipped three-critic losses,entropy;no optimizer step;auxiliary loss derivatives not separately measured',
        exact_original_FP32_samples_preserved=True,all_policy_weights_unchanged=True,
        native_worlds_started=0,optimizer_updates=0,live_job_configuration_changed=False,
        live_trainer_identity=trainer_identity,trainer_alive_after=trainer.is_running(),
        actual_training_speedup_not_established=True,native_qualification_still_required=True,
        deployment_forbidden=True)
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'report.json',report)
    (a.output/'diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k not in ('timing_repetitions',)},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('runtime','training','trace','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
