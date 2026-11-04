"""Scratch CUDA graph/eager parity; no original worlds, optimizer or deployment."""
from pathlib import Path
import argparse,gzip,hashlib,io,json,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def run(a):
    import numpy as np,torch,psutil
    from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_cuda_graph_step import CapturedEntityStep
    from fly_touhou.thrl_inputs import make_encoder,SEPARATED_CONTACT_INPUT as INPUT
    from fly_touhou.thrl_policy_actions import policy_logits
    from fly_touhou.state import Snapshot
    if a.output.exists():raise ValueError('Preserve previous CUDA qualification')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    a.output.mkdir(parents=True,exist_ok=False)
    if sys.platform=='win32':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(81631);torch.cuda.manual_seed_all(81631)
    with _shared_binary(a.checkpoint) as f:raw=f.read()
    checkpoint=a.output/'source_checkpoint.pt';checkpoint.write_bytes(raw)
    digest=hashlib.sha256(raw).hexdigest();p=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    if p['model_config']['input_schema']!=INPUT or p.get('optimizer_transaction_complete') is not True:
        raise ValueError('A complete V4 B1 checkpoint required')
    model=model_from_config(**p['model_config']).cuda().eval();model.load_state_dict(p['model'],strict=True)
    encoder=make_encoder(INPUT);observations=[]
    for path in sorted(a.raw_replay.rglob('*.json.gz')):
        receipt=read(path.with_suffix('.receipt.json'));content=path.read_bytes()
        if hashlib.sha256(content).hexdigest()!=receipt['sha256']:raise ValueError('Raw replay receipt differs')
        record=json.loads(gzip.decompress(content))
        observations.append(encoder.encode(Snapshot.from_dict(record['snapshot'])))
    if len(observations)!=20:raise ValueError('Expected bounded20 measured inference-only states')
    data={k:torch.from_numpy(np.stack([observations[i%20][k] for i in range(30)])).cuda()
        for k in ('features','maps','objects')}
    # Explicit synthetic state variants exercise zero-Bomb, protection, dying,
    # empty entities, and nonzero caller-managed recurrent state as well.
    data['features'][1,4]=0
    data['features'][2,21:25]=0;data['features'][2,24]=1;data['features'][2,19]=.5
    data['features'][3,20]=.5;data['features'][3,5]=1
    data['objects'][4]=0
    hidden=torch.randn(1,30,model.gru_hidden_size,device='cuda')*.1
    runner=CapturedEntityStep(model);errors=[]
    def batch(n):return data['features'][:n],data['maps'][:n],hidden[:,:n],data['objects'][:n]
    def call(n,captured):
        f,m,h,o=batch(n)
        return runner.forward_step(f,m,h,objects=o) if captured else model.forward_step(f,m,h,objects=o)
    try:
        with torch.no_grad():
            for n in (1,2,7,13,20,30,3,9,1):
                eager=call(n,False);captured=call(n,True);detail=[]
                for x,y in zip(eager,captured):
                    torch.testing.assert_close(x,y,rtol=1e-4,atol=3e-5)
                    finite=torch.isfinite(x)&torch.isfinite(y)
                    detail.append(float((x[finite]-y[finite]).abs().max()) if finite.any() else 0.)
                errors.append(dict(batch=n,maximum_absolute_errors_logits_values_hidden=detail))
            # In-place writes model an ordinary Adam storage update. No Adam
            # runs here; only this scratch copy changes, then is restored.
            bias=model.base.actor_head.bias;old=bias.clone();bias[3].add_(.01)
            for x,y in zip(call(7,False),call(7,True)):torch.testing.assert_close(x,y,rtol=1e-4,atol=3e-5)
            bias.copy_(old)
            # Invalid results must fail in Python before being returned; this
            # does not use a device assertion or poison the CUDA context.
            original=data['features'][0,0].clone();data['features'][0,0]=float('nan')
            rejected=False
            try:call(1,True)
            except FloatingPointError:rejected=True
            finally:data['features'][0,0]=original
            if not rejected:raise ValueError('Invalid output escaped graph guard')
            for x,y in zip(call(1,False),call(1,True)):torch.testing.assert_close(x,y,rtol=1e-4,atol=3e-5)
            timings=[]
            for n in (1,4,13,30):
                call(n,True)
                for repetition in range(2):
                    for captured in ((False,True) if repetition==0 else (True,False)):
                        torch.cuda.synchronize();begun=time.perf_counter()
                        for _ in range(20):
                            z,_,_=call(n,captured);policy_logits(z)
                        torch.cuda.synchronize()
                        timings.append(dict(batch=n,repetition=repetition,backend='graph' if captured else 'eager',
                            calls=20,wall_seconds=time.perf_counter()-begun))
        if any(not torch.equal(v.detach().cpu(),p['model'][k]) for k,v in model.state_dict().items()):
            raise ValueError('Qualification failed to restore the scratch weights')
        result=dict(status='qualified_CUDA_prototype_only',schema='B1-CUDA-graph-step-parity-v1',utc=utc_now(),
            checkpoint_copy=str(checkpoint),checkpoint_sha256=digest,
            source_checkpoint_candidate_decisions=p['candidate_optimized_decisions'],
            device=torch.cuda.get_device_name(0),parity=errors,maximum_relative_tolerance=1e-4,
            maximum_absolute_tolerance=3e-5,in_place_weight_update_parity=True,
            nonfinite_output_rejected_and_context_recovered=True,all_model_weights_restored=True,
            source_checkpoint_copy_unchanged=hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest,
            timings=timings,runner=runner.specification(),optimizer_updates=0,native_worlds_started=0,
            current_training_modified=False,fit_on_inference_states=False,
            benchmark_concurrent_original_worlds=sum((p.info['name'] or '').lower().startswith('th14') for p in psutil.process_iter(['name'])),
            end_to_end_native_acceleration_measured=False,production_activation=False)
        write(a.output/'report.json',result)
        print(json.dumps({k:result[k] for k in ('status','device','in_place_weight_update_parity','nonfinite_output_rejected_and_context_recovered','native_worlds_started','production_activation')},ensure_ascii=False))
    except BaseException:
        write(a.output/'failure.json',dict(utc=utc_now(),traceback=traceback.format_exc(),native_worlds_started=0,current_training_modified=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('checkpoint','raw-replay','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
