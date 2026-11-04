"""Compare lossless recurrent-history storage on saved native observations."""
from pathlib import Path
import argparse,sys,time,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def run(a):
    import numpy as np,torch
    from fly_touhou.thrl_recurrent import PolicyHistory
    from fly_touhou.thrl_model import MOActorCritic
    from fly_touhou.atomic_json_io import atomic_write_json as write
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=MOActorCritic(**payload['model_config']).cuda();model.load_state_dict(payload['model']);model.eval()
    source={k:np.load(a.demonstrations/'demonstrations'/f'{k}.npy',mmap_mode='r') for k in ['features','maps','objects']}
    # Exclude disk I/O from storage timings; both paths see identical arrays.
    rows=[(i,{k:v[t,i].copy() for k,v in source.items()}) for t in range(128,192) for i in range(4)]
    histories={};timing={}
    for mode in ['compressed-v1','raw-v1']:
        history=PolicyHistory(4,64,storage=mode);start=time.perf_counter()
        for i,row in rows:history.append(i,row)
        timing[mode]=dict(append_seconds=time.perf_counter()-start,summary=history.summary())
        histories[mode]=history
    states={}
    for mode,h in histories.items():h.refresh(torch,model)
    for mode,h in histories.items():
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(3):state=h.refresh(torch,model)
        torch.cuda.synchronize();timing[mode]['refresh_seconds_mean3']=(time.perf_counter()-start)/3
        states[mode]=state.detach().cpu()
    error=float((states['compressed-v1']-states['raw-v1']).abs().max())
    if error>1e-6:raise ValueError('Lossless history changed the refreshed state')
    result=dict(status='passed',observations=len(rows),worlds=4,history_length=64,
        maximum_hidden_state_difference=error,timing=timing,native_instances_started=0,
        estimated_raw_bytes_for30x128=30*128*sum(v.nbytes for v in rows[0][1].values()),
        scope='Same saved float32 observations and model; asynchronous training trajectory identity not claimed',
        active_training_modified=False)
    write(a.output,result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for n in ['checkpoint','demonstrations','output']:p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
