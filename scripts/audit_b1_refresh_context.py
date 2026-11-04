"""Read-only GRU refresh audit on stored held-out native prefixes; no game launch."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read
from evaluate_thrl_avoidance_probe import checkpoint_copy


def refreshed_outputs(torch,model,encoded,window):
    parts=[]
    for start in range(0,len(encoded),window):
        hidden=None
        if start:
            _,hidden=model.gru(encoded[max(0,start-window):start,None])
        value,_=model.gru(encoded[start:start+window,None],hidden)
        parts.append(value[:,0])
    return torch.cat(parts)


def run(a):
    import numpy as np
    import torch
    from fly_touhou.thrl_model import MOActorCritic
    if not torch.cuda.is_available():raise RuntimeError('CUDA audit explicitly required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.output.mkdir(parents=True,exist_ok=False)
    checkpoint=a.output/'frozen.pt';sha=checkpoint_copy(a.checkpoint,checkpoint)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=MOActorCritic(**payload['model_config']).cuda().eval();model.load_state_dict(payload['model'])
    if not model.is_recurrent:raise ValueError('This audit is exclusively recurrent')
    cost=dict(candidate_optimized_decisions=payload.get('candidate_optimized_decisions'),
        inherited_decisions=payload.get('config',{}).get('inherited_decisions',0))
    del payload
    worlds=sorted({x[0] for x in read(a.split)['heldout']})[:a.worlds]
    files={key:a.demonstrations/'demonstrations'/f'{key}.npy' for key in ['features','maps','objects','targets','dones']}
    results=[]
    with torch.inference_mode():
        for world in worlds:
            arrays={key:np.load(path,mmap_mode='r') for key,path in files.items()}
            if arrays['dones'][:,world].any():raise ValueError('Continuous-prefix audit cannot silently cross episode resets')
            encoded=[]
            for start in range(0,len(arrays['features']),64):
                batch={k:torch.as_tensor(arrays[k][start:start+64,world].copy(),device='cuda') for k in ['features','maps','objects']}
                encoded.append(model._encode(batch['features'],batch['maps'],batch['objects']))
            encoded=torch.cat(encoded)
            globals=torch.as_tensor(arrays['features'][:,world].copy(),device='cuda')
            target=torch.as_tensor(arrays['targets'][:,world].copy(),device='cuda')
            full,_=model.gru(encoded[:,None]);full=full[:,0]
            refresh=refreshed_outputs(torch,model,encoded,128)
            # The first two128-state blocks have exactly the same preceding history.
            error=float((full[:256]-refresh[:256]).abs().max())
            if error>3e-6:raise ValueError('Refresh audit failed its equivalent-prefix control')
            full_logits=model._policy_logits(full,globals);refresh_logits=model._policy_logits(refresh,globals)
            p=full_logits.softmax(-1);q=refresh_logits.softmax(-1)
            def fit(logits):
                logs=logits.log_softmax(-1);ce=-(target*logs.masked_fill(~logs.isfinite(),0.)).sum(-1)
                entropy=-(target*target.clamp_min(1e-20).log()).sum(-1)
                return dict(teacher_KL=float((ce-entropy).mean()),agreement=float((logits.argmax(-1)==target.argmax(-1)).float().mean()))
            results.append(dict(world=int(world),states=len(encoded),first256_control_max_error=error,
                all_state_top1_agreement=float((full_logits.argmax(-1)==refresh_logits.argmax(-1)).float().mean()),
                after256_top1_agreement=float((full_logits[256:].argmax(-1)==refresh_logits[256:].argmax(-1)).float().mean()),
                after256_mean_total_variation=float((p[256:]-q[256:]).abs().sum(-1).mean()/2),
                hidden_max_difference=float((full-refresh).abs().max()),
                continuous=fit(full_logits),bounded128_refresh=fit(refresh_logits)))
            del arrays,batch,encoded,full,refresh,globals,target,p,q
    report=dict(schema='B1-offline-fixed-parameter-refresh-context-v1',checkpoint=str(checkpoint),sha256=sha,
        cost=cost,worlds=results,device='cuda:0',native_instances=0,optimizer_updates=0,
        actual_training_modified=False,native_effect_measured=False,
        scope='Same stored held-out teacher prefixes and fixed weights; full continuous GRU vs re-encode last128 states at128-state block boundaries; does not reproduce parameter drift or on-policy state distribution')
    write(a.output/'report.json',report);print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['checkpoint','demonstrations','split','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--worlds',type=int,default=4)
    run(p.parse_args())
