"""Compare chunk-zero and real continuous B1 GRU inference on native holdout."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read
from fly_touhou.thrl_model import MOActorCritic


def run(a):
    torch.set_num_threads(2);p=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=MOActorCritic(**p['model_config']).cuda().eval();model.load_state_dict(p['model'])
    worlds=sorted({x[0] for x in read(a.split)['heldout']});folder=a.demonstrations/'demonstrations'
    arrays={k:np.load(folder/f'{k}.npy',mmap_mode='r') for k in ['features','maps','objects','targets','dones']}
    t=arrays['features'].shape[0];w=len(worlds)
    f=torch.tensor(arrays['features'][:,worlds].transpose(1,0,2).copy(),device='cuda')
    y=torch.tensor(arrays['targets'][:,worlds].transpose(1,0,2).copy(),device='cuda').reshape(-1,19)
    dones=arrays['dones'][:,worlds]
    if dones.any():raise ValueError('This bounded context audit expects uninterrupted native prefixes')
    encoded=[]
    with torch.inference_mode():
        for world in worlds:
            for start in range(0,t,128):
                batch={k:torch.tensor(arrays[k][start:start+128,world].copy(),device='cuda') for k in ['features','maps','objects']}
                encoded.append(model._encode(batch['features'],batch['maps'],batch['objects']))
        encoded=torch.cat(encoded).reshape(w,t,-1)
        contexts={}
        full,_=model.gru(encoded.transpose(0,1).contiguous())
        contexts['continuous_native_episode']=full.transpose(0,1).reshape(w*t,-1)
        chunks,_=model.gru(encoded.reshape(-1,16,encoded.shape[-1]).transpose(0,1).contiguous())
        contexts['zero_every16_BC_evaluation']=chunks.transpose(0,1).reshape(w*t,-1)
        result={}
        for name,h in contexts.items():
            x=model._policy_logits(h,f.reshape(-1,25));logs=x.log_softmax(-1)
            ent=-(y*y.clamp_min(1e-20).log()).sum(-1);ce=-(y*logs.masked_fill(~torch.isfinite(logs),0.)).sum(-1)
            action=x.argmax(-1).reshape(w,t)
            result[name]=dict(teacher_KL=float((ce-ent).mean()),argmax_agreement=float((x.argmax(-1)==y.argmax(-1)).float().mean()),
                first100_action_counts=torch.bincount(action[:,:100].reshape(-1),minlength=19).tolist(),
                hidden_norm_mean=float(h.norm(dim=-1).mean()))
    write(a.output,dict(native_prefixes=w,states=w*t,checkpoint=str(a.checkpoint),contexts=result,PPO_updates=0))
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for k in ['checkpoint','demonstrations','split','output']:p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
