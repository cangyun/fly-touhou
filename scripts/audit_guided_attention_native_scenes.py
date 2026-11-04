"""Read-only CPU attention audit on existing native precontact observations.

This does not fit a model, simulate the brain, or start a native/GPU job.
Scene-level attention is not a policy avoidance result.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source',type=Path,action='append',required=True)
    p.add_argument('--native-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    import psutil
    current=psutil.Process();current.cpu_affinity(current.cpu_affinity()[-2:])
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    import numpy as np
    import torch
    from torch import nn
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_pair_risk import observation_queries
    from fly_touhou.geometry_attention_teaching import per_object_first4,attention_teaching_loss
    torch.set_num_threads(2)
    begun=time.monotonic();scenes=[];abstentions={}
    for game in read(a.native_report)['episode_results']:
        with np.load(game['first_contact_tail']['path'],allow_pickle=False) as d:
            f,o,m,action=d['features'][-1],d['objects'][-1],d['maps'][-1],int(d['action'][-1])
        label,detail=per_object_first4(f,o,m)
        if label is None:
            reason=detail['reason'];abstentions[reason]=abstentions.get(reason,0)+1;continue
        margins,valid=label
        if valid.sum()<2 or margins[action,valid].min()>0:continue
        scenes.append(dict(game=game['evaluation_game'],f=f,o=o,action=action,margins=margins,valid=valid))
    if not scenes:raise ValueError('No known native contact scene for this point-attention diagnostic')
    f=torch.from_numpy(np.stack([s['f'] for s in scenes]));o=torch.from_numpy(np.stack([s['o'] for s in scenes]))
    margins=torch.from_numpy(np.stack([s['margins'] for s in scenes]));labels_valid=torch.from_numpy(np.stack([s['valid'] for s in scenes]))
    with torch.no_grad():pair,valid=observation_queries(f,o[...,:21])
    results=[]
    for path in a.source:
        with _shared_binary(path) as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest();stream.seek(0)
            saved=torch.load(stream,map_location='cpu',weights_only=True)
        payload=saved.get('model_payload',saved);state=payload['state_dict'];prefix='sensory_adapter.object_attention.'
        network=nn.Sequential(nn.Linear(28,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        network.load_state_dict({k[len(prefix+'pair_embedding.'):]:v for k,v in state.items() if k.startswith(prefix+'pair_embedding.')})
        with torch.no_grad():
            values=network(torch.where(valid[:,None,:,None],pair,0.));query=state[prefix+'object_pool_query']
            logits=(values*query).sum(-1)/math.sqrt(128.)
            weights=logits.masked_fill(~valid[:,None],-torch.inf).softmax(-1).numpy()
            loss,details=attention_teaching_loss(logits,margins,labels_valid)
        cases=[]
        for i,s in enumerate(scenes):
            action=s['action'];ids=np.flatnonzero(s['valid']);w=weights[i,action]
            critical=int(np.where(s['valid'],s['margins'][action],np.inf).argmin())
            cases.append(dict(game=s['game'],critical_object=critical,valid_objects=len(ids),
                highest_weight_object=int(w.argmax()),highest_weight_is_danger=bool(s['margins'][action,w.argmax()]<=0),
                critical_lift=float(w[critical]*len(ids)),danger_mass=float(w[s['margins'][action]<=0].sum()),
                normalized_entropy=float(-(w[ids]*np.log(w[ids]+1e-30)).sum()/np.log(len(ids)))))
        results.append(dict(source=str(path),sha256=digest,counters=payload['counters'],cases=len(cases),
            highest_weight_is_danger=sum(s['highest_weight_is_danger'] for s in cases),
            attention_KL=float(loss),query_norm=float(query.norm()),
            mean_critical_lift=float(np.mean([s['critical_lift'] for s in cases])),
            mean_normalized_entropy=float(np.mean([s['normalized_entropy'] for s in cases])),records=cases))
        del state,payload,saved,network
    result=dict(schema='fixed-native-point-attention-no-fitting-v1',native_report=str(a.native_report),
        scenes=len(scenes),abstentions=abstentions,results=results,wall_seconds=time.monotonic()-begun,
        observations_added_to_training=False,weights_modified=False,CUDA_used=False,native_started=False,
        scope='point VALUE pooling only;not complete-brain information transmission or gameplay avoidance')
    write_json(a.output,result)
    print(json.dumps(dict(result,results=[{k:v for k,v in r.items() if k!='records'} for r in results])))


if __name__=='__main__':main()
