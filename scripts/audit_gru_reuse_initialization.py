"""Verify the isolated four-pass experiment starts identically to each v1 seed."""
from pathlib import Path
import argparse
import json
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.experiment import file_sha256
from fly_touhou.native_diagnostic import write_json


def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    args=parser.parse_args()
    output=args.candidate/'initialization_comparison_v1.json'
    if output.exists():raise FileExistsError('Preserve the initialization comparison')
    entries=[]
    for seed in (11,29,47):
        paths=[root/f'seed{seed}/train/checkpoint_0000.pt' for root in (args.base,args.candidate)]
        first,second=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
        assert first['seed']==second['seed']==seed
        assert all(v==0 for v in first['counters'].values())
        assert all(v==0 for v in second['counters'].values())
        verified=[]
        for key in ('actor','critic','actor_optimizer','critic_optimizer','sampler','torch_rng','cuda_rng'):
            assert equal(first[key],second[key]),f'seed{seed}:{key}'
            verified.append(key)
        entries.append({'seed':seed,'base':{'path':str(paths[0]),'sha256':file_sha256(paths[0])},
            'candidate':{'path':str(paths[1]),'sha256':file_sha256(paths[1])},'equal_initial_fields':verified})
    result={'schema':'matched-initial-GRU-reuse-experiment-audit-v1','status':'passed','seeds':entries,
        'auditor_sha256':file_sha256(Path(__file__)),'native_actions':0,
        'scope':'identical initial learners and samplers;does not prove avoidance improvement',
        'G2_achieved':False,'final_acceptance':False}
    write_json(output,result,exclusive=True)
    print(json.dumps({'status':'passed','seeds':3,'output':str(output)},ensure_ascii=False))


if __name__=='__main__':main()
