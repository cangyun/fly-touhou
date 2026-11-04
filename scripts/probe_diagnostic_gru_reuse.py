"""Compare one versus four GRU optimizer passes on sealed TRAINING rollouts.

Private CUDA learners only. No original game, held-out data or live checkpoint is
modified. A smaller recorded objective is not evidence of better native survival.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import gzip
import json
import os
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from fly_touhou.diagnostic_gru import DiagnosticGRU, cpu_copy, fit_actor
from fly_touhou.diagnostic_gru_reuse import fit_actor_reuse
from fly_touhou.experiment import file_sha256
from fly_touhou.native_diagnostic import write_json


def seal(path):
    return {'path':str(path),'sha256':file_sha256(path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Every private probe must retain its original result')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA comparison required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True)
    cases=[]
    for seed in (11,29,47):
        for update in (16,64):
            behavior_path=args.root/f'seed{seed}/train/training/behavior_{update:04d}.pt.gz'
            with gzip.open(behavior_path,'rb') as stream:
                data=torch.load(stream,map_location='cpu',weights_only=True)
            parent=Path(data['parent_checkpoint']['path'])
            assert file_sha256(parent)==data['parent_checkpoint']['sha256']
            checkpoint=torch.load(parent,map_location='cpu',weights_only=True)
            results={}
            first_actor=None
            for epochs in (1,4):
                model=DiagnosticGRU().cuda()
                model.load_state_dict(checkpoint['actor'])
                optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
                optimizer.load_state_dict(deepcopy(checkpoint['actor_optimizer']))
                torch.cuda.synchronize()
                started=time.perf_counter()
                if epochs==1:
                    metric,proposals=fit_actor(model,optimizer,data)
                    first_actor=cpu_copy(model.state_dict())
                else:
                    metric,proposals=fit_actor_reuse(model,optimizer,data,epochs=epochs)
                    # Its first step must equal v1, before additional data reuse.
                    expected=results['1']['metric']['after']
                    assert metric['epochs'][0]['after']==expected
                    assert torch.equal(proposals['epochs'][0]['accepted_logp18'],
                                       results['1']['accepted_logp18'])
                torch.cuda.synchronize()
                wall=time.perf_counter()-started
                path=args.output/f'private_seed{seed}_update{update:04d}_passes{epochs}.pt'
                torch.save({'actor':cpu_copy(model.state_dict()),
                    'optimizer':cpu_copy(optimizer.state_dict()), 'proposals':proposals,
                    'parent':data['parent_checkpoint'],'native_actions':0,
                    'private_only_not_a_deployment_checkpoint':True},path)
                results[str(epochs)]={'metric':metric,'fit_wall_seconds':wall,
                    'checkpoint':seal(path),'accepted_logp18':proposals['accepted_logp18']}
                del model,optimizer,proposals
            before=results['1']['metric']['before']['total']
            one=results['1']['metric']['after']['total']
            four=results['4']['metric']['after']['total']
            for entry in results.values():del entry['accepted_logp18']
            assert file_sha256(parent)==data['parent_checkpoint']['sha256']
            case={'seed':seed,'update':update,'behavior':seal(behavior_path),
                'parent':data['parent_checkpoint'],'samples':len(data['actions']),
                'one_vs_four':results,'recorded_loss_reduction_one':before-one,
                'recorded_loss_reduction_four':before-four,
                'additional_recorded_reduction':one-four,
                'live_parent_unchanged':True,'held_out_samples':0,'native_actions':0}
            cases.append(case)
            write_json(args.output/'progress.json',{'cases':len(cases),'latest':case})
            print(json.dumps({'seed':seed,'update':update,'one':one,'four':four,
                'accepted_steps_four':results['4']['metric']['accepted_optimizer_steps'],
                'additional_recorded_reduction':one-four},ensure_ascii=False),flush=True)
    report={'schema':'private-GRU-rollout-reuse-comparison-v1','status':'finished',
        'GPU':torch.cuda.get_device_name(),'cases':cases,
        'sources':[seal(Path(__file__)),seal(ROOT/'src/fly_touhou/diagnostic_gru_reuse.py'),
                   seal(ROOT/'src/fly_touhou/diagnostic_gru.py')],
        'native_actions':0,'held_out_samples':0,'live_training_modified':False,
        'scope':'training objective and runtime feasibility only;not independent game survival',
        'G2_achieved':False,'final_acceptance':False}
    write_json(args.output/'report.json',report,exclusive=True)


if __name__=='__main__':
    main()
