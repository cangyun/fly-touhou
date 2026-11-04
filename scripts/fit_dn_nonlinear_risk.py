"""Finite CPU-only readout capacity experiment on existing frozen DN records."""
from pathlib import Path
import argparse,os,sys,time,json,copy,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--extra-data',type=Path)
    a=p.parse_args()
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    import psutil
    q=psutil.Process();q.cpu_affinity(q.cpu_affinity()[-2:]);q.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    import numpy as np
    import torch
    from torch.nn import functional as F
    from fly_touhou.dn_residual_risk import DNResidualRisk
    from fly_touhou.candidate_spiking_calibration import metrics
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    from fly_touhou.thrl_study_io import read_shared_json as read
    torch.set_num_threads(2);torch.manual_seed(202610317)
    a.output.mkdir(parents=True,exist_ok=False);begun=time.monotonic()
    write_json(a.output/'source-pins.json',{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        Path(__file__),ROOT/'src/fly_touhou/dn_residual_risk.py',ROOT/'src/fly_touhou/candidate_spiking_calibration.py')})
    write_json(a.output/'launch.json',dict(schema='DN-only-CPU-capacity-check-v1',PID=q.pid,create_time=q.create_time(),
        CUDA_started=False,brain_updates=0,native_instances=0,data=str(a.data),extra_data=str(a.extra_data) if a.extra_data else None,
        sampling='hard false-safe,danger,boundary,uniform quarter each'))
    datasets={}
    for fold in ('train','heldout'):
        with np.load(a.data/(fold+'-DN-queries.npz')) as z:
            datasets[fold]={k:torch.from_numpy(z[k]) for k in ('DN','margins','mask')}
        if a.extra_data:
            for record in read(a.extra_data/'manifest.json')['records']:
                if record['fold']!=fold:continue
                with np.load(record['path']) as z:
                    for k in datasets[fold]:datasets[fold][k]=torch.cat((datasets[fold][k],torch.from_numpy(z[k])))
    train,held=datasets['train'],datasets['heldout']
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    s=payload['state_dict'];head=DNResidualRisk(train['DN'].mean(0),train['DN'].std(0,unbiased=False),s['risk.weight'],s['risk.bias'])
    del payload,s
    optimizer=torch.optim.AdamW(head.parameters(),lr=.0003,weight_decay=.0001)
    valid=train['mask'][:,0];m=train['margins'][:,0]
    all_rows=valid.nonzero()[:,0]
    with torch.no_grad():false_safe=valid&(m<=0)&((8*head(train['DN'])[:,0]-1).sigmoid()<.2)
    pools=[false_safe.nonzero()[:,0],(valid&(m<=0)).nonzero()[:,0],(valid&(m.abs()<8)).nonzero()[:,0],all_rows]
    pools=[r if len(r) else all_rows for r in pools]
    with torch.no_grad():initial=metrics(head(held['DN']),held['margins'],held['mask'])
    best=copy.deepcopy(head.state_dict());best_metric=initial;best_step=0;stale=0
    for step in range(1,501):
        indices=torch.cat([r[torch.randint(len(r),(n,))] for r,n in zip(pools,(128,128,128,128))])
        scores=head(train['DN'][indices]);target=train['margins'][indices];mask=train['mask'][indices]
        weights=mask*scores.new_tensor([1.,.3,.1]);known=(target<=0)|(target>1)
        reg=F.smooth_l1_loss(scores.tanh(),torch.tanh((1-target)/8),reduction='none',beta=.1)
        bce=F.binary_cross_entropy_with_logits(8*scores-1,(target<=0).float(),reduction='none')
        loss=(weights*reg).sum()/weights.sum().clamp_min(1)+.2*(weights*bce*known).sum()/(weights*known).sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.);optimizer.step()
        if step%50==0:
            with torch.no_grad():report=metrics(head(held['DN']),held['margins'],held['mask'])
            row=dict(step=step,train_loss=float(loss.detach()),heldout=report,elapsed_seconds=time.monotonic()-begun)
            append_jsonl(a.output/'learning.jsonl',row);write_json(a.output/'progress.json',row)
            if report['balanced_loss']<best_metric['balanced_loss']:
                best=copy.deepcopy(head.state_dict());best_metric=report;best_step=step;stale=0
            else:stale+=1
            if stale>=4:break
    torch.save(dict(schema='DN-only-nonlinear-risk-v1',state_dict=best,width=128,brain_checkpoint=str(a.checkpoint),
        trained_input='2598 actual DN activities only',raw_geometry_or_teacher_action_bypass=False,
        training_extra_data=str(a.extra_data) if a.extra_data else None),a.output/'head.pt')
    write_json(a.output/'calibration.json',dict(before={'heldout':initial},after={'heldout':best_metric},
        head='128-wide nonlinear residual over DN-only linear readout',brain_updates=0,best_step=best_step,
        risk_targets='4/12/24-frame provisional geometric margins,not unobserved actual future',
        sampling='hard false-safe,danger,boundary,uniform quarters',native_performance_not_measured=True))
    write_json(a.output/'result.json',dict(status='completed_finite_CPU_readout_experiment',initial=initial,best=best_metric,
        best_step=best_step,updates=step,CUDA_started=False,brain_updates=0,native_performance_not_measured=True,
        train_queries=len(train['DN']),heldout_queries=len(held['DN']),hard_false_safe_queries=int(false_safe.sum()),
        train_valid_immediate_queries=int(train['mask'][:,0].sum()),heldout_valid_immediate_queries=int(held['mask'][:,0].sum()),
        elapsed_seconds=time.monotonic()-begun))


if __name__=='__main__':main()
