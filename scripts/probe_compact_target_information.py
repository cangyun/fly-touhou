"""CPU probes of target direction and proposed closing in recorded DN activity.

The real actor and its brain are never loaded or changed. Held-out worlds
separate readout fitting from evaluation; this is not game performance.
"""
from pathlib import Path
import argparse,json,sys,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import psutil,numpy as np,torch
    from torch import nn
    from sklearn.metrics import balanced_accuracy_score,roc_auc_score
    from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher
    from fly_touhou.thrl_durable_records import write_json
    q=psutil.Process();q.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);q.cpu_affinity(q.cpu_affinity()[-2:])
    torch.set_num_threads(2);torch.manual_seed(20261031)
    d=torch.load(a.rollout,map_location='cpu',weights_only=False)
    f=d['features'];act=d['actions'];dx=f[...,114]*368
    vx=torch.tensor(MovementSafetyTeacher().velocities[:,0],dtype=torch.float32)[act.clamp_max(17)]
    gain=dx.abs()-(dx-vx*4).abs()
    selected=(d['unprotected_mask']&(f[...,113]>.5)&(f[...,117]>.5)&(dx.abs()>32)
        &d['trusted'][...,0]&(d['margins'][...,0,:]>1).all(-1)&(act<18)&(vx.abs()>1e-5))
    t,w=selected.nonzero(as_tuple=True)
    DN=d['recorded_selected_DN'][t,w].clone();raw=torch.stack((dx[t,w]/368,vx[t,w]/4.5),-1)
    y=torch.stack(((dx[t,w]>0).float(),(gain[t,w]>0).float()),-1)
    train=(w!=1)&(w!=6);test=~train
    source_counts=dict(train=int(train.sum()),test=int(test.sum()),worlds=sorted(w.unique().tolist()),
        train_positive_fraction=y[train].mean(0).tolist(),test_positive_fraction=y[test].mean(0).tolist())
    del d,f,act,dx,vx,gain;gc.collect()
    if not bool(train.any() and test.any()):raise ValueError('Both training and held-out worlds required')
    mean=DN[train].mean(0);std=DN[train].std(0).clamp_min(1e-3)
    results={}
    for name in ('raw_target_and_action','DN_LayerNorm128','DN_feature_standardized128'):
        torch.manual_seed(20261031)
        x=raw if name=='raw_target_and_action' else DN
        if name=='DN_feature_standardized128':x=(x-mean)/std
        width=32 if name=='raw_target_and_action' else 128
        model=nn.Sequential(nn.LayerNorm(x.shape[1]) if name=='DN_LayerNorm128' else nn.Identity(),
            nn.Linear(x.shape[1],width),nn.SiLU(),nn.Linear(width,2))
        optimizer=torch.optim.Adam(model.parameters(),lr=.002)
        positive=y[train].mean(0);pos_weight=((1-positive)/positive.clamp_min(1e-3))
        loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        indices=train.nonzero()[:,0];curve=[]
        for epoch in range(40):
            shuffled=indices[torch.randperm(len(indices))]
            total=0.
            for first in range(0,len(shuffled),128):
                ii=shuffled[first:first+128];optimizer.zero_grad(set_to_none=True)
                loss=loss_fn(model(x[ii]),y[ii]);loss.backward();optimizer.step();total+=float(loss.detach())*len(ii)
            if epoch in (0,9,19,39):curve.append(dict(epoch=epoch+1,training_BCE=total/len(indices)))
        with torch.no_grad():prob=model(x).sigmoid().numpy()
        split={}
        for fold,mask in (('train',train),('heldout_worlds_1_6',test)):
            target=y[mask].numpy();value=prob[mask.numpy()]
            split[fold]={}
            for k,label in enumerate(('target_on_right','proposal_closes_horizontal_gap')):
                split[fold][label]=dict(balanced_accuracy=float(balanced_accuracy_score(target[:,k],value[:,k]>=.5)),
                    AUC=float(roc_auc_score(target[:,k],value[:,k])) if len(np.unique(target[:,k]))==2 else None)
        results[name]=dict(curve=curve,metrics=split)
    report=dict(schema='recorded-DN-target-information-CPU-probe-v1',source=str(a.rollout),counts=source_counts,
        task='Boss offset>32px;all18 directions immediate-safe;nonzero horizontal actual proposal;predict target side and closing sign',
        results=results,neural_policy_changed=False,brain_topology_changed=False,native_games_started=0,
        CUDA_initialized=torch.cuda.is_initialized(),
        limitation='Predictive readout on observed states;not a causal target perturbation or evidence of full-game improvement')
    write_json(a.output,report);print(json.dumps(report))


if __name__=='__main__':main()
