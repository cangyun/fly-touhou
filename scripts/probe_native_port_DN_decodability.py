"""Disposable CPU decoders for recorded native VP/DN and observed input rows.

Splits are by whole native game. Nothing is added to replay or deployed.
The targets are teacher geometry hypotheses, not native counterfactual truth.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--native-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=240)
    a=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    import psutil
    current=psutil.Process();cpus=current.cpu_affinity();current.cpu_affinity(cpus[-4:-2] if len(cpus)>=4 else cpus)
    current.nice(psutil.IDLE_PRIORITY_CLASS)
    import numpy as np
    import torch
    from torch import nn
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.geometry_attention_teaching import per_object_first4
    torch.set_num_threads(2)
    write_json(a.output.with_suffix('.launch.json'),dict(PID=current.pid,create_time=current.create_time(),argv=sys.argv,
        CUDA_used=False,native_started=False,production_training_modified=False))
    begun=time.monotonic();rows=[];abstentions={};native=read(a.native_report)
    for episode in native['episode_results']:
        with np.load(episode['first_contact_tail']['path'],allow_pickle=False) as data:
            for t in range(len(data['features'])):
                f,o,m=data['features'][t],data['objects'][t],data['maps'][t]
                labels,detail=per_object_first4(f,o,m)
                if labels is None:
                    reason=detail['reason'];abstentions[reason]=abstentions.get(reason,0)+1;continue
                per_object,valid=labels;margin=per_object[:,valid].min(-1)
                if not (margin<=0).any() or not (margin>1).any():continue
                row=dict(game=episode['evaluation_game'],margin=margin,
                    raw=np.concatenate((f,o.flatten())),VP=data['VP_activity'][t],DN=data['DN_activity'][t],
                    risk=data['risk_latent_scores'][t] if 'risk_latent_scores' in data else data['predicted_costs'][t],
                    risk_coordinate='latent_before_softplus' if 'risk_latent_scores' in data else 'legacy_nonnegative_cost',
                    policy=-data['noBomb_policy_logits'][t,:18],final=(t==len(data['features'])-1))
                if 'encoded384' in data:row['encoded384']=data['encoded384'][t]
                rows.append(row)
    games=np.unique([r['game'] for r in rows])
    if len(games)<10:raise ValueError('Too few whole games with known mixed directions for the diagnostic')
    margin=np.stack([r['margin'] for r in rows]);group=np.asarray([r['game'] for r in rows])
    positive=margin<=0;negative=margin>1;known=positive|negative
    y=torch.from_numpy(positive.astype(np.float32));mask=torch.from_numpy(known)

    def scores(pred,indices=None):
        ids=np.arange(len(rows)) if indices is None else np.flatnonzero(indices)
        z=np.asarray(pred)[ids];pos=positive[ids];neg=negative[ids]
        comparisons=z[:,:,None]-z[:,None,:];pairs=pos[:,:,None]&neg[:,None,:]
        auc=np.where(comparisons>0,1.,np.where(comparisons==0,.5,0.))
        per_scene=(auc*pairs).sum((1,2))/pairs.sum((1,2))
        chosen=z.argmin(-1)
        return dict(rows=len(ids),within_scene_ranking_AUC=float(per_scene.mean()),
            minimum_risk_action_proxy_safe=float(negative[ids,chosen].mean()),
            minimum_risk_action_proxy_unsafe=float(positive[ids,chosen].mean()))

    folds=[games[i::5] for i in range(5)]
    prior=np.zeros_like(margin);results={}
    views=('raw','encoded384','VP','DN') if all('encoded384' in r for r in rows) else ('raw','VP','DN')
    for view in views:
        x=torch.from_numpy(np.stack([r[view] for r in rows]).astype(np.float32))
        predicted=np.zeros_like(margin);training_pred=np.zeros_like(margin);fold_rows=[]
        for fold,held_games in enumerate(folds):
            held=np.isin(group,held_games);train=~held
            xtrain=x[train];center=xtrain.mean(0);scale=xtrain.std(0,unbiased=False)
            scale=scale.clamp_min(max(1e-5,float(scale.median())*.05))
            normalized=((x-center)/scale).clamp(-20,20)
            torch.manual_seed(292110+fold)
            net=nn.Sequential(nn.Linear(x.shape[-1],64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,18))
            optimizer=torch.optim.Adam(net.parameters(),lr=.003,weight_decay=1e-4)
            pos=y[train].sum(0);neg=((1-y[train])*mask[train]).sum(0)
            weight=(neg/pos.clamp_min(1)).clamp(.25,8.)
            for _ in range(a.steps):
                optimizer.zero_grad(set_to_none=True);logits=net(normalized[train])
                terms=torch.nn.functional.binary_cross_entropy_with_logits(logits,y[train],pos_weight=weight,reduction='none')
                loss=(terms*mask[train]).sum()/mask[train].sum()
                loss.backward();optimizer.step()
            with torch.no_grad():output=net(normalized).numpy()
            predicted[held]=output[held]
            prevalence=(pos/(pos+neg).clamp_min(1)).numpy()
            prior[held]=prevalence
            fold_rows.append(dict(fold=fold,held_games=held_games.tolist(),
                train_rows=int(train.sum()),held_rows=int(held.sum()),
                train=scores(output,train),held=scores(output,held),final_training_loss=float(loss.detach())))
        final=np.asarray([r['final'] for r in rows])
        results[view]=dict(input_dimensions=int(x.shape[-1]),hidden=[64,64],
            parameters=sum(p.numel() for p in net.parameters()),fitted_steps_per_fold=a.steps,
            held_all=scores(predicted),held_final_contacts=scores(predicted,final),folds=fold_rows)
        del x,net,optimizer
    result=dict(schema='disposable-whole-game-split-native-port-DN-decoding-v1',
        source_report=str(a.native_report),source_checkpoint_SHA256=native['checkpoint_sha256'],
        native_games=len(native['episode_results']),included_games=len(games),known_mixed_rows=len(rows),
        known_final_contact_rows=sum(r['final'] for r in rows),abstentions=abstentions,
        actual_risk_coordinates=sorted({r['risk_coordinate'] for r in rows}),
        encoded384_is_actual_same_forward=all('encoded384' in r for r in rows),
        geometry_target='known four-frame point motion;unsafe<=0,safe>1,border cells omitted from BCE',
        baseline=dict(constant_training_danger_prevalence=scores(prior),
            actual_frozen_risk=scores(np.stack([r['risk'] for r in rows])),
            actual_frozen_composed_policy=scores(np.stack([r['policy'] for r in rows]))),
        results=results,whole_games_separated=True,weights_exported=False,models_deployed=False,
        production_training_modified=False,development_observations_added_to_replay=False,
        CUDA_used=False,native_started=False,wall_seconds=time.monotonic()-begun,
        scope='exploratory decoding on few correlated precontact observations;different input widths and small game cohorts prevent a causal or gameplay claim')
    assert not torch.cuda.is_initialized()
    write_json(a.output,result)
    print(json.dumps(dict(result,results={k:{n:v for n,v in r.items() if n!='folds'} for k,r in results.items()})))


if __name__=='__main__':main()
