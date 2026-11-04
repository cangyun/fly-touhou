"""Fit a DN-only risk head from sealed training-game replay,using held-out worlds.

Writes a candidate,never declares native improvement or launches a game.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import psutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from torch.nn import functional as F
from fly_touhou.dn_residual_risk import DNLayerNormRisk,load_head
from fly_touhou.compact_story_ppo import risk_weights,risk_loss
from fly_touhou.thrl_durable_records import write_json


def reset_risk_optimizer_state(saved):
    """New head has different parameter meanings despite equal tensor counts."""
    groups=[dict(g,params=list(g['params'])) for g in saved['param_groups']]
    risk=[g for g in groups if g.get('name')=='risk']
    if len(risk)!=1 or len(risk[0]['params'])!=6:raise ValueError('Expected the six-parameter source risk optimizer group')
    removed=set(risk[0]['params'])
    return dict(saved,param_groups=groups,state={k:v for k,v in saved['state'].items() if k not in removed})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--updates',type=int,default=128)
    ap.add_argument('--resume-training',type=Path)
    ap.add_argument('--neural-risk-scale',type=float)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);begun=time.monotonic()
    proc=psutil.Process();write_json(a.output/'launch.json',dict(PID=proc.pid,create_time=proc.create_time(),argv=sys.argv,
        CUDA_started=False,training_only=True,development_data_used=False,planned_readout_updates=a.updates,new_environment_decisions=0))
    torch.set_num_threads(2);torch.manual_seed(612)
    data=torch.load(a.cache,map_location='cpu',weights_only=False)
    if data['metadata'].get('development_data_used') is not False:raise ValueError('Training-only DN cache required')
    if Path(data['metadata']['checkpoint']).resolve()!=a.checkpoint.resolve():raise ValueError('Cache/checkpoint binding differs')
    point=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    resume=torch.load(a.resume_training,map_location='cpu',weights_only=False) if a.resume_training else None
    if resume is not None:
        if set(resume['model_payload']['state_dict'])!=set(point['state_dict']) or any(
                not torch.equal(v,resume['model_payload']['state_dict'][k]) for k,v in point['state_dict'].items()):
            raise ValueError('Source Adam checkpoint and parameter checkpoint differ')
    old={k[5:]:v for k,v in point['state_dict'].items() if k.startswith('risk.')}
    baseline=load_head(dict(schema=point.get('risk_readout_kind','DN-only-nonlinear-risk-v1'),state_dict=old,
        width=point.get('risk_readout_width',128),input_soft_clip=point.get('risk_input_soft_clip'),
        base_input_soft_clip=point.get('risk_base_input_soft_clip'),baseline_trainable=point.get('trainable_risk_baseline',False)),device='cpu')
    dn=data['DN'];flat=dn.reshape(-1,dn.shape[-1]);m=data['margins'].transpose(-1,-2)
    margin=m.reshape(-1,3);trust=data['trusted'][:,None,:].expand(-1,18,-1).reshape(-1,3)
    normal=data['normal'][:,None].expand(-1,18).reshape(-1)
    heldout=data['world']>=12;fit=(~heldout)[:,None].expand(-1,18).reshape(-1)&normal&trust[:,0]
    pos=(fit&(margin[:,0]<=0)).nonzero().flatten();neg=(fit&(margin[:,0]>1)).nonzero().flatten()
    if not len(pos) or not len(neg):raise ValueError('Training risk examples lack a class')
    @torch.no_grad()
    def evaluate(head):
        scores=torch.cat([head(x) for x in flat.split(512)]).reshape(len(dn),18,3)
        logits=8*scores[...,0]-1;p=logits.sigmoid()
        valid=heldout[:,None]&data['normal'][:,None]&data['trusted'][:,:1]
        positive=valid&(m[...,0]<=0);negative=valid&(m[...,0]>1)
        if not positive.any() or not negative.any():raise ValueError('Held-out training worlds lack a risk class')
        contacts=heldout&(data['outcomes']==1)&data['normal']&(data['actions']<18)
        chosen=p.gather(-1,data['actions'].clamp(0,17)[:,None])[:,0]
        bands=[]
        for name,band in (('safe1to4',(m[...,0]>1)&(m[...,0]<=4)),
                          ('safe4to8',(m[...,0]>4)&(m[...,0]<=8)),('safe_gt8',m[...,0]>8)):
            mask=valid&band
            bands.append(dict(margin_bin=name,count=int(mask.sum()),
                alerts=int((p[mask]>=.2).sum()),
                alert_rate=float((p[mask]>=.2).float().mean()) if bool(mask.any()) else None))
        return dict(balanced_BCE=float((F.softplus(-logits[positive]).mean()+F.softplus(logits[negative]).mean())/2),
            danger_count=int(positive.sum()),safe_count=int(negative.sum()),
            danger_recall_at02=float((p[positive]>=.2).float().mean()),safe_false_alarm_at02=float((p[negative]>=.2).float().mean()),
            actual_contacts=int(contacts.sum()),contact_prealerts=int((chosen[contacts]>=.2).sum()),safe_margin_bands=bands)
    head=DNLayerNormRisk(flat.shape[-1],width=128);optimizer=torch.optim.Adam(head.parameters(),lr=3e-4)
    report=dict(source_checkpoint=str(a.checkpoint),source_cache=str(a.cache),baseline=evaluate(baseline),
        training_worlds=list(range(12)),heldout_training_worlds=list(range(12,16)),development_games_used=False,
        objective='Existing three-horizon smooth-L1 plus BCE;DN-only input',readout_width=128,nodes=[],
        production_native_performance_unverified=True,new_environment_decisions=0)
    best=None;best_loss=float('inf');generator=torch.Generator().manual_seed(612)
    for update in range(1,a.updates+1):
        idx=torch.cat((pos[torch.randint(len(pos),(256,),generator=generator)],neg[torch.randint(len(neg),(256,),generator=generator)]))
        target=margin[idx,:,None];reg,cls=risk_weights(target,trust[idx],normal[idx])
        loss=risk_loss(head(flat[idx])[:,None,:],target,reg,cls)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.);optimizer.step()
        if update in (32,64,128,256,512,a.updates):
            measured=evaluate(head);node=dict(update=update,loss=float(loss.detach()),validation=measured);report['nodes'].append(node)
            torch.save(head.state_dict(),a.output/f'readout-update{update:03}.pt')
            if measured['balanced_BCE']<best_loss:
                best_loss=measured['balanced_BCE'];best=copy.deepcopy(head.state_dict());report['selected_update']=update;report['selected_validation']=measured
            write_json(a.output/'progress.json',node);print(json.dumps(node),flush=True)
    point=copy.deepcopy(point)
    point['state_dict']={k:v for k,v in point['state_dict'].items() if not k.startswith('risk.')}
    point['state_dict'].update({'risk.'+k:v for k,v in best.items()})
    point.update(risk_readout_kind=head.readout_kind,risk_readout_width=128,risk_input_soft_clip=None,
        risk_base_input_soft_clip=None,trainable_risk_baseline=False)
    if a.neural_risk_scale is not None:
        from fly_touhou.compact_story_risk_composition import checked_risk_scale
        point['neural_risk_scale']=checked_risk_scale(a.neural_risk_scale)
    point['costs']['risk_readout_calibration_updates']=report['selected_update']
    point['risk_readout_calibration']=dict(training_only=True,source_cache=str(a.cache),selected_update=report['selected_update'],
        paid_readout_updates=a.updates,native_performance_unverified=True,brain_parameters_changed=False)
    torch.save(point,a.output/'model-parameters.pt')
    if resume is not None:
        resume['model_payload']=point;resume['optimizer']=reset_risk_optimizer_state(resume['optimizer'])
        resume['risk_optimizer_reset_reason']='LayerNorm head replaces inherited linear-plus-residual head;all other optimizer groups preserved'
        torch.save(resume,a.output/'resumable-training.pt')
        report['resumable_training']=str(a.output/'resumable-training.pt')
        report['new_risk_Adam_state_reset']=True;report['other_Adam_groups_preserved']=True
    report.update(complete=True,paid_readout_updates=a.updates,elapsed_seconds=time.monotonic()-begun,
        output=str(a.output/'model-parameters.pt'),candidate_requires_frozen_native_evaluation=True)
    write_json(a.output/'report.json',report)


if __name__=='__main__':main()
