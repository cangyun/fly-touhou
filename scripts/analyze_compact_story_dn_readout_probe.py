"""CPU diagnostic of risk information in frozen DN, isolated by native game.

This fits diagnostic readouts on development-game traces. It must never be
exported as a production policy or counted as an independent native test.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from torch.nn import functional as F
from fly_touhou.dn_residual_risk import load_head
from fly_touhou.compact_story_ppo import risk_weights,risk_loss


def write(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--diagnostics',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--updates',type=int,default=128)
    ap.add_argument('--probe',choices=('inherited','layernorm','input47'),default='inherited')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(611)
    begun=time.monotonic()
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    state={k[5:]:v for k,v in payload['state_dict'].items() if k.startswith('risk.')}
    initial=load_head(dict(schema='DN-only-nonlinear-risk-v1',state_dict=state,
        width=state['residual.0.weight'].shape[0],baseline_trainable=True,
        input_soft_clip=payload.get('risk_input_soft_clip'),
        base_input_soft_clip=payload.get('risk_base_input_soft_clip')),device='cpu')
    unique={};contacts=set()
    for path in sorted(a.diagnostics.glob('event-*.pt')):
        item=torch.load(path,map_location='cpu',weights_only=False)
        world=item['summary']['world']
        for row in item['pre_action_tail']:
            key=(world,row['context']['episode_frames'])
            unique[key]=row
        if 'contact_window_entered' in item['summary']['kinds']:
            row=item['pre_action_tail'][-1]
            if row['normal'] and row['action']<18:contacts.add((world,row['context']['episode_frames']))
    keys=sorted(unique);rows=[unique[k] for k in keys]
    dn=torch.stack([r['candidate_DN'].float() for r in rows])
    margins=torch.stack([r['margins'].T for r in rows])
    trusted=torch.tensor([r['trusted'] for r in rows])
    normal=torch.tensor([r['normal'] for r in rows])
    worlds=torch.tensor([k[0] for k in keys]);actions=torch.tensor([r['action'] for r in rows])
    contact=torch.tensor([k in contacts for k in keys])
    recorded=torch.stack([r['horizon_scores'] for r in rows])
    flat=dn.reshape(-1,dn.shape[-1]);original_flat=flat
    if a.probe=='input47':
        from fly_touhou.compact_story_model import story_candidates
        flat=story_candidates(torch.stack([r['features'].float() for r in rows])).reshape(-1,47)
    margin=margins.reshape(-1,3)
    trust=trusted[:,None,:].expand(-1,18,-1).reshape(-1,3)
    norm=normal[:,None].expand(-1,18).reshape(-1)

    @torch.no_grad()
    def predict(head,x=None):
        return torch.cat([head(v) for v in (flat if x is None else x).split(512)]).reshape(len(rows),18,3)

    @torch.no_grad()
    def metrics(prediction,row_mask):
        p=(8*prediction[...,0]-1).sigmoid()
        eligible=row_mask[:,None]&normal[:,None]&trusted[:,:1]
        positive=eligible&(margins[...,0]<=0);negative=eligible&(margins[...,0]>1)
        known=positive|negative;y=positive[known].float();prob=p[known]
        logits=8*prediction[...,0][known]-1
        pos_loss=F.softplus(-logits[y.bool()]).mean();neg_loss=F.softplus(logits[~y.bool()]).mean()
        indices=row_mask&contact
        selected=p.gather(-1,actions.clamp(0,17)[:,None])[:,0]
        # This is forecast agreement,not actual alternate-action survival.
        costs=-8*p-4*(8*prediction[...,1]-1).sigmoid()-2*(8*prediction[...,2]-1).sigmoid()
        best=costs.argmax(-1);chosen_margin=margins[...,0].gather(-1,best[:,None])[:,0]
        mixed=row_mask&normal&trusted[:,0]&(margins[...,0]>1).any(-1)&(margins[...,0]<=0).any(-1)
        return dict(rows=int(row_mask.sum()),known_danger_directions=int(positive.sum()),known_safe_directions=int(negative.sum()),
            balanced_BCE=float((pos_loss+neg_loss)/2),Brier=float((prob-y).square().mean()),
            danger_recall_at02=float((p[positive]>=.2).float().mean()),safe_false_alarm_at02=float((p[negative]>=.2).float().mean()),
            native_contact_rows=int(indices.sum()),native_contacts_p4_ge02=int((selected[indices]>=.2).sum()),
            native_contact_p4=[float(v) for v in selected[indices]],known_mixed_rows=int(mixed.sum()),
            neural_risk_only_forecast_unsafe=int(((chosen_margin<=0)&mixed).sum()))

    original=predict(initial,original_flat)
    reconstructed=((8*original[...,0]-1).sigmoid()-(8*recorded[...,0]-1).sigmoid()).abs()
    report=dict(schema='CPU-game-heldout-DN-risk-readout-diagnostic-v1',
        checkpoint=str(a.checkpoint),source_diagnostics=str(a.diagnostics),unique_near_event_rows=len(rows),
        native_contact_rows=int(contact.sum()),DN_storage='Recorded FP16 DN,converted to FP32 for CPU readout',
        baseline_reconstruction_max_p4_error=float(reconstructed.max()),
        protocol=f'Two complementary native-game folds:0,1 versus2,3. Fixed{a.updates} updates;brain and inputs fixed.',
        probe=a.probe,probe_input='47 actual candidate sensory features' if a.probe=='input47' else '2598 actual DN activities',
        scope='Development trace decodability diagnostic,not independent test or native counterfactual. Never deploy these heads because development games were used for fitting.',
        CUDA_used=False,production_model_modified=False,production_training_budget_extended=False,folds=[])
    write(a.output/'report.json',report)
    for train_worlds in ((0,1),(2,3)):
        train_rows=(worlds==train_worlds[0])|(worlds==train_worlds[1]);heldout=~train_rows
        fit=train_rows[:,None].expand(-1,18).reshape(-1)&norm&trust[:,0]
        pos=(fit&(margin[:,0]<=0)).nonzero().flatten();neg=(fit&(margin[:,0]>1)).nonzero().flatten()
        if not len(pos) or not len(neg):raise RuntimeError('Readout diagnostic lacks both risk classes')
        if a.probe=='inherited':head=copy.deepcopy(initial)
        elif a.probe=='layernorm':head=torch.nn.Sequential(torch.nn.LayerNorm(flat.shape[-1]),torch.nn.Linear(flat.shape[-1],128),torch.nn.SiLU(),torch.nn.Linear(128,3))
        else:head=torch.nn.Sequential(torch.nn.Linear(47,128),torch.nn.SiLU(),torch.nn.Linear(128,3))
        optimizer=torch.optim.Adam(head.parameters(),lr=3e-4)
        fold=dict(training_worlds=list(train_worlds),heldout_worlds=sorted(set(worlds.tolist())-set(train_worlds)),
            fit_danger_directions=len(pos),fit_safe_directions=len(neg),baseline=metrics(original,heldout),nodes=[])
        generator=torch.Generator().manual_seed(611+train_worlds[0])
        for update in range(1,a.updates+1):
            idx=torch.cat((pos[torch.randint(len(pos),(256,),generator=generator)],neg[torch.randint(len(neg),(256,),generator=generator)]))
            target=margin[idx,:,None];reg,cls=risk_weights(target,trust[idx],norm[idx])
            prediction=head(flat[idx])[:,None,:];loss=risk_loss(prediction,target,reg,cls)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.);optimizer.step()
            if update in (32,64,a.updates):
                node=dict(update=update,training_loss=float(loss.detach()),heldout=metrics(predict(head),heldout))
                fold['nodes'].append(node);print(json.dumps({'fold':train_worlds,**node}),flush=True)
        fold['final']=fold['nodes'][-1]['heldout'];report['folds'].append(fold)
        report['diagnostic_Adam_updates']=sum(f['nodes'][-1]['update'] for f in report['folds'])
        report['elapsed_seconds']=time.monotonic()-begun;write(a.output/'report.json',report)
    report['complete']=True;write(a.output/'report.json',report)


if __name__=='__main__':main()
