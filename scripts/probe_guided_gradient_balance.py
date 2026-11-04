"""Full-CUDA gradient attribution on sealed training chunks; no optimizer/game."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
PIN=Path('./outputs/legacy-build/source_snapshots/20261019-v4-64')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PIN/'src'));sys.path.insert(0,str(ROOT/'artifacts/runtime'))
import numpy as np
import torch
from torch.nn import functional as F
from buffered_rollout_reader import load_checked_rollout
from fly_touhou.connectome import load_connectome
from fly_touhou.experiment import _file_lock,file_sha256
from fly_touhou.geometry_action_feedback import action_set_loss,movement_entropy
from fly_touhou.native_teacher_feedback import teacher_cross_entropy
from fly_touhou.whole_brain_learning import load_whole_brain_checkpoint
from fly_touhou.whole_brain_ppo import WholeBrainRollout,ValueNetwork,replay_segment,generalized_advantages,clipped_policy_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--first-updates',type=int,default=32)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with _file_lock(a.run/'.events.lock'):
        lines=(a.run/'events.jsonl').read_text(encoding='utf-8').splitlines()
    events=[json.loads(x) for x in lines]
    ends=[i for i,e in enumerate(events) if e['event']=='checkpoint' and e['payload']['tag']=='ppo-latest']
    assert len(ends)>=a.first_updates
    end=ends[a.first_updates-1]+1;events=events[:end]
    (a.output/'events_prefix.jsonl').write_text('\n'.join(lines[:end])+'\n',encoding='utf-8')
    cfg=json.loads((a.run/'manifest.json').read_text(encoding='utf-8'))['config']
    cp={e['payload']['epoch']:e['payload'] for e in events if e['event']=='checkpoint' and e['payload']['tag']=='ppo-latest'}
    wanted={'interior_combat_anchor','shooting_lane','power','return_anchor','measured_deathbomb'};selected={}
    for e in events:
        if e['event']!='ppo_behavior_rollout_sealed':continue
        seal=e['payload'];epoch=int(Path(seal['path']).name.split('.')[0]);data=load_checked_rollout(seal['path'],seal['sha256'])
        rollout=WholeBrainRollout(**{k:(data[k].numpy() if torch.is_tensor(data[k]) else data[k]) for k in WholeBrainRollout.__dataclass_fields__})
        for i,d in enumerate(data['teacher_target_metadata']['details']):
            goal=d['expert']['target'];kind=goal.get('kind') if isinstance(goal,dict) else None
            kinds=[kind,data['action_set_metadata']['details'][i]['reason']]
            for category in kinds:
                if category in wanted and category not in selected and float(data['teacher_target_weights'][i])>0:
                    start,stop=next((x,y) for x,y in rollout.segments() if x<=i<y)
                    parent=cp.get(epoch-1,{'path':cfg['initial_checkpoint_path'],'sha256':cfg['initial_checkpoint_sha256']})
                    selected[category]={'category':category,'epoch':epoch,'start':start,'stop':stop,'index':i,'seal':seal,'parent':parent}
        del data,rollout
        if wanted<=set(selected):break
    assert wanted<=set(selected),set(selected)
    graph=load_connectome(a.source_dir,max_neurons=None)
    assert graph.n_neurons==138639 and graph.n_edges==15091983
    report={'schema':'full-brain-guided-gradient-attribution-v1','status':'running',
        'native_game_started':False,'optimizer_step_executed':False,'source_checkpoints_modified':False,
        'scope':'local first-update gradient balance on five real training chunks; not a policy-improvement or global causal claim','cases':[]}
    def add(vectors):return [sum(values) for values in zip(*vectors)]
    def norm(vector):return sum(float(x.square().sum()) for x in vector)**.5
    def cosine(x,y):return sum(float((a*b).sum()) for a,b in zip(x,y))/max(norm(x)*norm(y),1e-300)
    for case in selected.values():
        parent=case['parent'];assert file_sha256(parent['path'])==parent['sha256']
        saved=torch.load(parent['path'],map_location='cpu',weights_only=True)
        data=load_checked_rollout(case['seal']['path'],case['seal']['sha256'])
        assert all(torch.equal(v,saved['trainable_parameters'][n]) for n,v in data['behavior_parameters'].items())
        rollout=WholeBrainRollout(**{k:(data[k].numpy() if torch.is_tensor(data[k]) else data[k]) for k in WholeBrainRollout.__dataclass_fields__})
        policy=load_whole_brain_checkpoint(parent['path'],graph,'cuda');model=policy.student;model.enable_cuda_capture()
        critic=ValueNetwork(policy.encoder.size,tuple(saved['rl_state']['critic_hidden_sizes']),
            geometry_prefix_dim=model.geometry_prefix_dim).cuda();critic.load_state_dict(saved['rl_state']['critic_state_dict'])
        params=tuple(model.parameters())+tuple(critic.parameters());actor_count=len(tuple(model.parameters()))
        first,last=case['start'],case['stop'];sl=slice(first,last)
        advantage,returns=generalized_advantages(rollout.rewards,rollout.values,rollout.next_values,
            rollout.terminated,rollout.truncated,gamma=cfg['gamma'],gae_lambda=cfg['gae_lambda'],independent_boundaries=rollout.independent_boundaries)
        normalized=(advantage-advantage.mean())/max(float(advantage.std()),1e-8)
        logits=replay_segment(model,rollout,first,last);lp=F.log_softmax(logits.double(),-1)
        action=torch.as_tensor(rollout.actions[sl],device='cuda');old=torch.as_tensor(rollout.old_log_probabilities[sl],device='cuda')
        chosen=-F.nll_loss(lp,action,reduction='none');error=float((chosen-old).abs().max().detach());assert error<=1e-9
        x=torch.as_tensor(rollout.observations[sl],device='cuda');prediction=critic(x).double()
        target=torch.as_tensor(returns[sl],device='cuda');old_values=torch.as_tensor(rollout.values[sl],device='cuda')
        squared=(prediction-target).square();clipped=old_values+(prediction-old_values).clamp(-cfg['value_clip'],cfg['value_clip'])
        value=.5*torch.maximum(squared,(clipped-target).square()).mean()
        policy_loss=clipped_policy_loss(chosen,old,torch.as_tensor(normalized[sl],device='cuda'),clip_range=cfg['clip_range'])['loss']
        terms={'policy':policy_loss,'value':cfg['value_coefficient']*value,
            'entropy':-cfg['entropy_coefficient']*movement_entropy(lp),
            'action_set':cfg['action_feedback']['coefficient']*action_set_loss(lp,
                torch.as_tensor(rollout.action_set_mask[sl],device='cuda'),torch.as_tensor(rollout.action_set_weights[sl],device='cuda')),
            'teacher_unit':teacher_cross_entropy(lp,torch.as_tensor(rollout.teacher_targets[sl],device='cuda'),
                torch.as_tensor(rollout.teacher_target_weights[sl],device='cuda'))}
        gradients={}
        for name,loss in terms.items():
            g=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            gradients[name]=[z.detach().cpu().double() if z is not None else torch.zeros_like(p,device='cpu',dtype=torch.float64)
                for z,p in zip(g,params)]
        coefficient=cfg['teacher_feedback']['coefficient']
        total=sum(terms[k] for k in ('policy','value','entropy','action_set'))+coefficient*terms['teacher_unit']
        full=torch.autograd.grad(total,params,allow_unused=True)
        full=[z.detach().cpu().double() if z is not None else torch.zeros_like(p,device='cpu',dtype=torch.float64) for z,p in zip(full,params)]
        base=add([gradients[k] for k in ('policy','value','entropy','action_set')]);teacher=gradients['teacher_unit']
        combined=add([base,[coefficient*z for z in teacher]])
        relative=norm([u-v for u,v in zip(full,combined)])/max(norm(full),1e-300);assert relative<5e-5
        alphas={}
        for alpha in (.05,.2,1.):
            combined=add([base,[alpha*z for z in teacher]]);scale=min(1.,cfg['gradient_clip']/max(norm(combined),1e-300))
            alphas[str(alpha)]={'teacher_actor_gradient_norm':alpha*norm(teacher[:actor_count]),
                'teacher_actor_norm_over_other_actor_terms':alpha*norm(teacher[:actor_count])/max(norm(base[:actor_count]),1e-300),
                'combined_global_gradient_norm':norm(combined),'global_clip_multiplier':scale}
        item={**case,'chosen_logp_error':error,'loss_values':{k:float(v.detach()) for k,v in terms.items()},
            'gradient_global_norms':{k:norm(v) for k,v in gradients.items()},
            'gradient_actor_norms':{k:norm(v[:actor_count]) for k,v in gradients.items()},
            'teacher_actor_cosine_with_other_actor_terms':cosine(teacher[:actor_count],base[:actor_count]),
            'linear_decomposition_relative_error':relative,'coefficient_cases':alphas}
        report['cases'].append(item);print(json.dumps({'category':case['category'],'alpha_cases':alphas,
            'teacher_actor_cosine':item['teacher_actor_cosine_with_other_actor_terms']}),flush=True)
        assert file_sha256(parent['path'])==parent['sha256']
        assert all(p.grad is None for p in params)
        for name,value in model.named_parameters():assert torch.equal(value.detach().cpu(),saved['trainable_parameters'][name])
        model.clear_cuda_capture();model.clear_cached_propagation()
        del policy,model,critic,params,terms,gradients,full,base,teacher,combined,logits,lp,total,data,saved,rollout
        gc.collect();torch.cuda.empty_cache()
    report['status']='passed';(a.output/'probe_script.py').write_bytes(Path(__file__).read_bytes())
    with (a.output/'report.json').open('x',encoding='utf-8') as out:json.dump(report,out,ensure_ascii=False,indent=2)
    print(json.dumps({'status':'passed','real_training_categories_checked':len(report['cases'])}),flush=True)


if __name__=='__main__':main()
