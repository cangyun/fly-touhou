"""Discarded CUDA transaction for the new B1 loss recipe, never a skill score.

All scratch observations/returns are synthetic. The controlled old reference
exercises V10's support transformation; it is not claimed as V9 native output.
The chosen parent file and actual training weights are never overwritten.
"""
from pathlib import Path
from dataclasses import replace
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(a):
    import torch,numpy as np,psutil
    from torch.distributions import Categorical
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_branching import warm_start
    from fly_touhou.thrl_ppo import PPOConfig,RolloutBatch,make_optimizer,update
    from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher
    from fly_touhou.resource_teacher_v10 import soften_route_reference
    from fly_touhou.thrl_policy_actions import policy_logits
    from fly_touhou.thrl_native_env import SOURCE19_TO_NATIVE36
    c=read(a.config)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('CUDA reserve required for scratch qualification')
    parent=Path(c['parent_PPO_policy']['checkpoint']);parent_sha=c['parent_PPO_policy']['sha256']
    if sha(parent)!=parent_sha or sha(ROOT/'source_manifest.json')!=c['source_manifest_sha256']:raise ValueError('Pinned source/parent differs')
    a.output.mkdir(parents=True,exist_ok=False);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=torch.load(parent,map_location='cpu',weights_only=False)
    model=model_from_config(**p['model_config']).cuda()
    config=PPOConfig(**c['parameters']);optimizer=make_optimizer(model,config)
    inherited=warm_start(torch,model,optimizer,parent,parent_sha,exact_optimizer=True,model_factory=model_from_config)
    del p
    steps,worlds=32,2
    f=torch.zeros(steps,worlds,25);f[...,0]=.5;f[...,1]=.84;f[...,2]=.5;f[...,4]=.375
    f[...,8]=1;f[...,13:16]=1/64;f[...,21]=1
    m=torch.zeros(steps,worlds,16,92,96);o=torch.zeros(steps,worlds,96,29)
    o[:,:,0,0:2]=1;o[:,:,0,6]=20/384;o[:,:,0,7]=-8/448;o[:,:,0,10]=2/128;o[:,:,0,12:14]=1
    o[:,:,1,0]=1;o[:,:,1,3]=1;o[:,:,1,7]=-14/448;o[:,:,1,9]=3/12
    o[:,:,1,10]=3/128;o[:,:,1,12:14]=1;o[:,:,1,21]=80/384;o[:,:,1,26]=1;o[:,:,1,27]=.2
    f[-2:,:,21]=0;f[-2:,:,24]=1;f[-2:,:,19]=.5
    teacher=MovementSafetyTeacher();targets=torch.zeros(steps,worlds,19)
    native=np.asarray(SOURCE19_TO_NATIVE36[:18]);controlled=np.zeros(36);controlled[native]=.8/18;controlled[18]=.2
    for t in range(steps):
        for i in range(worlds):
            observation=dict(features=f[t,i].numpy(),objects=o[t,i].numpy(),maps=m[t,i].numpy())
            q,_=soften_route_reference(controlled,observation,teacher)
            targets[t,i,:18]=torch.from_numpy(q[native]);targets[t,i,18]=float(q[18:].sum())
    model.eval();h=torch.zeros(1,worlds,model.gru_hidden_size,device='cuda');hidden=[];actions=[];old=[];values=[]
    with torch.no_grad():
        for t in range(steps):
            hidden.append(h[0].cpu().clone())
            z,v,h=model.forward_step(f[t].cuda(),m[t].cuda(),h,objects=o[t].cuda())
            d=Categorical(logits=policy_logits(z));action=d.sample()
            actions.append(action.cpu());old.append(d.log_prob(action).cpu());values.append(v.cpu())
        _,bootstrap,_=model.forward_step(f[-1].cuda(),m[-1].cuda(),h,objects=o[-1].cuda())
    frozen={name:{k:v.detach().cpu().clone() for k,v in getattr(model,name).state_dict().items()} for name in ['risk','laser_risk']}
    body={k:v.detach().cpu().clone() for k,v in model.base.state_dict().items()}
    starts=torch.zeros(steps,worlds,dtype=torch.bool);starts[0]=True;weights=torch.ones(steps,worlds);weights[::2]=0
    rewards=torch.zeros(steps,worlds,3);rewards[:,:,0]=torch.linspace(-.03,.02,steps)[:,None]
    batch=RolloutBatch(features=f,maps=m,objects=o,actions=torch.stack(actions),old_log_probs=torch.stack(old),
        values=torch.stack(values),rewards=rewards,hidden_states=torch.stack(hidden),terminated=torch.zeros_like(starts),
        episode_starts=starts,bootstrap_values=bootstrap.cpu(),movement_outcomes=torch.full((steps,worlds),-1),
        teacher_targets=targets,teacher_weights=weights)
    scratch=replace(config,seqs_per_batch=2,ppo_epochs=1,target_kl=None,update_index=0)
    metrics=update(model,optimizer,batch,scratch);torch.cuda.synchronize()
    fixed=all(torch.equal(frozen[name][k],v.detach().cpu()) for name in frozen for k,v in getattr(model,name).state_dict().items())
    changed=sum(not torch.equal(body[k],v.detach().cpu()) for k,v in model.base.state_dict().items())
    if not fixed or not changed or metrics['optimizer_updates']!=2 or not metrics.get('actor_output_loss_gradients'):
        raise ValueError('Scratch loss, frozen heads or gradient diagnostics did not qualify')
    if sha(parent)!=parent_sha:raise ValueError('The chosen parent file was modified')
    result=dict(schema='B1-route-credit-discarded-CUDA-transaction-v1',status='qualified-scratch-only',utc=utc_now(),
        source_runtime=str(ROOT),source_manifest_sha256=c['source_manifest_sha256'],parent_sha256=parent_sha,
        device=torch.cuda.get_device_name(0),common_Adam_states=inherited['common_Adam_states'],
        changed_body_tensors=changed,both_frozen_risk_heads_exact=fixed,scratch_Adam_steps=metrics['optimizer_updates'],
        input_states_returns_and_reference_prior_synthetic=True,V10_support_transformation_exercised=True,
        teacher_distribution_is_not_claimed_raw_native_V9_reference=True,scratch_weights_discarded=True,
        native_worlds_started=0,actual_training_optimized_decisions=0,native_skill_improvement_demonstrated=False,
        metrics=metrics)
    write(a.output/'report.json',result)
    print(json.dumps({k:result[k] for k in ['status','device','common_Adam_states','changed_body_tensors','scratch_Adam_steps','both_frozen_risk_heads_exact']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
