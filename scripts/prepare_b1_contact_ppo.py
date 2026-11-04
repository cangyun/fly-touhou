"""Preserve B1 PPO weights/Adam/RNG for the measured contact-sensor correction."""
from pathlib import Path
import argparse,hashlib,json,sys
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(a):
    import torch
    from torch.distributions import Categorical
    from fly_touhou.thrl_contact_migration import contact_parent
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_branching import warm_start
    from fly_touhou.thrl_ppo import PPOConfig,RolloutBatch,make_optimizer,update
    from fly_touhou.thrl_policy_actions import policy_logits
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.atomic_json_io import atomic_write_json as write
    if not torch.cuda.is_available():raise RuntimeError('CUDA qualification required;CPU fallback forbidden')
    if sha(a.parent)!=a.parent_sha256:raise ValueError('Source PPO identity changed')
    if a.output.exists():raise ValueError('Never overwrite a previous migration')
    evidence=read_shared_json(a.qualification)
    evidence.update(qualification_sha256=sha(a.qualification),source_checkpoint_sha256=a.parent_sha256)
    source=torch.load(a.parent,map_location='cpu',weights_only=False)
    payload=contact_parent(source,evidence=evidence)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    before=model_from_config(**source['model_config'])
    before.load_state_dict(source['model'],strict=True)
    model=model_from_config(**payload['model_config']).cuda()
    model.load_state_dict(payload['model'],strict=True)
    if list(dict(before.named_parameters()))!=list(dict(model.named_parameters())):
        raise ValueError('Migration changed optimizer parameter order')
    if any(not torch.equal(model.state_dict()[k].cpu(),v) for k,v in source['model'].items()):
        raise ValueError('Migration changed model weights')
    a.output.mkdir(parents=True,exist_ok=False)
    checkpoint=a.output/'parent.pt';torch.save(payload,checkpoint)
    cfg=PPOConfig(**source['config']['parameters'])
    optimizer=make_optimizer(model,cfg)
    inherited=warm_start(torch,model,optimizer,checkpoint,sha(checkpoint),exact_optimizer=True,model_factory=model_from_config)
    if inherited['common_Adam_states']!=len(source['optimizer']['state']):
        raise ValueError('Migration lost Adam moments')
    # Synthetic transaction qualification only. This scratch model is discarded;
    # no test-game observation is fitted and these updates are never saved.
    steps,envs=16,2
    f=torch.zeros(steps,envs,25);f[:,:,2]=.5;f[:,:,4]=.125;f[:,:,8]=1;f[:,:,21]=1
    f[:,:,0]=.5;f[:,:,1]=.84;f[:,:,13:16]=1/64
    maps=torch.zeros(steps,envs,16,92,96)
    objects=torch.zeros(steps,envs,96,21)
    objects[:,:,0,0]=1;objects[:,:,0,1]=1;objects[:,:,0,7]=-.2
    objects[:,:,0,9]=.1;objects[:,:,0,10]=2/128;objects[:,:,0,12:14]=1
    raw=dict(features=f,maps=maps,objects=objects)
    model.eval();h=torch.zeros(1,envs,model.gru_hidden_size,device='cuda')
    hidden=[];actions=[];old=[];values=[]
    with torch.no_grad():
        for t in range(steps):
            hidden.append(h[0].cpu().clone())
            z,v,h=model.forward_step(f[t].cuda(),maps[t].cuda(),h,objects=objects[t].cuda())
            d=Categorical(logits=policy_logits(z));action=d.sample()
            actions.append(action.cpu());old.append(d.log_prob(action).cpu());values.append(v.cpu())
        _,bootstrap,_=model.forward_step(f[-1].cuda(),maps[-1].cuda(),h,objects=objects[-1].cuda())
    risk={k:v.detach().cpu().clone() for k,v in model.risk.state_dict().items()}
    body={k:v.detach().cpu().clone() for k,v in model.base.state_dict().items()}
    starts=torch.zeros(steps,envs,dtype=torch.bool);starts[0]=True
    rewards=torch.zeros(steps,envs,3);rewards[:,:,0]=torch.linspace(-.03,.02,steps)[:,None]
    batch=RolloutBatch(**raw,actions=torch.stack(actions),old_log_probs=torch.stack(old),values=torch.stack(values),
        rewards=rewards,hidden_states=torch.stack(hidden),terminated=torch.zeros_like(starts),
        episode_starts=starts,bootstrap_values=bootstrap.cpu())
    smoke=replace(cfg,seqs_per_batch=2,ppo_epochs=1,target_kl=None,
        deathbomb_auxiliary_coefficient=0.,movement_teacher_coefficient=0.,movement_collision_coefficient=0.,
        movement_dense_coefficient=0.,online_teacher_coefficient=0.,cache_frozen_entity_risk=True)
    metrics=update(model,optimizer,batch,smoke);torch.cuda.synchronize()
    frozen=all(torch.equal(risk[k],v.detach().cpu()) for k,v in model.risk.state_dict().items())
    changed=sum(not torch.equal(body[k],v.detach().cpu()) for k,v in model.base.state_dict().items())
    if not frozen or not changed or metrics['optimizer_updates']<1:raise RuntimeError('Actual CUDA update qualification failed')
    result=dict(status='qualified',schema='B1-contact-v4-parent-CUDA-qualification-v1',
        source_checkpoint=str(a.parent),source_sha256=a.parent_sha256,
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),model_config=payload['model_config'],
        migration=payload['input_migration'],warm_start=inherited,device=torch.cuda.get_device_name(0),
        preserved_Adam_states=inherited['common_Adam_states'],parameter_counts=model.parameter_counts(),
        lifetime_optimized_PPO_decisions=source['lifetime_decisions'],
        scratch_CUDA_Adam_steps=metrics['optimizer_updates'],scratch_changed_body_tensors=changed,
        scratch_frozen_risk_exact=frozen,scratch_uses_synthetic_observations_and_returns=True,
        scratch_weights_discarded=True,new_native_worlds=0,training_native_updates=0,
        source_candidate_optimized_decisions=source['candidate_optimized_decisions'])
    write(a.output/'qualification.json',result)
    print(json.dumps({k:result[k] for k in ['status','device','checkpoint_sha256','preserved_Adam_states',
        'scratch_CUDA_Adam_steps','scratch_changed_body_tensors','scratch_frozen_risk_exact']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ('parent','qualification','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);run(p.parse_args())
