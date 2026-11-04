"""Preserve real PPO Adam/RNG, add the verified frozen readout, test CUDA PPO."""
from pathlib import Path
from dataclasses import replace
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(a):
    import numpy as np,torch
    from torch.distributions import Categorical
    from fly_touhou.thrl_entity_ppo import training_parent,sealed_avoidance_baseline
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_branching import warm_start
    from fly_touhou.thrl_ppo import PPOConfig,RolloutBatch,make_optimizer,update
    from fly_touhou.thrl_policy_actions import policy_logits
    from fly_touhou.atomic_json_io import atomic_write_json
    if not torch.cuda.is_available():raise RuntimeError('CUDA required, no CPU fallback')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha(a.base)!=a.base_sha256 or sha(a.entity)!=a.entity_sha256:raise ValueError('Verified parent identities changed')
    a.output.mkdir(parents=True,exist_ok=False)
    base=torch.load(a.base,map_location='cpu',weights_only=False)
    entity=torch.load(a.entity,map_location='cpu',weights_only=False)
    payload=training_parent(torch,base,entity)
    payload['entity_PPO_transfer'].update(original_PPO_checkpoint=str(a.base),original_PPO_sha256=sha(a.base),
        verified_entity_checkpoint=str(a.entity),verified_entity_sha256=sha(a.entity))
    checkpoint=a.output/'parent.pt';torch.save(payload,checkpoint)
    cfg=PPOConfig(**base['config']['parameters'])
    model=model_from_config(**payload['model_config']).cuda();optimizer=make_optimizer(model,cfg)
    inherited=warm_start(torch,model,optimizer,checkpoint,sha(checkpoint),exact_optimizer=True,model_factory=model_from_config)
    reference=model_from_config(**entity['model_config']).cuda().eval();reference.load_state_dict(entity['model'])
    report=a.development_report
    baseline=dict(report=str(report),report_sha256=sha(report),checkpoint=str(a.entity),checkpoint_sha256=sha(a.entity),
        condition='trained_sampled',episodes=20,seed=372191,max_frames=16384)
    initial=sealed_avoidance_baseline(torch,model,baseline)
    arrays={k:np.load(a.data/'teacher30-v1/demonstrations'/(k+'.npy'),mmap_mode='r') for k in ['features','maps','objects']}
    raw={k:torch.as_tensor(v[128:144,:2].copy()) for k,v in arrays.items()}
    steps,envs=raw['features'].shape[:2];h=torch.zeros(1,envs,model.gru_hidden_size,device='cuda')
    hidden=[];actions=[];logprobs=[];values=[];max_error=0.
    model.eval()
    with torch.inference_mode():
        for t in range(steps):
            obs={k:v[t].cuda() for k,v in raw.items()};hidden.append(h[0].cpu())
            z,v,hn=model.forward_step(obs['features'],obs['maps'],h,objects=obs['objects'])
            rz,rv,rh=reference.forward_step(obs['features'],obs['maps'],h,objects=obs['objects'])
            for x,y in [(z,rz),(v,rv),(hn,rh)]:
                torch.testing.assert_close(x,y,rtol=0,atol=0)
                finite=torch.isfinite(x)&torch.isfinite(y)
                if finite.any():max_error=max(max_error,float((x[finite]-y[finite]).abs().max()))
            d=Categorical(logits=policy_logits(z));act=d.sample()
            actions.append(act.cpu());logprobs.append(d.log_prob(act).cpu());values.append(v.cpu());h=hn
        _,bootstrap,_=model.forward_step(raw['features'][-1].cuda(),raw['maps'][-1].cuda(),h,objects=raw['objects'][-1].cuda())
    del reference
    old_risk={k:v.detach().cpu().clone() for k,v in model.risk.state_dict().items()}
    old_body={k:v.detach().cpu().clone() for k,v in model.base.state_dict().items()}
    starts=torch.zeros(steps,envs,dtype=torch.bool);starts[0]=True
    reward=torch.zeros(steps,envs,3);reward[:,:,0]=torch.linspace(-.03,.02,steps)[:,None]
    batch=RolloutBatch(**raw,actions=torch.stack(actions),old_log_probs=torch.stack(logprobs),values=torch.stack(values),
        rewards=reward,hidden_states=torch.stack(hidden),terminated=torch.zeros_like(starts),episode_starts=starts,
        bootstrap_values=bootstrap.cpu())
    smoke=replace(cfg,seqs_per_batch=2,ppo_epochs=1,target_kl=None,deathbomb_auxiliary_coefficient=0.,
        movement_teacher_coefficient=0.,movement_collision_coefficient=0.,movement_dense_coefficient=0.,online_teacher_coefficient=0.)
    metrics=update(model,optimizer,batch,smoke)
    torch.cuda.synchronize()
    frozen=all(torch.equal(old_risk[k],v.detach().cpu()) for k,v in model.risk.state_dict().items())
    changed=sum(not torch.equal(old_body[k],v.detach().cpu()) for k,v in model.base.state_dict().items())
    if not frozen or not changed or metrics['optimizer_updates']<1:raise RuntimeError('Frozen readout or actual PPO update qualification failed')
    result=dict(status='qualified',schema='B1-frozen-entity-PPO-CUDA-qualification-v1',device=torch.cuda.get_device_name(0),
        prepared_checkpoint=str(checkpoint),prepared_checkpoint_sha256=sha(checkpoint),model_config=payload['model_config'],
        transfer=payload['entity_PPO_transfer'],warm_start=inherited,baseline_contract=baseline,
        initial_native_development_mean=initial['score'],initial_native_development_episodes=initial['episodes'],
        exact_initial_logits_values_hidden=True,max_initial_error=max_error,smoke_PPO_Adam_steps=metrics['optimizer_updates'],
        smoke_changed_body_tensors=changed,smoke_readout_exact=frozen,native_instances=0,
        smoke_uses_stored_training_observations_and_synthetic_returns=True,smoke_is_not_native_efficacy_or_training_job=True,
        scratch_updates_saved_to_training_parent=False,lifetime_PPO_optimized_decisions=1259520)
    atomic_write_json(a.output/'CUDA-qualification.json',result)
    print(json.dumps({k:result[k] for k in ['status','device','prepared_checkpoint_sha256','initial_native_development_mean','smoke_PPO_Adam_steps','smoke_changed_body_tensors','smoke_readout_exact']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ['base','entity','data','development-report','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--base-sha256',required=True);p.add_argument('--entity-sha256',required=True)
    run(p.parse_args())
