"""Seal a V5 laser parent and qualify one discarded CUDA PPO transaction."""
from pathlib import Path
from dataclasses import replace
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(a):
    import torch
    from torch.distributions import Categorical
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.thrl_laser_migration import laser_parent,EXPANDED_PARAMETER
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_branching import warm_start
    from fly_touhou.thrl_ppo import PPOConfig,RolloutBatch,make_optimizer,update
    from fly_touhou.thrl_policy_actions import policy_logits
    from fly_touhou.thrl_laser_observation import bounded
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    if a.output.exists():raise ValueError('Never overwrite a prepared parent')
    if sha(a.parent)!=a.parent_sha256:raise ValueError('Original PPO parent identity differs')
    sensor=read(a.sensor_qualification);sensor['qualification_sha256']=sha(a.sensor_qualification)
    fit=read(a.fit/'result.json');fit['fit_report_sha256']=sha(a.fit/'result.json')
    if sha(a.fit/'best.pt')!=fit['checkpoint_sha256']:raise ValueError('Fitted laser head identity differs')
    source=torch.load(a.parent,map_location='cpu',weights_only=False)
    laser=torch.load(a.fit/'best.pt',map_location='cpu',weights_only=False)
    payload=laser_parent(torch,source,laser,sensor_evidence=sensor,fit_evidence=fit)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    before=model_from_config(**source['model_config']).cuda();before.load_state_dict(source['model'],strict=True);before.eval()
    model=model_from_config(**payload['model_config']).cuda();model.load_state_dict(payload['model'],strict=True);model.eval()
    cfg=PPOConfig(**source['config']['parameters']);optimizer=make_optimizer(model,cfg)
    a.output.mkdir(parents=True);checkpoint=a.output/'parent.pt';torch.save(payload,checkpoint)
    inherited=warm_start(torch,model,optimizer,checkpoint,sha(checkpoint),exact_optimizer=True,model_factory=model_from_config)
    if inherited['common_Adam_states']!=len(source['optimizer']['state']):raise ValueError('Lost original Adam moments')
    # Check every preserved old tensor/moment exactly, including the expanded
    # projection's old columns and Adam's unmodified historical step counter.
    exact=expanded=0
    for key,value in source['model'].items():
        new=payload['model'][key]
        if key==EXPANDED_PARAMETER:
            if not torch.equal(new[:,:value.shape[1]],value) or new[:,value.shape[1]:].count_nonzero():
                raise ValueError('Shared object projection changed')
            expanded+=1
        elif not torch.equal(new,value):raise ValueError('Shared old state changed: '+key)
        else:exact+=1
    for old_id,state in source['optimizer']['state'].items():
        for name,value in state.items():
            current=payload['optimizer']['state'][old_id][name]
            if isinstance(value,torch.Tensor):
                if current.shape!=value.shape:current=current[:,:value.shape[1]]
                if not torch.equal(current,value):raise ValueError('Shared Adam state changed')
    steps,envs=16,2
    f=torch.zeros(steps,envs,25);f[:,:,0]=.5;f[:,:,1]=.84;f[:,:,2]=.5;f[:,:,4]=.375
    f[:,:,8]=1;f[:,:,13:16]=1/64;f[:,:,21]=1
    maps=torch.zeros(steps,envs,16,92,96);objects=torch.zeros(steps,envs,96,29)
    objects[:,:,0,0:2]=1;objects[:,:,0,6]=20/384;objects[:,:,0,7]=-8/448
    objects[:,:,0,10]=2/128;objects[:,:,0,12:14]=1
    h=torch.zeros(1,envs,model.gru_hidden_size,device='cuda')
    with torch.no_grad():
        old=before.forward_step(f[0].cuda(),maps[0].cuda(),h,objects=objects[0,...,:21].cuda())
        new=model.forward_step(f[0].cuda(),maps[0].cuda(),h,objects=objects[0].cuda())
    for x,y in zip(old,new):torch.testing.assert_close(x,y,rtol=1e-4,atol=4e-5)
    del before
    # Synthetic current laser: y354,width6,length160,observed downward3/frame.
    objects[:,:,1,0]=1;objects[:,:,1,3]=1;objects[:,:,1,6]=0
    objects[:,:,1,7]=float(bounded(-14,448));objects[:,:,1,9]=float(bounded(3,12))
    objects[:,:,1,10]=float(bounded(3,128));objects[:,:,1,12:14]=1
    objects[:,:,1,21]=float(bounded(80,384));objects[:,:,1,26]=1;objects[:,:,1,27]=.2
    # Two actual schema-defined deathbomb-like scratch rows; these states and
    # rewards are synthetic and never claimed as original-game observations.
    f[-2:,:,21]=0;f[-2:,:,24]=1;f[-2:,:,19]=.5
    hidden=[];actions=[];old=[];values=[]
    with torch.no_grad():
        for t in range(steps):
            hidden.append(h[0].cpu().clone())
            z,v,h=model.forward_step(f[t].cuda(),maps[t].cuda(),h,objects=objects[t].cuda())
            distribution=Categorical(logits=policy_logits(z));action=distribution.sample()
            actions.append(action.cpu());old.append(distribution.log_prob(action).cpu());values.append(v.cpu())
        _,bootstrap,_=model.forward_step(f[-1].cuda(),maps[-1].cuda(),h,objects=objects[-1].cuda())
    frozen={name:{k:v.detach().cpu().clone() for k,v in getattr(model,name).state_dict().items()}
        for name in ('risk','laser_risk')}
    body={k:v.detach().cpu().clone() for k,v in model.base.state_dict().items()}
    starts=torch.zeros(steps,envs,dtype=torch.bool);starts[0]=True
    rewards=torch.zeros(steps,envs,3);rewards[:,:,0]=torch.linspace(-.03,.02,steps)[:,None]
    batch=RolloutBatch(features=f,maps=maps,objects=objects,actions=torch.stack(actions),old_log_probs=torch.stack(old),
        values=torch.stack(values),rewards=rewards,hidden_states=torch.stack(hidden),terminated=torch.zeros_like(starts),
        episode_starts=starts,bootstrap_values=bootstrap.cpu(),movement_outcomes=torch.full((steps,envs),-1),
        teacher_targets=torch.full((steps,envs,19),1/19),teacher_weights=torch.ones(steps,envs))
    batch.teacher_weights[::2]=0
    smoke=replace(cfg,seqs_per_batch=2,ppo_epochs=1,target_kl=None,cache_frozen_entity_risk=True,
        deathbomb_auxiliary_coefficient=.05,movement_teacher_coefficient=0,movement_collision_coefficient=.02,
        movement_dense_coefficient=.2,movement_dense_mode='receding-step-v2',online_teacher_coefficient=.1)
    metrics=update(model,optimizer,batch,smoke);torch.cuda.synchronize()
    fixed=all(torch.equal(frozen[name][k],v.detach().cpu()) for name in frozen for k,v in getattr(model,name).state_dict().items())
    changed=sum(not torch.equal(body[k],v.detach().cpu()) for k,v in model.base.state_dict().items())
    if not fixed or not changed or not metrics['optimizer_updates']:raise RuntimeError('Actual CUDA transaction failed')
    result=dict(schema='B1-observed-laser-V5-PPO-parent-CUDA-qualification-v1',status='qualified',
        source_checkpoint=str(a.parent),source_sha256=a.parent_sha256,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        model_config=payload['model_config'],migration=payload['input_migration'],warm_start=inherited,
        sensor_qualification=str(a.sensor_qualification),fitted_risk=str(a.fit),
        device=torch.cuda.get_device_name(0),parameter_counts=model.parameter_counts(),
        exact_shared_state_tensors=exact,expanded_shared_state_tensors=expanded,preserved_Adam_states=inherited['common_Adam_states'],
        inherited_optimized_PPO_decisions=source['lifetime_decisions'],scratch_CUDA_Adam_steps=metrics['optimizer_updates'],
        scratch_changed_body_tensors=changed,scratch_frozen_both_risk_heads_exact=fixed,
        scratch_dense_labels=metrics.get('movement_guidance',{}).get('dense',{}).get('states'),
        scratch_all_observations_and_targets_synthetic=True,scratch_weights_discarded=True,
        parent_saved_before_scratch=True,new_native_instances=0,native_PPO_updates=0,
        no_laser_forward_matches_V4_within_FP32_tolerance=True,already_evasive=False)
    write(a.output/'qualification.json',result)
    print(json.dumps({k:result[k] for k in ['status','device','checkpoint_sha256','preserved_Adam_states',
        'scratch_CUDA_Adam_steps','scratch_changed_body_tensors','scratch_frozen_both_risk_heads_exact','parameter_counts']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ['parent','sensor-qualification','fit','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);run(p.parse_args())
