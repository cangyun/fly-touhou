"""Offline CPU/CUDA qualification and matched M0 initialization; no native EXE."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('baseline-runtime','baseline-config','output','prepared'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False);a.prepared.mkdir(parents=True,exist_ok=False)
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_model import MOActorCritic
    from fly_touhou.thrl_ppo import PPOConfig,RolloutBatch,make_optimizer,update
    import torch
    torch.set_num_threads(2)
    cfg=read_shared_json(a.baseline_config)
    spec=importlib.util.spec_from_file_location('fly_touhou._frozen_gru_model',a.baseline_runtime/'src/fly_touhou/thrl_model.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    kwargs={**cfg['model_overrides'],'attention':cfg['attention'],'action_dim':cfg['policy_action_count']}
    torch.manual_seed(11);old=frozen.MOActorCritic(**kwargs);old_rng=torch.get_rng_state().clone()
    torch.manual_seed(11);new=MOActorCritic(**kwargs)
    assert old.model_config()==new.model_config() and torch.equal(old_rng,torch.get_rng_state())
    assert old.state_dict().keys()==new.state_dict().keys()
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    f=torch.randn(1,20);maps=torch.randn(1,16,92,96);objects=torch.zeros(1,96,21);h=torch.randn(1,1,256)
    with torch.no_grad():
        before=old.forward_step(f,maps,h,objects=objects);after=new.forward_step(f,maps,h,objects=objects)
    for x,y in zip(before,after):torch.testing.assert_close(x,y,rtol=0,atol=0)
    # Native execution and PPO formulas must retain their exact source bytes.
    unchanged=['src/fly_touhou/thrl_ppo.py','src/fly_touhou/thrl_rewards.py',
        'src/fly_touhou/thrl_native_env.py','src/fly_touhou/native_replay.py',
        'src/fly_touhou/frida_controller.py','src/fly_touhou/thrl_geometry_observation.py',
        'src/fly_touhou/thrl_object_transformer.py','src/fly_touhou/hooks/th14_headless.js',
        'src/fly_touhou/hooks/th14_responsive.js']
    for name in unchanged:
        assert (ROOT/name).read_bytes()==(a.baseline_runtime/name).read_bytes(),name
    initial=Path(cfg['warm_start_checkpoint'])
    assert hashlib.sha256(initial.read_bytes()).hexdigest()==cfg['warm_start_sha256']
    original=torch.load(initial,map_location='cpu',weights_only=False)
    assert original['decisions']==0 and original['adam_minibatch_steps']==0 and not original['optimizer']['state']
    kwargs.update(temporal_core='mlp',mlp_hidden_size=768)
    torch.manual_seed(11);mlp=MOActorCritic(**kwargs)
    state=mlp.state_dict();shared=[]
    for name,value in original['model'].items():
        if name.startswith('gru.'):continue
        assert name in state and state[name].shape==value.shape
        state[name]=value.clone();shared.append(name)
    mlp.load_state_dict(state,strict=True)
    assert not any(name.startswith('gru.') for name in state)
    optimizer=torch.optim.Adam(mlp.parameters(),lr=1e-4)
    payload=copy.deepcopy(original)
    payload.update(model_config=mlp.model_config(),model=mlp.state_dict(),optimizer=optimizer.state_dict(),
        optimizer_parameter_names=list(dict(mlp.named_parameters())),curriculum_state=None,
        initializer=dict(temporal_core='mlp',shared_initial_checkpoint=str(initial),
            shared_initial_sha256=cfg['warm_start_sha256'],shared_tensors=len(shared),
            trained_GRU_weights_used=False,MLP_core_random_seed=11,training_started=False))
    payload['config']=copy.deepcopy(cfg)
    payload['config'].update(branch='M0',teacher_pretraining=None,model_overrides={**cfg['model_overrides'],'temporal_core':'mlp','mlp_hidden_size':768})
    checkpoint=a.prepared/'initial_M0.pt';torch.save(payload,checkpoint)
    counts=mlp.parameter_counts()
    assert counts['total']==4981943 and counts['shared_MLP']==492544
    assert all(torch.equal(payload['model'][name],original['model'][name]) for name in shared)
    gpu=dict(available=torch.cuda.is_available(),native_processes_launched=0,training_job_started=False)
    if not gpu['available']:raise RuntimeError('Offline CUDA checks required for this GPU experiment')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    mlp=mlp.cuda();T,N=64,2
    f=torch.randn(T,N,20);maps=torch.randn(T,N,16,92,96);objects=torch.zeros(T,N,96,21)
    objects[:,:,0,0]=1;objects[:,:,0,1]=1
    with torch.no_grad():
        logits,values,_=mlp.forward_step(f.reshape(T*N,20).cuda(),maps.reshape(T*N,16,92,96).cuda(),objects=objects.reshape(T*N,96,21).cuda())
        dist=torch.distributions.Categorical(logits=logits);actions=dist.sample();logs=dist.log_prob(actions)
    starts=torch.zeros(T,N,dtype=torch.bool);starts[0]=True
    rollout=RolloutBatch(features=f,maps=maps,objects=objects,actions=actions.cpu().reshape(T,N),
        old_log_probs=logs.cpu().reshape(T,N),values=values.cpu().reshape(T,N,3),
        rewards=torch.randn(T,N,3),hidden_states=torch.zeros(T,N,256),
        terminated=torch.zeros(T,N,dtype=torch.bool),episode_starts=starts,bootstrap_values=torch.zeros(N,3))
    before=mlp.mlp_core[0].weight.detach().clone();ppo=PPOConfig()
    metrics=update(mlp,make_optimizer(mlp,ppo),rollout,ppo)
    assert metrics['optimizer_updates']==6 and not torch.equal(before,mlp.mlp_core[0].weight)
    assert all(torch.isfinite(p).all() for p in mlp.parameters())
    gpu.update(device=torch.cuda.get_device_name(0),synthetic_decisions=T*N,
        synthetic_Adam_updates=6,finite_gradients=True,GRU_modules=0,
        scope='Synthetic tensors only; no original-game rollout, no learning-efficacy claim')
    write(a.output/'report.json',dict(status='passed',baseline_runtime=str(a.baseline_runtime),
        legacy_GRU_identical_initial_tensors_rng_and_outputs=True,unchanged_native_and_PPO_sources=unchanged,
        shared_initial_tensors=len(shared),parameter_counts=counts,offline_CUDA=gpu,
        initialization=dict(path=str(checkpoint.resolve()),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()),
        training_started=False,learning_gain_claimed=False))
    print(str(a.output/'report.json'))


if __name__=='__main__':main()
