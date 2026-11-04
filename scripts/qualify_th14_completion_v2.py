"""Full-size synthetic CUDA PPO check. No native process or training job."""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json,utc_now


def main():
    import torch
    from fly_touhou.thrl_model import MOActorCritic
    from fly_touhou.thrl_ppo import RolloutBatch,PPOConfig,make_optimizer,update
    from fly_touhou.th14_completion_protocol import CompletionRewardModel,RECIPE,baseline_parameters,exploration_parameters
    output=ROOT/'artifacts/engineering/completion-v2-20261023/offline-CUDA-v1'
    output.mkdir(exist_ok=False)
    cfg=read_shared_json(ROOT/'artifacts/engineering/mlp-experiment-20261023/branch_M0_config.json')
    torch.set_num_threads(2);torch.manual_seed(11)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    payload=torch.load(cfg['warm_start_checkpoint'],map_location='cpu',weights_only=False)
    model=MOActorCritic(**payload['model_config']).cuda();model.load_state_dict(payload['model'],strict=True)
    assert payload['decisions']==0 and not payload['optimizer']['state']
    recipe=dict(training_recipe=RECIPE,rollout_total_decisions=2560,entropy_decay_decisions=3001344)
    parameters=baseline_parameters(recipe);parameters.update(exploration_parameters(recipe,parameters['entropy_coeff_start']))
    ppo=PPOConfig(**parameters)
    T,N=64,2;features=torch.randn(T,N,20);maps=torch.randn(T,N,16,92,96)
    objects=torch.zeros(T,N,96,21);objects[:,:,0,0]=1;objects[:,:,0,1]=1
    with torch.no_grad():
        logits,values,_=model.forward_step(features.reshape(T*N,20).cuda(),maps.reshape(T*N,16,92,96).cuda(),objects=objects.reshape(T*N,96,21).cuda())
        dist=torch.distributions.Categorical(logits=logits);actions=dist.sample();log_probs=dist.log_prob(actions)
    rewards=torch.zeros(T,N,3);rewards[:,:,0]=.001;rewards[20,0,0]-=1;rewards[36,1,2]-=.1
    terminal=torch.zeros(T,N,dtype=torch.bool);terminal[-1]=True;rewards[-1,:,0]-=2
    starts=torch.zeros(T,N,dtype=torch.bool);starts[0]=True
    rollout=RolloutBatch(features=features,maps=maps,objects=objects,actions=actions.cpu().reshape(T,N),
        old_log_probs=log_probs.cpu().reshape(T,N),values=values.cpu().reshape(T,N,3),rewards=rewards,
        hidden_states=torch.zeros(T,N,256),terminated=terminal,episode_starts=starts,bootstrap_values=torch.zeros(N,3))
    optimizer=make_optimizer(model,ppo);before=model.mlp_core[0].weight.detach().clone()
    metrics=update(model,optimizer,rollout,ppo)
    assert metrics['optimizer_updates']==6 and not torch.equal(before,model.mlp_core[0].weight)
    assert all(torch.isfinite(value).all() for value in model.parameters())
    assert all(math.isfinite(float(v)) for k,v in metrics.items() if isinstance(v,(float,int)))
    write(output/'report.json',dict(status='passed',created_utc=utc_now(),device=torch.cuda.get_device_name(0),
        synthetic_CUDA_PPO_Adam_updates=6,finite_gradients=True,parameters=parameters,
        new_reward=CompletionRewardModel(gamma=ppo.gamma).specification(),
        parameter_counts=model.parameter_counts(),initialization_checkpoint_unchanged=True,
        native_processes_started=0,actual_training_jobs_started=0,learning_gain_claimed=False))
    print(str(output/'report.json'))


if __name__=='__main__':main()
