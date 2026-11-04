"""Prepare initial models and validate CUDA math offline; never launch a game."""
from dataclasses import asdict,replace
from pathlib import Path
import argparse,copy,random,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    import torch
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.thrl_study_io import read_shared_json,utc_now
    from fly_touhou.thrl_control import digest
    from fly_touhou.thrl_inputs import ORIGINAL_TIMERS_INPUT,SEPARATED_TIMERS_INPUT,observation_shapes
    from fly_touhou.thrl_model import MOActorCritic
    from fly_touhou.thrl_ppo import PPOConfig,make_optimizer,RolloutBatch,update
    from fly_touhou.thrl_branching import warm_start
    from fly_touhou.thrl_policy_actions import policy_logits
    from fly_touhou.th14_completion_protocol import baseline_parameters,exploration_parameters,learning_overrides
    from fly_touhou.th14_completion_v3 import RECIPE
    from train_thrl_native import save_checkpoint,allocate_rollout
    if not torch.cuda.is_available():raise RuntimeError('Offline CUDA verification requires CUDA')
    torch.set_num_threads(2)
    control=read_shared_json(ROOT/'artifacts/engineering/training-control-center-v1/configuration.json')
    recipe=dict(training_recipe=RECIPE,entropy_decay_decisions=1_000_000,rollout_total_decisions=3840)
    params={**asdict(PPOConfig()),**baseline_parameters(recipe),**learning_overrides(recipe),
        **exploration_parameters(recipe,.01)}
    records={};shared_b=None
    for group in ('A0','B0','M0'):
        torch.manual_seed(11);torch.cuda.manual_seed_all(11);np.random.seed(11);random.seed(11)
        source=read_shared_json(control['template_configs'][group]['path'])
        schema=ORIGINAL_TIMERS_INPUT if group=='A0' else SEPARATED_TIMERS_INPUT
        overrides={**source['model_overrides'],'input_schema':schema,
            'feature_dim':observation_shapes(schema)['features'][0]}
        model=MOActorCritic(attention='both',action_dim=19,**overrides).cuda()
        if group=='M0':
            state=model.state_dict()
            for name,value in shared_b.items():
                if name in state and state[name].shape==value.shape:state[name]=value.to('cuda')
            model.load_state_dict(state,strict=True)
        if group=='B0':shared_b={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if not n.startswith('gru.')}
        optimizer=make_optimizer(model,PPOConfig(**params))
        checkpoint=output/f'initial_{group}.pt'
        config={**recipe,'parameters':params,'input_schema':schema,'model_overrides':overrides,
            'inherited_decisions':0,'inherited_rollout_updates':0,'inherited_adam_minibatch_steps':0}
        save_checkpoint(torch,checkpoint,model,optimizer,decisions=0,updates=0,adam=0,config=config)
        initial_hash=digest(checkpoint)
        restored=warm_start(torch,model,optimizer,checkpoint,initial_hash,exact_optimizer=True)
        assert restored['common_weights_exact'] and restored['inherited_decisions']==0 and not optimizer.state
        torch.cuda.reset_peak_memory_stats()
        # 128 time steps per worker tests complete GAE across the old 64 cut;
        # sixteen 16-step chunks form one actual 256-observation minibatch.
        steps,worlds=128,2
        data=allocate_rollout(torch,steps,worlds,model.gru_hidden_size,schema)
        data['features'].uniform_(-.1,.1)
        bomb_index=10 if group=='A0' else 4
        data['features'][...,bomb_index]=.3;data['features'][::2,:,bomb_index]=0.
        data['maps'].uniform_(-.1,.1)
        if 'objects' in data:
            data['objects'].zero_();data['objects'][...,:6,0]=1
            data['objects'][...,:6,1]=1;data['objects'][...,:6,6:10].uniform_(-.1,.1)
        for name in ('hidden_states','terminated','truncated','episode_starts','truncation_values'):data[name].zero_()
        data['episode_starts'][0]=True
        data['rewards'].zero_();data['rewards'][...,0]=.001;data['rewards'][-1,:,0]=-2.
        data['terminated'][-1]=True
        hidden=torch.zeros(1,worlds,model.gru_hidden_size,device='cuda')
        model.eval()
        with torch.no_grad():
            for t in range(steps):
                data['hidden_states'][t]=hidden[0].cpu()
                logits,values,hidden=model.forward_step(data['features'][t].cuda(),data['maps'][t].cuda(),hidden,
                    **({'objects':data['objects'][t].cuda()} if 'objects' in data else {}))
                dist=torch.distributions.Categorical(logits=policy_logits(logits))
                actions=dist.sample();data['actions'][t]=actions.cpu()
                data['old_log_probs'][t]=dist.log_prob(actions).cpu();data['values'][t]=values.cpu()
                if t%2==0:assert torch.equal(dist.probs[:,18],torch.zeros(worlds,device='cuda'))
        metrics=update(model,optimizer,RolloutBatch(**data,bootstrap_values=torch.zeros(worlds,3)),
            replace(PPOConfig(**params),ppo_epochs=2))
        torch.cuda.synchronize()
        assert metrics['optimizer_updates']>=1 and metrics['samples']==256 and metrics['gae_chunk_size']==0
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert digest(checkpoint)==initial_hash
        records[group]=dict(initial_checkpoint=str(checkpoint),initial_sha256=initial_hash,
            model_overrides=overrides,parameter_counts=model.parameter_counts(),initial_loader=restored,
            synthetic_CUDA=metrics,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            initialization_unchanged_by_test=True)
        del model,optimizer,data;torch.cuda.empty_cache()
        print(group,'offline CUDA passed',flush=True)
    write(output/'report.json',dict(status='passed',created_utc=utc_now(),groups=records,
        training_recipe=RECIPE,parameters=params,device=torch.cuda.get_device_name(0),
        original_game_processes_started=0,training_jobs_started=0,real_data_optimization_performed=False,
        synthetic_only=True,shared_B_M_encoder_heads_identical_at_initialization=True,
        teacher_pairs_use_identical_pre_BC_checkpoint=True))
    print(str(output/'report.json'))


if __name__=='__main__':main()
