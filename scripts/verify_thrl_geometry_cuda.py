"""Real native sensor -> 4-layer attention/GRU -> original PPO CUDA update."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--executable',required=True,type=Path);p.add_argument('--profile',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--steps',type=int,default=128)
    p.add_argument('--ready',action='store_true')
    p.add_argument('--input-schema',choices=('original273-v1','semantic-separated-v1'),default='semantic-separated-v1')
    p.add_argument('--teacher',action='store_true',help='Also qualify fresh schema-matched teacher BC and PPO transfer')
    p.add_argument('--start-bank',type=Path,help='Qualify the production mixture of verified prefixes and full Story starts')
    a=p.parse_args()
    if not 1<=a.workers<=8:raise ValueError('Functional run limited to 1..8 workers')
    if a.steps<32 or a.steps>512 or a.steps*a.workers%64:raise ValueError('Whole PPO minibatches required')
    a.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    from torch.distributions import Categorical
    from fly_touhou.thrl_inputs import observation_shapes, forward_observation
    from fly_touhou.thrl_model import MOActorCritic
    from fly_touhou.thrl_ppo import PPOConfig,RolloutBatch,make_optimizer,update
    from fly_touhou.thrl_vector import THRLNativeVector
    from train_thrl_native import allocate_rollout
    report={'schema':'thrl-geometry-cuda-functional-v1','status':'running','learning_efficacy_test':False}
    vector=None
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
        torch.set_num_threads(2)
        torch.manual_seed(11022)
        model=MOActorCritic(feature_dim=observation_shapes(a.input_schema)['features'][0],action_dim=19,attention='both',
            object_embed_dim=256,object_attention_heads=8,object_attention_layers=4,
            spatial_attention_layers=2,input_schema=a.input_schema).cuda()
        cfg=PPOConfig(ppo_epochs=1)
        optimizer=make_optimizer(model,cfg)
        if a.teacher:
            from dataclasses import asdict
            import hashlib
            from train_thrl_native import save_checkpoint
            from fly_touhou.thrl_teacher_training import pretrain_teacher_branch
            from fly_touhou.thrl_branching import warm_start
            initial=a.output/'qualification_initial.pt'
            save_checkpoint(torch,initial,model,optimizer,decisions=0,updates=0,adam=0,
                            config={'parameters':asdict(cfg)})
            config={'attention':'both','policy_action_count':19,'input_schema':a.input_schema,
                'model_overrides':{k:v for k,v in model.model_config().items() if k not in ('attention','action_dim')},
                'warm_start_checkpoint':str(initial),'warm_start_sha256':hashlib.sha256(initial.read_bytes()).hexdigest(),
                'game_exe':str(a.executable),'profile':str(a.profile),'seed':11,'headless':True,
                'match_PPO_initial_RNG':True,
                'teacher_pretraining':{'samples':512,'workers':2,'epochs':1,'seq_len':16,
                    'seqs_per_batch':4,'learning_rate':.0001,'max_grad_norm':.5}}
            if a.start_bank:
                config.update(start_bank=str(a.start_bank),full_story_fraction=.25)
            report['teacher']=pretrain_teacher_branch(torch,config,a.output,on_progress=lambda row:None,check_stop=lambda:None)
            report['teacher_transfer']=warm_start(torch,model,optimizer,
                report['teacher']['checkpoint'],report['teacher']['checkpoint_sha256'])
        vector=THRLNativeVector([dict(executable=str(a.executable),profile_path=str(a.profile),
            private_appdata=str(a.output/'native'/str(i)/'userdata'),policy_action_count=19,
            headless=True,binary_reader=True,native_seed=12345+i,input_schema=a.input_schema,
            **({'start_bank':str(a.start_bank),'full_story_fraction':.25,'curriculum_seed':11+1009*i} if a.start_bank else {}))
            for i in range(a.workers)],a.output/'workers')
        observation,infos=vector.reset()
        data=allocate_rollout(torch,a.steps,a.workers,model.gru_hidden_size,a.input_schema)
        hidden=torch.zeros(1,a.workers,model.gru_hidden_size,device='cuda')
        starts=torch.ones(a.workers,dtype=torch.bool)
        begun=time.perf_counter()
        for t in range(a.steps if not a.ready else 0):
            for key in observation:
                data[key][t].copy_(torch.from_numpy(observation[key]))
            data['hidden_states'][t].copy_(hidden[0].cpu())
            data['episode_starts'][t].copy_(starts)
            with torch.no_grad():
                logits,values,next_hidden=forward_observation(torch,model,
                    {key:data[key][t] for key in observation},hidden)
                distribution=Categorical(logits=logits.clamp(-cfg.logit_clamp,cfg.logit_clamp))
                actions=distribution.sample()
                data['actions'][t].copy_(actions.cpu());data['values'][t].copy_(values.cpu())
                data['old_log_probs'][t].copy_(distribution.log_prob(actions).cpu())
            observation,_,terminated,truncated,infos=vector.step(actions.cpu().numpy())
            if truncated.any():raise RuntimeError('Unexpected functional-run truncation')
            data['rewards'][t].copy_(torch.as_tensor(np.stack([i['reward_vector'] for i in infos])))
            data['terminated'][t].copy_(torch.from_numpy(terminated));data['truncated'][t].zero_()
            if terminated.any():
                vector.reset_at(np.flatnonzero(terminated).tolist());observation=vector.observation
            hidden=next_hidden*(~torch.as_tensor(terminated,device='cuda'))[None,:,None]
            starts=torch.from_numpy(terminated)
        if a.ready:
            from fly_touhou.thrl_ready_collection import collect_ready
            observation,hidden,starts,infos=collect_ready(torch,model,vector,data,hidden,starts,infos,
                logit_clamp=cfg.logit_clamp,check_stop=lambda:None,on_step=lambda *args:None)
        collect=time.perf_counter()-begun
        with torch.no_grad():
            _,bootstrap,_=forward_observation(torch,model,observation,hidden)
        before=model.actor_head.weight.detach().clone()
        begun=time.perf_counter()
        metrics=update(model,optimizer,RolloutBatch(**data,bootstrap_values=bootstrap.cpu()),cfg)
        torch.cuda.synchronize()
        states=[s for s in optimizer.state.values() if 'exp_avg' in s]
        if not states or any(s['exp_avg'].device.type!='cuda' for s in states):
            raise RuntimeError('CUDA Adam state missing')
        if torch.equal(before,model.actor_head.weight):raise RuntimeError('PPO did not change actor parameters')
        report.update(status='passed',device=torch.cuda.get_device_name(),workers=a.workers,
            start_bank=str(a.start_bank) if a.start_bank else None,
            input_schema=a.input_schema,observation_shapes={key:list(value.shape[1:]) for key,value in observation.items()},
            policy_decisions=a.steps*a.workers,collector='ready' if a.ready else 'synchronous',
            collect_seconds=collect,update_seconds=time.perf_counter()-begun,
            metrics=metrics,adam_tensors_on_cuda=len(states),actor_weights_changed=True,
            model_config=model.model_config(),parameter_counts=model.parameter_counts(),
            attention=model.attention_specification(),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            final_workers=[{k:i.get(k) for k in ('pid','stage','stage_frame','lives','bombs','native_execution','native_binary_reader','prefix_reset')} for i in infos])
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if vector is not None:vector.close()
        (a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
