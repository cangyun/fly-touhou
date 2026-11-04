"""Real-size CUDA gradient/target smoke check on saved native observations.

The input actions are historical demonstration actions, not a fresh on-policy
rollout. This is an engineering check only, with no learning-effect claim.
"""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def run(a):
    import numpy as np,torch
    from fly_touhou.thrl_model import MOActorCritic
    from fly_touhou.thrl_ppo import PPOConfig,RolloutBatch,update,make_optimizer
    from fly_touhou.thrl_policy_actions import policy_logits
    from fly_touhou.atomic_json_io import atomic_write_json as write
    torch.set_num_threads(2);torch.manual_seed(241024)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    p=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=MOActorCritic(**p['model_config']).cuda();model.load_state_dict(p['model']);model.eval()
    keys=('features','objects','maps','actions','dones')+(('targets',) if a.online_reference else ())
    arrays={k:np.load(a.demonstrations/'demonstrations'/f'{k}.npy',mmap_mode='r') for k in keys}
    # Several actual failure neighborhoods, preserving observation/action pairs.
    targets=np.load(a.targets/'targets.npz');t,w=arrays['features'].shape[:2]
    facts=targets['outcomes'].reshape(w,t);starts=[]
    for world in range(4):
        hits=np.flatnonzero(facts[world]==1);s=int(hits[0]-8) if len(hits) else 180
        starts.append(min(t-16,max(0,s)))
    batch={k:torch.from_numpy(np.stack([np.array(v[s:s+16,i]) for i,s in enumerate(starts)],axis=1)) for k,v in arrays.items()}
    outcomes=torch.from_numpy(np.stack([facts[i,s:s+16] for i,s in enumerate(starts)],axis=1))
    cfg=PPOConfig(learning_rate=3e-5,seq_len=16,seqs_per_batch=4,ppo_epochs=1,
        gae_chunk_size=0,target_kl=.03,entropy_coeff_start=.001,entropy_coeff_end=.0001,
        movement_dense_coefficient=.2,movement_dense_max_labels=64,movement_collision_coefficient=.02,
        movement_dense_mode=a.dense_mode,
        online_teacher_coefficient=.1 if a.online_reference else 0.,
        deathbomb_auxiliary_coefficient=.05)
    h=torch.zeros(1,4,model.gru_hidden_size,device='cuda');hs=[];vs=[];logps=[]
    with torch.no_grad():
        for step in range(16):
            hs.append(h[0].cpu());x,v,h=model.forward_step(batch['features'][step].cuda(),batch['maps'][step].cuda(),h,objects=batch['objects'][step].cuda())
            vs.append(v.cpu());logps.append(torch.distributions.Categorical(logits=policy_logits(x)).log_prob(batch['actions'][step].cuda()).cpu())
            h*=~batch['dones'][step].cuda()[None,:,None]
    data=RolloutBatch(features=batch['features'],maps=batch['maps'],objects=batch['objects'],
        actions=batch['actions'],old_log_probs=torch.stack(logps),values=torch.stack(vs),
        rewards=torch.zeros(16,4,3),hidden_states=torch.stack(hs),terminated=batch['dones'],
        episode_starts=torch.cat([torch.ones(1,4,dtype=torch.bool),batch['dones'][:-1].bool()]),
        bootstrap_values=torch.zeros(4,3),movement_outcomes=outcomes,
        teacher_targets=batch['targets'] if a.online_reference else None)
    started=time.monotonic();metrics=update(model,make_optimizer(model,cfg),data,cfg);torch.cuda.synchronize()
    finite=all(torch.isfinite(v).all() for v in model.parameters())
    result=dict(status='passed' if finite else 'failed',device=torch.cuda.get_device_name(0),
        parameter_counts=model.parameter_counts(),metrics=metrics,seconds=time.monotonic()-started,
        cuda_peak_bytes=torch.cuda.max_memory_allocated(),engineering_only=True,
        historical_actions_not_a_fresh_on_policy_rollout=True,game_instances_started=0,native_performance_claimed=False)
    write(a.output,result);print(json.dumps(result))
    if not finite:raise FloatingPointError('Nonfinite CUDA model')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for n in ('checkpoint','demonstrations','targets','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--dense-mode',choices=['legacy-straight-v1','receding-step-v2'],default='legacy-straight-v1')
    p.add_argument('--online-reference',action='store_true')
    run(p.parse_args())
