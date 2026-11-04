"""Scratch CUDA equivalence of repeated versus cached frozen-risk PPO.

Uses saved training-partition observations and synthetic returns. No original
game starts, no updated weights are saved, and this is not policy efficacy.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import copy
import hashlib
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def run(args):
    import numpy as np
    import torch
    from torch.distributions import Categorical
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.thrl_study_io import utc_now
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_policy_actions import policy_logits
    from fly_touhou.thrl_ppo import PPOConfig,RolloutBatch,make_optimizer,update
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA qualification required')
    if torch.cuda.mem_get_info()[0]<8*1024**3:
        raise RuntimeError('Insufficient CUDA reserve for a scratch qualification')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    sha=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if sha!=args.sha256:
        raise ValueError('Immutable source checkpoint hash differs')
    args.output.mkdir(parents=True,exist_ok=False)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    original=model_from_config(**payload['model_config']).cuda()
    original.load_state_dict(payload['model'],strict=True)
    if not original.frozen_readout:
        raise ValueError('Only the fitted frozen readout qualifies')
    cfg=PPOConfig(**payload['config']['parameters'])
    cfg=replace(cfg,seq_len=16,seqs_per_batch=16,ppo_epochs=3,target_kl=10.,
        deathbomb_auxiliary_coefficient=0.,movement_teacher_coefficient=0.,
        movement_collision_coefficient=0.,movement_dense_coefficient=0.,
        online_teacher_coefficient=0.,cache_frozen_entity_risk=False)
    root=args.data/'teacher30-v1/demonstrations'
    seeds=np.load(root/'native_seeds.npy',mmap_mode='r')[128:144,:16]
    if seeds.shape!=(16,16) or not np.all(seeds%10<6):
        raise ValueError('Qualification input must remain in the native training partition')
    arrays={k:np.load(root/(k+'.npy'),mmap_mode='r') for k in ('features','objects','maps')}
    raw={k:torch.as_tensor(np.asarray(v[128:144,:16],dtype=np.float32).copy()) for k,v in arrays.items()}
    if raw['features'].shape!=(16,16,25) or raw['objects'].shape!=(16,16,96,21):
        raise ValueError('Saved training observations do not match the fixed B1 sensor')
    hidden=torch.zeros(1,16,original.gru_hidden_size,device='cuda')
    states=[];actions=[];logprobs=[];values=[]
    original.eval()
    with torch.no_grad():
        for tick in range(16):
            states.append(hidden[0].cpu())
            logits,value,hidden=original.forward_step(raw['features'][tick].cuda(),raw['maps'][tick].cuda(),
                hidden,objects=raw['objects'][tick].cuda())
            distribution=Categorical(logits=policy_logits(logits))
            action=distribution.sample()
            actions.append(action.cpu());logprobs.append(distribution.log_prob(action).cpu());values.append(value.cpu())
    starts=torch.zeros(16,16,dtype=torch.bool);starts[0]=True
    rewards=torch.zeros(16,16,3);rewards[:,:,0]=torch.linspace(-.03,.02,16)[:,None]
    batch=RolloutBatch(**raw,actions=torch.stack(actions),old_log_probs=torch.stack(logprobs),
        values=torch.stack(values),rewards=rewards,hidden_states=torch.stack(states),
        terminated=torch.zeros_like(starts),episode_starts=starts,bootstrap_values=torch.zeros(16,3))
    cached=copy.deepcopy(original)
    for model in (original,cached):
        model.gru.flatten_parameters()
    aopt,bopt=make_optimizer(original,cfg),make_optimizer(cached,cfg)
    aopt.load_state_dict(payload['optimizer']);bopt.load_state_dict(copy.deepcopy(payload['optimizer']))
    risk={k:v.detach().cpu().clone() for k,v in original.risk.state_dict().items()}
    torch.cuda.synchronize()
    seed=torch.get_rng_state().clone();gpu_seeds=[value.clone() for value in torch.cuda.get_rng_state_all()]
    started=time.perf_counter();a=update(original,aopt,batch,cfg);torch.cuda.synchronize();direct_seconds=time.perf_counter()-started
    after_cpu=torch.get_rng_state().clone();after_gpu=[value.clone() for value in torch.cuda.get_rng_state_all()]
    torch.set_rng_state(seed);torch.cuda.set_rng_state_all(gpu_seeds)
    started=time.perf_counter();b=update(cached,bopt,batch,replace(cfg,cache_frozen_entity_risk=True));torch.cuda.synchronize();cached_seconds=time.perf_counter()-started
    rng_exact=torch.equal(after_cpu,torch.get_rng_state()) and all(torch.equal(x,y) for x,y in zip(after_gpu,torch.cuda.get_rng_state_all()))
    max_parameter_error=0.
    for name,value in original.state_dict().items():
        other=cached.state_dict()[name]
        torch.testing.assert_close(other,value,atol=2e-6,rtol=2e-5)
        if value.is_floating_point():max_parameter_error=max(max_parameter_error,float((other-value).abs().max()))
    max_Adam_error=0.
    for key,state in aopt.state_dict()['state'].items():
        for name,value in state.items():
            other=bopt.state_dict()['state'][key][name]
            torch.testing.assert_close(other,value,atol=1e-7,rtol=2e-5)
            if value.is_floating_point():max_Adam_error=max(max_Adam_error,float((other-value).abs().max()))
    frozen=all(torch.equal(risk[k],v.detach().cpu()) for model in (original,cached) for k,v in model.risk.state_dict().items())
    if not frozen or not rng_exact or a['optimizer_updates']!=b['optimizer_updates']:
        raise RuntimeError('Frozen readout, sampling RNG or update-count equivalence failed')
    result=dict(status='qualified',utc=utc_now(),schema='B1-CUDA-frozen-risk-cache-qualification-v1',
        device=torch.cuda.get_device_name(),source_checkpoint=str(args.checkpoint),source_sha256=sha,
        native_instances_started=0,source_training_samples=256,source_native_seeds=sorted(set(map(int,seeds.ravel()))),
        synthetic_returns=True,scratch_updates_saved=False,main_optimizer_updates=0,
        scratch_Adam_per_copy=a['optimizer_updates'],risk_exact=frozen,RNG_exact=rng_exact,
        max_parameter_error=max_parameter_error,max_Adam_state_error=max_Adam_error,
        direct_seconds=direct_seconds,cached_seconds=cached_seconds,
        timing_is_single_nonisolated_pair=True,isolated_speed_acceptance=False,
        both_GRU_parameter_storage_compacted=True,
        input_sample_time_range=[128,144],input_sample_world_range=[0,16],
        source_hashes={name:hashlib.sha256((ROOT/'src/fly_touhou'/name).read_bytes()).hexdigest()
            for name in ('thrl_ppo.py','thrl_pair_policy.py')},
        risk_cache=b['frozen_entity_risk_cache'],native_policy_efficacy_claimed=False)
    write(args.output/'report.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
