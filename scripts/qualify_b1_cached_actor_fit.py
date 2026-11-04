"""CPU scratch equivalence on sealed FP32 inputs and actual recurrent context.

Recorded model actions are numerical exercise labels, not teacher targets.
No updated policy is saved. This is not learning or native performance proof.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    import numpy as np
    import torch
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_policy_actions import policy_logits
    from fly_touhou.thrl_cached_actor_fit import begin_cached_actor_phase, forward_cached_actor
    torch.set_num_threads(1)
    device=torch.device(getattr(args,'device','cpu'))
    if device.type=='cuda':
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<8*1024**3:
            raise RuntimeError('CUDA and8GiB free reserve required for this scratch comparison')
        torch.cuda.set_per_process_memory_fraction(.25)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    atol,rtol=(0.,0.) if device.type=='cpu' else (2e-6,2e-5)
    def synchronize():
        if device.type=='cuda':torch.cuda.synchronize()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = read(args.report)
    if report['status'] != 'completed' or read(args.report.parent/'cleanup.json') != dict(closed=True, close_errors=[]):
        raise ValueError('Completed closed native observation source required')
    checkpoint = Path(report['pins']['trained']['copy'])
    if sha(checkpoint) != report['pins']['trained']['sha256']:
        raise ValueError('Original source checkpoint differs')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    direct = model_from_config(**payload['model_config'])
    direct.load_state_dict(payload['model'], strict=True)
    direct.to(device)
    cached = copy.deepcopy(direct)
    for model in (direct,cached):model.gru.flatten_parameters()
    del payload
    configuration = begin_cached_actor_phase(direct)
    if configuration != begin_cached_actor_phase(cached):
        raise ValueError('Trainable parameter scopes differ')
    selected = [next(r for r in report['episode_results']
                     if r['policy'] == 'trained_greedy' and r['stage'] == stage) for stage in (1, 2)]
    rows, receipts = [], []
    for row in selected:
        receipt_path = Path(row['failure_trace']['receipt']);receipt = read(receipt_path)
        arrays = Path(receipt['arrays_file'])
        if (sha(arrays) != receipt['sha256'] or receipt['checkpoint_sha256'] != sha(checkpoint)
                or receipt.get('earlier_history_full_map_coverage_known') is not True):
            raise ValueError('Actual FP32 native sequence identity differs')
        with np.load(arrays, allow_pickle=False) as z:
            item = {key:z[key][-16:].copy() for key in ('features','objects','maps','action','probabilities')}
            item['hidden'] = z['hidden'][-16].copy()
        rows.append(item);receipts.append(dict(path=str(receipt_path),sha256=sha(receipt_path)))
    tensors = {key:torch.from_numpy(np.stack([r[key] for r in rows])).to(device)
               for key in ('features','objects','maps')}
    hidden = torch.from_numpy(np.stack([r['hidden'] for r in rows]))[None].contiguous().to(device)
    dones = torch.zeros((2,16), dtype=torch.bool,device=device)
    actions = torch.from_numpy(np.stack([r['action'] for r in rows])).long().reshape(-1).to(device)
    with torch.no_grad():
        flat = {key:value.reshape(-1,*value.shape[2:]) for key,value in tensors.items()}
        encoded = cached._encode(flat['features'],flat['maps'],flat['objects']).reshape(2,16,-1)
        risk = cached.precompute_frozen_risk(flat['features'],flat['objects']).reshape(2,16,18)
    frozen = {name:value.clone() for name,value in direct.state_dict().items()
              if not name.startswith(('base.gru.','base.actor_trunk1.','base.actor_trunk2.','base.actor_head.'))}
    direct_optimizer = torch.optim.Adam([p for p in direct.parameters() if p.requires_grad],lr=1e-4)
    cached_optimizer = torch.optim.Adam([p for p in cached.parameters() if p.requires_grad],lr=1e-4)
    synchronize();started=time.perf_counter()
    logits,_ = direct.forward_sequence(tensors['features'],tensors['maps'],hidden,dones,objects=tensors['objects'])
    direct_loss=torch.nn.functional.cross_entropy(policy_logits(logits),actions)
    direct_loss.backward();synchronize();direct_seconds=time.perf_counter()-started
    actual = torch.from_numpy(np.stack([r['probabilities'] for r in rows])).reshape(-1,18).to(device)
    replay_error=float((logits[:,:18].softmax(-1)-actual).abs().max().detach())
    if replay_error>3e-5:
        raise ValueError('Actual recorded policy sequence does not replay: '+str(replay_error))
    synchronize();started=time.perf_counter()
    other=forward_cached_actor(cached,encoded,tensors['features'],risk,hidden,dones)
    cached_loss=torch.nn.functional.cross_entropy(policy_logits(other),actions)
    cached_loss.backward();synchronize();cached_seconds=time.perf_counter()-started
    torch.testing.assert_close(other,logits,atol=atol,rtol=rtol)
    torch.testing.assert_close(cached_loss,direct_loss,atol=atol,rtol=rtol)
    max_gradient_error=0.
    for (name,p),(other_name,q) in zip(direct.named_parameters(),cached.named_parameters()):
        if name!=other_name:
            raise ValueError('Model parameter layout differs')
        if p.requires_grad:
            if p.grad is None or q.grad is None or not torch.isfinite(p.grad).all():
                raise ValueError('Expected finite actor/recurrent gradients absent')
            torch.testing.assert_close(q.grad,p.grad,atol=atol,rtol=rtol)
            max_gradient_error=max(max_gradient_error,float((p.grad-q.grad).abs().max()))
        elif p.grad is not None or q.grad is not None:
            raise ValueError('Frozen module received optimizer gradients')
    direct_optimizer.step();cached_optimizer.step()
    max_parameter_error=0.
    for name,value in direct.state_dict().items():
        compare=cached.state_dict()[name]
        torch.testing.assert_close(compare,value,atol=atol,rtol=rtol)
        max_parameter_error=max(max_parameter_error,float((compare-value).abs().max()))
        if name in frozen and (not torch.equal(value,frozen[name]) or not torch.equal(compare,frozen[name])):
            raise ValueError('Frozen encoder/risk/critic tensors changed')
    if device.type=='cpu' and torch.cuda.is_initialized():
        raise RuntimeError('Do not initialize CUDA during the independent native test')
    result=dict(schema='B1-cached-encoder-temporary-actor-fit-qualification-v2',utc=utc_now(),status='passed',
        device=str(device),actual_device_name=torch.cuda.get_device_name() if device.type=='cuda' else 'CPU',
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),source_receipts=receipts,
        actual_native_states=32,actual_recurrent_context=True,configuration=configuration,
        actual_recorded_conditional_probability_replay_error=replay_error,
        maximum_actor_gradient_error=max_gradient_error,maximum_post_Adam_parameter_error=max_parameter_error,
        direct_loss=float(direct_loss.detach()),cached_loss=float(cached_loss.detach()),
        single_device_exercise_seconds=dict(direct=direct_seconds,cached=cached_seconds),
        encoding_and_risk_precomputation_excluded_from_timing=True,timing_not_whole_job_speed=True,
        frozen_encoder_risk_and_critic_tensors_exact=True,scratch_Adam_per_copy=1,
        original_stepwise_GRU_order_retained=True,declared_atol=atol,declared_rtol=rtol,
        main_optimizer_updates=0,native_worlds_started=0,CUDA_initialized=torch.cuda.is_initialized(),
        scratch_updated_weights_saved=False,recorded_actions_used_as_numerical_exercise_not_teacher_labels=True,
        test_observations_not_used_for_deployable_fitting=True,
        GPU_equivalence_not_yet_qualified=device.type!='cuda',full_data_transfer_or_efficacy_not_claimed=True,
        source_hashes={name:sha(ROOT/'src/fly_touhou'/name) for name in
            ('thrl_cached_actor_fit.py','thrl_pair_policy.py','thrl_model.py')})
    args.output.mkdir(parents=True)
    write(args.output/'report.json',result)
    (args.output/'qualification_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key:value for key,value in result.items() if key not in ('source_receipts','source_hashes')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    run(p.parse_args())
