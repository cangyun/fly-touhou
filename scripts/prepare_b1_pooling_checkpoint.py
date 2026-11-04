"""Retain every learned tensor/Adam/RNG value; version only inference pooling."""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(a):
    import torch,psutil
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_policy_actions import policy_logits
    from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);torch.set_num_threads(1)
    if a.output.exists():raise FileExistsError(a.output)
    if sha(a.checkpoint)!=a.sha256:raise ValueError('Original parent checkpoint identity differs')
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('CUDA qualification reserve unavailable')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    original=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if original.get('optimizer_transaction_complete') is False:raise ValueError('Partial optimizer transaction')
    legacy=model_from_config(**original['model_config']).cuda().eval().requires_grad_(False)
    legacy.load_state_dict(original['model'],strict=True)
    configuration=deepcopy(original['model_config']);configuration['pair_risk_wrapper'].update(
        aggregation='worst-known-margin-v2',laser_buffer_weight=.1)
    changed=model_from_config(**configuration).cuda().eval().requires_grad_(False)
    changed.load_state_dict(original['model'],strict=True)
    if not all(torch.equal(v.cpu(),original['model'][k]) for k,v in changed.state_dict().items()):
        raise ValueError('Inference migration changed a learned tensor')
    inventory=read(a.inventory);teacher=MovementSafetyTeacher();rows=[]
    if inventory['checkpoint_sha256']!=a.sha256:raise ValueError('Wrong recorded policy')
    with torch.inference_mode():
        for entry in inventory['rows']:
            receipt=read(entry['source_receipt']);path=Path(receipt['arrays'])
            if sha(path)!=entry['source_trace_sha256']:raise ValueError('Sealed recorded state changed')
            with np.load(path,allow_pickle=False) as z:
                x={k:z[k][-1].copy() for k in ('features','objects','maps','hidden','probabilities')}
            obs={k:torch.from_numpy(x[k][None]).cuda() for k in ('features','objects','maps')}
            h=torch.from_numpy(x['hidden'][None,None]).cuda()
            base,_,_=legacy.base.forward_step(obs['features'],obs['maps'],h,objects=obs['objects'])
            old=legacy._adjust(base,obs['features'],obs['objects'])
            new=changed._adjust(base,obs['features'],obs['objects'])
            old_full=policy_logits(old)[0].softmax(-1).cpu().numpy()
            new_full=policy_logits(new)[0].softmax(-1).cpu().numpy()
            error=float(np.abs(old_full-x['probabilities']).max())
            if error>3e-5:raise ValueError('Legacy reference probabilities no longer replay: '+str(error))
            present=obs['objects'][0,:,0]>.5
            known=present&((obs['objects'][0,:,3]>.5)|(obs['objects'][0,:,4]>.5))&(obs['objects'][0,:,26]>.5)
            if not known.any() and not torch.equal(old,new):raise ValueError('No-known-laser policy logits changed')
            bomb_error=abs(float(old_full[18]-new_full[18]))
            if bomb_error>2e-6:raise ValueError('Pooling changed Bomb marginal')
            row=dict(game=entry['game'],stage=entry['stage'],known_lasers=int(known.sum()),
                legacy_probability_replay_error=error,no_known_laser_logits_bitwise_equal=bool(torch.equal(old,new)) if not known.any() else None,
                Bomb_probability_difference=bomb_error,legacy_movement_argmax=int(old[0,:18].argmax()),
                pooling_v2_movement_argmax=int(new[0,:18].argmax()),source_receipt=entry['source_receipt'])
            reason=teacher._uncertainty(x['features'],x['objects'],x['maps'],horizon=4)
            row['geometry_uncertainty']=reason
            if not reason:
                origin=teacher._scene(x['features'],x['objects'])[0]
                margins=teacher.clearances(x['features'],x['objects'],np.repeat(origin[None],18,axis=0),teacher.velocities,duration=4)
                safe=margins>1
                row.update(proxy_buffered_options=int(safe.sum()),critical=bool(safe.any() and not safe.all()),
                    legacy_proxy_unsafe_mass=float(old[0,:18].softmax(-1).cpu().numpy()[~safe].sum()),
                    pooling_v2_proxy_unsafe_mass=float(new[0,:18].softmax(-1).cpu().numpy()[~safe].sum()))
            rows.append(row)
    a.output.mkdir(parents=True)
    metadata=dict(schema='B1-fixed-weight-known-margin-pooling-inference-migration-v1',utc=utc_now(),
        original_checkpoint=str(a.checkpoint),original_sha256=a.sha256,source_runtime=str(ROOT),
        source_manifest_sha256=sha(ROOT/'source_manifest.json'),
        before_model_config=original['model_config'],after_model_config=configuration,
        learned_tensors_unchanged=True,optimizer_moments_unchanged=True,sampling_RNG_unchanged=True,
        optimizer_updates=0,native_worlds_started=0,former_development_scores_do_not_certify_new_operator=True)
    converted=deepcopy(original);converted['model_config']=configuration;converted['inference_migration']=metadata
    config=converted['config'];retired={}
    for key in ('sealed_avoidance_baseline','candidate_initial_checkpoint','candidate_initial_sha256'):
        if key in config:retired[key]=config.pop(key)
    config.update(model_config=configuration,policy_model_config=configuration,source_runtime=str(ROOT),
        source_manifest_sha256=metadata['source_manifest_sha256'],profile=str(ROOT/'profiles/th14_1.00b_original.json'),
        inference_migration=dict(metadata,retired_development_metadata=retired),
        experiment='B1-fixed-weight-known-margin-pooling-v2-effect-only')
    target=a.output/'policy.pt';torch.save(converted,target)
    sealed=torch.load(target,map_location='cpu',weights_only=False)
    def equal(x,y):
        if isinstance(x,torch.Tensor):return isinstance(y,torch.Tensor) and torch.equal(x,y)
        if isinstance(x,np.ndarray):return isinstance(y,np.ndarray) and np.array_equal(x,y)
        if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
        if isinstance(x,(tuple,list)):return type(x)==type(y) and len(x)==len(y) and all(equal(u,v) for u,v in zip(x,y))
        return x==y
    for key in ('model','optimizer','optimizer_parameter_names','torch_rng','cuda_rng','numpy_rng','python_rng'):
        if not equal(sealed[key],original[key]):raise ValueError('Migration failed exact preservation: '+key)
    critical=[r for r in rows if r.get('critical')]
    result=dict(metadata,status='qualified-for-finite-native-effect-test',checkpoint=str(target),checkpoint_sha256=sha(target),
        tensor_keys=len(original['model']),Adam_states=len(original['optimizer']['state']),
        raw_native_observed_events_replayed=len(rows),maximum_legacy_probability_replay_error=max(r['legacy_probability_replay_error'] for r in rows),
        no_known_laser_bitwise_equal_cases=sum(r['no_known_laser_logits_bitwise_equal'] is True for r in rows),
        maximum_Bomb_probability_difference=max(r['Bomb_probability_difference'] for r in rows),
        trusted_critical_proxy_cases=critical,rows=rows,native_efficacy_demonstrated=False,
        script_sha256=sha(__file__))
    write(a.output/'report.json',result);(a.output/'qualification_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:result[k] for k in ('status','checkpoint_sha256','tensor_keys','Adam_states','raw_native_observed_events_replayed',
        'maximum_legacy_probability_replay_error','no_known_laser_bitwise_equal_cases','maximum_Bomb_probability_difference')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
