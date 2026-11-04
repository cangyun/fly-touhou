"""Compare accepted V5 inputs and test discarded exact CUDA recovery.

This launches no original game and never saves the scratch-trained weights.
Recorded full native observations and controlled curve fixtures are separate.
"""
from pathlib import Path
from dataclasses import replace
import argparse,gzip,hashlib,json,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(args):
    sys.path.insert(0,str(args.runtime/'src'))
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.state import Snapshot
    from fly_touhou.thrl_inputs import make_encoder,SEPARATED_LASER_INPUT
    results=[]
    for case in read(args.dataset)['cases']:
        encoder=make_encoder(SEPARATED_LASER_INPUT)
        hashes=[]
        for raw in case['snapshots']:
            observation=encoder.encode(Snapshot.from_dict(raw))
            hashes.append({key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in observation.items()})
        results.append(dict(name=case['name'],hashes=hashes))
    write(args.output,dict(runtime=str(args.runtime),manifest_sha256=sha(args.runtime/'source_manifest.json'),cases=results))


def run(args):
    sys.path.insert(0,str(args.runtime/'src'));sys.path.insert(1,str(ROOT/'tests'))
    from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.state import Snapshot
    from test_thrl_laser_input_v5 import current,laser
    import numpy as np
    if args.output.exists():raise FileExistsError(args.output)
    if sha(args.checkpoint)!=args.sha256:raise ValueError('Checkpoint identity differs')
    args.output.mkdir(parents=True)
    sensor=read(args.sensor_qualification);cases=[]
    for row in sensor['snapshot_results']:
        path=Path(row['path'])
        if sha(path)!=row['sha256']:raise ValueError('Raw original observation changed')
        raw=json.loads(gzip.decompress(path.read_bytes()))
        cases.append(dict(name='recorded-native-'+str(len(cases)),snapshots=[raw['snapshot']]))
    for i in range(64):
        count=2+i%7
        nodes=tuple((float(j*10-50),float(330+(j%2)*12)) for j in range(count))
        if i%8==0:nodes=tuple(nodes[0] for _ in nodes)
        value=laser(kind=2,nodes=nodes,width=float((i%4)*4),active=bool(i%2),native_iframes=i%5)
        later=replace(value,age=104,nodes=tuple((x+2,y+1) for x,y in nodes))
        cases.append(dict(name='controlled-accepted-curve-'+str(i),snapshots=[current(value).to_dict(),current(later,frame=104).to_dict()]))
    for i in range(32):
        value=laser(angle=(i-16)*.13,width=float(i%8),length=float(i*23),active=bool(i%2))
        later=replace(value,age=104,x=value.x+4,length=value.length+12)
        cases.append(dict(name='controlled-accepted-line-'+str(i),snapshots=[current(value).to_dict(),current(later,frame=104).to_dict()]))
    dataset=args.output/'accepted_cases.json';write(dataset,dict(cases=cases))
    for label,runtime in [('reference',args.reference_runtime),('repair',args.runtime)]:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--encode-only','--runtime',str(runtime),
                        '--dataset',str(dataset),'--output',str(args.output/(label+'.json'))],check=True,capture_output=True,text=True)
    reference=read(args.output/'reference.json');repair=read(args.output/'repair.json')
    if reference['cases']!=repair['cases']:raise ValueError('Previously accepted input bytes changed')
    import torch,random
    from torch.distributions import Categorical
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_inputs import make_encoder,SEPARATED_LASER_INPUT
    from fly_touhou.thrl_branching import warm_start
    from fly_touhou.thrl_ppo import PPOConfig,RolloutBatch,make_optimizer,update
    from fly_touhou.thrl_policy_actions import policy_logits
    if not torch.cuda.is_available():raise RuntimeError('CUDA required;no fallback')
    torch.set_num_threads(2)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if payload.get('optimizer_transaction_complete') is not True:raise ValueError('Partial optimizer transaction')
    model=model_from_config(**payload['model_config']).cuda()
    cfg=PPOConfig(**payload['config']['parameters']);optimizer=make_optimizer(model,cfg)
    audit=warm_start(torch,model,optimizer,args.checkpoint,args.sha256,exact_optimizer=True,model_factory=model_from_config)
    restored=optimizer.state_dict();source=payload['optimizer']
    assert restored['param_groups']==source['param_groups']
    for key,state in source['state'].items():
        for field,value in state.items():
            actual=restored['state'][key][field]
            assert torch.equal(actual.cpu(),value.cpu()) if isinstance(value,torch.Tensor) else actual==value
    assert torch.equal(torch.get_rng_state(),payload['torch_rng'])
    assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),payload['cuda_rng']))
    nr=np.random.get_state();saved=payload['numpy_rng']
    assert nr[0]==saved[0] and np.array_equal(nr[1],saved[1]) and nr[2:]==saved[2:]
    assert random.getstate()==payload['python_rng']
    if audit['common_Adam_states']!=114 or not audit['common_weights_exact']:raise ValueError('Recovery lost shared optimizer/model state')
    observations=[make_encoder(SEPARATED_LASER_INPUT).encode(current(laser(kind=2,nodes=((0.,368.),)))),
                  make_encoder(SEPARATED_LASER_INPUT).encode(current())]
    steps,envs=16,2
    arrays={}
    for key in observations[0]:
        values=torch.from_numpy(np.stack([row[key] for row in observations]))
        arrays[key]=values[None].repeat((steps,)+(1,)*values.ndim)
    f,m,o=arrays['features'],arrays['maps'],arrays['objects']
    hidden=torch.zeros(1,envs,model.gru_hidden_size,device='cuda');hs=[];actions=[];old=[];values=[]
    with torch.no_grad():
        for t in range(steps):
            hs.append(hidden[0].cpu().clone())
            logits,value,hidden=model.forward_step(f[t].cuda(),m[t].cuda(),hidden,objects=o[t].cuda())
            dist=Categorical(logits=policy_logits(logits));action=dist.sample()
            actions.append(action.cpu());old.append(dist.log_prob(action).cpu());values.append(value.cpu())
        _,bootstrap,_=model.forward_step(f[-1].cuda(),m[-1].cuda(),hidden,objects=o[-1].cuda())
    frozen={name:{k:v.cpu().clone() for k,v in getattr(model,name).state_dict().items()} for name in ('risk','laser_risk')}
    body={k:v.detach().cpu().clone() for k,v in model.base.state_dict().items()}
    starts=torch.zeros(steps,envs,dtype=torch.bool);starts[0]=True
    rewards=torch.zeros(steps,envs,3);rewards[:,:,0]=torch.linspace(-.03,.02,steps)[:,None]
    weights=torch.ones(steps,envs);weights[:,0]=0  # Curve future is unknown;no invented teacher action.
    batch=RolloutBatch(features=f,maps=m,objects=o,actions=torch.stack(actions),old_log_probs=torch.stack(old),
        values=torch.stack(values),rewards=rewards,hidden_states=torch.stack(hs),terminated=torch.zeros_like(starts),
        episode_starts=starts,bootstrap_values=bootstrap.cpu(),movement_outcomes=torch.full((steps,envs),-1),
        teacher_targets=torch.full((steps,envs,19),1/19),teacher_weights=weights)
    metrics=update(model,optimizer,batch,replace(cfg,seqs_per_batch=2,ppo_epochs=1,target_kl=None))
    torch.cuda.synchronize()
    fixed=all(torch.equal(v.cpu(),frozen[name][k]) for name in frozen for k,v in getattr(model,name).state_dict().items())
    changed=sum(not torch.equal(v.cpu(),body[k]) for k,v in model.base.state_dict().items())
    if not fixed or not changed or not metrics['optimizer_updates']:raise ValueError('Discarded CUDA transaction failed')
    result=dict(schema='B1-curve-domain-extension-exact-recovery-qualification-v1',utc=utc_now(),status='qualified',
        checkpoint=str(args.checkpoint),checkpoint_sha256=args.sha256,candidate_optimized_decisions=payload['candidate_optimized_decisions'],
        reference_runtime=str(args.reference_runtime),repair_runtime=str(args.runtime),
        reference_manifest_sha256=reference['manifest_sha256'],repair_manifest_sha256=repair['manifest_sha256'],
        recorded_full_native_snapshots=len(sensor['snapshot_results']),controlled_accepted_sequences=96,
        accepted_cases=len(cases),all_formerly_accepted_modality_bytes_exact=True,
        exact_optimizer_state_tensors=True,exact_four_sampling_RNG_states=True,warm_start=audit,
        scratch_CUDA_Adam_steps=metrics['optimizer_updates'],scratch_changed_body_tensors=changed,
        scratch_frozen_both_risk_heads_exact=fixed,scratch_weights_discarded=True,
        scratch_inputs_controlled=True,new_native_instances=0,native_PPO_updates=0,
        native_single_node_confirmation_required_separately=True,checkpoint_unchanged=sha(args.checkpoint)==args.sha256)
    write(args.output/'report.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ['runtime','reference-runtime','sensor-qualification','checkpoint','dataset','output']:p.add_argument('--'+key,type=Path)
    p.add_argument('--sha256');p.add_argument('--encode-only',action='store_true');args=p.parse_args()
    encode(args) if args.encode_only else run(args)
