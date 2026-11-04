"""Compare frozen B1 policies on the exact same stored BC-held trajectories.

No game, optimizer or guidance modification. Geometry is a proxy, not native
counterfactual truth. Existing BC-held states are not a fresh native test set.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from evaluate_thrl_avoidance_probe import checkpoint_copy


def held_worlds(split,offset,worlds,length,limit):
    groups={}
    if set(map(tuple,split['train']))&set(map(tuple,split['heldout'])):
        raise ValueError('BC train/held descriptors overlap')
    for world,start in split['heldout']:
        if offset<=world<offset+worlds:
            if start<0 or start+16>length:raise ValueError('Held descriptor exceeds native prefix')
            mask=groups.setdefault(world-offset,np.zeros(length,bool));mask[start:start+16]=True
    return [(world,groups[world]) for world in sorted(groups)[:limit]]


def recurrent_prefix(torch,model,encoded,dones):
    if dones.shape!=(len(encoded),):raise ValueError('Reset clock does not match prefix')
    cuts=[0,*list(np.flatnonzero(dones[:-1])+1),len(encoded)]
    return torch.cat([model.gru(encoded[a:b,None])[0][:,0] for a,b in zip(cuts,cuts[1:])])


def state_statistics(logits,targets,margins,trusted,actions,outcomes,held):
    import torch
    from fly_touhou.thrl_policy_actions import policy_logits
    x=policy_logits(logits);device=x.device
    y=torch.as_tensor(targets,device=device,dtype=x.dtype)
    legal=torch.isfinite(x);y=y*legal;y=y/y.sum(-1,keepdim=True)
    logp=x.log_softmax(-1).masked_fill(~legal,0.)
    ce=-(y*logp).sum(-1);entropy=-(y*y.clamp_min(1e-20).log()).sum(-1)
    mask=torch.as_tensor(held,device=device);p=x[:,:18].softmax(-1)
    safe=torch.as_tensor(margins[:,0]>1,device=device)
    valid=torch.as_tensor(trusted[:,0],device=device)
    critical=mask&valid&safe.any(-1)&~safe.all(-1)
    risk=(p*~safe).sum(-1);top=x[:,:18].argmax(-1)
    bad_top=~safe.gather(1,top[:,None])[:,0]
    a=torch.as_tensor(actions,device=device);actual_hit=torch.as_tensor(outcomes==1,device=device)&mask
    if ((a[actual_hit]<0)|(a[actual_hit]>=18)).any():raise ValueError('Actual collision action must be movement')
    chosen=p[actual_hit].gather(1,a[actual_hit,None])[:,0]
    return dict(held_states=int(mask.sum()),teacher_CE_sum=float(ce[mask].sum()),
        teacher_KL_sum=float((ce-entropy)[mask].sum()),
        teacher_argmax_agreement_sum=int((x.argmax(-1)[mask]==y.argmax(-1)[mask]).sum()),
        critical4_states=int(critical.sum()),unsafe4_mass_sum=float(risk[critical].sum()),
        unsafe4_argmax_count=int(bad_top[critical].sum()),actual_observed_hit_states=int(actual_hit.sum()),
        executed_hit_action_conditional_probability_sum=float(chosen.sum()))


def mean_statistics(rows):
    total={key:sum(row[key] for row in rows) for key in rows[0]}
    def ratio(n,d):return total[n]/total[d] if total[d] else None
    return dict(**total,teacher_CE=ratio('teacher_CE_sum','held_states'),
        teacher_KL=ratio('teacher_KL_sum','held_states'),
        teacher_argmax_agreement=ratio('teacher_argmax_agreement_sum','held_states'),
        modeled_unsafe4_movement_mass=ratio('unsafe4_mass_sum','critical4_states'),
        modeled_unsafe4_movement_argmax_rate=ratio('unsafe4_argmax_count','critical4_states'),
        conditional_probability_of_previously_executed_hit_action=ratio('executed_hit_action_conditional_probability_sum','actual_observed_hit_states'))


def run(a):
    import psutil,torch
    from fly_touhou.thrl_model import MOActorCritic
    if not torch.cuda.is_available():raise RuntimeError('CUDA explicitly required')
    if torch.cuda.mem_get_info()[0]<4*1024**3:raise RuntimeError('Four GiB free CUDA reserve required; no CPU fallback')
    if not 1<=a.worlds<=4:raise ValueError('Bounded stored-prefix audit required')
    torch.cuda.set_per_process_memory_fraction(.12);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sys.platform=='win32':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'identity.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time()))
    split=read(a.split);base=a.data;reports={};pins={};input_hashes={};source_pins={}
    datasets=[('teacher30-v1','teacher-horizon-targets-v1',0),('student-aggregation-v1','student-horizon-targets-v1',30)]
    expected_config=None
    for label,source_checkpoint in [('parent',a.parent),('candidate',a.candidate)]:
        pin=a.output/(label+'.pt');sha=checkpoint_copy(source_checkpoint,pin)
        payload=torch.load(pin,map_location='cpu',weights_only=False)
        if payload.get('optimizer_transaction_complete') is False:raise ValueError('Incomplete optimizer checkpoint')
        if expected_config is not None and payload['model_config']!=expected_config:raise ValueError('Model architectures differ')
        expected_config=payload['model_config']
        for name in ['thrl_model.py','thrl_object_transformer.py','thrl_policy_actions.py']:
            source=Path(payload['config']['source_runtime'])/'src/fly_touhou'/name
            if source.read_bytes()!=(ROOT/'src/fly_touhou'/name).read_bytes():raise ValueError('Frozen inference implementation differs: '+name)
        model=MOActorCritic(**payload['model_config']).cuda().eval();model.load_state_dict(payload['model'])
        pins[label]=dict(source=str(source_checkpoint),sha256=sha,candidate_optimized_decisions=payload.get('candidate_optimized_decisions'),
            lifetime_rollout_updates=payload.get('lifetime_rollout_updates'))
        del payload;policies={}
        with torch.inference_mode():
            for dataset,targets_name,offset in datasets:
                folder=base/dataset;gold=read(base/targets_name/'report.json')
                source_sha=hashlib.sha256((folder/'demonstrations.json').read_bytes()).hexdigest()
                if gold['status']!='completed' or source_sha!=gold['source_manifest_sha256']:
                    raise ValueError('Geometry targets differ from their sealed native source')
                arrays={key:np.load(folder/'demonstrations'/(key+'.npy'),mmap_mode='r') for key in
                    ['features','objects','maps','targets','dones','native_episodes','episode_starts']}
                t,w=arrays['features'].shape[:2]
                if w!=30 or not arrays['episode_starts'][0].all() or (arrays['episode_starts'][1:]!=arrays['dones'][:-1]).any() or ((arrays['native_episodes'][1:]!=arrays['native_episodes'][:-1])!=arrays['dones'][:-1]).any():
                    raise ValueError('Stored native reset clocks disagree')
                with np.load(base/targets_name/'targets.npz',allow_pickle=False) as z:
                    if int(z['worlds'])!=w or int(z['frames'])!=t:raise ValueError('Geometry dimensions differ from source')
                    targets={key:z[key].copy() for key in ['margins','trusted_horizons','actions','outcomes']}
                source_pins[dataset]=dict(native_report_sha256=source_sha,
                    geometry_bundle_sha256=hashlib.sha256((base/targets_name/'targets.npz').read_bytes()).hexdigest(),
                    exact_input_prefix_hashes='recorded per world for both policies; includes original FP32 maps')
                rows=[]
                for world,held in held_worlds(split,offset,w,t,a.worlds):
                    encoded=[];digest=hashlib.sha256()
                    for start in range(0,t,32):
                        cpu={key:arrays[key][start:start+32,world].copy() for key in ['features','objects','maps']}
                        for key in ['features','objects','maps']:digest.update(cpu[key].tobytes())
                        batch={key:torch.as_tensor(value,device='cuda') for key,value in cpu.items()}
                        encoded.append(model._encode(batch['features'],batch['maps'],batch['objects']))
                    key=f'{dataset}:world{world}';sha=digest.hexdigest()
                    if key in input_hashes and input_hashes[key]!=sha:raise ValueError('Policies received different actual input prefixes')
                    input_hashes[key]=sha
                    f=torch.as_tensor(arrays['features'][:,world].copy(),device='cuda')
                    h=recurrent_prefix(torch,model,torch.cat(encoded),arrays['dones'][:,world])
                    logits=model._policy_logits(h,f);span=slice(world*t,(world+1)*t)
                    row=state_statistics(logits,arrays['targets'][:,world].copy(),targets['margins'][span],
                        targets['trusted_horizons'][span],targets['actions'][span],targets['outcomes'][span],held)
                    rows.append(row);write(a.output/'progress.json',dict(utc=utc_now(),policy=label,dataset=dataset,world=world,
                        held_states=row['held_states'],native_instances=0,optimizer_updates=0))
                    del encoded,batch,cpu,h,logits,f
                policies[dataset]=dict(summary=mean_statistics(rows),worlds=rows)
                del arrays,targets
        reports[label]=policies;del model;torch.cuda.empty_cache()
    result=dict(schema='B1-offline-fixed-state-retention-v1',status='completed',utc=utc_now(),policies=reports,pins=pins,
        source_pins=source_pins,input_prefix_sha256=input_hashes,split_sha256=hashlib.sha256(a.split.read_bytes()).hexdigest(),
        device='cuda:0',native_instances=0,optimizer_updates=0,current_training_modified=False,
        scope='Same original FP32 stored prefixes, continuous GRU with measured episode resets; score only BC-held16-state descriptors. Existing train-partition sources, not fresh native validation or executed alternative actions.',
        geometry_is_counterfactual_prediction=True,native_avoidance_improvement_claimed=False)
    write(a.output/'report.json',result)
    print(json.dumps({label:{name:p['summary'] for name,p in pools.items()} for label,pools in reports.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ['parent','candidate','data','split','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--worlds',type=int,default=4);run(p.parse_args())
