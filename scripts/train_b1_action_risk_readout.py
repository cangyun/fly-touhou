"""CUDA-only frozen-B1 risk probe on sealed training-partition native sources.

Original teacher/student BC native groups remain split. Test8/9 failures are
not opened, scored or optimized here. No game is launched and the original
policy stays byte-identical. A fitted probe is not native dodge acceptance.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from evaluate_thrl_avoidance_probe import checkpoint_copy


def run(a):
    import torch,psutil
    from fly_touhou.thrl_model import MOActorCritic
    from fly_touhou.thrl_action_risk_readout import ActionRiskReadout,native_risk_targets,descriptor_indices
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<4*1024**3:raise RuntimeError('CUDA with four GiB reserve required')
    torch.cuda.set_per_process_memory_fraction(.12);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    if sys.platform=='win32':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    write(a.output/'launch.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now()))
    def stop():
        if (a.output/'STOP').exists():raise RuntimeError('Operator stopped risk readout')
    def publish(stage,**kw):write(a.output/'progress.json',dict(stage=stage,utc=utc_now(),device='cuda:0',native_instances=0,**kw))
    pin=a.output/'base.pt';sha=checkpoint_copy(a.checkpoint,pin);p=torch.load(pin,map_location='cpu',weights_only=False)
    if p.get('optimizer_transaction_complete') is False:raise ValueError('Incomplete base policy')
    for name in ['thrl_model.py','thrl_object_transformer.py','thrl_policy_actions.py']:
        if (Path(p['config']['source_runtime'])/'src/fly_touhou'/name).read_bytes()!=(ROOT/'src/fly_touhou'/name).read_bytes():
            raise ValueError('Base inference implementation changed')
    model=MOActorCritic(**p['model_config']).cuda().eval();model.load_state_dict(p['model'])
    if not model.is_recurrent or model.input_schema!='semantic-separated-timers-object-priority-v3':raise ValueError('Require current separated B1 GRU')
    configuration=p['model_config'];del p
    split=read(a.split);cache=None;labels=[];masks=[];source_pins=[];prefix_hashes={};offset=0;t=None
    with torch.inference_mode():
        for dataset,goldname in [('teacher30-v1','teacher-horizon-targets-v1'),('student-aggregation-v1','student-horizon-targets-v1')]:
            folder=a.data/dataset;goldpath=a.data/goldname/'targets.npz';gold=read(a.data/goldname/'report.json')
            report_sha=hashlib.sha256((folder/'demonstrations.json').read_bytes()).hexdigest()
            if gold['status']!='completed' or gold['source_manifest_sha256']!=report_sha:raise ValueError('Native geometry provenance differs')
            arrays={key:np.load(folder/'demonstrations'/(key+'.npy'),mmap_mode='r') for key in
                ['features','objects','maps','dones','episode_starts','native_episodes','native_seeds']}
            length,worlds=arrays['features'].shape[:2]
            if t is None:t=length
            if length!=t or worlds!=30 or not arrays['episode_starts'][0].all():raise ValueError('Expected equal complete-opening prefixes')
            if (arrays['native_seeds']%10>=6).any():raise ValueError('Reserved development/test seeds cannot enter fitting')
            if (arrays['episode_starts'][1:]!=arrays['dones'][:-1]).any() or ((arrays['native_episodes'][1:]!=arrays['native_episodes'][:-1])!=arrays['dones'][:-1]).any():
                raise ValueError('Measured native reset clocks differ')
            with np.load(goldpath,allow_pickle=False) as z:
                if int(z['worlds'])!=30 or int(z['frames'])!=t:raise ValueError('Target source dimensions differ')
                y,known=native_risk_targets(z['margins'],z['trusted_horizons'],z['actions'],z['outcomes'])
            labels.append(y);masks.append(known)
            for world in range(worlds):
                stop();encoded=[];digest=hashlib.sha256()
                for begin in range(0,t,32):
                    cpu={k:arrays[k][begin:begin+32,world].copy() for k in ['features','objects','maps']}
                    for k in ['features','objects','maps']:digest.update(cpu[k].tobytes())
                    batch={k:torch.as_tensor(v,device='cuda') for k,v in cpu.items()}
                    encoded.append(model._encode(batch['features'],batch['maps'],batch['objects']))
                encoded=torch.cat(encoded);done=arrays['dones'][:,world]
                cuts=[0,*list(np.flatnonzero(done[:-1])+1),t]
                hidden=torch.cat([model.gru(encoded[first:last,None])[0][:,0] for first,last in zip(cuts,cuts[1:])])
                x=torch.cat((encoded,hidden),-1).cpu().numpy()
                if cache is None:cache=np.lib.format.open_memmap(a.output/'features.npy',mode='w+',dtype=np.float32,shape=(t*60,x.shape[-1]))
                cache[(offset+world)*t:(offset+world+1)*t]=x
                prefix_hashes[f'{dataset}:world{world}']=digest.hexdigest()
                publish('encoding_frozen_B1',completed_worlds=offset+world+1,target_worlds=60,base_policy_modified=False)
                del encoded,hidden,x,batch,cpu
            source_pins.append(dict(dataset=dataset,native_report_sha256=report_sha,geometry_sha256=hashlib.sha256(goldpath.read_bytes()).hexdigest(),native_seed_partition='train0..5'))
            offset+=worlds;del arrays
    cache.flush();input_dim=cache.shape[-1];del model;torch.cuda.empty_cache()
    y=np.concatenate(labels);known=np.concatenate(masks)
    train=descriptor_indices(split['train'],t,60);held=descriptor_indices(split['heldout'],t,60)
    if np.intersect1d(train,held).size:raise ValueError('Training and holdout overlap')
    train=train[known[train].any(-1)];held=held[known[held].any(-1)]
    x=torch.as_tensor(np.asarray(cache).copy(),device='cuda');targets=torch.as_tensor(y,device='cuda');valid=torch.as_tensor(known,device='cuda')
    train=torch.as_tensor(train,device='cuda');held=torch.as_tensor(held,device='cuda');head=ActionRiskReadout(input_dim,a.hidden).cuda()
    optimizer=torch.optim.Adam(head.parameters(),lr=a.learning_rate)
    positive=int((targets[train]*valid[train]).sum());negative=int(valid[train].sum())-positive
    weight=min(20.,max(1.,negative/max(1,positive)))
    write(a.output/'plan.json',dict(base_checkpoint=str(a.checkpoint),base_sha256=sha,base_model_config=configuration,
        feature_dim=input_dim,head_hidden=a.hidden,head_parameters=sum(v.numel() for v in head.parameters()),
        source_pins=source_pins,prefix_input_sha256=prefix_hashes,split_sha256=hashlib.sha256(a.split.read_bytes()).hexdigest(),
        train_states=len(train),held_states=len(held),train_positive_labels=positive,train_negative_labels=negative,positive_weight=weight,
        learning_rate=a.learning_rate,epochs_max=a.epochs,patience=a.patience,seed=a.seed,
        optimizer_scope='only new readout; frozen original B1 encoder,GRU,actor,critic',native_instances=0,test_failure_examples_in_training=0))
    def assess():
        sums=np.zeros(6,float)
        with torch.inference_mode():
            for begin in range(0,len(held),2048):
                ix=held[begin:begin+2048];z=head(x[ix]);target=targets[ix];mask=valid[ix]
                loss=torch.nn.functional.binary_cross_entropy_with_logits(z,target,reduction='none')
                pos=mask&(target>.5);neg=mask&~pos;pred=z>0
                sums+=np.array([float(loss[pos].sum()),int(pos.sum()),float(loss[neg].sum()),int(neg.sum()),int((pred&pos).sum()),int((pred&neg).sum())])
        return dict(balanced_BCE=.5*(sums[0]/sums[1]+sums[2]/sums[3]),positive_labels=int(sums[1]),negative_labels=int(sums[3]),
            dangerous_action_recall=sums[4]/sums[1],safe_action_false_positive_rate=sums[5]/sums[3])
    best=float('inf');stale=0;steps=0;history=[]
    for epoch in range(1,a.epochs+1):
        stop();head.train();permutation=train[torch.randperm(len(train),device='cuda')]
        for begin in range(0,len(train),1024):
            ix=permutation[begin:begin+1024];z=head(x[ix]);target=targets[ix];mask=valid[ix]
            weights=mask.to(z.dtype)*torch.where(target>.5,weight,1.)
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,target,reduction='none')*weights).sum()/weights.sum()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite risk readout loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite readout gradient')
            optimizer.step();steps+=1
        head.eval();scores=assess();row=dict(epoch=epoch,Adam_steps=steps,**scores);history.append(row)
        if scores['balanced_BCE']<best-1e-5:
            best=scores['balanced_BCE'];stale=0
            temp=a.output/'best.tmp';torch.save(dict(schema='B1-frozen-action-risk-readout-v1',model=head.state_dict(),input_dim=input_dim,
                hidden_dim=a.hidden,base_sha256=sha,base_checkpoint=str(pin),epoch=epoch,Adam_steps=steps,held_metrics=scores,
                original_policy_modified=False,native_acceptance=False),temp);os.replace(temp,a.output/'best.pt')
        else:stale+=1
        publish('fitting_action_risk',**row,stale_epochs=stale,base_policy_modified=False)
        with (a.output/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        if stale>=a.patience:break
    best_checkpoint=torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)
    result=dict(status='completed',utc=utc_now(),base_sha256=sha,head_parameters=sum(v.numel() for v in head.parameters()),epochs=len(history),Adam_steps=steps,
        best_epoch=best_checkpoint['epoch'],best_held=best_checkpoint['held_metrics'],wall_seconds=time.monotonic()-started,
        cache_bytes=cache.nbytes,native_instances=0,original_policy_modified=False,readout_deployed=False,native_avoidance_improvement_claimed=False,
        interpretation='Risk classification feasibility on existing BC grouped holdout; no reserved8/9 source fitted. Requires original-game policy validation before any efficacy claim.')
    write(a.output/'result.json',result);publish('finished',result=result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ['checkpoint','data','split','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed',type=int,default=11);p.add_argument('--hidden',type=int,default=128)
    p.add_argument('--learning-rate',type=float,default=.001);p.add_argument('--epochs',type=int,default=60);p.add_argument('--patience',type=int,default=8)
    run(p.parse_args())
