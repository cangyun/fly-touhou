"""Finite CUDA calibration of B1's point-risk branch; no PPO or original worlds."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.thrl_full_geometry_data import build_archive,digest,calibrated_state
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.thrl_durable_records import write_json as write,append_jsonl


def choose_states(data,held,maximum,rng):
    import numpy as np
    f,o=data['features'],data['objects']
    point=(o[:,:,0]>.5)&((o[:,:,1]>.5)|(o[:,:,2]>.5))&(o[:,:,13]>.5)
    mask=(data['held']==held)&(f[:,21]>.5)&(f[:,20]<=0)&(f[:,5]<=0)&point.any(1)
    pool=np.flatnonzero(mask);chosen=[]
    for stage in range(1,7):
        for large in (False,True):
            group=pool[(data['stage'][pool]==stage)&((f[pool,13]*64>3)==large)]
            if len(group):chosen.extend(rng.choice(group,min(len(group),max(1,maximum//12)),replace=False).tolist())
    rest=np.setdiff1d(pool,chosen);count=min(len(rest),max(0,maximum-len(chosen)))
    if count:chosen.extend(rng.choice(rest,count,replace=False).tolist())
    if not chosen:raise ValueError('No unprotected native point-hazard states in partition')
    return np.asarray(sorted(chosen),np.int64)


def write_pairs(torch,data,indices,folder,rng,training,stop,publish):
    import numpy as np
    from fly_touhou.thrl_movement_teacher import action_velocities
    from fly_touhou.thrl_pair_risk_labels import paired_clearance
    from fly_touhou.thrl_pair_risk import query_features,QUERY_DIM
    folder.mkdir();capacity=len(indices)*18*12*(3 if training else 2)
    q=np.lib.format.open_memmap(folder/'queries.npy',mode='w+',dtype='float32',shape=(capacity,QUERY_DIM))
    y=np.lib.format.open_memmap(folder/'clearance.npy',mode='w+',dtype='float32',shape=(capacity,))
    group=np.lib.format.open_memmap(folder/'enlarged.npy',mode='w+',dtype='bool',shape=(capacity,))
    kind=np.lib.format.open_memmap(folder/'origin.npy',mode='w+',dtype='uint8',shape=(capacity,))
    count=0;velocities=action_velocities()
    for begin in range(0,len(indices),16):
        stop();ix=indices[begin:begin+16];f=data['features'][ix];o=data['objects'][ix];n=len(ix)
        valid=(o[:,:,0]>.5)&((o[:,:,1]>.5)|(o[:,:,2]>.5))&(o[:,:,13]>.5)
        relative=o[:,:,6:8]*[384,448];motion=o[:,:,8:10]*12
        sq=(motion*motion).sum(-1);tau=np.clip(-(relative*motion).sum(-1)/(sq+1e-6),0,4)
        rank=np.linalg.norm(relative+motion*tau[:,:,None],axis=-1)-o[:,:,10]*128
        nearest=np.argsort(np.where(valid,rank,np.inf),axis=1)[:,:8]
        random_rank=rng.random((n,96));random_rank[~valid]=np.inf
        np.put_along_axis(random_rank,nearest,np.inf,axis=1)
        other=np.argsort(random_rank,axis=1)[:,:4]
        slots=np.concatenate((nearest,other),axis=1)
        selected=o[np.arange(n)[:,None],slots]
        keep=np.concatenate((valid[np.arange(n)[:,None],nearest],
            np.isfinite(random_rank[np.arange(n)[:,None],other])),axis=1)
        mask=np.broadcast_to(keep[:,None],(n,18,12)).reshape(-1)
        def spread(value):return np.broadcast_to(value,(n,18,12,*value.shape[3:])).reshape(-1,*value.shape[3:])[mask]
        origin=np.stack((f[:,0]*368-184,f[:,1]*400+32),-1)
        starts=spread(origin[:,None,None,:]);vel=spread(velocities[None,:,None,:])
        positions=spread((origin[:,None,:]+selected[:,:,6:8]*[384,448])[:,None])
        motions=spread((selected[:,:,8:10]*12)[:,None])
        circle=spread(((selected[:,:,1]>.5)&(selected[:,:,12]>.5))[:,None,:])
        bullet=spread(selected[:,None,:,10]*128)
        original_player=spread(np.broadcast_to(f[:,None,None,13]*64,(n,1,12)))
        base_extents=spread((selected[:,:,10:12]*128)[:,None])
        half=spread(np.broadcast_to(f[:,None,14:16]*64,(n,12,2))[:,None])
        offsets=spread(np.broadcast_to(f[:,None,16:18]*64,(n,12,2))[:,None])
        radius=np.sqrt(bullet**2+original_player**2)+.05;extents=base_extents+half+.05
        def append(ps,vs,rs,es,which,enlarged):
            nonlocal count
            target=paired_clearance(ps,vel,vs,motions,circle,rs,es,offsets)
            tensors=[torch.as_tensor(np.array(v,copy=True),device='cuda',dtype=torch.float32)
                for v in (ps,vel,vs,motions,circle,rs,es,offsets)]
            with torch.inference_mode():queries=query_features(*tensors).cpu().numpy()
            length=len(target);q[count:count+length]=queries;y[count:count+length]=np.clip(target,-64,64)
            group[count:count+length]=enlarged;kind[count:count+length]=which;count+=length
        def boundary(r,e):
            ps=rng.uniform([-184,32],[184,432],starts.shape)
            near=rng.random(len(ps))<.5;axis=rng.integers(0,2,len(ps));side=rng.integers(0,2,len(ps))
            edge=np.where(side==0,np.array([-184,32])[axis],np.array([184,432])[axis])
            ps[np.arange(len(ps))[near],axis[near]]=edge[near]+np.where(side[near]==0,1,-1)*rng.uniform(0,24,len(ps))[near]
            times=rng.uniform(0,4,len(ps));angles=rng.uniform(-math.pi,math.pi,len(ps));jitter=rng.uniform(-6,6,len(ps))
            circular=np.stack((np.cos(angles),np.sin(angles)),-1)*(r+jitter)[:,None]
            rectangle=rng.uniform(-1,1,e.shape)*e;ax=rng.integers(0,2,len(ps));sign=rng.choice([-1,1],len(ps))
            rectangle[np.arange(len(ps)),ax]=sign*(e[np.arange(len(ps)),ax]+jitter)
            delta=np.where(circle[:,None],circular,offsets-rectangle)
            contact=np.clip(ps+vel*times[:,None],[-184,32],[184,432])+delta
            return ps,contact-motions*times[:,None]
        append(starts,positions,radius,extents,0,original_player>3)
        if training:
            ps,pos=boundary(radius,extents);append(ps,pos,radius,extents,1,original_player>3)
        # Explicit modeled augmentation, not fabricated original observations.
        # Vary player circle and rectangle dimensions independently: circles
        # must not learn irrelevant rectangle dimensions as a radius proxy.
        player=rng.uniform(2,24,len(starts));h=rng.uniform(1,3,half.shape)
        r2=np.sqrt(bullet**2+player**2)+.05;e2=base_extents+h+.05
        ps,pos=boundary(r2,e2);append(ps,pos,r2,e2,2,player>3)
        if begin%256==0:publish('preparing_full_geometry_queries',partition='train' if training else 'held',
            completed_states=min(begin+16,len(indices)),states=len(indices),queries_written=count)
    for value in (q,y,group,kind):value.flush()
    np.save(folder/'source_state_indices.npy',indices)
    return count


def run(plan_path):
    import numpy as np
    import psutil
    import torch
    from fly_touhou.thrl_pair_risk import PairRiskNetwork
    plan=read(plan_path)
    if digest(ROOT/'source_manifest.json')!=plan['source_manifest_sha256']:raise ValueError('Calibration source changed')
    for row in read(ROOT/'source_manifest.json')['files']:
        if digest(ROOT/row['path'])!=row['sha256']:raise ValueError('Frozen calibration file changed')
    for path,sha in [(plan['parent'],plan['parent_sha256']),(Path(plan['archive'])/'demonstrations.json',plan['archive_manifest_sha256'])]:
        if digest(path)!=sha:raise ValueError('Declared native source or neural parent changed')
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<4*2**30:raise RuntimeError('CUDA and4GiB free memory required')
    output=Path(plan['output']);output.mkdir(parents=True,exist_ok=False)
    run.created_output=output
    begun=time.monotonic();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed']);rng=np.random.default_rng(plan['seed'])
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    write(output/'launch.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now(),
        original_worlds=0,CUDA_required=True,PPO_optimizer_updates=0))
    write(output/'plan.json',plan)
    cfg=read(plan['environment_config']);cfg.update(workers=0,offline_neural_calibration=True,
        display_name='B1 · 人物尺寸风险校准（无原作采样、无PPO）')
    write(output/'study_config.json',cfg)
    def stop():
        if (output/'STOP').exists():raise RuntimeError('Operator stopped the finite risk calibration')
    def publish(stage,**extra):
        row=dict(status='calibration',stage=stage,updated_utc=utc_now(),device='cuda:0',workers=0,
            actual_decisions=0,adam_minibatch_steps=0,PPO_optimizer_updates=0,native_worlds_started=0,
            wall_seconds=time.monotonic()-begun,**extra)
        write(output/'progress.json',row)
        write(output/'study_status.json',dict(status='training',phase='calibration',current_run=str(output),
            updated_utc=utc_now(),current_progress=row))
    publish('extracting_full_native_geometry')
    from register_b1_goal_phase import register
    write(output/'panel_registration.json',register(output,plan['panel_job_id'],'calibration',
        control_root=Path(plan['control_center_root'])))
    data,manifest=build_archive(plan['archive'],plan['held_native_seeds'],output/'native_geometry',publish,stop)
    train=choose_states(data,False,8192,rng);held=choose_states(data,True,2048,rng)
    if set(data['native_seed'][train])&set(data['native_seed'][held]):raise ValueError('Fitting and held native seeds overlap')
    counts={name:write_pairs(torch,data,ix,output/name,rng,name=='train',stop,publish)
        for name,ix in [('train',train),('held',held)]}
    arrays={}
    for name in counts:
        arrays[name]={key:torch.as_tensor(np.load(output/name/(filename+'.npy'),mmap_mode='r')[:counts[name]].copy(),device='cuda')
            for key,filename in [('q','queries'),('y','clearance'),('big','enlarged'),('origin','origin')]}
    parent=torch.load(plan['parent'],map_location='cpu',weights_only=False)
    state={k.removeprefix('risk.'):v for k,v in parent['model'].items() if k.startswith('risk.')}
    width=parent['model_config']['pair_risk_wrapper']['width'];net=PairRiskNetwork(width).cuda();net.load_state_dict(state,strict=True)
    optimizer=torch.optim.Adam(net.parameters(),lr=.001)
    def assess():
        net.eval();item=arrays['held'];pred=[]
        with torch.inference_mode():
            for begin in range(0,len(item['q']),32768):pred.append(net(item['q'][begin:begin+32768]))
        pred=torch.cat(pred);y=item['y'];metrics={}
        for name,mask in [('all',torch.ones_like(y,dtype=torch.bool)),('normal',~item['big']),('enlarged',item['big']),
                          ('original_native_geometry',item['origin']==0),('modeled_geometry_augmentation',item['origin']==2)]:
            near=mask&(y.abs()<8);danger=mask&(y<=0);safe=mask&(y>1)
            mae=float((pred[near]-y[near]).abs().mean()) if near.any() else 0.
            fn=float((pred[danger]>=1).float().mean()) if danger.any() else 0.
            fp=float((pred[safe]<1).float().mean()) if safe.any() else 0.
            metrics[name]=dict(pairs=int(mask.sum()),boundary_MAE=mae,danger_pairs=int(danger.sum()),
                dangerous_pair_recall=1-fn,safe_false_positive_rate=fp,score=mae+5*fn+fp)
        return dict(groups=metrics,selection_score=(metrics['normal']['score']+metrics['enlarged']['score'])/2)
    initial=assess();write(output/'initial_frozen_risk_metrics.json',initial)
    pool=arrays['train'];y=pool['y'];pools=[torch.nonzero(y<=0).flatten(),torch.nonzero(y.abs()<8).flatten(),torch.arange(len(y),device='cuda')]
    if any(len(p)==0 for p in pools):raise ValueError('Calibration needs both collision and boundary targets')
    best=float('inf');stale=0;steps=0;history=[]
    for epoch in range(1,41):
        stop();net.train();rate=.001*(.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/40)))
        for group in optimizer.param_groups:group['lr']=rate
        for _ in range(200):
            stop();ix=torch.cat([p[torch.randint(len(p),(n,),device='cuda')] for p,n in zip(pools,[1024,2048,1024])])
            target=y[ix];pred=net(pool['q'][ix]);known=(target<=0)|(target>1)
            loss=torch.nn.functional.smooth_l1_loss(pred/8,target/8,beta=.1)+.2*torch.nn.functional.binary_cross_entropy_with_logits(
                -(pred[known]-.5),(target[known]<=0).float())
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite geometry calibration loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite geometry calibration gradient')
            optimizer.step();steps+=1
        metrics=assess();row=dict(epoch=epoch,risk_fit_Adam_steps=steps,learning_rate=rate,**metrics);history.append(row)
        append_jsonl(output/'risk_fit_history.jsonl',row)
        if metrics['selection_score']<best-1e-4:
            best=metrics['selection_score'];stale=0
            torch.save(dict(schema='B1-full-geometry-point-risk-fit-v1',model=net.state_dict(),width=width,
                epoch=epoch,risk_fit_Adam_steps=steps,held=metrics),output/'best_risk.pt')
        else:stale+=1
        publish('fitting_full_geometry_point_risk',**row,stale_epochs=stale)
        if stale>=8:break
    chosen=torch.load(output/'best_risk.pt',map_location='cpu',weights_only=False)
    updated,changed=calibrated_state(parent['model'],chosen['model'])
    candidate=dict(schema='B1-risk-calibrated-frozen-candidate-v1',model=updated,model_config=parent['model_config'],
        config=parent['config'],parent=plan['parent'],parent_sha256=plan['parent_sha256'],
        changed_state_tensors=changed,optimizer_state_valid_for_resume=False,evaluation_only=True,
        risk_fit_Adam_steps=chosen['risk_fit_Adam_steps'],PPO_optimizer_updates=0,
        model_decisions_inherited=parent.get('decisions'),input_reward_actor_GRU_critic_and_laser_unchanged=True)
    target=output/'candidate_frozen.pt';torch.save(candidate,target)
    with target.open('r+b') as f:f.flush();os.fsync(f.fileno())
    result=dict(schema='B1-full-geometry-point-risk-calibration-result-v1',status='completed',utc=utc_now(),
        initial=initial,selected=chosen['held'],selected_epoch=chosen['epoch'],epochs_executed=len(history),
        paid_risk_fit_Adam_steps=steps,selected_risk_fit_Adam_steps=chosen['risk_fit_Adam_steps'],
        selected_proxy_improved=chosen['held']['selection_score']<initial['selection_score'],
        source_native_geometry=manifest['partitions'],query_pairs=counts,
        checkpoint=str(target),checkpoint_sha256=digest(target),parent_sha256=plan['parent_sha256'],
        changed_state_tensors=changed,actor_GRU_critic_laser_and_rewards_unchanged=True,
        actual_native_worlds_started=0,PPO_optimizer_updates=0,actual_device=str(next(net.parameters()).device),
        unexecuted_geometric_alternatives_not_native_outcomes=True,native_efficacy_not_yet_tested=True,
        wall_seconds=time.monotonic()-begun)
    write(output/'result.json',result);write(output/'progress.json',{**result,'status':'finished','stage':'finished'})
    write(output/'study_status.json',dict(status='finished',phase='completed',current_run=str(output),updated_utc=utc_now()))
    print(json.dumps({k:result[k] for k in ['status','selected_epoch','paid_risk_fit_Adam_steps','selected_proxy_improved']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    try:run(a.plan)
    except BaseException:
        out=getattr(run,'created_output',None)
        if out is not None:write(out/'failure.json',dict(utc=utc_now(),traceback=traceback.format_exc(),no_automatic_restart=True,
            native_worlds_started=0,PPO_optimizer_updates=0))
        raise
