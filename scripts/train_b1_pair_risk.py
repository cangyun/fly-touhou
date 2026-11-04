"""CUDA learning of action/entity geometry from sealed B1 training observations.

No original game, teacher execution or reserved test data. Optional contact
augmentation is explicitly modeled geometry, never labeled a native outcome.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(a):
    import torch,psutil
    from fly_touhou.thrl_action_risk_readout import descriptor_indices
    from fly_touhou.thrl_movement_teacher import action_velocities
    from fly_touhou.thrl_pair_risk_labels import paired_clearance
    from fly_touhou.thrl_pair_risk import PairRiskNetwork,query_features,QUERY_DIM,QUERY_SCHEMA,action_clearances
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<4*1024**3:
        raise RuntimeError('CUDA and four GiB reserve required; no CPU fallback')
    torch.cuda.set_per_process_memory_fraction(.2);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);rng=np.random.default_rng(a.seed)
    if sys.platform=='win32':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    write(a.output/'identity.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time()))
    def publish(stage,**kw):write(a.output/'progress.json',dict(stage=stage,utc=utc_now(),device='cuda:0',native_instances=0,**kw))
    def stop():
        if (a.output/'STOP').exists():raise RuntimeError('Operator stopped pair risk learning')
    features=[];objects=[];seeds=[];trust=[];margins=[];pins=[]
    for dataset,goldname in [('teacher30-v1','teacher-horizon-targets-v1'),('student-aggregation-v1','student-horizon-targets-v1')]:
        folder=a.data/dataset/'demonstrations';report=read(a.data/goldname/'report.json')
        if report['status']!='completed' or report['source_manifest_sha256']!=sha(a.data/dataset/'demonstrations.json'):
            raise ValueError('Native geometry provenance differs')
        f=np.load(folder/'features.npy',mmap_mode='r');o=np.load(folder/'objects.npy',mmap_mode='r')
        s=np.load(folder/'native_seeds.npy',mmap_mode='r');t,w=f.shape[:2]
        if (t,w)!=(1024,30) or o.shape!=(t,w,96,21) or np.any(s%10>=6):
            raise ValueError('Only original train-partition0..5 observations allowed')
        features.append(np.asarray(f).transpose(1,0,2).reshape(-1,25).copy())
        objects.append(np.asarray(o).transpose(1,0,2,3).reshape(-1,96,21).copy())
        seeds.append(np.asarray(s).T.reshape(-1).copy())
        with np.load(a.data/goldname/'targets.npz') as z:
            trust.append(z['trusted_horizons'][:,0].copy());margins.append(z['margins'][:,0].copy())
        pins.append(dict(dataset=dataset,features_sha256=sha(folder/'features.npy'),objects_sha256=sha(folder/'objects.npy'),
            native_seeds_sha256=sha(folder/'native_seeds.npy'),geometry_sha256=sha(a.data/goldname/'targets.npz')))
    f=np.concatenate(features);o=np.concatenate(objects);native=np.concatenate(seeds);trusted=np.concatenate(trust);gold=np.concatenate(margins)
    del features,objects,seeds,trust,margins
    split=read(a.split);train=descriptor_indices(split['train'],1024,60);held=descriptor_indices(split['heldout'],1024,60)
    if np.intersect1d(train,held).size or set(native[train])&set(native[held]):
        raise ValueError('Native seed groups overlap between fitting and holdout')
    valid=(o[...,0]>.5)&((o[...,1]>.5)|(o[...,2]>.5))&(o[...,13]>.5)
    def select(ix,count):
        ix=ix[trusted[ix]&valid[ix].any(-1)]
        critical=ix[gold[ix].min(-1)<12]
        chosen=rng.choice(critical,min(len(critical),count//2),replace=False)
        remaining=np.setdiff1d(ix,chosen);take=min(count-len(chosen),len(remaining))
        return np.r_[chosen,rng.choice(remaining,take,replace=False)]
    indices={'train':select(train,a.train_states),'held':select(held,a.held_states)}
    velocities=action_velocities();datasets={};source_counts={}
    for name,ix in indices.items():
        augment=name=='train';capacity=len(ix)*12*18*(2 if augment else 1)
        directory=a.output/name;directory.mkdir()
        cache=np.lib.format.open_memmap(directory/'queries.npy',mode='w+',dtype=np.float32,shape=(capacity,QUERY_DIM))
        target=np.lib.format.open_memmap(directory/'clearances.npy',mode='w+',dtype=np.float32,shape=(capacity,))
        origin_kind=np.lib.format.open_memmap(directory/'modeled_augmentation.npy',mode='w+',dtype=np.uint8,shape=(capacity,))
        count=0;native_count=0
        for begin in range(0,len(ix),32):
            stop();selected=ix[begin:begin+32];ff=f[selected];oo=o[selected];keep=valid[selected];bs=len(selected)
            origin=np.stack((ff[:,0]*368-184,ff[:,1]*400+32),-1)
            relative=oo[...,6:8]*[384,448];motion=oo[...,8:10]*12
            speed2=(motion*motion).sum(-1);tau=np.clip(-(relative*motion).sum(-1)/(speed2+1e-6),0,4)
            rank=np.linalg.norm(relative+motion*tau[...,None],axis=-1)-oo[...,10]*128
            rank=np.where(keep,rank,np.inf);nearest=np.argsort(rank,axis=1)[:,:8]
            random_rank=rng.random((bs,96));random_rank[~keep]=np.inf
            np.put_along_axis(random_rank,nearest,np.inf,axis=1)
            other=np.argsort(random_rank,axis=1)[:,:4];slots=np.concatenate((nearest,other),axis=1)
            selected_objects=oo[np.arange(bs)[:,None],slots]
            active=keep[np.arange(bs)[:,None],slots]
            mask=np.broadcast_to(active[:,None],(bs,18,12)).reshape(-1)
            def spread(x):return np.broadcast_to(x,(bs,18,12,*x.shape[3:])).reshape(-1,*x.shape[3:])[mask]
            starts=spread(origin[:,None,None,:]);vel=spread(velocities[None,:,None,:])
            positions=spread((origin[:,None,:]+selected_objects[...,6:8]*[384,448])[:,None])
            motions=spread((selected_objects[...,8:10]*12)[:,None])
            circle=spread(((selected_objects[...,1]>.5)&(selected_objects[...,12]>.5))[:,None,:])
            radius=spread((np.sqrt((selected_objects[...,10]*128)**2+(ff[:,None,13]*64)**2)+.05)[:,None,:])
            extents=spread((selected_objects[...,10:12]*128+ff[:,None,14:16]*64+.05)[:,None])
            offsets=spread(np.broadcast_to(ff[:,None,16:18]*64,(bs,12,2))[:,None])
            def append(position,kind,query_starts=None):
                nonlocal count,native_count
                query_starts=starts if query_starts is None else query_starts
                values=paired_clearance(query_starts,vel,position,motions,circle,radius,extents,offsets)
                tensors=[torch.as_tensor(np.array(x,copy=True),device='cuda',dtype=torch.float32)
                         for x in [query_starts,vel,position,motions,circle,radius,extents,offsets]]
                with torch.inference_mode():q=query_features(*tensors).cpu().numpy()
                n=len(q);cache[count:count+n]=q;target[count:count+n]=np.clip(values,-64,64);origin_kind[count:count+n]=kind
                count+=n;native_count+=n if kind==0 else 0
            append(positions,0)
            if augment and len(starts):
                # Contact-centered augmentation: model geometry only, no claim
                # these changed positions were ever executed in the original.
                time_at=rng.uniform(0,4,len(starts));angle=rng.uniform(-math.pi,math.pi,len(starts))
                # Geometry is spatially homogeneous except for legal walls.
                # Spread modeled contact configurations across the legal field
                # and include wall approaches rather than retaining native
                # demonstration-specific absolute player positions.
                augmented_starts=rng.uniform([-184.,32.],[184.,432.],starts.shape)
                near_wall=rng.random(len(starts))<.5
                wall_axis=rng.integers(0,2,len(starts));wall_side=rng.integers(0,2,len(starts))
                edge=np.where(wall_side==0,np.array([-184.,32.])[wall_axis],np.array([184.,432.])[wall_axis])
                wall_distance=rng.uniform(0,24,len(starts))
                wall_distance[rng.random(len(starts))<.5]=0.
                edge+=np.where(wall_side==0,1,-1)*wall_distance
                augmented_starts[np.arange(len(starts))[near_wall],wall_axis[near_wall]]=edge[near_wall]
                corner=rng.random(len(starts))<.1
                augmented_starts[corner]=np.where(rng.integers(0,2,(int(corner.sum()),2))==0,
                                                 [-184.,32.],[184.,432.])
                jitter=rng.uniform(-6,6,len(starts));unit=np.stack((np.cos(angle),np.sin(angle)),-1)
                circular=unit*(radius+jitter)[:,None]
                rectangle=rng.uniform(-1,1,extents.shape)*extents
                axis=rng.integers(0,2,len(starts));sign=rng.choice([-1.,1.],len(starts))
                rectangle[np.arange(len(starts)),axis]=sign*(extents[np.arange(len(starts)),axis]+jitter)
                boundary=np.where(circle[:,None],circular,offsets-rectangle)
                contact=np.clip(augmented_starts+vel*time_at[:,None],[-184,32],[184,432])+boundary
                append(contact-motions*time_at[:,None],1,augmented_starts)
            publish('preparing_pair_queries',partition=name,completed_states=min(begin+32,len(ix)),states=len(ix),written_pairs=count)
        cache.flush();target.flush();origin_kind.flush();del cache,target,origin_kind
        datasets[name]=(directory,count);source_counts[name]=dict(native_pairs=native_count,modeled_augmentation_pairs=count-native_count,states=len(ix))
        np.save(directory/'source_state_indices.npy',ix)
    net=PairRiskNetwork(a.width).cuda();optimizer=torch.optim.Adam(net.parameters(),lr=a.learning_rate)
    def load(name):
        directory,n=datasets[name]
        x=torch.as_tensor(np.load(directory/'queries.npy',mmap_mode='r')[:n].copy(),device='cuda')
        y=torch.as_tensor(np.load(directory/'clearances.npy',mmap_mode='r')[:n].copy(),device='cuda')
        return x,y
    tx,ty=load('train');hx,hy=load('held')
    pools=[torch.where(ty<=0)[0],torch.where(ty.abs()<8)[0],torch.arange(len(ty),device='cuda')]
    if any(len(pool)==0 for pool in pools):raise ValueError('No dangerous/near/ordinary pair diversity')
    write(a.output/'plan.json',dict(utc=utc_now(),source_pins=pins,split_sha256=sha(a.split),query_dim=QUERY_DIM,query_schema=QUERY_SCHEMA,
        source_counts=source_counts,native_seed_groups_disjoint=True,native_partition='train0..5',
        source_parameters=sum(x.numel() for x in net.parameters()),width=a.width,learning_rate=a.learning_rate,
        epochs_max=a.epochs,steps_per_epoch=a.steps_per_epoch,batch_size=4096,seed=a.seed,
        native_instances=0,original_B1_modified=False,augmentation_is_actual_native_data=False,
        teacher_execution_or_collision_solver_at_inference=False,targets='modeled four-frame signed clearance, actual geometry fields; not alternative native outcomes'))
    def assess():
        sums=np.zeros(7)
        with torch.inference_mode():
            for begin in range(0,len(hy),8192):
                y=hy[begin:begin+8192];p=net(hx[begin:begin+8192]);near=y.abs()<8;positive=y<=0;negative=y>1
                sums+=np.array([float((p[near]-y[near]).abs().sum()),int(near.sum()),
                    int((p[positive]>=1).sum()),int(positive.sum()),int((p[negative]<1).sum()),int(negative.sum()),
                    float((p-y).abs().sum())])
        mae=sums[0]/max(1,sums[1]);fn=sums[2]/max(1,sums[3]);fp=sums[4]/max(1,sums[5])
        return dict(boundary_MAE=mae,danger_recall=1-fn,safe_false_positive_rate=fp,
            held_danger_pairs=int(sums[3]),held_safe_pairs=int(sums[5]),all_MAE=sums[6]/len(hy),selection_score=mae+5*fn+fp)
    best=float('inf');stale=0;steps=0;history=[]
    for epoch in range(1,a.epochs+1):
        stop();net.train();rate=a.learning_rate*(.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/a.epochs)))
        for group in optimizer.param_groups:group['lr']=rate
        for _ in range(a.steps_per_epoch):
            ix=torch.cat([pool[torch.randint(len(pool),(count,),device='cuda')] for pool,count in zip(pools,[1024,2048,1024])])
            y=ty[ix];pred=net(tx[ix]);reg=torch.nn.functional.smooth_l1_loss(pred/8,y/8,beta=.1)
            known=(y<=0)|(y>1);label=(y<=0).to(pred.dtype)
            classification=torch.nn.functional.binary_cross_entropy_with_logits(-(pred[known]-.5),label[known])
            loss=reg+.1*classification
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite pair risk loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            optimizer.step();steps+=1
            if steps%10==0:time.sleep(.005)
        net.eval();score=assess();row=dict(epoch=epoch,Adam_steps=steps,learning_rate=rate,**score);history.append(row)
        if score['selection_score']<best-1e-4:
            best=score['selection_score'];stale=0;temp=a.output/'best.tmp'
            torch.save(dict(schema='B1-action-entity-risk-v3',model=net.state_dict(),width=a.width,query_dim=QUERY_DIM,query_schema=QUERY_SCHEMA,
                epoch=epoch,Adam_steps=steps,held=score,native_acceptance=False),temp);os.replace(temp,a.output/'best.pt')
        else:stale+=1
        publish('fitting_pair_risk',**row,stale_epochs=stale);write(a.output/'history.json',history)
        if stale>=8:break
    best_payload=torch.load(a.output/'best.pt',map_location='cuda',weights_only=False);net.load_state_dict(best_payload['model']);net.eval()
    scene_counts=np.zeros(6)
    with torch.inference_mode():
        ix=indices['held']
        for begin in range(0,len(ix),16):
            chosen=ix[begin:begin+16];pred=action_clearances(net,torch.as_tensor(f[chosen],device='cuda'),torch.as_tensor(o[chosen],device='cuda'))
            y=torch.as_tensor(gold[chosen],device='cuda');near=y.abs()<8;pos=y<=0;neg=y>1
            scene_counts+=np.array([float((pred[near]-y[near]).abs().sum()),int(near.sum()),int((pred[pos]>=1).sum()),int(pos.sum()),int((pred[neg]<1).sum()),int(neg.sum())])
    scene=dict(states=len(indices['held']),boundary_MAE=scene_counts[0]/max(1,scene_counts[1]),
        dangerous_action_recall=1-scene_counts[2]/max(1,scene_counts[3]),safe_action_false_positive_rate=scene_counts[4]/max(1,scene_counts[5]),
        danger_labels=int(scene_counts[3]),safe_labels=int(scene_counts[5]),all96_entities_checked=True)
    result=dict(status='completed',utc=utc_now(),epochs=len(history),Adam_steps=steps,best_epoch=best_payload['epoch'],
        selected_Adam_steps=best_payload['Adam_steps'],best_pair_held=best_payload['held'],full_scene_held=scene,
        source_counts=source_counts,parameters=sum(x.numel() for x in net.parameters()),
        native_instances=0,original_B1_modified=False,native_efficacy_demonstrated=False,wall_seconds=time.monotonic()-started)
    write(a.output/'result.json',result);publish('finished',**{k:v for k,v in result.items() if k not in ('status','utc','native_instances')});print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ['data','split','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed',type=int,default=271);p.add_argument('--width',type=int,default=128)
    p.add_argument('--train-states',type=int,default=8192);p.add_argument('--held-states',type=int,default=2048)
    p.add_argument('--learning-rate',type=float,default=.001);p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--steps-per-epoch',type=int,default=200)
    run(p.parse_args())
