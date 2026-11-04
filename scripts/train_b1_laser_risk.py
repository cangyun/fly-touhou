"""CUDA fitting of conservative segment proxies, with sealed native geometry.

No original-game worlds, policy updates, test-game replay fitting or controls.
Native laser shapes/motions come only from the earlier on-policy TRAIN logs;
player routes are deliberately modeled placements, never native outcomes.
"""
from pathlib import Path
import argparse,json,os,sys,time,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import utc_now


def load_native(folder):
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    rows=[];sources=[]
    for path in sorted(folder.glob('worker*.jsonl')):
        worker=int(path.stem.removeprefix('worker'))
        with _shared_binary(path) as f:raw=f.read()
        complete=raw[:raw.rfind(b'\n')+1]
        sources.append(dict(path=str(path),sha256=hashlib.sha256(complete).hexdigest()))
        for line in complete.splitlines():
            row=json.loads(line)
            if (row.get('schema')!='th14-measured-laser-motion-pairs-v1'
                    or not row.get('age_continuous') or row['elapsed_frames']!=4
                    or row['before']['node_count'] or row['after']['node_count']):continue
            row['worker']=worker;rows.append(row)
    if not rows:raise ValueError('Sealed native training laser pairs required')
    return rows,sources


def cases(rng,rows,count,*,native_fraction=.5):
    n=count;choose=rng.random(n)<native_fraction
    angle=rng.uniform(-np.pi,np.pi,n);direction=np.stack((np.cos(angle),np.sin(angle)),-1)
    length=rng.uniform(.2,1000,n);a=rng.uniform([-250,-50],[250,500],(n,2));b=a+direction*length[:,None]
    va=rng.uniform(-10,10,(n,2));vb=va+rng.normal(0,3,(n,2));width=rng.uniform(.25,28,n)
    rate=rng.uniform(-1,1,n);known=rng.random(n)>.1;state=rng.integers(0,6,n)/8;iframes=np.zeros(n)
    for i in np.flatnonzero(choose):
        row=rows[int(rng.integers(len(rows)))];r=row['after'];theta=r['angle']
        a[i]=(r['x'],r['y']);b[i]=a[i]+r['length']*np.asarray((np.cos(theta),np.sin(theta)))
        va[i]=row['measured_origin_velocity'];vb[i]=row['measured_endpoint_velocity']
        width[i]=r['width']/2;rate[i]=row['measured_width_rate']/2
        state[i]=r['native_state']/(abs(r['native_state'])+8) if r['native_state'] is not None else -1
        counter=r['native_iframes'];iframes[i]=counter/(abs(counter)+60) if counter is not None else -1
    half=rng.uniform(.5,5,(n,2));offset=rng.uniform(-1,1,(n,2));circle=rng.uniform(.5,8,n)
    segment=b-a;norm=np.linalg.norm(segment,axis=-1).clip(1e-9)
    normal=np.stack((-segment[:,1],segment[:,0]),-1)/norm[:,None]
    fraction=rng.uniform(-.15,1.15,n)
    starts=a+fraction[:,None]*segment+normal*rng.normal(0,14,(n,1))
    uniform=rng.random(n)<.2;starts[uniform]=rng.uniform([-184,32],[184,432],(int(uniform.sum()),2))
    starts=np.clip(starts,[-184,32],[184,432])
    from fly_touhou.thrl_movement_teacher import action_velocities
    velocities=action_velocities()[rng.integers(18,size=n)]
    # Additional synthetic near-contact scenes deliberately cover every legal
    # player location, including long segment interiors and clamped walls.
    local=(~choose)&(rng.random(n)<.7)
    shift=(starts-(a+fraction[:,None]*segment))+normal*rng.normal(0,10,(n,1))
    a[local]+=shift[local];b[local]+=shift[local]
    return (starts,velocities,a,b,va,vb,width,rate,half,offset,circle,known,state,iframes),choose


def run(a):
    import torch,psutil
    from fly_touhou.thrl_laser_risk import LaserRiskNetwork,segment_query_features,QUERY_SCHEMA
    from fly_touhou.thrl_laser_risk_labels import segment_clearance
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    if a.output.exists():raise ValueError('Never overwrite a fitted readout')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.18)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    if sys.platform=='win32':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    a.output.mkdir(parents=True);started=time.monotonic()
    write(a.output/'identity.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),native_instances=0))
    rows,sources=load_native(a.source)
    split={name:[r for r in rows if condition(r['worker'])] for name,condition in
        [('train',lambda x:x<20),('heldout',lambda x:20<=x<25),('test',lambda x:25<=x<30)]}
    if any(not v for v in split.values()):raise ValueError('Native worker-group holdouts required')
    write(a.output/'plan.json',dict(schema='B1-observed-laser-risk-proxy-training-v1',seed=a.seed,
        device='cuda:0',gpu=torch.cuda.get_device_name(0),native_instances=0,source=sources,
        native_pair_counts={k:len(v) for k,v in split.items()},source_is_training_geometry=True,
        synthetic_player_placements=True,native_outcome_labels=False,native_collision_state_verified=False,
        query_schema=QUERY_SCHEMA,train_examples=a.train_examples,heldout_examples=a.held_examples,
        native_worker_split=dict(train=list(range(20)),heldout=list(range(20,25)),test=list(range(25,30)))))
    rng=np.random.default_rng(a.seed);data={}
    for name,n in [('train',a.train_examples),('heldout',a.held_examples),('test',a.held_examples)]:
        values,native=cases(rng,split[name],n)
        y=segment_clearance(*values[:12]);q=[]
        for first in range(0,n,8192):
            tensors=[torch.as_tensor(np.array(x[first:first+8192],copy=True),device='cuda',dtype=torch.float32) for x in values]
            with torch.inference_mode():q.append(segment_query_features(*tensors).cpu())
        data[name]=(torch.cat(q),torch.from_numpy(np.clip(y,-64,64).astype(np.float32)))
        write(a.output/(name+'_distribution.json'),dict(examples=n,native_geometry=int(native.sum()),
            modeled_intersection=int((y<=0).sum()),near_boundary=int((abs(y)<8).sum()),
            unknown_motion=int((~values[11]).sum())))
    net=LaserRiskNetwork(a.width).cuda();optimizer=torch.optim.Adam(net.parameters(),lr=a.learning_rate)
    def assess(name):
        q,y=data[name];errors=[];predictions=[]
        with torch.inference_mode():
            for i in range(0,len(q),8192):predictions.append(net(q[i:i+8192].cuda()).cpu())
        p=torch.cat(predictions);near=abs(y)<8;danger=y<=0;safe=y>1
        result=dict(MAE=float(abs(p-y).mean()),boundary_MAE=float(abs(p[near]-y[near]).mean()),
            dangerous_recall=float((p[danger]<1).float().mean()),
            safe_false_positive=float((p[safe]<1).float().mean()),danger_examples=int(danger.sum()),safe_examples=int(safe.sum()))
        result['selection_score']=result['boundary_MAE']+20*(1-result['dangerous_recall'])+2*result['safe_false_positive']
        return result
    q,y=data['train'];history=[];steps=0;best=float('inf');stale=0
    for epoch in range(1,a.epochs+1):
        if (a.output/'STOP').exists():raise RuntimeError('Operator stopped laser proxy fitting')
        net.train();order=torch.randperm(len(q))
        rate=a.learning_rate*(.2+.8*(1-(epoch-1)/a.epochs))
        for group in optimizer.param_groups:group['lr']=rate
        for i in range(0,len(order),a.batch_size):
            ix=order[i:i+a.batch_size];target=y[ix].cuda();prediction=net(q[ix].cuda())
            weight=1+5*(abs(target)<8).to(target.dtype)+3*(target<=0).to(target.dtype)
            loss=(torch.nn.functional.smooth_l1_loss(prediction,target,reduction='none',beta=.5)*weight).sum()/weight.sum()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite laser risk loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite laser risk gradient')
            optimizer.step();steps+=1
        net.eval();held=assess('heldout');row=dict(epoch=epoch,Adam_steps=steps,learning_rate=rate,**held);history.append(row)
        if held['selection_score']<best-1e-4:
            best=held['selection_score'];stale=0
            torch.save(dict(schema='B1-learned-laser-segment-risk-v1',query_schema=QUERY_SCHEMA,width=a.width,
                model=net.state_dict(),epoch=epoch,Adam_steps=steps,heldout=held,
                native_acceptance=False,native_collision_state_verified=False),a.output/'best.pt')
        else:stale+=1
        write(a.output/'history.json',history)
        write(a.output/'progress.json',dict(status='training',stage='CUDA-laser-proxy-fitting',utc=utc_now(),native_instances=0,**row))
        if stale>=10:break
    payload=torch.load(a.output/'best.pt',map_location='cuda',weights_only=False);net.load_state_dict(payload['model']);net.eval()
    test=assess('test')
    qualified=(test['boundary_MAE']<1.5 and test['dangerous_recall']>=.98 and test['safe_false_positive']<=.15)
    result=dict(schema='B1-laser-risk-proxy-fit-result-v1',status='completed',utc=utc_now(),qualified_proxy_only=qualified,
        epochs=len(history),Adam_steps=steps,selected_Adam_steps=payload['Adam_steps'],best_epoch=payload['epoch'],
        heldout=payload['heldout'],test=test,parameters=sum(p.numel() for p in net.parameters()),
        checkpoint_sha256=hashlib.sha256((a.output/'best.pt').read_bytes()).hexdigest(),native_instances=0,
        native_efficacy_demonstrated=False,native_collision_state_verified=False,
        actor_or_PPO_updates=0,wall_seconds=time.monotonic()-started)
    write(a.output/'result.json',result);write(a.output/'progress.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ['source','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed',type=int,default=31863);p.add_argument('--width',type=int,default=64)
    p.add_argument('--train-examples',type=int,default=196608);p.add_argument('--held-examples',type=int,default=32768)
    p.add_argument('--batch-size',type=int,default=4096);p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--learning-rate',type=float,default=.001);run(p.parse_args())
