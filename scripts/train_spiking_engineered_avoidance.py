"""Finite CUDA warmup and original NoBomb tests for the new LIF controller.

Teacher demonstrations and historical student failures are supervised data,
not current SNN rollouts or PPO samples. All sensory/core/actor/risk parameters
learn together. Three DN value heads are retained for subsequent on-policy PPO.
"""
from pathlib import Path
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','connectome','expert-data','historical-student-data','environment-config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-sha256',required=True)
    p.add_argument('--resume-training',type=Path)
    p.add_argument('--resume-training-sha256')
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--development-every',type=int,default=1000)
    p.add_argument('--first-development',type=int,default=250)
    p.add_argument('--seed',type=int,default=202610301)
    p.add_argument('--evaluation-seed',type=int,default=2027432926)
    p.add_argument('--max-seconds',type=float,default=28800)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if bool(a.resume_training)!=bool(a.resume_training_sha256):p.error('A full training checkpoint and its SHA256 are required together')
    if not a.execute:
        print(json.dumps(dict(status='prepared',CUDA=False,updates=a.updates,output=str(a.output))));return
    import psutil
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    from fly_touhou.thrl_study_io import read_shared_json as read
    current=psutil.Process()
    from fly_touhou.owned_worker_affinity import worker_cpu_pool
    worker_cpus=worker_cpu_pool(current.cpu_affinity())
    current.cpu_affinity(current.cpu_affinity()[-2:]);current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-SNN-engineered-'+str(current.pid))
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    costs=dict(joint_Adam=0,supervised_Adam=0,PPO_Adam=0,new_student_training_actions=0,
        teacher_learning_rows=0,historical_student_learning_rows=0)
    launch=dict(schema='engineered-fullbrain-LIF-supervised-native-v1',PID=current.pid,create_time=current.create_time(),
        argv=sys.argv,status='initializing',started_UTC=datetime.now(timezone.utc).isoformat(),CUDA_started=False,native_started=False)
    write_json(a.output/'launch.json',launch)
    pins={}
    for path in (Path(__file__), ROOT/'src/fly_touhou/spiking_engineered_control.py',ROOT/'src/fly_touhou/continuous_sparse.py',
                 ROOT/'src/fly_touhou/thrl_movement_teacher.py',ROOT/'src/fly_touhou/continuous_no_bomb_first_contact.py'):
        pins[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(a.output/'source-pins.json',pins)
    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-started>a.max_seconds:raise TimeoutError('Finite whole-phase wall budget')
    def progress(phase,**extra):write_json(a.output/'progress.json',dict(phase=phase,costs=dict(costs),elapsed_seconds=time.monotonic()-started,**extra))
    try:
        import numpy as np
        import torch
        from fly_touhou.spiking_engineered_control import (SCHEMA,SpikingState,SpikingNativePolicy,
            engineering_features,build_spiking_model,forward_sequence)
        from fly_touhou.chronological_avoidance_data import TeacherRows,FixedStudentTrainingRows,ChronologicalBatches,recorded_state_metrics
        from fly_touhou.no_bomb_teacher_data import NoBombTeacherDataset
        from fly_touhou.fly_gru_aligned_guidance import gru_aligned_guidance_loss
        from fly_touhou.continuous_object_avoidance import GUIDANCE_CONFIG
        from fly_touhou.continuous_signed_clearance import signed_clearance4f_terms
        from fly_touhou.continuous_no_bomb_first_contact import evaluate_no_bomb
        from fly_touhou.whole_brain_ppo import clip_gradients_fp64
        from fly_touhou.thrl_dodge_diagnostics import _shared_binary
        from fly_touhou.thrl_job_paths import configure_native_temp
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        cfg=read(a.environment_config)
        for process in psutil.process_iter(['pid','name','cmdline']):
            if (process.info['name'] or '').lower()==Path(cfg['game_exe']).name.lower():raise RuntimeError('Another native game is active')
            if process.pid!=current.pid and any('train_chronological_fullbrain_avoidance.py' in x for x in process.info['cmdline'] or []):
                raise RuntimeError('Previous CUDA training must exit first')
        configure_native_temp(a.output)
        with _shared_binary(a.source) as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=a.source_sha256:raise ValueError('Source SHA mismatch')
        with _shared_binary(a.source) as f:source=torch.load(f,map_location='cpu',weights_only=True)
        progress('building_trainable_spiking_brain')
        model=build_spiking_model(source,a.connectome,seed=a.seed);del source;gc.collect()
        groups=[dict(name='brain',params=list(model.core.internal_parameters()),lr=.0003,eps=1e-10),
            dict(name='sensory_ports',params=[*model.core.input_encoder.parameters(),*model.core.geometry_encoder.parameters(),*model.core.visual_norm.parameters()],lr=.0003),
            dict(name='perception',params=list(model.sensory_adapter.parameters()),lr=.0001),
            dict(name='actor',params=list(model.actor.parameters()),lr=.001),
            dict(name='risk',params=list(model.risk_head.parameters()),lr=.001),
            dict(name='three_values',params=list(model.critic.parameters()),lr=.0003)]
        ids=[id(x) for g in groups for x in g['params']]
        if len(ids)!=len(set(ids)) or set(ids)!={id(x) for x in model.parameters()}:raise ValueError('Disjoint exhaustive named optimizer groups required')
        optimizer=torch.optim.Adam(groups,weight_decay=0.)
        launch.update(status='training',CUDA_started=True,CUDA_device=torch.cuda.get_device_name());write_json(a.output/'launch.json',launch)
        dataset=NoBombTeacherDataset(a.expert_data)
        sources=dict(teacher=TeacherRows(dataset),historical=FixedStudentTrainingRows(a.historical_student_data))
        order=np.random.default_rng(a.seed).permutation(30).tolist()
        split=dict(teacher=dataset.split(),historical=dict(train=order[6:],heldout=order[:6]))
        class EngineeringStream(ChronologicalBatches):
            def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.engineering_cache={}
            def next(self):
                data,task,meta=super().next();e=np.zeros((*data['valid_mask'].shape,144),np.float32)
                for spec in meta['sources']:
                    record,first,stop,lane=(spec[k] for k in ('record','first','stop','lane'))
                    pieces=[]
                    for begin in range(first//32*32,(stop+31)//32*32,32):
                        key=record,begin
                        if key not in self.engineering_cache:
                            raw=self.source.fetch(record,begin,min(begin+32,self.source.lengths[record]))
                            self.engineering_cache[key]=engineering_features(raw['features'],raw['objects'],raw['maps'])
                        pieces.append(self.engineering_cache[key][max(first-begin,0):min(stop-begin,32)])
                    e[:stop-first,lane]=np.concatenate(pieces)
                data['engineering']=e
                return data,task,meta
        streams={k:EngineeringStream(v,split[k]['train'],seed=a.seed+i,width=16,length=16) for i,(k,v) in enumerate(sources.items())}
        states={k:model.initial_state(16) for k in streams}
        nodes=[]
        if a.resume_training is not None:
            # This is an explicitly pinned, locally produced full training state.
            # Preserve Adam, raw chronology, all four neural states and RNG.
            with _shared_binary(a.resume_training) as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=a.resume_training_sha256:raise ValueError('Full training checkpoint SHA mismatch')
            with _shared_binary(a.resume_training) as f:resumed=torch.load(f,map_location='cpu',weights_only=False)
            point=resumed['model_payload']
            if (resumed.get('schema')!=SCHEMA+'-training' or point['base_sha256']!=a.source_sha256 or point['seed']!=a.seed
                    or point['core']['graph']['graph_fingerprint']!=model.core.config()['graph']['graph_fingerprint']):
                raise ValueError('Cannot resume a different brain, seed, or source')
            model.load_state_dict(point['state_dict'],strict=True);optimizer.load_state_dict(resumed['optimizer'])
            costs.update(resumed['costs'])
            for key,stream in streams.items():
                stream.load_state_dict(resumed['streams'][key]);states[key]=resumed['states'][key].to(model.device)
            torch.set_rng_state(resumed['torch_RNG']);torch.cuda.set_rng_state_all(resumed['CUDA_RNG'])
            prior_evaluations=a.resume_training.parent/'evaluations.json'
            if prior_evaluations.exists():nodes=[n for n in read(prior_evaluations)['nodes'] if n['updates']<=costs['joint_Adam']]
            launch.update(resumed_from=str(a.resume_training),resumed_training_sha256=a.resume_training_sha256,
                resumed_completed_updates=costs['joint_Adam'],training_and_neural_history_restored=True)
            write_json(a.output/'launch.json',launch)
            del resumed,point;gc.collect()
            progress('full_training_checkpoint_restored')
        write_json(a.output/'algorithm.json',dict(schema=SCHEMA,core=model.core.config(),
            perception='raw maps+objects+numeric,all trainable;old weights initialize perception only',
            engineering='observation-only kinematic4/12/24 clearance,known flags,3continuation escape proxy,wall margin;no post-action facts',
            losses='original GRU safe/future/hit/rule plus unit signed4f Huber+BCE;all sensory/core/actor/risk joint',
            stage='supervised warmup,not PPO or fresh SNN DAgger',three_value_heads='created and preserved,trained by subsequent on-policy PPO only',
            clipping='independent5norm for each disjoint optimizer group;heads not scaled by brain gradient norm',
            teacher_and_historical_batches=[2,1],BPTT=16,neural_ticks=8,batch=16,max_grad_norm=5.,
            updates=a.updates,first_development=a.first_development,development_every=a.development_every,split=split,base_source=str(a.source),source_sha256=a.source_sha256,
            neuronal_restrictions='topology/direction/sign preserved;hard spikes;learnable edge amplitude,thresholds,leaks',
            reset='actual opening only;historical replay maintains full history;not synthetic stage endings'))
        def save(tag):
            payload=model.parameter_payload(a.source,a.source_sha256,costs)
            for suffix,value in [('parameters',payload),('training',dict(schema=SCHEMA+'-training',model_payload=payload,
                optimizer=optimizer.state_dict(),costs=dict(costs),streams={k:v.state_dict() for k,v in streams.items()},
                states={k:v.detach().to('cpu') for k,v in states.items()},torch_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state_all()))]:
                path=a.output/f'{tag}-{suffix}.pt';tmp=path.with_suffix('.pt.tmp')
                with tmp.open('wb') as file:torch.save(value,file);file.flush();os.fsync(file.fileno())
                os.replace(tmp,path)
            return a.output/f'{tag}-parameters.pt'
        save('initial')
        def test(tag):
            path=save(tag)
            with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
            model.eval().requires_grad_(False);optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            launch['native_started']=True;write_json(a.output/'launch.json',launch)
            policy=SpikingNativePolicy(model).eval()
            report=evaluate_no_bomb(torch,policy,cfg,a.output/(tag+'-development20'),checkpoint_sha256=digest,
                episodes=20,workers=20,seed=a.evaluation_seed,check_stop=guard,worker_cpus=worker_cpus,
                progress=lambda item:progress('frozen_spiking_NoBomb20',evaluation=item))
            confirmation=None
            if report['first_stage_without_observed_contact_count']>=8:
                confirmation=evaluate_no_bomb(torch,SpikingNativePolicy(model).eval(),cfg,a.output/(tag+'-independent20'),checkpoint_sha256=digest,
                    episodes=20,workers=20,seed=a.evaluation_seed+1000000,check_stop=guard,worker_cpus=worker_cpus,
                    progress=lambda item:progress('independent_spiking_NoBomb20',evaluation=item))
            node=dict(updates=costs['joint_Adam'],source=str(path),sha256=digest,development=report,confirmation=confirmation)
            nodes.append(node);write_json(a.output/'evaluations.json',dict(nodes=nodes))
            model.train().requires_grad_(True)
            return confirmation is not None and confirmation['first_stage_without_observed_contact_count']>=8
        passed=False
        completed=costs['joint_Adam']
        if completed and (completed==a.first_development or completed%a.development_every==0 or completed==a.updates):
            if not any(n['updates']==completed for n in nodes):passed=test(f'node-{completed:04}')
        for step in range(completed,a.updates):
            if passed:break
            guard();kind='teacher' if step%3<2 else 'historical'
            raw,labels,meta=streams[kind].next()
            data={k:torch.from_numpy(v).to(model.device) for k,v in raw.items()}
            task={k:torch.from_numpy(v).to(model.device) for k,v in labels.items()}
            optimizer.zero_grad(set_to_none=True)
            result=forward_sequence(model,data,data['engineering'],states[kind])
            guide=gru_aligned_guidance_loss(result['composed_logits'],**task,actions=data['actions'],valid_mask=data['valid_mask'],config=GUIDANCE_CONFIG)
            risk=signed_clearance4f_terms(result['predicted_margins'],task,data['features'],data['valid_mask'])
            loss=guide['total_loss']+risk['loss']
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite spiking supervision')
            loss.backward()
            if step==0 or (not (a.output/'first-hazard-gradient.json').exists() and int(risk['danger_pool_queries'])>0):
                gradients={g['name']:dict(tensors=len(g['params']),gradient_tensors=sum(x.grad is not None for x in g['params']),
                    norm=sum(float(x.grad.detach().double().square().sum()) for x in g['params'] if x.grad is not None)**.5) for g in groups}
                write_json(a.output/('initial-gradient.json' if step==0 else 'first-hazard-gradient.json'),gradients)
            norms={g['name']:clip_gradients_fp64(g['params'],5.) for g in groups}
            if not all(math.isfinite(x) for x in norms.values()):raise FloatingPointError('Nonfinite spiking gradient group')
            norm=max(norms.values())
            optimizer.step();states[kind]=result['following_state'].detach()
            costs['joint_Adam']+=1;costs['supervised_Adam']+=1
            costs['teacher_learning_rows' if kind=='teacher' else 'historical_student_learning_rows']+=meta['real_rows']
            row=dict(step=costs['joint_Adam'],stream=kind,loss=float(loss.detach()),risk_loss=float(risk['loss'].detach()),
                guide_loss=float(guide['total_loss'].detach()),gradient_norm=norm,real_rows=meta['real_rows'],
                population_spike_fraction=float(states[kind].spikes.mean()),mean_membrane=float(states[kind].membrane.mean()),
                DN_rate_mean=float(states[kind].rate[:,model.core.output_indices].mean()),
                DN_membrane_std=float(states[kind].membrane[:,model.core.output_indices].std()),elapsed_seconds=time.monotonic()-started)
            row.update(recorded_state_metrics(result,data,task))
            row.update({'gradient_'+name:value for name,value in norms.items()})
            append_jsonl(a.output/'learning.jsonl',row);progress('joint_spiking_supervision',learning=row)
            del raw,labels,data,task,result,guide,risk,loss
            if (step+1)%25==0:save('latest')
            if step+1==a.first_development or (step+1)%a.development_every==0 or step+1==a.updates:
                if test(f'node-{step+1:04}'):
                    passed=True;break
        save('final');write_json(a.output/'result.json',dict(status='completed',costs=costs,evaluations=nodes,
            basic_avoidance_target_met=passed,full_story_training_started=False,elapsed_seconds=time.monotonic()-started))
        launch['status']='completed';write_json(a.output/'launch.json',launch);progress('completed')
    except BaseException as error:
        write_json(a.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),costs=costs,
            no_automatic_retry=True,elapsed_seconds=time.monotonic()-started));raise


if __name__=='__main__':main()
