"""Short, evidence-gated compact111 whole-connectome SNN experiment.

64 risk-supervised updates -> frozen native20 ->64 joint updates ->native20.
If needed, one2048-decision current-student DAgger collection and64 updates.
No map, CNN, object Transformer, PPO, frame-limited native tests or teacher
action substitution. All learning passes through the complete spiking brain.
"""
from pathlib import Path
import argparse,copy,gc,hashlib,json,math,os,sys,time,traceback
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','connectome','environment-config','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--seed',type=int,default=202610307)
    p.add_argument('--candidate',action='store_true')
    p.add_argument('--extra-data',type=Path)
    p.add_argument('--baseline-report',type=Path)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:
        print(dict(status='prepared',input=111,maps=False,phase_updates=[64,64,64],maximum_new_student_actions=2048));return
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    from fly_touhou.owned_worker_affinity import worker_cpu_pool
    current=psutil.Process();worker_cpus=worker_cpu_pool(current.cpu_affinity())
    current.cpu_affinity(current.cpu_affinity()[-2:]);current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    config=read(a.environment_config)
    for process in psutil.process_iter(['name','cmdline']):
        if (process.info['name'] or '').lower()==Path(config['game_exe']).name.lower():raise RuntimeError('Another original game is active')
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-compact111-'+str(current.pid))
    a.output.mkdir(parents=True,exist_ok=False);begun=time.monotonic();updates=0;new_actions=0
    launch=dict(schema='compact111-fullbrain-spiking-short-cycle-v1',PID=current.pid,create_time=current.create_time(),
        started_UTC=datetime.now(timezone.utc).isoformat(),argv=sys.argv,CUDA_started=False,native_started=False,
        input_features=111,map_in_student=False,optimizer_budget=64 if a.candidate else 192,new_student_action_budget=0 if a.candidate else 2048,
        candidate_conditioned=a.candidate)
    write_json(a.output/'launch.json',launch)
    write_json(a.output/'source-pins.json',{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        Path(__file__),ROOT/'src/fly_touhou/compact_spiking_sensation.py',ROOT/'src/fly_touhou/compact_spiking_model.py',
        ROOT/'src/fly_touhou/compact_spiking_run.py',ROOT/'src/fly_touhou/compact_spiking_native.py',ROOT/'src/fly_touhou/continuous_sparse.py',
        ROOT/'src/fly_touhou/candidate_spiking_model.py')})
    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-begun>14400:raise TimeoutError('Finite short-cycle wall budget')
    def progress(phase,**extra):
        write_json(a.output/'progress.json',dict(phase=phase,joint_updates=updates,new_student_actions=new_actions,
            elapsed_seconds=time.monotonic()-begun,**extra))
    try:
        import numpy as np
        import torch
        if a.candidate:
            from fly_touhou.candidate_spiking_model import build_compact_model,sequence,risk_supervision
        else:
            from fly_touhou.compact_spiking_model import build_compact_model,sequence,risk_supervision
        from fly_touhou.compact_spiking_data import Streams
        from fly_touhou.compact_spiking_run import evaluate,acquire
        from fly_touhou.continuous_object_avoidance import GUIDANCE_CONFIG
        from fly_touhou.fly_gru_aligned_guidance import gru_aligned_guidance_loss
        from fly_touhou.whole_brain_ppo import clip_gradients_fp64
        from fly_touhou.thrl_job_paths import configure_native_temp
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
        configure_native_temp(a.output)
        progress('building_compact_fullbrain_SNN')
        model=build_compact_model(a.connectome,seed=a.seed)
        groups=[dict(name='brain',params=list(model.core.internal_parameters()),lr=.0003,eps=1e-10),
            dict(name='ports',params=list(model.core.input_encoder.parameters()),lr=.0005),
            dict(name='risk',params=list(model.risk.parameters()),lr=.001),dict(name='actor',params=list(model.actor.parameters()),lr=.001)]
        optimizer=torch.optim.Adam(groups,weight_decay=0.)
        manifest=read(a.data/'manifest.json');train=[r for r in manifest['records'] if r['fold']=='train'];held=[r for r in manifest['records'] if r['fold']=='heldout']
        if a.extra_data is not None:
            extra=read(a.extra_data/'manifest.json')['records'];train += [r for r in extra if r['fold']=='train'];held += [r for r in extra if r['fold']=='heldout']
        stream=Streams(train,seed=a.seed);state=model.initial_state(16);nodes=[]
        launch.update(CUDA_started=True,CUDA_device=torch.cuda.get_device_name());write_json(a.output/'launch.json',launch)
        write_json(a.output/'algorithm.json',dict(core=model.core.config(),map=False,CNN=False,object_Transformer=False,
            stimulus='bounded111 -> opponent VP populations plus trainable residual ->hard spikes',
            readout=('shared linear3 horizon risks per candidate;all18 evaluated with same brain/readout;selected following state only' if a.candidate
                else 'linear18 risk from1299 DN filtered spikes + membrane;linear19 actor;three DN value heads retained'),
            risk_loss='bounded danger-hint Huber +.2 known-danger BCE,query pools25/50/25',
            composition=('shared neural4/12/24 risks -> movement scores;predicted immediate-safe subset when available;neutral stay prior' if a.candidate else
                'actor -8*learned danger probability;physical risk cost detached from policy-guidance gradients'),
            phases=(['64 risk-query updates,brain/ports/shared3-risk all trainable,then original20;no long continuation'] if a.candidate else
                ['64 risk calibration,brain/ports/risk all trainable','64 joint original GRU movement guidance',
                'if needed2048 current-student decisions,teacher annotations only,then64 joint updates']),
            BPTT=16,neural_ticks=model.core.K,neural_batch=32 if a.candidate else 16,
            candidate_conditioned=a.candidate,candidate_neural_inputs=8 if a.candidate else None,
            formal_gate='NoBomb first-stage clean8/20 plus separate8/20;also compare stationary behavior',
            all_native_hazards=True,no_96_object_limit_at_runtime=True,old_data_are_existing_training_worlds_only=True,
            full_story_training_started=False,PPO_started=False))
        def save(tag):
            payload=model.parameter_payload(dict(joint_updates=updates,new_student_actions=new_actions))
            path=a.output/(tag+'-parameters.pt');temp=path.with_suffix('.pt.tmp')
            with temp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
            os.replace(temp,path)
            training=a.output/(tag+'-training.pt');temp=training.with_suffix('.pt.tmp')
            with temp.open('wb') as f:torch.save(dict(model_payload=payload,optimizer=optimizer.state_dict(),stream=stream.state_dict(),
                state=state.detach().to('cpu'),torch_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state_all()),f);f.flush();os.fsync(f.fileno())
            os.replace(temp,training);return path
        def tensors(raw):return {k:torch.from_numpy(v).to(model.device) for k,v in raw.items()}
        def task_of(data):return {**{k:data[k] for k in ('margins','trusted','unprotected_mask','outcomes','rule_probabilities','rule_weights')},
            'deathbomb_mask':torch.zeros_like(data['valid'])}
        def fit(count,joint):
            nonlocal state,updates
            joint=joint and not a.candidate
            model.train();model.actor.requires_grad_(joint);model.critic.requires_grad_(False)
            for _ in range(count):
                guard();data=tensors(stream.next());task=task_of(data)
                optimizer.zero_grad(set_to_none=True)
                result=sequence(model,data['features'],data['episode_starts'],state,actions=data['actions'])
                risk,metrics=risk_supervision(result,task,data['valid'])
                guide=gru_aligned_guidance_loss(result['logits'],**task,actions=data['actions'],valid_mask=data['valid'],config=GUIDANCE_CONFIG)['total_loss'] if joint else risk*0
                loss=risk+guide
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite compact SNN loss')
                loss.backward();norms={g['name']:clip_gradients_fp64(g['params'],5.) for g in groups}
                if not all(math.isfinite(v) for v in norms.values()):raise FloatingPointError('Nonfinite actual SNN gradient')
                optimizer.step();state=result['state'].detach();updates+=1
                row=dict(step=updates,phase='joint' if joint else 'risk',loss=float(loss.detach()),risk_loss=float(risk.detach()),guide_loss=float(guide.detach()),
                    **metrics,VP_spike_fraction=float(state.spikes[:,model.core.input_indices].mean()),
                    DN_rate=float(state.rate[:,model.core.output_indices].mean()),DN_membrane_std=float(state.membrane[:,model.core.output_indices].std()),
                    gradients=norms,elapsed_seconds=time.monotonic()-begun)
                append_jsonl(a.output/'learning.jsonl',row);progress('compact_joint_learning' if joint else 'compact_risk_calibration',learning=row)
                del data,task,result,risk,guide,loss
                if updates%16==0:save('latest')
        @torch.no_grad()
        def heldout():
            model.eval();sampler=Streams(held,seed=a.seed+1);s=model.initial_state(16);recall=fp=nd=ns=0;unsafe=contrast=0
            while True:
                data=tensors(sampler.next());task=task_of(data);r=sequence(model,data['features'],data['episode_starts'],s,checkpointed=False,actions=data['actions']);s=r['state'].detach()
                mask=data['valid']&task['trusted'][...,0]&task['unprotected_mask'];m=task['margins'][...,0,:]
                danger=mask[...,None]&(m<=0);safe=mask[...,None]&(m>1);pred=r['predicted_margin']<=0
                recall+=int((danger&pred).sum());fp+=int((safe&pred).sum());nd+=int(danger.sum());ns+=int(safe.sum())
                mixed=mask&(m.min(-1).values<=0)&(m.max(-1).values>1);chosen=r['logits'].argmax(-1)
                unsafe+=int((mixed&(m.gather(-1,chosen.clamp_max(17)[...,None])[...,0]<=0)).sum());contrast+=int(mixed.sum())
                if not sampler.queue and all(slot is None for slot in sampler.slots):break
            return dict(danger_recall=recall/max(nd,1),danger_queries=nd,safe_false_alarm=fp/max(ns,1),unsafe_chosen=unsafe,mixed_scenes=contrast)
        def test(tag):
            path=save(tag);model.eval();optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            offline=({'all_candidate_offline_sweep_skipped':True,'reason':'prioritize full native20 instead of another expensive18x replay'} if a.candidate else heldout())
            write_json(a.output/(tag+'-heldout.json'),offline)
            launch['native_started']=True;write_json(a.output/'launch.json',launch)
            report=evaluate(model,config,a.output/(tag+'-development20'),worker_cpus=worker_cpus,guard=guard,
                progress=lambda x:progress('frozen_compact_native20',evaluation=x))
            confirmation=None
            if report['first_stage_without_observed_contact_count']>=8:
                confirmation=evaluate(model,config,a.output/(tag+'-independent20'),seed=2028432926,worker_cpus=worker_cpus,guard=guard,
                    progress=lambda x:progress('independent_compact_native20',evaluation=x))
            node=dict(updates=updates,parameters=str(path),heldout=offline,development=report,confirmation=confirmation)
            nodes.append(node);write_json(a.output/'evaluations.json',dict(nodes=nodes))
            return confirmation is not None and confirmation['first_stage_without_observed_contact_count']>=8
        save('initial')
        if a.baseline_report is not None:
            baseline=read(a.baseline_report)
            if (baseline.get('status')!='completed' or baseline.get('policy')!='stationary' or baseline.get('episodes')!=20
                    or baseline.get('seed')!=2027432926 or baseline.get('input_schema')!='connectome-compact111-no-map-v1'):
                raise ValueError('Reuse only the matching completed original stationary20')
            write_json(a.output/'stationary-baseline-reference.json',dict(source=str(a.baseline_report),report=baseline))
        else:
            baseline=evaluate(None,config,a.output/'stationary-baseline20',worker_cpus=worker_cpus,guard=guard,
                progress=lambda x:progress('stationary_original_baseline20',evaluation=x))
        fit(64,False);passed=test('risk064')
        if not passed and not a.candidate:fit(64,True);passed=test('joint128')
        if not passed and not a.candidate:
            own=acquire(model,config,a.output/'current-student-DAgger',worker_cpus=worker_cpus,guard=guard,
                progress=lambda x:progress('compact_current_student_DAgger',acquisition=x))
            new_actions=2048
            # A new replay phase really begins at recorded actual openings.
            train += [r for r in own if r['fold']=='train'];held += [r for r in own if r['fold']=='heldout']
            stream=Streams(train,seed=a.seed+2);state=model.initial_state(16)
            fit(64,True);passed=test('DAgger192')
        save('final')
        best=max(nodes,key=lambda n:(n['development']['first_stage_without_observed_contact_count'],n['development']['observed_contact_frames']['median'] or 0))
        write_json(a.output/'result.json',dict(status='completed_finite_experiment',basic_native_gate_passed=passed,
            goal_complete_claimed=False,needs_comparison_to_stationary=True,baseline_clean=baseline['first_stage_without_observed_contact_count'],
            baseline_contact_median=baseline['observed_contact_frames']['median'],best_parameters=best['parameters'],
            best_clean=best['development']['first_stage_without_observed_contact_count'],joint_updates=updates,new_student_actions=new_actions,
            inputs=111,maps=False,all_source_neurons_and_edges_retained=True,PPO_Adam=0,full_story_started=False,
            elapsed_seconds=time.monotonic()-begun))
        progress('completed_finite_experiment')
    except BaseException as error:
        write_json(a.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),joint_updates=updates,new_student_actions=new_actions,no_automatic_retry=True));raise


if __name__=='__main__':main()
