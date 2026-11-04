"""Original GRU avoidance guidance on chronological raw demonstrations/states.

One unchanged complete trainable brain; no PPO labels from teacher actions,
no risk-ranking loss, no frozen CNN. CUDA starts after the identified corpus
collector exits. The independent uncapped original-game evaluation decides
whether basic avoidance has actually been learned.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
SCHEMA='chronological-fullbrain-original-GRU-avoidance-supervision-v1'


def arguments(argv=None):
    p=argparse.ArgumentParser(__doc__)
    for key in ('source','connectome','environment-config','expert-data','output'):
        p.add_argument('--'+key,type=Path,required=True)
    own=p.add_mutually_exclusive_group(required=True)
    own.add_argument('--own-training-trace',type=Path)
    own.add_argument('--current-student-corpus',type=Path)
    p.add_argument('--after-student-pid',type=int)
    p.add_argument('--after-student-create-time',type=float)
    p.add_argument('--teacher-batches-per-cycle',type=int,default=3)
    p.add_argument('--own-batches-per-cycle',type=int,default=1)
    p.add_argument('--risk-readout-substeps',type=int,default=0)
    p.add_argument('--risk-readout-initial-learning-rate',type=float,default=.001)
    p.add_argument('--development-every',type=int,default=0)
    p.add_argument('--risk-only-pretraining',action='store_true')
    p.add_argument('--actor-calibration-updates',type=int,default=0)
    p.add_argument('--own-acquisition-already-counted',action='store_true')
    p.add_argument('--max-grad-norm',type=float,default=.5)
    p.add_argument('--source-sha256',required=True)
    p.add_argument('--after-corpus-pid',type=int,required=True)
    p.add_argument('--after-corpus-create-time',type=float,required=True)
    p.add_argument('--updates',type=int,default=600)
    p.add_argument('--readout-learning-rate',type=float,default=3e-4)
    p.add_argument('--brain-learning-rate',type=float,default=3e-5)
    p.add_argument('--brain-epsilon',type=float,help='Explicit brain-only Adam denominator change;inherited moments remain intact')
    p.add_argument('--seed',type=int,default=2026102963)
    p.add_argument('--evaluation-seed',type=int,default=2027432926)
    p.add_argument('--max-seconds',type=int,default=28800)
    p.add_argument('--tensorboard-config',type=Path)
    p.add_argument('--execute',action='store_true')
    return p.parse_args(argv)


def checkpoint_payload(model,optimizer,context,costs,algorithm,provenance,streams,states):
    import torch
    from fly_touhou.continuous_guided_dagger import SCHEMA as GUIDED,parameter_payload
    from fly_touhou.guided_optimizer_io import parameter_signature
    total=copy.deepcopy(context['inherited'])
    total['supervised_Adam']=total.get('supervised_Adam',0)+costs['supervised_Adam']
    total['brain_updates_in_this_phase']=costs['joint_Adam']
    total['risk_readout_extra_Adam']=total.get('risk_readout_extra_Adam',0)+costs['risk_readout_Adam']
    total['actor_readout_extra_Adam']=total.get('actor_readout_extra_Adam',0)+costs['actor_readout_Adam']
    total['student_actions']=total.get('student_actions',0)+costs['new_student_training_actions']
    if costs['new_student_training_actions']:
        total['completed_rounds']=total.get('completed_rounds',0)+1
    payload=parameter_payload(model,context['base_source'],context['base_sha256'],context['calibration'],total)
    return dict(schema=GUIDED+'-training',model_payload=payload,optimizer=optimizer.state_dict(),
        optimizer_parameter_signature=parameter_signature(model,optimizer),costs=total,
        phase_costs=copy.deepcopy(costs),algorithm=copy.deepcopy(algorithm),provenance=copy.deepcopy(provenance),
        numpy_RNG=copy.deepcopy(context['numpy_RNG']),torch_RNG=torch.get_rng_state(),
        CUDA_RNG=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        chronological_streams={key:value.state_dict() for key,value in streams.items()},
        chronological_activity={key:value.activity.detach().cpu().clone() for key,value in states.items()},
        neural_midgame_resume=False,offline_teacher_history_saved=True,teacher_actions_at_inference=0)


def execute(a):
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    current=psutil.Process()
    from fly_touhou.owned_worker_affinity import worker_cpu_pool
    worker_cpus=worker_cpu_pool(current.cpu_affinity())
    current.cpu_affinity(current.cpu_affinity()[-2:]);current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-chronological-avoidance-'+str(current.pid))
    a.output.mkdir(parents=True,exist_ok=False)
    begun=time.monotonic();model=optimizer=None;streams={};states={};context=None;algorithm=None
    costs=dict(supervised_Adam=0,joint_Adam=0,risk_only_Adam=0,actor_readout_Adam=0,
               risk_readout_Adam=0,risk_readout_query_visits=0,
               teacher_learning_rows=0,own_state_learning_rows=0,
               new_student_training_actions=0,PPO_Adam=0)
    launch=dict(schema=SCHEMA,PID=current.pid,create_time=current.create_time(),argv=sys.argv,
        started_UTC=datetime.now(timezone.utc).isoformat(),status='waiting_for_corpus',
        CUDA_started=False,native_started=False)
    write_json(a.output/'launch.json',launch)
    def guard():
        if (a.output/'STOP').exists() or (a.expert_data/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-begun>a.max_seconds:raise RuntimeError('Finite phase wall budget')
    def progress(phase,**extra):
        write_json(a.output/'progress.json',dict(schema=SCHEMA,phase=phase,costs=copy.deepcopy(costs),
            elapsed_seconds=time.monotonic()-begun,**extra))
    def after_live():
        try:return abs(psutil.Process(a.after_corpus_pid).create_time()-a.after_corpus_create_time)<.01
        except psutil.NoSuchProcess:return False
    def save(tag):
        import torch
        value=checkpoint_payload(model,optimizer,context,costs,algorithm,provenance,streams,states)
        for suffix,data in (('training',value),('parameters',value['model_payload'])):
            path=a.output/f'{tag}-{suffix}.pt';temp=path.with_suffix('.pt.tmp')
            with temp.open('wb') as file:
                torch.save(data,file);file.flush();os.fsync(file.fileno())
            os.replace(temp,path)
        return a.output/f'{tag}-parameters.pt'
    def frozen_node(tag,development_folder,confirmation_folder):
        import torch
        from fly_touhou.continuous_no_bomb_first_contact import make_no_bomb_policy,evaluate_no_bomb
        path=save(tag);optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        model.eval().requires_grad_(False)
        versions=tuple(p._version for p in model.parameters())
        def frozen_guard():
            guard()
            if tuple(p._version for p in model.parameters())!=versions:raise RuntimeError('Frozen policy changed')
        with path.open('rb') as file:digest=hashlib.file_digest(file,'sha256').hexdigest()
        try:
            progress('frozen_NoBomb_development',node=tag)
            launch['native_started']=True;write_json(a.output/'launch.json',launch)
            policy=make_no_bomb_policy(torch,model,family='deep',episodes=20)
            report=evaluate_no_bomb(torch,policy,config,a.output/development_folder,checkpoint_sha256=digest,
                episodes=20,workers=20,seed=a.evaluation_seed,check_stop=frozen_guard,worker_cpus=worker_cpus,
                progress=lambda row:progress('frozen_NoBomb_development',node=tag,evaluation=row))
            confirmation=None
            if report['first_stage_without_observed_contact_count']>=8:
                progress('independent_NoBomb_confirmation',node=tag)
                confirmation=evaluate_no_bomb(torch,policy,config,a.output/confirmation_folder,checkpoint_sha256=digest,
                    episodes=20,workers=20,seed=a.evaluation_seed+1000000,check_stop=frozen_guard,worker_cpus=worker_cpus,
                    progress=lambda row:progress('independent_NoBomb_confirmation',node=tag,evaluation=row))
            return dict(joint_step=costs['joint_Adam'],parameter_path=str(path),development=report,
                        confirmation=confirmation,basic_avoidance_target_met=confirmation is not None and
                        confirmation['first_stage_without_observed_contact_count']>=8)
        finally:
            model.train().requires_grad_(True)
    try:
        predecessor=read(a.expert_data/'launch.json')
        if predecessor['PID']!=a.after_corpus_pid or abs(predecessor['create_time']-a.after_corpus_create_time)>.01:
            raise ValueError('Explicit corpus executor identity differs')
        while after_live():
            guard();progress('waiting_for_complete_NoBomb_raw_corpus',CUDA_started=False,native_worlds=0)
            time.sleep(5)
        if (a.expert_data/'failure.json').exists():raise RuntimeError('Corpus failed;no automatic neural retry')
        if read(a.expert_data/'native_cleanup.json').get('close_errors')!=[]:
            raise RuntimeError('Corpus native cleanup incomplete')
        if a.current_student_corpus:
            own_launch=read(a.current_student_corpus/'launch.json')
            if (own_launch['PID']!=a.after_student_pid
                    or abs(own_launch['create_time']-a.after_student_create_time)>.01):
                raise ValueError('Explicit current student acquisition executor differs')
            while True:
                try:live=abs(psutil.Process(a.after_student_pid).create_time()-a.after_student_create_time)<.01
                except psutil.NoSuchProcess:live=False
                if not live:break
                guard();progress('waiting_for_current_student_acquisition',CUDA_started=False,native_worlds=0)
                time.sleep(5)
            if (a.current_student_corpus/'failure.json').exists():
                raise RuntimeError('Current student acquisition failed;no automatic neural retry')
            if read(a.current_student_corpus/'native_cleanup.json').get('close_errors')!=[]:
                raise RuntimeError('Current student native cleanup incomplete')
        config=read(a.environment_config)
        if any((p.info['name'] or '').lower()==Path(config['game_exe']).name.lower()
               for p in psutil.process_iter(['name'])):
            raise RuntimeError('Another original-game job is active')
        from fly_touhou.thrl_job_paths import configure_native_temp
        configure_native_temp(a.output)
        import numpy as np
        import torch
        import math
        from fly_touhou.thrl_dodge_diagnostics import _shared_binary
        from fly_touhou.no_bomb_teacher_data import NoBombTeacherDataset
        from fly_touhou.chronological_avoidance_data import (TeacherRows,OwnTrainingRows,FixedStudentTrainingRows,ChronologicalBatches,
            original_guidance_block,alternating_original_guidance_block,recorded_state_metrics)
        from fly_touhou.continuous_guided_PPO_bridge import restore_guided_PPO_source
        from fly_touhou.continuous_connectome import ContinuousConnectomeState
        from fly_touhou.whole_brain_ppo import clip_gradients_fp64
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        dataset=NoBombTeacherDataset(a.expert_data);split=dataset.split()
        teacher=TeacherRows(dataset)
        own=FixedStudentTrainingRows(a.current_student_corpus) if a.current_student_corpus else OwnTrainingRows(a.own_training_trace)
        if a.current_student_corpus and not a.own_acquisition_already_counted:
            costs['new_student_training_actions']=own.actual_student_actions
        order=np.random.default_rng(a.seed).permutation(30).tolist()
        own_split=dict(train=order[6:],heldout=order[:6],whole_worlds_separated=True)
        provenance=dict(source=str(a.source),source_sha256=a.source_sha256,
            corpus=str(a.expert_data),corpus_manifest_sha256=dataset.manifest_sha256,
            teacher_split=split,own_state_source=str(a.current_student_corpus or a.own_training_trace),own_state_sha256=own.identity,
            own_split=own_split,development_observations_used_for_training=False)
        if a.current_student_corpus:
            provenance.update(own_acquisition_source_parameters_sha256=own.source_parameters_sha256,
                own_chronological_rows_per_world=own.lengths[0],own_rollouts=len(own.parts),
                own_state_source_is_current_student=not a.own_acquisition_already_counted)
        write_json(a.output/'data-split.json',provenance)
        with _shared_binary(a.source) as file:
            if hashlib.file_digest(file,'sha256').hexdigest()!=a.source_sha256:raise ValueError('Training source SHA differs')
        progress('restoring_complete_brain_and_named_Adam')
        with _shared_binary(a.source) as file:source=torch.load(file,map_location='cpu',weights_only=False)
        model,optimizer,inherited,context=restore_guided_PPO_source(source,a.connectome,device='cuda:0')
        del source
        if (model.core.n_neurons,model.core.n_edges,len(model.core.input_indices),len(model.core.output_indices))!=(138639,15091983,8053,1299):
            raise ValueError('Same complete brain, all visual ports and descending outputs required')
        model.train().requires_grad_(True)
        if a.risk_only_pretraining:
            model.actor.requires_grad_(False);model.critic.requires_grad_(False)
        for group in optimizer.param_groups:
            if group['name'] in ('actor','risk'):group['lr']=a.readout_learning_rate
            elif group['name']=='brain':
                group['lr']=a.brain_learning_rate
                if a.brain_epsilon is not None:group['eps']=a.brain_epsilon
        streams=dict(teacher=ChronologicalBatches(teacher,split['train'],seed=a.seed),
                     own=ChronologicalBatches(own,own_split['train'],seed=a.seed+1))
        states={key:ContinuousConnectomeState(torch.zeros(16,model.core.n_neurons,device=model.device)) for key in streams}
        algorithm=dict(schema=SCHEMA,PPO=False,returns=False,value_loss=False,
            critic_heads_retained=3,critic_parameters_preserved=True,critic_updated_in_this_supervised_phase=False,
            losses='original safe/future/hit/rule;deathbomb0;signed4f.05;no ranking or D95 distillation',
            raw_input='original FP32 features25/maps16x92x96/objects96x29',
            graph_neurons=model.core.n_neurons,graph_edges=model.core.n_edges,K=8,BPTT=16,
            all_8053_visual_ports=True,DN_outputs=1299,brain_and_all_sensory_trainable=True,
            policy_cost_gradient=model.policy_cost_gradient,readout_learning_rate=a.readout_learning_rate,
            brain_learning_rate=a.brain_learning_rate,requested_brain_epsilon=a.brain_epsilon,
            optimizer_groups=[dict(name=g['name'],lr=g['lr'],epsilon=g['eps']) for g in optimizer.param_groups],
            original_optimizer_moments_preserved=True,neural_history='chronological carry;reset at actual opening only',
            sample_schedule=f'{a.teacher_batches_per_cycle} teacher then {a.own_batches_per_cycle} own chronological batches',
            own_state_source_is_current_student=bool(a.current_student_corpus) and not a.own_acquisition_already_counted,
            risk_only_pretraining=a.risk_only_pretraining,actor_calibration_update_budget=a.actor_calibration_updates,
            max_grad_norm=a.max_grad_norm,acquisition_already_counted=a.own_acquisition_already_counted,
            risk_readout_substeps=a.risk_readout_substeps,
            risk_readout_substeps_protocol='same original signed4f loss,unit coefficient,clip5,on detached SAME-forward DN;then original joint objective',
            risk_readout_learning_rate_schedule=f'{a.risk_readout_initial_learning_rate}*(.1+.9*.5*(1+cos(pi*joint_step/joint_budget)))',
            all_nonrisk_parameters_unchanged_during_substeps=True,
            all_sensory_and_brain_joint_update_budget=a.updates,
            development_every_joint_updates=a.development_every,
            teacher_input_data_stage_scope=1,student_native_episode_stage_cutoff=False,
            teacher_actions_are_supervised_data_not_PPO=True,live_student_actions_overridden=False,
            updates=a.updates,RAM_or_VRAM_caps=None)
        if a.risk_only_pretraining:
            algorithm.update(losses='unit original signed4f Huber+BCE through all raw sensory/core/risk;actor+three critics preserved;then actor-only original safe/future/hit/rule on final-body DN',
                phase_order=['raw_history_physical_risk','final_body_full_history_DN_cache','bounded_actor_readout','frozen_native20'],
                actor_updated_during_risk_pretraining=False,
                body_and_map_fixed_only_during_bounded_actor_readout_fitting=True)
        write_json(a.output/'algorithm.json',algorithm)
        launch.update(status='training',CUDA_started=True,CUDA_device=torch.cuda.get_device_name(0))
        write_json(a.output/'launch.json',launch)
        if a.tensorboard_config:
            board=read(a.tensorboard_config)
            board.setdefault('historical_fly_algorithm_sources',{}).update(board.get('fly_algorithm_sources') or {})
            label=('chronological_risk_first_plastic_brain_v10' if a.risk_only_pretraining else
                'chronological_current_student_risk_balanced_v9' if a.risk_readout_substeps else
                'chronological_current_student_DAgger_v8' if a.current_student_corpus else 'chronological_fullbrain_avoidance_v7')
            board['fly_algorithm_sources']={label:str(a.output)}
            write_json(a.tensorboard_config,board)
        save('initial')
        development_nodes=[];passed_node=None
        for step in range(a.updates):
            guard();cycle=a.teacher_batches_per_cycle+a.own_batches_per_cycle
            kind='teacher' if step%cycle<a.teacher_batches_per_cycle else 'own'
            raw,labels,meta=streams[kind].next()
            data={key:torch.from_numpy(value).to(model.device) for key,value in raw.items()}
            task={key:torch.from_numpy(value).to(model.device) for key,value in labels.items()}
            optimizer.zero_grad(set_to_none=True)
            risk_extra=[]
            if a.risk_readout_substeps:
                lr=a.risk_readout_initial_learning_rate*(.1+.9*.5*(1.+math.cos(math.pi*step/a.updates)))
                result,guide,risk,loss,risk_extra=alternating_original_guidance_block(
                    model,optimizer,data,task,states[kind],risk_substeps=a.risk_readout_substeps,
                    risk_learning_rate=lr)
                costs['risk_readout_Adam']+=len(risk_extra)
                costs['risk_readout_query_visits']+=sum(x['valid_queries'] for x in risk_extra)
                costs['supervised_Adam']+=len(risk_extra)
            else:
                result,guide,risk,loss=original_guidance_block(model,data,task,states[kind])
            if a.risk_only_pretraining:loss=risk['loss']
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite original guidance objective')
            loss.backward()
            norm=clip_gradients_fp64(list(model.parameters()),a.max_grad_norm)
            if step==0 or (not (a.output/'first-hazard-gradient.json').exists() and float(risk['danger_pool_queries'])>0):
                gradients={g['name']:dict(tensors=len(g['params']),gradient_tensors=sum(p.grad is not None for p in g['params']),
                    norm=sum(float(p.grad.detach().double().square().sum()) for p in g['params'] if p.grad is not None)**.5)
                    for g in optimizer.param_groups}
                write_json(a.output/('initial-gradient.json' if step==0 else 'first-hazard-gradient.json'),gradients)
            optimizer.step()
            states[kind]=ContinuousConnectomeState(result['following_state'].activity.detach())
            costs['supervised_Adam']+=1
            costs['joint_Adam']+=1
            if a.risk_only_pretraining:costs['risk_only_Adam']+=1
            costs['teacher_learning_rows' if kind=='teacher' else 'own_state_learning_rows']+=meta['real_rows']
            row=dict(step=costs['supervised_Adam'],stream=kind,real_rows=meta['real_rows'],
                stream_epochs_completed=meta['epochs_completed'],loss=float(loss.detach()),gradient_norm=norm,
                safe_loss=float(guide['safe_loss'].detach()),rule_loss=float(guide['rule_loss'].detach()),
                future_loss=float(guide['future_loss'].detach()),hit_loss=float(guide['collision_loss'].detach()),
                signed4f_loss=float(risk['loss'].detach()),known_risk_scenes=float(risk['valid_scenes']),
                joint_step=costs['joint_Adam'],risk_readout_substeps_paid=len(risk_extra),
                risk_readout_substep_loss_first=risk_extra[0]['loss'] if risk_extra else None,
                risk_readout_substep_loss_last=risk_extra[-1]['loss'] if risk_extra else None,
                risk_readout_learning_rate=risk_extra[-1]['learning_rate'] if risk_extra else None,
                elapsed_seconds=time.monotonic()-begun)
            row.update(recorded_state_metrics(result,data,task))
            append_jsonl(a.output/'learning.jsonl',row)
            progress('chronological_physical_risk_pretraining' if a.risk_only_pretraining else
                     'chronological_original_GRU_supervision',learning=row)
            del raw,labels,data,task,result,guide,risk,loss
            if costs['joint_Adam']%30==0:save('latest')
            if a.development_every and costs['joint_Adam']%a.development_every==0 and step+1<a.updates:
                tag=f'node-{costs["joint_Adam"]:04}'
                node=frozen_node(tag,f'development_{costs["joint_Adam"]:04}',f'confirmation_{costs["joint_Adam"]:04}')
                development_nodes.append(node)
                write_json(a.output/'development-nodes.json',dict(nodes=development_nodes))
                if node['basic_avoidance_target_met']:
                    passed_node=node;break
        if a.actor_calibration_updates:
            save('risk-pretrained')
            from fly_touhou.chronological_actor_readout import build_cache,fit_actor
            cache=build_cache(model,dict(teacher=teacher,own=own),
                dict(teacher=split,own=own_split),a.output/'final-body-DN-cache',check_stop=guard,
                progress=lambda row:progress('final_body_history_replay_for_actor',cache=row))
            def actor_progress(item):
                costs['actor_readout_Adam']+=1;costs['supervised_Adam']+=1
                row=dict(item,step=costs['supervised_Adam'],actor_step=item['step'],
                    stream='actor_readout',joint_step=costs['joint_Adam'],elapsed_seconds=time.monotonic()-begun)
                append_jsonl(a.output/'learning.jsonl',row)
                progress('bounded_actor_readout_calibration',learning=row)
            actor_result=fit_actor(model,optimizer,cache,updates=a.actor_calibration_updates,
                seed=a.seed+77,check_stop=guard,progress=actor_progress,
                on_checkpoint=lambda step:save(f'actor-{step:04}'))
            write_json(a.output/'actor-calibration-result.json',actor_result)
            del cache;gc.collect()
        selected=passed_node or frozen_node('final','development','confirmation20')
        if passed_node:save('final')  # Preserve the usual full-resume artifact too.
        report=selected['development'];confirmation=selected['confirmation']
        write_json(a.output/'result.json',dict(schema=SCHEMA,status='completed',costs=costs,algorithm=algorithm,
            development=report,confirmation=confirmation,basic_avoidance_target_met=confirmation is not None and
                confirmation['first_stage_without_observed_contact_count']>=8,
            selected_parameter_path=selected['parameter_path'],development_nodes=development_nodes,
            full_story_PPO_resumed=False,elapsed_seconds=time.monotonic()-begun))
        launch['status']='completed';write_json(a.output/'launch.json',launch)
        progress('completed',first_stage_clean=report['first_stage_without_observed_contact_count'])
    except BaseException as error:
        write_json(a.output/'failure.json',dict(schema=SCHEMA,error=repr(error),traceback=traceback.format_exc(),
            costs=costs,no_automatic_retry=True,elapsed_seconds=time.monotonic()-begun))
        raise


def main(argv=None):
    a=arguments(argv)
    if not (a.updates>0 and a.max_seconds>0 and 0<a.readout_learning_rate<=.01 and 0<a.brain_learning_rate<=.01
            and a.teacher_batches_per_cycle>=1 and a.own_batches_per_cycle>=1):
        raise ValueError('Explicit finite supervised budget and learning rates required')
    if not (0<=a.risk_readout_substeps<=16 and 0<a.risk_readout_initial_learning_rate<=.01 and a.development_every>=0):
        raise ValueError('Finite risk-readout substeps and explicit bounded learning rate required')
    if a.brain_epsilon is not None and not 0<a.brain_epsilon<=1e-3:
        raise ValueError('Explicit positive brain Adam epsilon required')
    if a.max_grad_norm<=0 or a.actor_calibration_updates<0:
        raise ValueError('Positive gradient clipping and finite actor budget required')
    if a.risk_only_pretraining and (a.risk_readout_substeps or a.development_every or not a.actor_calibration_updates):
        raise ValueError('Risk-first uses its direct loss and completes actor warmup before native evaluation')
    if a.current_student_corpus and (a.after_student_pid is None or a.after_student_create_time is None):
        raise ValueError('Current student acquisition must have an identified executor')
    if not a.execute:
        print(json.dumps(dict(schema=SCHEMA,status='prepared_not_executed',configuration={
            key:str(value) if isinstance(value,Path) else value for key,value in vars(a).items()},CUDA_started=False,native_started=False)))
        return
    execute(a)


if __name__=='__main__':main()
