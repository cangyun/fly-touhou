"""Full CUDA proof of an opt-in auxiliary objective, never a native policy run."""
import argparse
from copy import deepcopy
import gc
import gzip
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch.nn import functional as F
from fly_touhou.connectome import load_connectome
from fly_touhou.experiment import file_sha256
from fly_touhou.geometry_migration import verify_previous_geometry
from fly_touhou.geometry_action_feedback import (GeometryActionFeedback,action_set_loss,movement_entropy,
    balance_target_weights,ACTION_FEEDBACK_COEFFICIENT,ACTION_FEEDBACK_SCHEMA)
from fly_touhou.ppo_cuda_capture import PpoLossCudaCapture
from fly_touhou.state import Snapshot
from fly_touhou.whole_brain_learning import load_whole_brain_checkpoint,graph_fingerprint
from fly_touhou.whole_brain_ppo import (ValueNetwork,WholeBrainRollout,state_record,
    ppo_update,replay_segment,_native_observation)
from probe_captured_ppo_update import step_proof,tensor_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','migrated','trace','source-dir','reference-source','output','learning-checkpoint','previous-source'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--teacher-reference',type=Path)
    parser.add_argument('--teacher-shadow-reference',type=Path)
    parser.add_argument('--critical-full-rollout-first-step',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'starting','native_game_started':False,'numerical_learning_deployable':False}
    def progress(phase,**data):
        value={'phase':phase,'updated_unix':time.time(),**data}
        (args.output/'progress.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(value,ensure_ascii=False),flush=True)
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        learning_hash=file_sha256(args.learning_checkpoint)
        learning=torch.load(args.learning_checkpoint,map_location='cpu',weights_only=True)
        old=verify_previous_geometry(learning,args.previous_source)
        progress('full_zero_extension_and_default_PPO_baseline')
        baseline=args.output/'geometry_baseline'
        command=[sys.executable,str(Path(__file__).with_name('prove_geometry_extension.py'))]
        for name in ('parent','migrated','trace','source-dir','reference-source'):
            command.extend(['--'+name,str(getattr(args,name.replace('-','_')))])
        command.extend(['--output',str(baseline),'--decisions','16'])
        subprocess.run(command,check=True,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        report=json.loads((baseline/'report.json').read_text(encoding='utf-8'))
        assert report['status']=='passed'
        report.update(status='running',script_sha256=file_sha256(__file__),
            geometry_baseline={'path':str((baseline/'report.json').resolve()),'sha256':file_sha256(baseline/'report.json')},
            learning_checkpoint_sha256=learning_hash,
            learning_checkpoint_path=str(args.learning_checkpoint.resolve()),
            previous_verified_source={'path':old['source'],'manifest_sha256':old['manifest_sha256']},
            numerical_learning_deployable=False)
        assert learning['model_config']==report['model_config'] and learning['encoder']==report['encoder']
        assert learning['graph_fingerprint']==report['graph_fingerprint']
        graph=load_connectome(args.source_dir,max_neurons=None)
        assert graph.n_neurons==138639 and graph.n_edges==15091983 and graph_fingerprint(graph)==report['graph_fingerprint']
        feedback=GeometryActionFeedback(frame_skip=learning['training_config']['frame_skip'])
        report['action_feedback']=feedback.specification()
        expert=None
        if args.teacher_reference is not None:
            from fly_touhou.native_teacher_feedback import (NativeV6TrainingFeedback,verify_closed_teacher_reference,
                restrict_teacher_target,validate_teacher_targets,teacher_cross_entropy,TEACHER_FEEDBACK_SCHEMA,
                TEACHER_FEEDBACK_COEFFICIENT)
            assert args.teacher_shadow_reference is not None
            shadow=json.loads(args.teacher_shadow_reference.read_text(encoding='utf-8'))
            assert shadow['status']=='passed' and shadow['counts']['queries']>=1024
            assert shadow['independent_evaluation_used'] is False and shadow['learning_performed'] is False
            assert file_sha256(args.teacher_shadow_reference.parent/'source_archive/native_teacher_feedback.py')==file_sha256(ROOT/'src/fly_touhou/native_teacher_feedback.py')
            expert=NativeV6TrainingFeedback()
            report['teacher_feedback']=expert.specification()
            report['teacher_reference']=verify_closed_teacher_reference(args.teacher_reference)
            report['teacher_shadow_reference']={'path':str(args.teacher_shadow_reference.resolve()),
                'sha256':file_sha256(args.teacher_shadow_reference)}
            report['inherited_action_feedback_transition']=deepcopy(learning['training_config']['action_feedback_transition'])
        snapshots=[];input_rows=[];input_meta=None
        with gzip.open(args.trace,'rb') as stream:
            for raw in stream:
                row=json.loads(raw)
                if row.get('record_type')=='metadata':input_meta=row
                if row.get('record_type')=='transition':
                    snapshots.append(Snapshot.from_dict(row['snapshot']));input_rows.append(raw)
        assert len(snapshots)==16
        assert input_meta['schema']=='action-feedback-numeric-input-bundle-v1' and len(input_meta['selected'])==16
        import hashlib
        sources={}
        for selected,raw in zip(input_meta['selected'],input_rows):
            source=selected['source'];path=Path(source['path'])
            assert hashlib.sha256(raw).hexdigest()==source['raw_line_sha256']
            sources.setdefault(str(path),{'sha256':source['sha256'],'lines':{}})['lines'][source['line']]=raw
        for name,item in sources.items():
            assert file_sha256(name)==item['sha256']
            found=set()
            with gzip.open(name,'rb') as stream:
                for number,raw in enumerate(stream,1):
                    if number in item['lines']:
                        assert raw==item['lines'][number];found.add(number)
            assert found==set(item['lines'])
        report['numeric_source_lines_verified']=16

        def load():
            policy=load_whole_brain_checkpoint(args.learning_checkpoint,graph,'cuda',allow_unproved_geometry=True)
            model=policy.student
            critic=ValueNetwork(policy.encoder.size,tuple(learning['rl_state']['critic_hidden_sizes']),
                geometry_prefix_dim=model.geometry_prefix_dim).to('cuda')
            critic.load_state_dict(learning['rl_state']['critic_state_dict'],strict=True)
            optimizer=torch.optim.Adam([p for m in (model,critic) for p in m.parameters()],
                lr=learning['training_config']['learning_rate'])
            optimizer.load_state_dict(deepcopy(learning['rl_state']['optimizer_state_dict']))
            model.enable_cuda_capture()
            return policy,model,critic,optimizer

        policy,model,critic,optimizer=load();observations=[];masks=[];confidence=[];details=[]
        for snapshot in snapshots:
            observation=policy.encoder.encode(snapshot)
            _native_observation(observation,policy.encoder,snapshot,{'exe_sha256':snapshot.extra['exe_sha256']})
            observations.append(observation)
            mask,c,d=feedback.target(snapshot);masks.append(mask);confidence.append(c);details.append(d)
        masks=np.asarray(masks,dtype=bool);weights=balance_target_weights(masks,confidence)
        assert any(d['reason']=='measured_deathbomb' for d in details)
        assert any(d['reason']=='immune_through_next_observation' for d in details)
        assert any(d['reason']=='positive_clearance_movement' for d in details)
        report['checks']['training_action_targets_verified']=True
        expert_options={}
        if expert is not None:
            expert_targets=[];expert_weights=[]
            for i,(snapshot,raw) in enumerate(zip(snapshots,input_rows)):
                source=json.loads(raw);match=re.search(r':world([0-9]+):',source['episode_id'])
                probabilities,_=expert.query(int(match.group(1)),source['episode_id'],snapshot)
                target,weight,_=restrict_teacher_target(probabilities,masks[i],weights[i])
                expert_targets.append(target);expert_weights.append(weight)
            expert_targets=np.asarray(expert_targets);expert_weights=np.asarray(expert_weights)
            validate_teacher_targets(expert_targets,expert_weights,16)
            expert_options={'teacher_targets':expert_targets,'teacher_target_weights':expert_weights,
                'teacher_target_metadata':{'schema':TEACHER_FEEDBACK_SCHEMA,'before_state_queries':16,
                    'active_targets':int((expert_weights>0).sum()),'numerical_fixture_only':True,
                    'mode_context':'artificial numerical source ordering; actual continuous native mode replay checked by shadow reference'}}
            report['checks']['teacher_targets_verified_from_measured_states']=True
        obs=np.asarray(observations,dtype=np.float32);x=torch.as_tensor(obs,device='cuda')
        initial_states={};keys=[];counters=[];next_counters=[];actions=[];logps=[];values=[]
        generator=torch.Generator().manual_seed(44)
        with torch.no_grad():
            state=model.initial_state(rng_seed=23456)
            for i in range(16):
                if i%4==0:initial_states[i]=state_record(state)
                keys.append(int(state.rng_keys[0]));counters.append(int(state.noise_counter[0]))
                logits,state=model(x[i:i+1],state);lp=F.log_softmax(logits.double(),-1)[0]
                action=int(torch.multinomial(lp.exp().cpu(),1,generator=generator))
                actions.append(action);logps.append(float(lp[action]));values.append(float(critic(x[i:i+1])[0]))
                next_counters.append(int(state.noise_counter[0]))
        values=np.asarray(values,dtype=np.float64);ended=np.zeros(16,dtype=bool);limited=ended.copy();limited[-1]=True
        metadata={'schema':ACTION_FEEDBACK_SCHEMA,'before_state_queries':16,'details':details,
            'specification':feedback.specification(),'numerical_fixture_only':True,'native_actions_executed':False}
        rollout=WholeBrainRollout(obs,np.asarray(actions,dtype=np.int64),np.asarray(logps,dtype=np.float64),
            np.asarray([.01*(-1)**i for i in range(16)],dtype=np.float64),values,
            np.r_[values[1:],values[-1]],ended,limited,np.asarray([True]+[False]*15),initial_states,
            np.asarray(keys),np.asarray(counters),np.asarray(next_counters),['action-set-numeric-fixture']*16,
            4,{'not_native_policy_rollout':True},0,action_set_mask=masks,action_set_weights=weights,
            action_set_metadata=metadata,**expert_options)
        payload={k:torch.as_tensor(v) if isinstance(v,np.ndarray) else v for k,v in rollout.__dict__.items()}
        torch.save({'schema':'nondeployable-action-feedback-numeric-fixture-v1','fixture':payload},args.output/'numeric_fixture.pt')
        del x,state
        progress('auxiliary_only_full_brain_gradient_probe')
        logits=replay_segment(model,rollout,0,4)
        gate=torch.zeros(4,2,1,dtype=torch.float64,device='cuda',requires_grad=True)
        reference_logits=logits.detach().double().reshape(4,2,18)
        gate_entropy=movement_entropy(F.log_softmax((reference_logits+gate).reshape(4,36),-1))
        gate_gradient=torch.autograd.grad(gate_entropy,gate)[0]
        shifted=reference_logits+torch.tensor([60.,-60.],dtype=torch.float64,device='cuda')[None,:,None]
        shifted_entropy=movement_entropy(F.log_softmax(shifted.reshape(4,36),-1))
        assert float(gate_gradient.abs().max())<1e-12 and float((shifted_entropy-gate_entropy).abs())<1e-12
        report['Bomb_gate_entropy_probe']={'gradient_max_abs':float(gate_gradient.abs().max()),
            'group_shift_entropy_difference':float((shifted_entropy-gate_entropy).abs()),'full_actor_logits_used':True}
        report['checks']['Bomb_gate_entropy_not_rewarded']=True
        del gate,reference_logits,gate_entropy,gate_gradient,shifted,shifted_entropy
        aux=action_set_loss(F.log_softmax(logits.double(),-1),torch.as_tensor(masks[:4],device='cuda'),
            torch.as_tensor(weights[:4],device='cuda'))
        gradients=torch.autograd.grad(aux,tuple(model.parameters()))
        norms={name:float(g.detach().double().norm()) for (name,_),g in zip(model.named_parameters(),gradients)}
        assert norms['geometry_input_encoder.weight']>0 and norms['recurrent_gain_logits']>0
        report['auxiliary_only_gradient_norms']=norms
        report['checks']['auxiliary_gradient_through_full_brain_nonzero']=True
        del logits,aux,gradients
        if expert is not None:
            teacher_logits=replay_segment(model,rollout,0,4)
            teacher_loss=teacher_cross_entropy(F.log_softmax(teacher_logits.double(),-1),
                torch.as_tensor(expert_targets[:4],device='cuda'),torch.as_tensor(expert_weights[:4],device='cuda'))
            teacher_gradients=torch.autograd.grad(teacher_loss,tuple(model.parameters()))
            report['teacher_only_gradient_norms']={name:float(g.detach().double().norm())
                for (name,_),g in zip(model.named_parameters(),teacher_gradients)}
            assert report['teacher_only_gradient_norms']['recurrent_gain_logits']>0
            assert report['teacher_only_gradient_norms']['geometry_input_encoder.weight']>0
            report['checks']['teacher_gradient_through_full_brain_nonzero']=True
            del teacher_logits,teacher_loss,teacher_gradients
        cfg=learning['training_config']
        options={k:cfg[k] for k in ('gamma','gae_lambda','clip_range','entropy_coefficient','value_coefficient','value_clip','gradient_clip')}
        options.update(epochs=2,minibatch_chunks=1,target_kl=None,action_set_coefficient=ACTION_FEEDBACK_COEFFICIENT)
        if expert is not None:options['teacher_coefficient']=TEACHER_FEEDBACK_COEFFICIENT
        if args.critical_full_rollout_first_step:
            assert expert is not None
            from fly_touhou.critical_full_rollout import specification,active_critical_indices
            report['critical_full_rollout_first_step']=specification()
            options['critical_full_rollout_first_step']=True
        def update(model,critic,optimizer,capture=None):
            proofs=[];original=optimizer.step
            def step(*a,**kw):proofs.append(step_proof(model,critic,optimizer,original))
            optimizer.step=step
            try:metrics=ppo_update(model,critic,optimizer,rollout,rng=random.Random(343),loss_capture=capture,**options)
            finally:optimizer.step=original
            return proofs,metrics
        progress('ordinary_auxiliary_continuing_Adam')
        expected,ordinary=update(model,critic,optimizer)
        assert expected and ordinary['action_set_loss']>0 and ordinary['behavior_replay_max_logp_error']==0.
        model.clear_cuda_capture();model.clear_cached_propagation();del policy,model,critic,optimizer
        gc.collect();torch.cuda.empty_cache()
        progress('captured_auxiliary_continuing_Adam')
        policy,model,critic,optimizer=load()
        capture=PpoLossCudaCapture(model,critic,clip_range=cfg['clip_range'],value_clip=cfg['value_clip'],
            value_coefficient=cfg['value_coefficient'],entropy_coefficient=cfg['entropy_coefficient'],
            action_set_coefficient=ACTION_FEEDBACK_COEFFICIENT,
            **({'teacher_coefficient':TEACHER_FEEDBACK_COEFFICIENT} if expert is not None else {}))
        actual,captured=update(model,critic,optimizer,capture)
        assert len(actual)==len(expected)==(5 if args.critical_full_rollout_first_step else 8)
        for i,(a,b) in enumerate(zip(expected,actual)):assert a==b,('auxiliary gradient/parameter/Adam parity',i)
        assert captured['action_set_loss']==ordinary['action_set_loss']
        assert captured['behavior_replay_max_logp_error']==0.
        report.update(auxiliary_ordinary_per_step_proofs=expected,auxiliary_captured_per_step_proofs=actual,
            auxiliary_ordinary_metrics=ordinary,auxiliary_captured_metrics=captured,auxiliary_capture=capture.specification())
        report['checks']['auxiliary_loss_and_continuing_Adam_equal']=True
        if expert is not None:
            assert ordinary['teacher_loss']>0 and captured['teacher_loss']==ordinary['teacher_loss']
            report['checks']['teacher_combined_loss_and_continuing_Adam_equal']=True
        if args.critical_full_rollout_first_step:
            assert ordinary['critical_first_step']==captured['critical_first_step']
            first=ordinary['critical_first_step']
            assert first['applied'] and first['samples']==16 and first['microchunks']==4 and first['optimizer_steps']==1
            assert sorted(first['optimized_indices'])==list(range(16))
            assert first['critical_indices']==active_critical_indices(rollout) and first['critical_indices']
            assert first['global_target_weights_preserved']
            report['checks']['critical_and_all_controls_share_first_optimizer_step']=True
            report['checks']['critical_first_step_ordinary_captured_continuing_Adam_equal']=True
        capture.close();model.clear_cuda_capture();model.clear_cached_propagation()
        assert file_sha256(args.learning_checkpoint)==learning_hash
        assert file_sha256(args.trace)==report['inputs']['trace']['sha256']
        report.update(status='passed',scope='full-size numerical and ancestor validation only; no deployable numerical training result')
        progress('passed',auxiliary_optimizer_steps=len(actual))
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
