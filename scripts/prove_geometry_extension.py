"""Full CUDA geometry extension proof, isolated from live games and learners.

Measured input snapshots feed a numerical fixture; its sampled actions and
synthetic rewards are never a native campaign or a deployable learned policy.
The initial migrated checkpoint remains untouched throughout this proof.
"""
import argparse
from copy import deepcopy
import gc
import gzip
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from fly_touhou.brain_cuda_capture import _tensors  # noqa: E402
from fly_touhou.connectome import load_connectome  # noqa: E402
from fly_touhou.experiment import file_sha256  # noqa: E402
from fly_touhou.geometry_migration import PROOF_SCHEMA,PROOF_SOURCE_FILES  # noqa: E402
from fly_touhou.ppo_cuda_capture import PpoLossCudaCapture  # noqa: E402
from fly_touhou.state import Snapshot  # noqa: E402
from fly_touhou.whole_brain_learning import load_whole_brain_checkpoint,_restore_trainable,graph_fingerprint  # noqa: E402
from fly_touhou.whole_brain_ppo import (ValueNetwork,WholeBrainRollout,state_record,
    ppo_update,_native_observation)  # noqa: E402
from probe_captured_ppo_update import (step_proof,tensor_digest,optimizer_digest)  # noqa: E402


def frozen_module(path,name):
    spec=importlib.util.spec_from_file_location('fly_touhou.'+name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','migrated','trace','source-dir','reference-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--decisions',type=int,default=16)
    args=parser.parse_args()
    if args.decisions<8 or args.decisions%4:
        raise ValueError('At least two full four-decision chunks are required')
    args.output.mkdir(parents=True,exist_ok=False)
    report={'schema':PROOF_SCHEMA,'status':'running','native_game_started':False,
        'numerical_fixture_actions_executed_in_game':False,
        'numerical_fixture_learning_deployable':False,
        'scope':'full-size numerical validation only, not a native student or teacher acceptance',
        'script_sha256':file_sha256(__file__),
        'helper_sha256':file_sha256(Path(__file__).with_name('probe_captured_ppo_update.py')),
        'source_files':{name:file_sha256(ROOT/name) for name in PROOF_SOURCE_FILES},
        'inputs':{name:{'path':str(getattr(args,name).resolve()),'sha256':file_sha256(getattr(args,name))}
                  for name in ('parent','migrated','trace')},'checks':{}}
    def write():
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    def progress(phase,**kw):
        value={'phase':phase,'updated_unix':time.time(),**kw}
        (args.output/'progress.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(value,ensure_ascii=False),flush=True)
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        parent,migrated=[torch.load(p,map_location='cpu',weights_only=True) for p in (args.parent,args.migrated)]
        transition=migrated['geometry_observation_transition']
        assert transition['parent_checkpoint_sha256']==report['inputs']['parent']['sha256']
        report.update(lineage_id=transition['lineage_id'],model_config=migrated['model_config'],
            encoder=migrated['encoder'],graph_fingerprint=migrated['graph_fingerprint'],
            migrated_checkpoint_sha256=report['inputs']['migrated']['sha256'])
        graph=load_connectome(args.source_dir,max_neurons=None)
        assert graph.n_neurons==138639 and graph.n_edges==15091983
        assert graph_fingerprint(graph)==parent['graph_fingerprint']==migrated['graph_fingerprint']
        report['checks']['full_source_graph_identity']=True
        report['graph']={'neurons':graph.n_neurons,'signed_connection_records':graph.n_edges}

        def load():
            policy=load_whole_brain_checkpoint(args.migrated,graph,'cuda',allow_unproved_geometry=True)
            model=policy.student
            critic=ValueNetwork(policy.encoder.size,tuple(migrated['rl_state']['critic_hidden_sizes']),
                geometry_prefix_dim=model.geometry_prefix_dim).to('cuda')
            critic.load_state_dict(migrated['rl_state']['critic_state_dict'],strict=True)
            params=[p for module in (model,critic) for p in module.parameters()]
            optimizer=torch.optim.Adam(params,lr=migrated['training_config']['learning_rate'])
            optimizer.load_state_dict(deepcopy(migrated['rl_state']['optimizer_state_dict']))
            model.enable_cuda_capture()
            return policy,model,critic,optimizer

        progress('load_full_new_learner')
        policy,model,critic,optimizer=load()
        observations=[]
        with gzip.open(args.trace,'rt',encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line)
                if row.get('record_type')!='transition':continue
                snap=Snapshot.from_dict(row['snapshot'])
                value=policy.encoder.encode(snap)
                _native_observation(value,policy.encoder,snap,{'exe_sha256':snap.extra['exe_sha256']})
                observations.append(value)
                if len(observations)==args.decisions:break
        if len(observations)!=args.decisions:
            raise ValueError('Not enough closed measured geometry snapshots')
        obs=np.asarray(observations,dtype=np.float32)
        x=torch.as_tensor(obs,device='cuda')
        report['checks']['native_geometry_inputs_validated']=True
        report['input_observation_sha256']=tensor_digest([('observations',torch.as_tensor(obs))])

        # Use the actual pre-extension implementations, not a second instance
        # of the edited class, to test the zero-extension preservation claim.
        progress('compare_frozen_parent_full_states')
        baseline_manifest=json.loads((args.reference_source/'snapshot_manifest.json').read_text(encoding='utf-8'))
        entries={Path(r['path']).as_posix():r['sha256'] for r in baseline_manifest['files']}
        old_paths={n:args.reference_source/'src/fly_touhou'/n for n in ('whole_brain.py','whole_brain_ppo.py')}
        for name,path in old_paths.items():assert file_sha256(path)==entries['src/fly_touhou/'+name]
        old_brain=frozen_module(old_paths['whole_brain.py'],'_geometry_reference_brain56')
        old_ppo=frozen_module(old_paths['whole_brain_ppo.py'],'_geometry_reference_ppo56')
        old=old_brain.WholeBrainStudent(graph=graph,device='cuda',**parent['model_config'])
        # The present capture storage guard knows this absent optional field;
        # adding zero metadata does not change the frozen forward function.
        old.geometry_prefix_dim=0
        _restore_trainable(old,parent)
        old.enable_cuda_capture()
        old_critic=old_ppo.ValueNetwork(parent['model_config']['obs_dim'],
            tuple(parent['rl_state']['critic_hidden_sizes'])).to('cuda')
        old_critic.load_state_dict(parent['rl_state']['critic_state_dict'],strict=True)
        old_names=[('actor',n) for n,_ in old.named_parameters()]+[('critic',n) for n,_ in old_critic.named_parameters()]
        new_names=[('actor',n) for n,_ in model.named_parameters()]+[('critic',n) for n,_ in critic.named_parameters()]
        old_ids=parent['rl_state']['optimizer_state_dict']['param_groups'][0]['params']
        for key,old_id in zip(old_names,old_ids):
            new_id=new_names.index(key)
            before=parent['rl_state']['optimizer_state_dict']['state'][old_id]
            after=optimizer.state_dict()['state'][new_id]
            assert set(before)==set(after)
            assert all(torch.equal(before[k].cpu(),after[k].cpu()) for k in before)
            role,name=key
            saved=parent['trainable_parameters'] if role=='actor' else parent['rl_state']['critic_state_dict']
            actual=dict((model if role=='actor' else critic).named_parameters())[name]
            assert torch.equal(saved[name],actual.detach().cpu())
        for name in ('completed_timesteps','native_frames','updates','episodes_started'):
            assert parent['rl_state'][name]==migrated['rl_state'][name]
        assert torch.equal(parent['rl_state']['action_sampler_state'],migrated['rl_state']['action_sampler_state'])
        report['checks']['original_Adam_state_preserved']=True
        with torch.no_grad():
            a,b=old.initial_state(rng_seed=12345),model.initial_state(rng_seed=12345)
            for index in range(len(x)):
                lp,a=old(x[index:index+1,:model.geometry_prefix_dim].contiguous(),a)
                ln,b=model(x[index:index+1],b)
                assert torch.equal(lp,ln)
                assert all(torch.equal(u,v) for u,v in zip(_tensors(a),_tensors(b)))
            assert torch.equal(old_critic(x[:,:model.geometry_prefix_dim].contiguous()),critic(x))
        report['checks']['frozen_parent_forward_state_bitwise_equal']=True
        report['checks']['critic_zero_extension_bitwise_equal']=True
        report['zero_extension_comparison']={'decisions':len(x),'all_logits_and_all_neural_state_tensors_bitwise_equal':True,
            'old_source_files':{k:file_sha256(v) for k,v in old_paths.items()},'neural_ms_per_decision':model.neural_ms_per_decision}
        old.clear_cuda_capture();old.clear_cached_propagation()
        del old,old_critic,a,b,lp,ln
        gc.collect();torch.cuda.empty_cache()

        progress('construct_numeric_fixture')
        initial_states={};keys=[];counters=[];next_counters=[];actions=[];logps=[];values=[]
        generator=torch.Generator().manual_seed(44)
        with torch.no_grad():
            state=model.initial_state(rng_seed=23456)
            for i in range(len(x)):
                if i%4==0:initial_states[i]=state_record(state)
                keys.append(int(state.rng_keys[0]));counters.append(int(state.noise_counter[0]))
                logits,state=model(x[i:i+1],state)
                lp=F.log_softmax(logits.double(),-1)[0]
                action=int(torch.multinomial(lp.exp().cpu(),1,generator=generator))
                actions.append(action);logps.append(float(lp[action]));values.append(float(critic(x[i:i+1])[0]))
                next_counters.append(int(state.noise_counter[0]))
        n=len(x);terminated=np.zeros(n,dtype=bool);truncated=terminated.copy();truncated[-1]=True
        values=np.asarray(values,dtype=np.float64)
        rollout=WholeBrainRollout(obs,np.asarray(actions,dtype=np.int64),np.asarray(logps,dtype=np.float64),
            np.asarray([.01*(-1)**i for i in range(n)],dtype=np.float64),values,
            np.r_[values[1:],values[-1]],terminated,truncated,np.asarray([True]+[False]*(n-1)),
            initial_states,np.asarray(keys),np.asarray(counters),np.asarray(next_counters),
            ['geometry-numeric-fixture']*n,4,{'not_native_policy_rollout':True},0)
        torch.save({'schema':'geometry-extension-numeric-fixture-v1',
            'native_actions_executed':False,'fixture':rollout.__dict__},args.output/'numeric_fixture.pt')
        del x,state

        cfg=migrated['training_config']
        options={k:cfg[k] for k in ('gamma','gae_lambda','clip_range','entropy_coefficient',
                                   'value_coefficient','value_clip','gradient_clip')}
        options.update(epochs=2,minibatch_chunks=1,target_kl=None)
        def run_update(current,current_critic,current_optimizer,capture=None):
            proofs=[];geometry_norms=[];original_step=current_optimizer.step
            def step(*a,**kw):
                geometry_norms.append(float(current.geometry_input_encoder.weight.grad.detach().double().norm()))
                proofs.append(step_proof(current,current_critic,current_optimizer,original_step))
            current_optimizer.step=step
            try:
                metrics=ppo_update(current,current_critic,current_optimizer,rollout,
                    rng=random.Random(343),loss_capture=capture,**options)
            finally:current_optimizer.step=original_step
            return proofs,geometry_norms,metrics

        progress('ordinary_continuing_Adam_fixture_update')
        expected,gradients,ordinary=run_update(model,critic,optimizer)
        assert expected and any(v>0 for v in gradients)
        assert float(model.geometry_input_encoder.weight.detach().abs().sum())>0
        report['ordinary_per_step_proofs']=expected
        report['geometry_gradient_norms']=gradients
        report['ordinary_update']=ordinary
        report['checks']['new_geometry_gradient_nonzero']=True
        torch.save({'schema':'geometry-extension-numeric-final-v1','deployable':False,
            'actor':{n:v.detach().cpu() for n,v in model.named_parameters()},
            'critic':{n:v.detach().cpu() for n,v in critic.named_parameters()},
            'optimizer':optimizer.state_dict()},args.output/'ordinary_numeric_final.pt')
        model.clear_cuda_capture();model.clear_cached_propagation()
        del model,critic,optimizer,policy
        gc.collect();torch.cuda.empty_cache()

        progress('captured_continuing_Adam_fixture_update')
        policy,model,critic,optimizer=load()
        capture=PpoLossCudaCapture(model,critic,clip_range=cfg['clip_range'],value_clip=cfg['value_clip'],
            value_coefficient=cfg['value_coefficient'],entropy_coefficient=cfg['entropy_coefficient'])
        actual,actual_gradients,captured=run_update(model,critic,optimizer,capture)
        assert len(actual)==len(expected)
        for index,(reference,candidate) in enumerate(zip(expected,actual)):
            assert reference==candidate,('gradient/parameter/Adam parity',index)
        assert ordinary['behavior_replay_max_logp_error']==captured['behavior_replay_max_logp_error']==0.
        report['captured_per_step_proofs']=actual
        report['captured_update']=captured
        report['checks']['captured_objective_and_gradients_equal']=True
        report['checks']['continuing_Adam_and_parameters_equal']=True
        report['capture']=capture.specification()
        capture.close();model.clear_cuda_capture();model.clear_cached_propagation()
        assert file_sha256(args.parent)==report['inputs']['parent']['sha256']
        assert file_sha256(args.migrated)==report['inputs']['migrated']['sha256']
        report.update(status='passed',max_allocated_cuda_bytes=torch.cuda.max_memory_allocated(),
                      numerical_fixture_optimizer_steps=len(actual))
        progress('passed',optimizer_steps=len(actual))
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        write()


if __name__=='__main__':main()
