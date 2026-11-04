"""Final-theta cachedDN -> fresh signed risk8k -> fixed-risk actorBC2k.

Default NoTorch plan. Requires an already-complete DN cache: this entry never
advances the brain. Root separately dispatches frozen native20 if appropriate.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
SCHEMA='continuous-fullhistory-DN-head-calibration-entry-v1'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training-source','label-manifest','visual-manifest','connectome','DN-cache','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('training-source-sha256','label-manifest-sha256','visual-manifest-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--max-seconds',type=int,required=True);p.add_argument('--execute-cuda',action='store_true')
    a=p.parse_args(argv)
    if a.max_seconds<1:p.error('Positive finite wall budget required')
    for key in ('training_source_sha256','label_manifest_sha256','visual_manifest_sha256'):
        value=getattr(a,key)
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):p.error('Explicit artifact SHA256 required')
    return a


def plan(a):
    from fly_touhou.continuous_rule_episode_pretraining import inspect_fixed8
    data=inspect_fixed8(a.label_manifest,a.visual_manifest,expected_sha256=a.label_manifest_sha256,
        expected_visual_sha256=a.visual_manifest_sha256)
    committed=(a.DN_cache/'COMMITTED.json').is_file()
    return dict(schema=SCHEMA,status='prepared_not_executed',Torch_imported='torch' in sys.modules,
        NumPy_imported='numpy' in sys.modules,source=str(a.training_source),source_sha256=a.training_source_sha256,
        data=data,DN_cache=str(a.DN_cache),cache_committed=committed,execution_ready=data['ready'] and committed,
        output=str(a.output),risk=dict(fresh=True,seed=71931,hidden64=True,outputs18=True,updates8000=True,
            scalar_batch4096=True,query_counts=[1024,2048,1024],clip=5.,baseLR=.001,epochs40_steps200=True,
            coordinate='train z=margin/8;prediction8*z,not cost-score;Huber+.2BCE',
            statistics='only6train eligible finalthetaDN;FP64 populationSD active>=1e-4 median floor',
            runtime='fixed new normalization;score=2-16*z,softplus;no first-layer folding',
            parent_main_risk_states_reset=4),
        actor=dict(updates=2000,batch=1024,learning_rate=.0003,clip=.5,weight=1.,
            target='original rule softp/confidence;world000 excluded',
            objective='legal weighted teacher CE on actual combined rawActor+newFixedRisk19',
            only_parent_actor6_Adam_advances=True,risk_and_body_frozen=True,other_group_LRs_unchanged=True),
        held='two fit-held before/after on cachedDN only;not fit/selected;no native claim',
        maximum_seconds=a.max_seconds,RAM_or_VRAM_caps=None,core_forwards=0,brain_updates=0,
        PPO_updates=0,value_targets=0,native_actions=0,teacher_queries=0,CUDA_started=False,optimizer_instances=0)


def execute(a,prepared):
    if 'torch' in sys.modules:raise RuntimeError('Fresh executor,CVD before Torch required')
    if not prepared['execution_ready']:raise RuntimeError('Finish the separate complete DN cache first;this entry never replays brain')
    if a.output.exists():raise FileExistsError('Fresh calibration output required;no auto retry')
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    import psutil
    process=psutil.Process();process.cpu_affinity(process.cpu_affinity()[-2:]);process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    a.output.mkdir(parents=True);started=time.monotonic()
    from fly_touhou.thrl_durable_records import write_json
    costs=dict(risk_Adam=0,actor_Adam=0,risk_backwards=0,actor_backwards=0,
        brain_forwards=0,brain_updates=0,critic_updates=0,native_actions=0,teacher_queries=0)
    launch=dict(prepared,status='running',PID=os.getpid(),create_time=process.create_time(),
        started_UTC=datetime.now(timezone.utc).isoformat())
    write_json(a.output/'launch.json',launch)
    model=optimizer=net=None;normalization_spec=None;algorithm=provenance=None;inherited={}
    risk_installed=False;actor_transaction_complete=True;latest_checkpoint=None
    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-started>a.max_seconds:raise RuntimeError('Finite head-calibration wall budget reached')
    def progress(phase,**kwargs):
        write_json(a.output/'progress.json',dict(schema=SCHEMA,phase=phase,costs=copy.deepcopy(costs),
            elapsed_seconds=time.monotonic()-started,**kwargs))
    def atomic_pt(path,value):
        temporary=path.with_suffix(path.suffix+'.tmp')
        with temporary.open('wb') as stream:torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    def save(tag,*,parameters=False):
        nonlocal latest_checkpoint
        if body_before!=fixed_body_identity(model,optimizer,exclude_actor_LR=True):
            raise RuntimeError('Brain/input/perception/critic parameters,Adam or nonactor LRs changed')
        counters=copy.deepcopy(inherited)
        counters['fullhistory_readout_risk_Adam']=counters.get('fullhistory_readout_risk_Adam',0)+costs['risk_Adam']
        counters['fullhistory_readout_actor_BC_Adam']=counters.get('fullhistory_readout_actor_BC_Adam',0)+costs['actor_Adam']
        payload=training_checkpoint_payload(model,optimizer,algorithm=algorithm,counters=counters,provenance=provenance)
        tag_normalized_payload(payload,normalization_spec)
        atomic_pt(a.output/f'{tag}-training.pt',payload)
        if parameters:atomic_pt(a.output/f'{tag}-parameters.pt',payload['model_payload'])
        latest_checkpoint=str(a.output/f'{tag}-training.pt')
    try:
        import numpy as np
        import torch
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(False)
        from fly_touhou.continuous_policy_cost_routing import restore_routed_source
        from fly_touhou.continuous_connectome_training import training_checkpoint_payload
        from fly_touhou.continuous_normalized_risk_coordinate import tag_normalized_payload,tensor_sha
        from fly_touhou.continuous_fullhistory_head_calibration import (build_DN_cache,load_DN_cache,
            fixed_body_identity,install_new_normalized_risk,calibrate_actor,predict_cost_cache,evaluate_cached_heads)
        from fly_touhou.continuous_signed_risk_pretraining import SignedClearanceReadout,fit_signed_head
        from fly_touhou.continuous_risk_calibration import train_statistics
        guard()
        if sha(a.training_source)!=a.training_source_sha256:raise ValueError('Explicit final source checkpoint bytes differ')
        source=torch.load(a.training_source,map_location='cpu',weights_only=True)
        model,optimizer,inherited,_=restore_routed_source(source,a.connectome,policy_cost_gradient='shared')
        parent_algorithm=copy.deepcopy(source['algorithm']);del source
        model.eval().requires_grad_(False)
        body_before=fixed_body_identity(model,optimizer,exclude_actor_LR=True)
        # Existing COMMITTED is mandatory,so this function only validates/reuses
        # the one frozen-theta prefix. It cannot enter its core-forward branch.
        cache=build_DN_cache(model,optimizer,prepared['data'],a.DN_cache,source_sha256=a.training_source_sha256,check_stop=guard)
        if not cache['reused_without_core_forward']:raise RuntimeError('Head entry must not replay core')
        DN,data=load_DN_cache(a.DN_cache)
        progress('held_before')
        held_before=evaluate_cached_heads(model,DN,data,check_stop=guard)
        write_json(a.output/'held-before.json',held_before)
        f=data['features'];normal=(f[:,21]>.5)&(f[:,20]<=0)&(f[:,5]<=0)
        train=data['train_role']&data['trusted4']&data['unprotected']&normal
        mean,scale,statistics=train_statistics(DN,train)
        statistics['population']='eligible signed4f final-theta DN from all six TRAIN trajectories only'
        stat_path=a.output/'risk-finaltheta-train-statistics.npz'
        np.savez(stat_path,mean=mean.numpy(),scale=scale.numpy(),train_eligible=train.numpy())
        normalization_spec=dict(protocol='fullhistory-finaltheta-DN-fixed-risk-normalization-v1',
            statistics_path=str(stat_path),statistics_sha256=sha(stat_path),mean_sha256=tensor_sha(mean),
            scale_sha256=tensor_sha(scale),buffers=['layers.DN_mean','layers.DN_scale'],fixed_train_only=True,
            origin='six complete TRAIN final-theta own DN histories;no held fit or oldDN',
            training_coordinate='z=margin/8',runtime_score='2-16*z')
        net=SignedClearanceReadout(1299,seed=71931).to(model.device)
        def risk_progress(row):
            costs['risk_Adam']=row['optimizer_updates'];costs['risk_backwards']=row['backward_passes']
            progress('risk_training',learning=row)
        risk_learning=fit_signed_head(net,DN,data['margin4'],train,mean,scale,seed=71931,
            check_stop=guard,progress=risk_progress)
        risk_progress(risk_learning)
        risk_preservation=install_new_normalized_risk(model,optimizer,net,mean,scale,normalization_spec)
        risk_installed=True
        algorithm=dict(schema=SCHEMA,phase='fixed_body_staged_risk_then_combined_actor_calibration',
            core=model.core.config(),risk_supervision='signed-clearance4f',
            risk_pretraining_protocol='signed-clearance4f-GRU-v1',risk_fitting='fresh64 z=margin/8 original8000',
            actor_BC='combined source19 weighted original teacher CE;fixed newrisk;2000x1024',
            PPO_updates=False,brain_updates=0,critic_updates=0)
        provenance=dict(parent=dict(path=str(a.training_source),sha256=a.training_source_sha256,
                counters=inherited,algorithm=parent_algorithm),
            cache=dict(path=str(a.DN_cache),COMMITTED_sha256=sha(a.DN_cache/'COMMITTED.json'),
                DN_source_identity=cache['DN_source_identity'],original_prefix_counts=cache['counts'],new_core_forwards=0),
            data_binding=cache['data_binding'],statistics=statistics,risk_learning=risk_learning,
            risk_preservation=risk_preservation,normalization=normalization_spec,
            entry_sha256=sha(__file__),helper_sha256=sha(ROOT/'src/fly_touhou/continuous_fullhistory_head_calibration.py'),
            interpretation='Body was already trained;this stage calibrates readouts only,not a substitute for brain learning',
            held_not_native=True)
        save('risk-completed');progress('risk_completed',learning=risk_learning)
        del net;net=None
        fixed_cost=predict_cost_cache(model,DN,check_stop=guard)
        def actor_progress(row):
            costs['actor_Adam']=row['optimizer_updates'];costs['actor_backwards']=row['backward_passes']
            progress('actor_BC',learning=row)
        actor_learning=calibrate_actor(model,optimizer,DN,data,fixed_cost,seed=71932,
            check_stop=guard,progress=actor_progress)
        actor_progress(actor_learning);provenance['actor_learning']=actor_learning
        save('final',parameters=True)
        progress('held_after')
        held_after=evaluate_cached_heads(model,DN,data,check_stop=guard)
        write_json(a.output/'held-after.json',held_after)
        write_json(a.output/'result.json',dict(schema=SCHEMA,status='completed',costs=costs,
            held_before=held_before,held_after=held_after,risk_learning=risk_learning,actor_learning=actor_learning,
            provenance=provenance,algorithm=algorithm,brain_input_CNN_critic_and_Adam_bitwise_preserved=True,
            risk4_main_Adam_reset=True,only_actor6_main_Adam_advanced=True,
            final_training=dict(path=str(a.output/'final-training.pt'),sha256=sha(a.output/'final-training.pt')),
            final_parameters=dict(path=str(a.output/'final-parameters.pt'),sha256=sha(a.output/'final-parameters.pt')),
            elapsed_seconds=time.monotonic()-started,native_development=None,native_clear_claim=False))
        progress('completed')
    except BaseException as error:
        risk=getattr(error,'signed_risk_fit_diagnostics',None);actor=getattr(error,'actor_BC_diagnostics',None)
        if risk:costs.update(risk_Adam=risk['optimizer_updates'],risk_backwards=risk['backward_passes'])
        if actor:
            costs.update(actor_Adam=actor['optimizer_updates'],actor_backwards=actor['backward_passes'])
            actor_transaction_complete=actor['optimizer_transaction_complete']
        recovery_error=None
        if risk_installed and algorithm is not None and actor_transaction_complete:
            try:save('stopped-after-head-update')
            except BaseException as second:recovery_error=repr(second)
        work=getattr(error,'signed_risk_work',None)
        if work is not None:
            try:
                work.update(mean=mean,scale=scale,coordinate='z=margin/8,not runtime cost',source=str(a.training_source))
                atomic_pt(a.output/'stopped-signed-work.pt',work)
            except BaseException as second:recovery_error=repr(second)
        write_json(a.output/'failure.json',dict(schema=SCHEMA,error=repr(error),costs=costs,
            risk_diagnostics=risk,actor_diagnostics=actor,latest_checkpoint=latest_checkpoint,
            optimizer_transaction_complete=actor_transaction_complete,recovery_error=recovery_error,
            brain_updates=0,native_actions=0,automatic_retry=False));raise
    finally:
        write_json(a.output/'training-cleanup.json',dict(training_closed=True,close_errors=[],native_worlds_created=0))
        model=optimizer=net=None


def main(argv=None):
    a=arguments(argv);prepared=plan(a)
    if not a.execute_cuda:print(json.dumps(prepared,ensure_ascii=False,indent=2));return 0
    execute(a,prepared);return 0


if __name__=='__main__':raise SystemExit(main())
