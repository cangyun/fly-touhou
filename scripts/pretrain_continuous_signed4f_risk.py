"""Fresh signed4f risk64 pretraining on frozen final-theta DN;NoTorch by default.

The existing continuous checkpoint keeps its graph, actor, three critics and
non-risk Adam bytes. This entry never constructs a core,teacher or native game.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]
SCHEMA = 'continuous-signed4f-risk-pretraining-entry-v1'
PROTOCOL = 'signed-clearance4f-GRU-v1'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('training-source','DN-cache','output'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--training-source-sha256')
    p.add_argument('--seed',type=int,default=71931)
    p.add_argument('--max-seconds',type=int,required=True)
    p.add_argument('--execute-cuda',action='store_true')
    a = p.parse_args(argv)
    if a.max_seconds<1 or not 0<=a.seed<2**31: p.error('Finite wall budget and explicit private seed required')
    if a.training_source_sha256 is not None and (len(a.training_source_sha256)!=64
            or any(c not in '0123456789abcdef' for c in a.training_source_sha256)):
        p.error('Optional source artifact identity must be lowercase SHA256')
    return a


def plan(a):
    return dict(schema=SCHEMA,status='prepared_not_executed',Torch_imported='torch' in sys.modules,
        source=str(a.training_source),DN_cache=str(a.DN_cache),output=str(a.output),seed=a.seed,
        protocol=PROTOCOL,risk_supervision='signed-clearance4f',source_code_SHA_load_gate=False,
        checkpoint_family_unchanged=True,full_graph_constructors=0,core_forwards=0,
        training=dict(fresh_risk64=True,CPU500_weights_used=False,parameters=84370,tensors=4,
            signed_network='standardized rawDN1299->64SiLU->18linear; margin=8*linear',
            labels='trusted4f & unprotected & normal;clip original margin[-64,64]',
            split='gameID%5==0 readout-fit-held only; PPO already saw these trajectories',
            statistic_population='trusted TRAIN geometry scenes only;populationSD median floor',
            scalar_query_batch=4096,pools=[1024,2048,1024],overlap=True,replacement=True,
            loss='SmoothL1(pred/8,y/8,beta.1)+.2BCE(-(pred-.5),y<=0);known y<=0 or y>1',
            epochs=40,steps_per_epoch=200,updates=8000,Adam_lr=.001,clip_grad_norm=5.,
            LR='per epoch .001*(.1+.9*.5*(1+cos(pi*epoch_index/40)))',
            selection='fixed final8000; no held selection or early stop'),
        runtime=dict(architecture='original rawDN1299->64SiLU->18softplus',beta=8.,
            first_layer='Wraw=Wstd/scale; braw=bstd-Wraw@mean',
            last_layer='Wcost=-16*Wsign; bcost=2-16*bsign',
            cost='softplus(2-2*predicted_margin); original D95 point formula, not total point+laser+comfort',
            new_normalizer=False,raw_observation_bypass=False),
        Adam=dict(full_source_retained=True,only_risk4_states_reset=True,nonrisk_bitwise_preserved=True),
        max_seconds=a.max_seconds,RAM_or_VRAM_caps=None,automatic_retry=False,
        GPU_started=False,native_actions=0,teacher_queries=0,PPO_updates=0)


def execute(a):
    if 'torch' in sys.modules: raise RuntimeError('Fresh executor required;CVD before Torch')
    if a.output.exists(): raise FileExistsError('Fresh output required;no overwrite or auto retry')
    import psutil
    from calibrate_continuous_risk_head import _completed_predecessor
    current = psutil.Process(); predecessor = _completed_predecessor(a.training_source,psutil)
    current.cpu_affinity(current.cpu_affinity()[-2:])
    if os.name=='nt': current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    a.output.mkdir(parents=True); begun=time.monotonic()
    from fly_touhou.thrl_durable_records import write_json
    costs=dict(signed_risk_Adam=0,backward_passes=0,scalar_training_queries=0,
        unique_scene_head_forward_rows=0,head_inference_rows=0,core_forwards=0,
        PPO_Adam=0,native_actions=0,teacher_queries=0)
    write_json(a.output/'launch.json',dict(plan(a),PID=os.getpid(),create_time=current.create_time(),
        started_UTC=datetime.now(timezone.utc).isoformat(),predecessor=predecessor))
    net=None; source=None
    def guard():
        if (a.output/'STOP').exists(): raise RuntimeError('Operator STOP')
        if time.monotonic()-begun>a.max_seconds: raise RuntimeError('Finite wall budget reached')
    def progress(phase,**kwargs):
        write_json(a.output/'progress.json',dict(schema=SCHEMA,phase=phase,costs=copy.deepcopy(costs),
            elapsed_seconds=time.monotonic()-begun,**kwargs))
    def save(path,payload):
        temporary=path.with_suffix(path.suffix+'.tmp')
        with temporary.open('wb') as stream:
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    try:
        import numpy as np
        import torch
        from torch.nn import functional as F
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(False)
        from fly_touhou.continuous_signed_risk_pretraining import (load_signed_cache,train_statistics,
            SignedClearanceReadout,fit_signed_head,predict_signed,folded_cost_state,predict_cost_state,
            nonrisk_payload_identity,install_cost_state)
        from fly_touhou.continuous_risk_calibration import prediction_error
        guard(); progress('loading_exact_frozen_DN_and_original_geometry')
        DN,target,train,held,geometry,data,cache=load_signed_cache(a.DN_cache,check_stop=guard)
        source_sha=sha(a.training_source)
        if a.training_source_sha256 is not None and source_sha!=a.training_source_sha256:
            raise ValueError('Caller-supplied source artifact identity differs')
        # These two recorded sources share exactly the final-theta body. This
        # is data/cache binding, not an arbitrary source-code compatibility gate.
        norm_result=json.loads((a.DN_cache.parent/'result.json').read_text('utf-8'))
        recorded={norm_result['provenance']['parent']['sha256'],norm_result['final_training']['sha256']}
        if (source_sha not in recorded or not norm_result['main_Adam_preservation']['nonrisk_parameter_and_Adam_bitwise_preserved']):
            raise ValueError('Source must share the recorded final-theta DN body;cache binding differs')
        source=torch.load(a.training_source,map_location='cpu',weights_only=True)
        if source.get('schema')!='continuous-full-connectome-threevalue-training-v1':
            raise ValueError('Complete continuous model+Adam training payload required')
        config=source['model_payload']['configuration']['core']
        if (config['n_neurons'],config['n_edges'],config['output_ports'],config['K'])!=(138639,15091983,1299,8):
            raise ValueError('Same complete continuous K8 family required')
        before_identity=nonrisk_payload_identity(source)
        mean,scale,statistics=train_statistics(DN,train)
        statistics['population']='trusted4 & normal & unprotected TRAIN scenes;not query-gated D95 aggregate-cost rows'
        np.savez(a.output/'signed-train-statistics.npz',mean=mean.numpy(),scale=scale.numpy(),
            train_geometry=train.numpy(),heldout=held.numpy(),geometry=geometry)
        net=SignedClearanceReadout(1299,seed=a.seed).to('cuda:0')
        source_cost=predict_cost_state(source['model_payload']['risk_head'],DN,device='cuda:0',check_stop=guard,paid=costs)
        initial_signed=predict_signed(net,DN,mean,scale,check_stop=guard,paid=costs)
        def metrics(score):
            score=np.asarray(score,np.float32);margin=target.numpy(); result={}
            hard=geometry&(margin<=0).any(-1)&(margin>1).any(-1)
            for role,mask in (('train',train.numpy()),('held',geometry&held.numpy()),
                    ('train_hard',train.numpy()&hard),('held_hard',held.numpy()&hard)):
                s,m=score[mask],margin[mask];choice=s.argmin(-1);chosen=m[np.arange(len(m)),choice]
                ranks=[]
                for start in range(0,len(m),1024):
                    mm=m[start:start+1024];ss=s[start:start+1024]
                    pairs=(mm[:,:,None]<=0)&(mm[:,None,:]>1);counts=pairs.sum((1,2))
                    delta=ss[:,:,None]-ss[:,None,:]
                    wins=(((delta>0)+.5*(delta==0))*pairs).sum((1,2))
                    ranks.extend(np.divide(wins,counts,out=np.full(len(counts),np.nan),where=counts>0).tolist())
                finite=np.isfinite(ranks)
                result[role]=dict(rows=int(mask.sum()),danger=int((chosen<=0).sum()),
                    grey=int(((chosen>0)&(chosen<=1)).sum()),safe=int((chosen>1).sum()),
                    chosen_clearance_regret=float((m.max(-1)-chosen).mean()) if len(m) else None,
                    scene_danger_safe_rank=float(np.asarray(ranks)[finite].mean()) if finite.any() else None,
                    action_counts=np.bincount(choice,minlength=18).tolist())
            return result
        before=dict(source_risk_cost=metrics(source_cost.numpy()),fresh_signed=metrics(-initial_signed.numpy()))
        write_json(a.output/'before.json',dict(statistics=statistics,metrics=before))
        def fit_progress(row):
            costs.update(signed_risk_Adam=row['optimizer_updates'],backward_passes=row['backward_passes'],
                scalar_training_queries=row['scalar_training_queries'],
                unique_scene_head_forward_rows=row['unique_scene_head_forward_rows'])
            progress('signed4f_head_fitting',learning=row)
        learning=fit_signed_head(net,DN,target,train,mean,scale,seed=a.seed,check_stop=guard,progress=fit_progress)
        fit_progress(learning)
        signed=predict_signed(net,DN,mean,scale,check_stop=guard,paid=costs)
        folded=folded_cost_state(net,mean,scale)
        point_cost=predict_cost_state(folded,DN,device='cuda:0',check_stop=guard,paid=costs)
        fold_error=prediction_error(point_cost,F.softplus(2.-2.*signed))
        progress('folding_signed_to_original_raw_DN_cost',fold_error=fold_error)
        preservation=install_cost_state(source,folded)
        if nonrisk_payload_identity(source)!=before_identity: raise RuntimeError('Source body/Adam changed')
        source['algorithm']=copy.deepcopy(source['algorithm'])
        source['algorithm'].update(risk_supervision='signed-clearance4f',risk_pretraining_protocol=PROTOCOL)
        source['counters']=copy.deepcopy(source['counters'])
        source['counters']['signed_clearance4f_pretraining_Adam']=source['counters'].get('signed_clearance4f_pretraining_Adam',0)+8000
        provenance=dict(protocol=plan(a),parent=dict(path=str(a.training_source),sha256=source_sha),
            cache=cache,statistics=statistics,learning=learning,fold_error=fold_error,preservation=preservation,
            entry_sha256=sha(__file__),helper_sha256=sha(ROOT/'src/fly_touhou/continuous_signed_risk_pretraining.py'),
            GPU_name=torch.cuda.get_device_name(0),core_or_teacher_or_native_constructed=False,
            runtime_normalization_added=False,all_source_brain_parameters_available_for_joint_training=True,
            geometric_counterfactuals_are_native_outcomes=False,
            limitation='Pure signed4f point-style cost;not original full D95 point+laser+comfort aggregate')
        source['provenance']=copy.deepcopy(source.get('provenance') or {})
        source['provenance']['signed_clearance4f_pretraining']=provenance
        save(a.output/'final-training.pt',source)
        save(a.output/'final-parameters.pt',source['model_payload'])
        np.savez(a.output/'fixed-final-predictions.npz',signed_margin4=signed.numpy(),point_cost=point_cost.numpy(),
            source_cost=source_cost.numpy(),game_id=data['game_ids'],target_margin4=target.numpy(),
            geometry=geometry,heldout=held.numpy())
        result=dict(schema=SCHEMA,status='completed',protocol=PROTOCOL,costs=costs,learning=learning,
            preservation=preservation,statistics=statistics,fold_error=fold_error,before=before,
            after=metrics(point_cost.numpy()),source_model_and_Adam_family_unchanged=True,
            final_training=dict(path=str(a.output/'final-training.pt'),sha256=sha(a.output/'final-training.pt')),
            final_parameters=dict(path=str(a.output/'final-parameters.pt'),sha256=sha(a.output/'final-parameters.pt')),
            provenance=provenance,elapsed_seconds=time.monotonic()-begun,development=None,
            native_efficacy_or_clear_claim=False)
        write_json(a.output/'result.json',result);progress('completed')
    except BaseException as error:
        learning=getattr(error,'signed_risk_fit_diagnostics',None)
        if learning:
            costs.update(signed_risk_Adam=learning['optimizer_updates'],backward_passes=learning['backward_passes'],
                scalar_training_queries=learning['scalar_training_queries'],
                unique_scene_head_forward_rows=learning['unique_scene_head_forward_rows'])
        work=getattr(error,'signed_risk_work',None)
        if work is None and net is not None and costs['signed_risk_Adam']:
            work=dict(head={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},record=costs)
        recovery=None
        if work is not None:
            try:
                work.update(schema='signed-standardized-risk64-work-only-v1',mean=mean,scale=scale,
                    source=str(a.training_source),raw_cost_runtime_checkpoint=False,automatic_resume=False)
                save(a.output/'stopped-signed-work.pt',work);recovery=str(a.output/'stopped-signed-work.pt')
            except BaseException as second: recovery=repr(second)
        write_json(a.output/'failure.json',dict(schema=SCHEMA,error=repr(error),costs=costs,learning=learning,
            work_recovery=recovery,source_checkpoint_unchanged=True,automatic_retry=False))
        raise
    finally:
        write_json(a.output/'training-cleanup.json',dict(training_closed=True,close_errors=[],native_worlds_created=0))
        source=net=None


def main(argv=None):
    a=parse_args(argv)
    if not a.execute_cuda:
        print(json.dumps(plan(a),ensure_ascii=False,indent=2));return 0
    execute(a);return 0


if __name__=='__main__': raise SystemExit(main())
