"""CPU-only risk-coordinate variant;default prints a NoTorch plan.

No core,perception,critic,optimizer or native game is constructed. The fixed
signed-pretraining checkpoint and its exact statistics are read once;the live
PPO checkpoint is never the default input. No candidate is installed anywhere.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
BASE=Path('./outputs/distillation')
SCHEMA='continuous-normalized-risk-coordinate-CPU-conversion-v1'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-source',type=Path,default=BASE/'continuous-risk-signed4f8000-v1/final-training.pt')
    p.add_argument('--statistics',type=Path)
    p.add_argument('--activity-dir',type=Path,default=BASE/'continuous-signed4f-PPO983040-v2/neural-diagnostics/rollout-000000')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=int,default=120)
    p.add_argument('--execute-cpu',action='store_true')
    a=p.parse_args(argv)
    if a.max_seconds<1:p.error('Finite positive wall budget required')
    if a.statistics is None:a.statistics=a.training_source.parent/'signed-train-statistics.npz'
    return a


def plan(a):
    return dict(schema=SCHEMA,status='prepared_not_executed',Torch_imported='torch' in sys.modules,
        fixed_source=str(a.training_source),original_signed_statistics=str(a.statistics),
        committed64_activity=str(a.activity_dir),output=str(a.output),
        coordinate_change='Wnorm=Wraw*scale;bnorm=braw+Wraw@mean',
        runtime='NormalizedRiskLayers.forward:(DN-mean)/scale then same64SiLU18;[0]/[2] unchanged',
        normalization_shared_by_cost_and_signed_layers=True,new_trainable_parameters=0,
        source_original_schema_rejected_for_normalized_weights=True,
        Adam='only first-layer W/b invalid states removed if actually present;last2 and all others unchanged',
        numeric_optimizer_resume=False,Adam_second_cross_moments_available=False,
        initial_dL_dDN_reduction_claim=False,global_clip_parameter_norm_may_change=True,
        max_seconds=a.max_seconds,optimizer_instances=0,optimizer_steps=0,
        core_constructors=0,core_forwards=0,GPU_started=False,native_actions=0,
        live_source_modified=False,candidate_installed=False)


def execute(a):
    if 'torch' in sys.modules:raise RuntimeError('Fresh CPU executor required;CVD before Torch')
    if a.output.exists():raise FileExistsError('Fresh CPU preparation output required')
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    import psutil
    process=psutil.Process();allowed=process.cpu_affinity()
    process.cpu_affinity([18,19] if {18,19}<=set(allowed) else allowed[-2:])
    if os.name=='nt':process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    a.output.mkdir(parents=True);begun=time.monotonic()
    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-begun>a.max_seconds:raise RuntimeError('Finite CPU wall budget reached')
    def write(path,value):
        temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
        os.replace(temporary,path)
    write(a.output/'launch.json',dict(plan(a),PID=os.getpid(),create_time=process.create_time(),
        started_UTC=datetime.now(timezone.utc).isoformat(),cpu_affinity=process.cpu_affinity()))
    try:
        import numpy as np
        import torch
        from torch import nn
        from torch.nn import functional as F
        torch.set_num_threads(2)
        if torch.cuda.is_initialized():raise RuntimeError('CUDA must remain uninitialized')
        from fly_touhou.continuous_normalized_risk_coordinate import (convert_risk_layers,
            convert_training_payload,tensor_sha,PROTOCOL,TRAINING_SCHEMA,PARAMETER_SCHEMA)
        from fly_touhou.fly_dn_cost_native_policy import (compose_from_active_mask,
            normal_unprotected_from_features,source_policy_logits)
        guard();source_sha=sha(a.training_source);stats_sha=sha(a.statistics)
        with np.load(a.statistics,allow_pickle=False) as archive:
            mean=torch.from_numpy(archive['mean'].copy());scale=torch.from_numpy(archive['scale'].copy())
        index_path=a.activity_dir/'index.json';packet=a.activity_dir/'activations.npz'
        index=json.loads(index_path.read_text('utf-8'));packet_sha=sha(packet)
        if packet_sha!=index['payload_sha256']:raise ValueError('Committed real64 packet byte identity differs')
        with np.load(packet,allow_pickle=False) as archive:
            data={k:archive[k].copy() for k in ('DN_activation','features','raw_logits','predicted_costs','action_probs')}
        DN=torch.from_numpy(data['DN_activation']);features=torch.from_numpy(data['features'])
        if DN.shape!=(64,1299) or DN.dtype!=torch.float32 or len(index['records'])!=64:
            raise ValueError('The existing committed64 raw ANN DN scenes required')
        source=torch.load(a.training_source,map_location='cpu',weights_only=True)
        if source['model_payload']['configuration']['risk_beta']!=8.:
            raise ValueError('Same beta8 risk execution function required')
        raw_risk={k:v.detach().clone() for k,v in source['model_payload']['risk_head'].items()}
        # Only small actor/risk heads are instantiated;no whole model/core.
        class RiskOnly(nn.Module):
            def __init__(self):
                super().__init__();self.layers=nn.Sequential(nn.Linear(1299,64),nn.SiLU(),nn.Linear(64,18))
            def forward(self,x):return F.softplus(self.layers(x))
        with torch.random.fork_rng(devices=[]):
            risk=RiskOnly()
            actor=nn.Sequential(nn.LayerNorm(1299),nn.Linear(1299,256),nn.SiLU(),nn.Linear(256,19))
        risk.load_state_dict(raw_risk,strict=True);actor.load_state_dict(source['model_payload']['actor'],strict=True)
        risk.requires_grad_(False).eval();actor.requires_grad_(False).eval()
        def error(actual,reference):
            delta=(actual.double()-reference.double()).abs()
            return dict(max_abs=float(delta.max()),RMS=float(delta.square().mean().sqrt()),
                max_relative_to_one_plus_reference=float((delta/(1.+reference.double().abs())).max()),
                all_finite=bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()))
        def probability(raw,cost):
            composed=compose_from_active_mask(raw,cost,normal_unprotected_from_features(features),beta=8.,check_finite=False)
            return source_policy_logits(composed,features,check_finite=False).softmax(-1)
        guard()
        with torch.no_grad():
            raw=actor(DN);before_score=risk.layers(DN);before_margin=1.-.5*before_score
            before_cost=F.softplus(before_score);before_p=probability(raw,before_cost)
            convert_risk_layers(risk,mean,scale)
            after_score=risk.layers(DN);after_margin=1.-.5*after_score
            after_cost=risk(DN);after_p=probability(raw,after_cost)
        validation=dict(rows=64,DN_units='dimensionless ANN activation',
            margin=error(after_margin,before_margin),cost=error(after_cost,before_cost),probability19=error(after_p,before_p),
            greedy19_changes=int((after_p.argmax(-1)!=before_p.argmax(-1)).sum()),
            signed_and_forward_share_exact_layers=bool(torch.equal(after_cost,F.softplus(after_score))),
            source_CPU_vs_recorded_GPU=dict(raw_actor=error(raw,torch.from_numpy(data['raw_logits'])),
                cost=error(before_cost,torch.from_numpy(data['predicted_costs'])),
                probability19=error(before_p,torch.from_numpy(data['action_probs']))),
            actor_real_row_forwards=64,risk_real_row_forwards=192,
            fixed_error_rejection_threshold=None,
            limitation='One committed64-scene arithmetic check;not generalization,training improvement or native performance')
        if not all(validation[k]['all_finite'] for k in ('margin','cost','probability19')):
            raise FloatingPointError('Nonfinite normalized-risk forward')
        spec=dict(protocol=PROTOCOL,statistics_path=str(a.statistics),statistics_sha256=stats_sha,
            mean_sha256=tensor_sha(mean),scale_sha256=tensor_sha(scale),
            buffers=['layers.DN_mean','layers.DN_scale'],fixed_train_only=True,
            origin='original signed4f train-only mean/scale;not recomputed from these64 or current PPO',
            coordinate_change='Wnorm=Wraw*scale;bnorm=braw+Wraw@mean',
            transform_parameters=['risk_head.layers.0.weight','risk_head.layers.0.bias'],
            numeric_optimizer_resume=False)
        guard();preservation=convert_training_payload(source,risk.state_dict(),spec)
        source['counters']=copy.deepcopy(source['counters'])
        source['counters']['risk_coordinate_conversions']=source['counters'].get('risk_coordinate_conversions',0)+1
        provenance=dict(parent=dict(path=str(a.training_source),sha256=source_sha),normalization=spec,
            activity=dict(index_path=str(index_path),index_sha256=sha(index_path),packet_path=str(packet),packet_sha256=packet_sha,
                policy_version=index['policy_version'],records=index['records']),
            validation=validation,preservation=preservation,new_training_phase=True,
            interpretation='Forward-equivalent within measured FP32 rounding;not equivalent diagonal-Adam/global-clip optimization',
            bottleneck_or_improvement_claim=False,optimizer_instances=0,optimizer_steps=0,
            core_constructors=0,core_forwards=0,GPU_initialized=torch.cuda.is_initialized(),native_actions=0)
        source['provenance']=copy.deepcopy(source.get('provenance') or {})
        source['provenance']['normalized_risk_coordinate_conversion']=provenance
        def save(path,value):
            temporary=path.with_suffix(path.suffix+'.tmp')
            with temporary.open('wb') as stream:torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,path)
        save(a.output/'converted-training.pt',source)
        save(a.output/'converted-parameters.pt',source['model_payload'])
        np.savez(a.output/'same64-forward.npz',DN=DN.numpy(),features=features.numpy(),raw_actor=raw.numpy(),
            before_margin=before_margin.numpy(),after_margin=after_margin.numpy(),
            before_cost=before_cost.numpy(),after_cost=after_cost.numpy(),
            before_probabilities=before_p.numpy(),after_probabilities=after_p.numpy())
        result=dict(schema=SCHEMA,status='completed_CPU_conversion_not_installed',source_sha256=source_sha,
            normalized_training_schema=TRAINING_SCHEMA,normalized_parameter_schema=PARAMETER_SCHEMA,
            normalization=spec,validation=validation,preservation=preservation,
            provenance=provenance,elapsed_seconds=time.monotonic()-begun,CUDA_initialized=torch.cuda.is_initialized(),
            optimizer_instances=0,optimizer_steps=0,backward_passes=0,core_forwards=0,native_actions=0,
            converted_training=dict(path=str(a.output/'converted-training.pt'),sha256=sha(a.output/'converted-training.pt')),
            converted_parameters=dict(path=str(a.output/'converted-parameters.pt'),sha256=sha(a.output/'converted-parameters.pt')),
            candidate_installed=False,live_source_modified=False)
        write(a.output/'report.json',result)
        print(json.dumps(dict(output=str(a.output),source_sha256=source_sha,validation=validation,
            actual_states_removed=preservation['states_removed'],CPU_wall_seconds=result['elapsed_seconds']),ensure_ascii=False))
        return result
    except BaseException as e:
        write(a.output/'failure.json',dict(schema=SCHEMA,error=repr(e),optimizer_steps=0,core_forwards=0,
            native_actions=0,candidate_installed=False,live_source_modified=False));raise


def main(argv=None):
    a=parse_args(argv)
    if not a.execute_cpu:print(json.dumps(plan(a),ensure_ascii=False,indent=2));return 0
    execute(a);return 0


if __name__=='__main__':raise SystemExit(main())
