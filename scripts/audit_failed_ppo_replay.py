"""Offline reproduce a sealed PPO mismatch; never starts or controls a game."""
from pathlib import Path
import argparse
import json
import os
import sys
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--rollout',type=Path,required=True)
parser.add_argument('--source-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--repeats',type=int,default=3)
parser.add_argument('--mode',choices=('default','deterministic','bmm-deterministic','cached-deterministic-probe'),default='default')
parser.add_argument('--capture-inference',action='store_true')
parser.add_argument('--require-exact',action='store_true',help='Fail instead of merely reporting any error beyond the existing 1e-9 behavior tolerance')
args=parser.parse_args()
if args.mode!='default':
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch.nn import functional as F
from fly_touhou.connectome import load_connectome,file_sha256
from fly_touhou.whole_brain_learning import load_whole_brain_checkpoint
from fly_touhou.whole_brain_ppo import WholeBrainRollout,load_rollout,replay_segment,restore_state

if args.output.exists():
    raise FileExistsError('Preserve previous replay audits')
torch.set_num_threads(4)
torch.use_deterministic_algorithms(args.mode!='default')
graph=load_connectome(args.source_dir,max_neurons=None)
policy=load_whole_brain_checkpoint(args.checkpoint,graph,'cuda')
model=policy.student.eval()
model.verify_propagation_buffers()
if args.capture_inference:
    model.enable_cuda_capture()
if args.mode=='bmm-deterministic':
    # Diagnostic backend substitution only, explicitly reported below. The
    # permanent model path must also use this adjoint in training.
    base=model.recurrent.coalesce()
    indices=torch.cat((torch.zeros((1,base._nnz()),dtype=torch.int64,device=model.device),base.indices()))
    batched=torch.sparse_coo_tensor(indices,base.values(),(1,model.n_neurons,model.n_neurons),is_coalesced=True)
    def deterministic_current(spikes):
        result=torch.bmm(batched,spikes.T.unsqueeze(0)).squeeze(0).T
        if model.recurrent_gain_logits is not None:
            result=result*(1+model.recurrent_gain_limit*torch.tanh(model.recurrent_gain_logits))
        return result
    model.recurrent_current=deterministic_current
if args.mode=='cached-deterministic-probe':
    from benchmark_cached_cusparse import CachedCooProbe, ProbePropagation
    vendor_header=Path(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include\cusparse.h')
    incoming=CachedCooProbe(model.recurrent,vendor_header)
    outgoing=CachedCooProbe(model.recurrent_transpose,vendor_header)
    def cached_current(spikes):
        result=ProbePropagation.apply(spikes,incoming,outgoing)
        if model.recurrent_gain_logits is not None:
            result=result*(1+model.recurrent_gain_limit*torch.tanh(model.recurrent_gain_logits))
        return result
    model.recurrent_current=cached_current
saved=load_rollout(args.rollout)
for name,value in model.named_parameters():
    if not torch.equal(value.detach().cpu(),saved['behavior_parameters'][name]):
        raise ValueError('Audit parent differs from sealed behavior weights')
rollout=WholeBrainRollout(**{key:(saved[key].numpy() if isinstance(saved[key],torch.Tensor) else saved[key])
    for key in WholeBrainRollout.__dataclass_fields__})
report={'schema':'failed-full-brain-ppo-offline-replay-audit-v1','mode':args.mode,
    'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':file_sha256(args.checkpoint),
    'rollout':str(args.rollout.resolve()),'rollout_sha256':file_sha256(args.rollout),
    'torch':torch.__version__,'cuda':torch.version.cuda,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
    'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
    'neurons':model.n_neurons,'edges':model.graph.n_edges,'substeps':model.substeps,
    'actor_matches_sealed_behavior_exactly':True,'new_native_actions':0,'passes':[]}
report['model_propagation_backend']=model.propagation_backend
report['native_sparse_implementation']=model.numerical_implementation_metadata()
report['inference_execution']=model.inference_execution_specification()
report['requires_exact_replay']=args.require_exact or args.mode=='cached-deterministic-probe'
report['strict_behavior_log_probability_tolerance']=1e-9
if args.mode=='cached-deterministic-probe':
    report['prototype']={'method':'local no-game cached COO ALG2 forward/adjoint override',
        'implementation_path':str(Path(__file__).with_name('benchmark_cached_cusparse.py')),
        'implementation_sha256':file_sha256(Path(__file__).with_name('benchmark_cached_cusparse.py')),
        'library_sha256':file_sha256(incoming.library_path),'library_version':incoming.version,
        'production_configuration_changed':False,'strict_tolerance':1e-9}
try:
    previous=None
    with torch.no_grad():
        sample=restore_state(rollout.initial_states[0],model.device).delay_queue[0]
        reference=model.recurrent_current(sample)
        biggest=0.
        differing=0
        for _ in range(100):
            current=model.recurrent_current(sample)
            error=float((current-reference).abs().max())
            biggest=max(biggest,error)
            differing+=int(error!=0.)
        report['identical_input_propagation']={'calls':100,'nonidentical_calls':differing,'max_absolute_error_mv':biggest}
        for repetition in range(args.repeats):
            started=time.monotonic()
            actual=[]
            for start,end in rollout.segments():
                logits=replay_segment(model,rollout,start,end)
                probability=F.log_softmax(logits.double(),-1).gather(1,
                    torch.as_tensor(rollout.actions[start:end],device=model.device).unsqueeze(1)).squeeze(1)
                actual.extend(probability.cpu().tolist())
            actual=np.asarray(actual)
            errors=actual-rollout.old_log_probabilities
            item={'pass':repetition,'seconds':time.monotonic()-started,
                'max_error_vs_recorded':float(np.abs(errors).max()),
                'different_indices_vs_recorded':np.flatnonzero(np.abs(errors)>1e-9).tolist(),
                'log_probability_errors':errors.tolist(),
                'max_difference_vs_previous_pass':None if previous is None else float(np.abs(actual-previous).max())}
            report['passes'].append(item)
            if report['requires_exact_replay'] and float(np.abs(errors).max())>1e-9:
                raise AssertionError('Cached ALG2 violates the existing strict behavior replay tolerance')
            previous=actual
            print(json.dumps({k:v for k,v in item.items() if k!='log_probability_errors'}),flush=True)
except BaseException as error:
    report['error']={'type':type(error).__name__,'message':str(error)}
    raise
finally:
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as output:
        json.dump(report,output,ensure_ascii=False,indent=2,allow_nan=False)
