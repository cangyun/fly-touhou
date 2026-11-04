"""One finite balanced input/gain pilot; frozen motor, no native/teacher model.

Original cache and DAgger Phi are observations, never neural states. Each
prefix/replay starts at row0 with its original key. Detached training history
becomes stale after an update; only final frozen held replay rebuilds an exact
own history under one parameter version. No numerical training resume exists.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import torch

from .fly_cached_distillation import LabelRows, Metrics, atomic_json, episode_key, file_sha, FitStop
from .fly_distillation_student import legal_action_distillation_kl
from .fly_gain_centered_distillation import GainCenteredFlyDistillationStudent, GAIN_NAME

SCHEMA='fly-gain01-balanced-input-brain-finite-pilot-v1'
INPUT_NAMES=('brain.input_encoder.weight','brain.input_encoder.bias')
BODY_SHA='58e9fd9ba99d6384b94e05a438e8ad7f9199daa47eb32133f8f07edeeab5379f'


def tensor_sha(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@dataclass
class Prefix:
    name: str
    key: int
    phi: np.ndarray
    target: np.ndarray
    legal: np.ndarray
    provenance: dict
    metadata: list | None = None

    def __post_init__(self):
        n=len(self.phi)
        if (n<1 or self.phi.shape!=(n,9721) or self.phi.dtype!=np.float32 or not np.isfinite(self.phi).all()
                or np.any(np.abs(self.phi)>1) or self.target.shape!=(n,19) or self.target.dtype!=np.float32
                or not np.isfinite(self.target).all() or np.any(self.target<0)
                or not np.allclose(self.target.sum(-1),1.,atol=2e-6,rtol=0)
                or self.legal.shape!=(n,19) or self.legal.dtype!=np.bool_ or not self.legal[:,:18].all()
                or np.any(self.target[~self.legal]!=0) or not np.array_equal(self.legal[:,18],self.phi[:,4]>0)
                or type(self.key) is not int or not 0<=self.key<2**31):
            raise ValueError('Require original finite FP32 Phi9721/final T1 legal19 prefix')
        for value in (self.phi,self.target,self.legal):value.setflags(write=False)


def load_original_prefix(reader,name,count,guard,audit):
    labels=LabelRows(guard);phis=[];targets=[];legals=[];metadata=[]
    inspection=reader.inspect(name)
    if inspection.get('eligible') is not True:raise ValueError('Complete formal source cache required')
    with closing(reader.iter_episode(name,max_rows=count,audit_dir=audit)) as rows:
        for row in rows:
            target,legal=labels.read(row)
            if row.metadata.source_row!=len(phis):raise ValueError('Old source must start row0 and remain adjacent')
            phis.append(row.phi.copy());targets.append(target.copy());legals.append(legal.copy());metadata.append(row.metadata)
    if len(phis)!=count:raise ValueError('Requested original continuous prefix incomplete')
    return Prefix(name,episode_key(name,613711),np.stack(phis),np.stack(targets),np.stack(legals),
        dict(source='formal original Phi/label bank',rows=count,source_rows=inspection['rows'],
            complete_episode=metadata[-1].dones,inspection=inspection,label_files=labels.files_read),metadata)


def load_round03_prefixes(index_path,student,guard,*,expected_index_sha256):
    # Reuse the already exercised DAgger validation; its DN arrays are checked
    # but never enter this model/loss. Only Phi is passed to the complete brain.
    from refit_fly_DN_motor_KL_DAgger import load_dagger
    path=Path(index_path)
    if file_sha(path)!=expected_index_sha256:raise ValueError('Explicit round03 data index SHA differs')
    declared=json.loads(path.read_text(encoding='utf-8'))
    body=dict(outer_checkpoint_sha256=BODY_SHA,fixed_mean_raw_sha256=tensor_sha(student.brain.input_encoder.fixed_train_mean),
        graph_fingerprint=student.graph_fingerprint,ports_manifest_sha256=student.interface_provenance['manifest_sha256'])
    index,arrays,refs=load_dagger(path,body,guard,expected_student_sha256=declared['student_checkpoint_sha256'])
    if index.get('brain_seed')!=613711:
        raise ValueError('Use final round03 train collection with its original brain keys')
    parameters=dict(student.named_parameters())
    if any(index['student_parameter_raw_sha256'][name]!=tensor_sha(parameters[name]) for name in INPUT_NAMES):
        raise ValueError('Collected body/input differs from explicit centered migration')
    result=[]
    for game,episode in enumerate(index['episodes']):
        limit=min(352,episode['rows']);pieces=[];selected_refs=[]
        for spec,ref in zip(index['chunks'],refs):
            if spec['episode_index']!=game or spec['source_row_start']>=limit:continue
            guard.check('round03_fixed_Phi_read')
            with np.load(path.parent/spec['name'],allow_pickle=False) as archive:phi=archive['phi']
            if phi.dtype!=np.float32 or phi.shape!=(spec['rows'],9721) or not np.isfinite(phi).all():
                raise ValueError('DAgger packet does not contain actual fixed FP32 Phi')
            pieces.append(phi[:limit-spec['source_row_start']].copy());selected_refs.append(ref)
        selected=(arrays['episode_index']==game)&(arrays['source_row']<limit)
        phi=np.concatenate(pieces)
        if len(phi)!=limit or not np.array_equal(arrays['source_row'][selected],np.arange(limit)):
            raise ValueError('Round03 prefix cannot skip rows or start from a late zero state')
        result.append(Prefix('round03/episode'+str(game),int(episode['brain_rng_key']),phi,
            arrays['teacher_probabilities'][selected].copy(),arrays['legal_action_mask'][selected].copy(),
            dict(source='round03 actual student trajectory',rows=limit,source_rows=episode['rows'],
                complete_episode=limit==episode['rows'],episode=episode,selected_payloads=selected_refs,
                collection_master_seed=index['seed'],data_index_sha256=expected_index_sha256)))
    return result


class Cursor:
    def __init__(self,prefixes):
        if not prefixes:raise ValueError('Nonempty ordered prefixes required')
        self.prefixes=prefixes;self.index=0;self.row=0;self.state=None;self.rows=0;self.restarts=0

    def forward(self,student):
        prefix=self.prefixes[self.index]
        if self.row==len(prefix.phi):
            self.index=(self.index+1)%len(self.prefixes);self.row=0;self.state=None;self.restarts+=1
            prefix=self.prefixes[self.index]
        if self.state is None:
            if self.row!=0:raise ValueError('No late-frame zero-state training')
            self.state=student.initial_state(1,rng_seed=prefix.key)
        i=self.row
        logits,self.state=student.brain(torch.tensor(prefix.phi[i],device=student.device)[None],self.state)
        if int(self.state.noise_counter.item())!=(i+1)*100 or int(self.state.rng_keys.item())!=prefix.key:
            raise ValueError('Own full state RNG/counter continuity differs')
        self.row+=1;self.rows+=1
        return logits,prefix.target[i],prefix.legal[i]

    def detach(self):
        if self.state is not None:self.state=self.state.detach()

    def audit(self):
        return dict(prefix=self.prefixes[self.index].name,next_source_row=self.row,rows=self.rows,
            prefix_restarts_from_row0=self.restarts,noise_counter=None if self.state is None else int(self.state.noise_counter.item()))


def fit_joint(student,old,new,held,output,guard,*,max_updates=512,provenance=None,synthetic_fixture=False):
    if (not isinstance(student,GainCenteredFlyDistillationStudent) or student.brain.recurrent_gain_limit!=.1
            or type(max_updates) is not int or not 1<=max_updates<=512):
        raise ValueError('Typed gain.1 student and finite <=512 updates required')
    if synthetic_fixture:
        if student.device.type!='cpu' or student.brain.require_full_graph:raise ValueError('Artificial CPU non-full graph only')
    elif student.device.type!='cuda' or not student.brain.require_full_graph:
        raise ValueError('Production pilot explicitly requires the full CUDA brain')
    params=dict(student.named_parameters())
    if set(params)!={*INPUT_NAMES,GAIN_NAME,'brain.action_readout.weight'}:raise ValueError('Unexpected parameter boundary')
    for name,p in params.items():p.requires_grad_(name in (*INPUT_NAMES,GAIN_NAME))
    motor_sha=tensor_sha(params['brain.action_readout.weight']);mean_sha=tensor_sha(student.brain.input_encoder.fixed_train_mean)
    if bool(torch.count_nonzero(params[GAIN_NAME])):raise ValueError('Start only from explicit zero-gain migration')
    output=Path(output)
    report=dict(schema=SCHEMA,status='running',phase='initial_parameters',provenance=provenance or {},synthetic_fixture=synthetic_fixture,
        loss='.5 mean legal T1 KL(old consecutive4) + .5 mean legal T1 KL(new consecutive4)',lambda_value=0.,
        batch=1,tbptt=4,gain_limit=.1,motor_frozen=True,input_learning_rate=1e-5,gain_learning_rate=3e-3,
        optimizer_updates=0,optimizer_step_attempts=0,backward_passes=0,training_data_forwards=0,
        held_data_forwards=0,capture_setup_forwards=0,total_model_forward_attempts=0,transaction_complete=True,
        complete_update_checkpoint_only=True,numerical_resume=False,optimizer_or_neural_state_serialized=False,
        native_worlds_started=0,teacher_model_loaded=False,teacher_hidden_or_values_used=False,
        history='Training states retain detached pre-update history; stale after parameter updates. Final held is fresh frozen row0 replay.',
        training_prefixes={'old':[p.provenance for p in old],'new':[p.provenance for p in new]},
        checkpoints=[],held_validation=None)
    cursors={'old':Cursor(old),'new':Cursor(new)};last_saved=-1
    def progress():
        report['resources']=guard.report();report['cursors']={k:c.audit() for k,c in cursors.items()}
        atomic_json(output/'progress.json',report)
    def save(name,reason):
        nonlocal last_saved
        if not report['transaction_complete']:return
        path=output/name;tmp=output/('.parameters-'+uuid.uuid4().hex+'.tmp')
        try:
            with tmp.open('xb') as f:torch.save(student.parameter_checkpoint_payload(),f);f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:tmp.unlink(missing_ok=True)
        record=dict(path=str(path.resolve()),sha256=file_sha(path),optimizer_updates=report['optimizer_updates'],reason=reason)
        report['checkpoints'].append(record);report['last_completed_model']=record;last_saved=report['optimizer_updates']
        if name=='initial-parameters.pt':report['initial_model']=record
        if name=='final-parameters.pt':report['final_model']=record
        progress()
    optimizer=torch.optim.Adam([{'params':[params[n] for n in INPUT_NAMES],'lr':1e-5},
        {'params':[params[GAIN_NAME]],'lr':3e-3}],weight_decay=0.)
    try:
        save('initial-parameters.pt','zero_gain_migration_before_optimizer')
        report['phase']='train_balanced_TBPTT4';student.train()
        for _ in range(max_updates):
            optimizer.zero_grad(set_to_none=True);terms={}
            for group in ('old','new'):
                losses=[]
                for _ in range(4):
                    guard.check('before_'+group+'_full_brain_forward');report['total_model_forward_attempts']+=1
                    logits,target,legal=cursors[group].forward(student);report['training_data_forwards']+=1
                    loss=legal_action_distillation_kl(logits,torch.tensor(target,device=student.device)[None],
                        torch.tensor(legal,device=student.device)[None],temperature=1.)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite joint legal KL')
                    losses.append(loss)
                term=torch.stack(losses).mean();terms[group]=float(term.detach())
                guard.check('before_'+group+'_backward');(.5*term).backward();report['backward_passes']+=1
                cursors[group].detach();losses=[];del loss,term,logits
            allowed=[params[n] for n in (*INPUT_NAMES,GAIN_NAME)]
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in allowed):
                raise FloatingPointError('Missing/nonfinite input or gain surrogate gradient')
            report['last_gradient_RMS']={n:float(params[n].grad.square().mean().sqrt()) for n in (*INPUT_NAMES,GAIN_NAME)}
            guard.check('before_joint_Adam');report['transaction_complete']=False;report['optimizer_step_attempts']+=1
            optimizer.step()
            if student.device.type=='cuda':torch.cuda.synchronize(student.device)
            if any(not bool(torch.isfinite(p).all()) for p in allowed):raise FloatingPointError('Nonfinite parameter after Adam')
            report['transaction_complete']=True;report['optimizer_updates']+=1
            report['last_completed_update']=dict(optimizer_updates=report['optimizer_updates'],old_mean_KL=terms['old'],
                new_mean_KL=terms['new'],balanced_KL=.5*(terms['old']+terms['new']))
            if report['optimizer_updates']%64==0:save('last-completed-parameters.pt','periodic_complete_update')
            progress()
        optimizer.zero_grad(set_to_none=True)
        if tensor_sha(params['brain.action_readout.weight'])!=motor_sha or tensor_sha(student.brain.input_encoder.fixed_train_mean)!=mean_sha:
            raise ValueError('Frozen motor/fixed mean changed')
        student.brain.verify_propagation_buffers();save('final-parameters.pt','training_complete_validation_pending')
        report['phase']='frozen_after_held010';student.eval().requires_grad_(False)
        guard.check('before_frozen_held_capture')
        if student.device.type=='cuda':
            report['total_model_forward_attempts']+=4;student.brain.enable_cuda_capture(1);report['capture_setup_forwards']=4
        metrics=Metrics();state=student.initial_state(1,rng_seed=held.key)
        with torch.no_grad():
            for i in range(len(held.phi)):
                guard.check('before_frozen_held_forward');report['total_model_forward_attempts']+=1
                logits,state=student.brain(torch.tensor(held.phi[i],device=student.device)[None],state)
                report['held_data_forwards']+=1
                if int(state.noise_counter.item())!=100*(i+1):raise ValueError('Frozen held own state counter differs')
                loss=legal_action_distillation_kl(logits,torch.tensor(held.target[i],device=student.device)[None],
                    torch.tensor(held.legal[i],device=student.device)[None],temperature=1.)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite frozen held KL')
                meta=held.metadata[i] if held.metadata is not None else SimpleNamespace(stage=1)
                metrics.add(SimpleNamespace(phi=held.phi[i],metadata=meta),held.target[i],held.legal[i],logits,float(loss),state,student.brain.output_indices)
                if (i+1)%32==0:progress()
        report['held_validation']=dict(rows=len(held.phi),metrics=metrics.report(),parameters_frozen_for_entire_replay=True,
            own_history_rebuilt_from_row0=True,episode_rng_key=held.key,noise_counter_at_end=int(state.noise_counter.item()),
            complete_episode=held.provenance.get('complete_episode',False),before_validation_run=False)
        report.update(status='completed_finite_joint_pilot',phase='completed',frozen_motor_and_mean_unchanged=True,
            gain_summary=student.brain.recurrent_plasticity_summary())
    except BaseException as error:
        for cursor in cursors.values():cursor.detach()
        student.zero_grad(set_to_none=True)
        report.update(status='stopped_finite_budget' if isinstance(error,FitStop) else 'failed',reason=repr(error),
            unvalidated_after_update=report['optimizer_updates']>0 and report['held_validation'] is None)
        if report['transaction_complete'] and last_saved!=report['optimizer_updates']:
            try:save('last-completed-parameters.pt','finite_exit_last_complete_update')
            except Exception as save_error:report['checkpoint_save_error']=repr(save_error)
    finally:
        student.brain.clear_cuda_capture();progress();atomic_json(output/'result.json',report)
    return report
