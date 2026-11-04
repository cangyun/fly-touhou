"""Finite D95 final19 distillation through the existing cached whole brain.

Import and default CLI inspection are NoTorch. Production learning requires
explicit CUDA; CPU execution is an explicit artificial-fixture API only.
No graph/reader/bank mutation, native process, service or resume is provided.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import uuid

from .windows_physical_memory import available_physical_memory_bytes

SCHEMA = 'fly-cached-complete-connectome-finite-distillation-v1'
D95_SHA = 'd95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'
BANK_ID = '1c6006dc7fbbbe5465e7afcef5f43afc8ffebfc4aba8e7833f4ad2b55eebb4f5'
GIB = 1024**3
ROOT = Path(__file__).resolve().parents[2]
META_KEYS = ('source_row', 'before_frames', 'after_observation_frames', 'episode_starts', 'dones',
             'native_seeds', 'stage', 'stage_frame', 'native_misses', 'native_bombs_used', 'behaviour_actions')


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path); temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class CachedFitConfig:
    max_rows: int = 32  # global training-row cap, from original row0
    validation_rows: int = 32  # independent row0 prefix budget for EACH frozen phase
    max_updates: int = 1
    max_model_decisions: int = 256  # all train + before/after validation
    max_seconds: float = 300.
    tbptt: int = 1
    learning_rate: float = 1e-4
    checkpoint_every_updates: int = 10
    gpu_fraction: float = .2
    minimum_ram_GiB: float = 10.
    minimum_disk_GiB: float = 4.
    rng_seed: int = 613711

    def validate(self):
        for name in ('max_rows', 'validation_rows', 'max_model_decisions', 'checkpoint_every_updates'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError('Positive finite integer required: '+name)
        if type(self.max_updates) is not int or self.max_updates < 0 or type(self.tbptt) is not int or not 1 <= self.tbptt <= 4:
            raise ValueError('Nonnegative update budget and explicit TBPTT1..4 required')
        for name in ('max_seconds', 'learning_rate', 'minimum_ram_GiB', 'minimum_disk_GiB'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Positive finite value required: '+name)
        if (type(self.gpu_fraction) not in (int, float) or not 0 < self.gpu_fraction <= .25
                or type(self.rng_seed) is not int or not 0 <= self.rng_seed < 2**31):
            raise ValueError('Explicit GPU quota <=.25 and RNG seed required')
        return self


class FitStop(RuntimeError):
    """Finite stop only; no automatic retries or looser resource limits."""


class FitGuard:
    def __init__(self, output, config, *, available_ram=None, free_disk=None, clock=None):
        self.output = self.destination = Path(output)
        self.config = config.validate()
        self.clock = clock or time.monotonic
        self.started = self.clock()
        self.available_ram = available_ram or available_physical_memory_bytes
        def disk_now():
            ancestor = self.output.parent
            while not ancestor.exists(): ancestor = ancestor.parent
            return shutil.disk_usage(ancestor).free
        self.free_disk = free_disk or disk_now
        self.minimum_ram = self.minimum_disk = None

    def check(self, stage):
        ram, disk = self.available_ram(), self.free_disk()
        if type(ram) is not int or ram < 0:
            raise ValueError('Unknown physical RAM; no zero or psutil fallback')
        self.minimum_ram = ram if self.minimum_ram is None else min(ram, self.minimum_ram)
        self.minimum_disk = disk if self.minimum_disk is None else min(disk, self.minimum_disk)
        if (self.output/'STOP').exists(): raise FitStop('STOP:'+stage)
        if self.clock()-self.started >= self.config.max_seconds: raise FitStop('wall_budget:'+stage)
        if ram < self.config.minimum_ram_GiB*GIB: raise FitStop('physical_RAM_floor:'+stage)
        if disk < self.config.minimum_disk_GiB*GIB: raise FitStop('disk_floor:'+stage)

    def report(self):
        return dict(elapsed_seconds=self.clock()-self.started, max_seconds=self.config.max_seconds,
                    minimum_available_RAM_bytes=self.minimum_ram, minimum_free_disk_bytes=self.minimum_disk,
                    RAM_backend='fresh GlobalMemoryStatusEx.ullAvailPhys on Windows')


def import_reader_preserving_CVD():
    """Caller must use this before importing/initializing Torch for CUDA."""
    present, original = 'CUDA_VISIBLE_DEVICES' in os.environ, os.environ.get('CUDA_VISIBLE_DEVICES')
    stages = [dict(stage='before_CPU_reader', CVD=original, Torch_imported='torch' in sys.modules)]
    try:
        module = importlib.import_module('fly_touhou.fly_sensory_full_episode_reader')
        stages.append(dict(stage='after_CPU_reader', CVD=os.environ.get('CUDA_VISIBLE_DEVICES'), Torch_imported='torch' in sys.modules))
    finally:
        if present: os.environ['CUDA_VISIBLE_DEVICES'] = original
        else: os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    stages.append(dict(stage='restored_before_Torch', CVD=os.environ.get('CUDA_VISIBLE_DEVICES'), Torch_imported='torch' in sys.modules))
    return module, stages


def validate_lists(train, held):
    train, held = list(train), list(held)
    if (not train or len(set(train)) != len(train) or len(set(held)) != len(held) or set(train)&set(held)
            or any(not isinstance(n, str) or Path(n).name != n or n in ('', '.', '..') for n in train+held)):
        raise ValueError('Distinct ordered train/held episode lists required; no overlap or unsafe names')
    return train, held


def inspect_episodes(reader, train, held):
    train, held = validate_lists(train, held)
    result = {name:reader.inspect(name) for name in train+held}
    if any(row.get('eligible') is not True or row.get('complete_producer_episode') is not True for row in result.values()):
        raise ValueError('Only independently complete COMMITTED Phi episodes may supply a continuous prefix')
    return result


def episode_key(name, seed):
    return int.from_bytes(hashlib.sha256((str(seed)+':'+name).encode()).digest()[:4], 'little') & 0x7fffffff


def process_identity():
    import psutil
    return dict(pid=os.getpid(),create_time=psutil.Process().create_time())


class LabelRows:
    """One verified <=32-row final19 shard; never read value/hidden/actor arrays."""
    def __init__(self, guard, *, synthetic_fixture=False):
        self.guard, self.fixture = guard, synthetic_fixture
        self.key = self.arrays = None
        self.files_read = self.bytes_read = 0

    def read(self, row):
        import numpy as np
        ref, meta = row.teacher, row.metadata
        root, path = Path(ref.bank_root).resolve(), Path(ref.path).resolve()
        if (path.parent != root/'episodes'/meta.episode or not path.name.startswith('labels-') or path.suffix != '.npz'
                or (not self.fixture and ref.bank_identity_sha256 != BANK_ID)
                or ref.episode != meta.episode or ref.source_row != meta.source_row
                or type(ref.local_row) is not int or type(ref.shard_rows) is not int or not 1 <= ref.shard_rows <= 32
                or not 0 <= ref.local_row < ref.shard_rows or ref.shard_start+ref.local_row != meta.source_row):
            raise ValueError('Invalid immutable D95 label-file/row reference')
        key = (str(path), ref.sha256, ref.file_bytes, ref.shard_start, ref.shard_rows)
        if key != self.key:
            self.guard.check('label_SHA_before_read')
            if path.stat().st_size != ref.file_bytes or file_sha(path) != ref.sha256:
                raise ValueError('Label SHA/byte count differs from immutable reference')
            with np.load(path, allow_pickle=False) as archive:
                values = {name:archive[name] for name in (*META_KEYS, 'teacher_probabilities', 'legal_action_mask')}
            n = ref.shard_rows
            for name in META_KEYS:
                dtype = np.bool_ if name in ('episode_starts', 'dones') else np.int64
                if values[name].shape != (n,) or values[name].dtype != dtype:
                    raise ValueError('Label source metadata dtype/shape differs')
            target, legal = values['teacher_probabilities'], values['legal_action_mask']
            if (target.dtype != np.float32 or target.shape != (n,19) or legal.dtype != np.bool_ or legal.shape != (n,19)
                    or not np.isfinite(target).all() or np.any(target<0)
                    or not np.allclose(target.sum(-1,dtype=np.float64),1.,atol=2e-6,rtol=0)
                    or not legal[:,:18].all() or not (target*legal).sum(-1).all()):
                raise ValueError('Expected normalized final19 probabilities and bank-only inventory legal mask')
            self.key, self.arrays = key, values
            self.files_read += 1; self.bytes_read += ref.file_bytes
        i = ref.local_row
        if any(self.arrays[name][i].item() != getattr(meta,name) for name in META_KEYS):
            raise ValueError('Teacher label source row/clock/seed/reset metadata differs')
        return self.arrays['teacher_probabilities'][i], self.arrays['legal_action_mask'][i]


class Metrics:
    def __init__(self):
        self.groups = {}; self.per_action = {str(i):dict(rows=0, correct=0) for i in range(19)}
        self.teacher_greedy = [0]*19; self.student_greedy = [0]*19
        self.hard = dict(whole_brain_spikes=0, descending_spikes=0, input_hard_events=None,
                        rows_measured=0,active_DN_sum=0,active_brain_neurons_sum=0,DN_rate_mean_Hz_sum=0.,
                        input_event_limit='Original WholeBrainState does not retain Poisson input counts; not recomputed')

    def add(self, row, target, legal, logits, loss, state, output_indices):
        import numpy as np
        import torch
        phi = row.phi; objects = phi[25:25+96*29].reshape(96,29)
        present = objects[:,0]>.5
        hazards = present & (objects[:,[1,3,4]].max(-1)>.5)
        near = bool((hazards & (np.linalg.norm(objects[:,6:8],axis=1)<.15)).any())
        count = int(present.sum()); occupancy = '0' if count==0 else '1..15' if count<16 else '16..63' if count<64 else '64..96'
        teacher_action = int(np.where(legal,target,-1).argmax())
        predicted = int(logits.detach().masked_fill(~torch.tensor(legal,device=logits.device)[None],-float('inf')).argmax(-1).item())
        correct = int(predicted==teacher_action)
        groups = ['all','stage/'+str(row.metadata.stage),'objects/'+occupancy,
                  'hazard/'+('near_token_proxy' if near else 'visible' if hazards.any() else 'none'),
                  'objects_nonempty/'+str(bool(count)), 'Phi_nonzero/'+str(bool(np.any(phi!=0))),
                  'selected_laser/'+str(bool((present & (objects[:,3:5].max(-1)>.5)).any())),
                  'player_circle_large_proxy/'+str(bool(phi[13]>4/64))]
        if phi[24]>.5 and phi[19]>0: groups.append('rare/observed_deathbomb_state')
        if phi[24]>.5 and phi[4]>0 and legal[18] and .25<=phi[19]<=1: groups.append('rare/usable_deathbomb_inventory_timer_proxy')
        if max(abs(float(phi[13])),float(np.linalg.norm(phi[14:16])))>4/64: groups.append('rare/enlarged_geometry_proxy')
        for name in groups:
            item = self.groups.setdefault(name,dict(rows=0,KL_sum=0.,correct=0))
            item['rows']+=1; item['KL_sum']+=float(loss); item['correct']+=correct
        self.per_action[str(teacher_action)]['rows']+=1; self.per_action[str(teacher_action)]['correct']+=correct
        self.teacher_greedy[teacher_action]+=1;self.student_greedy[predicted]+=1
        self.hard['whole_brain_spikes']+=int(state.window_spike_counts.sum())
        self.hard['descending_spikes']+=int(state.window_spike_counts[:,output_indices].sum())
        self.hard['rows_measured']+=1
        self.hard['active_DN_sum']+=int((state.window_spike_counts[:,output_indices]>0).sum())
        self.hard['active_brain_neurons_sum']+=int((state.window_spike_counts>0).sum())
        self.hard['DN_rate_mean_Hz_sum']+=float(state.rate.detach().mean())

    def report(self):
        return dict(groups={key:{**row,'mean_legal_KL':row['KL_sum']/row['rows'],'argmax_agreement':row['correct']/row['rows']} for key,row in self.groups.items()},
            per_teacher_action={key:{**row,'accuracy':row['correct']/row['rows'] if row['rows'] else None} for key,row in self.per_action.items()},
            teacher_greedy_counts=self.teacher_greedy,student_greedy_counts=self.student_greedy,
            hard_window_counts={**self.hard,
                'mean_active_DN':self.hard['active_DN_sum']/self.hard['rows_measured'] if self.hard['rows_measured'] else None,
                'mean_active_brain_neurons':self.hard['active_brain_neurons_sum']/self.hard['rows_measured'] if self.hard['rows_measured'] else None,
                'mean_DN_filtered_rate_Hz':self.hard['DN_rate_mean_Hz_sum']/self.hard['rows_measured'] if self.hard['rows_measured'] else None},
            limits='Potential hazard/normalized token-distance and enlarged-geometry proxies only; not collision truth. Phi_nonzero alone can be only globals. No rare positives means uncovered, not solved. Hard count aggregates do not certify event timing or per-row event equality.')


def fit_cached_student(student, reader, train, held, output, config, *, guard=None, synthetic_fixture=False):
    """Fresh finite fit; parameter-only checkpoints are deliberately not resumes."""
    import torch
    from .cached_whole_brain_bridge import CachedWholeBrainBridge
    from .fly_distillation_student import legal_action_distillation_kl
    config.validate(); train,held=validate_lists(train,held)
    output=Path(output)
    if output.exists(): raise ValueError('Fresh output required; no resume or automatic restart')
    if synthetic_fixture:
        if student.device.type!='cpu' or not getattr(reader,'synthetic_fixture',False): raise ValueError('Explicit CPU artificial student/reader fixture required')
    elif student.device.type!='cuda' or getattr(reader,'synthetic_fixture',False):
        raise ValueError('Production learning requires explicit CUDA and formal reader; no CPU fallback')
    CachedWholeBrainBridge(student,synthetic_fixture=synthetic_fixture)
    if student.brain.recurrent_gain_limit or student.brain.recurrent_gain_logits is not None:
        raise ValueError('This baseline trains I/O only; brain gain must remain0')
    groups=student.trainable_parameter_groups()
    allowed=(*groups['sensory_interface'],*groups['motor_interface'])
    if {id(p) for p in allowed}!={id(p) for p in student.parameters()} or not all(p.requires_grad for p in allowed):
        raise ValueError('Only the original trainable sensory/motor I/O parameters are permitted')
    guard=guard or FitGuard(output,config); guard.check('before_new_output')
    metadata=inspect_episodes(reader,train,held)
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema=SCHEMA,status='running',config=asdict(config),train=train,heldout=held,
        process=process_identity(),
        teacher_SHA256=D95_SHA,teacher_bank_identity=BANK_ID,loss='final19_legal_KL',temperature=1.,lambda_value=0.,gain_limit=0.,
        batch=1,substeps=100,dt_ms=.2,synthetic_fixture=synthetic_fixture,optimizer='fresh Adam',optimizer_instantiated=False,
        optimizer_or_neural_state_serialized=False,numerical_resume=False,automatic_restart=False,native_worlds_started=0,
        scopes={},costs={},optimizer_updates=0,backward_passes=0,backward_attempts=0,optimizer_step_attempts=0,
        dropped_pending_rows=0,parameter_version=0,history_after_update='detached pre-update history; approximate for changed parameters',
        episode_rng_keys={name:episode_key(name,config.rng_seed) for name in train+held},checkpoints=[],unvalidated_after_update=False,
        reader_audit_scope='Reader audit flags describe the reader itself; they are not process-wide claims that this trainer has not used Torch/CUDA',
        metric_contract=dict(Phi_objects='Phi[25:2809].reshape(96,29)',present=0,bullet=1,line_laser=3,curve_laser=4,
            XY=[6,7],near_token_threshold=.15,near_token_is_not_native_distance_or_collision_truth=True,
            player_circle='Phi[13], effective circle component/64',large_circle_proxy_threshold=4/64,
            state4='Phi[24]>.5',observed_deathbomb_timer='Phi[19]',inventory='Phi[4]',legal_mask_source='teacher bank only'))
    if student.device.type=='cuda':
        properties=torch.cuda.get_device_properties(student.device)
        torch.cuda.reset_peak_memory_stats(student.device)
        report['CUDA']=dict(name=properties.name,total_memory_bytes=properties.total_memory,
            peak_reset=True,peak_reset_stage='fit_entry_after_student_to_CUDA; includes current persistent allocation, not earlier transient transfer peak')
    else:report['CUDA']=dict(device='CPU artificial fixture',peak_reset=False,name=None,total_memory_bytes=None)
    total_forwards=0; last_saved=-1; transaction_complete=True; pending=[]; active_bridge=None

    def progress():
        report.update(total_model_decisions=total_forwards,total_forward_passes=total_forwards,resources=guard.report())
        atomic_json(output/'progress.json',report)

    def save_parameters(reason):
        nonlocal last_saved
        path=output/('initial-parameters.pt' if reason=='initial_IO_parameters' else 'last-completed-parameters.pt')
        tmp=output/('.parameters.'+uuid.uuid4().hex+'.tmp')
        try:
            with tmp.open('xb') as stream:
                torch.save(student.parameter_checkpoint_payload(),stream);stream.flush();os.fsync(stream.fileno())
            os.replace(tmp,path)
        finally: tmp.unlink(missing_ok=True)
        last_saved=report['parameter_version']
        record=dict(path=str(path.resolve()),sha256=file_sha(path),parameter_version=last_saved,
            optimizer_updates=report['optimizer_updates'],reason=reason,optimizer_or_neural_state_serialized=False)
        report['checkpoints'].append(record);report['last_completed_model']=record
        if reason=='initial_IO_parameters':report['initial_model']=record
        atomic_json(output/'last-completed-parameters.json',record);progress()

    def phase(names, label, *, training=False, optimizer=None):
        nonlocal total_forwards,pending,active_bridge,transaction_complete
        metrics=Metrics();labels=LabelRows(guard,synthetic_fixture=synthetic_fixture)
        cost=dict(forwards=0,rows=0,updated_labels=0,episode_records=[],frozen_parameters_for_entire_phase=not training)
        report['costs'][label]=cost;report['phase']=label
        row_budget=config.max_rows if training else config.validation_rows
        try:
            for name in names:
                if cost['rows']>=row_budget or training and report['optimizer_updates']>=config.max_updates: break
                guard.check('before_episode_'+label)
                bridge=CachedWholeBrainBridge(student,synthetic_fixture=synthetic_fixture);active_bridge=bridge
                limit=min(row_budget-cost['rows'],metadata[name]['rows'])
                audit=output/'reader-audits'/(label+'-'+name);audit.parent.mkdir(exist_ok=True)
                count=0;terminal=False;pending=[]
                episode_record=dict(episode=name,rows=0,complete_episode=False,scope='continuous_row0_prefix_development_only')
                cost['episode_records'].append(episode_record)
                with closing(reader.iter_episode(name,max_rows=limit,audit_dir=audit)) as stream:
                    for row in stream:
                        guard.check('before_forward_'+label)
                        if total_forwards>=config.max_model_decisions: raise FitStop('combined_model_decision_budget')
                        target,legal=labels.read(row)
                        logits,state=bridge.step(row,episode_rng_key=report['episode_rng_keys'][name] if row.metadata.source_row==0 else None)
                        total_forwards+=1;cost['forwards']+=1;cost['rows']+=1;count+=1;terminal=row.metadata.dones
                        episode_record.update(rows=count,complete_episode=bool(terminal and count==metadata[name]['rows']),
                            scope='complete_episode_teacher_KL' if terminal and count==metadata[name]['rows'] else 'continuous_row0_prefix_development_only')
                        loss=legal_action_distillation_kl(logits,torch.tensor(target,device=student.device)[None],torch.tensor(legal,device=student.device)[None],temperature=1.)
                        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite final19 legal KL')
                        metrics.add(row,target,legal,logits,float(loss.detach()),state,student.brain.output_indices)
                        if training:
                            pending.append(loss)
                            if len(pending)==config.tbptt:
                                guard.check('before_backward')
                                student.zero_grad(set_to_none=True);report['backward_attempts']+=1
                                chunk_loss=torch.stack(pending).mean()
                                chunk_loss.backward()
                                if student.device.type=='cuda':torch.cuda.synchronize(student.device)
                                report['backward_passes']+=1
                                if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in allowed):
                                    raise ValueError('Missing or nonfinite I/O surrogate gradients')
                                bridge.detach_state();guard.check('before_optimizer')
                                transaction_complete=False;report['optimizer_step_attempts']+=1
                                optimizer.step()
                                if student.device.type=='cuda':torch.cuda.synchronize(student.device)
                                if not all(bool(torch.isfinite(parameter).all()) for parameter in allowed):
                                    raise ValueError('Nonfinite parameter after optimizer; preserve preceding atomic checkpoint')
                                transaction_complete=True
                                report['optimizer_updates']+=1;report['parameter_version']+=1
                                report['last_completed_update']=dict(optimizer_updates=report['optimizer_updates'],
                                    parameter_version=report['parameter_version'],rows=len(pending),mean_legal_KL=float(chunk_loss.detach()))
                                cost['updated_labels']+=len(pending);pending=[]
                                student.zero_grad(set_to_none=True)
                                if report['optimizer_updates']%config.checkpoint_every_updates==0:save_parameters('periodic_complete_update')
                                progress()
                                if report['optimizer_updates']>=config.max_updates:break
                report['dropped_pending_rows']+=len(pending);pending=[];bridge.detach_state()
        finally:
            report['dropped_pending_rows']+=len(pending);pending=[]
            if active_bridge is not None:active_bridge.detach_state()
            cost.update(label_files_read=labels.files_read,label_bytes_read=labels.bytes_read)
            report['scopes'][label]=dict(rows=cost['rows'],metrics=metrics.report(),episode_records=cost['episode_records'],
                parameter_version_at_end=report['parameter_version'],own_history_rebuilt_from_row0=True,
                parameters_frozen_for_entire_replay=not training,
                history='approximate detached pre-update history after updates' if training else 'frozen parameters, independent complete consumed own prefix')
            progress()

    try:
        save_parameters('initial_IO_parameters')
        with torch.no_grad():
            phase(train,'frozen_before_train');phase(held,'frozen_before_held')
        optimizer=torch.optim.Adam(allowed,lr=config.learning_rate) if config.max_updates else None
        report['optimizer_instantiated']=optimizer is not None
        report['optimizer_defaults']=None if optimizer is None else dict(optimizer.defaults)
        phase(train,'train',training=True,optimizer=optimizer)
        if last_saved!=report['parameter_version']:save_parameters('training_boundary')
        with torch.no_grad():
            phase(train,'frozen_after_train');phase(held,'frozen_after_held')
        report['status']='completed_finite_cached_distillation'
    except FitStop as error:
        report.update(status='stopped_finite_budget',reason=str(error),unvalidated_after_update=report['optimizer_updates']>0)
        student.zero_grad(set_to_none=True)
        if transaction_complete and last_saved!=report['parameter_version']:save_parameters('finite_stop_last_completed_update')
    except BaseException as error:
        report.update(status='failed',reason=str(error),traceback=traceback.format_exc(),
            optimizer_transaction_complete=transaction_complete,unvalidated_after_update=report['optimizer_updates']>0,
            last_atomic_checkpoint_preserved=True,failed_GPU_transaction_not_checkpointed=not transaction_complete)
    finally:
        if student.device.type=='cuda':
            try:report['CUDA'].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(student.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(student.device))
            except Exception as error:report['CUDA']['peak_collection_error']=str(error)
        progress();atomic_json(output/'result.json',report)
    return report


def memory_plan(tbptt):
    return dict(batch=1,tbptt=tbptt,source_graph_and_transpose_bytes=603679320,IO_parameters_bytes=39920036,
        IO_gradients_bytes=39920036,Adam_moments_bytes=79840072,state_bytes_per_episode=8323552,
        allocated_GiB_unmeasured_range=[1.,1.8] if tbptt==1 else [1.8,3.5],
        reserved_headroom='Add about30%+0.25GiB for reserved workspace/fragmentation; not a measured upper bound',
        basis='Existing full-graph single-100-microstep backward receipt allocated960782848B, without Adam; TBPTT1/4 trainer not benchmarked here',
        CPU_assembly_GiB_unmeasured_range=[2.,6.],no_automatic_quota_or_substep_change=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root',type=Path,required=True);parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--train',nargs='+',required=True);parser.add_argument('--held',nargs='*',default=[])
    parser.add_argument('--split',type=Path,default=ROOT/'artifacts/engineering/B1-autonomous-dodge-20261024/fly-migration-20261026/frozen-episode-split-v1.json')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute-cuda',action='store_true')
    parser.add_argument('--student-checkpoint',type=Path);parser.add_argument('--connectome',type=Path,default=ROOT/'data/connectome')
    for flag,kind,default in (('max-rows',int,32),('validation-rows',int,32),('max-updates',int,1),('max-model-decisions',int,None),('max-seconds',float,300.),
                             ('tbptt',int,1),('learning-rate',float,1e-4),('checkpoint-every-updates',int,10),('gpu-fraction',float,.2),
                             ('minimum-ram-GiB',float,10.),('minimum-disk-GiB',float,4.),('rng-seed',int,613711)):
        parser.add_argument('--'+flag,type=kind,default=default)
    args=parser.parse_args(argv)
    if args.output.exists(): raise ValueError('New output directory required; never resume')
    if args.max_model_decisions is None: args.max_model_decisions=args.max_rows+4*args.validation_rows
    config=CachedFitConfig(**{name:getattr(args,name) for name in CachedFitConfig.__dataclass_fields__}).validate()
    guard=FitGuard(args.output,config)
    try:
        if 'torch' in sys.modules: raise ValueError('Fresh CLI must import CPU reader before Torch')
        module,CVD=import_reader_preserving_CVD()
        reader=module.FullEpisodeSensoryReader(args.cache_root,args.bank,guard)
        inspect=inspect_episodes(reader,args.train,args.held)
        split=json.loads(args.split.read_text(encoding='utf-8'))
        if (split.get('bank_identity_sha256')!=BANK_ID or not set(args.train)<=set(split['train'])
                or not set(args.held)<=set(split['heldout'])):raise ValueError('Requested episodes must stay within the frozen D95 train/development split')
        provenance=dict(schema=SCHEMA,CVD_import_sequence=CVD,split_path=str(args.split.resolve()),split_SHA256=file_sha(args.split),
            source_SHA256=file_sha(__file__),inspection=inspect,memory_plan=memory_plan(config.tbptt),Torch_imported=False,
            teacher_SHA256=D95_SHA,train=args.train,heldout=args.held,config=asdict(config),process=process_identity(),
            reader_audit_scope='Only reader side effects; not a process-wide Torch/CUDA assertion for subsequent training')
        if not args.execute_cuda:
            args.output.mkdir(parents=True,exist_ok=False)
            atomic_json(args.output/'inspect.json',{**provenance,'status':'NoTorch_inspect_only','GPU_or_optimizer_started':False})
            return 0
        if min(config.max_rows,config.validation_rows)<32:
            raise ValueError('Formal reader consumes whole packets; use train/validation row caps>=32 for production prefix pilots')
        guard.check('before_Torch')
        import torch
        provenance['Torch_imported']=True;provenance['CUDA_initialized_before_setup']=torch.cuda.is_initialized()
        if provenance['CUDA_initialized_before_setup']:raise ValueError('Fresh CUDA context required')
        torch.set_num_threads(2);torch.manual_seed(config.rng_seed)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if torch.is_autocast_enabled() or torch.is_autocast_enabled('cpu'):raise ValueError('Strict FP32 without autocast required')
        torch.cuda.set_per_process_memory_fraction(config.gpu_fraction,0)
        from .fly_distillation_student import FlyDistillationStudent
        student=(FlyDistillationStudent.from_parameter_checkpoint(args.student_checkpoint,args.connectome,device='cpu')
                 if args.student_checkpoint else FlyDistillationStudent.from_full_connectome(args.connectome,substeps=100,device='cpu',rng_seed=config.rng_seed))
        guard.check('before_student_CUDA');student.to('cuda:0')
        report=fit_cached_student(student,reader,args.train,args.held,args.output,config,guard=guard)
        provenance['CUDA']=report['CUDA']
        atomic_json(args.output/'provenance.json',provenance)
        return 0 if report['status']=='completed_finite_cached_distillation' else 2
    except BaseException as error:
        if not args.output.exists(): args.output.mkdir(parents=True,exist_ok=False)
        atomic_json(args.output/'failure.json',dict(schema=SCHEMA,status='failed_preparation',reason=str(error),traceback=traceback.format_exc(),
                    automatic_restart=False,native_worlds_started=0,Torch_imported='torch' in sys.modules))
        return 1


if __name__=='__main__':
    raise SystemExit(main())
