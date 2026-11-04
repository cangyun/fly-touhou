"""Read-only first-eight pre-update diagnostics; no learner imports or actions."""
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .atomic_json_io import atomic_write_json

MAX_ROLLOUTS = 8
ARRAY_FIELDS = ('observations','actions','rewards','episode_starts','values',
                'log_probs','advantages','returns','hidden_states_pi',
                'cell_states_pi','hidden_states_vf','cell_states_vf')


class TraceReadOnlyViolation(RuntimeError):
    pass


def copy_array(value):
    if not isinstance(value,np.ndarray):
        value=value.detach().cpu().numpy()
    result=np.array(value,copy=True)
    if result.dtype.hasobject:
        raise ValueError('Diagnostic arrays must not require pickle')
    return result


def fingerprint(value):
    sha=hashlib.sha256()
    def visit(item):
        if isinstance(item,Mapping):
            sha.update(b'dict')
            for key in sorted(item,key=repr):
                visit(key);visit(item[key])
        elif isinstance(item,(tuple,list)):
            sha.update(type(item).__name__.encode())
            for child in item: visit(child)
        elif isinstance(item,np.ndarray) or hasattr(item,'detach'):
            array=copy_array(item)
            sha.update(str((array.dtype.str,array.shape)).encode());sha.update(array.tobytes())
        elif item is None or isinstance(item,(str,int,float,bool,bytes,np.generic)):
            sha.update((type(item).__name__+repr(item)).encode())
        else:
            raise TypeError('Unsupported diagnostic state: '+type(item).__name__)
    visit(value)
    return sha.hexdigest()


def model_signature(model,capture_rng):
    buffer=model.rollout_buffer
    return fingerprint(( {key:getattr(buffer,key) for key in ARRAY_FIELDS},
        (buffer.full,buffer.pos,getattr(buffer,'generator_ready',False)),
        model._last_obs,model._last_episode_starts,getattr(model,'_last_lstm_states',None),
        model.policy.state_dict(),model.policy.optimizer.state_dict(),capture_rng()))


class RolloutTraceObserver:
    def __init__(self,output,save_checkpoint,capture_rng,optimizer_steps,*,steps=128,envs=2):
        self.output=Path(output)/'pre_update_diagnostic'
        self.save_checkpoint,self.capture_rng,self.optimizer_steps=save_checkpoint,capture_rng,optimizer_steps
        self.steps,self.envs=steps,envs
        self.completed=0
        self.disabled=False
        self.failure=None
        self.initial_decisions=None

    def summary(self):
        return {'completed_archives':self.completed,'hard_cap':MAX_ROLLOUTS,
                'disabled':self.disabled,'failure':self.failure,
                'phase':'pre_update_diagnostic','regular_training_resume':False}

    def capture(self,model):
        if self.disabled or self.completed>=MAX_ROLLOUTS:
            return None
        buffer=model.rollout_buffer
        if not buffer.full or buffer.pos!=buffer.buffer_size:
            return None  # A stopped/partial collection is not a completed rollout.
        if (buffer.buffer_size!=self.steps or buffer.n_envs!=self.envs
                or getattr(buffer,'generator_ready',False)):
            raise TraceReadOnlyViolation('Expected a complete, unflattened pre-train rollout')
        before=model_signature(model,self.capture_rng)
        arrays={key:copy_array(getattr(buffer,key)) for key in ARRAY_FIELDS}
        arrays['last_observation']=copy_array(model._last_obs)
        arrays['last_episode_starts']=copy_array(model._last_episode_starts)
        states=getattr(model,'_last_lstm_states',None)
        if states is not None:
            for branch in ('pi','vf'):
                for label,value in zip(('hidden','cell'),getattr(states,branch)):
                    arrays['last_'+branch+'_'+label]=copy_array(value)
        if any(arrays[key].shape[:2]!=(self.steps,self.envs) for key in ARRAY_FIELDS[:8]):
            raise TraceReadOnlyViolation('Rollout leading dimensions changed')
        if sum(value.nbytes for value in arrays.values())>8*1024*1024:
            raise TraceReadOnlyViolation('Unexpected trace layout exceeds 8MiB array bound')
        index=self.completed+1
        directory=self.output/f'rollout_{index:02d}_sampled_{int(model.num_timesteps):07d}'
        metadata={'schema':'PPO-pre-update-diagnostic-v1','phase':'pre_update_diagnostic',
            'regular_training_resume':False,'current_rollout_updated':False,
            'sampled_total_decisions':int(model.num_timesteps),
            'rollout_decisions_pending_update':self.steps*self.envs,
            'previous_committed_updates':self.optimizer_steps(model),'previous_committed_updates_unit':'Adam minibatches',
            'initial_training_decisions':self.initial_decisions,
            'complete_rollout_index':index,'hard_cap':MAX_ROLLOUTS,
            'buffer_full':True,'buffer_pos':int(buffer.pos),'generator_ready':False,
            'arrays':{key:{'shape':list(value.shape),'dtype':value.dtype.str,'bytes':value.nbytes} for key,value in arrays.items()},
            'read_only_signature_before':before,'checkpoint_receipt_unmodified':True,
            'warning':'This rollout is sampled but not trained. Never use this diagnostic checkpoint for ordinary resume.'}
        failure=None
        try:
            directory.mkdir(parents=True,exist_ok=False)
            with (directory/'rollout_arrays.npz').open('xb') as stream:
                np.savez(stream,**arrays)
            self.save_checkpoint(model,directory/'pre_update')
            receipt=json.loads((directory/'pre_update.receipt.json').read_text(encoding='utf-8'))
            if receipt['decisions']!=int(model.num_timesteps) or receipt['optimizer_steps']!=metadata['previous_committed_updates']:
                raise TraceReadOnlyViolation('Pre-update checkpoint counters differ from observer')
            metadata['checkpoint_receipt_sha256']=hashlib.sha256((directory/'pre_update.receipt.json').read_bytes()).hexdigest()
            metadata['archive_sha256']=hashlib.sha256((directory/'rollout_arrays.npz').read_bytes()).hexdigest()
            if sum(path.stat().st_size for path in directory.iterdir() if path.is_file())>32*1024*1024:
                raise OSError('Diagnostic block exceeds fixed 32MiB storage limit')
        except TraceReadOnlyViolation:
            raise
        except Exception as error:
            failure=error
        finally:
            after=model_signature(model,self.capture_rng)
            if before!=after:
                raise TraceReadOnlyViolation('Observer changed buffer/model/optimizer/learner RNG')
        try:
            if failure is not None:
                raise failure
            metadata.update(status='complete',read_only_signature_after=after)
            atomic_write_json(directory/'diagnostic.json',metadata)
            self.completed=index
        except Exception as error:
            self.disabled=True
            self.failure=repr(error)
            try:
                atomic_write_json(self.output/'trace_status.json',dict(self.summary(),status='failed_disabled',partial_directory=str(directory)))
            except Exception:
                print(json.dumps({'pre_update_diagnostic':'failed_disabled','reason':self.failure}),flush=True)
        return metadata if not self.disabled else None
