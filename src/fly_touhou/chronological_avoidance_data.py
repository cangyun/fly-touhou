"""Chronological raw teacher/own-state streams for original GRU guidance.

Only offline supervision is constructed here. Real student episodes are not
cut at a stage boundary and no recorded action becomes a PPO sample. Within
each stream, neural history continues rather than restarting every16 rows.
"""
from __future__ import annotations
from collections import deque
import copy
import json
from pathlib import Path

import numpy as np

from .no_bomb_teacher_data import NoBombTeacherDataset
from .thrl_inputs import SEPARATED_LASER_INPUT, observation_shapes
from .thrl_study_io import read_shared_json as read


class TeacherRows:
    def __init__(self, dataset):
        self.dataset = dataset
        self.identity = dataset.manifest_sha256
        self.lengths = {i: int(record['reader'].index['original_stage_row_ranges']['1']['stop'])
                        for i, record in enumerate(dataset.records)}
        if any(n < 1 for n in self.lengths.values()):
            raise ValueError('Actual first-stage demonstration prefix required')

    def fetch(self, record, first, stop):
        raw = self.dataset.records[record]['reader'].fetch(first, stop)
        if (np.any(raw['stage'] != 1) or np.any(raw['actions'] >= 18)
                or np.any(raw['native_bombs_used'] != 0)
                or not np.array_equal(raw['episode_starts'], np.arange(first, stop) == 0)
                or not np.array_equal(raw['before_frames'][1:], raw['after_observation_frames'][:-1])):
            raise ValueError('Teacher prefix or before-action chronology differs')
        result = {key: raw[key] for key in ('features','maps','objects','actions','episode_starts','outcomes')}
        result.update(rule_probabilities=raw['targets'], rule_weights=raw['teacher_guidance_weight'])
        return result


class OwnTrainingRows:
    """First complete fixed-policy training rollout; no development tails."""
    def __init__(self, path):
        from .thrl_dodge_diagnostics import _shared_binary
        import hashlib
        self.path = Path(path)
        if self.path.name != 'rollout-000.npz':
            raise ValueError('The actual first rollout has genuine zero-history openings')
        self.archive = self.path.with_name(self.path.stem+'-raw-maps.zip')
        index = read(self.archive.with_suffix('.zip.index.json'))
        self.lookup = {(r['time_start'],r['world']):r for r in index['members']}
        with _shared_binary(self.path) as stream:
            with np.load(stream, allow_pickle=False) as data:
                self.data = {key:data[key] for key in ('features','objects','actions','episode_starts',
                    'rule_probabilities','rule_weights','task_outcomes')}
        time, worlds = self.data['actions'].shape
        if worlds != 30 or not self.data['episode_starts'][0].all() or (self.data['actions'] >= 18).any():
            raise ValueError('Actual30-world fresh-opening NoBomb training source required')
        self.lengths = dict.fromkeys(range(worlds),time)
        self.identity = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def fetch(self, record, first, stop):
        from .continuous_full_sensory_ppo import read_raw_map_member
        blocks=[]
        for begin in range(first//16*16,(stop+15)//16*16,16):
            raw=read_raw_map_member(self.archive,self.lookup[(begin,record)])
            blocks.append(raw[max(first-begin,0):min(stop-begin,16)])
        result={key:values[first:stop,record].copy() for key,values in self.data.items()
                if key != 'task_outcomes'}
        result.update(maps=np.concatenate(blocks),outcomes=self.data['task_outcomes'][first:stop,record].copy())
        return result


class FixedStudentTrainingRows:
    """Consecutive student-controlled DAgger rollouts under one frozen policy.

Only the acquisition schema is accepted. PPO rollouts can change the body
between files and must never be concatenated as one neural history.
    """
    def __init__(self, root):
        import hashlib
        from .thrl_dodge_diagnostics import _shared_binary
        self.root=Path(root)
        launch=read(self.root/'launch.json');result=read(self.root/'result.json')
        if (launch.get('schema')!='current-fullbrain-student-controlled-DAgger-acquisition-v1'
                or result.get('status')!='completed' or result.get('optimizer_updates')!=0
                or result.get('teacher_actions')!=0 or result.get('PPO_samples') is not False
                or result.get('own_world_history_carried_across_rollouts') is not True
                or launch['parameters_sha256']!=result['source_parameters_sha256']):
            raise ValueError('Completed fixed-policy student-controlled acquisition required')
        self.parts=[];self.offsets=[0];digest=hashlib.sha256()
        digest.update(launch['parameters_sha256'].encode())
        fields=('features','objects','actions','episode_starts','rule_probabilities','rule_weights',
                'task_outcomes','task_terminated','game_ids','native_episodes','episode_steps')
        for number in range(result['rollouts']):
            path=self.root/'traces'/f'rollout-{number:03}.npz'
            archive=path.with_name(path.stem+'-raw-maps.zip')
            index=read(archive.with_suffix('.zip.index.json'))
            with _shared_binary(path) as stream:
                digest.update(hashlib.file_digest(stream,'sha256').digest());stream.seek(0)
                with np.load(stream,allow_pickle=False) as arrays:data={key:arrays[key] for key in fields}
            steps,worlds=data['actions'].shape
            if not 0<steps<=256 or worlds!=30 or (data['actions']>=18).any():
                raise ValueError('Only actual30-world NoBomb acquisition rows are eligible')
            if not self.parts:
                if not data['episode_starts'][0].all():raise ValueError('Actual opening history required')
            else:
                previous=self.parts[-1]['data'];starts=data['episode_starts'][0]
                if not np.array_equal(starts,previous['task_terminated'][-1]):
                    raise ValueError('File boundary cannot invent or drop a game reset')
                for key in ('game_ids','native_episodes'):
                    if not np.array_equal(data[key][0,~starts],previous[key][-1,~starts]):
                        raise ValueError('Continuing games changed at a rollout boundary')
                if not np.array_equal(data['episode_steps'][0,~starts],previous['episode_steps'][-1,~starts]+1):
                    raise ValueError('Continuing game decision history has a gap')
            self.parts.append(dict(data=data,archive=archive,
                lookup={(m['time_start'],m['world']):m for m in index['members']}))
            self.offsets.append(self.offsets[-1]+steps)
        self.data={key:np.concatenate([part['data'][key] for part in self.parts]) for key in fields}
        if self.data['actions'].size!=result['student_actions']:
            raise ValueError('Acquired decision count differs from archived actual rows')
        self.lengths=dict.fromkeys(range(30),self.offsets[-1]);self.identity=digest.hexdigest()
        self.source_parameters_sha256=launch['parameters_sha256']
        self.actual_student_actions=result['student_actions']

    def fetch(self,record,first,stop):
        from .continuous_full_sensory_ppo import read_raw_map_member
        if not 0<=first<stop<=self.lengths[record]:raise ValueError('Actual recorded row interval required')
        blocks=[]
        for part,offset,end in zip(self.parts,self.offsets,self.offsets[1:]):
            start=max(first,offset)-offset;finish=min(stop,end)-offset
            if start>=finish:continue
            for begin in range(start//16*16,(finish+15)//16*16,16):
                raw=read_raw_map_member(part['archive'],part['lookup'][(begin,record)])
                blocks.append(raw[max(start-begin,0):min(finish-begin,16)])
        keys=('features','objects','actions','episode_starts','rule_probabilities','rule_weights')
        result={key:self.data[key][first:stop,record].copy() for key in keys}
        result.update(maps=np.concatenate(blocks),outcomes=self.data['task_outcomes'][first:stop,record].copy())
        return result


class ChronologicalBatches:
    def __init__(self, source, records, *, width=16, length=16, seed=1, shuffle=True):
        if not records or len(set(records)) != len(records) or min(width,length) < 1:
            raise ValueError('Distinct raw streams and positive dimensions required')
        self.source, self.records = source, list(records)
        self.width,self.length,self.shuffle = width,length,shuffle
        self.rng=np.random.default_rng(seed)
        self.queue=deque()
        self.slots=[None]*width
        self.epochs_started=self.epochs_completed=0
        self.label_cache={}

    def _labels(self, record, first, stop):
        from .continuous_guided_dagger import direction_labels
        pieces={key:[] for key in ('margins','trusted','unprotected_mask')}
        for begin in range(first//32*32,(stop+31)//32*32,32):
            key=(record,begin)
            if key not in self.label_cache:
                data=self.source.fetch(record,begin,min(begin+32,self.source.lengths[record]))
                direction_labels(data)  # Original4/12/24 geometry and native-fact veto, all rows.
                self.label_cache[key]={name:data[name] for name in pieces}
            for name in pieces:
                pieces[name].append(self.label_cache[key][name][max(first-begin,0):min(stop-begin,32)])
        return {key:np.concatenate(values) for key,values in pieces.items()}

    def next(self):
        if not self.queue and all(slot is None for slot in self.slots):
            order=self.rng.permutation(self.records).tolist() if self.shuffle else list(self.records)
            self.queue.extend(order)
            self.epochs_started+=1
        for lane in range(self.width):
            if self.slots[lane] is None and self.queue:
                self.slots[lane]=dict(record=self.queue.popleft(),position=0)
        shape=(self.length,self.width)
        data={key:np.zeros((*shape,*tail),np.float32)
              for key,tail in observation_shapes(SEPARATED_LASER_INPUT).items()}
        data.update(actions=np.zeros(shape,np.int64),episode_starts=np.ones(shape,np.bool_),
                    valid_mask=np.zeros(shape,np.bool_),legal_actions=np.ones((*shape,19),np.bool_))
        data['legal_actions'][...,18]=False
        task=dict(margins=np.zeros((*shape,3,18),np.float32),trusted=np.zeros((*shape,3),np.bool_),
                  unprotected_mask=np.zeros(shape,np.bool_),outcomes=np.full(shape,-1,np.int8),
                  rule_probabilities=np.zeros((*shape,19),np.float32),rule_weights=np.zeros(shape,np.float32),
                  deathbomb_mask=np.zeros(shape,np.bool_))
        sources=[]
        for lane,slot in enumerate(self.slots):
            if slot is None:continue
            record,first=slot['record'],slot['position']
            stop=min(first+self.length,self.source.lengths[record]);n=stop-first
            raw=self.source.fetch(record,first,stop)
            labels=self._labels(record,first,stop)
            for key in ('features','maps','objects','actions','episode_starts'):
                data[key][:n,lane]=raw[key]
            for key in task:
                if key=='deathbomb_mask':continue
                task[key][:n,lane]=(labels if key in labels else raw)[key]
            data['valid_mask'][:n,lane]=True
            sources.append(dict(lane=lane,record=record,first=first,stop=stop,epoch=self.epochs_started))
            slot['position']=stop
            if stop==self.source.lengths[record]:self.slots[lane]=None
        if not self.queue and all(slot is None for slot in self.slots):
            self.epochs_completed=self.epochs_started
        if not data['valid_mask'].any():raise RuntimeError('No artificial all-padding learning step')
        return data,task,dict(sources=sources,real_rows=int(data['valid_mask'].sum()),
            epochs_started=self.epochs_started,epochs_completed=self.epochs_completed)

    def state_dict(self):
        return dict(source_identity=self.source.identity,records=self.records,width=self.width,length=self.length,
                    shuffle=self.shuffle,queue=list(self.queue),slots=copy.deepcopy(self.slots),
                    epochs_started=self.epochs_started,epochs_completed=self.epochs_completed,
                    numpy_RNG=copy.deepcopy(self.rng.bit_generator.state))

    def load_state_dict(self,state):
        if any(state[key]!=getattr(self,key) for key in ('records','width','length','shuffle')) or state['source_identity']!=self.source.identity:
            raise ValueError('A chronological checkpoint cannot switch raw sources or streams')
        self.queue=deque(state['queue']);self.slots=copy.deepcopy(state['slots'])
        self.epochs_started=state['epochs_started'];self.epochs_completed=state['epochs_completed']
        self.rng.bit_generator.state=copy.deepcopy(state['numpy_RNG'])


def original_guidance_block(model,data,task,before,*,checkpoint=True):
    from .continuous_full_sensory_ppo import _forward_full_sequence
    from .continuous_object_avoidance_ppo import avoidance_PPO_config
    from .continuous_object_avoidance import GUIDANCE_CONFIG
    from .fly_gru_aligned_guidance import gru_aligned_guidance_loss
    from .continuous_signed_clearance import signed_clearance4f_terms
    result=_forward_full_sequence(model,data,before,cfg=avoidance_PPO_config(),
                                  use_checkpoint=checkpoint,counts={})
    guide=gru_aligned_guidance_loss(result['composed_logits'],**task,actions=data['actions'],
        valid_mask=data['valid_mask'],config=GUIDANCE_CONFIG)
    risk=signed_clearance4f_terms(result['predicted_margins'],task,data['features'],data['valid_mask'])
    return result,guide,risk,guide['total_loss']+risk['weighted_loss']


def alternating_original_guidance_block(model,optimizer,data,task,before,*,risk_substeps,
                                       risk_learning_rate,checkpoint=True):
    """Fit risk on this exact DN, then differentiate the original joint loss.

No input/core/actor weights are changed before their pending backward. The
old risk-head graph is discarded and re-evaluated with the fitted weights;
the same DN remains connected to the trainable sensory and recurrent graph.
There are no old-policy latent replays or teacher/runtime shortcuts.
    """
    import torch
    from .continuous_full_sensory_ppo import _forward_full_sequence
    from .continuous_object_avoidance_ppo import avoidance_PPO_config
    from .continuous_object_avoidance import GUIDANCE_CONFIG,no_bomb_policy_logits
    from .fly_gru_aligned_guidance import gru_aligned_guidance_loss
    from .continuous_signed_clearance import signed_clearance4f_terms
    from .whole_brain_ppo import clip_gradients_fp64
    if type(risk_substeps) is not int or risk_substeps<1 or risk_learning_rate<=0:
        raise ValueError('Positive explicit risk-readout substeps/rate required')
    result=_forward_full_sequence(model,data,before,cfg=avoidance_PPO_config(),
        use_checkpoint=checkpoint,counts={},return_DN=True)
    head_parameters=list(model.risk_head.parameters());head_ids={id(p) for p in head_parameters}
    groups=[group for group in optimizer.param_groups if group.get('name')=='risk']
    if len(groups)!=1 or {id(p) for p in groups[0]['params']}!=head_ids:
        raise ValueError('Existing named risk Adam coordinates must be retained')
    detached_DN=result['following_DN'].detach();group=groups[0];original_lr=group['lr']
    measurements=[]
    try:
        group['lr']=risk_learning_rate
        for _ in range(risk_substeps):
            optimizer.zero_grad(set_to_none=True)
            prediction=1.-.5*model.risk_head.layers(detached_DN)
            terms=signed_clearance4f_terms(prediction,task,data['features'],data['valid_mask'])
            if not int(terms['count']):break  # No fabricated risk update on unlabelled rows.
            if not bool(torch.isfinite(terms['loss'])):raise FloatingPointError('Nonfinite original signed-risk loss')
            terms['loss'].backward()
            if any(p.grad is not None for p in model.parameters() if id(p) not in head_ids):
                raise RuntimeError('A risk-only substep reached outside the DN readout')
            norm=clip_gradients_fp64(head_parameters,5.)
            optimizer.step()
            measurements.append(dict(loss=float(terms['loss'].detach()),gradient_norm=norm,
                valid_queries=int(terms['count']),learning_rate=risk_learning_rate))
    finally:
        group['lr']=original_lr
    optimizer.zero_grad(set_to_none=True)
    score=model.risk_head.layers(result['following_DN'])
    combined=model.compose(result['raw_logits'],torch.nn.functional.softplus(score),data['features'],check_finite=False)
    result.update(predicted_margins=1.-.5*score,
        composed_logits=combined.masked_fill(~data['legal_actions'],-torch.inf),
        actual_logits=no_bomb_policy_logits(combined,data['features']))
    result['log_probs']=result['actual_logits'].log_softmax(-1).gather(-1,data['actions'][...,None]).squeeze(-1)
    guide=gru_aligned_guidance_loss(result['composed_logits'],**task,actions=data['actions'],
        valid_mask=data['valid_mask'],config=GUIDANCE_CONFIG)
    risk=signed_clearance4f_terms(result['predicted_margins'],task,data['features'],data['valid_mask'])
    return result,guide,risk,guide['total_loss']+risk['weighted_loss'],measurements


def recorded_state_metrics(result,data,task):
    """Same-forward training proxies, explicitly distinct from gameplay scores."""
    import torch
    with torch.no_grad():
        f=data['features'];target=task['margins'][...,0,:]
        good=(data['valid_mask']&task['trusted'][...,0]&task['unprotected_mask']&
              (f[...,21]>.5)&(f[...,20]<=0)&(f[...,5]<=0))
        bad=(target<=0)&good[...,None];safe=(target>1)&good[...,None]
        mixed=bad.any(-1)&safe.any(-1)
        prediction=result['predicted_margins']
        logits=result['actual_logits'][...,:18]
        action=logits.argmax(-1)
        chosen_bad=bad.gather(-1,action[...,None]).squeeze(-1)
        queries=int(bad.sum());scenes=int(mixed.sum())
        return dict(proxy_danger_queries=queries,proxy_mixed_scenes=scenes,
            proxy_danger_recall=float((bad&(prediction<=0)).sum())/queries if queries else None,
            proxy_false_safe_fraction=float((bad&(prediction>1)).sum())/queries if queries else None,
            proxy_final_mode_unsafe_count=int((mixed&chosen_bad).sum()),
            proxy_final_mode_unsafe_fraction=float((mixed&chosen_bad).sum())/scenes if scenes else None,
            proxy_policy_unsafe_mass=float((logits.softmax(-1)*bad).sum(-1)[mixed].mean()) if scenes else None)
