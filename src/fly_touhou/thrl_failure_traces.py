"""Bounded copies of actual diagnostic inputs/actions; never select an action."""
from collections import deque
from pathlib import Path
import hashlib
import numpy as np
from .atomic_json_io import atomic_write_json as write

MODES=('trained_sampled','trained_greedy')


class FailureTraces:
    modes=MODES
    action_count=18
    def __init__(self,output,*,history=128,capture_history_maps=False):
        if type(history) is not int or not 1<=history<=128:raise ValueError('Bounded trace history required')
        if type(capture_history_maps) is not bool:raise ValueError('Explicit historical map capture flag required')
        self.output=Path(output);self.history=history;self.slots={}
        self.capture_history_maps=capture_history_maps

    def reset(self,index):self.slots.pop(index,None)

    def before_action(self,index,job,observation,probabilities,hidden,info,action):
        if job['policy'] not in self.modes:return
        if not 0<=action<self.action_count:raise ValueError('Trace action is outside its declared protocol')
        p=np.asarray(probabilities,dtype=np.float32)
        if p.shape!=(self.action_count,) or not np.isfinite(p).all() or (p<0).any() or abs(float(p.sum())-1)>2e-5:
            raise ValueError('Trace needs the actual movement distribution')
        key=(job['policy'],job['game'])
        if index not in self.slots:self.slots[index]=dict(key=key,count=0,rows=deque(maxlen=self.history),actions=bytearray())
        slot=self.slots[index]
        if slot['key']!=key:raise ValueError('Trace slot reused without resetting its episode')
        slot['count']+=1
        if any(np.asarray(observation[k]).dtype!=np.float32 for k in ('features','objects','maps')):
            raise ValueError('Diagnostic capture must preserve original FP32 observations')
        slot['actions'].append(int(action))
        def clock(key):return int(info[key]) if info.get(key) is not None else -1
        row={k:np.asarray(observation[k]).copy() for k in ('features','objects')}
        if self.capture_history_maps:row['maps']=np.asarray(observation['maps']).copy()
        row.update(probabilities=p.copy(),hidden=np.asarray(hidden,dtype=np.float32).copy(),
            action=int(action),native_frame=clock('observation_native_frame'),
            episode_frame=clock('episode_frames'),native_episode=clock('native_episode'),
            stage=clock('stage'),stage_frame=clock('stage_frame'))
        slot['rows'].append(row);slot['last_maps']=np.asarray(observation['maps']).copy()

    def finish(self,index,job,after,info,checkpoint_sha256):
        if job['policy'] not in self.modes:return None
        slot=self.slots.pop(index);endpoint=info['probe_episode'];rows=list(slot['rows'])
        if slot['key']!=(job['policy'],job['game']) or slot['count']!=endpoint['policy_decisions']:
            raise ValueError('Captured actions do not match the acknowledged native endpoint')
        if len(slot['actions'])!=slot['count']:raise ValueError('Whole-episode action tape is incomplete')
        folder=self.output/'failure_traces';folder.mkdir(parents=True,exist_ok=True)
        path=folder/f"{job['policy']}_game{job['game']:03d}.npz"
        arrays={key:np.stack([r[key] for r in rows]) for key in rows[0]}
        arrays.update(last_preaction_maps=slot['last_maps'],
            **{'after_'+k:np.asarray(after[k]).copy() for k in ('features','objects','maps')})
        with path.open('xb') as stream:np.savez_compressed(stream,**arrays)
        action_path=path.with_suffix('.actions.bin');action_bytes=bytes(slot['actions'])
        with action_path.open('xb') as stream:stream.write(action_bytes)
        tape=dict(schema='B1-original-source18-four-tick-action-tape-v1',path=str(action_path),
            sha256=hashlib.sha256(action_bytes).hexdigest(),decisions=len(action_bytes),
            encoding='one uint8 source18 action per acknowledged policy decision',
            replay_in_original_game_verified=False)
        receipt=dict(schema=('B1-native-diagnostic-input-action-tail-v2-full-map-history'
            if self.capture_history_maps else 'B1-native-diagnostic-input-action-tail-v1'),policy=job['policy'],game=job['game'],
            checkpoint_sha256=checkpoint_sha256,history_rows=len(rows),whole_episode_decisions=slot['count'],
            endpoint=endpoint,after_native_frame=info.get('observation_native_frame'),
            after_native_episode=info.get('native_episode'),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),arrays_file=str(path),
            stored_map_precision=('original observation FP32; every retained preaction map and endpoint'
                if self.capture_history_maps else 'original observation FP32; only final preaction and endpoint maps'),
            full_episode_action_tape=tape,
            teacher_queries=0,actions_changed=False,
            earlier_history_full_map_coverage_known=self.capture_history_maps,unexecuted_native_counterfactuals_measured=False)
        write(path.with_suffix('.json'),receipt)
        return dict(receipt=str(path.with_suffix('.json')),arrays=str(path),sha256=receipt['sha256'],full_episode_action_tape=tape)
