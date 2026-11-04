"""Actual full19 collision tails across all stages, without controlling actions.

Keep one rolling128-decision input/map/memory buffer per live game. Save the
first acknowledged movement collision per stage and one natural terminal tail.
This bounds saved events to seven per complete six-stage game.
"""
from pathlib import Path
import hashlib
import os
import re

import numpy as np

from .thrl_failure_traces import FailureTraces
from .thrl_durable_records import write_json as write

FACT_FIELDS = ('native_episode','observation_native_frame','observation_source',
    'episode_frames','stage','stage_frame','misses','bombs_used','policy_action',
    'elapsed_input_ticks')


class FullStoryFailureTraces(FailureTraces):
    modes=('full_story',)
    action_count=19

    def __init__(self,output,*,checkpoint_sha256,history=128,
                 memory_kind='model_recurrent_hidden'):
        if not isinstance(checkpoint_sha256,str) or not re.fullmatch('[0-9a-f]{64}',checkpoint_sha256):
            raise ValueError('Explicit frozen full-game model hash required')
        super().__init__(output,history=history,capture_history_maps=True)
        self.checkpoint_sha256=checkpoint_sha256
        self.memory_kind=memory_kind

    def before_action(self,index,job,observation,probabilities,hidden,info,action):
        super().before_action(index,dict(policy='full_story',game=job['game']),
            observation,probabilities,hidden,info,action)
        slot=self.slots[index]
        slot['last_before_facts']={k:info.get(k) for k in FACT_FIELDS}
        slot.setdefault('collision_stages',set())
        slot.setdefault('saved_events',[])

    def finish(self,*args,**kwargs):
        raise ValueError('Full19 natural-game traces use finish_game, not the no-Bomb tape protocol')

    def _save(self,index,job,after,info,*,event,collision_observed,terminal):
        slot=self.slots[index]
        if slot['count']!=job['decisions'] or len(slot['actions'])!=job['decisions']:
            raise ValueError('Full-game trace must match acknowledged actions')
        rows=list(slot['rows'])
        if not rows:raise ValueError('A saved full-game event needs actual preaction inputs')
        folder=self.output/'full_game_traces';folder.mkdir(parents=True,exist_ok=True)
        path=folder/f"game{job['game']:03d}_{event}.npz"
        arrays={key:np.stack([row[key] for row in rows]) for key in rows[0]}
        arrays.update(last_preaction_maps=slot['last_maps'],
            **{'after_'+key:np.asarray(after[key]).copy() for key in ('features','objects','maps')})
        with path.open('xb') as stream:
            np.savez_compressed(stream,**arrays)
            stream.flush();os.fsync(stream.fileno())
        tape_path=path.with_suffix('.actions.bin');tape=bytes(slot['actions'])
        if 'action_hash' in job and hashlib.sha256(tape).hexdigest()!=job['action_hash'].hexdigest():
            raise ValueError('Captured full19 action prefix differs from the dispatched game actions')
        with tape_path.open('xb') as stream:
            stream.write(tape)
            stream.flush();os.fsync(stream.fileno())
        receipt=dict(schema='B1-full19-native-observed-event-tail-v1',
            event=event,collision_observed=collision_observed,natural_terminal=terminal,
            game=job['game'],evaluation_seed=job['environment_seed'],action_seed=job['action_seed'],
            native_seed=job['native_seed'],checkpoint_sha256=self.checkpoint_sha256,
            policy_action_count=19,Bomb_allowed=True,history_rows=len(rows),
            acknowledged_decisions=job['decisions'],arrays=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            action_tape=dict(schema='B1-source19-acknowledged-opening-action-prefix-v1',
                path=str(tape_path),sha256=hashlib.sha256(tape).hexdigest(),decisions=len(tape),
                complete_natural_episode=terminal,replay_verified=False),
            before_facts=slot['last_before_facts'],after_facts={k:info.get(k) for k in FACT_FIELDS},
            episode_summary=info.get('episode_summary') if terminal else None,
             actual_full_map_history=True,
             actual_preaction_recurrent_memory=self.memory_kind=='model_recurrent_hidden',
             preaction_memory_kind=self.memory_kind,
            actions_changed=False,native_counterfactuals_executed=False)
        write(path.with_suffix('.json'),receipt)
        pointer=dict(receipt=str(path.with_suffix('.json')),arrays=str(path),sha256=receipt['sha256'],event=event)
        slot['saved_events'].append(pointer)
        return pointer

    def after_step(self,index,job,after,info,outcome):
        if outcome!=1:return None
        slot=self.slots[index];stage=slot['last_before_facts']['stage']
        if type(stage) is not int or not 1<=stage<=6:
            raise ValueError('Observed full-game collision needs its native stage')
        if stage in slot['collision_stages']:return None
        pointer=self._save(index,job,after,info,event=f'stage{stage}_first_observed_collision',
            collision_observed=True,terminal=False)
        slot['collision_stages'].add(stage)
        return pointer

    def finish_game(self,index,job,after,info):
        self._save(index,job,after,info,event='natural_terminal_tail',
            collision_observed=False,terminal=True)
        slot=self.slots.pop(index)
        if len(slot['saved_events'])>7:raise ValueError('Full-game event capture exceeded its bound')
        return dict(schema='B1-full19-native-event-trace-list-v1',events=slot['saved_events'],
            action_count=19,Bomb_allowed=True,maximum_saved_events=7,
            history_rows_max=self.history,actions_changed=False)
