"""Sparse lossless native snapshots during a fixed-action diagnostic replay.

No sensors, actions, rewards or game fields are changed. Records retain all
native candidates before the normal96-token selection, including raw flags.
Only a few declared preaction checkpoints and the actual endpoint are saved.
"""
from pathlib import Path
import gzip
import hashlib
import json
import math
import struct

import numpy as np

from .atomic_json_io import atomic_write_json
from .thrl_avoidance_probe import AvoidanceProbeEnv

SCHEMA='B1-sparse-original-replay-snapshot-v1'
ENEMY_SCHEMA='B1-unfiltered-native-enemy-manager-v1'


def read_unfiltered_enemy_manager(controller,snapshot):
    """Read the pinned manager list without the normal animation/HP filters.

    Flags are retained as measurements/profile associations, not a new native
    collision oracle. Nonfinite unused fields keep their exact raw bits.
    """
    reader=controller.reader
    before=reader.global_frame()
    if before!=snapshot.frame:
        raise ValueError('Raw enemy manager does not match the acknowledged snapshot frame')
    offsets=reader.o['enemies'];memory=reader.mem
    manager=reader.pointer('enemies','enemy_manager_pointer')
    rows=[]
    retained={enemy.id for enemy in snapshot.enemies}

    def pair(address):
        raw=memory.read(address,8)
        values=struct.unpack('<ff',raw)
        finite=all(math.isfinite(x) for x in values)
        return dict(f32_hex=raw.hex(),finite=finite,values=list(values) if finite else None)

    if manager:
        for identity in reader._entries(memory.integer(manager+offsets['zEnemyManager_list']),limit=2048):
            data=identity+offsets['zEnemy_data']
            flags=memory.integer(data+offsets['zEnemyData_flags'],size=8)
            boss=bool(flags&offsets['zEnemyFlags_is_boss'])
            animation=memory.integer(data+offsets['zEnemyData_anm_vm_id'])
            hp=memory.integer(data+offsets['zEnemyData_hp'],signed=True)
            rows.append(dict(id=identity,native_flags=flags,native_hp=hp,
                native_animation_vm_id=animation,boss_flag=boss,
                intangible_flag=bool(flags&offsets['zEnemyFlags_intangible']),
                no_hitbox_flag=bool(flags&offsets['zEnemyFlags_no_hitbox']),
                contactable_by_existing_profile_flags=not bool(flags&(
                    offsets['zEnemyFlags_no_hitbox']|offsets['zEnemyFlags_intangible'])),
                present_in_normal_snapshot=identity in retained,
                normal_reader_animation_filter_passes=boss or bool(animation),
                existing_encoder_positive_HP_filter_passes=hp>0,
                position=pair(data+offsets['zEnemyData_pos']),
                velocity=pair(data+offsets['zEnemyData_vel']),
                raw_hitbox_pair=pair(data+offsets['zEnemyData_hitbox'])))
    after=reader.global_frame()
    if after!=before:
        raise ValueError('Raw enemy manager read crossed native frames')
    return dict(schema=ENEMY_SCHEMA,observed=True,snapshot_frame=snapshot.frame,
        native_frame_before=before,native_frame_after=after,manager=manager,
        enemies=rows,normal_snapshot_ids=sorted(retained),native_collision_gate_verified=False,
        modifications_to_game_memory=0)


def json_numeric(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError('Unsupported raw native snapshot value: '+type(value).__name__)


class SparseRawReplayCapture:
    def __init__(self, directory, schedule):
        if not isinstance(schedule,dict) or not 1<=len(schedule)<=20:
            raise ValueError('Bounded explicit replay capture schedule required')
        self.schedule={}
        for seed,positions in schedule.items():
            seed=int(seed)
            if (not 0<=seed<2**32 or not isinstance(positions,(list,tuple))
                    or not 1<=len(positions)<=8
                    or any(type(x) is not int or not 0<=x<16384 for x in positions)):
                raise ValueError('Capture only bounded declared replay decision indices')
            if seed in self.schedule:
                raise ValueError('Duplicate normalized evaluation seed')
            self.schedule[seed]=frozenset(positions)
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.written=set()

    def write(self,snapshot,*,evaluation_seed,native_episode,decision,side,enemy_manager=None):
        if side not in ('before','endpoint'):
            raise ValueError('Only preaction and acknowledged endpoint capture allowed')
        if evaluation_seed not in self.schedule:
            raise ValueError('Undeclared original replay seed')
        if side=='before' and decision not in self.schedule[evaluation_seed]:
            return None
        snapshot.validate()
        if enemy_manager is not None and (enemy_manager.get('schema')!=ENEMY_SCHEMA
                or enemy_manager.get('observed') is not True
                or any(enemy_manager.get(k)!=snapshot.frame for k in
                       ('snapshot_frame','native_frame_before','native_frame_after'))):
            raise ValueError('Unfiltered enemy sidecar must belong to the same exact native frame')
        key=(evaluation_seed,native_episode,decision,side)
        if key in self.written:
            raise ValueError('Original replay checkpoint already captured')
        payload=dict(schema=SCHEMA,evaluation_seed=evaluation_seed,
            native_episode=native_episode,decision=decision,side=side,
            snapshot=snapshot.to_dict(),unfiltered_enemy_manager=enemy_manager,
            policy_input_changed=False,
            actions_changed=False,unexecuted_native_alternatives_verified=False)
        encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False,
            separators=(',',':'),default=json_numeric).encode('utf-8')
        if len(encoded)>32*1024**2:
            raise ValueError('Bounded raw snapshot exceeds32MiB')
        compressed=gzip.compress(encoded,compresslevel=1,mtime=0)
        path=self.directory/f'seed{evaluation_seed}_episode{native_episode}_decision{decision:05d}_{side}.json.gz'
        with path.open('xb') as stream:
            stream.write(compressed)
        receipt=dict(schema=SCHEMA,evaluation_seed=evaluation_seed,
            native_episode=native_episode,decision=decision,side=side,
            path=str(path.resolve()),sha256=hashlib.sha256(compressed).hexdigest(),
            uncompressed_sha256=hashlib.sha256(encoded).hexdigest(),
            compressed_bytes=len(compressed),uncompressed_bytes=len(encoded),
            native_frame=snapshot.frame,stage=snapshot.stage,stage_frame=snapshot.stage_frame,
            native_bullets=len(snapshot.bullets),native_enemies=len(snapshot.enemies),
            native_lasers=len(snapshot.lasers),all_native_snapshot_fields_retained=True,
            unfiltered_enemy_manager_observed=enemy_manager is not None,
            policy_input_changed=False,actions_changed=False)
        atomic_write_json(path.with_suffix('.receipt.json'),receipt)
        self.written.add(key)
        return receipt


class RawCapturingAvoidanceProbeEnv(AvoidanceProbeEnv):
    def __init__(self,*args,raw_capture_directory,raw_capture_schedule,**kwargs):
        self._raw_capture=SparseRawReplayCapture(raw_capture_directory,raw_capture_schedule)
        self._capture_seed=None
        super().__init__(*args,**kwargs)

    def reset(self,*,seed=None,options=None):
        if not options or set(options)!={'evaluation_seed'}:
            raise ValueError('Raw replay requires an explicit original evaluation seed')
        self._capture_seed=int(options['evaluation_seed'])
        if self._capture_seed not in self._raw_capture.schedule:
            raise ValueError('Raw replay seed was not declared before native launch')
        return super().reset(seed=seed,options=options)

    def step(self,action):
        position=self._probe_decisions
        if position in self._raw_capture.schedule[self._capture_seed]:
            enemies=read_unfiltered_enemy_manager(self.controller,self.snapshot)
            self._raw_capture.write(self.snapshot,evaluation_seed=self._capture_seed,
                native_episode=self._full_story_episode,decision=position,side='before',enemy_manager=enemies)
        result=super().step(action)
        if result[2] or result[3]:
            enemies=read_unfiltered_enemy_manager(self.controller,self.snapshot)
            self._raw_capture.write(self.snapshot,evaluation_seed=self._capture_seed,
                native_episode=self._full_story_episode,decision=position,side='endpoint',enemy_manager=enemies)
        return result


def raw_capture_probe_factory(kwargs):
    return RawCapturingAvoidanceProbeEnv(**kwargs)
