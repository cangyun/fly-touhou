"""Verified story-prefix starting points; no arbitrary savestate/ECL patch.

Reset replays actual native input in one or more batched commands. Resources,
deaths, RNG evolution and scripts remain original. Replaying a prefix is real
simulation work, reported separately from new on-policy PPO decisions.
"""
from dataclasses import asdict
from copy import deepcopy
import hashlib
import json

START_SCHEMA='th14-native-story-prefix-v2'
VISUAL_ALIGNMENT_SCHEMA='th14-prefix-visual-rng-alignment-v1'


def validate_tape(segments,*,max_ticks=262144):
    if not isinstance(segments,(list,tuple)) or not 1<=len(segments)<=max_ticks:
        raise ValueError('Empty or unbounded native tape')
    result=[];total=0
    for row in segments:
        if not isinstance(row,(list,tuple)) or len(row)!=2:
            raise ValueError('Native tape segment must be [mask,ticks]')
        mask,ticks=row
        if type(mask) is not int or not 0<=mask<=0xffffff or type(ticks) is not int or ticks<1:
            raise ValueError('Invalid native tape mask/ticks')
        total+=ticks
        if total>max_ticks:raise ValueError('Native input tape tick budget exceeded')
        if result and result[-1][0]==mask:result[-1][1]+=ticks
        else:result.append([mask,ticks])
    return result


def tape_chunks(segments,max_ticks=1024):
    if type(max_ticks) is not int or not 1<=max_ticks<=36000:raise ValueError('Invalid tape chunk bound')
    chunk=[];used=0
    for mask,ticks in validate_tape(segments):
        while ticks:
            take=min(ticks,max_ticks-used);chunk.append([mask,take]);used+=take;ticks-=take
            if used==max_ticks:yield chunk;chunk=[];used=0
    if chunk:yield chunk


def semantic_state(s):
    def without_id(row):return {k:v for k,v in row.items() if k!='id'}
    def ordered(rows):return sorted(rows,key=lambda row:json.dumps(row,sort_keys=True))
    geometry=s.extra['native_collision_geometry']
    shape={row['id']:without_id(row) for row in geometry['bullets']}
    raw={row['id']:without_id(row) for row in s.extra['hazard_metadata']['bullets']}
    bullets=[dict(bullet=without_id(asdict(b)),geometry=shape[b.id],hazard=raw[b.id]) for b in s.bullets]
    enemies={row['id']:without_id(row) for row in s.extra['resources']['enemies']}
    return {'stage':s.stage,'stage_frame':s.stage_frame,'player':asdict(s.player) if s.player else None,
        'player_geometry':geometry['player'],'bullets':ordered(bullets),
        'enemies':ordered([dict(enemy=without_id(asdict(e)),resource=enemies[e.id]) for e in s.enemies]),
        'lasers':ordered([without_id(asdict(l)) for l in s.lasers]),
        'items':ordered([without_id(row) for row in s.extra['resources']['items']]),
        'rng':s.extra['native_rng_state'],'lives':s.lives,'bombs':s.bombs,'power':s.power,
        'score':s.score,'graze':s.graze,'dialogue':s.dialogue,'character':s.character,
        'difficulty':s.difficulty,'subshot':s.subshot,'run_mode':s.extra['run_mode'],
        'continues':s.extra['continues'],'player_can_shoot':s.extra['resources']['player_can_shoot']}


def state_fingerprint(snapshot):
    return semantic_fingerprint(semantic_state(snapshot))


def semantic_fingerprint(state):
    data=json.dumps(state,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def visual_alignment_target(reference,current,expected_hash):
    """Only the two declared visual RNG fields may differ; never ignore physics."""
    if semantic_fingerprint(reference)!=expected_hash:
        raise ValueError('Prefix reference document does not match the verified base hash')
    first,second=deepcopy(reference),deepcopy(current)
    target={key:first['rng'].pop(key) for key in ('visual_seed','visual_counter')}
    for key in target:second['rng'].pop(key)
    if first!=second:
        raise ValueError('Prefix differs beyond visual RNG; refusing all alignment')
    return target


def _mismatch(controller,snapshot,entry,phase):
    from pathlib import Path
    from .atomic_json_io import atomic_write_json
    path=Path(controller.private_appdata).parent/'prefix_checkpoint_mismatch.json'
    atomic_write_json(path,{'entry_id':entry['id'],'phase':phase,
        'expected_sha256':entry[phase+'_state_sha256'],'actual_sha256':state_fingerprint(snapshot),
        'actual_state':semantic_state(snapshot),'native_seed_audit':snapshot.extra.get('native_seed')})
    raise ValueError(f'Native prefix {phase} checkpoint differs; no approximate landing; diagnostic={path}')


def replay_entry(controller,entry,*,checkpoints=None,on_checkpoint=None):
    if entry.get('schema')!=START_SCHEMA or type(entry.get('native_seed')) is not int:
        raise ValueError('Unversioned native prefix entry')
    seed=(controller.read().extra.get('native_seed') or {}).get('seed')
    if seed!=entry['native_seed']:raise ValueError('Native prefix reset has a different seed')
    s=controller.read()
    while s.stage==1 and s.stage_frame<entry['base_stage_frame']:
        s=controller.advance(1,1)
    alignment=None
    if state_fingerprint(s)!=entry['base_state_sha256']:
        declared=entry.get('base_visual_rng_alignment',{})
        if declared.get('schema')==VISUAL_ALIGNMENT_SCHEMA:
            target=visual_alignment_target(declared['reference_state'],semantic_state(s),entry['base_state_sha256'])
            alignment=controller.align_prefix_visual_rng(s.extra['native_rng_state'],target)
            s=controller.read()
        if state_fingerprint(s)!=entry['base_state_sha256']:
            _mismatch(controller,s,entry,'base')
    barriers=entry.get('loading_barriers',[])
    applied=0;barrier_receipts=[]
    if checkpoints is None and not barriers:
        for chunk in tape_chunks(entry['raw_input_segments']):s=controller.advance_tape(chunk)
    else:
        from .thrl_course_store import slice_tape
        offset=0
        def advance_to(target):
            nonlocal s,offset,applied
            for barrier in barriers:
                if barrier['end_ticks']<=offset or barrier['end_ticks']>target:continue
                if offset>barrier['start_ticks']:raise ValueError('Checkpoint falls inside a variable native loading interval')
                if offset<barrier['start_ticks']:
                    for chunk in tape_chunks(slice_tape(entry['raw_input_segments'],offset,barrier['start_ticks'])):
                        s=controller.advance_tape(chunk);applied+=sum(t for _,t in chunk)
                if state_fingerprint(s)!=barrier['prior_state_sha256']:
                    raise ValueError('Native pre-loading checkpoint does not match')
                used=0
                if barrier['initial_tape']:
                    for chunk in tape_chunks(barrier['initial_tape']):
                        s=controller.advance_tape(chunk);used+=sum(t for _,t in chunk)
                # Asset-loading duration is wall-dependent. Keep the recorded
                # active input, then the already-established shot-idle loading
                # input, until the EXACT loaded native stage/frame is reached.
                # A stage ID with the old stage timer is never a valid landing.
                while not (s.screen==7 and s.stage==barrier['stage'] and s.stage_frame==barrier['stage_frame']):
                    if used>=3600:raise ValueError('Bounded semantic native loading barrier did not complete')
                    s=controller.advance(1,1);used+=1
                if state_fingerprint(s)!=barrier['state_sha256']:
                    raise ValueError('Native post-loading physics/RNG checkpoint does not match')
                applied+=used
                barrier_receipts.append(dict(stage=s.stage,stage_frame=s.stage_frame,
                    recorded_input_ticks=barrier['end_ticks']-barrier['start_ticks'],actual_input_ticks=used))
                offset=barrier['end_ticks']
            if offset<target:
                for chunk in tape_chunks(slice_tape(entry['raw_input_segments'],offset,target)):
                    s=controller.advance_tape(chunk);applied+=sum(t for _,t in chunk)
            offset=target
        for row in checkpoints or []:
            target=row['ticks']
            if target<offset:raise ValueError('Recurrent replay checkpoints are out of order')
            advance_to(target)
            if state_fingerprint(s)!=row['state_sha256']:
                raise ValueError(f"Native trajectory checkpoint mismatch at input tick {target}; no approximate reset")
            if on_checkpoint is not None:on_checkpoint(s,row)
        final=sum(ticks for _,ticks in entry['raw_input_segments'])
        advance_to(final)
    if state_fingerprint(s)!=entry['landing_state_sha256']:
        _mismatch(controller,s,entry,'landing')
    tracking=entry.get('reward_tracking',{})
    if any(s.extra.get('thrl_events',{}).get(key)!=tracking.get(key)
           for key in ('misses','bombs_used','point_items')):
        raise ValueError('Native prefix event counters differ from captured reward state')
    if alignment:
        from dataclasses import replace
        s=replace(s,extra={**s.extra,'prefix_visual_rng_alignment':alignment})
    if barriers:
        from dataclasses import replace
        s=replace(s,extra={**s.extra,'prefix_replay_cost':dict(actual_input_ticks=applied,
            recorded_input_ticks=sum(t for _,t in entry['raw_input_segments']),loading_barriers=barrier_receipts)})
    return s
