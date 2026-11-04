"""Original input replay with adaptive loading intervals, no gameplay edits."""
from copy import deepcopy
import json
from .native_replay import semantic_state,tape_chunks
from .thrl_course_store import slice_tape


def state(snapshot):
    # Normalize tuples/lists exactly as the captured JSON, without fingerprints.
    return json.loads(json.dumps(semantic_state(snapshot),allow_nan=False))


def differing_fields(expected,actual):
    return [k for k in sorted(set(expected)|set(actual)) if expected.get(k)!=actual.get(k)]


def replay(controller,record,*,guard=lambda:None):
    if record.get('schema')!='frozen-SNN-actual-prefix-commands-v1' or record.get('status') not in ('stage5_entry','stage6_entry'):
        raise ValueError('A completed actual frozen prefix is required')
    s=controller.read();base=record['start']
    if (s.extra.get('native_seed') or {}).get('seed')!=record['native_seed']:
        raise ValueError('Prefix and fresh original opening have different native seeds')
    while s.stage==base['stage'] and s.stage_frame<base['stage_frame']:
        guard();s=controller.advance(1,1)
    actual=state(s);alignment=None
    if differing_fields(base,actual)==['rng']:
        first,second=deepcopy(base),deepcopy(actual)
        target={k:first['rng'].pop(k) for k in ('visual_seed','visual_counter')}
        for k in target:second['rng'].pop(k)
        if first==second:
            alignment=controller.align_prefix_visual_rng(actual['rng'],target)
            s=controller.read();actual=state(s)
    differences=differing_fields(base,actual)
    if differences:raise ValueError('Original prefix start differs: '+','.join(differences))
    tape=record['raw_input_segments'];offset=0;applied=0;barriers=[]
    def advance_to(target):
        nonlocal s,offset,applied
        for chunk in tape_chunks(slice_tape(tape,offset,target)):
            guard();s=controller.advance_tape(chunk);applied+=sum(t for _,t in chunk)
        offset=target
    for barrier in record['transitions']:
        before,after=barrier['before'],barrier['after']
        if before['ticks']>offset:advance_to(before['ticks'])
        if (s.stage,s.stage_frame)!=(before['stage'],before['stage_frame']):
            raise ValueError('Original pre-loading stage/frame differs')
        if before.get('rng') is not None and s.extra['native_rng_state']!=before['rng']:
            raise ValueError('Original pre-loading RNG differs')
        initial=barrier.get('initial_tape')
        if initial is None:
            # The first v6 transition was recorded before bridge metadata was
            # added. Preserve its actual first4tick move or1+3tick Bomb pair.
            index=before['segment'];initial=[tape[index]]
            if initial[0][1]==1 and initial[0][0]&2 and tape[index+1][1]==3:
                initial.append(tape[index+1])
        used=0
        for chunk in tape_chunks(initial):
            guard();s=controller.advance_tape(chunk);used+=sum(t for _,t in chunk)
        while (s.screen,s.stage,s.stage_frame)!=(7,after['stage'],after['stage_frame']):
            if used>=3600:raise ValueError('Native loading interval did not reach its recorded landing')
            guard();s=controller.advance(1,1);used+=1
        if barrier.get('state') is not None:
            differences=differing_fields(barrier['state'],state(s))
            if differences:raise ValueError('Post-loading original state differs: '+','.join(differences))
        applied+=used;offset=after['ticks']
        barriers.append(dict(stage=s.stage,stage_frame=s.stage_frame,
            recorded_ticks=after['ticks']-before['ticks'],actual_ticks=used,
            loading_bridge_ticks=used-sum(t for _,t in initial)))
    if record['ticks']>offset:advance_to(record['ticks'])
    differences=differing_fields(record['landing'],state(s))
    if differences:raise ValueError('Original stage6 landing differs: '+','.join(differences))
    events={k:s.extra['thrl_events'][k] for k in ('misses','bombs_used','point_items')}
    if record.get('native_events') is not None and events!=record['native_events']:
        raise ValueError('Replayed native Miss/Bomb/item history differs')
    return s,dict(status='matched_original_landing',recorded_input_ticks=record['ticks'],
        actual_input_ticks=applied,barriers=barriers,initial_visual_alignment=alignment,
        native_events=events,teacher_actions=0,optimizer_updates=0,gameplay_state_edits=False)


def restore_environment(env,path):
    """Fast-forward the EXE using its real input; rewards start at the landing."""
    from pathlib import Path
    import time
    from .thrl_study_io import read_shared_json
    from .thrl_durable_records import write_json
    record=read_shared_json(Path(path))
    started=time.perf_counter()
    snapshot,receipt=replay(env.controller,record)
    receipt['replay_seconds']=time.perf_counter()-started
    env.snapshot=env._check_snapshot(snapshot)
    env.episode_frames=snapshot.frame-env._episode_start_frame
    env._native_input_ticks=receipt['actual_input_ticks']
    env._nonplay_ticks=sum(r['loading_bridge_ticks'] for r in receipt['barriers'])
    env.stage_history=list(dict.fromkeys([record['start']['stage'],
        *[r['after']['stage'] for r in record['transitions']]]))
    # Full event counters are native hooks; do not manufacture Miss/Bomb totals.
    # Rewards are rebased at this real landing and explicitly reported as suffix-only.
    mapping=env.reward_model.reset(snapshot)
    env._mapping_unsupported.update(mapping['reward_unsupported'])
    env.encoder.reset()
    observation=env._encode(snapshot)
    receipt.update(source=str(path),prefix_native_frames=env.episode_frames,
        episode_statistics_scope='native-total-events;live-suffix-policy-reward-and-diagnostics',
        full_story_prefix_played_by_current_policy=False)
    env._frozen_prefix_replay_receipt=receipt
    write_json(Path(env.private_appdata)/'frozen-prefix-replay.json',receipt)
    info=env._info();info.update(mapping)
    from .compact_frozen_stage6_metrics import observe
    observe(env,info,False)
    return observation,info


def restore_brain(model,path,features,opening_seed,*,receipt_path=None):
    """Continue the corresponding brain; never substitute a fresh zero state."""
    import numpy as np
    import torch
    data=torch.load(path,map_location='cpu',weights_only=False)
    if data['schema'] not in ('frozen-SNN-stage6-entry-brain-v1','frozen-SNN-stage-entry-brain-v2') or data['opening_seed']!=opening_seed:
        raise ValueError('Brain capsule belongs to a different original opening')
    if not np.array_equal(data['features'],features):
        raise ValueError('Replayed first observation differs from the recorded brain input')
    fresh=model.initial_state(1)
    target=fresh.tensors();source=data['state']
    if len(source)!=len(target) or any(a.shape!=b.shape for a,b in zip(source,target)):
        raise ValueError('Saved brain state does not match the unchanged complete connectome')
    restored=type(fresh)(*(a.to(device=b.device,dtype=b.dtype) for a,b in zip(source,target)))
    if receipt_path is not None:
        from pathlib import Path
        from .thrl_durable_records import write_json
        write_json(Path(receipt_path),dict(status='restored',source=str(path),opening_seed=opening_seed,
            context=data.get('context'),first_observation_equal=True,tensor_shapes=[list(v.shape) for v in source],
            optimizer_updates=0))
    return restored
