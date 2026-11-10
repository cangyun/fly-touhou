"""Warm-start an actual TH14 attack with current-policy recurrent burn-in."""
from pathlib import Path
import json
import time
import numpy as np

SCHEMA = 'th14-nbnm-attack-start-bank-v1'


def choose_entry(path, split, task, rng, *, native_seed=None):
    path = Path(path)
    bank = json.loads(path.read_text(encoding='utf-8-sig'))
    if bank.get('schema') != SCHEMA: raise ValueError('Expected a native NBNM attack bank')
    from .nbnm_task import entry_matches
    rows = [r for r in bank['entries'] if r['split']==split and entry_matches(r['task'],task)]
    if native_seed is not None:rows=[r for r in rows if r['native_seed']==native_seed]
    if not rows: raise ValueError(f'No {split} starts for {task}; prepare that bank first')
    chosen = dict(rows[int(rng.integers(len(rows)))])
    prefix = Path(chosen['prefix'])
    if not prefix.is_absolute(): prefix=path.parent/prefix
    chosen['prefix'] = str(prefix)
    return chosen


def source_action(mask):
    from .background import action_mask
    from .thrl_native_env import SOURCE19_TO_NATIVE36
    if mask & 2: return 18
    for action,native in enumerate(SOURCE19_TO_NATIVE36[:18]):
        if action_mask(native)==mask: return action
    raise ValueError('Warm-up command is outside the original source19 action set')


def restore_entry(env, entry):
    from .compact_frozen_prefix_replay import replay
    record=json.loads(Path(entry['prefix']).read_text(encoding='utf-8-sig'))
    begun=time.perf_counter()
    snapshot,receipt=replay(env.controller,record)
    env.snapshot=env._check_snapshot(snapshot)
    env.encoder.reset()
    features=[];actions=[]
    target_seen_frames=[]
    applied=0
    for item in entry['warmup_commands']:
        commands=item['commands'];action=int(item['action'])
        if sum(t for _,t in commands) != 4: raise ValueError('Warm-up must use the policy four-frame interval')
        observation=env._encode(env.snapshot)
        from .nbnm_sensation import current_attack
        from .nbnm_task import attack_key
        if attack_key(current_attack(env.snapshot))==attack_key(entry['task'].get('attack')):
            target_seen_frames.append(env.snapshot.frame)
        features.append(observation['features'].tolist());actions.append(action)
        env.snapshot=env._check_snapshot(env.controller.advance_tape(commands))
        applied+=4
    env.episode_frames=env.snapshot.frame-env._episode_start_frame
    env._native_input_ticks=receipt['actual_input_ticks']+applied
    env.stage_history=([env.snapshot.stage] if entry.get('entry_protocol')=='native_stage_practice'
        else list(range(1,env.snapshot.stage+1)))
    env.reward_model.reset(env.snapshot)
    env._reset_episode_reward_diagnostics()
    observation=env._encode(env.snapshot)
    receipt.update(replay_seconds=time.perf_counter()-begun,task=entry['task'],
        brain_initialization='zero plus current-weight committed-action burn-in',
        burn_in_decisions=len(actions),prefix_is_NBNM_evidence=False,
        target_attack_preplayed_frames=(env.snapshot.frame-min(target_seen_frames) if target_seen_frames else 0))
    env._nbnm_entry_receipt=receipt
    from .thrl_durable_records import write_json
    write_json(Path(env.private_appdata)/'nbnm-prefix.json',receipt)
    burn=dict(features=features,actions=actions) if actions else None
    return observation,env._info(),burn


def burn_in(model, info, state, *, use_graph=None):
    """Known prefix actions need ONE candidate, not all18 candidate brains."""
    import torch
    history=info.get('policy_burn_in')
    if not history:return state
    from .spiking_engineered_control import SpikingState
    if use_graph is None:use_graph=getattr(model,'nbnm',{}).get('history_cuda_graph',False)
    if use_graph and model.device.type=='cuda' and len(history['actions'])>=32:
        with torch.no_grad():return _graph_history(model,history,state)
    with torch.no_grad():
        for feature,action in zip(history['features'],history['actions']):
            f=torch.as_tensor(feature,device=model.device,dtype=torch.float32)[None]
            cue=model.candidate_features(f)[:,0 if action==18 else action]
            _,state=model.core(cue,state)
    return state.detach()


def _graph_history(model, history, state):
    """Current-weight committed-action burn-in with fewer CUDA launches.

    Exactly one candidate/state update per historical action, just like the
    eager path. The short-lived graph is discarded before any Adam update.
    """
    import torch
    from .spiking_engineered_control import SpikingState
    features=torch.as_tensor(np.asarray(history['features']),device=model.device,dtype=torch.float32)
    actions=torch.as_tensor(history['actions'],device=model.device,dtype=torch.long)
    cues=model.candidate_features(features)[torch.arange(len(actions),device=model.device),
        torch.where(actions==18,0,actions)].contiguous()
    current=SpikingState(*(v.clone() for v in state.tensors()));cue=cues[:1].clone()
    cached=model.core.cache_inference_parameters;model.core.cache_inference_parameters=False
    try:
        stream=torch.cuda.Stream(device=model.device);stream.wait_stream(torch.cuda.current_stream(model.device))
        with torch.cuda.stream(stream):
            for _ in range(2):model.core(cue,current)
        torch.cuda.current_stream(model.device).wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _,following=model.core(cue,current)
            for target,value in zip(current.tensors(),following.tensors()):target.copy_(value)
        # Capture itself may run operators. Restore the true initial memory.
        for target,value in zip(current.tensors(),state.tensors()):target.copy_(value)
        for index in range(len(actions)):
            cue.copy_(cues[index:index+1]);graph.replay()
        return current.detach()
    finally:model.core.cache_inference_parameters=cached


def replay_resets(saved, chosen, length, device):
    result={}
    for column,(first,world) in enumerate(chosen):
        for t in range(length):
            state=saved.get((first+t,world))
            if state is not None:result.setdefault(t,[]).append((column,state.to(device)))
    return result


def apply_resets(state, rows):
    if not rows:return state
    import torch
    from .spiking_engineered_control import SpikingState
    indices=torch.tensor([i for i,_ in rows],device=state.membrane.device)
    source=SpikingState.pack([s for _,s in rows])
    return SpikingState(*(v.index_copy(0,indices,s) for v,s in zip(state.tensors(),source.tensors())))
