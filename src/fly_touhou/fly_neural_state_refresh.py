"""Explicit finite-tail current-parameter refresh;no game/Adam/visual calls.

The old complete behavior boundary remains the initial condition. Replaying
the last16 encoded observations is an approximation,not full-prefix recovery
and NOT an extension of the learner's T4 neural backpropagation window.
"""
from __future__ import annotations
from dataclasses import fields
import math
import time

import numpy as np
import torch

from .brain_cuda_capture import _pointers,_tensors
from .fly_centered_native_evaluation import native_brain_key
from .fly_visual_native_batch import unpack_whole_brain_state
from .fly_visual_online_collector import copy_brain_state
from .fly_visual_online_accum_wide import _pad
from .fly_visual_ppo_cuda_graph import tensor_reset_state

SCHEMA='fly-current-parameter-finite-tail-neural-state-refresh-v1'
WIDTH=16
GROUPS=(tuple(range(16)),tuple(range(16,30)))


def _versions(states):
    return tuple(tuple((t.data_ptr(),t._version) for t in _tensors(s)) for s in states)


def _difference(old,new):
    result={};total=0.
    for field in fields(old):
        a,b=getattr(old,field.name),getattr(new,field.name)
        pairs=zip(a,b) if field.name=='delay_queue' else ((a,b),)
        squared=maximum=0.;changed=elements=0
        for x,y in pairs:
            d=y.detach().double()-x.detach().double()
            squared+=float(d.square().sum());maximum=max(maximum,float(d.abs().max()))
            changed+=int((x!=y).sum());elements+=x.numel()
        result[field.name]=dict(changed_elements=changed,elements=elements,L2=math.sqrt(squared),max_abs=maximum)
        total+=squared
    return dict(all_fields=result,total_L2=math.sqrt(total),changed_fields=[k for k,v in result.items() if v['changed_elements']])


@torch.no_grad()
def refresh_collector_state(model,collector,rollout,history_steps=16,check_stop=lambda:None):
    """After all Adam, before save/next collect;all30 states commit together.

    Source keys/clocks and within-tail natural resets are reproduced exactly.
    A natural terminal on the final source row keeps the already-created new
    game's fresh collector state. No native/gameclock/action RNG is advanced.
    Failure attaches actual partial costs as neural_state_refresh_diagnostics.
    """
    started=time.perf_counter();CPU_started=time.process_time()
    report=dict(schema=SCHEMA,status='not_committed',committed=False,history_steps=history_steps,
        neural_lane_forwards=0,real_neural_lane_forwards=0,padding_neural_lane_forwards=0,neural_batch_forwards=0,
        neural_batch_forward_attempts=0,attempted_neural_lanes=0,
        visual_encoder_forwards=0,critic_forwards=0,optimizer_updates=0,native_actions=0,
        current_parameter_tail_replay=True,old_parameter_boundary_retained=True,
        complete_prefix_equivalence_claimed=False,gradient_enabled=False,training_BPTT_changed=False,
        sampling_capture_created=False,worlds_restarted=0)
    try:
        if (collector.model is not model or collector.neural_batch_size!=WIDTH
                or tuple(tuple(g) for g in collector.worker_groups)!=GROUPS or len(collector.states)!=30
                or type(history_steps) is not int or not 1<=history_steps<=len(rollout.actions)
                or history_steps%4 or (len(rollout.actions)-history_steps)%4
                or model.brain.substeps!=100 or model.brain.dt_ms!=.2 or model.brain.n_actions!=19
                or rollout.encoded_visual.shape!=(len(rollout.actions),30,384)
                or rollout.encoded_visual.dtype!=torch.float32 or rollout.neural_batch_size!=WIDTH):
            raise ValueError('Original30-world B16 groups and an existing4-aligned finite FP32 history required')
        T=len(rollout.actions);start=T-history_steps
        if (rollout.episode_starts.shape!=(T,30) or rollout.episode_starts.dtype!=torch.bool
                or rollout.terminated.shape!=(T,30) or rollout.terminated.dtype!=torch.bool or bool(rollout.truncated.any())
                or rollout.brain_rng_keys.shape!=(T,30) or rollout.noise_counters.shape!=(T,30)
                or rollout.brain_rng_keys.dtype!=torch.int64 or rollout.noise_counters.dtype!=torch.int64):
            raise ValueError('Original natural-game starts,terminations and own key/counter history required')
        states=collector.states;versions=_versions(states);ids=tuple(id(s) for s in states)
        starts=np.asarray(collector.starts).copy();games=tuple(collector.game_ids)
        source_parameters=tuple((p.data_ptr(),p._version) for p in model.parameters())
        source_buffers=tuple((n,_pointers(t),t._version) for n,t in model.named_buffers())
        was_training=model.training;device=model.device
        if device.type=='cuda' and not torch.cuda.is_initialized():raise ValueError('Refresh cannot initialize CUDA')
        devices=[] if device.type!='cuda' else [device.index if device.index is not None else torch.cuda.current_device()]
        CPU_rng=torch.random.get_rng_state().clone()
        CUDA_rng=None if not devices else torch.cuda.get_rng_state(devices[0]).clone()
        terminal=rollout.terminated[-1].cpu().numpy().astype(bool)
        proposed=list(states);world_reports=[None]*30
        for world,current in enumerate(states):
            if terminal[world]:
                if not starts[world] or games[world]==int(rollout.game_ids[-1,world]):
                    raise ValueError('Final natural terminal must already have a fresh acknowledged game')
                fresh=model.initial_state(1,rng_seed=native_brain_key(games[world],collector.brain_seed))
                if not all(torch.equal(a,b) for a,b in zip(_tensors(current),_tensors(fresh))):
                    raise ValueError('Collector terminal successor is not the exact new-game fresh state')
            elif (starts[world] or int(current.rng_keys[0])!=int(rollout.brain_rng_keys[-1,world])
                    or int(current.noise_counter[0])!=int(rollout.noise_counters[-1,world])+100
                    or games[world]!=int(rollout.game_ids[-1,world])):
                raise ValueError('Current collector game/key/counter differs from the completed source rollout')
        check_stop()
        with torch.random.fork_rng(devices=devices):
            fresh=model.initial_state(WIDTH,rng_seed=0)
            for group,workers in enumerate(GROUPS):
                state=copy_brain_state(rollout.block_states[(start,group)],device=device)
                if state.membrane.shape[0]!=WIDTH:raise ValueError('Original complete B16 block_state required')
                for t in range(start,T):
                    check_stop()
                    reset=_pad(rollout.episode_starts[t,list(workers)].to(device),WIDTH,fill=True)
                    keys=_pad(rollout.brain_rng_keys[t,list(workers)].to(device),WIDTH)
                    state=tensor_reset_state(state,fresh,reset,keys)
                    if (not torch.equal(state.rng_keys[:len(workers)],rollout.brain_rng_keys[t,list(workers)].to(device))
                            or not torch.equal(state.noise_counter[:len(workers)],rollout.noise_counters[t,list(workers)].to(device))):
                        raise ValueError('Finite replay own key/counter history differs before a real observation')
                    encoded=_pad(rollout.encoded_visual[t,list(workers)].to(device),WIDTH)
                    report['neural_batch_forward_attempts']+=1;report['attempted_neural_lanes']+=WIDTH
                    _,state=model.brain(encoded,state)
                    report['neural_batch_forwards']+=1;report['neural_lane_forwards']+=WIDTH
                    report['real_neural_lane_forwards']+=len(workers);report['padding_neural_lane_forwards']+=WIDTH-len(workers)
                for world,replayed in zip(workers,unpack_whole_brain_state(state,count=len(workers))):
                    if terminal[world]:
                        world_reports[world]=dict(world=world,current_game=games[world],kept_fresh_after_final_terminal=True,
                            clock_before=int(states[world].noise_counter[0]),clock_after=int(states[world].noise_counter[0]),
                            key=int(states[world].rng_keys[0]),reset_rows=int(rollout.episode_starts[start:,world].sum()),
                            state_difference=_difference(states[world],states[world]))
                        continue
                    if (not torch.equal(replayed.rng_keys,states[world].rng_keys)
                            or not torch.equal(replayed.noise_counter,states[world].noise_counter)
                            or any(not bool(torch.isfinite(t).all()) for t in _tensors(replayed) if t.is_floating_point())):
                        raise ValueError('Replayed endpoint differs from actual game/key/clock or is nonfinite;no commit')
                    world_reports[world]=dict(world=world,current_game=games[world],kept_fresh_after_final_terminal=False,
                        clock_before=int(states[world].noise_counter[0]),clock_after=int(replayed.noise_counter[0]),
                        key=int(replayed.rng_keys[0]),reset_rows=int(rollout.episode_starts[start:,world].sum()),
                        state_difference=_difference(states[world],replayed))
                    proposed[world]=replayed
        check_stop()
        invariants=dict(parameters_unchanged=source_parameters==tuple((p.data_ptr(),p._version) for p in model.parameters()),
            model_buffers_unchanged=source_buffers==tuple((n,_pointers(t),t._version) for n,t in model.named_buffers()),
            original_collector_states_unchanged=collector.states is states and tuple(id(s) for s in states)==ids and _versions(states)==versions,
            collector_games_starts_unchanged=tuple(collector.game_ids)==games and np.array_equal(collector.starts,starts),
            training_mode_unchanged=model.training==was_training,global_CPU_RNG_unchanged=torch.equal(CPU_rng,torch.random.get_rng_state()),
            global_CUDA_RNG_unchanged=CUDA_rng is None or torch.equal(CUDA_rng,torch.cuda.get_rng_state(devices[0])))
        if not all(invariants.values()):raise RuntimeError('Refresh read-only source/parameter/RNG invariant failed;no commit')
        report.update(source_start=start,source_end_exclusive=T,worlds=world_reports,invariants=invariants,
            elapsed_seconds=time.perf_counter()-started,CPU_seconds=time.process_time()-CPU_started,
            sampling_capture_reused=getattr(model.brain,'_captured_inference',None) is not None,
            natural_fresh_states_retained=int(terminal.sum()),live_states_refreshed=int((~terminal).sum()),
            note='16 is no-grad suffix refresh from an old behavior boundary;not16-step BPTT or complete-prefix equality')
        collector.states=proposed  # Single atomic reference replacement after ALL30 validations.
        report.update(status='committed',committed=True)
        return report
    except BaseException as error:
        report.update(status='failed_without_state_commit',elapsed_seconds=time.perf_counter()-started,
            CPU_seconds=time.process_time()-CPU_started,error_type=type(error).__name__,error=str(error),
            incomplete_forward_internal_work_unknown=report['neural_batch_forward_attempts']!=report['neural_batch_forwards'])
        error.neural_state_refresh_diagnostics=report
        raise
