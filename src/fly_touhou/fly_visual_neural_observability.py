"""Opt-in read-only CPU snapshots and verified sparse 0.2ms replay observations."""
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .whole_brain import WholeBrainState, _HardSpike
from .fly_visual_native_batch import pack_whole_brain_states, unpack_whole_brain_state

VERSION = 'fly-visual-neural-observability-v1'
CORE_SHA256 = 'b2090d153c73df24da2b3dea299e072476d408729316ccbafe0075507c913365'


def _array(tensor):
    if tensor.device.type != 'cpu':
        raise ValueError('Statistics require existing CPU block snapshots')
    return tensor.detach().numpy()


def _stats(value):
    a = np.asarray(value, dtype=np.float64)
    if not a.size:
        return None
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite neural statistics')
    return dict(mean=float(a.mean()), std=float(a.std()), min=float(a.min()), max=float(a.max()))


def _groups(brain):
    vp = brain.input_indices.detach().cpu().numpy()
    dn = brain.output_indices.detach().cpu().numpy()
    if np.intersect1d(vp, dn).size:
        raise ValueError('Input and output ports overlap')
    middle = np.setdiff1d(np.arange(brain.n_neurons), np.concatenate((vp,dn)))
    return dict(VP_input_ports=vp, DN_output_ports=dn, remaining_intermediate=middle)


def _clone(state, device):
    values = []
    for field in fields(WholeBrainState):
        value = getattr(state,field.name)
        values.append(tuple(t.detach().to(device).clone() for t in value) if field.name=='delay_queue'
                      else value.detach().to(device).clone())
    return WholeBrainState(*values)


def _snapshot(brain, rollout, groups):
    width = rollout.block_states[(0,0)].membrane.shape[0]
    total = rollout.actions.shape[1]
    result, scanned = [], 0
    gain = None if brain.recurrent_gain_logits is None else brain.recurrent_gain_logits.detach().cpu().numpy()
    for group in range((total+width-1)//width):
        state = rollout.block_states[(0,group)]
        n = min(width,total-group*width)
        v, c = _array(state.membrane)[:n], _array(state.window_spike_counts)[:n]
        previous = _array(state.noise_counter)[:n] >= brain.substeps
        scanned += v.nbytes+c.nbytes+_array(state.rate)[:n].nbytes
        row = dict(physical_group=group, workers=list(range(group*width,group*width+n)),
            physical_batch=width, real_lanes=n, snapshot='pre-input at rollout t0',
            rng_keys=_array(state.rng_keys)[:n].tolist(), noise_counters=_array(state.noise_counter)[:n].tolist(),
            previous_20ms_window_available_rows=int(previous.sum()), populations={})
        for name, indices in groups.items():
            m, counts = v[:,indices], c[previous][:,indices]
            rates = counts.astype(np.float64)*(1000./brain.neural_ms_per_decision)
            row['populations'][name] = dict(neurons=len(indices), membrane_mv=_stats(m),
                near_threshold_fraction=float((np.abs(m-brain.parameters_source.v_threshold_mv)<=1.).mean()) if m.size else None,
                near_threshold_definition='abs(pre-input membrane - threshold)<=1mV',
                preceding_window_firing_hz=_stats(rates),
                preceding_window_active_fraction=float((counts>0).mean()) if counts.size else None,
                preceding_window_spikes=int(counts.sum()),
                gain_multiplier=_stats(1+brain.recurrent_gain_limit*np.tanh(gain[indices])) if gain is not None else dict(mean=1.,std=0.,min=1.,max=1.))
        row['DN_current_EMA_rate_hz'] = _stats(_array(state.rate)[:n])
        result.append(row)
    return dict(time='pre-input; count window is preceding neural decision, not the upcoming input',
        count_window_ms=brain.neural_ms_per_decision, new_game_zero_counter_is_not_measured_zero_firing=True,
        preceding_window_may_use_previous_policy_theta=bool(rollout.metadata.get('state_carry_crossed_parameter_version',False)),
        biological_anatomy_or_NT_labels_used=False, populations='engineering port partition only',
        batches=result, CPU_snapshot_payload_bytes=scanned)


def _traced_forward(brain, observations, state, cells, populations, records):
    """Exact _forward_validated tensor order, with read-only sparse sampling hooks."""
    p = brain.parameters_source
    probabilities = torch.sigmoid(brain.sensory_logits(observations))*brain.max_input_rate_hz*brain.dt_ms/1000.
    voltage, conductance = state.membrane, state.synaptic
    spikes, refractory, rate = state.spikes, state.refractory, state.rate
    queue, counter = state.delay_queue, state.noise_counter
    counts = torch.zeros_like(state.window_spike_counts)
    for _ in range(brain.substeps):
        available = refractory<=0
        integrated_v = p.v_rest_mv+brain._em*(voltage-p.v_rest_mv)+brain._coupling*conductance
        integrated_v = torch.where(available,integrated_v,voltage)
        integrated_g = torch.where(available,conductance*brain._es,conductance)
        spikes = _HardSpike.apply(integrated_v-p.v_threshold_mv,brain.surrogate_scale,brain.surrogate_slope_per_mv)*available
        synaptic_events = brain.recurrent_current(queue[0])
        conductance = integrated_g+synaptic_events
        uniform = brain._uniform(state.rng_keys,counter)
        hard_input = (uniform<probabilities).to(probabilities.dtype)
        input_events = probabilities+(hard_input-probabilities).detach()
        voltage = integrated_v.index_add(1,brain.input_indices,input_events*(p.poisson_weight_multiplier*p.weight_per_synapse_mv))
        hard_spikes = spikes.detach()
        voltage = voltage*(1.-hard_spikes)+p.v_reset_mv*hard_spikes
        conductance = conductance*(1.-hard_spikes)
        refractory = torch.where(hard_spikes.bool(),brain.refractory_limits[None,:],torch.clamp(refractory-1,min=0))
        output_spikes = spikes[:,brain.output_indices]
        rate = brain._readout_decay*rate+(1.-brain._readout_decay)*output_spikes*(1000./brain.dt_ms)
        queue = queue[1:]+(spikes,)
        counter = counter+1
        counts = counts+hard_spikes.to(torch.int32)
        records['prethreshold_mv'].append(integrated_v[0,cells].clone())
        records['postreset_mv'].append(voltage[0,cells].clone())
        records['binary_spikes'].append(hard_spikes[0,cells].to(torch.uint8).clone())
        records['population_spikes'].append(torch.stack([hard_spikes[0,g].sum().to(torch.int32) for g in populations]))
        records['noise_counter_after'].append(counter[0].clone())
    logits = brain.action_readout(rate/p.reference_poisson_rate_hz)
    return logits,WholeBrainState(voltage,conductance,spikes,refractory,rate,queue,state.rng_keys,counter,counts)


def _reset_boundary(brain,state,rollout,t,width,n,cost):
    lanes = list(unpack_whole_brain_state(state))
    for lane in range(width):
        if lane>=n or bool(rollout.episode_starts[t,lane]):
            key = 0 if lane>=n else int(rollout.brain_rng_keys[t,lane])
            lanes[lane] = brain.initial_state(1,rng_seed=key)
            cost['extra_state_initializations'] += 1
    return pack_whole_brain_states(lanes)


def _replay(brain,rollout,groups,output,tag,cost):
    width = rollout.block_states[(0,0)].membrane.shape[0]
    if width!=16 or len(rollout.actions)<5 or (4,0) not in rollout.block_states:
        return dict(status='prepared_not_available',reason='Require physical B16 and cached t4 boundary')
    core = Path(__file__).with_name('whole_brain.py')
    if hashlib.sha256(core.read_bytes()).hexdigest()!=CORE_SHA256:
        return dict(status='prepared_not_available',reason='Pinned core source differs')
    n = min(width,rollout.actions.shape[1]); device=brain.device
    selected=[]
    for name,indices in groups.items():
        number=min(32 if name=='remaining_intermediate' else 16,len(indices))
        selected.extend(indices[np.linspace(0,len(indices)-1,number,dtype=int)].tolist() if number else [])
    cells=torch.tensor(selected,dtype=torch.long,device=device)
    populations=[torch.tensor(g,dtype=torch.long,device=device) for g in groups.values()]
    records={key:[] for key in ('prethreshold_mv','postreset_mv','binary_spikes','population_spikes','noise_counter_after')}
    state=_clone(rollout.block_states[(0,0)],device); checks=[]
    for t in range(4):
        state=_reset_boundary(brain,state,rollout,t,width,n,cost)
        if (not torch.equal(state.rng_keys[:n].cpu(),rollout.brain_rng_keys[t,:n])
                or not torch.equal(state.noise_counter[:n].cpu(),rollout.noise_counters[t,:n])):
            return dict(status='replay_unverified',reason='Own keys/counters differ',checks=checks)
        encoded=rollout.encoded_visual[t,:n].to(device)
        if n<width:encoded=torch.cat((encoded,encoded.new_zeros(width-n,encoded.shape[1])))
        cost['extra_model_calls']+=1;cost['extra_neural_lane_forwards']+=width
        cost['extra_neural_microstep_lane_updates']+=width*brain.substeps
        raw,state=_traced_forward(brain,encoded,state,cells,populations,records)
        expected=rollout.raw_logits[t,:n]
        observed=raw[:n].cpu()
        checks.append(dict(t=t,rng_key_worker0=int(state.rng_keys[0]),
            noise_counter_before_worker0=int(rollout.noise_counters[t,0]),
            game_id_worker0=int(rollout.game_ids[t,0]) if hasattr(rollout,'game_ids') else None,
            raw_logits_exact=bool(torch.equal(observed,expected)),
            raw_logits_max_abs_error=float((observed-expected).abs().max())))
    state=_reset_boundary(brain,state,rollout,4,width,n,cost)
    expected=rollout.block_states[(4,0)];state_checks={}
    for field in fields(WholeBrainState):
        a,b=getattr(state,field.name),getattr(expected,field.name)
        if field.name=='delay_queue':
            state_checks[field.name]=all(torch.equal(x.detach().cpu(),y) for x,y in zip(a,b)) and len(a)==len(b)
        else:state_checks[field.name]=bool(torch.equal(a.detach().cpu(),b))
    verified=all(c['raw_logits_exact'] for c in checks) and all(state_checks.values())
    arrays={key:torch.stack(value).cpu().numpy() for key,value in records.items()}
    arrays.update(cell_indices=np.asarray(selected,np.int64),tick=np.arange(4*brain.substeps,dtype=np.int32),
        rollout_t=np.repeat(np.arange(4,dtype=np.int32),brain.substeps),
        neural_ms=(np.arange(4*brain.substeps,dtype=np.float64)+1)*brain.dt_ms)
    path=output/(tag+'-cells.npz');np.savez_compressed(path,**arrays)
    return dict(status='replay_verified' if verified else 'replay_unverified',checks=checks,
        all_boundary_state_fields_exact=state_checks,worker=0,physical_batch=width,
        actual_trace_claimed=verified,cell_count=len(selected),cell_root_ids=[str(brain.graph.node_ids[i]) for i in selected],
        cell_selection='deterministic VP16,DN16,remaining32,clipped only for tiny fixtures',
        population_order=list(groups),neural_ticks=4*brain.substeps,dt_ms=brain.dt_ms,
        voltage_timing='integrated prethreshold and post-input/reset; binary spikes from actual threshold comparison',
        native_time_scope='100 neural ticks share one current encoded observation;no subdecision native snapshots',
        NPZ=str(path),NPZ_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),NPZ_bytes=path.stat().st_size,
        sparse_uncompressed_bytes=sum(a.nbytes for a in arrays.values()),
        no_teacher_calls_or_native_actions=True,full_neuron_timeseries_saved=False)


def diagnose_neural_activity(model,rollout,*,output:Path,include_replay=False,enabled=False):
    """Fail-open diagnostic, called BEFORE update at the unchanged behavior theta."""
    if enabled is not True:
        return dict(schema=VERSION,status='disabled',extra_model_calls=0,extra_neural_lane_forwards=0)
    begun=time.perf_counter();cpu=time.process_time()
    result=dict(schema=VERSION,status='unavailable',extra_model_calls=0,extra_neural_lane_forwards=0,
        extra_neural_microstep_lane_updates=0,extra_state_initializations=0,extra_teacher_calls=0,
        native_actions=0,optimizer_updates=0,include_replay=bool(include_replay))
    try:
        with torch.no_grad():
            brain=model.brain;versions=tuple(p._version for p in model.parameters())
            groups=_groups(brain)
            result.update(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),core_sha256=CORE_SHA256,
                policy_version=rollout.metadata.get('policy_version'),neural_ms_per_decision=brain.neural_ms_per_decision,
                same_theta_required='caller invokes before any update; replay verifies against actual logits/state',
                interpretation='Activity/near-threshold or gain changes alone do not prove task information or causal usefulness',
                snapshot=_snapshot(brain,rollout,groups),status='snapshot_available')
            output=Path(output);output.mkdir(parents=True,exist_ok=True)
            tag=f"rollout-{int(rollout.metadata['policy_version']):06d}"
            result['replay']=(_replay(brain,rollout,groups,output,tag,result) if include_replay
                              else dict(status='disabled'))
            if tuple(p._version for p in model.parameters())!=versions:
                result.update(status='unavailable',reason='Unexpected parameter version change')
                result['replay'].update(status='replay_unverified',actual_trace_claimed=False)
            else:result['status']='available'
    except Exception as error:
        result.update(status='unavailable',reason=repr(error))
    result.update(elapsed_seconds=time.perf_counter()-begun,CPU_seconds=time.process_time()-cpu)
    try:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        tag=f"rollout-{int(getattr(rollout,'metadata',{}).get('policy_version',-1)):06d}"
        path=output/(tag+'-summary.json')
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
        result['summary_json']=str(path);result['summary_json_bytes']=path.stat().st_size
    except Exception as error:
        result['write_error']=repr(error)
    return result
