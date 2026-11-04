"""Read-only population tracing from ACTUAL CPU rollout block states.

Definitions only. No graph/model is loaded, no neural/teacher forward is run,
and no running recorder, collector or learner is edited. Call after collect(),
before Adam; t-before state is the preceding row's following state only after
the explicit actual-world/game/key/counter/reset checks below.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .fly_representation_trace import _cpu, _sha, _raw_sha, _atomic_json, _json_value

SCHEMA = 'fly-actual-population-path-CPU-blockstate-trace-v1'
ANNOTATIONS_SHA256 = '30be6c73975a70c56d930e27911f36455d3886e15abf383b78edd2a5d679e0b6'
DEFAULT_ANNOTATIONS = Path(__file__).resolve().parents[2]/'data/connectome/annotations/flywire_annotations_v2.1.0.tsv'
ANNOTATION_FIELDS = ('super_class','cell_class','cell_sub_class','cell_type','hemibrain_type','side','top_nt')
ROLES = ('input_ports','middle_remaining','DN_outputs')
CHUNK_ROWS = 128
CELLS_PER_TYPE_SIDE = 4


def _digest(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def _readonly(value):
    value=np.asarray(value);value.setflags(write=False);return value


def _identity(brain):
    roots=tuple(str(x) for x in brain.graph.node_ids)
    inputs=brain.input_indices.detach().to(device='cpu',copy=True).numpy().astype(np.int64,copy=False)
    outputs=brain.output_indices.detach().to(device='cpu',copy=True).numpy().astype(np.int64,copy=False)
    n=len(roots)
    if (n!=brain.n_neurons or len(set(roots))!=n or not n
            or any(x.ndim!=1 or not len(x) or len(np.unique(x))!=len(x) or x.min()<0 or x.max()>=n
                   for x in (inputs,outputs)) or np.intersect1d(inputs,outputs).size):
        raise ValueError('Exact graph root order and disjoint actual input/DN ports required')
    return roots,inputs.copy(),outputs.copy(),_digest([list(roots),inputs.tolist(),outputs.tolist()])


@dataclass(frozen=True)
class PopulationLayout:
    identity_sha256: str
    roots: tuple
    input_indices: np.ndarray
    output_indices: np.ndarray
    role_labels: np.ndarray
    role_present: np.ndarray
    partition_labels: np.ndarray
    population_sizes: np.ndarray
    populations: tuple
    sampled_indices: np.ndarray
    sampled_cells: tuple
    provenance: dict


def build_population_layout(brain,annotations_path=DEFAULT_ANNOTATIONS,*,synthetic_fixture=False):
    """CPU annotation join by root ID, never row position/root spacing/hop.

    All neurons belong to one engineering-role x actual annotation-class/side
    partition. Role totals overlap those partitions deliberately. Fine-type
    cells use cell_type, then explicitly named hemibrain_type, or unknown; at
    most4 per role/type/side are selected by fixed hash without any RNG draws.
    """
    roots,inputs,outputs,identity=_identity(brain)
    path=Path(annotations_path);annotation_sha=_sha(path)
    if not synthetic_fixture and (len(roots)!=138639 or len(inputs)!=1024 or len(outputs)!=1299
                                 or annotation_sha!=ANNOTATIONS_SHA256):
        raise ValueError('Pinned full graph/1024/1299/local annotation identity required')
    lookup={root:i for i,root in enumerate(roots)}
    annotations=[None]*len(roots);matched=0
    with path.open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream,delimiter='\t')
        if 'root_id' not in (reader.fieldnames or ()):
            raise ValueError('Annotations require explicit root_id, never positional matching')
        columns=tuple(reader.fieldnames)
        for row in reader:
            index=lookup.get(row['root_id'])
            if index is None:continue
            if annotations[index] is not None:raise ValueError('Duplicate graph root in annotations')
            annotations[index]={k:(row.get(k,'').strip() or 'unknown') for k in ANNOTATION_FIELDS}
            matched+=1
    missing=lambda:{k:'unknown' for k in ANNOTATION_FIELDS}
    annotations=[missing() if a is None else a for a in annotations]
    roles=np.full(len(roots),1,np.int64);roles[inputs]=0;roles[outputs]=2
    partition_keys=[(ROLES[int(r)],a['super_class'],a['cell_class'],a['cell_sub_class'],a['side'])
                    for r,a in zip(roles,annotations)]
    keys=sorted(set(partition_keys));key_id={k:i for i,k in enumerate(keys)}
    partition=np.asarray([key_id[k] for k in partition_keys],np.int64)
    present=np.unique(roles);role_counts=np.bincount(roles,minlength=3)[present]
    groups=[]
    def detail(indices,kind,role,**labels):
        return dict(kind=kind,engineering_role=role,neurons=len(indices),**labels,
            annotation_label_counts={field:dict(sorted(Counter(annotations[i][field] for i in indices).items()))
                                     for field in ANNOTATION_FIELDS})
    for role in present:
        indices=np.flatnonzero(roles==role)
        groups.append(detail(indices,'engineering_role_total',ROLES[int(role)],side='mixed_or_unknown'))
    by_partition=defaultdict(list)
    for index,group in enumerate(partition):by_partition[int(group)].append(index)
    for group,key in enumerate(keys):
        role,super_class,cell_class,subclass,side=key
        groups.append(detail(by_partition[group],'annotation_class_side_partition',role,
            super_class=super_class,cell_class=cell_class,cell_sub_class=subclass,side=side))
    for i,g in enumerate(groups):g['population_id']=i
    strata=defaultdict(list)
    for index,(role,a) in enumerate(zip(roles,annotations)):
        field=('cell_type' if a['cell_type']!='unknown' else
               'hemibrain_type' if a['hemibrain_type']!='unknown' else 'unknown')
        label=a[field] if field!='unknown' else 'unknown'
        strata[(ROLES[int(role)],field,label,a['side'])].append(index)
    sampled=[]
    for stratum,indices in sorted(strata.items()):
        indices=sorted(indices,key=lambda i:_digest([SCHEMA,'cells',list(stratum),roots[i]]))
        sampled.extend(indices[:CELLS_PER_TYPE_SIDE])
    sampled=sorted(sampled)
    cells=tuple(dict(graph_index=i,root_id=roots[i],engineering_role=ROLES[int(roles[i])],
                     **annotations[i]) for i in sampled)
    sizes=np.concatenate((role_counts,np.bincount(partition,minlength=len(keys)))).astype(np.int64)
    return PopulationLayout(identity,roots,_readonly(inputs),_readonly(outputs),_readonly(roles),
        _readonly(present),_readonly(partition),_readonly(sizes),tuple(groups),_readonly(np.asarray(sampled,np.int64)),
        cells,dict(annotation_path=str(path.resolve()),annotation_sha256=annotation_sha,
            annotation_columns=list(columns),missing_annotation_fields=[k for k in ANNOTATION_FIELDS if k not in columns],
            graph_neurons=len(roots),matched_annotation_rows=matched,missing_graph_annotation_rows=len(roots)-matched,
            unknown_label_counts={k:sum(a[k]=='unknown' for a in annotations) for k in ANNOTATION_FIELDS},
            sampled_cell_strata=len(strata),sampled_cells=len(sampled),maximum_cells_per_type_side=CELLS_PER_TYPE_SIDE,
            sampled_type_precedence=['actual cell_type','explicit hemibrain_type when cell_type unknown','unknown'],
            sample_rule='fixed SHA256 of role/type-field/type/side/root;no RNG',
            middle_definition='all graph neurons outside engineered input/DN ports;not a functional brain region',
            hemisphere_source='annotation.side only;never inferred from root order/coordinates/hops',
            NT_source='annotation.top_nt label only;not used to overwrite signed edges',synthetic_fixture=synthetic_fixture))


def _population_sum(layout,value):
    return np.concatenate((np.bincount(layout.role_labels,weights=value,minlength=3)[layout.role_present],
        np.bincount(layout.partition_labels,weights=value,minlength=len(layout.populations)-len(layout.role_present))))


def _snapshot_contract(state,width,n,outputs):
    for name,dtype,shape in (('membrane',torch.float32,(width,n)),('synaptic',torch.float32,(width,n)),
        ('spikes',torch.float32,(width,n)),('refractory',torch.int64,(width,n)),
        ('rate',torch.float32,(width,outputs)),('window_spike_counts',torch.int32,(width,n)),
        ('rng_keys',torch.int64,(width,)),('noise_counter',torch.int64,(width,))):
        _cpu(getattr(state,name),shape,dtype,'block_state.'+name)
    if not isinstance(state.delay_queue,tuple) or len(state.delay_queue)!=9:
        raise ValueError('Original complete9-microstep conduction delay required')
    # Validate shape/dtype here; binary values are checked on selected rows.
    for value in state.delay_queue:
        if (not torch.is_tensor(value) or value.shape!=(width,n) or value.dtype!=torch.float32
                or value.device.type!='cpu' or value.requires_grad):
            raise ValueError('Original detached CPU full-neuron delay queue required')


def record_population_path_trace(model,rollout,output,max_scenes=512,*,annotations_path=DEFAULT_ANNOTATIONS,layout=None):
    """Atomic CPU trace; output JSON metadata, no state/parameter/RNG mutation.

    Actual20ms input/middle/DN firing counts and DN EMA are distinct channels.
    Expected external Poisson injection rate is reconstructed separately from
    recorded standardized384 and copied inputW/b; realized injected events are
    NOT saved by the state and are not invented. No causal decoding is claimed.
    """
    began=time.monotonic()
    if type(max_scenes) is not int or not 1<=max_scenes<=512:
        raise ValueError('Bounded max_scenes1..512 required')
    output=Path(output)
    if output.exists():raise FileExistsError('Fresh population trace output required')
    brain=model.brain;metadata=rollout.metadata;steps,worlds=rollout.actions.shape
    groups=tuple(tuple(g) for g in metadata.get('worker_groups',()))
    if (metadata.get('neural_batch_size')!=16 or worlds!=30 or tuple(map(len,groups))!=(16,14)
            or sorted(w for g in groups for w in g)!=list(range(30))):
        raise ValueError('Explicit actual worker_groups16/14 required;never infer world//16')
    version=metadata.get('policy_version');fixture=metadata.get('synthetic_fixture') is True
    if type(version) is not int or version<0 or (brain.substeps,brain.dt_ms,brain.obs_dim,brain.n_actions)!=(100,.2,384,19):
        raise ValueError('Original policy version and100x.2ms visual384/source19 required')
    if brain.geometry_input_encoder is not None or brain.input_encoder.bias is None or brain.action_readout.bias is not None:
        raise ValueError('Existing visual->input->wholebrain->biasfreeDN19 path required')
    identity=_identity(brain)[-1]
    if layout is None:layout=build_population_layout(brain,annotations_path,synthetic_fixture=fixture)
    if not isinstance(layout,PopulationLayout) or layout.identity_sha256!=identity:
        raise ValueError('Population layout graph/root/port order differs')
    n=len(layout.roots);ndn=len(layout.output_indices);nin=len(layout.input_indices)
    fields={'encoded_visual':((steps,worlds,384),torch.float32),'features':((steps,worlds,25),torch.float32),
        'raw_logits':((steps,worlds,19),torch.float32),'action_probs':((steps,worlds,19),torch.float32),
        'legal_actions':((steps,worlds,19),torch.bool),'actions':((steps,worlds),torch.int64),
        'old_log_probs':((steps,worlds),torch.float32),'episode_starts':((steps,worlds),torch.bool),
        'terminated':((steps,worlds),torch.bool),'game_ids':((steps,worlds),torch.int64),
        'native_episodes':((steps,worlds),torch.int64),'brain_rng_keys':((steps,worlds),torch.int64),
        'noise_counters':((steps,worlds),torch.int64)}
    source={k:_cpu(getattr(rollout,k),shape,dtype,k) for k,(shape,dtype) in fields.items()}
    task=rollout.task_targets
    for name,shape,dtype in (('margins',(steps,30,3,18),np.float32),('trusted',(steps,30,3),np.bool_),
        ('outcomes',(steps,30),np.int8),('valid_mask',(steps,30),np.bool_),('unprotected_mask',(steps,30),np.bool_),
        ('actions',(steps,30),np.int64)):
        a=np.asarray(getattr(task,name))
        if a.shape!=shape or a.dtype!=dtype:raise ValueError('Original task field differs:'+name)
    if (not np.isfinite(task.margins).all() or not np.array_equal(task.actions,source['actions'].numpy())
            or len(task.step_evidence)!=steps or len(rollout.step_info)!=steps):
        raise ValueError('Original aligned geometry/native facts required')
    optional={}
    for name,tail,dtype in (('composed_logits',(19,),torch.float32),('predicted_costs',(18,),torch.float32),
        ('d95_risk_costs',(18,),torch.float32),('d95_risk_queried_mask',(),torch.bool),
        ('d95_risk_valid_mask',(),torch.bool)):
        value=getattr(rollout,name,None)
        if value is not None:optional[name]=_cpu(value,(steps,30,*tail),dtype,name)
    if ('d95_risk_costs' in optional and not {'d95_risk_queried_mask','d95_risk_valid_mask'}<=optional.keys()):
        raise ValueError('Teacher cost placeholders require both query and validity masks')
    versions=tuple((p.data_ptr(),p._version) for p in model.parameters())
    with torch.no_grad():
        weight=brain.input_encoder.weight.detach().to(device='cpu',copy=True)
        bias=brain.input_encoder.bias.detach().to(device='cpu',copy=True)
        head=brain.action_readout.weight.detach().to(device='cpu',copy=True)
    if (weight.shape!=(nin,384) or bias.shape!=(nin,) or head.shape!=(19,ndn)
            or any(v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) for v in (weight,bias,head))):
        raise ValueError('Actual finite FP32 input/readout parameters required')
    reference=float(brain.parameters_source.reference_poisson_rate_hz)
    window_ms=brain.substeps*brain.dt_ms
    if not np.isfinite(reference) or reference<=0:raise ValueError('Original positive DN reference rate required')
    excluded=Counter();candidates=[]
    times=sorted(t for t,g in rollout.block_states if g==0 and 0<t<steps)
    for t in times:
        if t%4:raise ValueError('Original4-step CPU snapshots required')
        previous=t-1
        for g,workers in enumerate(groups):
            state=rollout.block_states[(t,g)];_snapshot_contract(state,16,n,ndn)
            for lane,world in enumerate(workers):
                if source['episode_starts'][t,world]:reason='new_episode_at_snapshot'
                elif source['terminated'][previous,world]:reason='previous_action_terminated'
                elif not task.valid_mask[previous,world]:reason='invalid_source_row'
                elif source['game_ids'][previous,world]!=source['game_ids'][t,world]:reason='game_changed'
                elif source['native_episodes'][previous,world]!=source['native_episodes'][t,world]:reason='native_episode_changed'
                elif (state.rng_keys[lane]!=source['brain_rng_keys'][previous,world]
                      or state.rng_keys[lane]!=source['brain_rng_keys'][t,world]):reason='RNG_key_mismatch'
                elif (state.noise_counter[lane]!=source['noise_counters'][previous,world]+100
                      or state.noise_counter[lane]!=source['noise_counters'][t,world]):reason='counter_mismatch'
                else:
                    endpoint=task.step_evidence[previous]['before_native_endpoints'][world]
                    reason=('source_native_endpoint_mismatch' if int(endpoint[0])!=int(source['native_episodes'][previous,world])
                        or rollout.step_info[previous][world].get('source_native_episode')!=int(endpoint[0]) else None)
                if reason:excluded[reason]+=1;continue
                m=task.margins[previous,world,0]
                critical=bool(task.trusted[previous,world,0] and task.unprotected_mask[previous,world]
                    and np.any(m<=0) and np.any(m>1))
                candidates.append((previous,world,g,lane,int(source['game_ids'][previous,world]),critical))
    rank=lambda r,kind:_digest([SCHEMA,kind,version,r[0],r[1],r[4]])
    critical=sorted((r for r in candidates if r[-1]),key=lambda r:rank(r,'critical'))
    selected=[(r,'critical') for r in critical[:max_scenes//2]]
    taken={(r[0],r[1]) for r,_ in selected}
    rest=sorted((r for r in candidates if (r[0],r[1]) not in taken),key=lambda r:rank(r,'uniform'))
    selected+=[(r,'uniform') for r in rest[:max_scenes-len(selected)]]
    selected.sort(key=lambda item:(item[0][0],item[0][1]))
    output.mkdir(parents=True,exist_ok=False)
    catalog=dict(schema=SCHEMA,identity_sha256=identity,annotation_provenance=layout.provenance,
        populations=list(layout.populations),sampled_cells=list(layout.sampled_cells),
        input_indices=layout.input_indices.tolist(),output_indices=layout.output_indices.tolist(),
        population_sizes=layout.population_sizes.tolist(),population_overlap='role totals plus disjoint class/side partition')
    _atomic_json(output/'populations.json',catalog)
    parts=[];facts=[];error_max=0.;danger_pairs=[]
    for start in range(0,len(selected),CHUNK_ROWS):
        records=[]
        for row,selection in selected[start:start+CHUNK_ROWS]:
            t,w,g,lane,game,_=row;state=rollout.block_states[(t+1,g)]
            counts=state.window_spike_counts[lane].numpy()
            v=state.membrane[lane].numpy();syn=state.synaptic[lane].numpy()
            if np.any((counts<0)|(counts>brain.substeps)) or not np.isfinite(v).all() or not np.isfinite(syn).all():
                raise ValueError('Actual finite full-neuron window/membrane/synaptic state required')
            pop_sum=_population_sum(layout,counts)
            recent=[]
            for delayed in state.delay_queue:
                spikes=delayed[lane].numpy()
                if not np.all((spikes==0)|(spikes==1)):raise ValueError('Actual delayed spikes must be binary')
                recent.append(_population_sum(layout,spikes).astype(np.int32))
            DN_EMA=state.rate[lane].numpy().copy()
            with torch.no_grad():
                predicted=F.linear(torch.from_numpy(DN_EMA)/reference,head)
            actual_raw=source['raw_logits'][t,w]
            err=float((predicted-actual_raw).abs().max());error_max=max(error_max,err)
            if not torch.allclose(predicted,actual_raw,atol=1e-4,rtol=1e-4):
                raise ValueError('Following DN/current raw head mismatch;record BEFORE Adam')
            probabilities=source['action_probs'][t,w];action=int(source['actions'][t,w]);legal=source['legal_actions'][t,w]
            if (not 0<=action<19 or not legal[action] or not torch.isclose(probabilities.sum(),probabilities.new_tensor(1.),atol=2e-6,rtol=0)
                    or bool((probabilities<0).any()) or bool((probabilities[~legal]!=0).any()) or probabilities[action]<=0
                    or not torch.isclose(probabilities[action].log(),source['old_log_probs'][t,w],atol=2e-5,rtol=2e-5)):
                raise ValueError('Actual19 probability/action/logp/legality mismatch')
            record={k:source[k][t,w].clone().numpy() for k in
                ('encoded_visual','features','raw_logits','action_probs','legal_actions','actions','old_log_probs')}
            record.update({k:x[t,w].clone().numpy() for k,x in optional.items()})
            raw_clean=actual_raw.clamp(-80.,80.).clone()
            if source['features'][t,w,4]<=0:raw_clean[18]=-torch.inf
            raw_probs=raw_clean.softmax(-1)
            trusted_danger=bool(task.trusted[t,w,0] and task.unprotected_mask[t,w])
            danger=(task.margins[t,w,0]<=0)&trusted_danger
            if trusted_danger and danger.any():
                danger_pairs.append((float(probabilities[:18][torch.from_numpy(danger)].sum()),
                    float(raw_probs[:18][torch.from_numpy(danger)].sum())))
            record.update(time=np.int64(t),world=np.int16(w),group=np.int16(g),lane=np.int16(lane),game_id=np.int64(game),
                native_episode=np.int64(source['native_episodes'][t,w]),selection_kind=np.asarray(selection),
                following_snapshot_time=np.int64(t+1),brain_rng_key=np.int64(state.rng_keys[lane]),
                noise_counter_before=np.int64(source['noise_counters'][t,w]),noise_counter_after=np.int64(state.noise_counter[lane]),
                episode_start=np.bool_(source['episode_starts'][t,w]),terminated=np.bool_(source['terminated'][t,w]),
                margins=task.margins[t,w].copy(),trusted=task.trusted[t,w].copy(),native_outcome=np.int8(task.outcomes[t,w]),
                unprotected_mask=np.bool_(task.unprotected_mask[t,w]),valid_mask=np.bool_(task.valid_mask[t,w]),
                raw_policy_probabilities=raw_probs.numpy(),policy_probability_change_from_raw=(probabilities-raw_probs).numpy(),
                population_spikes=pop_sum.astype(np.int64),population_mean_neural_Hz=(pop_sum/layout.population_sizes*1000/window_ms).astype(np.float32),
                population_active_fraction=(_population_sum(layout,counts>0)/layout.population_sizes).astype(np.float32),
                population_mean_membrane_mV=(_population_sum(layout,v)/layout.population_sizes).astype(np.float32),
                population_mean_synaptic_mV=(_population_sum(layout,syn)/layout.population_sizes).astype(np.float32),
                population_recent_delay_spikes=np.stack(recent),
                sampled_neural_Hz=(counts[layout.sampled_indices].astype(np.float32)*np.float32(1000/window_ms)),
                input_port_neural_Hz=counts[layout.input_indices].astype(np.float32)*np.float32(1000/window_ms),
                DN_neural_Hz=counts[layout.output_indices].astype(np.float32)*np.float32(1000/window_ms),
                DN_EMA_Hz=DN_EMA,DN_membrane_mV=v[layout.output_indices].copy(),DN_synaptic_mV=syn[layout.output_indices].copy())
            records.append(record)
            facts.append(dict(time=t,world=w,game_id=game,
                before_native_endpoint=_json_value(task.step_evidence[t]['before_native_endpoints'][w]),
                transition=_json_value(rollout.step_info[t][w])))
        with torch.no_grad():
            encoded=torch.from_numpy(np.stack([r['encoded_visual'] for r in records]))
            rates=torch.sigmoid(F.linear(encoded,weight,bias))*brain.max_input_rate_hz
            probability=rates*brain.dt_ms/1000.
        for i,record in enumerate(records):
            record['expected_external_drive_Hz']=rates[i].numpy().copy()
            record['expected_external_event_probability']=probability[i].numpy().copy()
        payload={k:np.stack([r[k] for r in records]) for k in records[0]}
        if any(not np.isfinite(a).all() for a in payload.values() if a.dtype.kind=='f'):
            raise ValueError('Nonfinite selected population trace')
        path=output/f'part-{len(parts):04}.npz';temporary=path.with_suffix('.npz.tmp')
        with temporary.open('xb') as stream:np.savez_compressed(stream,**payload)
        os.replace(temporary,path)
        parts.append(dict(file=path.name,rows=len(records),start=start,bytes=path.stat().st_size,sha256=_sha(path)))
    if tuple((p.data_ptr(),p._version) for p in model.parameters())!=versions:
        raise ValueError('Model parameters changed during read-only population tracing')
    result=dict(schema=SCHEMA,status='complete',policy_version=version,source_steps=steps,worlds=worlds,
        neural_batch_size=16,worker_groups=[list(g) for g in groups],source_module_sha256=_sha(__file__),
        alignment_helper_sha256=_sha(Path(__file__).with_name('fly_representation_trace.py')),
        graph_fingerprint=getattr(model.student,'graph_fingerprint',None),layout_identity_sha256=identity,
        populations_file='populations.json',populations_sha256=_sha(output/'populations.json'),
        annotation_provenance=layout.provenance,parts=parts,scene_facts=facts,max_scenes=max_scenes,
        candidate_rows=len(candidates),candidate_critical_rows=len(critical),excluded=dict(excluded),selected_rows=len(selected),
        selection_counts=dict(Counter(kind for _,kind in selected)),selection='half critical;remainder deterministic uniform hash,without replacement',
        input_ports=nin,middle_neurons=int(np.sum(layout.role_labels==1)),DN_outputs=ndn,
        populations=len(layout.populations),sampled_cells=len(layout.sampled_indices),geometry_horizons=[4,12,24],
        neural_window_ms=window_ms,neural_Hz_definition='actual window_spike_counts*1000/(100*.2ms)',
        input_external_scope='CPU FP32 expected Poisson injection rate;NOT realized input events or input-neuron firing',
        input_neural_scope='actual firing of the VP/input graph neurons in the same20ms window',
        DN_EMA_scope='original continuous descending EMA;different from20ms count rate',
        recent_spike_scope='only last9*.2ms delayed binary spikes;not full20ms temporal order',
        raw_head_max_abs_error=error_max,source_parameter_raw_sha256={
            'input_encoder.weight':_raw_sha(weight),'input_encoder.bias':_raw_sha(bias),'action_readout.weight':_raw_sha(head)},
        teacher_cost_available='d95_risk_costs' in optional,
        teacher_cost_scope='recorded continuous D95 proxy,query/valid masks mandatory;unqueried0 is not safe',
        danger_probability_comparison=(dict(rows=len(danger_pairs),actual_mean=float(np.mean(danger_pairs,axis=0)[0]),
            raw_mean=float(np.mean(danger_pairs,axis=0)[1]),delta_actual_minus_raw=float(np.mean(np.subtract(*np.asarray(danger_pairs).T))))
            if danger_pairs else None),
        geometry_is_counterfactual_proxy=True,population_means_prove_information=False,
        decoding_performed=False,causal_path_verified=False,full_brain_state_saved=False,
        synthetic_fixture=fixture,extra_brain_forwards=0,teacher_queries=0,optimizer_updates=0,
        model_or_state_moved=False,state_or_global_RNG_modified=False,elapsed_seconds=time.monotonic()-began)
    _atomic_json(output/'index.json',result)
    return result
