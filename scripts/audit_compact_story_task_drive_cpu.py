"""Small paired full-LIF task-port audit on archived real block states only."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def load(path):
    import torch
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    with _shared_binary(path) as stream:return torch.load(stream,map_location='cpu',weights_only=False)


def distributions(old,new):
    import torch
    p=old.softmax(-1);q=new.softmax(-1)
    positive=p>0
    kl=(p[positive]*(p[positive].log()-q[positive].clamp_min(1e-35).log())).sum()
    finite=torch.isfinite(old)&torch.isfinite(new)
    return dict(KL_old_to_new=float(kl),total_variation=float((p-q).abs().sum()/2),
        raw_logits_max_error=float((old[finite]-new[finite]).abs().max()),
        legal_support_equal=bool(torch.equal(torch.isfinite(old),torch.isfinite(new))))


def choose(tape,count):
    import torch
    f=tape['features'];pool=[]
    for t in sorted(tape['block_states']):
        if t==0:continue
        for w in range(f.shape[1]):
            x=f[t,w]
            if (x[14]>.5 and x[113]>.5 and x[117]>.5 and x[114].abs()*368>32
                    and tape['rule_weights'][t,w]>0 and tape['trusted'][t,w,0]):
                pool.append((t,w))
    if not pool:raise ValueError('No teacher-labelled boss scene at a real noninitial block boundary')
    # Prefer a broad safe alignment case, then another world with a tighter
    # future risk pattern. No score is selected using the new-port result.
    safe=[q for q in pool if bool((tape['margins'][q[0],q[1],0]>1).all())]
    first=(safe or pool)[len(safe or pool)//2]
    chosen=[first]
    alternatives=[q for q in pool if q[1]!=first[1]]
    if count>1 and alternatives:
        second=min(alternatives,key=lambda q:float(tape['margins'][q[0],q[1],1:].min()))
        chosen.append(second)
    return chosen


def apply_configuration(model,payload):
    for key in ('temperature','bombs_disabled','phase_valid_control','protection_aware_risk',
            'anticipate_point_arming','forecast_growth_motion','native_motion_forecast','native_focus_gate',
            'protected_exit_forecast','primary_attack_target','risk_policy_gradient',
            'native_contact_risk_calibration','neural_risk_scale','idle_prior_scale',
            'full_story_action_selection','soft_risk_composition'):
        if key in payload:setattr(model,key,payload[key])
    model.pad_candidate_batch=False
    model.core.sparse.deterministic_spmm=False


def run(args):
    import psutil
    p=psutil.Process();p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);p.cpu_affinity(p.cpu_affinity()[-2:])
    import torch
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    from scipy.stats import spearmanr
    from fly_touhou.connectome import load_connectome
    import fly_touhou.continuous_connectome as cc
    import fly_touhou.compact_story_model as sm
    from fly_touhou.candidate_spiking_model import CandidateCore,CandidatePolicy
    from fly_touhou.compact_story_task_drive import task_progress_cues
    from fly_touhou.spiking_engineered_control import SpikingState
    from fly_touhou.compact_story_behavior import story_behavior_logits,rollout_sampling_temperature,rollout_bomb_sampling_temperature
    torch.set_num_threads(2);torch.set_num_interop_threads(1);pa.set_cpu_count(2)
    started=time.monotonic();args.output.parent.mkdir(parents=True,exist_ok=True)
    status=args.output.with_name(args.output.stem+'-status.json')
    def progress(phase,**extra):
        row=dict(phase=phase,pid=os.getpid(),seconds=time.monotonic()-started,**extra)
        save(status,row);print(json.dumps(row),flush=True)
    progress('loading_checkpoint_and_rollout')
    checkpoint_path=args.source/'latest-parameters.pt';payload=load(checkpoint_path)
    tape=load(args.source/'rollout-027.pt');chosen=choose(tape,args.scenes)
    features=torch.stack([tape['features'][t,w] for t,w in chosen])
    states=SpikingState.pack([tape['block_states'][t].row(w).reset(tape['episode_starts'][t,w].reshape(1)) for t,w in chosen])
    records=[dict(time=t,world=w,teacher=tape['rule_probabilities'][t,w].clone(),
        margins=tape['margins'][t,w].clone(),trusted=tape['trusted'][t,w].clone(),
        recorded_policy=tape['recorded_policy_logits'][t,w].clone(),
        recorded_behavior=tape.get('recorded_behavior_policy_logits',tape['recorded_policy_logits'])[t,w].clone(),
        sampled_action=int(tape['actions'][t,w]),old_selected_log_probability=float(tape['old_log_probs'][t,w])) for t,w in chosen]
    metadata=tape['metadata'];del tape
    progress('loading_full_graph',scenes=chosen)
    graph=load_connectome(args.connectome,max_neurons=None,verify=False)
    table=pd.read_csv(args.connectome/'annotations/flywire_annotations_v2.1.0.tsv',sep='\t',dtype=str,
        keep_default_na=False,usecols=['root_id','super_class'])
    lookup={root:i for i,root in enumerate(graph.node_ids)}
    inputs=tuple(sorted(lookup[root] for root in table.loc[table.super_class=='visual_projection','root_id'] if root in lookup))
    manifest=json.loads((args.connectome/'annotations/selected_ports_1024_v783.json').read_text(encoding='utf-8'))
    outputs=tuple(lookup[root] for root in manifest['output_root_ids'])
    del table,lookup,manifest
    # Process-local constructor bypass requested for this offline diagnostic;
    # no on-disk code, graph identity, topology, or checkpoint weight is edited.
    cc.graph_fingerprint=lambda graph:cc.PINNED_GRAPH_FINGERPRINT
    provenance=dict(annotation_sha256=cc.ANNOTATIONS_SHA256,graph_matched_visual_ports=len(inputs))
    def builder(directory,*,seed,device):
        core=CandidateCore(graph,inputs,outputs,K=16,trainable_brain=True,require_full_graph=True,
            port_provenance=provenance,edge_chunk_size=262144,seed=seed)
        return CandidatePolicy(core,seed=seed).to(device)
    sm.build_compact_model=builder
    progress('building_full_LIF')
    model=sm.from_story_checkpoint(args.connectome,payload,device='cpu').eval()
    apply_configuration(model,payload);model.core.input_encoder.task_progress=None
    model.task_progress_ports=False
    neural=sm.story_candidates(features);closing=task_progress_cues(neural)[...,0]>1e-6
    with torch.no_grad():
        base_ports=model.core.input_encoder(neural.reshape(-1,47)).reshape(len(features),18,-1)
        progress('baseline_LIF_forward')
        before,_,_=model.step(features,states,return_all_DN=True,actions=torch.zeros(len(features),dtype=torch.long))
        model.enable_task_progress_ports(amplitude=args.amplitude)
        new_ports=model.core.input_encoder(neural.reshape(-1,47)).reshape(len(features),18,-1)
        progress('task_ports_LIF_forward')
        after,_,_=model.step(features,states,return_all_DN=True,actions=torch.zeros(len(features),dtype=torch.long))
    scenarios=[]
    movement_temp=rollout_sampling_temperature(dict(metadata=metadata));bomb_temp=rollout_bomb_sampling_temperature(dict(metadata=metadata))
    baseline_behavior=story_behavior_logits(before['logits'],movement_temp,bomb_temp)
    altered_behavior=story_behavior_logits(after['logits'],movement_temp,bomb_temp)
    for i,record in enumerate(records):
        mask=closing[i];dn0=before['candidate_DN'][i];dn1=after['candidate_DN'][i]
        s0=before['horizon_scores'][i];s1=after['horizon_scores'][i]
        correlations=[]
        for h in range(3):
            corr=spearmanr(s0[:,h].numpy(),s1[:,h].numpy()).statistic
            correlations.append(float(corr) if np.isfinite(corr) else None)
        def policy_metrics(logits,behavior):
            return dict(joint19_closing_probability=float(logits.softmax(-1)[:18][mask].sum()),
                movement_conditional_closing_probability=float(logits[:18].softmax(-1)[mask].sum()),
                behavior_joint19_closing_probability=float(behavior.softmax(-1)[:18][mask].sum()),
                argmax_action=int(logits.argmax()),probabilities19=logits.softmax(-1).tolist())
        scenarios.append(dict(time=record['time'],world=record['world'],stage=int(round(float(features[i,111]*6))),
            target_dx_pixels=float(features[i,114]*368),trusted_horizons=record['trusted'].tolist(),
            min_actual_forecast_margins=record['margins'].min(-1).values.tolist(),
            teacher_closing_probability=float(record['teacher'][:18][mask].sum()),
            baseline=policy_metrics(before['logits'][i],baseline_behavior[i]),
            task_ports=policy_metrics(after['logits'][i],altered_behavior[i]),
            VP_mean_abs_change=float((new_ports[i]-base_ports[i]).abs().mean()),
            DN_mean_abs_change=float((dn1-dn0).abs().mean()),DN_max_abs_change=float((dn1-dn0).abs().max()),
            DN_relative_L2=float((dn1-dn0).norm()/dn0.norm().clamp_min(1e-8)),
            DN_changed_fraction=float(((dn1-dn0).abs()>1e-8).float().mean()),
            risk_rank_spearman_4_12_24=correlations,
            risk_probability_mean_abs_change=float(((8*s1-1).sigmoid()-(8*s0-1).sigmoid()).abs().mean()),
            risk_safety_threshold_flips=int((((8*s1-1).sigmoid()<.2)!=((8*s0-1).sigmoid()<.2)).sum()),
            recorded_policy_to_current=distributions(record['recorded_policy'],before['logits'][i]),
            recorded_behavior_to_current=distributions(record['recorded_behavior'],baseline_behavior[i]),
            recorded_selected_action=record['sampled_action'],
            selected_log_probability_change=float(baseline_behavior[i].log_softmax(-1)[record['sampled_action']]-record['old_selected_log_probability'])))
    result=dict(schema='paired-real-full-LIF-task-drive-CPU-v1',checkpoint=str(checkpoint_path),
        checkpoint_costs=payload['costs'],rollout='rollout-027.pt',rollout_metadata=metadata,
        full_graph=dict(nodes=graph.n_neurons,edges=graph.n_edges,VP_ports=len(inputs),DN_ports=len(outputs),K=model.core.K),
        device='cpu',cpu_affinity=p.cpu_affinity(),cpu_threads=2,priority='BELOW_NORMAL',
        hash_checks_performed=False,constructor_fingerprint_skipped_in_this_process=True,
        amplitude=args.amplitude,scenarios=scenarios,seconds=time.monotonic()-started,
        optimizer_updates=0,native_game_actions=0,
        scope='Paired one-decision forward sensitivity at real archived LIF states; no claim of learned alignment or improved game return',
        replay_scope='Current checkpoint has undergone updates since rollout collection; CPU CSR and unpadded18-candidate execution differ from production CUDA execution')
    save(args.output,result);progress('main_report_written')
    if args.behavior_replay and time.monotonic()-started<150:
        progress('loading_initial_policy_replay')
        initial=load(args.source/'initial-parameters.pt');old=load(args.source/'rollout-023.pt')
        t=next(t for t in sorted(old['block_states']) if t>0);w=0
        x=old['features'][t,w:w+1];state=old['block_states'][t].row(w).reset(old['episode_starts'][t,w:w+1])
        recorded=old.get('recorded_behavior_policy_logits',old['recorded_policy_logits'])[t,w].clone()
        old_metadata=old['metadata'];del old
        model.core.input_encoder.task_progress=None;model.task_progress_ports=False
        from fly_touhou.dn_residual_risk import load_head
        state_dict={k.removeprefix('risk.'):v for k,v in initial['state_dict'].items() if k.startswith('risk.')}
        model.risk=load_head(dict(schema=initial['risk_readout_kind'],state_dict=state_dict,
            width=initial['risk_readout_width'],input_soft_clip=initial.get('risk_input_soft_clip'),
            base_input_soft_clip=initial.get('risk_base_input_soft_clip'),baseline_trainable=initial.get('trainable_risk_baseline',False)),device='cpu')
        model.load_state_dict(initial['state_dict'],strict=True);apply_configuration(model,initial)
        progress('initial_policy_LIF_replay',time=t,world=w)
        with torch.no_grad():replayed,_,_=model.step(x,state,return_all_DN=True,actions=torch.zeros(1,dtype=torch.long))
        behavior=story_behavior_logits(replayed['logits'],old_metadata['movement_sampling_temperature'],old_metadata['bomb_sampling_temperature'])[0]
        result['collection_policy_replay']=dict(checkpoint='initial-parameters.pt',rollout='rollout-023.pt',time=t,world=w,
            initial_costs=initial['costs'],comparison=distributions(recorded,behavior),
            scope='One noninitial block, original collection parameters and exact stored FP32 state; CPU/CUDA numerical backend caveat applies')
        result['seconds']=time.monotonic()-started;save(args.output,result)
    elif args.behavior_replay:
        result['collection_policy_replay']=dict(skipped='main CPU audit exceeded150 seconds; no extra forward')
        save(args.output,result)
    progress('complete',report=str(args.output))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--connectome',type=Path,default=ROOT/'data/connectome')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--scenes',type=int,default=2)
    parser.add_argument('--amplitude',type=float,default=.025);parser.add_argument('--behavior-replay',action='store_true')
    arguments=parser.parse_args()
    try:run(arguments)
    except BaseException as error:
        save(arguments.output.with_name(arguments.output.stem+'-failure.json'),dict(type=type(error).__name__,message=str(error)))
        raise
