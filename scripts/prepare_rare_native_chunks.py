"""Seal real training-only deathbomb chunks and nearby control chunks; no learning."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

PIN=Path('./outputs/legacy-build/source_snapshots/20261019-v4-63')
sys.path.insert(0,str(PIN/'src'))
import numpy as np
import torch
from fly_touhou.experiment import _file_lock,file_sha256
from fly_touhou.geometry_action_feedback import GeometryActionFeedback,balance_target_weights
from fly_touhou.geometry_migration import verify_previous_geometry
from fly_touhou.geometry_observation import GeometryObservationEncoder
from fly_touhou.state import Snapshot
from fly_touhou.whole_brain_ppo import WholeBrainRollout,load_rollout,_native_observation


def positive(row):
    s=row['snapshot'];p=s['player']
    return p is not None and s['bombs']>0 and p['state']==4 and p['deathbomb_timer']>0


def world_index(row):
    match=re.search(r':world([01]):',row['episode_id'])
    if match is None:raise ValueError('Exact independent original-world identity required')
    return int(match.group(1))*32+row['rollout_step']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--previous-source',type=Path,required=True)
    p.add_argument('--first-updates',type=int,default=32)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    assert not a.output.exists() and a.first_updates>0
    manifest_path=a.run/'manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    config=manifest['config']
    assert config['training_kind']=='native-reward-complete-brain-recurrent-PPO'
    assert config['teacher_action_calls']==0 and config['safety_takeovers']==0
    assert config['campaign_runtime']['parallel_collection']['worlds']==2
    assert config['model_config']['require_full_graph'] is True
    assert config['encoder']['schema']=='native-collision-geometry-observation-v4'
    assert json.loads((a.run/'rootstatus.json').read_text(encoding='utf-8'))['phase']=='finished'
    with _file_lock(a.run/'.events.lock'):
        lines=(a.run/'events.jsonl').read_text(encoding='utf-8').splitlines()
    events=[json.loads(x) for x in lines]
    ends=[i for i,x in enumerate(events) if x['event']=='checkpoint' and x['payload']['tag']=='ppo-latest']
    assert len(ends)>=a.first_updates
    end=ends[a.first_updates-1]+1;events=events[:end]
    checkpoints={e['payload']['epoch']:e['payload'] for e in events if e['event']=='checkpoint' and e['payload']['tag']=='ppo-latest'}
    seals={Path(e['payload']['path']).name.split('.')[0]:e['payload'] for e in events if e['event']=='ppo_behavior_rollout_sealed'}
    positives=[]
    for epoch,seal in sorted(seals.items()):
        native=seal['native_trajectory'];assert file_sha256(native['path'])==native['sha256']
        with gzip.open(native['path'],'rb') as stream:
            for number,raw in enumerate(stream,1):
                row=json.loads(raw)
                if row.get('record_type')=='transition' and positive(row):
                    assert row['snapshot']['screen']==7 and row['snapshot']['pause']==2
                    assert row['snapshot']['extra']['native_dialogue_object']==0
                    positives.append({'epoch':int(epoch),'index':world_index(row),'episode_id':row['episode_id'],
                        'stage':row['snapshot']['stage'],'stage_frame':row['snapshot']['stage_frame'],
                        'timer':row['snapshot']['player']['deathbomb_timer'],'line':number,
                        'raw_line_sha256':hashlib.sha256(raw).hexdigest()})
    assert positives,'No actual stocked deathbomb before-states in this sealed TRAINING prefix'
    a.output.mkdir(parents=True,exist_ok=False)
    prefix=a.output/'training.events_prefix.jsonl'
    prefix.write_text('\n'.join(lines[:end])+'\n',encoding='utf-8')
    encoder=GeometryObservationEncoder.from_specification(config['encoder']);feedback=GeometryActionFeedback(frame_skip=4)
    chunks=[];sources=[];raw_sources={};case_counts=Counter()
    for epoch in sorted({x['epoch'] for x in positives}):
        seal=seals[f'{epoch:06d}'];assert file_sha256(seal['path'])==seal['sha256']
        saved=load_rollout(seal['path'])
        rollout=WholeBrainRollout(**{k:(saved[k].numpy() if torch.is_tensor(saved[k]) else saved[k])
            for k in WholeBrainRollout.__dataclass_fields__})
        assert len(rollout)==64 and rollout.independent_boundaries==(32,)
        parent=checkpoints.get(epoch-1,{'path':config['initial_checkpoint_path'],'sha256':config['initial_checkpoint_sha256']})
        assert file_sha256(parent['path'])==parent['sha256']
        before=torch.load(parent['path'],map_location='cpu',weights_only=True)
        old=verify_previous_geometry(before,a.previous_source)
        assert before['epoch']==epoch-1 and before['neurons']==138639 and before['edge_records']==15091983
        assert before['graph_fingerprint']==saved['graph_fingerprint']==config['graph_fingerprint']
        assert all(torch.equal(v,before['trainable_parameters'][name]) for name,v in saved['behavior_parameters'].items())
        native=seal['native_trajectory'];rows={};raws={};line_numbers={};targets={}
        with gzip.open(native['path'],'rb') as stream:
            for number,raw in enumerate(stream,1):
                row=json.loads(raw)
                if row.get('record_type')!='transition':continue
                i=world_index(row);assert i not in rows
                assert row['episode_id']==rollout.episode_ids[i] and row['action']==int(rollout.actions[i])
                snapshot=Snapshot.from_dict(row['snapshot'])
                _native_observation(rollout.observations[i],encoder,snapshot,{'exe_sha256':config['game_exe_sha256']})
                assert np.array_equal(encoder.encode(snapshot),rollout.observations[i])
                targets[i]=feedback.target(snapshot);rows[i]=row;raws[i]=raw;line_numbers[i]=number
        assert set(rows)==set(range(64))
        segments=rollout.segments();chosen={}
        for event in [x for x in positives if x['epoch']==epoch]:
            span=next((a,b) for a,b in segments if a<=event['index']<b)
            chosen[span]='real_deathbomb'
            assert targets[event['index']][2]['reason']=='measured_deathbomb'
            assert hashlib.sha256(raws[event['index']]).hexdigest()==event['raw_line_sha256']
        for start,stop in list(chosen):
            for kind in ('immune_through_next_observation','positive_clearance_movement'):
                candidates=[]
                for first,last in segments:
                    if any(positive(rows[i]) for i in range(first,last)):continue
                    if any(rows[i]['episode_id']!=rows[start]['episode_id'] for i in range(first,last)):continue
                    if any(targets[i][2]['reason']=='predicted_no_constant_escape' for i in range(first,last)):continue
                    if any(targets[i][2]['reason']==kind for i in range(first,last)):
                        candidates.append((abs(first-start),first,last))
                if candidates:
                    _,first,last=min(candidates);chosen.setdefault((first,last),'matched_'+kind)
        for (first,last),kind in sorted(chosen.items()):
            episode=rollout.episode_ids[first];world_start=0 if first<32 else 32
            burn_start=min(k for k in rollout.initial_states if max(world_start,first-16)<=k<=first
                and rollout.episode_ids[k]==episode)
            assert all(rollout.episode_ids[i]==episode for i in range(burn_start,last))
            initial=deepcopy(rollout.initial_states[burn_start])
            assert initial['membrane'].shape==(1,138639)
            assert int(initial['noise_counter'][0])==int(rollout.noise_counters[burn_start])
            assert int(initial['rng_keys'][0])==int(rollout.rng_keys[burn_start])
            masks=np.stack([targets[i][0] for i in range(first,last)])
            confidence=np.asarray([targets[i][1] for i in range(first,last)],dtype=np.float64)
            row_sources=[]
            for i in range(burn_start,last):
                identity=(native['path'],line_numbers[i]);raw_sources[identity]=raws[i]
                row_sources.append({'index_in_original_rollout':i,'source_line':line_numbers[i],
                    'raw_line_sha256':hashlib.sha256(raws[i]).hexdigest()})
            chunks.append({'id':f'{epoch:06d}:{first}:{last}','selection':kind,'episode_id':episode,
                'epoch':epoch,'source_span':[first,last],'burnin_span':[burn_start,first],
                'observations':torch.as_tensor(rollout.observations[first:last]).clone(),
                'burnin_observations':torch.as_tensor(rollout.observations[burn_start:first]).clone(),
                'initial_state':initial,'original_target_initial_state':deepcopy(rollout.initial_states[first]),
                'rng_keys':torch.as_tensor(rollout.rng_keys[burn_start:last]).clone(),
                'noise_counters':torch.as_tensor(rollout.noise_counters[burn_start:last]).clone(),
                'next_noise_counters':torch.as_tensor(rollout.next_noise_counters[burn_start:last]).clone(),
                'allowed':torch.as_tensor(masks),'confidence':torch.as_tensor(confidence),
                'balanced_chunk_weights':torch.as_tensor(balance_target_weights(masks,confidence)),
                'target_details':[targets[i][2] for i in range(first,last)],'source_rows':row_sources,
                'native_source':native,'behavior_rollout':seal,'behavior_checkpoint':parent,
                'original_behavior_state_is_current_actor_state':False})
            case_counts[kind]+=1
        sources.append({'epoch':epoch,'rollout':seal,'behavior_checkpoint':parent,
            'previous_source':old['source'],'previous_manifest_sha256':old['manifest_sha256']})
        del before,saved,rollout
    # Archive the complete immutable target implementation, rather than a mutable worktree pointer.
    source_manifest=json.loads((PIN/'snapshot_manifest.json').read_text(encoding='utf-8'))
    for entry in source_manifest['files']:
        content=(PIN/entry['path']).read_bytes();assert hashlib.sha256(content).hexdigest()==entry['sha256']
        out=a.output/'source_archive'/entry['path'];out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(content)
    (a.output/'source_archive/snapshot_manifest.json').write_bytes((PIN/'snapshot_manifest.json').read_bytes())
    builder=a.output/'source_archive/prepare_rare_native_chunks.py';builder.write_bytes(Path(__file__).read_bytes())
    raw_file=a.output/'original_transition_rows.jsonl.gz'
    with gzip.open(raw_file,'xb',compresslevel=3) as out:
        for identity,raw in sorted(raw_sources.items()):out.write(raw)
    dataset={'schema':'sealed-native-rare-action-chunks-v1','model_config':config['model_config'],
        'encoder':config['encoder'],'graph_fingerprint':config['graph_fingerprint'],
        'target_specification':feedback.specification(),'chunks':chunks,
        'scope':'training-only off-policy supervised candidate; current actor must burn in separately; never inject these states into a live world',
        'native_actions_executed_by_builder':0,'training_performed':False}
    path=a.output/'chunks.pt'
    with path.open('xb') as out:torch.save(dataset,out)
    restored=torch.load(path,map_location='cpu',weights_only=True)
    assert len(restored['chunks'])==len(chunks)
    for x,y in zip(restored['chunks'],chunks):
        assert torch.equal(x['observations'],y['observations']) and torch.equal(x['allowed'],y['allowed'])
    report={'schema':'sealed-rare-native-chunks-build-report-v1','status':'passed',
        'training_run':str(a.run.resolve()),'training_manifest_sha256':file_sha256(manifest_path),
        'event_prefix_sha256':file_sha256(prefix),'target_source_manifest_sha256':file_sha256(PIN/'snapshot_manifest.json'),
        'builder_sha256':file_sha256(builder),'dataset':{'path':str(path.resolve()),'sha256':file_sha256(path),'bytes':path.stat().st_size},
        'raw_rows':{'path':str(raw_file.resolve()),'sha256':file_sha256(raw_file),'unique_original_rows':len(raw_sources)},
        'sources':sources,'positive_observation_steps':positives,'chunk_selection':dict(case_counts),
        'target_reason_counts':dict(Counter(d['reason'] for c in chunks for d in c['target_details'])),
        'all_raw_observations_recomputed_bitwise_equal':True,'all_behavior_actor_parameters_match_preceding_checkpoint':True,
        'independent_evaluation_data_used':False,'numeric_fixture_used':False,'future_outcomes_used_for_targets':False,
        'candidate_dataset_only':True,'training_performed':False,'live_state_modified':False}
    with (a.output/'report.json').open('x',encoding='utf-8') as out:json.dump(report,out,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':'passed','dataset':report['dataset'],'positive_steps':len(positives),
        'chunk_selection':report['chunk_selection'],'target_counts':report['target_reason_counts']},ensure_ascii=False))


if __name__=='__main__':main()
