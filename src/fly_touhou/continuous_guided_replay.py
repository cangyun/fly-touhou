"""Direct-teacher expert data and genuinely aggregated raw student sequences."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .continuous_guided_dagger import direction_labels, SEQUENCE_LENGTH


def add_expert_sequences(replay, folder, rng, *, per_game=8, check_stop=lambda:None, progress=None,split_role='train'):
    from .thrl_full_teacher_data import FullTeacherDataset
    from .no_bomb_teacher_data import NoBombTeacherDataset, SCHEMA as NOBOMB_SCHEMA
    from .thrl_study_io import read_shared_json
    manifest=read_shared_json(Path(folder)/'demonstrations.json')
    no_bomb=manifest.get('schema')==NOBOMB_SCHEMA
    dataset=(NoBombTeacherDataset if no_bomb else FullTeacherDataset)(folder); split=dataset.split()
    if split_role not in ('train','heldout'):raise ValueError('Explicit whole-trajectory split required')
    added=[]
    try:
        for index in split[split_role]:
            record=dataset.records[index];reader=record['reader']
            span=reader.index.get('original_stage_row_ranges',{}).get('1') if no_bomb else None
            stop=int(span['stop']) if span is not None else min(4096,reader.rows)
            starts=np.arange(0,stop-SEQUENCE_LENGTH+1,16)
            if not len(starts):continue
            order=[0]+rng.permutation(starts[starts!=0]).tolist()
            accepted=0
            for start in order if no_bomb else order[:64]:
                check_stop()
                raw=reader.fetch(int(start),int(start)+SEQUENCE_LENGTH)
                # These are full-story teacher runs, often with Bombs. Only
                # untouched first-life/first-stage/no-Bomb prefixes are used;
                # the teacher's29/30 clears are NOT a NoBomb certificate.
                if (np.any(raw['stage']!=1) or np.any(raw['native_misses']!=0)
                        or np.any(raw['native_bombs_used']!=0) or np.any(raw['actions']>=18)):
                    continue
                data={k:raw[k] for k in ('features','maps','objects','actions','episode_starts')}
                data.update(rule_probabilities=raw['targets'],rule_weights=raw['teacher_guidance_weight'],
                    outcomes=raw.get('outcomes',np.full(SEQUENCE_LENGTH,-1,np.int8)))
                evidence=direction_labels(data)
                added.append(replay.add(data,origin='expert',round_index=-1,
                    metadata=dict(native_seed=record['native_seed'],original_start=int(start),
                        original_teacher='ResourceV10',labels=evidence,whole_game_split=split_role)))
                accepted+=1
                if accepted>=per_game:break
            dataset.close_caches()
            if progress:progress(dict(expert_games_processed=len({r['metadata']['native_seed'] for r in added}),
                expert_sequences=len(added)))
        if not added:raise RuntimeError('No original no-Bomb expert prefixes available')
        replay.commit()
        return dict(manifest_sha256=dataset.manifest_sha256,split=split,sequences=len(added),
            observations=len(added)*SEQUENCE_LENGTH,teacher='original ResourceV10;not D95',
            filtering='stage1/no previous or current Miss/no used Bombs/no Bomb actions',split_role=split_role,
            NoBomb_teacher_demonstrations=no_bomb,complete_first_stage_span_available=bool(no_bomb))
    finally:dataset.close_caches()


def add_initial_student_sequences(replay, trace_path, rng, *, limit=32, check_stop=lambda:None):
    """Read original training raw-map archives, never the frozen dev20 tails."""
    from .continuous_full_sensory_ppo import read_raw_map_member
    trace_path=Path(trace_path)
    archive=trace_path.with_name(trace_path.stem+'-raw-maps.zip')
    index=json.loads(archive.with_suffix('.zip.index.json').read_text(encoding='utf8'))
    lookup={(r['time_start'],r['world']):r for r in index['members']}
    with np.load(trace_path,allow_pickle=False) as source:
        fields={k:source[k] for k in ('features','objects','actions','episode_starts','rule_probabilities',
            'rule_weights','task_outcomes','first_observed_contact')}
    contacts=fields['first_observed_contact'];candidates=[]
    for first in range(0,len(contacts)-SEQUENCE_LENGTH+1,16):
        for world in range(contacts.shape[1]):
            end=first+SEQUENCE_LENGTH
            # Include entire 64-frame learning window plus64-frame burn-in.
            weight=int(contacts[first:end,world].any())
            candidates.append((weight,float(rng.random()),first,world))
    count=0
    for _,_,first,world in sorted(candidates,reverse=True)[:limit]:
        check_stop();end=first+SEQUENCE_LENGTH
        maps=np.concatenate([read_raw_map_member(archive,lookup[(t,world)]) for t in (first,first+16)])
        data={k:v[first:end,world].copy() for k,v in fields.items() if k not in ('task_outcomes','first_observed_contact')}
        data.update(maps=maps,outcomes=fields['task_outcomes'][first:end,world].copy())
        evidence=direction_labels(data)
        replay.add(data,origin='student',round_index=-1,hard=True,
            metadata=dict(source_training_trace=str(trace_path),time_start=first,world=world,labels=evidence))
        count+=1
    replay.commit();return dict(sequences=count,source=str(trace_path),development_data_used=False)


def add_student_rollout(replay, rollout, rng, *, round_index, recent_limit=48, hard_limit=48, check_stop=lambda:None):
    n,worlds=rollout.actions.shape
    candidates=[]
    for first in range(0,n-SEQUENCE_LENGTH+1,16):
        for world in range(worlds):
            end=first+SEQUENCE_LENGTH
            contact=bool(rollout.first_observed_contact[first:end,world].any())
            trust=np.asarray(rollout.task_targets.trusted[first:end,world,0])
            margins=np.asarray(rollout.task_targets.margins[first:end,world,0])
            predicted=rollout.predicted_costs[first:end,world].argmin(-1).numpy()
            fail=(margins[np.arange(SEQUENCE_LENGTH),predicted]<=0)&trust
            score=100*int(contact)+10*int(fail.sum())
            candidates.append((first,world,score,contact))
    hard_order=sorted(range(len(candidates)),key=lambda i:(candidates[i][2],float(rng.random())),reverse=True)
    hard_ids=hard_order[:hard_limit]
    remaining=[i for i in range(len(candidates)) if i not in set(hard_ids)]
    recent_ids=rng.choice(remaining,size=min(recent_limit,len(remaining)),replace=False).tolist()
    counts=dict(recent=0,hard=0,native_contact_sequences=0)
    for i in hard_ids+recent_ids:
        check_stop();first,world,score,contact=candidates[i];end=first+SEQUENCE_LENGTH
        data={k:getattr(rollout,k)[first:end,world].numpy().copy() for k in
            ('features','objects','actions','episode_starts','rule_probabilities','rule_weights')}
        data['maps']=rollout.raw_maps[first:end,world].numpy().copy()
        data['outcomes']=np.asarray(rollout.task_targets.outcomes[first:end,world]).copy()
        evidence=direction_labels(data)
        hard=i in hard_ids
        replay.add(data,origin='student',round_index=round_index,hard=hard,
            metadata=dict(time_start=first,world=world,hard_score=score,native_contact=contact,labels=evidence))
        counts['hard' if hard else 'recent']+=1;counts['native_contact_sequences']+=int(contact)
    replay.commit()
    return dict(counts,total_persistent_sequences=len(replay.records),
        replay_used_for_training=True,stored_DN_used_for_training=False,
        archive='lossless original FP32 observations; labels recomputed once on insertion')
