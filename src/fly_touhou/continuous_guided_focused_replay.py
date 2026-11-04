"""Prepared correction: danger must fall in the learning half of a sequence.

Old replay files stay intact. Native contact anchors end on the actual chosen
hit, keeping its preceding128 original frames instead of wasting most of the
learning window on the following game's opening.
"""
from __future__ import annotations
import numpy as np
from .continuous_guided_dagger import SequenceReplay,SEQUENCE_LENGTH,BURN_IN,direction_labels

SCHEMA='native-contact-anchored-raw-replay-v2'
RAW_FIELDS=('features','objects','actions','episode_starts','rule_probabilities','rule_weights')


def learn_window_evidence(data):
    y=np.asarray(data['outcomes']);trusted=np.asarray(data['trusted'])[BURN_IN:,0]
    margin=np.asarray(data['margins'])[BURN_IN:,0,:]
    safe=margin>1.
    mixed=trusted&safe.any(-1)&~safe.all(-1)
    return dict(trained_contacts=int((y[BURN_IN:]==1).sum()),
        burn_in_contacts=int((y[:BURN_IN]==1).sum()),mixed_training_rows=int(mixed.sum()))


class FocusedSequenceReplay(SequenceReplay):
    def measure(self,record):
        evidence=record.setdefault('metadata',{}).get('learning_window')
        if evidence is None:
            with np.load(record['path'],allow_pickle=False) as data:
                evidence=learn_window_evidence(data)
            record['metadata']['learning_window']=evidence
        return evidence

    def sample(self,rng,*,current_round,warmup=False,batch=16):
        expert=[r for r in self.records if r['origin']=='expert']
        recent=[r for r in self.records if r['origin']=='student' and r['round']==current_round]
        hard=[r for r in self.records if r['origin']=='student' and r['hard'] and
            (self.measure(r)['trained_contacts']>0 or self.measure(r)['mixed_training_rows']>=8)]
        pools=dict(expert=expert,recent=recent,historical_hard=hard)
        counts={k:0 for k in pools};chosen=[]
        for _ in range(batch):
            key='expert' if warmup else rng.choice(['expert','recent','historical_hard'],p=[.2,.4,.4])
            if not pools[key]:key='recent' if recent else 'expert'
            pool=pools[key]
            if key=='historical_hard':
                weights=np.asarray([3. if self.measure(r)['trained_contacts']>0 else 1. for r in pool])
                record=pool[int(rng.choice(len(pool),p=weights/weights.sum()))]
            else:record=pool[int(rng.integers(len(pool)))]
            chosen.append(record);counts[key]+=1
        arrays=[]
        for record in chosen:
            arrays.append(self.load_record(record))
        return {k:np.stack([a[k] for a in arrays],axis=1) for k in arrays[0]},counts

    def load_record(self,record):
        with np.load(record['path'],allow_pickle=False) as data:
            return {k:data[k] for k in data.files}


def raw_tail(rollout):
    result={k:getattr(rollout,k)[-31:].numpy().copy() for k in RAW_FIELDS+('game_ids','episode_steps')}
    result['maps']=rollout.raw_maps[-31:].numpy().copy()
    result['outcomes']=np.asarray(rollout.task_targets.outcomes[-31:]).copy()
    return result


def contact_window(rollout,end,world,previous=None):
    first=end-SEQUENCE_LENGTH
    values={k:getattr(rollout,k).numpy() for k in RAW_FIELDS}
    values.update(maps=rollout.raw_maps.numpy(),outcomes=np.asarray(rollout.task_targets.outcomes))
    if first<0:
        if (previous is None or int(previous['game_ids'][-1,world])!=int(rollout.game_ids[0,world])
                or int(previous['episode_steps'][-1,world])+1!=int(rollout.episode_steps[0,world])):
            return None
        data={k:np.concatenate((previous[k][first:,world],v[:end,world]),axis=0) for k,v in values.items()}
    else:data={k:v[first:end,world].copy() for k,v in values.items()}
    if len(data['features'])!=SEQUENCE_LENGTH:return None
    return data


def add_focused_student_rollout(replay,rollout,rng,*,round_index,previous=None,
        recent_limit=48,hard_limit=48,check_stop=lambda:None):
    contacts=np.asarray(rollout.first_observed_contact);n,worlds=contacts.shape
    hits=[(int(t)+1,int(w)) for t,w in zip(*np.nonzero(contacts))]
    rng.shuffle(hits);selected=[];used=set();skipped=0
    for end,world in hits:
        data=contact_window(rollout,end,world,previous)
        if data is None or np.any(data['episode_starts'][1:]):skipped+=1;continue
        selected.append((end,world,True,data));used.add((end,world))
        if len(selected)>=hard_limit:break
    candidates=[]
    for first in range(0,n-SEQUENCE_LENGTH+1,16):
        end=first+SEQUENCE_LENGTH
        for world in range(worlds):
            if (end,world) in used:continue
            m=np.asarray(rollout.task_targets.margins[first+16:end,world,0,:])
            trust=np.asarray(rollout.task_targets.trusted[first+16:end,world,0])
            chosen=rollout.predicted_costs[first+16:end,world].argmin(-1).numpy()
            fail=int(((m[np.arange(16),chosen]<=0)&trust).sum())
            candidates.append((fail,float(rng.random()),end,world))
    candidates.sort(reverse=True)
    for fail,_,end,world in candidates:
        if len(selected)>=hard_limit:break
        if fail<=0:continue
        data=contact_window(rollout,end,world)
        selected.append((end,world,True,data));used.add((end,world))
    remaining=[c for c in candidates if (c[2],c[3]) not in used]
    for i in rng.choice(len(remaining),size=min(recent_limit,len(remaining)),replace=False):
        _,_,end,world=remaining[int(i)];selected.append((end,world,False,contact_window(rollout,end,world)))
    count=dict(hard=0,recent=0,contact_anchored=0,trained_contacts=0,burn_in_contacts=0)
    for end,world,hard,data in selected:
        check_stop();labels=direction_labels(data);evidence=learn_window_evidence(data)
        replay.add(data,origin='student',round_index=round_index,hard=hard,
            metadata=dict(window_end=end,world=world,learning_window=evidence,labels=labels,
                contact_at_final_training_row=bool(data['outcomes'][-1]==1),schema=SCHEMA))
        count['hard' if hard else 'recent']+=1
        count['contact_anchored']+=int(data['outcomes'][-1]==1)
        for key in ('trained_contacts','burn_in_contacts'):count[key]+=evidence[key]
    replay.commit()
    return dict(count,schema=SCHEMA,skipped_missing_real_prefix=skipped,total_persistent_sequences=len(replay.records),
        replay_used_for_training=True,stored_DN_used_for_training=False),raw_tail(rollout)
