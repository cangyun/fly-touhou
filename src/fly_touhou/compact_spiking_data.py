"""Small chronological compact111 training caches; original sources untouched."""
from collections import deque
from pathlib import Path
import copy
import numpy as np
from .thrl_study_io import read_shared_json as read
from .thrl_durable_records import write_json


def prepare(expert,own,output,*,rows=256,on_progress=lambda x:None):
    from .no_bomb_teacher_data import NoBombTeacherDataset
    from .chronological_avoidance_data import TeacherRows,FixedStudentTrainingRows
    from .compact_spiking_sensation import encode_geometry,SCHEMA
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    dataset=NoBombTeacherDataset(expert);sources=dict(teacher=TeacherRows(dataset),historical=FixedStudentTrainingRows(own))
    split=dataset.split();order=np.random.default_rng(202610307).permutation(30).tolist()
    selection=[('teacher','train',i) for i in split['train'][:4]]+[('teacher','heldout',i) for i in split['heldout'][:2]]
    selection += [('historical','train',i) for i in order[6:18]]+[('historical','heldout',i) for i in order[:6]]
    records=[]
    for number,(kind,fold,record) in enumerate(selection):
        source=sources[kind];n=min(rows,source.lengths[record]);parts=[]
        for start in range(0,n,32):
            raw=source.fetch(record,start,min(start+32,n));xs=[];margins=[];trusted=[];normal=[]
            for f,o,action,outcome in zip(raw['features'],raw['objects'],raw['actions'],raw['outcomes']):
                x,d=encode_geometry(f,o,complete=False);trust=d['trusted'].copy()
                # Native outcome can veto a training label, never an input.
                gap=d['margins'][0,int(action)]
                if (outcome==1 and gap>0) or (outcome==0 and gap<0):trust[:]=False
                xs.append(x);margins.append(d['margins']);trusted.append(trust);normal.append(d['normal'])
            parts.append(dict(features=np.stack(xs),margins=np.stack(margins),trusted=np.stack(trusted),
                unprotected_mask=np.asarray(normal,bool),actions=raw['actions'],outcomes=raw['outcomes'],
                episode_starts=raw['episode_starts'],rule_probabilities=raw['rule_probabilities'],rule_weights=raw['rule_weights']))
        arrays={key:np.concatenate([p[key] for p in parts]) for key in parts[0]}
        path=output/f'record-{number:03}.npz'
        with path.open('xb') as stream:np.savez_compressed(stream,**arrays)
        eligible=arrays['trusted'][:,0]&arrays['unprotected_mask']
        danger=(arrays['margins'][:,0]<=0)&eligible[:,None]
        entry=dict(path=str(path),kind=kind,fold=fold,record=record,rows=n,danger_queries=int(danger.sum()),trusted_scenes=int(eligible.sum()))
        records.append(entry);on_progress(dict(phase='compressing_original_training_rows',complete=number+1,target=len(selection),latest=entry))
    manifest=dict(schema=SCHEMA+'-chronological-cache-v1',records=records,map_in_training_input=False,
        teacher_source=str(expert),historical_student_source=str(own),source_identity={k:v.identity for k,v in sources.items()},
        sample_split='whole source world/trajectory;formal native evaluation uses independent development RNG partition',
        post_action_facts_are_labels_only=True,new_native_student_actions=0)
    write_json(output/'manifest.json',manifest);return manifest


class Streams:
    def __init__(self,records,*,width=16,length=16,seed=1):
        self.records=records;self.width=width;self.length=length;self.rng=np.random.default_rng(seed)
        self.data=[]
        for r in records:
            with np.load(r['path'],allow_pickle=False) as z:self.data.append({k:z[k] for k in z.files})
        self.queue=deque();self.slots=[None]*width;self.epoch=0
    def next(self):
        if not self.queue and all(s is None for s in self.slots):
            self.queue.extend(self.rng.permutation(len(self.data)).tolist());self.epoch+=1
        for lane in range(self.width):
            if self.slots[lane] is None and self.queue:self.slots[lane]=[self.queue.popleft(),0]
        shape=(self.length,self.width)
        data={k:np.zeros((*shape,*v.shape[1:]),v.dtype) for k,v in self.data[0].items()}
        data['episode_starts'][:]=True;data['valid']=np.zeros(shape,bool)
        for lane,slot in enumerate(self.slots):
            if slot is None:continue
            record,first=slot;stop=min(first+self.length,len(self.data[record]['features']));n=stop-first
            for key,v in self.data[record].items():data[key][:n,lane]=v[first:stop]
            data['valid'][:n,lane]=True;slot[1]=stop
            if stop==len(self.data[record]['features']):self.slots[lane]=None
        return data
    def state_dict(self):return dict(queue=list(self.queue),slots=copy.deepcopy(self.slots),epoch=self.epoch,RNG=copy.deepcopy(self.rng.bit_generator.state))
