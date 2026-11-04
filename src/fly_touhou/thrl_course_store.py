"""Verified compact native trajectories with thousands of selectable starts.

Only hashes, actual input tapes, resource counters and phase metadata are
stored. Maps are regenerated from the original game; no synthetic resources,
memory savestates, model targets or future trajectories enter observations.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from .native_replay import START_SCHEMA, VISUAL_ALIGNMENT_SCHEMA, validate_tape
from .thrl_curriculum import seed_split

TRACE_SCHEMA='th14-verified-native-course-trajectory-v1'
STORE_SCHEMA='th14-progressive-course-store-v1'
PHASES=('opening','midboss','latter','boss')
LABELS=('道中前半','中Boss','道中后半','Boss')


def lesson_id(stage,phase):
    return f's{stage}_{phase}'


def lesson_order(key):
    match=re.fullmatch(r's([1-6])_(opening|midboss|latter|boss)',key)
    if not match:
        raise ValueError('Invalid native lesson identifier')
    return (int(match[1])-1)*4+PHASES.index(match[2])


class PhaseTracker:
    """Monotonic coarse course phase; never interpret disappearance as kill.

    Attack names remain metadata for per-attack diagnostics and random
    sampling. Midboss exit requires observed post-midboss ECL context plus
    sustained measured boss absence. Main boss finishes only by stage advance
    or an actual native ending, so utility/Hide phases cannot fake completion.
    """
    def __init__(self,phase=None,*,state=None):
        self.stage=1
        self.phase='opening'
        self.absent_since=None
        self.post_midboss_context=False
        if phase:
            self.stage=int(phase[1])
            self.phase=phase.split('_',1)[1]
        if state is not None:
            if state['stage']!=self.stage or state['phase']!=self.phase:
                raise ValueError('Course phase history belongs to another landing')
            self.absent_since=state['absent_since']
            self.post_midboss_context=state['post_midboss_context']

    def state_dict(self):
        return dict(stage=self.stage,phase=self.phase,absent_since=self.absent_since,
                    post_midboss_context=self.post_midboss_context)

    def observe(self,snapshot):
        if not 1<=snapshot.stage<=6:
            return lesson_id(self.stage,self.phase)
        if snapshot.stage != self.stage:
            self.stage=snapshot.stage;self.phase='opening';self.absent_since=None
            self.post_midboss_context=False
        roles=snapshot.extra.get('thrl_events',{}).get('ecl_roles',{})
        if roles.get('observed') is not True or roles.get('midboss_observed') is not True:
            self.absent_since=None
            return lesson_id(self.stage,self.phase)
        # TH14's MainLatter thread can start while MBossCard1 is still
        # alive, then becomes MainSub05 before the boss disappears. Preserve
        # that measured context across frames; absence alone never advances.
        if any(name.startswith('MainLatter') for name in roles.get('main_enemy_subs',())):
            self.post_midboss_context=True
        if roles.get('main_bosses'):
            self.phase='boss';self.absent_since=None
        elif roles.get('midbosses') and self.phase in ('opening','midboss'):
            self.phase='midboss';self.absent_since=None
        elif self.phase=='midboss':
            if self.post_midboss_context:
                if self.absent_since is None:self.absent_since=snapshot.stage_frame
                if snapshot.stage_frame-self.absent_since>=32:self.phase='latter'
            else:self.absent_since=None
        return lesson_id(self.stage,self.phase)


def slice_tape(segments,start,end):
    if type(start) is not int or type(end) is not int or not 0<=start<end:
        raise ValueError('Nonempty ordered native tape interval required')
    result=[];offset=0
    for mask,ticks in validate_tape(segments):
        take=max(0,min(offset+ticks,end)-max(offset,start))
        if take:result.append([mask,take])
        offset+=ticks
        if offset>=end:break
    if sum(t for _,t in result)!=end-start:
        raise ValueError('Native tape does not cover requested prefix')
    return validate_tape(result)


def trajectory_entry(trace,index):
    row=trace['records'][index]
    if row['ticks']<=0:
        raise ValueError('A prefix landing must follow the canonical base')
    entry={key:deepcopy(trace[key]) for key in ('native_seed','base_stage_frame','base_state_sha256',
               'game_exe_sha256','base_visual_rng_alignment')}
    entry.update(schema=START_SCHEMA,id=f"{trace['id']}@{row['ticks']}",
        landing_state_sha256=row['state_sha256'],raw_input_segments=slice_tape(trace['raw_input_segments'],0,row['ticks']),
        landing={key:row[key] for key in ('stage','stage_frame','lives','bombs','power')},
        reward_tracking=row['reward_tracking'],verified_independent_resets=trace['verified_independent_resets'],
        resources_injected=False,native_deaths=True)
    barriers=[]
    for before,after in zip(trace['records'][:index],trace['records'][1:index+1]):
        if before['stage']==after['stage']:continue
        if after['stage']!=before['stage']+1 or after['stage_frame']!=4:
            raise ValueError('Cross-stage course replay requires an exact loaded frame4 barrier')
        span=slice_tape(trace['raw_input_segments'],before['ticks'],after['ticks'])
        if span[-1][0]!=1:
            raise ValueError('Loading bridge must end with recorded native shot-idle input')
        barriers.append(dict(start_ticks=before['ticks'],end_ticks=after['ticks'],
            prior_state_sha256=before['state_sha256'],state_sha256=after['state_sha256'],
            stage=after['stage'],stage_frame=4,initial_tape=span[:-1]))
    if barriers:entry['loading_barriers']=barriers
    return entry


def validate_trace(trace):
    if (trace.get('schema')!=TRACE_SCHEMA or trace.get('verified_independent_resets',0)<2
            or trace.get('resources_injected') is not False
            or trace.get('native_deaths') is not True
            or seed_split(trace['native_seed'])!=trace.get('split')):
        raise ValueError('Unverified or mispartitioned native course trajectory')
    tape=validate_tape(trace['raw_input_segments'])
    ticks=sum(t for _,t in tape)
    prior=-1;order=-1
    for row in trace['records']:
        if not prior<row['ticks']<=ticks or row['ticks']<0:
            raise ValueError('Course checkpoints are not strictly ordered actual input ticks')
        new_order=lesson_order(row['phase'])
        if new_order<order:
            raise ValueError('Course phases must follow original game progression')
        if not isinstance(row.get('state_sha256'),str) or len(row['state_sha256'])!=64:
            raise ValueError('Native checkpoint hash is required')
        if row['stage']!=int(row['phase'][1]):raise ValueError('Stage and lesson mismatch')
        prior=row['ticks'];order=new_order
    if len(trace['records'])<3:
        raise ValueError('A trajectory must offer a nontrivial interval of reset states')
    return trace


class CourseStore:
    def __init__(self,manifest,expected_sha256=None,overlay=None,inherited_overlays=None):
        self.path=Path(manifest).resolve()
        raw=self.path.read_bytes()
        if expected_sha256 and hashlib.sha256(raw).hexdigest()!=expected_sha256:
            raise ValueError('Course manifest changed after preparation')
        self.manifest=json.loads(raw.decode('utf-8'))
        if self.manifest.get('schema')!=STORE_SCHEMA:
            raise ValueError('Unknown native course store')
        self.overlay=None if overlay is None else Path(overlay)
        self.inherited_overlays=[Path(p) for p in (inherited_overlays or [])]
        self.traces=[];self._known={}
        self.refresh()

    def refresh(self):
        descriptors=list(self.manifest['trajectories'])
        for root in self.manifest.get('additional_verified_sources',[]):
            descriptors += [dict(path=str(p),sha256=None) for p in sorted(Path(root).glob('seed*/verified.json'))]
        for folder in self.inherited_overlays+([self.overlay] if self.overlay else []):
            if folder.exists():
                descriptors += [dict(path=str(p),sha256=None) for p in sorted(folder.glob('verified_*.json'))]
        for descriptor in descriptors:
            p=Path(descriptor['path'])
            if not p.is_absolute():p=self.path.parent/p
            if str(p) in self._known:continue
            raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest()
            if descriptor.get('sha256') and sha!=descriptor['sha256']:
                raise ValueError('A verified course trajectory changed')
            trace=validate_trace(json.loads(raw.decode('utf-8')))
            self.traces.append(trace);self._known[str(p)]=sha

    def spans(self,lesson,split):
        spans=[]
        for trace in self.traces:
            if trace['split']!=split:continue
            indices=[i for i,r in enumerate(trace['records']) if r['phase']==lesson]
            if not indices:continue
            start,end=indices[0],indices[-1]+1
            # A partial trace may supply starts, but cannot claim this segment
            # was completed. The environment's goal always uses live phases.
            complete=end<len(trace['records']) or trace.get('native_six_stage_clear',False)
            last=trace['records'][min(end,len(trace['records'])-1)]
            spans.append(dict(trace=trace,start=start,end=end,complete=complete,
                length=max(4,last['frame']-trace['records'][start]['frame']),
                sampling_end=max((r['frame']+4 for r in trace['records'][start:end] if r['playable']),default=last['frame'])))
        return spans

    def lessons(self):
        result=[]
        for stage in range(1,7):
            for phase,label in zip(PHASES,LABELS):
                key=lesson_id(stage,phase);spans=self.spans(key,'train')
                result.append(dict(id=key,label=f'第{stage}面{label}',stage=stage,
                    max_depth=max([s['length'] for s in spans]+[300]),available=bool(spans),
                    distinct_native_seeds=len({s['trace']['native_seed'] for s in spans})))
        return result

    def reset_choices(self,assignment,split,min_tail):
        spans=self.spans(assignment['lesson'],split)
        # A partial opening capture can end just after midboss spawn. When
        # completed examples exist, that prefix is not a "last300 frames"
        # example of the midboss lesson. Prefer measured complete boundaries.
        complete=[s for s in spans if s['complete']]
        if complete:spans=complete
        choices=[]
        for span in spans:
            records=span['trace']['records'];end_frame=span['sampling_end']
            lower=(records[span['start']]['frame'] if assignment.get('depth') is None
                   else max(records[span['start']]['frame'],end_frame-int(assignment['depth'])))
            indices=[i for i in range(span['start'],span['end'])
                     if records[i]['ticks']>0 and records[i]['frame']>=lower
                     and records[i]['frame']<=end_frame-min_tail and records[i]['playable']]
            if indices:choices.append((span['trace'],indices))
        return choices

    def has_starts(self,assignment,split,min_tail):
        return bool(self.reset_choices(assignment,split,min_tail))

    def choose(self,assignment,rng,split,min_tail):
        choices=self.reset_choices(assignment,split,min_tail)
        if not choices:
            raise ValueError(f"No verified {split} reset interval for {assignment['lesson']}; prepare more native trajectories")
        trace,indices=choices[int(rng.integers(len(choices)))]
        return trace,indices[int(rng.integers(len(indices)))]

    def summary(self):
        return dict(trajectories=len(self.traces),
                    distinct_native_seeds=len({r['native_seed'] for r in self.traces}),
                    verified_checkpoints=sum(len(r['records']) for r in self.traces),
                    raw_map_files=0,seed_partitions={'train':'seed%10<6','development':'6<=seed%10<8','test':'seed%10>=8'})
