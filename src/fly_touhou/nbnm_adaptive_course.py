"""Allocate local-task decisions using recent student success and forgetting.

The stage/chain and full-story pools keep their existing segment quotas.
Local targets change at segment boundaries; a task changes only at a genuine
episode terminal. Credits count acknowledged decisions, not reset attempts.
"""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np

SCHEMA='nbnm-adaptive-local-decisions-v1'


class AdaptiveCourse:
    def __init__(self, keys, groups, *, window=20, uniform_fraction=.25, prior=2., seed=0,coverage_first=False,
                 group='local',strata=None,stratum_weights=None,priority_multipliers=None):
        self.keys=list(keys);self.groups=list(groups)
        if not self.keys or len(set(self.keys))!=len(self.keys):raise ValueError('Distinct local task keys required')
        if not 4<=window<=100 or not 0<uniform_fraction<=1 or prior<=0:raise ValueError('Invalid adaptive course settings')
        self.window=int(window);self.uniform_fraction=float(uniform_fraction);self.prior=float(prior)
        self.history={k:[] for k in self.keys};self.credit={k:0. for k in self.keys}
        self.decisions={k:0 for k in self.keys};self.assignments={k:0 for k in self.keys}
        self.active={};self.rng=np.random.default_rng(seed);self.version=0
        self.weights={k:1/len(self.keys) for k in self.keys}
        self.last_details={};self.total_decisions=0
        self.coverage_first=bool(coverage_first)
        self.group=group;self.strata=dict(strata or {});self.stratum_weights=dict(stratum_weights or {})
        self.priority_multipliers=dict(priority_multipliers or {})
        if any(k not in self.keys or not np.isfinite(v) or not 1<=v<=4 for k,v in self.priority_multipliers.items()):
            raise ValueError('Use declared tasks and bounded development-priority multipliers')
        if self.stratum_weights:
            if (set(self.strata)!=set(self.keys) or set(self.strata.values())!=set(self.stratum_weights)
                    or any(not np.isfinite(w) or w<=0 for w in self.stratum_weights.values())):
                raise ValueError('Every task needs one positive-weight sampling stratum')

    def _details(self, key):
        rows=self.history[key];recent=rows[-self.window:];older=rows[-2*self.window:-self.window]
        n=len(recent);wins=sum(r['success'] for r in recent)
        rate=(wins+self.prior)/(n+2*self.prior)
        old_rate=(sum(r['success'] for r in older)+self.prior)/(len(older)+2*self.prior) if older else rate
        forgetting=max(0.,old_rate-rate)
        # All-zero tasks receive extra exposure, but cannot monopolize the pool.
        priority=min(1.,.25+.75*(1-rate)+.5*forgetting)
        duration=float(np.mean([r['decisions'] for r in recent])) if recent else 512.
        return dict(attempts=n,successes=wins,raw_success_rate=wins/n if n else None,
            smoothed_success_rate=rate,forgetting=forgetting,priority=priority,
            expected_episode_decisions=max(16.,duration),
            provisional_mastery=n>=self.window and wins/n>=.9)

    def refresh(self):
        self.last_details={k:self._details(k) for k in self.keys}
        u=self.uniform_fraction
        partitions=(self.stratum_weights if self.stratum_weights else {'all':1.})
        self.weights={}
        for stratum,share in partitions.items():
            keys=[k for k in self.keys if not self.stratum_weights or self.strata[k]==stratum]
            priority={k:self.last_details[k]['priority']*self.priority_multipliers.get(k,1.) for k in keys}
            total=sum(priority.values())
            for k in keys:
                self.weights[k]=share/sum(partitions.values())*(u/len(keys)+(1-u)*priority[k]/total)
        self.version+=1

    def observe_episode(self, record):
        key=record.get('task_key')
        if key not in self.history:return
        allowed=('attack',) if self.group=='local' else ('stage','chain')
        if record.get('task',{}).get('mode') not in allowed:return
        success=bool(record['nbnm_success'])
        # A terminal native failure is a real trial. Prefixes, teacher runs,
        # interrupted worlds and unfinished attempts are not added here.
        if record.get('task_Bombs',0) or record.get('task_misses',0):success=False
        decisions=record.get('policy_decisions')
        if decisions is None:decisions=max(1,round(record['task_native_frames']/4))
        self.history[key].append(dict(success=success,decisions=int(decisions),native_seed=record.get('native_seed')))
        self.history[key]=self.history[key][-2*self.window:]

    def bind_worlds(self, infos):
        for i,info in enumerate(infos):
            if self.groups[i]!=self.group:continue
            key=info['nb_nm_task_key']
            if key not in self.weights:raise ValueError('Active task is outside adaptive local pool')
            previous=self.active.get(i)
            if previous and previous['key']==key:continue
            elapsed=max(0,int(info.get('task_native_frames',0))//4)
            remaining=max(16.,self._details(key)['expected_episode_decisions']-elapsed)
            self.active[i]=dict(key=key,remaining=remaining)

    def observe_step(self, world, key):
        if self.groups[world]!=self.group:return
        if key not in self.weights:raise ValueError('Unknown local decision task')
        for candidate in self.keys:self.credit[candidate]+=self.weights[candidate]
        self.credit[key]-=1.;self.decisions[key]+=1;self.total_decisions+=1
        active=self.active.get(world)
        if active and active['key']==key:active['remaining']=max(0.,active['remaining']-1.)

    def finish(self, world, record):
        if self.groups[world]!=self.group:return None
        self.observe_episode(record);self.active.pop(world,None)
        return self.choose(world)

    def choose(self, world):
        if self.groups[world]!=self.group:return None
        reserved={k:0. for k in self.keys}
        for active in self.active.values():reserved[active['key']]+=max(16.,active['remaining'])
        scores=np.array([self.credit[k]-reserved[k] for k in self.keys])
        if self.coverage_first:
            active_keys={v['key'] for v in self.active.values()}
            unseen=np.array([not self.history[k] and k not in active_keys for k in self.keys])
            if unseen.any():scores[~unseen]=-np.inf
        best=np.flatnonzero(np.isclose(scores,scores.max(),rtol=0,atol=1e-9))
        key=self.keys[int(self.rng.choice(best))]
        self.assignments[key]+=1
        self.active[world]=dict(key=key,remaining=self._details(key)['expected_episode_decisions'])
        return dict(task_key=key)

    def state_dict(self):
        return dict(schema=SCHEMA,keys=list(self.keys),groups=list(self.groups),window=self.window,
            uniform_fraction=self.uniform_fraction,prior=self.prior,history=deepcopy(self.history),
            credit=dict(self.credit),decisions=dict(self.decisions),assignments=dict(self.assignments),
            active=deepcopy(self.active),weights=dict(self.weights),last_details=deepcopy(self.last_details),
            total_decisions=self.total_decisions,version=self.version,rng=deepcopy(self.rng.bit_generator.state),
            group=self.group,strata=self.strata,stratum_weights=self.stratum_weights,
            priority_multipliers=self.priority_multipliers)

    def load_state_dict(self, state):
        if state['schema']!=SCHEMA or state['keys']!=self.keys or state['groups']!=self.groups:
            raise ValueError('Adaptive course belongs to a different task catalog')
        for name in ('history','credit','decisions','assignments','weights','last_details','total_decisions','version'):
            setattr(self,name,deepcopy(state[name]))
        self.active={int(k):deepcopy(v) for k,v in state['active'].items()}
        self.rng.bit_generator.state=deepcopy(state['rng'])

    def summary(self):
        return dict(schema=SCHEMA,scope='recent changing-policy '+self.group+' NBNM attempts; not frozen qualification',
            version=self.version,window=self.window,uniform_fraction=self.uniform_fraction,
            local_decisions=self.total_decisions,allocation_unit='acknowledged student decisions',
            coverage_first=self.coverage_first,unattempted_tasks=[k for k in self.keys if not self.history[k]],
            group=self.group,stratum_weights=self.stratum_weights,priority_multipliers=self.priority_multipliers,
            task_switch='only after native episode terminal',
            tasks={k:dict(self._details(k),target_local_decision_fraction=self.weights[k],
                actual_decisions=self.decisions[k],new_assignments=self.assignments[k],
                credit=self.credit[k]) for k in self.keys})


def attach(collector, spec, output, continuation=None):
    settings=spec.get('adaptive_course') or {}
    if not settings.get('enabled',False):return
    course=AdaptiveCourse(spec['task_pools']['local'],spec['world_groups'],
        window=settings.get('window',20),uniform_fraction=settings.get('uniform_fraction',.25),
        prior=settings.get('prior',2.),seed=spec['seed']+917,coverage_first=settings.get('coverage_first',False),
        strata={k:('spell' if spec['tasks'][k].get('spell_id') is not None or 'Card' in spec['tasks'][k].get('attack','') else 'nonspell')
                for k in spec['task_pools']['local']},stratum_weights=settings.get('stratum_weights'),
        priority_multipliers=settings.get('priority_multipliers'))
    saved=(continuation or {}).get('adaptive_course')
    if saved:course.load_state_dict(saved)
    else:
        for path in sorted(Path(output).glob('native-training-*/training-episodes.jsonl')):
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                try:row=json.loads(line)
                except json.JSONDecodeError:break
                course.observe_episode(row)
        course.refresh()
    course.bind_worlds(collector.infos)
    collector.adaptive_course=course
    collector.coverage_group_weights=spec.get('coverage_group_weights')
    collector.default_group_weights=dict(spec['group_weights'])
    if settings.get('stage_chain_enabled'):
        chain=AdaptiveCourse(spec['task_pools']['stage_chain'],spec['world_groups'],group='stage_chain',
            window=settings.get('window',20),uniform_fraction=settings.get('uniform_fraction',.25),
            prior=settings.get('prior',2.),seed=spec['seed']+918,coverage_first=True,
            strata={k:spec['tasks'][k]['mode'] for k in spec['task_pools']['stage_chain']},
            stratum_weights=settings.get('stage_chain_stratum_weights'),
            priority_multipliers=settings.get('stage_chain_priority_multipliers'))
        state=(continuation or {}).get('adaptive_stage_chain')
        if state:chain.load_state_dict(state)
        else:
            from .thrl_study_io import read_shared_text
            paths=list(Path(output).glob('native-training-*/training-episodes.jsonl'))
            paths+=list(map(Path,settings.get('stage_chain_history_files',[])))
            for path in dict.fromkeys(paths):
                for line in read_shared_text(path).splitlines():
                    if line.strip():chain.observe_episode(json.loads(line))
        chain.refresh();chain.bind_worlds(collector.infos)
        collector.adaptive_stage_chain=chain
