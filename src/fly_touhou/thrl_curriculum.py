"""Progressive lesson scheduling, independent of Torch and native processes.

One policy learns all lessons. Advancement changes sampling only; it never
rejects an optimizer update. Ratios are measured in acknowledged decisions,
not numbers of resets. Curriculum completion is distinct from game clear.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import math
import numpy as np

SCHEMA = 'th14-progressive-rehearsal-curriculum-v1'
SCHEMA_V2 = 'th14-progressive-rehearsal-curriculum-v2'
CHAIN_LESSON_BOUNDARY = 'lesson-boundary-v1'
CHAIN_NATIVE_GAME = 'native-game-v1'


def chain_horizon(value=CHAIN_LESSON_BOUNDARY):
    if value not in (CHAIN_LESSON_BOUNDARY, CHAIN_NATIVE_GAME):
        raise ValueError('Explicit versioned curriculum chain horizon required')
    return value


def qualification_endpoint(summary):
    """A training chain's target receipt survives a later native Game Over.

    Only its assigned target is measured; passing subsequent lessons neither
    fabricates additional attempts nor unlocks them. Evaluation chains still
    end at their assigned boundary and use their ordinary episode summary.
    """
    if (summary.get('curriculum_mode') == 'chain'
            and summary.get('curriculum_chain_horizon') == CHAIN_NATIVE_GAME):
        return summary.get('curriculum_target_boundary') or summary
    return summary
DEFAULTS = dict(initial_depth=300, growth=1.5, threshold=.8, window=100,
                min_tail=32, current_fraction=.60, review_fraction=.25,
                chain_fraction=.15, burn_in=64, gate_episodes=20,
                gate_interval_decisions=16384, retention_tolerance=.15,
                max_misses=1, max_bombs_used=2)
V2_DEFAULTS={**DEFAULTS,'version':2,'current_fraction':.4,'review_fraction':.2,'chain_fraction':.4,
    'first_current_fraction':.5,'first_chain_fraction':.5,'chain_min_lives':1,'gate_interval_decisions':65536}


def schema_for(options):
    return SCHEMA_V2 if options.get('version')==2 else SCHEMA


def configuration(values=None):
    defaults=V2_DEFAULTS if (values or {}).get('version')==2 else DEFAULTS
    result = {**defaults, **(values or {})}
    if set(result) != set(defaults):
        raise ValueError('Unknown curriculum options: '+str(set(result)-set(defaults)))
    if defaults is V2_DEFAULTS:
        if type(result['version']) is not int or type(result['chain_min_lives']) is not int or result['chain_min_lives']<0:
            raise ValueError('Explicit v2 resource reserve required')
        fractions=[result[k] for k in ('first_current_fraction','first_chain_fraction')]
        if any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in fractions) or not math.isclose(sum(fractions),1):
            raise ValueError('First-lesson current/chain fractions must sum to one')
    for key in ('initial_depth','window','min_tail','burn_in','gate_episodes','gate_interval_decisions'):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError('Positive curriculum integer required: '+key)
    if not 1 <= result['burn_in'] <= 128 or result['initial_depth'] < result['min_tail']:
        raise ValueError('Invalid bounded recurrent history or curriculum depth')
    for key in ('max_misses', 'max_bombs_used'):
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError('Explicit native resource consumption allowance required')
    if not 0 < result['threshold'] <= 1 or not 1 < result['growth'] <= 4:
        raise ValueError('Invalid curriculum threshold/growth')
    ratios = [result[k] for k in ('current_fraction','review_fraction','chain_fraction')]
    if any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in ratios) or not math.isclose(sum(ratios),1):
        raise ValueError('Current/review/chain positive decision fractions must sum to one')
    if not 0 <= result['retention_tolerance'] <= .5:
        raise ValueError('Invalid retention tolerance')
    return result


def seed_split(seed):
    if type(seed) is not int or not 0 <= seed <= 65535:
        raise ValueError('Native seed must be uint16')
    remainder = seed % 10
    return 'train' if remainder < 6 else 'development' if remainder < 8 else 'test'


def native_seed(rng, split):
    if split not in ('train','development','test'):
        raise ValueError('Unknown native seed split')
    while True:
        value = int(rng.integers(0,65536))
        if seed_split(value) == split:
            return value


def qualified(summary, cfg):
    """Finite lives/Bomb remain enabled; resource costs accompany passage."""
    summary = qualification_endpoint(summary)
    if summary.get('curriculum_mode')=='chain':
        return (summary.get('segment_reached') is True and summary.get('continues',0)==0
                and summary.get('lives',-1)>=0)
    return (summary.get('segment_reached') is True
            and summary.get('continues',0) == 0
            and 0 <= summary.get('segment_misses',float('inf')) <= cfg['max_misses']
            and (cfg.get('version')==2 or 0 <= summary.get('segment_bombs_used',float('inf')) <= cfg['max_bombs_used']))


def resource_qualified(summary,cfg):
    summary = qualification_endpoint(summary)
    if not qualified(summary,cfg):return False
    if summary.get('curriculum_mode')!='chain' or cfg.get('version')!=2:return True
    return summary.get('outcome')=='clear' or summary.get('lives',-1)>=cfg['chain_min_lives']


class Curriculum:
    def __init__(self, lessons, *, seed=11, options=None, state=None,
                 curriculum_chain_horizon=CHAIN_LESSON_BOUNDARY):
        self.options = configuration(options)
        self.chain_horizon = chain_horizon(curriculum_chain_horizon)
        self.schema=schema_for(self.options)
        if not lessons or len({r['id'] for r in lessons}) != len(lessons):
            raise ValueError('Ordered distinct nonempty curriculum lessons required')
        self.lessons = deepcopy(lessons)
        self.rng = np.random.default_rng(seed)
        self.frontier = 0
        self.depth = float(self.options['initial_depth'])
        self.recent = deque(maxlen=self.options['window'])
        self.decisions = dict(current=0,review=0,chain=0)
        self.quota_decisions=dict(self.decisions)
        self.by_lesson = {r['id']:dict(attempts=0,passes=0,decisions=0) for r in lessons}
        self.gate = None
        self.last_gate_decisions = 0 if self.options.get('version')==2 else -self.options['gate_interval_decisions']
        self.promotions = []
        self.development_baselines = {}
        self.practice_priority='normal'
        if state is not None:
            if state.get('curriculum_chain_horizon', CHAIN_LESSON_BOUNDARY) != self.chain_horizon:
                raise ValueError('Cannot silently change the resumed curriculum chain horizon')
            if state.get('schema') != self.schema or state['lesson_ids'] != [r['id'] for r in lessons] or state['options'] != self.options:
                raise ValueError('Cannot silently change the resumed curriculum protocol')
            for key in ('frontier','depth','decisions','by_lesson','gate','last_gate_decisions','promotions','development_baselines'):
                setattr(self,key,deepcopy(state[key]))
            self.recent.extend(state['recent'])
            self.rng.bit_generator.state = state['rng']
            self.practice_priority=state.get('practice_priority','normal')
            self.quota_decisions=deepcopy(state.get('quota_decisions',self.decisions))

    @property
    def lesson(self):
        return self.lessons[self.frontier]

    def mixture(self):
        # Before any old task exists, spend its rehearsal quota on the current
        # lesson. A chain still starts at the actual game opening.
        ratios = {k:self.options[k+'_fraction'] for k in self.decisions}
        if self.options.get('version')==2:
            return dict(current=self.options['first_current_fraction'],chain=self.options['first_chain_fraction']) if self.frontier==0 else ratios
        if self.practice_priority in ('review','chain'):
            transfer=min(.25,ratios['current']-.1)
            ratios['current']-=transfer
            ratios[self.practice_priority]+=transfer
        if self.frontier == 0:
            ratios['current'] += ratios.pop('review')
        return ratios

    def sample(self):
        ratios = self.mixture()
        counts=self.quota_decisions if self.options.get('version')==2 else self.decisions
        total = sum(counts.values())
        # Deficit scheduling pays down long chain episodes instead of treating
        # a 30-step reset and a 3000-step episode as equal samples.
        deficits = {k:r*(total+1)-counts[k] for k,r in ratios.items()}
        candidates = [k for k in ratios if deficits[k] >= max(deficits.values())-1e-12]
        mode = candidates[int(self.rng.integers(len(candidates)))]
        index = self.frontier if mode != 'review' else int(self.rng.integers(self.frontier))
        lesson = self.lessons[index]
        return dict(schema=self.schema,mode=mode,lesson=lesson['id'],lesson_index=index,
                    depth=self.depth if mode=='current' else None,
                    frontier=self.frontier,attempt_id=int(self.rng.integers(2**53)))

    def acknowledge(self, assignment, info):
        mode, lesson = assignment['mode'], assignment['lesson']
        self.decisions[mode] += 1
        if (assignment['frontier']==self.frontier
                or mode=='chain' and self.chain_horizon==CHAIN_NATIVE_GAME):
            # A long native run remains chain exposure after a promotion;
            # excluding its continuing actions would undercount that quota.
            self.quota_decisions[mode]+=1
        # Full chains can visit still-locked phases. Record actual exposure
        # without changing frontier, lesson attempts, or the current gate.
        decision_lesson = (info.get('curriculum_decision_lesson',lesson)
                           if mode=='chain' and self.chain_horizon==CHAIN_NATIVE_GAME else lesson)
        self.by_lesson[decision_lesson]['decisions'] += 1
        summary = info.get('episode_summary')
        if not summary:
            return
        # Endpoints dispatched before a promotion cannot advance a new task.
        stat = self.by_lesson[lesson]
        stat['attempts'] += 1
        success = qualified(summary,self.options)
        stat['passes'] += int(success)
        if mode=='current' and assignment['frontier']==self.frontier and assignment['depth']==self.depth:
            self.recent.append(success)

    def training_ready(self):
        return len(self.recent)==self.options['window'] and float(np.mean(self.recent))>=self.options['threshold']

    def request_gate(self, decisions):
        ready=(len(self.recent)==self.options['window']) if self.options.get('version')==2 else self.training_ready()
        if (not ready or decisions-self.last_gate_decisions < self.options['gate_interval_decisions']):
            return None
        self.last_gate_decisions = decisions
        return dict(lesson=self.lesson['id'],frontier=self.frontier,depth=self.depth,
                    episodes=self.options['gate_episodes'],split='development',
                    full_span=self.depth>=self.lesson['max_depth'])

    def apply_gate(self, report, decisions):
        if report['lesson']!=self.lesson['id'] or report['depth']!=self.depth or report.get('split')!='development':
            raise ValueError('Gate belongs to another lesson/depth/seed split')
        if report.get('complete') is not True:
            raise ValueError('Incomplete/failed probes are not curriculum failures or zero scores')
        self.gate = deepcopy(report)
        threshold = self.options['threshold']
        if self.options.get('version')==2:
            self.practice_priority='chain' if report.get('chain_pass_rate',0)<threshold else 'normal'
        if report['pass_rate'] < threshold:
            return 'continue-learning'
        # New tasks may not hide a loss in the opening or a previously learned
        # lesson. These are fresh current-policy probes, never old PPO data.
        for key, rate in report.get('retention',{}).items():
            baseline = self.development_baselines.get(key,threshold)
            if rate < baseline-self.options['retention_tolerance']:
                self.practice_priority='review'
                return 'rehearse-previous-lessons'
        if self.depth < self.lesson['max_depth']:
            self.depth = min(float(self.lesson['max_depth']), self.depth*self.options['growth'])
            self.recent.clear()
            return 'expand-start-window'
        if report.get('chain_pass_rate',0) < threshold:
            self.practice_priority='chain'
            return 'practice-connections'
        if self.options.get('version')==2 and report.get('chain_resource_pass_rate',0)<threshold:
            self.practice_priority='chain'
            return 'practice-resource-preservation'
        self.practice_priority='normal'
        self.development_baselines[self.lesson['id']] = report['pass_rate']
        self.promotions.append(dict(lesson=self.lesson['id'],decisions=decisions,gate=deepcopy(report)))
        self.recent.clear()
        if self.frontier+1 == len(self.lessons):
            return 'catalogue-complete-needs-full-game-validation'
        self.frontier += 1
        self.quota_decisions=dict(current=0,review=0,chain=0)
        self.depth = float(self.options['initial_depth'])
        return 'unlock-next-lesson'

    def state_dict(self):
        return dict(schema=self.schema,lesson_ids=[r['id'] for r in self.lessons],options=self.options,
                    frontier=self.frontier,depth=self.depth,recent=list(self.recent),
                    decisions=self.decisions,by_lesson=self.by_lesson,gate=self.gate,
                    last_gate_decisions=self.last_gate_decisions,promotions=self.promotions,
                    development_baselines=self.development_baselines,practice_priority=self.practice_priority,
                    rng=self.rng.bit_generator.state,quota_decisions=self.quota_decisions,
                    **({'curriculum_chain_horizon':self.chain_horizon}
                       if self.chain_horizon!=CHAIN_LESSON_BOUNDARY else {}))

    def summary(self):
        total=sum(self.decisions.values())
        return dict(schema=self.schema,lesson=self.lesson['id'],label=self.lesson.get('label',self.lesson['id']),
            unlocked=self.frontier+1,catalogue_lessons=len(self.lessons),depth=int(self.depth),
            max_depth=self.lesson['max_depth'],recent_attempts=len(self.recent),
            recent_pass_rate=float(np.mean(self.recent)) if self.recent else None,
            threshold=self.options['threshold'],decision_counts=self.decisions,
            observed_fractions={k:v/max(total,1) for k,v in self.decisions.items()},
            gate=self.gate,promotions=len(self.promotions),burn_in=self.options['burn_in'],
            practice_priority=self.practice_priority,target_fractions=self.mixture(),
            curriculum_chain_horizon=self.chain_horizon,
            chain_episode_horizon=('native Game Over or clear' if self.chain_horizon==CHAIN_NATIVE_GAME
                                  else 'assigned lesson boundary'),
            phase_decision_counts={k:v['decisions'] for k,v in self.by_lesson.items()},
            policy_count=1,update_rejections=0,formal_six_stage_clear=False)
