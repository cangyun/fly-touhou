"""Bounded pre-event neural traces for complete original-game evaluations.

These are observations of the frozen student, never action overrides. A
contact-window event and a later lost life are deliberately distinct facts.
"""
from collections import deque
from pathlib import Path
import os
import numpy as np
import torch
from .thrl_durable_records import write_json,append_jsonl


class StoryDiagnostics:
    def __init__(self,output,worlds,*,history=32):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        # Retain128 original frames at the current4-frame action interval.
        # A final32-frame tail can start after every escape route is gone.
        # This only extends saved observations;policy inputs/state are untouched.
        self.history=history
        self.tails={i:deque(maxlen=history) for i in range(worlds)}
        self.events=0;self.event_counts={};self.rows=0
        self.mixed=0;self.mixed_unsafe=0;self.mixed_bomb=0
        self.no_forecast_escape=0;self.event_stats=[]
        self.bomb_opportunities=dict(stocked_ordinary_decisions=0,stocked_deathbomb_decisions=0,
            ordinary_Bomb_requests=0,deathbomb_Bomb_requests=0,actual_ordinary_Bombs=0,actual_deathbomb_Bombs=0,
            actual_ordinary_Bombs_with_safe4=0,actual_protected_Bombs=0)

    @staticmethod
    def context(info):
        return {k:info.get(k) for k in ('stage','stage_frame','episode_frames',
            'lives','bombs','misses','bombs_used','compact_observation_frame')}

    def before_actions(self,worlds,features,infos,out,actions):
        scores=out['horizon_scores'].detach().cpu()
        logits=out['logits'].detach().cpu()
        dn=out.get('candidate_DN')
        dn=None if dn is None else dn.detach().to(device='cpu',dtype=torch.float16)
        for j,i in enumerate(worlds):
            label=infos[i]['compact_targets'];m=np.asarray(label['margins'],np.float32)
            action=int(actions[j]);normal=bool(label['normal']);trusted=bool(label['trusted'][0])
            safe=m[0]>1.;danger=m[0]<=0;eligible=normal and trusted
            mixed=eligible and bool(safe.any() and danger.any())
            f=features[j]
            critical=bool(f[4]>0 and f[12]>0 and f[17]>.5)
            ordinary=bool(f[4]>0 and f[14]>.5 and not critical)
            self.bomb_opportunities['stocked_ordinary_decisions']+=int(ordinary)
            self.bomb_opportunities['stocked_deathbomb_decisions']+=int(critical)
            self.bomb_opportunities['ordinary_Bomb_requests']+=int(ordinary and action==18)
            self.bomb_opportunities['deathbomb_Bomb_requests']+=int(critical and action==18)
            self.rows+=1;self.mixed+=int(mixed)
            self.mixed_unsafe+=int(mixed and action<18 and danger[action])
            self.mixed_bomb+=int(mixed and action==18)
            self.no_forecast_escape+=int(eligible and bool(danger.all()))
            row=dict(context=self.context(infos[i]),features=torch.as_tensor(features[j]).clone(),
                action=action,logits=logits[j].clone(),horizon_scores=scores[j].clone(),
                margins=torch.from_numpy(m.copy()),trusted=list(label['trusted']),normal=normal,
                uncertainty=label.get('uncertainty'))
            if dn is not None:row['candidate_DN']=dn[j].clone()
            self.tails[i].append(row)

    def after_action(self,world,before,after,previous_features,features,*,terminated=False):
        kinds=[]
        if float(features[17])>.5 and float(previous_features[17])<.5:kinds.append('contact_window_entered')
        if after.get('misses',0)>before.get('misses',0):kinds.append('life_lost')
        if after.get('bombs_used',0)>before.get('bombs_used',0):kinds.append('Bomb_used')
        if after.get('stage')!=before.get('stage'):kinds.append('stage_transition')
        if terminated:kinds.append('native_terminal')
        if not kinds:return
        self.events+=1
        for k in kinds:self.event_counts[k]=self.event_counts.get(k,0)+1
        summary=dict(event=self.events,world=world,kinds=kinds,before=self.context(before),after=self.context(after))
        if self.tails[world]:
            last=self.tails[world][-1];m=last['margins'][0];a=last['action']
            summary.update(last_action=a,pre_action_normal=last['normal'],
                pre_action_trusted=last['trusted'],
                pre_action_uncertainty=last.get('uncertainty'),
                geometric_immediate_safe_candidates=int((m>1).sum()),
                chosen_immediate_margin=float(m[a]) if a<18 else None)
            f=last['features'];critical=bool(f[4]>0 and f[12]>0 and f[17]>.5)
            protected=bool(f[14]>.5 and (f[5]>0 or f[13]>0))
            summary.update(pre_action_deathbomb_window=critical,pre_action_protected=protected)
            used=max(0,int(after.get('bombs_used',0)-before.get('bombs_used',0)))
            self.bomb_opportunities['actual_deathbomb_Bombs' if critical else 'actual_ordinary_Bombs']+=used
            self.bomb_opportunities['actual_ordinary_Bombs_with_safe4']+=used*int(not critical and last['normal'] and last['trusted'][0] and bool((m>1).any()))
            self.bomb_opportunities['actual_protected_Bombs']+=used*int(protected)
        path=self.output/f'event-{self.events:05}.pt';temp=path.with_suffix('.pt.tmp')
        with temp.open('xb') as f:torch.save(dict(summary=summary,pre_action_tail=list(self.tails[world]),
            physical_transition=after.get('compact_contact_evidence')),f)
        os.replace(temp,path)
        summary['trace_file']=path.name;self.event_stats.append(summary)
        append_jsonl(self.output/'events.jsonl',summary)

    def finish(self):
        report=dict(schema='compact-SNN-full-story-event-diagnostics-v1',student_decisions=self.rows,
            pre_event_history_decisions=self.history,
            events=self.events,event_counts=self.event_counts,mixed_immediate_rows=self.mixed,
            chosen_predicted_unsafe_in_mixed=self.mixed_unsafe,Bomb_in_mixed=self.mixed_bomb,
            no_constant_action_forecast_escape_rows=self.no_forecast_escape,
            bomb_opportunities=self.bomb_opportunities,
            event_summaries=self.event_stats,
            metric_scope='Current-observation physical forecasts;not measured future collision probabilities. Contact windows and lost lives are distinct events.',
            policy_changed=False,teacher_actions=0,brain_traces='DN filtered spikes+subthreshold membrane for all18 candidates;FP16 storage only')
        write_json(self.output/'report.json',report);return report
