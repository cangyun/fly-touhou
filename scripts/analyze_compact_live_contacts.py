"""Read current training contact histories without model/game execution."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
import numpy as np
from fly_touhou.compact_story_route_attribution import route_warning_attribution
from fly_touhou.thrl_dodge_diagnostics import _shared_binary
from fly_touhou.thrl_durable_records import write_json


def analyze(path):
    contacts=[];legacy=0;partial=False;invalid=0
    with _shared_binary(path) as stream:
        # Only the bytes already present when opened belong to this snapshot.
        size=stream.seek(0,2);stream.seek(0)
        while stream.tell()<size:
            line=stream.readline(size-stream.tell())
            if not line.endswith(b'\n'):
                partial=True;break
            event=json.loads(line);trace=event.get('policy_context')
            if trace is None:legacy+=1;continue
            transition=event['transition']
            if not transition.get('entered_contact_window'):continue
            rows=[]
            for record in trace['pre_action_tail']:
                row=dict(record)
                for key in ('features','margins','horizon_scores','logits'):
                    row[key]=torch.as_tensor(np.asarray(record[key],dtype=np.float32))
                # A masked unavailable Bomb may have an infinite logit. It is
                # irrelevant to movement attribution and must not hide stock=0
                # contacts, the very cases this diagnostic needs to retain.
                if not (all(torch.isfinite(row[k]).all() for k in ('features','margins','horizon_scores'))
                        and torch.isfinite(row['logits'][:18]).all()):
                    invalid+=1;continue
                rows.append(row)
            if not rows or rows[-1]['context']!=trace['pre_action_tail'][-1]['context']:continue
            last=rows[-1];last['uncertainty']=event.get('before_uncertainty');a=last['action']
            if not last['normal'] or a>=18:continue
            attribution=route_warning_attribution(rows,risk_scale=trace['neural_risk_scale'])
            p=(8*last['horizon_scores']-1).sigmoid()
            margins=last['margins'];context=last['context']
            contacts.append(dict(world=event['world'],student_decision=event['training_student_decisions'],
                stage=context['stage'],episode_frame=context['episode_frames'],action=a,
                bombs=context['bombs'],lives=context['lives'],trusted=last['trusted'],
                selected_p4=float(p[a,0]),selected_margin4=float(margins[0,a]),
                safe_alternatives4=int((margins[0]>1).sum()),
                forecast_safe_but_actual_contact=bool(margins[0,a]>0),
                covered_forecast_safe_but_actual_contact=bool(last['trusted'][0] and margins[0,a]>0),
                route_warning_attribution=attribution))
    stages={}
    for stage in sorted({c['stage'] for c in contacts}):
        subset=[c for c in contacts if c['stage']==stage]
        stages[str(stage)]=dict(contacts=len(subset),without_Bomb=sum(c['bombs']==0 for c in subset),
            high_neural_risk=sum(c['selected_p4']>=.2 for c in subset),
            no_buffered_four_frame_direction=sum(c['safe_alternatives4']==0 for c in subset),
            covered_forecast_contradictions=sum(c['covered_forecast_safe_but_actual_contact'] for c in subset),
            contacts_with_earlier_warning=sum(any(w['frames_before_last_recorded_action']>0
                for w in c['route_warning_attribution']['warnings']) for c in subset))
    return dict(schema='compact-live-training-contact-review-v1',source=str(path),
        scope='Changing-policy training contacts only;not frozen tests. Alternative forecasts do not prove native alternate survival.',
        bytes_read=size,legacy_events_without_policy_context=legacy,trailing_record_incomplete=partial,
        nonfinite_rows_omitted=invalid,contacts=len(contacts),stages=stages,contact_records=contacts,
        CUDA_initialized=torch.cuda.is_initialized(),learner_modified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    result=analyze(args.events);write_json(args.output,result)
    print(dict(contacts=result['contacts'],stages=result['stages'],
               legacy_events_without_policy_context=result['legacy_events_without_policy_context']))
