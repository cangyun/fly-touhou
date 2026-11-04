"""Separate teacher entropy from recorded student movement cross entropy."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.compact_native_rule_replay import NativeRuleTeacherReplay
from fly_touhou.thrl_durable_records import write_json


def analyze(report_path: Path) -> dict:
    report=json.loads(report_path.read_text(encoding='utf-8'))
    data=NativeRuleTeacherReplay(report['source_manifest'],split='heldout')
    parent=report['reports']['parent']['rows'];candidate=report['reports']['candidate']['rows']
    selected=[]
    for stage in report['stages']:
        group=[i for i,row in enumerate(data.windows) if row['stage']==stage]
        count=sum(row['native_rule_stage']==stage for row in parent)
        selected.extend(group[round(i*(len(group)-1)/max(1,count-1))] for i in range(count))
    if not len(selected)==len(parent)==len(candidate):
        raise ValueError('Saved report and heldout window selection differ')
    by_stage=defaultdict(list)
    for index,old,new in zip(selected,parent,candidate):
        window=data.windows[index]
        if window['stage']!=old['native_rule_stage'] or window['stage']!=new['native_rule_stage']:
            raise ValueError('Saved report stages do not match selected windows')
        with np.load(data.path.parent/window['file'],allow_pickle=False) as archive:
            start=data.burn_in;features=archive['features'][start:]
            actions=archive['actions'][start:];weights=archive['teacher_guidance_weight'][start:]
            probabilities=archive['targets'][start:,:18].astype(np.float64)
        mass=probabilities.sum(-1)
        eligible=(weights>0)&(actions<18)&((features[:,14]>.5)|(features[:,17]>.5))&(mass>0)
        conditional=np.divide(probabilities,mass[:,None],out=np.zeros_like(probabilities),where=mass[:,None]>0)
        entropy=-(np.where(conditional>0,conditional*np.log(np.maximum(conditional,1e-300)),0.)).sum(-1)
        n=int(eligible.sum());weight=float(weights[eligible].sum())
        if n!=old['native_rule_movement_rows'] or n!=new['native_rule_movement_rows']:
            raise ValueError('Recorded eligible movement rows differ from the heldout report')
        H=float((entropy[eligible]*weights[eligible]).sum()/weight) if weight else None
        by_stage[window['stage']].append(dict(window_index=index,rows=n,weight=weight,
            teacher_entropy=H,parent_cross_entropy=old['native_rule_movement_loss'],
            candidate_cross_entropy=new['native_rule_movement_loss']))
    stages={}
    for stage,rows in sorted(by_stage.items()):
        den=sum(row['weight'] for row in rows)
        mean=lambda key:sum(row[key]*row['weight'] for row in rows if row['weight'])/den if den else None
        entropy=mean('teacher_entropy');old=mean('parent_cross_entropy');new=mean('candidate_cross_entropy')
        stages[str(stage)]=dict(windows=len(rows),teacher_rows=sum(row['rows'] for row in rows),
            teacher_entropy=entropy,parent_cross_entropy=old,candidate_cross_entropy=new,
            parent_teacher_KL=old-entropy if den else None,candidate_teacher_KL=new-entropy if den else None)
    return dict(report=str(report_path),stages=stages,
        scope=('Same teacher-controlled heldout-game windows as the saved frozen report; '
               'conditional 18-movement distributions. Teacher preferences and modeled risks '
               'are not counterfactual native outcomes; this is not a gameplay result.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.report)
    write_json(args.output,result)
    print(json.dumps(result,ensure_ascii=False))
