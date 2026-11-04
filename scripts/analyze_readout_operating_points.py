"""Describe the held-out Bomb/ordinary-movement tradeoff; no new fitting or game."""
from pathlib import Path
import argparse
import hashlib
import json
import math


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--negative-request-limit',type=float,default=.01)
    args=parser.parse_args()
    if args.output.exists() or not math.isfinite(args.negative_request_limit) or not 0<=args.negative_request_limit<1:
        raise ValueError('Exclusive output and a finite request-rate limit in [0,1) required')
    source_digest=digest(args.report)
    source=json.loads(args.report.read_text(encoding='utf-8'))
    if source.get('status')!='completed_readout_only_control':
        raise ValueError('Read only a completed readout experiment')
    candidates=[]
    for epoch in source['history']:
        metrics=epoch['validation']['by_difficulty']['Lunatic']
        groups=metrics['by_teacher_action_type']
        positive,negative=groups['bomb'],groups['movement']
        if min(positive['decisions'],negative['decisions'])<1:
            raise ValueError('Actual positive and negative held-out decisions required')
        tp=round(positive['decisions']*positive['predicted_bomb_fraction'])
        fp=round(negative['decisions']*negative['predicted_bomb_fraction'])
        candidates.append({'epoch':epoch['epoch'],'checkpoint':epoch['checkpoint'],
            'teacher_Bomb_decisions':positive['decisions'],'Bomb_requests_on_teacher_Bomb':tp,
            'teacher_movement_decisions':negative['decisions'],'Bomb_requests_on_teacher_movement':fp,
            'Bomb_type_recall':positive['predicted_bomb_fraction'],
            'negative_Bomb_request_fraction':negative['predicted_bomb_fraction'],
            'request_precision_against_teacher':tp/(tp+fp) if tp+fp else None,
            'movement_exact_agreement':negative['teacher_argmax_agreement'],
            'balanced_group_mixed_ce':.5*(positive['mixed_cross_entropy']+negative['mixed_cross_entropy']),
            'overall_mixed_ce':metrics['mixed_cross_entropy']})
    feasible=[v for v in candidates if v['negative_Bomb_request_fraction']<=args.negative_request_limit]
    specificity=max(feasible,key=lambda v:(v['Bomb_type_recall'],v['movement_exact_agreement'],
                                         -v['balanced_group_mixed_ce'],-v['epoch'])) if feasible else None
    selected=next(v for v in candidates if v['epoch']==source['best_epoch'])
    frontier=[v for v in candidates if not any(
        other['Bomb_type_recall']>=v['Bomb_type_recall']
        and other['negative_Bomb_request_fraction']<=v['negative_Bomb_request_fraction']
        and (other['Bomb_type_recall']>v['Bomb_type_recall']
             or other['negative_Bomb_request_fraction']<v['negative_Bomb_request_fraction'])
        for other in candidates)]
    for v in (selected,specificity):
        if v is not None:
            v['checkpoint_sha256']=digest(Path(v['checkpoint']))
    if digest(args.report)!=source_digest:
        raise RuntimeError('Source changed during analysis')
    report={'schema':'readout-heldout-Bomb-operating-points-v1','source':str(args.report.resolve()),
        'source_sha256':source_digest,'script_sha256':digest(Path(__file__)),
        'predeclared_selection':source['checkpoint_selection_kind'],'selected_checkpoint':selected,
        'exploratory_negative_request_limit':args.negative_request_limit,
        'exploratory_specificity_candidate':specificity,
        'Pareto_frontier':sorted(frontier,key=lambda v:(v['negative_Bomb_request_fraction'],v['epoch'])),
        'limits':['Teacher disagreement does not itself prove native Bomb waste or usefulness.',
                  'Specificity choice is exploratory on the same held-out set, requiring fresh native evaluation.',
                  'No new model, thresholds, physics, or game actions were fitted or changed.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('selected_checkpoint','exploratory_specificity_candidate')},ensure_ascii=False))


if __name__=='__main__':
    main()
