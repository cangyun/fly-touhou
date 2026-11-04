"""Describe risk coverage and executed contacts in a student training rollout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_durable_records import write_json


def analyze_tape(tape, *, source, recover_compact111_trust=False):
    features=tape['features']
    actions=tape['actions'].long()
    stage=(features[...,111]*6).round().long()
    outcome=tape['outcomes']
    elapsed=tape['native_action_frames']
    normal=tape['unprotected_mask'].bool()
    post_outcome_trusted=tape['trusted'][...,0].bool()
    margins=tape['margins'][...,0,:]
    movement=(actions>=0)&(actions<18)
    chosen=margins.gather(-1,actions.clamp(0,17)[...,None])[...,0]
    raw_trusted=tape.get('recorded_pre_action_trusted',tape.get('pre_action_trusted'))
    trust_source='recorded' if raw_trusted is not None else 'unavailable'
    if raw_trusted is None and recover_compact111_trust:
        # In the verified compact111 encoder, slots 72..74 are the same
        # three pre-action known flags emitted in compact_targets. Confirm
        # that outcome invalidation exactly reproduces the recorded mask.
        flags=features[...,72:75]
        if flags.shape[-1]!=3 or not bool(((flags==0)|(flags==1)).all()):
            raise ValueError('Compact111 forecast-known features are not binary')
        raw_trusted=flags.bool()
        contradicted=(normal&movement&(((outcome==1)&(chosen>0))|((outcome==0)&(chosen<0))))
        expected=raw_trusted&~contradicted[...,None]
        if not torch.equal(expected,tape['trusted'].bool()):
            raise ValueError('Compact111 flags do not reproduce stored training trust masks')
        trust_source='recovered_from_compact111_features'
    pre_action_trusted=raw_trusted[...,0].bool() if raw_trusted is not None else None
    mixed=(margins.le(0).any(-1)&margins.gt(1).any(-1))
    rule=tape['rule_weights']>0
    deathbomb=tape['deathbomb_mask'].bool()
    all_contacts=tape.get('recorded_observed_contacts',tape.get('observed_contacts'))
    x=features[...,0]*368-184
    y=features[...,1]*400+32
    wall=torch.minimum(torch.minimum(x+184,184-x),torch.minimum(y-32,432-y)).clamp_min(0)
    after_wall=features[...,93:111]*64
    inward_hint=((after_wall>wall[...,None]+3)&(tape['margins'][...,0,:]>4)
                 &(tape['margins'][...,1,:]>1))
    weak_inward=normal&~rule&inward_hint.any(-1)
    groups={}
    for number in range(1,7):
        mask=stage==number
        if not bool(mask.any()):
            continue
        measured=mask&normal&movement&(elapsed>0)&(elapsed<=4)&(outcome==1)
        retained=mask&normal&movement&post_outcome_trusted&mixed
        bomb_requests=mask&(actions==18)
        mixed_states=(mask&normal&pre_action_trusted&mixed
                      if pre_action_trusted is not None else None)
        comparable=(mask&normal&movement&pre_action_trusted&mixed
                    if pre_action_trusted is not None else None)
        groups[str(number)]=dict(decisions=int(mask.sum()),
            normal_movement_decisions=int((mask&normal&movement).sum()),
            observed_contacts_all_actions=(int((mask&all_contacts.bool()).sum())
                                           if all_contacts is not None else None),
            observed_contacts=int(measured.sum()),
            observed_contacts_pre_action_forecast_untrusted=(
                int((measured&~pre_action_trusted).sum()) if pre_action_trusted is not None else None),
            observed_contacts_with_trusted_safe_direction=(
                int((measured&pre_action_trusted&margins.gt(1).any(-1)).sum())
                if pre_action_trusted is not None else None),
            bomb_action_requests=int(bomb_requests.sum()),
            bomb_requests_during_observed_deathbomb=int((bomb_requests&deathbomb).sum()),
            bomb_requests_pre_action_forecast_untrusted=(
                int((bomb_requests&~pre_action_trusted).sum()) if pre_action_trusted is not None else None),
            bomb_requests_with_trusted_safe_direction=(
                int((bomb_requests&pre_action_trusted&margins.gt(1).any(-1)).sum())
                if pre_action_trusted is not None else None),
            observed_deathbomb_windows=int((mask&deathbomb).sum()),
            teacher_guidance_rows=int((mask&rule).sum()),
            post_outcome_retained_mixed_rows=int(retained.sum()),
            pre_action_mixed_direction_rows=int(comparable.sum()) if comparable is not None else None,
            selected_pre_action_modeled_danger_rows=(int((comparable&(chosen<=0)).sum())
                                                     if comparable is not None else None))
        contact_details=[]
        recorded_scores=tape.get('recorded_neural_horizon_scores')
        for time_index,world in measured.nonzero().tolist():
            action=int(actions[time_index,world])
            row=dict(time_index=time_index,world=world,action=action,
                bomb_available=bool(features[time_index,world,4]>0),
                teacher_guidance_weight=float(tape['rule_weights'][time_index,world]),
                forecast_trusted=(bool(pre_action_trusted[time_index,world])
                                  if pre_action_trusted is not None else None))
            if recorded_scores is not None:
                scores4=recorded_scores[time_index,world,:,0]
                selected=scores4[action]
                row.update(neural_risk_signal=float(torch.sigmoid(8*selected-1)),
                    lowest_neural_risk_signal=float(torch.sigmoid(8*scores4.min()-1)),
                    neural_risk_rank=int((scores4<selected).sum())+1)
            behavior=tape.get('recorded_behavior_policy_logits')
            if behavior is not None:
                probabilities=torch.softmax(behavior[time_index,world].float(),dim=-1)
                row['bomb_behavior_probability']=float(probabilities[18])
            starts=tape.get('episode_starts')
            for lookback in (32,64):
                prior=time_index-lookback
                if (prior<0 or stage[prior,world]!=number or
                    (starts is not None and bool(starts[prior+1:time_index+1,world].any()))):
                    continue
                x=float(features[prior,world,0])*368-184
                y=float(features[prior,world,1])*400+32
                row[f'wall_distance_{lookback}_decisions_before']=min(x+184,184-x,y-32,432-y)
                row[f'bomb_requests_previous_{lookback}_decisions']=int(
                    (actions[prior:time_index,world]==18).sum())
            prior=max(0,time_index-64)
            if starts is not None:
                boundaries=starts[prior+1:time_index+1,world].nonzero().flatten()
                if len(boundaries):prior+=int(boundaries[-1])+1
            segment=slice(prior,time_index)
            prev_action=actions[segment,world]
            prev_wall=wall[segment,world]
            candidate=weak_inward[segment,world]&(stage[segment,world]==number)
            actual_wall=after_wall[segment,world].gather(-1,prev_action.clamp(0,17)[:,None])[:,0]
            row['previous_64_inward_heuristic']={str(limit):dict(
                possible_inward_hints=int((candidate&(prev_wall<limit)).sum()),
                chosen_toward_wall=int((candidate&(prev_wall<limit)&(prev_action<18)
                                        &(actual_wall<prev_wall-1)).sum()))
                for limit in (8,32,48)}
            contact_details.append(row)
        groups[str(number)]['contact_details']=contact_details
        if recorded_scores is not None and bool(bomb_requests.any()):
            minimum=torch.sigmoid(8*recorded_scores[bomb_requests][...,0].min(-1).values-1)
            groups[str(number)]['bomb_requests_min_neural_risk_below_0_2']=int((minimum<.2).sum())
            groups[str(number)]['bomb_requests_min_neural_risk_over_0_8']=int((minimum>.8).sum())
        if mixed_states is not None and bool(mixed_states.any()):
            danger=margins[mixed_states]<=0
            safe=margins[mixed_states]>1
            groups[str(number)]['mixed_states_including_bomb_actions']=int(mixed_states.sum())
            groups[str(number)]['uniform_movement_danger_probability']=float(danger.float().mean())
            logits=tape.get('recorded_behavior_policy_logits')
            if logits is not None:
                probabilities=torch.softmax(logits[mixed_states][...,:18].float(),dim=-1)
                groups[str(number)]['policy_movement_danger_probability']=float((probabilities*danger).sum(-1).mean())
            scores=tape.get('recorded_neural_horizon_scores')
            if scores is not None:
                prediction=scores[mixed_states][...,0]
                pairs=danger[:,:,None]&safe[:,None,:]
                ordered=(prediction[:,:,None]>prediction[:,None,:]).float()
                ties=(prediction[:,:,None]==prediction[:,None,:]).float()
                groups[str(number)]['risk_danger_safe_pairs']=int(pairs.sum())
                groups[str(number)]['risk_pair_order_accuracy']=float(((ordered+.5*ties)*pairs).sum()/pairs.sum())
    return dict(rollout=str(source),stages=groups,
        pre_action_trust_source=trust_source,
        scope=('Student-controlled training rollout, not independent evaluation. '
               'Contact counts cover executed movements only. '
               'Post-outcome trust masks exclude contradictory outcomes and cannot measure avoidance. '
               'Pre-action direction risks are forecasts, not native counterfactuals. '
               'Neural risk signals are uncalibrated model outputs. '
               'Inward heuristic counts use untrusted modeled margins and do not certify safe alternatives.'))


def analyze(path: Path, *, recover_compact111_trust=False):
    return analyze_tape(torch.load(path,map_location='cpu',weights_only=False),source=path,
                        recover_compact111_trust=recover_compact111_trust)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--recover-compact111-trust',action='store_true',
        help='Recover verified pre-action flags from compact111 input slots 72..74')
    args=parser.parse_args()
    result=analyze(args.rollout,recover_compact111_trust=args.recover_compact111_trust)
    write_json(args.output,result)
    print(json.dumps(dict(rollout=result['rollout'],trust_source=result['pre_action_trust_source'],
        stages={stage:{key:values[key] for key in ('decisions','observed_contacts',
            'observed_contacts_pre_action_forecast_untrusted','bomb_action_requests')}
                for stage,values in result['stages'].items()}),ensure_ascii=False))


if __name__=='__main__':
    main()
