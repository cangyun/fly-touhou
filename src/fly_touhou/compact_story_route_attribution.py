"""Attribute recorded route warnings without rerunning or changing a policy.

Margins describe constant-action geometric forecasts. They are not measured
alternate trajectories; these categories locate disagreements, not causation.
"""
import torch


def route_warning_attribution(rows, *, risk_scale=1.):
    warnings = []
    coverage = {str(h): dict(eligible_rows=0, trusted_rows=0, untrusted_rows=0,
        untrusted_with_near_growth=0, untrusted_with_unknown_lasers=0,
        untrusted_reason_unavailable=0) for h in (4,12,24)}
    if not rows:
        return dict(recorded_decisions=0, recorded_span_frames=0, warnings=[],
            forecast_coverage_by_horizon=coverage)
    last_frame = rows[-1]['context']['episode_frames']
    for row in rows:
        action = int(row['action'])
        if not row['normal'] or action >= 18:
            continue
        features = row['features'].float()
        # Only attribute route selection during vulnerable, controllable play.
        # A protected movement can deliberately cross otherwise unsafe space.
        if features[14] <= .5 or features[13] > 0 or features[5] > 0:
            continue
        margins = row['margins'].float()
        probability = (8 * row['horizon_scores'].float() - 1).sigmoid()
        cost = -(probability * probability.new_tensor((8., 4., 2.))).sum(-1) * risk_scale
        logits = row['logits'].float()[:18]
        for h, horizon in enumerate((4, 12, 24)):
            count=coverage[str(horizon)]
            count['eligible_rows']+=1
            if row['trusted'][h]:
                count['trusted_rows']+=1
            else:
                count['untrusted_rows']+=1
                uncertainty=row.get('uncertainty')
                if not isinstance(uncertainty,dict):
                    count['untrusted_reason_unavailable']+=1
                else:
                    growth=uncertainty.get('near_growth_points_4_12_24')
                    count['untrusted_with_near_growth']+=int(growth is not None and growth[h]>0)
                    count['untrusted_with_unknown_lasers']+=int(uncertainty.get('unknown_laser_segments',0)>0)
            safe = margins[h] > 1.
            if not row['trusted'][h] or margins[h, action] > 0 or not bool(safe.any()):
                continue
            # The highest-scoring forecast-safe alternative is the smallest
            # policy-score change that could escape this warned choice.
            alternative = int(logits.masked_fill(~safe, -torch.inf).argmax())
            risk_delta = float(cost[alternative] - cost[action])
            final_delta = float(logits[alternative] - logits[action])
            best_risk_safe = int(cost.masked_fill(~safe, -torch.inf).argmax())
            warnings.append(dict(
                episode_frame=row['context']['episode_frames'],
                frames_before_last_recorded_action=last_frame-row['context']['episode_frames'],
                horizon=horizon, selected_action=action,
                selected_margin=float(margins[h, action]),
                selected_predicted_risk=float(probability[action, h]),
                safe_alternatives=int(safe.sum()), alternative=alternative,
                alternative_margin=float(margins[h, alternative]),
                alternative_predicted_risk=float(probability[alternative, h]),
                selected_three_continuation_fraction=float(features[75+action]),
                alternative_three_continuation_fraction=float(features[75+alternative]),
                risk_logit_advantage=risk_delta, final_logit_advantage=final_delta,
                route_and_stay_logit_advantage=final_delta-risk_delta,
                risk_prefers_alternative_but_route_overrides=risk_delta>0 and final_delta<=0,
                sampled_below_best_safe_score=final_delta>0,
                selected_danger_not_alerted=float(probability[action, h])<.2,
                all_safe_alternatives_alerted=bool((probability[safe, h]>=.2).all()),
                best_risk_safe_action=best_risk_safe,
                best_risk_safe_advantage=float(cost[best_risk_safe]-cost[action]),
            ))
    summary = {}
    for horizon in (4, 12, 24):
        subset = [w for w in warnings if w['horizon']==horizon]
        summary[str(horizon)] = dict(
            trusted_warning_rows=len(subset),
            earliest_warning_lead_frames=max((w['frames_before_last_recorded_action'] for w in subset), default=None),
            risk_prefers_alternative_but_route_overrides=sum(w['risk_prefers_alternative_but_route_overrides'] for w in subset),
            selected_danger_not_alerted=sum(w['selected_danger_not_alerted'] for w in subset),
            all_safe_alternatives_alerted=sum(w['all_safe_alternatives_alerted'] for w in subset),
            sampled_below_best_safe_score=sum(w['sampled_below_best_safe_score'] for w in subset),
        )
    return dict(recorded_decisions=len(rows),
        recorded_span_frames=last_frame-rows[0]['context']['episode_frames'],
        forecast_coverage_by_horizon=coverage,
        summary_by_horizon=summary, warnings=warnings,
        scope='Trusted vulnerable geometric forecasts only;no trusted warnings does not imply no earlier escape when trust coverage is absent. Overlapping horizon rows are not independent contacts or proof of alternate native survival. Continuation fraction tests three second actions after4 frames for12 further frames;positive means a predicted turning route exists,zero does not exhaust all possible routes.')


def training_contact_lookback(data, *, decisions=64, risk_scale=1.):
    """Longer history from sealed rollouts; offsets are decisions, not frames.

    Never cross a reset. A tape-boundary history is explicitly left-censored.
    Teacher probabilities count as guidance only when their weight is positive.
    """
    contacts=[]
    actions=data['actions'];features=data['features'];margins=data['margins']
    # Training may veto a contradicted forecast after observing the outcome.
    # Diagnose the information available before acting, not that later veto.
    trusted=data.get('recorded_pre_action_trusted',data['trusted'])
    native_frames=data.get('recorded_episode_frames')
    for t,w in ((data['outcomes']==1)&(actions<18)).nonzero().tolist():
        first=max(0,t-decisions)
        starts=data['episode_starts'][first:t+1,w].nonzero().flatten()
        if len(starts):first+=int(starts[-1])
        rows=[]
        for index in range(first,t+1):
            rows.append(dict(context={'episode_frames':index},
                action=int(actions[index,w]),normal=bool(data['unprotected_mask'][index,w]),
                features=features[index,w],margins=margins[index,w],
                trusted=trusted[index,w].tolist(),
                horizon_scores=data['recorded_neural_horizon_scores'][index,w],
                logits=data['recorded_policy_logits'][index,w]))
        result=route_warning_attribution(rows,risk_scale=risk_scale)
        warnings=result['warnings']
        for warning in warnings:
            index=warning.pop('episode_frame')
            warning['rollout_decision_index']=index
            warning['decisions_before_contact']=warning.pop('frames_before_last_recorded_action')
            h=(4,12,24).index(warning['horizon'])
            safe=margins[index,w,h]>1.
            teacher=data['rule_probabilities'][index,w]
            weight=float(data['rule_weights'][index,w])
            warning.update(teacher_guidance_weight=weight,
                teacher_safe_mass_if_active=float(teacher[:18][safe].sum()) if weight>0 else None,
                teacher_selected_mass_if_active=float(teacher[int(actions[index,w])]) if weight>0 else None)
            if native_frames is not None:
                warning['episode_frame']=int(native_frames[index,w])
                warning['native_frames_before_contact']=int(native_frames[t,w]-native_frames[index,w])
        eligible=(data['unprotected_mask'][first:t+1,w] & (actions[first:t+1,w]<18)
                  & (features[first:t+1,w,14]>.5) & (features[first:t+1,w,13]<=0)
                  & (features[first:t+1,w,5]<=0))
        guided=eligible&(data['rule_weights'][first:t+1,w]>0)
        contacts.append(dict(time=t,world=w,
            stage=int((features[t,w,111]*6).round()),
            history_decisions=t-first,
            left_censored_by_tape=first==0 and not bool(data['episode_starts'][0,w]),
            max_margin4_at_contact=float(margins[t,w,0].max()),
            forecast_coverage_by_horizon=result['forecast_coverage_by_horizon'],
            summary_by_horizon=result['summary_by_horizon'],
            eligible_history_rows=int(eligible.sum()),teacher_active_history_rows=int(guided.sum()),
            teacher_abstained_history_rows=int((eligible&~guided).sum()),
            contact_episode_frame=int(native_frames[t,w]) if native_frames is not None else None,
            warnings=warnings))
    return dict(scope='Changing-policy training contacts. History offsets count decisions,not measured native frames. Alternative directions are constant-action forecasts;not native counterfactual outcomes.',
        history_limit_decisions=decisions,neural_risk_scale=risk_scale,
        uses_recorded_pre_action_trust='recorded_pre_action_trusted' in data,
        contacts=len(contacts),contacts_with_earlier_warning=sum(any(x['decisions_before_contact']>0 for x in c['warnings']) for c in contacts),
        contacts_with_warning_more_than8_decisions_earlier=sum(any(x['decisions_before_contact']>8 for x in c['warnings']) for c in contacts),
        contact_records=contacts)
