"""Read recorded native contact events without running or changing a policy.

Alternative-action margins are geometric forecasts, not native outcomes.
Works on an atomic snapshot of completed trace files during an evaluation.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.compact_story_route_attribution import route_warning_attribution


def analyze(evaluation, *, risk_scale=1.):
    paths = sorted((evaluation / 'diagnostics').glob('event-*.pt'))
    contacts = []
    for path in paths:
        item = torch.load(path, map_location='cpu', weights_only=False)
        event = item['summary']
        if 'contact_window_entered' not in event['kinds']:
            continue
        row = item['pre_action_tail'][-1]
        action = row['action']
        if not row['normal'] or action >= 18:
            continue
        p = (8 * row['horizon_scores'].float() - 1).sigmoid()
        f = row['features'].float()
        remaining = 60 * f[13] / (1 - f[13]).clamp_min(1e-6)
        horizon = torch.tensor((4., 12., 24.))
        exposure = ((horizon - remaining) / horizon).clamp(0, 1) * (f[14] > .5)
        cost = -(p * p.new_tensor((8., 4., 2.)) * exposure).sum(-1) * risk_scale
        margin = row['margins'].float()
        safe = margin[0] > 1
        safe_count = int(safe.sum())
        logits = row['logits'].float()
        alternative = int(logits[:18].masked_fill(~safe, -torch.inf).argmax()) if safe_count else None
        risk_only = int(cost.argmax())
        x = float(f[0] * 368 - 184); y = float(f[1] * 400 + 32)
        edge_x = 184 - abs(x); edge_y = min(y - 32, 432 - y)
        details = dict(event=event['event'], world=event['world'],
            stage=row['context']['stage'], episode_frame=row['context']['episode_frames'],
            action=action, bombs_remaining=row['context']['bombs'],
            selected_p=p[action].tolist(), selected_margin=margin[:, action].tolist(),
            trusted=row['trusted'], safe_forecast_alternatives=safe_count,
            max_margin4=float(margin[0].max()), best_safe_logit_action=alternative,
            risk_only_action=risk_only, risk_only_margin4=float(margin[0, risk_only]),
            minimum_p4_action=int(p[:, 0].argmin()),
            minimum_p4=float(p[:, 0].min()), exposure=exposure.tolist(),
            player_position=[x, y], within16_of_edge=min(edge_x, edge_y) <= 16,
            within16_of_corner=max(edge_x, edge_y) <= 16,
            unknown_laser_segments=(event.get('pre_action_uncertainty') or {}).get('unknown_laser_segments',0),
            risk_favors_safe_but_route_overrides=False)
        if alternative is not None:
            # A common movement-logit normalizer cancels in these differences.
            risk_delta = float(cost[alternative] - cost[action])
            logit_delta = float(logits[alternative] - logits[action])
            details.update(safe_alternative_risk_advantage=risk_delta,
                safe_alternative_final_logit_advantage=logit_delta,
                safe_alternative_route_and_stay_advantage=logit_delta - risk_delta,
                risk_favors_safe_but_route_overrides=risk_delta > 0 and logit_delta <= 0)
        lookback = []
        for earlier in item['pre_action_tail'][:-1]:
            a = earlier['action']; m = earlier['margins'].float()
            if not earlier['normal'] or a >= 18:
                continue
            for h, ticks in enumerate((4, 12, 24)):
                if earlier['trusted'][h] and m[h, a] <= 0 and bool((m[h] > 1).any()):
                    lookback.append(dict(episode_frame=earlier['context']['episode_frames'],
                        horizon=ticks, action=a, selected_margin=float(m[h, a]),
                        safe_alternatives=int((m[h] > 1).sum())))
        details['earlier_trusted_forecast_warnings_with_alternative'] = lookback
        details['route_warning_attribution'] = route_warning_attribution(item['pre_action_tail'], risk_scale=risk_scale)
        contacts.append(details)
    summary = dict(native_vulnerable_movement_contacts=len(contacts),
        contacts_p4_ge02=sum(c['selected_p'][0] >= .2 for c in contacts),
        contacts_with_forecast_safe_alternative=sum(c['safe_forecast_alternatives'] > 0 for c in contacts),
        risk_favors_safe_but_route_overrides=sum(c['risk_favors_safe_but_route_overrides'] for c in contacts),
        no_positive_constant_action_margin=sum(c['max_margin4'] <= 0 for c in contacts),
        narrow_only_margin0_to1=sum(0 < c['max_margin4'] <= 1 for c in contacts),
        risk_only_forecast_safe=sum(c['risk_only_margin4'] > 1 for c in contacts),
        contacts_within16_of_edge=sum(c['within16_of_edge'] for c in contacts),
        contacts_within16_of_corner=sum(c['within16_of_corner'] for c in contacts),
        contacts_with_earlier_trusted_warning=sum(bool(c['earlier_trusted_forecast_warnings_with_alternative']) for c in contacts))
    summary['route_warning_contacts_by_horizon'] = {
        str(h): {key:sum(c['route_warning_attribution']['summary_by_horizon'][str(h)][key]>0 for c in contacts)
                 for key in ('trusted_warning_rows','risk_prefers_alternative_but_route_overrides',
                             'selected_danger_not_alerted','all_safe_alternatives_alerted')}
        for h in (4,12,24)}
    # Flat scalar names are picked up by the existing TensorBoard exporter.
    # Count only the vulnerable movement contacts represented in `contacts`.
    for stage in (3,4):
        subset=[c for c in contacts if c['stage']==stage]
        summary.update({
            f'stage{stage}_vulnerable_movement_contacts':len(subset),
            f'stage{stage}_contacts_within16_of_edge':sum(c['within16_of_edge'] for c in subset),
            f'stage{stage}_contacts_without_Bomb':sum(c['bombs_remaining']==0 for c in subset),
            f'stage{stage}_contacts_with_unknown_laser_segments':sum(c['unknown_laser_segments']>0 for c in subset),
            f'stage{stage}_contacts_without_predicted_safe_direction':sum(c['safe_forecast_alternatives']==0 for c in subset),
            f'stage{stage}_contacts_with_earlier_trusted_warning':sum(bool(c['earlier_trusted_forecast_warnings_with_alternative']) for c in subset),
        })
    return dict(schema='compact-story-recorded-native-contact-choices-v1', evaluation=str(evaluation),
        complete=(evaluation / 'report.json').is_file(), trace_files_in_snapshot=len(paths),
        risk_scale=risk_scale, protection_aware_risk=True, finite_risk_composition=True,
        scope='Native contacts are observed. Alternative margins are current-state constant-action forecasts, not native counterfactuals. An empty short lookback does not prove no earlier escape route.',
        source='Recorded FP32 scores/logits; no neural replay, optimizer or game launch.',
        summary=summary, contacts=contacts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--risk-scale', type=float, default=1.)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = analyze(args.evaluation, risk_scale=args.risk_scale)
    write_json(args.output, result)
    print(dict(complete=result['complete'], **result['summary']), flush=True)


if __name__ == '__main__':
    main()
