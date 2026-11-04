"""Look backwards through real failure tails for earlier modeled route loss.

Uses recorded actions/probabilities. Historical maps were not saved: at object
capacity those historical states abstain. Model paths allow one turn only and
are hypotheses about current motion, not executed native alternatives.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from audit_b1_failure_risk_attribution import failure_cases
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_native_trace_replay import RecordedAvoidanceCase
from fly_touhou.thrl_failure_trace_audit import classify
from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher, normal_unprotected


def run(args):
    if args.output.exists():
        raise ValueError('Preserve earlier route audits')
    report = read(args.report)
    if (report.get('status') != 'completed'
            or read(args.report.parent/'cleanup.json') != dict(closed=True, close_errors=[])):
        raise ValueError('Completed closed native source required')
    selected, failed = failure_cases(report)
    teacher = MovementSafetyTeacher()
    results = []
    for episode in failed:
        case = RecordedAvoidanceCase(args.report, episode)
        if case.receipt['checkpoint_sha256'] != report['pins']['trained']['sha256']:
            raise ValueError('Trace does not belong to the declared frozen policy')
        z = case.arrays
        full_history_maps = 'maps' in z
        if full_history_maps and (case.receipt.get('earlier_history_full_map_coverage_known') is not True
                or z['maps'].shape != (case.tail_count,16,92,96) or z['maps'].dtype != np.float32
                or not np.array_equal(z['maps'][-1],z['last_preaction_maps'])):
            raise ValueError('Historical maps must be the declared exact paired FP32 capture')
        final = classify(z, case.receipt)
        reasons = Counter()
        rows = []
        native_episode = int(z['native_episode'][-1])
        stage = int(z['stage'][-1])
        failure_preframe = int(z['native_frame'][-1])
        for t, (f, o, p, action) in enumerate(zip(z['features'], z['objects'], z['probabilities'], z['action'])):
            action = int(action)
            if (int(z['native_episode'][t]) != native_episode or int(z['stage'][t]) != stage
                    or int(z['native_frame'][t]) < 0):
                reasons['other_episode_stage_or_unknown_clock'] += 1
                continue
            if not normal_unprotected(f):
                reasons['protected_or_non_normal'] += 1
                continue
            if t+1 < len(z['action']) and int(z['native_frame'][t+1])-int(z['native_frame'][t]) != 4:
                reasons['next_observation_not_four_native_frames'] += 1
                continue
            present = int((o[:,0] > .5).sum())
            maps = z['maps'][t] if full_history_maps else z['last_preaction_maps'] if t == len(z['action'])-1 else None
            if maps is None and present >= 96:
                reasons['historical_full_maps_missing_at_object_capacity'] += 1
                continue
            unknown = teacher._uncertainty(f, o, maps, horizon=24)
            if unknown:
                reasons[unknown] += 1
                continue
            origin = teacher._scene(f, o)[0]
            starts = np.tile(origin, (18, 1))
            immediate = teacher.clearances(f, o, starts, teacher.velocities, duration=4)
            ends = np.clip(starts+teacher.velocities*4, [-184,32], [184,432])
            future = teacher.clearances(f, o, np.repeat(ends,18,axis=0),
                np.tile(teacher.velocities,(18,1)), duration=20, initial_time=4).reshape(18,18)
            routes = np.minimum(immediate[:,None], future)
            best_route = routes.max(-1)
            if np.isnan(best_route).any() or not 0 <= action < 18 or p.shape != (18,):
                raise ValueError('Invalid paired recorded action/route geometry')
            buffered = best_route > 1
            candidate = (t < len(z['action'])-1 and immediate[action] > 0
                         and best_route[action] <= 0 and buffered.any())
            rows.append(dict(tail_index=t,
                before_failure_native_frames=failure_preframe-int(z['native_frame'][t]),
                episode_frames=int(z['episode_frame'][t]), present_objects=present,
                coverage='stored full map' if maps is not None else 'padding slot;all supplied sensor entities fit',
                action=action, actual_action_probability=float(p[action]),
                immediate_margins_clipped64=np.clip(immediate,-64,64).tolist(),
                best_one_turn_route_margins_clipped64=np.clip(best_route,-64,64).tolist(),
                best_continuations=routes.argmax(-1).tolist(),
                chosen_immediate_margin=float(np.clip(immediate[action],-64,64)),
                chosen_one_turn_route_margin=float(np.clip(best_route[action],-64,64)),
                buffered_one_turn_first_actions=int(buffered.sum()),
                actual_policy_mass_on_buffered_one_turn_routes=float(p[buffered].sum()),
                actual_policy_mass_immediate_safe_but_no_modeled_route=float(p[(immediate>1)&(best_route<=0)].sum()),
                modeled_earlier_route_loss_candidate=bool(candidate)))
        candidates = [row for row in rows if row['modeled_earlier_route_loss_candidate']]
        results.append(dict(policy=episode['policy'],game=episode['game'],native_seed=episode['native_seed'],
            source_trace_sha256=case.source_trace_sha256, source_action_tape_sha256=case.action_sha256,
            final_category=final['category'], total_tail_rows=case.tail_count,
            actual_full_map_history=full_history_maps,
            qualified_24_frame_states=len(rows), abstentions=dict(reasons),
            earlier_route_loss_candidates=len(candidates),
            closest_route_loss_candidate=min(candidates,key=lambda row:row['before_failure_native_frames']) if candidates else None,
            rows=rows))
    summary = dict(selected_trained_games=len(selected), observed_failures=len(failed),
        trace_states=sum(row['total_tail_rows'] for row in results),
        qualified_24_frame_states=sum(row['qualified_24_frame_states'] for row in results),
        cases_with_qualified_states=sum(row['qualified_24_frame_states'] > 0 for row in results),
        cases_with_modeled_earlier_route_loss=sum(row['earlier_route_loss_candidates'] > 0 for row in results),
        final_no_immediate_escape_cases=sum(row['final_category']=='no_modeled_immediate_escape' for row in results),
        final_no_escape_with_earlier_route_loss=sum(row['final_category']=='no_modeled_immediate_escape'
            and row['earlier_route_loss_candidates'] > 0 for row in results))
    args.output.mkdir(parents=True)
    result = dict(schema='B1-actual-failure-tail-earlier-modeled-route-audit-v1', status='completed', utc=utc_now(),
        source_report=str(args.report), source_report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
        source_checkpoint_sha256=report['pins']['trained']['sha256'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), summary=summary, results=results,
        input_schema=report.get('input_schema'),
        recorded_policy_probabilities_used=True, earlier_maps_fabricated=False,
        original_counterfactuals_executed=False, optimizer_updates=0, native_instances_started=0,
        future_emissions_predicted=False, modeled_paths_allow_one_turn_only=True,
        scope='Qualified current-motion route hypotheses on recorded real failure tails;not proof that native alternatives survive')
    write(args.output/'report.json', result)
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
