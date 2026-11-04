"""Check restrictive one-turn teacher labels against recorded adaptive paths.

Uses sealed real tails only. A certified observed path is an existence proof
for its chosen first action under the same current-motion proxy, not a native
counterfactual and not proof about the cause of a much later collision.
"""
from pathlib import Path
from collections import Counter
import argparse,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.thrl_dodge_diagnostics import _shared_binary
from fly_touhou.thrl_native_trace_replay import RecordedAvoidanceCase
from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher,normal_unprotected
from fly_touhou.atomic_json_io import atomic_write_json as write


def run(args):
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    if args.output.exists():raise FileExistsError(args.output)
    audit=read(args.route_audit);teacher=MovementSafetyTeacher()
    with _shared_binary(args.probe/'episodes.jsonl') as stream:raw=stream.read()
    raw=raw[:raw.rfind(b'\n')+1];episodes={(r['policy'],r['game']):r for r in map(json.loads,raw.splitlines())}
    checked=0;reasons=Counter();results=[]
    for row in audit['results']:
        def one_turn_margin(state):
            key='chosen_one_turn_margin' if 'chosen_one_turn_margin' in state else 'chosen_one_turn_route_margin'
            return state[key]
        interesting=[r for r in row['rows'] if r['chosen_immediate_margin']>1 and one_turn_margin(r)<=1]
        if not interesting:continue
        case=RecordedAvoidanceCase(args.probe/'report.json',episodes[(row['policy'],row['game'])]);z=case.arrays
        if case.source_trace_sha256!=row['source_trace_sha256']:raise ValueError('Route source changed')
        for candidate in interesting:
            t=candidate['tail_index'];checked+=1
            if t+6>=case.tail_count:reasons['observed24_window_missing']+=1;continue
            f,o=z['features'][t],z['objects'][t]
            if (not all(normal_unprotected(z['features'][j]) for j in range(t,t+7))
                    or any(int(z['native_frame'][j+1])-int(z['native_frame'][j])!=4 for j in range(t,t+6))
                    or any(z['native_episode'][j]!=z['native_episode'][t] or z['stage'][j]!=z['stage'][t] for j in range(t,t+7))):
                reasons['native_window_not_normal_same_episode_and_stage']+=1;continue
            if not np.allclose(z['features'][t:t+7,13:18],f[13:18],rtol=0,atol=2e-6):
                reasons['player_geometry_changed']+=1;continue
            position=teacher._scene(f,o)[0];margins=[];errors=[];actions=[]
            for j in range(6):
                action=int(z['action'][t+j]);actions.append(action)
                if not 0<=action<18:raise ValueError('Recorded no-Bomb source action invalid')
                velocity=teacher.velocities[action]
                margins.append(float(teacher.clearances(f,o,position[None],velocity[None],duration=4,initial_time=4*j)[0]))
                position=np.clip(position+velocity*4,[-184,32],[184,432])
                actual=teacher._scene(z['features'][t+j+1],z['objects'][t+j+1])[0]
                errors.append(float(np.linalg.norm(position-actual)))
            if max(errors)>.05:reasons['player_path_transport_error_over0.05']+=1;continue
            if min(margins)-max(errors)<=1:reasons['actual_adaptive_proxy_path_not_buffered']+=1;continue
            safe,detail=teacher.label(f,o,z['maps'][t],force=True)
            restrictive=bool(safe.any()) and not bool(safe[actions[0]])
            results.append(dict(policy=row['policy'],game=row['game'],tail_index=t,
                before_failure_native_frames=candidate['before_failure_native_frames'],
                source_trace_sha256=case.source_trace_sha256,observed_next_six_actions=actions,
                all_next24_normal_unprotected=True,one_turn_best_margin=one_turn_margin(candidate),
                adaptive_current_motion_proxy_margin=min(margins),max_player_path_prediction_error=max(errors),
                conservative_buffer_after_path_error=min(margins)-max(errors),
                teacher_reason=detail['reason'],teacher_safe_action_count=int(safe.sum()),
                restrictive_teacher_mask_excludes_observed_buffered_adaptive_path=restrictive))
    counts=audit['summary']
    summary=dict(source_sealed_failures=len(audit['results']),source_qualified24_states=(counts['qualified_states']
        if 'qualified_states' in counts else counts['qualified_24_frame_states']),
        selected_one_turn_margin_at_most1=checked,abstentions=dict(reasons),
        certified_actual_buffered_adaptive_continuations=len(results),
        restrictive_mask_conflicts=sum(r['restrictive_teacher_mask_excludes_observed_buffered_adaptive_path'] for r in results))
    args.output.mkdir(parents=True)
    write(args.output/'report.json',dict(schema='B1-observed-adaptive-vs-one-turn-label-consistency-v1',utc=utc_now(),
        summary=summary,results=results,partial_failed_subset_is_selection_biased=True,
        labels_or_training_not_modified=True,new_native_worlds=0,optimizer_updates=0,
        original_counterfactuals_executed=False,causal_explanation_of_PPO_regression_claimed=False))
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ['probe','route-audit','output']:p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
