"""Audit sealed actual tails while a larger frozen native comparison continues.

No aggregate policy score: completed failures are an early-stop biased subset.
Reuse the same four/24-frame proxy as the complete earlier-route audit.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.thrl_dodge_diagnostics import _shared_binary
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_native_trace_replay import RecordedAvoidanceCase
from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher,normal_unprotected
from fly_touhou.thrl_failure_trace_audit import classify


def audit_case(case,teacher):
    z=case.arrays
    if ('maps' not in z or case.receipt.get('earlier_history_full_map_coverage_known') is not True):
        raise ValueError('Use actual complete historical maps')
    reasons=Counter();rows=[]
    episode=int(z['native_episode'][-1]);stage=int(z['stage'][-1]);end=int(z['native_frame'][-1])
    for t,(f,o,p,action) in enumerate(zip(z['features'],z['objects'],z['probabilities'],z['action'])):
        action=int(action)
        if int(z['native_episode'][t])!=episode or int(z['stage'][t])!=stage or int(z['native_frame'][t])<0:
            reasons['other_episode_stage_or_unknown_clock']+=1;continue
        if not normal_unprotected(f):reasons['protected_or_non_normal']+=1;continue
        if t+1<case.tail_count and int(z['native_frame'][t+1])-int(z['native_frame'][t])!=4:
            reasons['next_observation_not_four_native_frames']+=1;continue
        unknown=teacher._uncertainty(f,o,z['maps'][t],horizon=24)
        if unknown:reasons[unknown]+=1;continue
        origin=teacher._scene(f,o)[0];starts=np.tile(origin,(18,1))
        immediate=teacher.clearances(f,o,starts,teacher.velocities,duration=4)
        ends=np.clip(starts+teacher.velocities*4,[-184,32],[184,432])
        future=teacher.clearances(f,o,np.repeat(ends,18,axis=0),np.tile(teacher.velocities,(18,1)),
            duration=20,initial_time=4).reshape(18,18)
        routes=np.minimum(immediate[:,None],future);best=routes.max(-1);buffered=best>1
        if np.isnan(best).any() or not 0<=action<18 or p.shape!=(18,):raise ValueError('Invalid actual action/proxy pair')
        candidate=t<case.tail_count-1 and immediate[action]>0 and best[action]<=0 and buffered.any()
        rows.append(dict(tail_index=t,before_failure_native_frames=end-int(z['native_frame'][t]),
            action=action,chosen_immediate_margin=float(np.clip(immediate[action],-64,64)),
            chosen_one_turn_margin=float(np.clip(best[action],-64,64)),
            buffered_first_actions=int(buffered.sum()),recorded_mass_on_buffered_routes=float(p[buffered].sum()),
            modeled_earlier_route_loss_candidate=bool(candidate),
            best_continuations=routes.argmax(-1).tolist(),best_route_margins_clipped64=np.clip(best,-64,64).tolist()))
    return dict(policy=case.row['policy'],game=case.row['game'],native_seed=case.row['native_seed'],
        final_category=classify(z,case.receipt)['category'],actual_full_map_history=True,
        source_trace_sha256=case.source_trace_sha256,action_tape_sha256=case.action_sha256,
        tail_rows=case.tail_count,qualified_states=len(rows),abstentions=dict(reasons),
        earlier_route_loss_candidates=sum(row['modeled_earlier_route_loss_candidate'] for row in rows),rows=rows)


def run(args):
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    if args.output.exists() or not 1<=args.per_policy<=20:raise ValueError('Fresh bounded read-only audit required')
    progress=read(args.probe/'progress.json')
    with _shared_binary(args.probe/'episodes.jsonl') as stream:raw=stream.read()
    raw=raw[:raw.rfind(b'\n')+1];episodes=[json.loads(line) for line in raw.splitlines()]
    selected=[]
    for policy in ('trained_sampled','trained_greedy'):
        selected.extend(sorted((row for row in episodes if row['policy']==policy
            and row['first_collision_observed'] and row.get('failure_trace')),key=lambda row:row['game'])[:args.per_policy])
    if not selected:raise ValueError('No complete sealed trained failure')
    teacher=MovementSafetyTeacher();results=[]
    for row in selected:
        # The constructor uses this authoritative directory for path checks;
        # it reads sealed receipts/tapes, not a fabricated completed report.
        case=RecordedAvoidanceCase(args.probe/'report.json',row)
        if case.receipt['checkpoint_sha256']!=progress['pins']['trained']['sha256']:
            raise ValueError('Not the pinned frozen trained model')
        results.append(audit_case(case,teacher))
    reasons=Counter()
    for result in results:reasons.update(result['abstentions'])
    summary=dict(sealed_failures=len(results),tail_states=sum(r['tail_rows'] for r in results),
        qualified_states=sum(r['qualified_states'] for r in results),abstentions=dict(reasons),
        cases_with_earlier_modeled_route_loss=sum(r['earlier_route_loss_candidates']>0 for r in results),
        earlier_modeled_route_loss_states=sum(r['earlier_route_loss_candidates'] for r in results))
    args.output.mkdir(parents=True)
    write(args.output/'report.json',dict(schema='B1-sealed-partial-actual-map-route-audit-v1',utc=utc_now(),
        status='completed-read-only-partial-subset',source_progress=progress,
        source_episode_complete_prefix_sha256=hashlib.sha256(raw).hexdigest(),
        source_episode_complete_prefix_bytes=len(raw),partial_failures_are_selection_biased=True,
        not_an_aggregate_policy_score=True,original_counterfactuals_executed=False,new_native_worlds=0,
        optimizer_updates=0,future_emissions_predicted=False,modeled_paths_allow_one_turn_only=True,
        summary=summary,results=results))
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--per-policy',type=int,default=4);run(p.parse_args())
