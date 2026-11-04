"""Read the complete development failure tails without loading any policy.

Separate a teacher's abstention from missing native input. Predicted alternative
actions are not executed native counterfactuals, nor proof of a causal escape.
"""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import sys
import time

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(a):
    sys.path.insert(0,str(a.runtime/'src'))
    import numpy as np
    from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher,normal_unprotected
    if a.output.exists():raise FileExistsError(a.output)
    report=read(a.report)
    if (report.get('status')!='completed' or report.get('completed')!=40
            or report.get('native_seed_split')!='development'
            or read(a.report.parent/'cleanup.json')!=dict(closed=True,close_errors=[])):
        raise ValueError('Complete closed development40 required')
    episodes=sorted((r for r in report['episode_results'] if r['policy']=='trained_greedy'),key=lambda r:r['game'])
    if len(episodes)!=20 or [r['game'] for r in episodes]!=list(range(20)):
        raise ValueError('All twenty new-BC cases required')
    teacher=MovementSafetyTeacher();cases=[];started=time.monotonic()
    for episode in episodes:
        pointer=episode['failure_trace'];receipt=read(pointer['receipt']);path=Path(receipt['arrays_file'])
        if (sha(path)!=receipt['sha256'] or receipt['sha256']!=pointer['sha256']
                or receipt['checkpoint_sha256']!=report['pins']['trained']['sha256']
                or receipt.get('earlier_history_full_map_coverage_known') is not True):
            raise ValueError('Original FP32 tail provenance differs')
        with np.load(path,allow_pickle=False) as bundle:
            f=bundle['features'];o=bundle['objects'];maps=bundle['maps']
            frames=bundle['native_frame'];actions=bundle['action'];probabilities=bundle['probabilities']
        if f.shape!=(128,25) or o.shape!=(128,96,29) or maps.shape!=(128,16,92,96):
            raise ValueError('A complete original 128-observation tail required')
        if (not all(np.isfinite(x).all() for x in (f,o,maps,probabilities))
                or not np.allclose(probabilities.sum(-1),1,atol=2e-5,rtol=0)
                or not np.array_equal(actions,probabilities.argmax(-1))):
            raise ValueError('Finite modalities and actual greedy actions required')
        categories=Counter();witnesses=[];trusted=0;all_trapped=[]
        critical_rows=[];route_reasons=Counter();route_witnesses=[];route_queries=0
        for j in range(128):
            after=int(frames[j+1]) if j<127 else int(receipt['after_native_frame'])
            if after-int(frames[j])!=4:
                categories['step_span_not_four_native_frames']+=1;continue
            if not normal_unprotected(f[j]):
                categories['protected_or_non_normal']+=1;continue
            unknown=teacher._uncertainty(f[j],o[j],maps[j],horizon=4)
            if unknown:
                categories['abstain_'+unknown]+=1;continue
            trusted+=1
            origin,*_=teacher._scene(f[j],o[j])
            margins=teacher.clearances(f[j],o[j],np.repeat(origin[None],18,axis=0),teacher.velocities,duration=4)
            if np.isnan(margins).any():raise ValueError('NaN in modeled geometry')
            chosen=int(actions[j]);buffered=margins>1
            if buffered.any() and (margins<=0).any():
                critical_rows.append(dict(row=j,
                    uniform_predicted_collision_probability=float((margins<=0).mean()),
                    policy_predicted_collision_probability=float(probabilities[j][margins<=0].sum()),
                    executed_action_predicted_collision=bool(margins[chosen]<=0)))
            if not np.any(margins>0):
                category='no_modeled_immediate_escape';all_trapped.append(j)
            elif margins[chosen]<=0 and buffered.any():
                category='chosen_predicted_collision_with_buffered_alternative'
            elif margins[chosen]<=1 and buffered.any():
                category='chosen_below_buffer_with_buffered_alternative'
            else:category='chosen_current_motion_route_has_clearance'
            categories[category]+=1
            if category=='chosen_predicted_collision_with_buffered_alternative':
                witnesses.append(dict(row=j,before_native_frame=int(frames[j]),
                    native_frames_before_collision=int(receipt['after_native_frame'])-int(frames[j]),
                    action=chosen,chosen_margin=float(margins[chosen]),
                    best_modeled_margin=float(np.clip(margins.max(),-64,64)),
                    buffered_actions=int(buffered.sum()),
                    probability_on_predicted_colliding_actions=float(probabilities[j][margins<=0].sum())))
            # Fixed grid, chosen before observing route outcomes; never select
            # only frames where the new diagnostic supports the hypothesis.
            if j%8==0 or j==127:
                route_queries+=1
                safe,detail=teacher.label(f[j],o[j],maps[j],force=True)
                route_reasons[detail['reason']]+=1
                if safe.any() and not bool(safe[chosen]):
                    route_witnesses.append(dict(row=j,
                        native_frames_before_collision=int(receipt['after_native_frame'])-int(frames[j]),
                        action=chosen,chosen_four_frame_margin=float(margins[chosen]),
                        chosen_modeled_two_phase_margin=float(detail['route_clearance'][chosen]),
                        alternative_first_actions_with_modeled_buffer=int(safe.sum()),
                        captured_probability_on_two_phase_feasible_first_actions=float(probabilities[j][safe].sum())))
        last=o[-1];present=last[:,0]>.5;relevant=present&((last[:,1]>.5)|(last[:,2]>.5))
        distance=np.linalg.norm(last[:,6:8]*[384,448],axis=-1)
        near=distance<=(4.5*4+np.linalg.norm(last[:,8:10]*12,axis=-1)*4+np.linalg.norm(last[:,10:12]*128,axis=-1)+16)
        growth=relevant&near&(last[:,20]>.5)
        growing=[]
        for token in last[growth]:
            growing.append(dict(distance_from_player=float(np.linalg.norm(token[6:8]*[384,448])),
                collidable_now=bool(token[13]>.5),spawn_delay_frames_encoded=float(token[17]*8),
                age_frames_encoded=float(token[18]*120),stored_velocity=(token[8:10]*12).tolist(),
                half_size=(token[10:12]*128).tolist()))
        cases.append(dict(game=episode['game'],native_seed=episode['native_seed'],stage=episode['stage'],
            first_collision_frames=episode['native_frames'],verified_history_rows=128,
            original_trace_sha256=receipt['sha256'],four_frame_categories=dict(categories),
            trusted_rows=trusted,earlier_modeled_collision_choices=len(witnesses),
            earlier_modeled_collision_choice_witnesses=witnesses,
            critical_four_frame_rows=critical_rows,
            two_phase_fixed_grid_queries=route_queries,two_phase_fixed_grid_reasons=dict(route_reasons),
            two_phase_modeled_alternative_witnesses=route_witnesses,
            final_uncertainty=teacher._uncertainty(f[-1],last,maps[-1],horizon=4),
            final_selected_tokens=int(present.sum()),
            final_growth_phase_objects_near_four_frame_envelope=growing,
            first_modeled_trap_row=min(all_trapped) if all_trapped else None))
        del f,o,maps,frames,actions,probabilities
    total=Counter()
    for c in cases:total.update(c['four_frame_categories'])
    critical=[r for c in cases for r in c['critical_four_frame_rows']]
    route_total=Counter()
    for c in cases:route_total.update(c['two_phase_fixed_grid_reasons'])
    result=dict(schema='B1-BC-twenty-development-failure-tail-temporal-coverage-v2',
        status='completed',utc=utc_now(),source_report=str(a.report),source_report_sha256=sha(a.report),
        actual_neural_checkpoint_sha256=report['pins']['trained']['sha256'],
        all_twenty_development_cases=True,actual_observations=2560,
        analyzed_native_history_frames_per_case=512,cases=cases,
        four_frame_categories=dict(total),
        cases_with_earlier_modeled_colliding_choice=sum(c['earlier_modeled_collision_choices']>0 for c in cases),
        critical_four_frame_states_with_colliding_and_buffered_alternatives=len(critical),
        mean_uniform_predicted_collision_probability_on_critical_states=float(np.mean([r['uniform_predicted_collision_probability'] for r in critical])) if critical else None,
        mean_policy_predicted_collision_probability_on_critical_states=float(np.mean([r['policy_predicted_collision_probability'] for r in critical])) if critical else None,
        executed_predicted_collision_choices_on_critical_states=sum(r['executed_action_predicted_collision'] for r in critical),
        two_phase_fixed_grid_every_eight_retained_observations_plus_last=True,
        two_phase_queries_on_trusted_four_frame_states=sum(c['two_phase_fixed_grid_queries'] for c in cases),
        two_phase_reasons=dict(route_total),
        cases_with_two_phase_modeled_alternatives_not_selected=sum(bool(c['two_phase_modeled_alternative_witnesses']) for c in cases),
        two_phase_predicted_first_action_errors=sum(len(c['two_phase_modeled_alternative_witnesses']) for c in cases),
        growth_phase_semantics='Native bullet state2 spawn/growth animation;not proof of missing current collision size or size growth velocity',
        measured_dimensions_are_present_in_original_tokens=True,
        alternative_routes_are_current_constant_motion_predictions=True,
        native_counterfactual_alternatives_executed=False,
        causal_failure_source_proven=False,current_live_PPO_modified=False,
        optimizer_updates=0,models_loaded=0,CUDA_used=False,native_worlds_started=0,
        wall_seconds=time.monotonic()-started)
    a.output.mkdir(parents=True);write(a.output/'report.json',result)
    (a.output/'diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('report','runtime','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
