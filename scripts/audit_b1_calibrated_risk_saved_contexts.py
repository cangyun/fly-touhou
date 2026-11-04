"""Post-selection CPU diagnosis; saved test failures never train this candidate.

The candidate and live development plan were already frozen before this read.
Only its point network differs, so the original recorded base-GRU context is
valid for this conditional comparison. Alternatives are not native outcomes.
"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
FROZEN=Path('./outputs/source_snapshots/20261025-B1-full-geometry-paired-development-v2')
sys.path.insert(0,str(FROZEN/'src'));sys.path.insert(0,str(FROZEN/'scripts'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.thrl_durable_records import write_json as write
from fly_touhou.thrl_full_geometry_data import digest


def main():
    import numpy as np
    import torch
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_pair_risk import action_clearances
    from fly_touhou.thrl_policy_actions import policy_logits
    from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher
    from evaluate_b1_full_geometry_pair import validate_candidate
    torch.set_num_threads(2)
    r=ROOT/'artifacts/engineering/B1-autonomous-dodge-20261024'
    plan=read(r/'full-geometry-paired-development-v2-control/plan.json')
    payload={}
    for key,item in plan['policies'].items():
        if digest(item['checkpoint'])!=item['sha256']:raise ValueError('Already selected comparison checkpoint changed')
        payload[key]=torch.load(item['checkpoint'],map_location='cpu',weights_only=False)
    validate_candidate(torch,payload['baseline'],payload['candidate'])
    models={}
    for key in payload:
        models[key]=model_from_config(**payload[key]['model_config']);models[key].load_state_dict(payload[key]['model'],strict=True);models[key].eval()
    review=read(r/'stable-clear-wrapup-review-v1.json');teacher=MovementSafetyTeacher();rows=[]
    with torch.inference_mode():
        for case in review['cases']:
            probe=case['terminal_probe'];selected=[]
            if probe['probes'] and probe['probes'][-1]['modeled_directional_mistake_candidate']:
                selected.append((probe['probes'][-1]['row'],'original_executed_native_collision'))
            selected.extend((x['tail_row'],'measured_usable_deathbomb') for x in case['terminal_tail_missed_usable_Bomb_decisions'])
            if not selected:continue
            if digest(probe['arrays'])!=probe['arrays_sha256']:raise ValueError('Saved original observation changed')
            with np.load(probe['arrays'],allow_pickle=False) as z:
                for index,kind in selected:
                    f=torch.as_tensor(z['features'][index:index+1]);maps=torch.as_tensor(z['maps'][index:index+1])
                    o=torch.as_tensor(z['objects'][index:index+1]);hidden=torch.as_tensor(z['hidden'][index]).reshape(1,1,-1)
                    outputs={};following={}
                    for key,model in models.items():
                        logits,_,following[key]=model.forward_step(f,maps,hidden,objects=o)
                        probabilities=torch.softmax(policy_logits(logits),-1)[0].numpy()
                        outputs[key]=dict(action=int(probabilities.argmax()),Bomb_probability=float(probabilities[18]),
                            point_clearances=action_clearances(model.risk,f,o[...,:21])[0].numpy())
                    if not torch.equal(following['baseline'],following['candidate']):
                        raise ValueError('A supposedly fixed base-GRU produced a different next context')
                    action=int(z['action'][index]);new=outputs['candidate']['action']
                    origin,*_=teacher._scene(f[0].numpy(),o[0].numpy())
                    margins=teacher.clearances(f[0].numpy(),o[0].numpy(),np.repeat(origin[None],18,axis=0),teacher.velocities,duration=4)
                    row=dict(game=case['game'],kind=kind,row=index,original_action=action,candidate_action=new,
                        base_GRU_next_state_identical=True,
                        original_Bomb_probability=float(z['probabilities'][index,18]),
                        candidate_Bomb_probability=outputs['candidate']['Bomb_probability'],
                        candidate_original_action_point_margin=float(outputs['candidate']['point_clearances'][action]) if action<18 else None,
                        baseline_original_action_point_margin=float(outputs['baseline']['point_clearances'][action]) if action<18 else None,
                        candidate_chosen_model_geometry_margin=float(margins[new]) if new<18 and np.isfinite(margins[new]) else None,
                        candidate_action_executed_in_original=False)
                    rows.append(row)
    output=r/'full-geometry-post-selection-saved-context-audit-v1.json'
    write(output,dict(schema='B1-full-geometry-post-selection-saved-context-audit-v1',utc=utc_now(),
        candidate_sha256=plan['policies']['candidate']['sha256'],baseline_sha256=plan['policies']['baseline']['sha256'],
        used_for_training_or_checkpoint_selection=False,CUDA_initialized=False,native_worlds_started=0,
        optimizer_updates=0,development_comparison_already_launched=True,rows=rows,
        conditional_old_observation_history_only=True,native_alternative_safety_not_proven=True))
    for row in rows:print(row)


if __name__=='__main__':main()
