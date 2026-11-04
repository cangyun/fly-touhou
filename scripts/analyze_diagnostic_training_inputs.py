"""Read-only input/target diagnostics; never a game counterfactual or test."""
from pathlib import Path
import argparse
import gzip
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fly_touhou.experiment import file_sha256
from fly_touhou.native_diagnostic import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--through',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Keep prior analyses')
    torch.set_num_threads(2)
    manifest = json.loads((args.run/'manifest.json').read_text(encoding='utf-8'))
    protocol = json.loads(Path(manifest['files']['protocol']['path']).read_text(encoding='utf-8'))
    limit = protocol['encoder']['observation']['nearest_bullets']
    summaries, all_observations = [], []
    retained, swapped, truncated, total_states, max_bullets = 0, 0, 0, 0, 0
    previous_identity, previous_slots = None, {}
    bindings = []
    for update in range(1,args.through+1):
        path = args.run/'training'/f'behavior_{update:04d}.pt.gz'
        bindings.append({'path':str(path),'sha256':file_sha256(path)})
        with gzip.open(path,'rb') as stream: behavior = torch.load(stream,map_location='cpu',weights_only=True)
        p = behavior['old_logp18'].exp()
        costs, weights = behavior['forecast_costs'],behavior['danger_weights']
        expected = float(((p*costs).sum(-1)*weights).mean())
        uniform = float((costs.mean(-1)*weights).mean())
        constant_costs = (costs*weights[:,None]).mean(0)
        best_constant = float(constant_costs.min())
        marginal = p.mean(0)
        conditional_variation = float((p*(behavior['old_logp18']-marginal.log()[None])).sum(-1).mean())
        summaries.append({'update':update,'samples':len(p),'danger_samples':int(behavior['discriminative'].sum()),
            'expected_forecast_regret':expected,'uniform_forecast_regret_on_same_visited_states':uniform,
            'best_constant_forecast_regret_on_same_visited_states':best_constant,
            'best_constant_action':int(constant_costs.argmin()),
            'improvement_vs_uniform_on_these_records':uniform-expected,
            'improvement_vs_best_constant_on_these_records':best_constant-expected,
            'conditional_action_variation_meanKL_to_marginal':conditional_variation,
            'maximum_policy_probability_mean':float(p.max(-1).values.mean()),
            'mean_action_probabilities':marginal.tolist(),
            'training_states_only_no_holdout_or_native_alternative_actions':True})
        all_observations.append(behavior['observations'])
        trace = Path(behavior['native_trajectory']['path'])
        assert file_sha256(trace) == behavior['native_trajectory']['sha256']
        with gzip.open(trace,'rt',encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line); snapshot=row['snapshot'];player=snapshot['player']
                bullets=sorted((b for b in snapshot['bullets'] if b['active'] and not b['intangible']),
                    key=lambda b:((b['x']-player['x'])**2+(b['y']-player['y'])**2,b['id']))
                slots={b['id']:i for i,b in enumerate(bullets[:limit])}
                if row['episode_id'] == previous_identity:
                    common=slots.keys()&previous_slots.keys()
                    retained+=len(common)
                    swapped+=sum(slots[k]!=previous_slots[k] for k in common)
                previous_identity,previous_slots=row['episode_id'],slots
                total_states+=1;truncated+=int(len(bullets)>limit);max_bullets=max(max_bullets,len(bullets))
    observations=torch.cat(all_observations).double()
    std=observations.std(0,unbiased=False)
    result={'schema':'read-only-native-training-input-and-direction-diagnostics-v1','status':'finished',
        'run':str(args.run),'through':args.through,'source_behavior':bindings,
        'source_script_sha256':file_sha256(Path(__file__)),'updates':summaries,
        'input_summary':{'states':total_states,'dimensions':observations.shape[1],
            'constant_dimensions_on_this_sample':int((std==0).sum()),
            'near_constant_dimensions_std_below_1e-5':int((std<1e-5).sum()),
            'active_bullet_slot_limit':limit,'states_exceeding_active_bullet_slot_limit':truncated,
            'maximum_active_bullets':max_bullets,'retained_object_slot_comparisons':retained,
            'retained_objects_with_changed_slot':swapped,
            'retained_object_slot_change_fraction':swapped/retained if retained else None},
        'interpretation_limit':'conditionalvariation can reflect observation and recurrent/time history;no causal sensory ablation. constant/uniform forecasts use same visited records,not actual alternative game trajectories. slot swapping/omissions are hypotheses,not established causes.',
        'models_changed':False,'games_started':False,'heldout_data_used':False,
        'G2_or_G3_achieved':False,'final_acceptance':False}
    write_json(args.output,result,exclusive=True)
    print(json.dumps({'output':str(args.output),'input_summary':result['input_summary'],
        'first_update':summaries[0],'last_update':summaries[-1]},ensure_ascii=False))


if __name__ == '__main__':
    main()
