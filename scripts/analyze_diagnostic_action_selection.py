"""Training-only pointwise comparison of sampled and maximum-probability actions.

The maximum-probability action is evaluated with recorded geometry estimates at
the SAMPLED policy's visited states. This cannot replay a greedy controller's own
future observations/history and is not evidence of greedy native survival.
"""
from pathlib import Path
import argparse
import gzip
import json
import math
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.experiment import file_sha256
from fly_touhou.native_diagnostic import write_json


def seal(path):return {'path':str(path),'sha256':file_sha256(path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior diagnoses')
    torch.set_num_threads(2)
    cases=[];per_seed={}
    for seed in (11,29,47):
        run=args.root/f'seed{seed}/train'
        terminal=json.loads((run/'report.json').read_text(encoding='utf-8'))['counters']['updates']
        selected=sorted({64,80,terminal})
        combined=[]
        for update in selected:
            path=run/'training'/f'behavior_{update:04d}.pt.gz'
            with gzip.open(path,'rb') as stream:data=torch.load(stream,map_location='cpu',weights_only=True)
            trace=Path(data['native_trajectory']['path'])
            assert file_sha256(trace)==data['native_trajectory']['sha256']
            with gzip.open(trace,'rt',encoding='utf-8') as stream:rows=[json.loads(line) for line in stream]
            probabilities=data['old_logp18'].exp()
            assert probabilities.shape==data['forecast_costs'].shape==(len(rows),18)
            assert torch.allclose(probabilities.sum(-1),torch.ones(len(rows),dtype=torch.float64),rtol=0,atol=1e-12)
            active=data['danger_weights']>0
            greedy=probabilities.argmax(-1)
            costs=data['forecast_costs']
            expected=(probabilities*costs).sum(-1)
            maximum_cost=costs.gather(1,greedy[:,None]).squeeze(1)
            sampled_cost=costs.gather(1,data['actions'][:,None]).squeeze(1)
            unsafe_mass=(probabilities*(costs>.1)).sum(-1)
            entropy=-(probabilities*data['old_logp18']).sum(-1)
            pointwise=[];hit_records=[]
            for i,row in enumerate(rows):
                assert row['role']=='training' and row['action']==int(data['actions'][i])
                assert row['conditional_logp18']==data['old_logp18'][i].tolist()
                if active[i]:
                    pointwise.append([float(expected[i]),float(maximum_cost[i]),float(sampled_cost[i]),
                                      float(unsafe_mass[i]),float(probabilities[i].max()),float(entropy[i])])
                if row['hit']:
                    clearances=(row.get('movement_details') or {}).get('clearances')
                    values=None if clearances is None else [math.inf if c is None else float(c) for c in clearances]
                    arg=int(greedy[i]);sample=int(data['actions'][i])
                    hit_records.append({'episode_id':row['episode_id'],'step':row['episode_step'],
                        'actual_sampled_action':sample,'pointwise_greedy_action':arg,
                        'chosen_probability':float(probabilities[i,sample]),'maximum_probability':float(probabilities[i,arg]),
                        'sampled_forecast_clearance':None if values is None or values[sample]==math.inf else values[sample],
                        'greedy_forecast_clearance':None if values is None or values[arg]==math.inf else values[arg],
                        'greedy_forecast_positive':values is not None and values[arg]>0,
                        'greedy_forecast_at_least5':values is not None and values[arg]>=5,
                        'actions_differ':arg!=sample,'not_a_real_greedy_game_outcome':True})
            if not pointwise:raise ValueError('This probe needs discriminative training states')
            tensor=torch.tensor(pointwise,dtype=torch.float64)
            names=('expected_sampled_forecast_regret','pointwise_greedy_forecast_regret',
                   'realized_sampled_forecast_regret','unsafe_probability_mass',
                   'maximum_action_probability','categorical_entropy')
            averages=dict(zip(names,tensor.mean(0).tolist()))
            case={'seed':seed,'update':update,'samples':len(rows),'discriminative_training_states':len(pointwise),
                'behavior':seal(path),'trajectory':seal(trace),'means_on_discriminative_states':averages,
                'actual_hit_records':hit_records}
            cases.append(case);combined.extend(pointwise)
        tensor=torch.tensor(combined,dtype=torch.float64)
        per_seed[str(seed)]={'states':len(combined),**dict(zip(names,tensor.mean(0).tolist()))}
    result={'schema':'training-only-sampled-versus-greedy-forecast-probe-v1','cases':cases,'per_seed':per_seed,
        'script':seal(Path(__file__)),'heldout_data_read':False,'new_native_actions':0,'live_training_modified':False,
        'greedy_forecast_better_on_all3_seeds':all(v['pointwise_greedy_forecast_regret']<v['expected_sampled_forecast_regret'] for v in per_seed.values()),
        'scope':'pointwise recorded geometric estimates only;not original-game counterfactuals or greedy deployment results',
        'current_formal_test_selection_rule_unchanged':'categorical sampling temperature1',
        'G2_achieved':False,'final_acceptance':False}
    write_json(args.output,result,exclusive=True)
    print(json.dumps({'per_seed':per_seed,'greedy_forecast_better_on_all3_seeds':result['greedy_forecast_better_on_all3_seeds'],
        'actual_hit_states':sum(len(c['actual_hit_records']) for c in cases),'output':str(args.output)},ensure_ascii=False))


if __name__=='__main__':main()
