"""Independent, current-state-free baselines for the recorded mixed loss.

The optimal constant for a mixture of soft targets and argmax labels is the
same mixture of their training-set means. Beating only a soft-target constant
does not demonstrate that a mixed-loss student uses its observations.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fly_touhou.experiment import ExperimentLogger
from fly_touhou.whole_brain_learning import (
    DIFFICULTY_NAMES, N_ACTIONS, _Metrics, load_campaign_demonstrations,
    split_campaign_episodes,
)


def constant_metrics(probabilities, episodes, hard_weight):
    metrics = _Metrics(hard_weight)
    logits = torch.tensor(np.log(np.maximum(probabilities, 1e-30)), dtype=torch.float32)[None]
    for episode in episodes:
        metrics.add(logits.expand(len(episode), -1),
                    torch.from_numpy(episode.probabilities), torch.from_numpy(episode.actions))
    return metrics.summary()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hard-weight', type=float, default=.5)
    options = parser.parse_args()
    if not 0 <= options.hard_weight <= 1:
        parser.error('hard-weight must be in [0,1]')
    config = {'purpose':'state-independent control for actual mixed distillation loss',
              'seed':options.seed,'validation_fraction':.25,'hard_weight':options.hard_weight,
              'brain_model_evaluated':False,'final_game_acceptance_achieved':False}
    files = {'audit_script':Path(__file__),
             'learner_code':ROOT/'src/fly_touhou/whole_brain_learning.py'}
    files.update({f'trace_{i}':p for i,p in enumerate(options.traces)})
    logger=ExperimentLogger(options.output,config,files=files)
    with logger.phase('constant-policy-baselines'):
        dataset=load_campaign_demonstrations(options.traces)
        train,validation=split_campaign_episodes(dataset.episodes,seed=options.seed)
        logger.record_split([e.episode_id for e in train],[e.episode_id for e in validation])
        labels=np.concatenate([e.probabilities for e in train])
        soft=labels.mean(axis=0,dtype=np.float64)
        soft/=soft.sum()
        hard=np.bincount(labels.argmax(-1),minlength=N_ACTIONS)/len(labels)
        mixed=(1-options.hard_weight)*soft+options.hard_weight*hard
        baselines={}
        for name,probabilities in [('soft-only-mean',soft),('mixed-loss-optimal-constant',mixed)]:
            baselines[name]={'probabilities':probabilities.tolist(),
                'training':constant_metrics(probabilities,train,options.hard_weight),
                'validation':constant_metrics(probabilities,validation,options.hard_weight),
                'validation_by_difficulty':{DIFFICULTY_NAMES[d]:constant_metrics(probabilities,
                    [e for e in validation if e.native_context['difficulty']==d],options.hard_weight)
                    for d in sorted({e.native_context['difficulty'] for e in validation})}}
        result={'config':config,'sources':dataset.sources,'baselines':baselines,
                'scope':'same split as wholebrain-v1; constants estimated from training episodes only'}
        target=options.output/'constant_baselines.json'
        with target.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        logger.event('constant_baseline_result',result)
        print(json.dumps({name:value['validation_by_difficulty'] for name,value in baselines.items()}))


if __name__=='__main__':
    main()
