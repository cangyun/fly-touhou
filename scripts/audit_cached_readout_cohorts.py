"""Separate existing student-error and long-teacher validation in a sealed cache.

Read-only CPU statistics of actual previously replayed complete-brain features.
No game, neural model or learning is started. This is not native acceptance.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.cached_bomb_readout import load_verified_producer
from fly_touhou.connectome import file_sha256
from fly_touhou.whole_brain import ShiuParameters
from fly_touhou.whole_brain_readout import _load_readout_parent, _tensor_digest, evaluate_cached_head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer', type=Path, required=True)
    parser.add_argument('--cache-sha256', required=True)
    parser.add_argument('--checkpoint', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Preserve prior audits')
    parent, train, validation, history, files, proof = load_verified_producer(
        args.producer, args.cache_sha256)
    parameters = ShiuParameters(**parent['model_config']['parameters'])
    assert asdict(parameters) == parent['model_config']['parameters']
    expected_inputs = {k: _tensor_digest(v) for k, v in parent['trainable_parameters'].items()
                       if k.startswith('input_encoder.')}
    cohorts = {
        'student_error_validation': [e for e in validation if e['episode_id'].startswith('DAgger:')],
        'long_teacher_validation': [e for e in validation if not e['episode_id'].startswith('DAgger:')],
    }
    if sorted(len(group) for group in cohorts.values()) != [1, 5]:
        raise ValueError('Expected five student-error episodes and one long teacher validation episode')
    results = []
    for path in [Path(proof['parent_path']), *args.checkpoint]:
        candidate, digest = _load_readout_parent(path,
            expected_graph_fingerprint=parent['graph_fingerprint'], parameters=parameters)
        actual_inputs = {k: _tensor_digest(v) for k, v in candidate['trainable_parameters'].items()
                         if k.startswith('input_encoder.')}
        if (candidate['model_config'] != parent['model_config'] or candidate['encoder'] != parent['encoder']
                or actual_inputs != expected_inputs):
            raise ValueError('Candidate head differs in neural dynamics or frozen sensory inputs')
        if digest != proof['parent_sha256']:
            if (candidate.get('feature_cache_sha256') != args.cache_sha256
                    or candidate['train_episode_ids'] != [e['episode_id'] for e in train]
                    or candidate['validation_episode_ids'] != [e['episode_id'] for e in validation]
                    or candidate['split_history'] != history):
                raise ValueError('Candidate cache or complete-episode holdout identities differ')
        head = torch.nn.Linear(1299, 36, bias=False)
        head.weight.requires_grad_(False)
        head.weight.copy_(candidate['trainable_parameters']['action_readout.weight'])
        groups = {}
        for name, episodes in cohorts.items():
            metrics = evaluate_cached_head(head, episodes, report_action_types=True)
            labels = torch.cat([e['probabilities'].argmax(-1) for e in episodes])
            groups[name] = {
                'episode_ids': [e['episode_id'] for e in episodes],
                'teacher_argmax_histogram': torch.bincount(labels, minlength=36).tolist(),
                'metrics': metrics,
            }
        if file_sha256(path) != digest:
            raise RuntimeError('Candidate changed during read-only audit')
        results.append({'checkpoint': str(path.resolve()), 'sha256': digest,
                        'parameterization': candidate.get('readout_parameterization', 'ordinary-36-action'),
                        'cohorts': groups})
        print(json.dumps({'checkpoint': str(path.resolve()), 'cohorts': {
            k: {'decisions': v['metrics']['decisions'],
                'mixed_ce': v['metrics']['mixed_cross_entropy'],
                'movement': v['metrics']['by_teacher_action_type']['movement'],
                'bomb': v['metrics']['by_teacher_action_type']['bomb']} for k, v in groups.items()}}, allow_nan=False), flush=True)
    result = {'schema': 'sealed-descending-readout-cohort-audit-v1', 'cache_proof': proof,
              'results': results, 'new_neural_replay': False, 'game_started': False, 'model_trained': False,
              'evidence_scope': 'offline cached full-brain validation by source cohort; no native game acceptance'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
