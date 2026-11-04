"""CPU preflight of the prepared long complete-brain interface curriculum.

This reads all actual raw demonstrations, checks the reserved episode split,
and derives runtime arguments from the exact full-interface parent. It starts
no game, optimizer step or CUDA model.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.cached_bomb_readout import load_verified_producer
from fly_touhou.connectome import file_sha256
from fly_touhou.whole_brain_learning import load_campaign_demonstrations, split_campaign_episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer', type=Path, required=True)
    parser.add_argument('--cache-sha256', required=True)
    parser.add_argument('--job-spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():
        raise FileExistsError('Preserve previous admissions')
    parent, cached_train, cached_val, history, files, proof = load_verified_producer(
        args.producer, args.cache_sha256)
    spec = json.loads(args.job_spec.read_text(encoding='utf-8-sig'))
    control = Path(spec['control_directory'])
    if (control / 'launch.json').exists() or (control / 'worker.lock').exists():
        raise ValueError('Never alter a launched immutable job specification')
    argv = spec['arguments']
    assert argv[argv.index('--initial-checkpoint')+1] == proof['parent_path']
    start = argv.index('--traces') + 1
    end = next((i for i in range(start, len(argv)) if argv[i].startswith('--')), len(argv))
    dataset = load_campaign_demonstrations([Path(p) for p in argv[start:end]])
    train, validation = split_campaign_episodes(dataset.episodes, .25, 42,
        fixed_train_ids=parent['split_history']['training'], fixed_validation_ids=parent['split_history']['validation'])
    assert dataset.encoder.specification() == parent['encoder']
    assert [e.episode_id for e in train] == [e['episode_id'] for e in cached_train]
    assert [e.episode_id for e in validation] == [e['episode_id'] for e in cached_val]
    assert parent['model_config'].get('recurrent_gain_limit', 0.) == 0.
    assert 'rl_state' not in parent
    optimizer = parent['optimizer_state']
    assert len(optimizer['param_groups']) == 1 and len(optimizer['param_groups'][0]['params']) == 3
    assert len(optimizer['state']) == 3
    names = ('input_encoder.weight', 'input_encoder.bias', 'action_readout.weight')
    steps = {}
    for name, index in zip(names, optimizer['param_groups'][0]['params']):
        value = optimizer['state'][index]
        shape = parent['trainable_parameters'][name].shape
        assert value['exp_avg'].shape == value['exp_avg_sq'].shape == shape
        steps[name] = int(value['step'].item())
    for flag, value in {
        '--encoder-init-scale': parent['model_config']['encoder_init_scale'],
        '--substeps': parent['model_config']['substeps'], '--dt-ms': parent['model_config']['dt_ms'],
        '--warmup-ms': parent['training_config']['warmup_ms'],
        '--propagation-backend': parent['model_config']['propagation_backend'],
    }.items():
        argv[argv.index(flag)+1] = str(value)
    assert argv[argv.index('--device')+1] == 'cuda'
    assert '--capture-backward' in argv and '--capture-inference' in argv
    assert argv[argv.index('--selection-metric')+1] == 'balanced_action_type_ce'
    before_spec_sha = file_sha256(args.job_spec)
    args.job_spec.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding='utf-8')
    result = {'schema': 'long-complete-brain-interface-admission-v1', 'status': 'passed',
        'parent_sha256': proof['parent_sha256'], 'cache_proof': proof,
        'actual_raw_sources': dataset.sources, 'training_decisions': sum(map(len, train)),
        'validation_decisions': sum(map(len, validation)),
        'training_episode_ids': [e.episode_id for e in train],
        'validation_episode_ids': [e.episode_id for e in validation],
        'parent_optimizer_steps': steps, 'actual_parent_runtime_arguments_inherited': True,
        'job_spec_before_sha256': before_spec_sha, 'job_spec_after_sha256': file_sha256(args.job_spec),
        'job_launched': False, 'CUDA_model_allocated': False,
        'full_GPU_loss_backward_proof_still_required': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({'status': 'passed', 'train': result['training_decisions'],
        'validation': result['validation_decisions'], 'parent_optimizer_steps': steps,
        'output': str(args.output.resolve())}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
