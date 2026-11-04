"""Verify a completed cached Bomb fit before independent native deployment."""
import argparse
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.connectome import file_sha256
from fly_touhou.experiment import verify_events
from fly_touhou.whole_brain_readout import _tensor_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Preserve previous audits')
    run = args.run.resolve()
    manifest_path, report_path = run / 'manifest.json', run / 'report.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    report = json.loads(report_path.read_text(encoding='utf-8'))
    parameterization = report.get('readout_parameterization', 'shared-correction')
    factorized = parameterization == 'factorized-movement-bomb'
    expected_status = ('completed_factorized_Bomb_readout_diagnostic' if factorized
                       else 'completed_shared_Bomb_readout_diagnostic')
    if report['status'] != expected_status:
        raise ValueError('Require a completed diagnostic')
    events = verify_events(run / 'events.jsonl', run_id=manifest['run_id'])
    epoch = report['best_epoch']
    selected = [r for r in report['history'] if r['epoch'] == epoch]
    if len(selected) != 1:
        raise ValueError('Ambiguous selected epoch')
    path = Path(selected[0]['checkpoint']).resolve()
    sealed = [e['payload'] for e in events if e['event'] == 'cached_readout_checkpoint'
              and Path(e['payload']['path']).resolve() == path]
    if len(sealed) != 1 or sealed[0]['sha256'] != file_sha256(path):
        raise ValueError('Selected checkpoint differs from its sealed event')
    parent_path = Path(manifest['config']['initial_checkpoint_path']).resolve()
    if file_sha256(parent_path) != manifest['config']['initial_checkpoint_sha256']:
        raise ValueError('Readout parent changed')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
    before, after = parent['trainable_parameters'], checkpoint['trainable_parameters']
    assert set(before) == set(after) == {'input_encoder.weight', 'input_encoder.bias', 'action_readout.weight'}
    assert checkpoint['model_config'] == parent['model_config']
    assert checkpoint['graph_fingerprint'] == parent['graph_fingerprint']
    assert checkpoint['encoder'] == parent['encoder']
    assert all(torch.equal(before[k], after[k]) for k in ('input_encoder.weight', 'input_encoder.bias'))
    assert torch.equal(checkpoint['shared_bomb_parent_readout'], before['action_readout.weight'])
    if factorized:
        movement, correction = checkpoint['factorized_movement'], checkpoint['factorized_bomb']
        assert movement.shape == (18, 1299) and movement.dtype == torch.float32
        expected = torch.cat((movement, movement + correction), 0)
        optimizer_names, optimizer_shapes = ['factorized_movement', 'factorized_bomb'], [(18, 1299), (1, 1299)]
    else:
        correction = checkpoint['shared_bomb_correction']
        expected = torch.cat((before['action_readout.weight'][:18],
                              before['action_readout.weight'][18:] + correction), 0)
        optimizer_names, optimizer_shapes = ['shared_bomb_correction'], [(1, 1299)]
    assert correction.shape == (1, 1299) and correction.dtype == torch.float32
    assert torch.equal(after['action_readout.weight'], expected)
    assert checkpoint['feature_cache_sha256'] == report['cache_proof']['cache_sha256']
    state = checkpoint['optimizer_state']
    assert len(state['param_groups']) == 1 and len(state['param_groups'][0]['params']) == len(optimizer_names)
    expected_steps = epoch * math.ceil(report['cache_proof']['training_decisions'] /
                                      manifest['config']['batch_size'])
    if expected_steps:
        assert len(state['state']) == len(optimizer_names)
        for index, shape in zip(state['param_groups'][0]['params'], optimizer_shapes):
            item = state['state'][index]
            assert int(item['step'].item()) == expected_steps
            assert tuple(item['exp_avg'].shape) == tuple(item['exp_avg_sq'].shape) == shape
    assert checkpoint['optimizer_parameter_names'] == optimizer_names
    assert 'rl_state' not in checkpoint
    initial_path = Path(report['history'][0]['checkpoint']).resolve()
    initial_seals = [e['payload'] for e in events if e['event'] == 'cached_readout_checkpoint'
                     and Path(e['payload']['path']).resolve() == initial_path]
    assert len(initial_seals) == 1 and initial_seals[0]['sha256'] == file_sha256(initial_path)
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    parameter_keys = ['factorized_movement', 'factorized_bomb'] if factorized else ['shared_bomb_correction']
    changes = {name: {'coordinates': checkpoint[name].numel(),
                     'changed_from_actual_epoch_zero': int(torch.count_nonzero(checkpoint[name] != initial[name])),
                     'change_l2': float(torch.linalg.vector_norm((checkpoint[name] - initial[name]).double()))}
               for name in parameter_keys}
    result = {'schema': 'cached-Bomb-readout-parameter-audit-v1',
        'checkpoint': str(path), 'checkpoint_sha256': sealed[0]['sha256'],
        'parent': str(parent_path), 'parent_sha256': file_sha256(parent_path),
        'report_sha256': file_sha256(report_path), 'epoch': epoch,
        'ordinary_36x1299_fold_exact': True, 'input_parameters_unchanged': True,
        'movement_rows_unchanged': torch.equal(before['action_readout.weight'][:18], after['action_readout.weight'][:18]),
        'graph_and_dynamics_configuration_unchanged': True,
        'readout_parameterization': parameterization, 'optimizer_parameter_names': optimizer_names,
        'actual_epoch_zero_checkpoint': str(initial_path), 'actual_epoch_zero_sha256': initial_seals[0]['sha256'],
        'initial_policy_differs_from_parent': not torch.equal(initial['trainable_parameters']['action_readout.weight'],
                                                            before['action_readout.weight']),
        'learned_coordinate_changes': changes,
        'Bomb_vector_sha256': _tensor_digest(correction),
        'Bomb_vector_nonzero_parameters': int(torch.count_nonzero(correction)),
        'optimizer_steps_per_parameter': expected_steps,
        'no_PPO_learning_state_discarded': True,
        'validation_Bomb': selected[0]['validation']['by_teacher_action_type']['bomb'],
        'validation_movement': selected[0]['validation']['by_teacher_action_type']['movement'],
        'selection_score': selected[0]['selection_score'],
        'evidence_scope': 'CPU parameter, optimizer and sealed cache provenance audit; not native game acceptance'}
    if not factorized:
        result.update(shared_correction_sha256=result['Bomb_vector_sha256'],
                      shared_correction_nonzero_parameters=result['Bomb_vector_nonzero_parameters'],
                      shared_correction_adam_steps=expected_steps)
    if file_sha256(path) != sealed[0]['sha256']:
        raise RuntimeError('Checkpoint changed during the audit')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
