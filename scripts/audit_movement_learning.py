"""Verify actual movement labels, complete batch coverage and brain learning."""
from pathlib import Path
import argparse
from collections import Counter
import gzip
import io
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import torch

from fly_touhou.experiment import file_sha256, verify_events
from fly_touhou.movement_curriculum import MovementFeedback, balanced_weights
from fly_touhou.state import Snapshot
from fly_touhou.whole_brain_learning import _encoder


def equal(a, b):
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)): return type(a) is type(b) and len(a) == len(b) and all(equal(v, w) for v, w in zip(a, b))
    return type(a) is type(b) and a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--rollouts', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve the original learning audit')
    run = args.run.resolve()
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    events = verify_events(run/'events.jsonl')
    checkpoints = [e['payload'] for e in events if e['event'] == 'checkpoint' and e['payload']['epoch'] == args.rollouts]
    assert len(checkpoints) == 1
    binding = checkpoints[0]
    assert file_sha256(binding['path']) == binding['sha256']
    actual = torch.load(binding['path'], map_location='cpu', weights_only=True)
    parent_path = Path(actual['phase_transition']['parent_checkpoint'])
    assert file_sha256(parent_path) == actual['phase_transition']['parent_sha256']
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
    assert parent['graph_fingerprint'] == actual['graph_fingerprint'] and parent['model_config'] == actual['model_config']
    assert parent['encoder'] == actual['encoder']
    assert equal(actual['prior_rl_state'], parent['rl_state'])
    assert equal(actual['prior_training_config'], parent['training_config'])
    assert 'rl_state' not in actual and actual['phase_transition']['new_actor_only_optimizer']
    encoder = _encoder(parent['encoder'])
    feedback = MovementFeedback()
    previous_episode = None
    evidence = []
    active = None
    update_coverage = {}
    for e in events:
        if e['event'] == 'sealed_movement_rollout':
            active = e['payload']['index']
            if active < args.rollouts: evidence.append(e['payload'])
        elif e['event'] == 'movement_full_batch_update' and active is not None and active < args.rollouts:
            update_coverage.setdefault(active, []).append(e['payload'])
    assert len(evidence) == args.rollouts
    kinds, decisions, pids = Counter(), 0, set()
    for entry in evidence:
        assert file_sha256(entry['native']['path']) == entry['native']['sha256']
        assert file_sha256(entry['replay_path']) == entry['replay_sha256']
        data = torch.load(io.BytesIO(gzip.decompress(Path(entry['replay_path']).read_bytes())), map_location='cpu', weights_only=True)
        assert data['native_binding'] == entry['native']
        target, masks, local_kinds, observations, actions = [], [], [], [], []
        with gzip.open(entry['native']['path'], 'rt', encoding='utf-8') as stream:
            for index, raw in enumerate(stream):
                row = json.loads(raw)
                if row['episode_id'] != previous_episode:
                    feedback.reset(); previous_episode = row['episode_id']
                snapshot = Snapshot.from_dict(row['snapshot'])
                t, mask, kind, details = feedback.query(snapshot)
                assert row['rollout_step'] == index and row['episode_id'] == data['episode_ids'][index]
                assert row['action'] < 18 and row['teacher_action_calls'] == row['safety_takeovers'] == 0
                assert row['teacher_label_queries'] == 1
                assert row['movement_details'] == details and row['movement_kind'] == kind
                np.testing.assert_array_equal(np.asarray(row['movement_target']), t)
                np.testing.assert_array_equal(np.asarray(row['movement_allowed']), mask)
                target.append(t); masks.append(mask); local_kinds.append(kind)
                observations.append(encoder.encode(snapshot)); actions.append(row['action'])
                if 'pid' in snapshot.extra: pids.add(snapshot.extra['pid'])
        np.testing.assert_array_equal(data['data']['observations'].numpy(), np.asarray(observations))
        np.testing.assert_array_equal(data['data']['targets'].numpy(), np.asarray(target))
        np.testing.assert_array_equal(data['data']['allowed'].numpy(), np.asarray(masks))
        np.testing.assert_array_equal(data['data']['weights'].numpy(), balanced_weights(local_kinds))
        np.testing.assert_array_equal(data['data']['actions'].numpy(), actions)
        assert data['kinds'] == local_kinds
        n = len(actions)
        if entry['active_samples']:
            passes = update_coverage[entry['index']]
            assert [p['pass'] for p in passes] == [1, 2]
            assert all(p['sample_coverage']['samples'] == n and p['Adam_steps'] == 1 for p in passes)
        kinds.update(local_kinds); decisions += n
    changes = {name: {'nonzero_changes': int((actual['trainable_parameters'][name] != value).sum()),
        'maximum_absolute_change': float((actual['trainable_parameters'][name] - value).abs().max())}
        for name, value in parent['trainable_parameters'].items()}
    assert changes['recurrent_gain_logits']['nonzero_changes'] > 0
    assert changes['geometry_input_encoder.weight']['nonzero_changes'] > 0
    steps = sorted(set(float(v['step']) for v in actual['movement_learning_state']['optimizer_state_dict']['state'].values()))
    assert steps == [args.rollouts * 2]
    assert torch.equal(parent['trainable_parameters']['action_readout.weight'][18:], actual['trainable_parameters']['action_readout.weight'][18:])
    result = {'schema': 'actual-full-brain-movement-first-learning-audit-v1', 'status': 'passed',
        'run': str(run), 'checkpoint': binding, 'parent_sha256': actual['phase_transition']['parent_sha256'],
        'actual_native_decisions': decisions, 'actual_rollouts': args.rollouts, 'sample_kinds': dict(kinds),
        'all_before_state_labels_masks_weights_observations_recomputed_equal': True,
        'every_full_batch_pass_covers_all_decisions': True, 'actual_actor_Adam_steps': steps,
        'changes': changes, 'prior_complete_PPO_state_preserved_bitwise': True,
        'Bomb_readout_rows_unchanged_in_movement_lesson': True,
        'teacher_actions_executed': 0, 'safety_takeovers': 0,
        'independent_native_evaluation_required': True, 'final_acceptance': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream: json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'status': 'passed', 'native_decisions': decisions, 'kinds': dict(kinds),
        'Adam_steps': steps, 'changes': changes, 'output': str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
