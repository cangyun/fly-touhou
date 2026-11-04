"""Check native demonstration labels and real complete-brain imitation updates."""
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

from audit_movement_learning import equal
from fly_touhou.avoidance_teacher import AvoidanceOnlyFeedback, avoidance_specification
from fly_touhou.experiment import file_sha256, verify_events
from fly_touhou.movement_curriculum import balanced_weights
from fly_touhou.state import Snapshot
from fly_touhou.whole_brain_learning import _encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--rollouts', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve previous imitation audits')
    run = args.run.resolve()
    events = verify_events(run/'events.jsonl')
    records = [e['payload'] for e in events if e['event'] == 'checkpoint' and e['payload']['epoch'] == args.rollouts]
    assert len(records) == 1
    binding = records[0]
    assert file_sha256(binding['path']) == binding['sha256']
    cp = torch.load(binding['path'], map_location='cpu', weights_only=True)
    transition = cp['phase_transition']
    assert file_sha256(transition['parent_checkpoint']) == transition['parent_sha256']
    parent = torch.load(transition['parent_checkpoint'], map_location='cpu', weights_only=True)
    assert equal(cp['prior_rl_state'], parent['rl_state'])
    assert cp['model_config'] == parent['model_config'] and cp['graph_fingerprint'] == parent['graph_fingerprint']
    lesson = cp['training_config']['movement_lesson']
    assert lesson['movement_curriculum'] == avoidance_specification()
    assert lesson['training_action_source'] == 'teacher-demo'
    feedback = AvoidanceOnlyFeedback()
    encoder = _encoder(parent['encoder'])
    current, passes, entries = None, {}, []
    for event in events:
        if event['event'] == 'sealed_movement_rollout':
            current = event['payload']['index']
            if current < args.rollouts: entries.append(event['payload'])
        elif event['event'] == 'movement_full_batch_update' and current is not None and current < args.rollouts:
            passes.setdefault(current, []).append(event['payload'])
    assert len(entries) == args.rollouts
    count, teacher_actions, kinds, previous_id = 0, 0, Counter(), None
    for entry in entries:
        assert file_sha256(entry['native']['path']) == entry['native']['sha256']
        assert file_sha256(entry['replay_path']) == entry['replay_sha256']
        replay = torch.load(io.BytesIO(gzip.decompress(Path(entry['replay_path']).read_bytes())), map_location='cpu', weights_only=True)
        expected_targets, expected_masks, expected_observations, local_kinds = [], [], [], []
        with gzip.open(entry['native']['path'], 'rt', encoding='utf-8') as stream:
            for index, raw in enumerate(stream):
                row = json.loads(raw)
                if row['episode_id'] != previous_id:
                    feedback.reset(); previous_id = row['episode_id']
                s = Snapshot.from_dict(row['snapshot'])
                target, allowed, kind, detail = feedback.query(s)
                assert row['rollout_step'] == index and row['episode_id'] == replay['episode_ids'][index]
                assert row['teacher_action_calls'] == 1 and row['teacher_label_queries'] == 1 and row['safety_takeovers'] == 0
                assert row['action_source'] == 'pure-avoidance-teacher-demonstration'
                assert row['action'] == int(target.argmax()) < 18
                assert row['movement_details'] == detail
                np.testing.assert_array_equal(row['movement_target'], target)
                np.testing.assert_array_equal(row['movement_allowed'], allowed)
                expected_targets.append(target); expected_masks.append(allowed)
                expected_observations.append(encoder.encode(s)); local_kinds.append(kind)
                count += 1; teacher_actions += 1
        for name, expected in [('targets', expected_targets), ('allowed', expected_masks), ('observations', expected_observations),
                               ('weights', balanced_weights(local_kinds))]:
            np.testing.assert_array_equal(replay['data'][name].numpy(), np.asarray(expected))
        assert [p['pass'] for p in passes[entry['index']]] == [1, 2]
        assert all(p['sample_coverage']['samples'] == len(local_kinds) for p in passes[entry['index']])
        kinds.update(local_kinds)
    changes = {name: int((cp['trainable_parameters'][name] != p).sum()) for name, p in parent['trainable_parameters'].items()}
    assert changes['recurrent_gain_logits'] > 0 and changes['geometry_input_encoder.weight'] > 0
    adam_steps = sorted(set(float(v['step']) for v in cp['movement_learning_state']['optimizer_state_dict']['state'].values()))
    assert adam_steps == [2. * args.rollouts]
    assert torch.equal(cp['trainable_parameters']['action_readout.weight'][18:], parent['trainable_parameters']['action_readout.weight'][18:])
    from fly_touhou.geometry_migration import validate_geometry_deployment
    validate_geometry_deployment(cp)
    result = {'schema': 'native-pure-avoidance-complete-brain-imitation-learning-audit-v1', 'status': 'passed',
        'run': str(run), 'checkpoint': binding, 'native_demonstration_steps': count,
        'training_teacher_actions_explicitly_executed': teacher_actions, 'sample_kinds': dict(kinds),
        'all_labels_and_current_observations_recomputed_equal': True,
        'all_samples_processed_every_pass': True, 'actual_new_Adam_steps': adam_steps,
        'changed_parameters': changes, 'parent_PPO_history_preserved_bitwise': True,
        'sensory_deployment_proof_retained_and_validated': True,
        'evaluation_teacher_actions_required': 0, 'independent_student_evaluation_pending': True,
        'scope': 'actual GPU whole-brain imitation of native teacher data; teacher demonstration survival is not student performance',
        'final_acceptance': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream: json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'status': 'passed', 'native_teacher_steps': count, 'kinds': dict(kinds),
        'Adam_steps': adam_steps, 'changed_parameters': changes, 'output': str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
