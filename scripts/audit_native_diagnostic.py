"""Independent CPU readback of original-game diagnostic trajectories."""
from pathlib import Path
import argparse
import gzip
import json
import math
import sys

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE/'src'))
from fly_touhou.background import action_mask, BackgroundTouhouEnv
from fly_touhou.diagnostic_gru import DiagnosticGRU
from fly_touhou.experiment import file_sha256, verify_events
from fly_touhou.first_hit_policy_gradient import first_hit_reward
from fly_touhou.geometry_observation import GeometryObservationEncoder
from fly_touhou.native_diagnostic import checked_snapshot, summarize_episodes, write_json
from fly_touhou.state import Snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():
        raise FileExistsError('Preserve prior independent audits')
    manifest = json.loads((args.run/'manifest.json').read_text(encoding='utf-8'))
    events = verify_events(args.run/'events.jsonl')
    assert events[0]['payload']['manifest_sha256'] == file_sha256(args.run/'manifest.json')
    for item in manifest['files'].values():
        assert file_sha256(item['path']) == item['sha256']
    protocol = json.loads(Path(manifest['files']['protocol']['path']).read_text(encoding='utf-8'))
    encoder = GeometryObservationEncoder.from_specification(protocol['encoder'])
    report = json.loads((args.run/'report.json').read_text(encoding='utf-8'))
    trace = Path(report['trajectory']['path'])
    assert file_sha256(trace) == report['trajectory']['sha256']
    actor_paths = {}
    if 'actor' in report:
        actor_paths['untrained'] = report['actor']
    if 'trained_checkpoint' in report:
        actor_paths['gru'] = report['trained_checkpoint']
        actor_paths['untrained'] = report['untrained_checkpoint']
    models = {}
    for name, item in actor_paths.items():
        assert file_sha256(item['path']) == item['sha256']
        checkpoint = torch.load(item['path'], map_location='cpu', weights_only=True)
        model = DiagnosticGRU()
        model.load_state_dict(checkpoint['actor'])
        model.eval()
        models[name] = model
    sampler = torch.Generator(device='cpu').manual_seed(manifest['config']['seed'] +
        (1_000_003 if 'trained_checkpoint' in report else 0))
    groups, order, previous = {}, [], None
    max_model_error = 0.
    actions = 0
    with gzip.open(trace, 'rt', encoding='utf-8') as stream, torch.no_grad():
        for line in stream:
            row = json.loads(line)
            identity = row['episode_id']
            kind = row['action_source']
            before = Snapshot.from_dict(row['snapshot'])
            after = Snapshot.from_dict(row['next_snapshot'])
            checked_snapshot(before, protocol); checked_snapshot(after, protocol)
            fresh = identity not in groups
            if fresh:
                assert identity not in order
                order.append(identity)
                groups[identity] = {'frames': 0, 'decisions': 0, 'return': 0., 'policy_kind': kind, 'hit': False}
                if kind in models:
                    state = models[kind].prime(torch.from_numpy(encoder.encode(before))[None],
                        protocol['warmup_repeated_observation_passes'])
            else:
                assert previous['episode_id'] == identity
                assert not previous['terminated'] and not previous['truncated']
                assert previous['next_snapshot'] == row['snapshot']
            assert row['episode_step'] == groups[identity]['decisions']
            assert row['teacher_action_calls'] == row['teacher_label_queries'] == row['safety_takeovers'] == 0
            action = row['action']
            assert 0 <= action < 18 and row['requested_native_input_mask'] == action_mask(action)
            assert row['requested_native_input_mask'] & 1 and not row['requested_native_input_mask'] & 2
            assert row['elapsed_frames'] == after.frame-before.frame == 4
            assert row['hit'] == BackgroundTouhouEnv.reward(before, after, 4)[1]
            assert not row['hit'] or row['terminated']
            if kind in models:
                logits, state = models[kind](torch.from_numpy(encoder.encode(before))[None], state)
                lp = logits[0].double().log_softmax(-1)
                error = float((lp-torch.tensor(row['conditional_logp18'], dtype=torch.float64)).abs().max())
                max_model_error = max(max_model_error, error)
                assert error < 2e-6
            if kind == 'static':
                assert action == 0 and row['selection_rule'] == 'fixed-action0'
            else:
                logged = torch.tensor(row['conditional_logp18'], dtype=torch.float64)
                if kind == 'random':
                    assert float((logged+math.log(18)).abs().max()) < 1e-12
                assert int(torch.multinomial(logged.exp(), 1, generator=sampler)) == action
                assert row['selected_conditional_log_probability'] == float(logged[action])
            reward, components = first_hit_reward(row)
            assert reward == row['reward'] and components == row['reward_components']
            item = groups[identity]
            item['frames'] += 4; item['decisions'] += 1; item['return'] += reward
            item['hit'] = row['hit']
            assert item['frames'] <= 1024
            assert row['truncated'] == (item['frames'] == 1024)
            if report['phase'] == 'G1-native-qualification':
                assert row['geometry_reference_parity_checked']
            previous = row
            actions += 1
    assert previous['terminated'] or previous['truncated']
    assert [e['episode_id'] for e in report['episodes']] == order
    for episode in report['episodes']:
        actual = groups[episode['episode_id']]
        assert all(episode[k] == actual[k] for k in ('frames', 'decisions', 'return', 'policy_kind', 'hit'))
        assert episode['horizon_completed_without_hit'] == (not actual['hit'] and actual['frames'] == 1024)
    assert summarize_episodes(report['episodes']) == report['summary']
    result = {'schema': 'independent-original-diagnostic-CPU-audit-v1', 'status': 'passed',
        'auditor_path': str(Path(__file__)), 'auditor_sha256': file_sha256(Path(__file__)),
        'report_sha256': file_sha256(args.run/'report.json'), 'actions_verified': actions,
        'episodes_verified': len(order), 'maximum_GPU_CPU_recurrent_logp_error': max_model_error,
        'original_snapshot_observation_action_sampler_reward_and_first_hit_verified': True,
        'teacher_actions': 0, 'safety_takeovers': 0, 'Bomb_requests': 0,
        'G2_or_G3_achieved': False, 'final_acceptance': False}
    write_json(args.output, result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
