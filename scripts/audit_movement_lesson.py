"""Read-only audit of a completed independent ordinary18 first-hit diagnostic."""
from pathlib import Path
import argparse
import gzip
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import torch

from fly_touhou.experiment import file_sha256, verify_events
from fly_touhou.whole_brain_learning import FORMAL_GAME_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.output.exists():
        raise FileExistsError('Preserve previous completed audits')
    report = json.loads((run/'report.json').read_text(encoding='utf-8'))
    assert report['status'] == 'finished' and report['config']['mode'] == 'evaluate'
    config = report['config']
    assert config['difficulty'] == 3 and config['movement_curriculum']['action_domain'] == list(range(18))
    assert config['game_exe_sha256'] == FORMAL_GAME_SHA256
    checkpoint = torch.load(config['checkpoint'], map_location='cpu', weights_only=True)
    assert file_sha256(config['checkpoint']) == config['checkpoint_sha256']
    assert checkpoint['neurons'] == 138639 and checkpoint['edge_records'] == 15091983
    weight = checkpoint['trainable_parameters']['action_readout.weight'].double().numpy()
    reference = checkpoint['model_config']['parameters']['reference_poisson_rate_hz']
    events = verify_events(run/'events.jsonl')
    sealed = [e['payload'] for e in events if e['event'] == 'sealed_native_movement_evaluation_part']
    observed, next_steps = {}, {}
    decisions = 0
    maximum_readout_error = 0.
    for binding in sealed:
        path = Path(binding['path'])
        assert path.is_relative_to(run) and file_sha256(path) == binding['sha256']
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for raw in stream:
                row = json.loads(raw)
                assert row['record_type'] == 'transition' and row['action'] < 18
                assert row['teacher_action_calls'] == row['teacher_label_queries'] == row['safety_takeovers'] == 0
                assert row['action_source'] == 'complete-brain-ordinary18-lesson'
                identity = row['episode_id']
                assert row['episode_step'] == next_steps.get(identity, 0)
                next_steps[identity] = row['episode_step'] + 1
                for snapshot in (row['snapshot'], row['next_snapshot']):
                    assert snapshot['source'] == 'native-process' and snapshot['difficulty'] == 3
                    assert snapshot['extra']['desktop_input'] is False
                    assert snapshot['extra'].get('exe_sha256', FORMAL_GAME_SHA256) == FORMAL_GAME_SHA256
                    assert snapshot['extra']['run_mode'] == snapshot['extra']['continues'] == snapshot['extra']['replay_playback'] == 0
                evidence = row['action_evidence']
                rates = np.asarray(evidence['descending_filtered_rates_hz'], dtype=np.float32)
                features = (rates / np.float32(reference)).astype(np.float64)
                rebuilt = weight @ features
                declared = np.asarray(evidence['logits36'], dtype=np.float64)
                error = float(np.abs(rebuilt - declared).max())
                maximum_readout_error = max(maximum_readout_error, error)
                assert error < 5e-5 and row['action'] == int(declared[:18].argmax())
                assert evidence['action_domain'] == list(range(18))
                assert row['noise_counter_after'] - row['noise_counter_before'] == checkpoint['model_config']['substeps']
                entry = observed.setdefault(identity, {'native_frames': 0, 'hits': 0, 'vulnerable_frames': 0})
                entry['native_frames'] += row['elapsed_frames']
                entry['hits'] += int(row['hit'])
                p = row['snapshot']['player']
                if p and p['state'] == 1 and p['invulnerable_frames'] == 0:
                    entry['vulnerable_frames'] += row['elapsed_frames']
                decisions += 1
    assert len(observed) == len(report['episodes']) == 4
    for episode in report['episodes']:
        actual = observed[episode['episode_id']]
        assert episode['native_frames_to_first_hit_or_limit'] == actual['native_frames']
        assert episode['vulnerable_frames_before_end'] == actual['vulnerable_frames']
        assert actual['hits'] == int(episode['hit']) and episode['bomb_actions'] == 0
    result = {'schema': 'independent-movement-first-hit-native-readout-audit-v1', 'status': 'passed',
        'run': str(run), 'report_sha256': file_sha256(run/'report.json'),
        'checkpoint_sha256': config['checkpoint_sha256'], 'native_decisions_verified': decisions,
        'maximum_CPU_GPU_descending_readout_error': maximum_readout_error,
        'frames_to_first_hit_or_limit': [e['native_frames_to_first_hit_or_limit'] for e in report['episodes']],
        'vulnerable_frames': [e['vulnerable_frames_before_end'] for e in report['episodes']],
        'Bombs_requested': 0, 'teacher_queries_or_actions': 0, 'safety_takeovers': 0,
        'action_domain': list(range(18)), 'scope': 'explicit first-hit movement diagnostic; not unrestricted six-stage campaign acceptance',
        'final_acceptance': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
