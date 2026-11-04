"""Measure causal response to emergency player flags in a full-brain replay.

Two branches share the exact pre-decision neural state and counter-based noise.
Only current deathbomb_timer/player_state inputs differ. Repeating the input
afterward is a frozen-stimulus diagnostic, not an alternate native trajectory.
"""
from pathlib import Path
import argparse
import gzip
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import torch
from fly_touhou.connectome import load_connectome, file_sha256
from fly_touhou.state import Snapshot
from fly_touhou.whole_brain import WholeBrainState
from fly_touhou.whole_brain_learning import load_whole_brain_checkpoint, _prime_state
from fly_touhou.whole_brain_readout import input_parameter_hashes


def paired_state(state):
    def double(value):
        return torch.cat((value, value), dim=0)
    return WholeBrainState(**{name: (tuple(double(v) for v in value) if name == 'delay_queue'
                                   else double(value)) for name, value in vars(state).items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cases-per-episode', type=int, default=3)
    parser.add_argument('--response-decisions', type=int, default=4)
    parser.add_argument('--capture-trajectory', action='store_true', help='Capture only the ordinary batch1 trajectory; paired branches use exact validated tensor operations')
    parser.add_argument('--normal-player-neutral', action='store_true', help='Control values are encoded player state1 and zero deathbomb timer, rather than the legacy zero-vector intervention')
    parser.add_argument('--save-descending-features', action='store_true', help='Retain paired DN-only features so another head with exactly the same sensory/brain identity can be inspected offline')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Do not overwrite input-response audits')
    if args.cases_per_episode < 1 or args.response_decisions < 1:
        parser.error('Case/response limits must be positive')
    torch.set_num_threads(4)
    graph = load_connectome(args.source_dir, max_neurons=None)
    policy = load_whole_brain_checkpoint(args.checkpoint, graph, args.device)
    model = policy.student.eval()
    model.verify_propagation_buffers()
    if args.capture_trajectory:
        model.enable_cuda_capture()
    features = list(policy.encoder.feature_names)
    indices = [features.index(name) for name in ('deathbomb_timer', 'player_state')]
    cases, state, episode, case_count, last_case = [], None, None, 0, -100
    started = time.monotonic()
    with torch.no_grad(), gzip.open(args.trace, 'rt', encoding='utf-8') as stream:
        header = json.loads(next(stream))
        if header['encoder'] != policy.encoder.specification():
            raise ValueError('Trace observation contract differs from the checkpoint')
        for line in stream:
            row = json.loads(line)
            if row.get('record_type') != 'transition':
                continue
            snapshot = Snapshot.from_dict(row['snapshot'])
            observation = torch.as_tensor(policy.encoder.encode(snapshot), device=model.device)
            if row['episode_id'] != episode:
                episode, case_count, last_case = row['episode_id'], 0, -100
                state = _prime_state(model, observation, policy.warmup_ms)
            emergency = snapshot.bombs > 0 and (snapshot.player.state == 4 or snapshot.player.deathbomb_timer > 0)
            if emergency and case_count < args.cases_per_episode and row['step'] - last_case > 10:
                neutral = observation.clone()
                if args.normal_player_neutral:
                    ordinary = snapshot.to_dict()
                    ordinary['player']['state'] = 1
                    ordinary['player']['deathbomb_timer'] = 0
                    neutral_encoding = policy.encoder.encode(Snapshot.from_dict(ordinary))
                    neutral[indices] = torch.as_tensor(neutral_encoding[indices], device=model.device)
                else:
                    neutral[indices] = 0
                paired = torch.stack((observation, neutral))
                branch = paired_state(state)
                model._check_state(branch, 2)
                if not bool(torch.isfinite(paired).all()):
                    raise ValueError('Nonfinite paired observation intervention')
                firing_probability = torch.sigmoid(model.input_encoder(paired)) * model.max_input_rate_hz*model.dt_ms/1000
                case = {'episode_id': episode, 'step': row['step'], 'native_frame': snapshot.frame,
                    'native_player_state': snapshot.player.state, 'native_deathbomb_timer': snapshot.player.deathbomb_timer,
                    'changed_features': {features[i]: float(observation[i]) for i in indices},
                    'control_features': {features[i]: float(neutral[i]) for i in indices},
                    'input_probability_l1_difference': float((firing_probability[0]-firing_probability[1]).abs().sum()),
                    'response': []}
                for window in range(1, args.response_decisions+1):
                    # The optional ordinary-trajectory capture has batch1.
                    # These shape/finite-validated paired branches share exact
                    # full LIF operations, with no capture batch substitution.
                    logits, branch = model._forward_validated(paired, branch)
                    probability = torch.softmax(logits, -1)
                    case['response'].append({'decisions': window,
                        'neural_ms': window*model.neural_ms_per_decision,
                        'max_logit_difference': float((logits[0]-logits[1]).abs().max()),
                        'action_total_variation': float((probability[0]-probability[1]).abs().sum()/2),
                        'emergency_bomb_probability': float(probability[0,18:].sum()),
                        'neutral_bomb_probability': float(probability[1,18:].sum()),
                        'different_descending_spike_counts': int((branch.window_spike_counts[0,model.output_indices]
                            != branch.window_spike_counts[1,model.output_indices]).sum()),
                        'different_descending_filtered_rates': int((branch.rate[0] != branch.rate[1]).sum()),
                        'different_brain_spike_counts': int((branch.window_spike_counts[0]
                            != branch.window_spike_counts[1]).sum())})
                    if args.save_descending_features:
                        case['response'][-1]['descending_spike_features'] = model.readout_features(branch).cpu().tolist()
                cases.append(case)
                case_count += 1
                last_case = row['step']
            _, state = model(observation, state)
    report = {'schema': 'whole-brain-emergency-input-causal-response-v1',
        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': file_sha256(args.checkpoint),
        'trace': str(args.trace.resolve()), 'trace_sha256': file_sha256(args.trace),
        'script_sha256': file_sha256(Path(__file__)), 'neurons': model.n_neurons, 'edges': model.graph.n_edges,
        'model_config': model.config(), 'cases': cases, 'wall_seconds': time.monotonic()-started,
        'input_parameter_hashes': input_parameter_hashes(model),
        'inference_execution': model.inference_execution_specification(),
        'control_kind': 'encoded state1 and deathbomb0; only those two coordinates changed' if args.normal_player_neutral else 'legacy zero values at two coordinates',
        'source_files': {name: file_sha256(Path(__file__).parents[1]/'src/fly_touhou'/f'{name}.py')
                         for name in ('whole_brain','whole_brain_learning','hazard_observation','cached_cusparse','brain_cuda_capture')},
        'teacher_action_calls': 0, 'new_native_actions': 0,
        'boundary': 'controlled frozen-input offline full-brain response; not alternative native survival evidence'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as target:
        json.dump(report, target, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'output': str(args.output), 'cases': len(cases), 'seconds': report['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
