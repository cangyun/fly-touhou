"""Check sealed native action evidence and summarize resource-conditioned Bomb use.

This is a read-only, offline audit. It starts no native game and trains nothing.
Recorded DN rates are checked against the pinned checkpoint's bias-free head.
"""
from pathlib import Path
import argparse
import json
import math

import numpy as np
import torch

from audit_native_policy_trajectories import (
    analyze_episode, file_evidence, rows, stable_bytes, verified_manifest, verify_events,
)


def summarize_group(records):
    result = {'decisions': len(records)}
    if records:
        result.update(mean_bomb_probability=float(np.mean([r['bomb_probability'] for r in records])),
            bomb_request_fraction=float(np.mean([r['bomb_requested'] for r in records])),
            mean_policy_entropy=float(np.mean([r['entropy'] for r in records])))
    return result


def audit_run(run):
    manifest, provenance = verified_manifest(run)
    events = verify_events(stable_bytes(run/'events.jsonl'))
    provenance['events'] = file_evidence(run/'events.jsonl')
    proofs = [e['payload'] for e in events if e['event'] == 'whole_brain_verified']
    if not proofs:
        raise ValueError('Missing full-brain runtime proof')
    proof = proofs[-1]
    if (proof.get('neurons') != 138639 or proof.get('directed_edges') != 15091983
            or proof.get('neuron_coverage') != 1 or proof.get('connection_record_coverage') != 1):
        raise ValueError('Incomplete brain in recorded native evaluation')
    if proof.get('recurrent_weights_frozen') is not True:
        gain = proof.get('internal_plasticity', {})
        if (gain.get('schema') != 'bounded-postsynaptic-gain-v1' or gain.get('enabled') is not True
                or gain.get('base_signed_weights_frozen') is not True or gain.get('limit') != .1
                or gain.get('topology_and_signs_preserved') is not True or gain.get('parameter_count') != 138639):
            raise ValueError('Unrecognized internal weight change in full-brain proof')
    config = manifest['config']
    # Resolve the exact artifact recorded by the evaluation, not mutable latest.pt.
    checkpoint_records = [v for v in manifest['files'].values()
                          if v.get('sha256') == config['checkpoint_sha256']]
    if not checkpoint_records:
        raise ValueError('Evaluation manifest does not pin its actor checkpoint')
    checkpoint_path = Path(checkpoint_records[0]['path'])
    provenance['checkpoint'] = file_evidence(checkpoint_path, config['checkpoint_sha256'])
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    head = checkpoint['trainable_parameters']['action_readout.weight'].double().numpy()
    reference = checkpoint['model_config']['parameters']['reference_poisson_rate_hz']
    if head.shape != (36, 1299) or not np.isfinite(head).all() or reference != 150:
        raise ValueError('Invalid pinned descending-spike readout')
    summary = json.loads(stable_bytes(run/'evaluation_summary.json'))
    provenance['evaluation_summary'] = file_evidence(run/'evaluation_summary.json')
    runtime_config = {k: v for k, v in config.items() if k != 'final_acceptance_criterion'}
    results = []
    for i, campaign in enumerate(summary['campaigns']):
        seal = campaign['acceptance_gate']['evidence']['trajectory']
        trajectory = Path(seal['path'])
        trace_evidence = file_evidence(trajectory, seal['sha256'])
        data = rows(trajectory)
        metadata = next(x for x in data if x['record_type'] == 'metadata')
        if metadata['config'] != runtime_config:
            raise ValueError('Runtime configuration differs from evaluation manifest')
        transitions = [x for x in data if x['record_type'] == 'transition']
        analyzed, features, deaths = analyze_episode(transitions, run.name, str(i), config, campaign['result'])
        max_logit_error = max_probability_error = 0.
        records = []
        for t, f in zip(transitions, features):
            e = t.get('action_evidence', {})
            if (e.get('schema') != 'whole-brain-action-evidence-v1'
                    or e.get('action_path') != 'complete brain descending spikes only'
                    or e.get('selected_action') != t['action']
                    or e.get('noise_counter') != t['neural_noise_counter']):
                raise ValueError('Missing or inconsistent individual native action evidence')
            rates = np.asarray(e['descending_filtered_rates_hz'], dtype=np.float64)
            logits = np.asarray(e['logits'], dtype=np.float64).reshape(-1)
            probabilities = np.asarray(e['probabilities'], dtype=np.float64).reshape(-1)
            if rates.shape != (1299,) or logits.shape != (36,) or probabilities.shape != (36,):
                raise ValueError('Invalid action evidence shape')
            if not all(np.isfinite(x).all() for x in (rates, logits, probabilities)):
                raise ValueError('Nonfinite action evidence')
            logit_error = float(np.max(np.abs(head @ (rates/reference) - logits)))
            p = np.exp(logits-logits.max())
            p /= p.sum()
            probability_error = float(np.max(np.abs(p-probabilities)))
            if logit_error > 1e-4 or probability_error > 1e-6:
                raise ValueError('Recorded action is inconsistent with pinned DN-only readout')
            max_logit_error = max(max_logit_error, logit_error)
            max_probability_error = max(max_probability_error, probability_error)
            entropy = float(-(p*np.log(np.maximum(p, 1e-300))).sum())
            if abs(entropy-e['conditional_policy_entropy']) > 1e-5:
                raise ValueError('Recorded policy entropy differs from actual distribution')
            records.append({**f, 'bomb_probability': float(probabilities[18:].sum()), 'entropy': entropy})
        conditions = {
            'all': lambda r: True,
            'deathbomb_window_with_inventory': lambda r: r['player_state'] == 4 and r['deathbomb_timer'] > 0 and r['bombs'] > 0,
            'vulnerable_with_inventory': lambda r: r['player_state'] == 1 and r['invulnerable'] == 0 and r['bombs'] > 0,
            'invulnerable_with_inventory': lambda r: r['invulnerable'] > 0 and r['bombs'] > 0,
            'zero_inventory': lambda r: r['bombs'] <= 0,
        }
        results.append({'episode': i, 'trajectory': trace_evidence, 'result': analyzed['result'],
            'brain_numerical_runtime': proof.get('numerical_runtime'),
            'internal_plasticity': proof.get('internal_plasticity'),
            'action_evidence': {'decisions_verified': len(records), 'head_shape': list(head.shape),
                'maximum_logit_error_float64_cpu_vs_recorded_float32': max_logit_error,
                'maximum_probability_error': max_probability_error},
            'Bomb_conditions': {k: summarize_group([r for r in records if condition(r)]) for k, condition in conditions.items()},
            'bomb_use': analyzed['bombs'], 'position_controllable_state1': analyzed['position_controllable_state1'],
            'shooting': analyzed['shooting'], 'max_power': max(r['power'] for r in records),
            'death_entries': [{'stage_frame': d['stage_frame'], 'bombs_before_hit': d['bombs_before_hit'],
                               'action': d['action']} for d in deaths],
            'acceptance_passed': campaign['acceptance_gate']['reported_measurements_pass_gate']})
    return {'run': str(run), 'checkpoint_sha256': config['checkpoint_sha256'], 'provenance': provenance,
            'episodes': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior native diagnostics')
    torch.set_num_threads(2)
    report = {'schema': 'sealed-native-DN-action-readout-audit-v1',
        'script': file_evidence(Path(__file__)),
        'support_script': file_evidence(Path(__file__).with_name('audit_native_policy_trajectories.py')),
        'game_started': False, 'model_trained': False,
        'limits': ['Inventory decrements may combine native use and pickups.',
                   'Small independent campaign samples with uncontrolled native RNG do not prove learning gains.',
                   'DN evidence verifies the recorded readout; it does not substitute for replaying the full brain.'],
        'runs': [audit_run(run.resolve()) for run in args.run]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    for run in report['runs']:
        for e in run['episodes']:
            print(json.dumps({'run': Path(run['run']).name, 'episode': e['episode'],
                'frames': e['result']['native_frames'], 'Bomb': e['Bomb_conditions'],
                'readout_max_error': e['action_evidence']['maximum_logit_error_float64_cpu_vs_recorded_float32']},
                ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
