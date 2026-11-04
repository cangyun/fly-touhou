"""Compare a declared brain-Adam epsilon intervention on real frozen games.

The training-source hashes intentionally differ. Actual saved initial model,
moments, RNG and named parameter groups must match except for the one epsilon.
This never weakens the separate same-full-source comparison contract.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json
from summarize_object_avoidance import completed_avoidance_report, paired_frozen_evaluations, quantile


def initial_checkpoint_proof(control, candidate):
    """Load dictionaries only on CPU; never instantiate or forward a model."""
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    import torch
    torch.set_num_threads(2)
    from fly_touhou.continuous_object_learning_model import _same
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    snapshots, digests = [], []
    for directory in (control, candidate):
        path = directory / 'initial-training.pt'
        with _shared_binary(path) as stream:
            digests.append(hashlib.file_digest(stream, 'sha256').hexdigest())
            stream.seek(0)
            snapshots.append(torch.load(stream, map_location='cpu', weights_only=True))
    a, b = snapshots
    assert a['schema'] == b['schema'] == 'continuous-action-query-object-DN-deep128x3-named-Adam-training-v1'
    for key in ('model_payload', 'RNG', 'optimizer_parameter_signature', 'counters'):
        assert _same(a[key], b[key]), 'Initial mismatch: ' + key
    assert _same(a['optimizer']['state'], b['optimizer']['state']), 'Initial named Adam moments/steps differ'
    ag, bg = a['optimizer']['param_groups'], b['optimizer']['param_groups']
    assert len(ag) == len(bg)
    changes = []
    for left, right in zip(ag, bg):
        expected = dict(left)
        if left.get('name') == 'brain':
            assert left['eps'] == 1e-8 and right['eps'] == 1e-10
            changes.append(dict(group='brain', old_epsilon=left['eps'], new_epsilon=right['eps']))
            expected['eps'] = right['eps']
        assert _same(expected, right), 'Unexpected Adam setting change'
    assert len(changes) == 1
    assert not torch.cuda.is_initialized()
    return dict(schema='brain-epsilon-actual-initial-snapshot-proof-v1',
        initial_checkpoint_SHA256=dict(control=digests[0], candidate=digests[1]),
        complete_model_payload_equal=True, all_Adam_moments_and_steps_equal=True,
        all_RNG_equal=True, initial_counters_equal=True, exact_named_signature_equal=True,
        only_optimizer_setting_changes=changes, model_forwards=0, native_actions=0,
        optimizer_updates=0, CUDA_initialized=False)


def summary(e):
    values = [r['native_frames'] for r in e['episode_results']]
    return dict(endpoint_median_lower_bound=statistics.median(values),
        endpoint_q25_lower_bound=quantile(values, .25),
        natural_clear_censored=sum(r['right_censored_at_natural_clear'] for r in e['episode_results']),
        first_stage_without_contact=sum(r['first_stage_without_observed_contact'] for r in e['episode_results']))


def summarize(control, candidate, output, *, control_specification, candidate_specification,
        initial_proof=None):
    if output.exists():
        raise FileExistsError('Fresh comparison output required')
    cr, ce, _ = completed_avoidance_report(control)
    qr, qe, _ = completed_avoidance_report(candidate)
    assert cr['schema'] == qr['schema'] == 'continuous-action-query-object-NoBomb-survival-PPO-entry-v1'
    assert cr['costs']['student_actions'] == qr['costs']['student_actions'] == 30720
    assert cr['algorithm']['PPO'] == qr['algorithm']['PPO']
    cl, ql = (read_shared_json(p / 'launch.json') for p in (control, candidate))
    for key in ('environment_config_sha256', 'student_decisions', 'worlds', 'steps_per_world',
            'training_seed', 'evaluation_seed', 'K', 'brain_BPTT', 'source_kind'):
        assert cl[key] == ql[key], ('Control configuration differs', key)
    specs = [read_shared_json(p) for p in (control_specification, candidate_specification)]
    matched_arguments = {}
    for flag in ('--seed', '--action-seed', '--training-seed', '--evaluation-seed',
            '--decisions', '--environment-config', '--connectome', '--source-kind'):
        values = [s['argv'][s['argv'].index(flag) + 1] for s in specs]
        assert values[0] == values[1], ('Dispatch argument differs', flag)
        matched_arguments[flag] = values[0]
    if initial_proof is None:
        proof = initial_checkpoint_proof(control, candidate)
    else:
        proof = read_shared_json(initial_proof)
        assert proof['schema'] == 'brain-epsilon-actual-initial-snapshot-proof-v1'
        for key in ('complete_model_payload_equal', 'all_Adam_moments_and_steps_equal',
                'all_RNG_equal', 'initial_counters_equal', 'exact_named_signature_equal'):
            assert proof[key] is True
        assert proof['only_optimizer_setting_changes'] == [
            dict(group='brain', old_epsilon=1e-8, new_epsilon=1e-10)]
        for label, directory in (('control', control), ('candidate', candidate)):
            with (directory / 'initial-training.pt').open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == proof['initial_checkpoint_SHA256'][label]
        proof = dict(proof, reused_hash_bound_actual_proof=True)
    pairs, _, details = paired_frozen_evaluations(qe, ce, 'brain_epsilon_intervention')
    result = dict(schema='NoBomb-brain-epsilon-intervention-frozen20-v1',
        UTC=datetime.now(timezone.utc).isoformat(), control_path=str(control), candidate_path=str(candidate),
        control_source_SHA=cr['provenance']['parent']['sha256'],
        candidate_source_SHA=qr['provenance']['parent']['sha256'],
        different_training_source_hashes_are_intentional=True, initialization_proof=proof,
        matched_dispatch_arguments=matched_arguments, control=summary(ce), candidate=summary(qe),
        paired=pairs, episode_pairs=details, actual_costs=dict(control=cr['costs'], candidate=qr['costs']),
        scope='Same actual initial model/moments/RNG, one brain-group epsilon change; '
            'reused development20 with natural-clear censoring. Not formal stability, '
            'not proof a particular mechanism caused any improvement.',
        model_forwards=0, native_actions=0, optimizer_updates=0, full_story_PPO_resumed=False)
    output.mkdir(parents=True)
    write_json(output / 'report.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--control-specification', type=Path, required=True)
    p.add_argument('--candidate-specification', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--initial-proof', type=Path)
    a = p.parse_args()
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    result = summarize(a.control, a.candidate, a.output,
        control_specification=a.control_specification, candidate_specification=a.candidate_specification,
        initial_proof=a.initial_proof)
    print(json.dumps({k:result[k] for k in ('control', 'candidate', 'paired')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
