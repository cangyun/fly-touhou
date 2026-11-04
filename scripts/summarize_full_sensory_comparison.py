"""Frozen versus adaptable sensation with actual initialization evidence.

Read dictionaries and completed native reports only. No model is constructed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json
from summarize_object_avoidance import completed_avoidance_report, paired_frozen_evaluations


def initialization_proof(control, candidate):
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    import torch
    torch.set_num_threads(2)
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    from fly_touhou.continuous_object_learning_model import _same
    snapshots, hashes = [], {}
    for label, directory in (('control', control), ('candidate', candidate)):
        with _shared_binary(directory / 'initial-training.pt') as stream:
            hashes[label] = hashlib.file_digest(stream, 'sha256').hexdigest()
            stream.seek(0)
            snapshots.append(torch.load(stream, map_location='cpu', weights_only=True))
    a, b = snapshots
    assert a['schema'] == 'continuous-action-query-object-DN-deep128x3-named-Adam-training-v1'
    assert b['schema'] == 'continuous-full-sensory-action-query-DN-deep128x3-named-Adam-training-v1'
    p = b['model_payload']
    assert _same(a['model_payload'], p['query_model_payload'])
    frozen = a['model_payload']['object_model_payload']['base_parameters']['perception']
    for field, prefix in (('current_CNN_state', 'map_encoder.'), ('current_numeric_state', 'feature_proj.')):
        expected = {key[len(prefix):]: value for key, value in frozen.items() if key.startswith(prefix)}
        assert expected and _same(expected, p[field]), field
    for key in ('RNG', 'counters'):
        assert _same(a[key], b[key]), key
    assert _same(a['optimizer']['state'], b['optimizer']['state'])
    old_groups, new_groups = a['optimizer']['param_groups'], b['optimizer']['param_groups']
    assert len(new_groups) == len(old_groups) + 1
    assert _same(old_groups, new_groups[:-1])
    assert _same(a['optimizer_parameter_signature'], b['optimizer_parameter_signature'][:-1])
    added = b['optimizer_parameter_signature'][-1]
    assert new_groups[-1]['name'] == 'sensory_CNN_numeric' and len(added) == 32
    assert all(item['name'].startswith(('sensory_adapter.map_encoder.', 'sensory_adapter.feature_proj.')) for item in added)
    assert not any(i in b['optimizer']['state'] for i in new_groups[-1]['params'])
    assert p['configuration']['perception']['frozen_parameters'] == []
    assert not torch.cuda.is_initialized()
    return dict(schema='full-sensory-actual-initialization-proof-v1', initial_checkpoint_SHA256=hashes,
        all_effective_parameter_values_equal=True, old_named_Adam_states_and_groups_equal=True,
        RNG_equal=True, counters_equal=True, newly_trainable_CNN_numeric_tensors=32,
        new_CNN_numeric_Adam_states=0, new_weight_parameters=0,
        source_CNN_snapshot_is_not_authoritative_current_CNN=True,
        model_forwards=0, native_actions=0, optimizer_updates=0, CUDA_initialized=False)


def summarize(control, candidate, output, *, initial_proof):
    if output.exists():
        raise FileExistsError('Fresh comparison evidence required')
    cr, ce, _ = completed_avoidance_report(control)
    qr, qe, _ = completed_avoidance_report(candidate)
    assert cr['schema'] == 'continuous-action-query-object-NoBomb-survival-PPO-entry-v1'
    assert qr['schema'] == 'continuous-full-sensory-NoBomb-survival-PPO-entry-v1'
    assert cr['costs']['student_actions'] == qr['costs']['student_actions'] == 30720
    assert cr['algorithm']['PPO'] == qr['algorithm']['PPO']
    cl, ql = (read_shared_json(d / 'launch.json') for d in (control, candidate))
    for key in ('environment_config_sha256', 'worlds', 'steps_per_world', 'student_decisions',
            'training_seed', 'evaluation_seed', 'K', 'brain_BPTT'):
        assert cl[key] == ql[key], key
    proof = read_shared_json(initial_proof)
    assert proof['schema'] == 'full-sensory-actual-initialization-proof-v1'
    assert proof['all_effective_parameter_values_equal'] and proof['old_named_Adam_states_and_groups_equal']
    assert proof['RNG_equal'] and proof['newly_trainable_CNN_numeric_tensors'] == 32
    for label, directory in (('control', control), ('candidate', candidate)):
        with (directory / 'initial-training.pt').open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == proof['initial_checkpoint_SHA256'][label]
    paired, reference, details = paired_frozen_evaluations(qe, ce, 'full_sensory_minus_frozen')
    result = dict(schema='full-sensory-versus-frozen-native20-v1', UTC=datetime.now(timezone.utc).isoformat(),
        control_path=str(control), candidate_path=str(candidate), initialization_proof=proof,
        control=reference, candidate=qe['observed_endpoint_frames'],
        candidate_stage1_no_contact=qe['first_stage_without_observed_contact_count'],
        paired=paired, episode_pairs=details, actual_costs=dict(control=cr['costs'], candidate=qr['costs']),
        scope='Same initial effective model/oldAdam/RNG, CNN+spatial+numeric adaptation enabled; '
            'same PPO/reward budget. Reused development20,not formal stability or proof of a unique cause.',
        model_forwards=0, native_actions=0, optimizer_updates=0, full_story_PPO_resumed=False)
    output.mkdir(parents=True)
    write_json(output / 'report.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--initial-proof', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    report = summarize(a.control, a.candidate, a.output, initial_proof=a.initial_proof)
    print(json.dumps({k:report[k] for k in ('control', 'candidate', 'paired')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
