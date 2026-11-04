"""Repair missing inherited proof metadata without any new learning or game."""
from pathlib import Path
import argparse
from copy import deepcopy
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import torch

from fly_touhou.experiment import file_sha256
from fly_touhou.movement_checkpoint import inherit_geometry_validation


def equal(a, b):
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)): return type(a) is type(b) and len(a) == len(b) and all(equal(v, w) for v, w in zip(a, b))
    return type(a) is type(b) and a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Keep the failed export and use a fresh qualified artifact')
    original = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    transition = original['phase_transition']
    parent_path = Path(transition['parent_checkpoint'])
    assert file_sha256(parent_path) == transition['parent_sha256']
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
    qualified = deepcopy(original)
    inherit_geometry_validation(qualified, parent)
    qualified['inherited_parent_epoch'] = original['epoch']
    qualified['epoch'] = original['movement_learning_state']['updates']
    retained = ('trainable_parameters', 'movement_learning_state', 'prior_rl_state',
        'prior_training_config', 'metrics', 'model_config', 'encoder', 'graph_fingerprint')
    checks = {key: equal(original[key], qualified[key]) for key in retained}
    assert all(checks.values())
    args.output.mkdir(parents=True)
    path = args.output/'qualified.pt'
    torch.save(qualified, path)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert all(equal(saved[key], original[key]) for key in retained)
    report = {'schema': 'movement-checkpoint-inherited-geometry-metadata-repair-v1', 'status': 'passed',
        'original_checkpoint': str(args.checkpoint.resolve()), 'original_sha256': file_sha256(args.checkpoint),
        'qualified_checkpoint': str(path.resolve()), 'qualified_sha256': file_sha256(path),
        'parent_checkpoint': str(parent_path), 'parent_sha256': transition['parent_sha256'],
        'all_learning_fields_preserved_bitwise': checks,
        'metadata_added': ['geometry_observation_transition', 'geometry_validation_proof'],
        'phase_counter_corrected': {'old_top_level_epoch': original['epoch'], 'movement_updates': saved['epoch']},
        'numerical_or_native_updates': 0, 'scope': 'restore missing inherited sensory proof binding; not a new model or new validation success'}
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
