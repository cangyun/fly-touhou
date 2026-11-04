"""CPU-only formal-trace admission with immutable inherited whole-episode splits."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
from fly_touhou.connectome import file_sha256
from fly_touhou.whole_brain_learning import (load_campaign_demonstrations,
    split_campaign_episodes, _split_history, FORMAL_GAME_SHA256)
from audit_native_policy_trajectories import verify_events, verified_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--traces', type=Path, nargs='+', required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--inherited-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'formal-curriculum-source-admission-v1', 'game_started': False,
        'model_trained': False, 'device': 'cpu', 'script_sha256': file_sha256(Path(__file__)),
        'loader_sha256': file_sha256(ROOT / 'src/fly_touhou/whole_brain_learning.py'),
        'parent_sha256': file_sha256(args.parent), 'status': 'running'}
    try:
        torch.set_num_threads(2)
        parent = torch.load(args.parent, map_location='cpu', weights_only=True)
        if parent['training_config']['game_exe_sha256'] != FORMAL_GAME_SHA256:
            raise ValueError('Parent uses a different executable')
        manifest, evidence = verified_manifest(args.inherited_run)
        prefix = (args.inherited_run / 'events.jsonl').read_bytes()
        prefix = prefix[:prefix.rfind(b'\n') + 1]
        events = verify_events(prefix)
        recorded_parent = [e['payload'] for e in events if e['event'] == 'initialized_from_parent_curriculum'][-1]
        if recorded_parent['sha256'] != report['parent_sha256']:
            raise ValueError('Inherited course is not linked to this exact parent')
        split = [e['payload'] for e in events if e['event'] == 'dataset_split'][-1]
        inherited = _split_history(parent, split['train'], split['validation'])
        with (args.output / 'inherited_events_prefix.jsonl').open('xb') as stream:
            stream.write(prefix)
        report.update(inherited_run_manifest=evidence['manifest'],
            inherited_event_prefix_sha256=hashlib.sha256(prefix).hexdigest(), inherited_split=inherited)
        print(json.dumps({'phase': 'validate_and_encode_complete_native_traces',
            'traces': [str(x) for x in args.traces]}), flush=True)
        started = time.perf_counter()
        dataset = load_campaign_demonstrations(args.traces)
        if dataset.encoder.specification() != parent['encoder']:
            raise ValueError('Extended source observation contract differs from the trained parent')
        train, val = split_campaign_episodes(dataset.episodes, .5, args.seed,
            fixed_train_ids=inherited['training'], fixed_validation_ids=inherited['validation'])
        cumulative = _split_history({'split_history': inherited,
            'train_episode_ids': [], 'validation_episode_ids': []},
            [e.episode_id for e in train], [e.episode_id for e in val])
        def summarize(episodes):
            return [{'id': e.episode_id, 'decisions': len(e), 'native_frames': e.native_frames,
                'difficulty': e.native_context['difficulty'],
                'stages_observed': e.native_context.get('stages_observed'),
                'teacher_Bomb_targets': int((e.probabilities.argmax(-1) >= 18).sum())}
                for e in episodes]
        report.update(status='passed', validation_seconds=time.perf_counter() - started,
            sources=dataset.sources, training=summarize(train), validation=summarize(val),
            cumulative_split=cumulative, independent_eval_role='No student evaluation is conducted',
            scope='Existing strict loader accepted complete original-game episodes and exact encoder; no new training, no staged cropping, and no reassignment of historical held-out episodes')
        with (args.output / 'report.json').open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps({'status': 'passed', 'training': report['training'],
            'validation': report['validation']}, ensure_ascii=False), flush=True)
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        with (args.output / 'failure.json').open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        raise


if __name__ == '__main__':
    main()
