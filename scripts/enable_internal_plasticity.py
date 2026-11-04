"""Initialize positive bounded internal gains from an audited distilled brain."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
from fly_touhou.connectome import load_connectome, file_sha256
from fly_touhou.whole_brain_learning import load_whole_brain_checkpoint
from fly_touhou.internal_plasticity import enable_bounded_recurrent_gain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=float, default=.1)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('Preserve existing migration artifacts')
    torch.set_num_threads(4)
    digest = file_sha256(args.parent)
    graph = load_connectome(args.source_dir, max_neurons=None)
    policy = load_whole_brain_checkpoint(args.parent, graph, 'cpu')
    parent = torch.load(args.parent, weights_only=True, map_location='cpu')
    if file_sha256(args.parent) != digest:
        raise RuntimeError('Parent changed while loading')
    result = enable_bounded_recurrent_gain(parent, limit=args.limit, parent_sha256=digest)
    result['internal_plasticity_migration']['parent_path'] = str(args.parent.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        torch.save(result, stream)
    restored = load_whole_brain_checkpoint(args.output, graph, 'cpu')
    for name, value in policy.student.named_parameters():
        if not torch.equal(value, dict(restored.student.named_parameters())[name]):
            raise RuntimeError('Existing actor tensor changed during migration')
    # Same real graph, states and counter-based noise. Adding zero gains must
    # reproduce actual forward spikes and logits exactly before any training.
    observation = torch.zeros(policy.student.obs_dim)
    with torch.no_grad():
        first_logits, first = policy.student(observation)
        second_logits, second = restored.student(observation)
        torch.testing.assert_close(first_logits, second_logits, rtol=0, atol=0)
        for field in ('membrane', 'synaptic', 'spikes', 'rate', 'noise_counter', 'window_spike_counts'):
            torch.testing.assert_close(getattr(first, field), getattr(second, field), rtol=0, atol=0)
        for first_delay, second_delay in zip(first.delay_queue, second.delay_queue):
            torch.testing.assert_close(first_delay, second_delay, rtol=0, atol=0)
    report = {**result['internal_plasticity_migration'], 'output': str(args.output.resolve()),
        'output_sha256': file_sha256(args.output), 'real_full_brain_identity_forward_verified': True,
        'source_code_sha256': file_sha256(Path(__file__)),
        'plasticity_code_sha256': file_sha256(ROOT/'src/fly_touhou/internal_plasticity.py'),
        'plasticity': restored.student.recurrent_plasticity_specification(),
        'initial_gains': restored.student.recurrent_plasticity_summary()}
    with args.output.with_suffix('.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
