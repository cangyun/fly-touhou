"""Restore actual saved native worlds without training or selecting new actions."""
from pathlib import Path
import argparse
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worlds', type=int, nargs='+')
    args = parser.parse_args()
    import torch
    from fly_touhou import thrl_vector
    from fly_touhou.compact_full_story import factory, story_kwargs
    from fly_touhou.nbnm_run import read, runtime, environment
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    args.output.mkdir(parents=True, exist_ok=False)
    temp = runtime(args.output)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    capsule = saved['native_continuation']
    decisions, updates = saved['completed_decisions'], saved['optimizer_updates']
    del saved
    spec = read(args.config)
    indices = args.worlds or list(range(len(capsule['worlds'])))
    kwargs = story_kwargs(environment(spec), args.output / 'native', spec['workers'], 'train', teacher=True)
    selected = [capsule['worlds'][i] for i in indices]
    del capsule
    vector = None
    started = time.monotonic()
    try:
        write_json(args.output / 'progress.json', dict(phase='restoring_saved_native_worlds', worlds=indices))
        thrl_vector.worker_factory = factory
        vector = thrl_vector.THRLNativeVector([kwargs[i] for i in indices], args.output / 'workers', timeout=180)
        _, infos = vector.reset(options=[dict(continuation=w) for w in selected])
        report = dict(status='matched', checkpoint=str(args.checkpoint), decisions=decisions,
            optimizer_updates=updates, new_optimizer_updates=0, new_PPO_decisions=0,
            worlds=[dict(index=i, task=w['task'], replayed_actions=len(w['actions']),
                result=info['native_continuation']) for i, w, info in zip(indices, selected, infos)],
            seconds=time.monotonic()-started)
        write_json(args.output / 'report.json', report)
        write_json(args.output / 'progress.json', dict(phase='complete', worlds=indices))
    except BaseException:
        write_json(args.output / 'failure.json', dict(error=traceback.format_exc()))
        raise
    finally:
        if vector is not None:
            vector.close()
        cleanup_native_temp(temp)


if __name__ == '__main__':
    main()
