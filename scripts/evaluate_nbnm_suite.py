"""Evaluate frozen NBNM checkpoints on matched native tasks, without training."""
from pathlib import Path
import argparse
import copy
import gc
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--tasks', nargs='+', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--outcomes-only', action='store_true',
                        help='Measure original-game outcomes without querying the teacher or writing per-decision traces')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    from fly_touhou.nbnm_run import read, runtime, load_model, evaluate
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    from fly_touhou.thrl_training_mode import seed_split

    if not 1 <= args.workers <= 100:
        raise ValueError('Use 1..100 evaluation workers')
    if len(set(args.seeds)) != len(args.seeds) or any(seed_split(s) != 'development' for s in args.seeds):
        raise ValueError('Use distinct development seeds for this diagnostic suite')
    spec = copy.deepcopy(read(args.config))
    for key in args.tasks:
        if key not in spec['tasks']:
            raise ValueError('Unknown task: ' + key)
        entry = spec['task_entries'][key]
        if entry.get('entry_mode') not in ('native_spell_practice','native_stage_practice'):
            raise ValueError('This suite uses native spell or complete stage practice entries')
        entry.setdefault('practice_evaluation_seeds',{})['development'] = list(args.seeds)
    spec['evaluation_workers'] = args.workers
    # Labels and scores are observed for diagnosis; only the frozen student acts.
    spec['evaluation_diagnostics'] = False if args.outcomes_only else 'teacher'
    spec['nbnm'].setdefault('practice_evaluation_seeds',{})['development'] = list(args.seeds)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'configuration.json', spec)
    entry_modes={key:spec['task_entries'][key]['entry_mode'] for key in args.tasks}
    modes=set(entry_modes.values())
    plan = dict(schema='frozen-nbnm-spell-suite-v1', source_config=str(args.config),
                checkpoints=list(map(str, args.checkpoints)), tasks=args.tasks,
                native_seeds=args.seeds, workers=args.workers,
                target_attempts=len(args.checkpoints) * len(args.tasks) * len(args.seeds),
                action_selection='joint_mode', network_training=False,
                entry_mode=next(iter(modes)) if len(modes)==1 else 'mixed_native_tasks',
                entry_modes=entry_modes,
                recurrent_state='native task entry', outcomes_only=args.outcomes_only,
                difficulty='Lunatic', Bombs_disabled=True,
                scope='Matched original-game development spell and stage starts')
    write_json(args.output / 'plan.json', plan)
    temporary = runtime(args.output)
    import torch
    import psutil

    proc = psutil.Process()
    proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    started = time.monotonic()
    completed = []
    active = {}

    def progress(row):
        write_json(args.output / 'progress.json', dict(
            row, active=active, completed_tasks=len(completed),
            completed_attempts=sum(x['attempts'] for x in completed) + row.get('completed', 0),
            completed_successes=sum(x['successes'] for x in completed) + row.get('successes', 0),
            target_attempts=plan['target_attempts'], network_training=False,
            PID=proc.pid, create_time=proc.create_time(), elapsed_seconds=time.monotonic()-started))

    try:
        for checkpoint in args.checkpoints:
            label = checkpoint.stem.replace('-parameters', '')
            active = dict(checkpoint=str(checkpoint), label=label)
            progress(dict(phase='loading_frozen_checkpoint'))
            torch.manual_seed(spec['seed'])
            torch.cuda.manual_seed_all(spec['seed'])
            model, _ = load_model(spec, checkpoint)
            model.requires_grad_(False)
            for key in args.tasks:
                active = dict(checkpoint=str(checkpoint), label=label, task=key)
                destination = args.output / label / key
                destination.parent.mkdir(parents=True, exist_ok=True)
                progress(dict(phase='starting_native_spell'))
                report = evaluate(model, spec, destination, task=spec['tasks'][key], progress=progress)
                report['checkpoint'] = str(checkpoint)
                report['task_key'] = key
                write_json(destination / 'report.json', report)
                completed.append(dict(label=label, checkpoint=str(checkpoint), task=key,
                    attempts=report['attempts'], successes=report['successes'],
                    spell_captures=report['spell_captures'],
                    spell_survived_timeouts=report['spell_survived_timeouts'],
                    successful_mean_native_frames=report['successful_mean_native_frames'],
                    report=str(destination / 'report.json')))
                write_json(args.output / 'results.json', dict(rows=completed, complete=False,
                    network_training=False, target_attempts=plan['target_attempts']))
                progress(dict(phase='task_complete'))
            del model
            gc.collect()
            torch.cuda.empty_cache()
        write_json(args.output / 'results.json', dict(rows=completed, complete=True,
            network_training=False, target_attempts=plan['target_attempts']))
        active = {}
        progress(dict(phase='complete'))
    except BaseException:
        write_json(args.output / 'failure.json', dict(active=active, error=traceback.format_exc()))
        progress(dict(phase='failed'))
        raise
    finally:
        cleanup_native_temp(temporary)


if __name__ == '__main__':
    main()
