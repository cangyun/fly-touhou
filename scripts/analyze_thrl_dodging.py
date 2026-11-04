"""Export existing dodge evidence without loading Torch or starting a game."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fly_touhou.thrl_dodge_diagnostics import DodgeDiagnosticsReader, summarize_rows
from fly_touhou.thrl_study_io import read_shared_json, utc_now


def build_report(run, *, window=30, latest_development=5):
    run = Path(run).resolve()
    errors = []

    def read(path):
        try:
            document = read_shared_json(path)
            if not isinstance(document, dict):
                raise ValueError('Expected JSON object')
            return document
        except (OSError, ValueError) as error:
            errors.append({'path': str(path), 'error': str(error)})
            return {}

    config = read(run / 'config.json')
    progress = read(run / 'progress.json')
    diagnostics = DodgeDiagnosticsReader(run, window_episodes=window).poll(
        progress=progress, protocol_metadata=config)
    # Worker prefixes remain available in the UI. Keep this export compact.
    diagnostics['live'].pop('workers', None)
    metrics = progress.get('metrics') or {}
    health_keys = ('entropy', 'entropy_coeff', 'approx_kl', 'target_kl', 'clip_fraction',
        'grad_norm', 'optimizer_updates', 'samples', 'sample_passes', 'planned_sample_passes',
        'epochs_completed', 'epochs_planned', 'max_post_epoch_kl', 'early_stop_reason',
        'explained_variance_before', 'value_loss', 'deathbomb_auxiliary', 'reward_scale',
        'return_scale', 'advantage_scale', 'objective_weights', 'custom_update_rejections',
        'KL_early_stopping')
    developments = sorted(run.glob('development_*/report.json'))
    evaluations = []
    paths = [('development', path) for path in developments[-latest_development:]]
    # Only reports inside this exact trial: never attribute another candidate's
    # final evaluation to the current trial based on a parent directory name.
    paths += [('final-test', path) for pattern in ('validation*/report.json',
        'final_evaluation*/report.json') for path in sorted(run.glob(pattern))]
    for kind, path in paths:
        report = read(path)
        if not report:
            continue
        rows = report.get('episode_results')
        if not isinstance(rows, list):
            errors.append({'path': str(path), 'error': 'Missing episode_results list'})
            continue
        try:
            summary = summarize_rows(rows,
                scope='frozen-policy-' + kind + '-episodes')
        except ValueError as error:
            errors.append({'path': str(path), 'error': str(error)})
            continue
        expected = config.get('development_episodes' if kind == 'development'
                              else 'validation_episodes_per_policy')
        count = report.get('episodes')
        complete = (type(count) is int and count > 0
            and count == summary['episodes'] == len(rows)
            and not summary['excluded_nonterminal_or_invalid_rows']
            and (expected is None or count == expected)
            and report.get('status') in (None, 'completed', 'complete', 'finished', 'success', 'ok')
            and report.get('complete') is not False and not report.get('error'))
        evaluations.append({'path': str(path), 'kind': kind, 'node': path.parent.name,
            'expected_episodes': expected,
            'reported_episodes': count, 'reported_status': report.get('status'),
            'complete': complete,
            'summary': summary, 'reported_score': report.get('score'),
            'score_definition': report.get('score_definition'),
            'seed_split': report.get('native_seed_split'),
            'formal_test': report.get('formal_test'),
            'selection_action': report.get('selection_action')})
    return {
        'schema': 'th14-read-only-dodge-report-v1', 'observed_utc': utc_now(),
        'run_directory': str(run),
        'source_consistency': 'Files are individually shared-read; not an atomic learner snapshot.',
        'process_liveness_checked': False,
        'protocol': {key: config.get(key) for key in ('branch', 'training_mode', 'input_schema',
            'episode_protocol', 'reward_schema', 'learning_revision', 'teacher_pretraining',
            'deathbomb_auxiliary_coefficient', 'model_config', 'parameter_counts')},
        'progress': {key: progress.get(key) for key in ('updated_utc', 'status', 'stage',
            'actual_decisions', 'candidate_optimized_decisions', 'candidate_target_decisions',
            'rollout_updates', 'adam_minibatch_steps', 'workers', 'episodes', 'wall_seconds',
            'decisions_per_second', 'device', 'latest_rollout_timing')},
        'learning_health': {key: metrics.get(key) for key in health_keys},
        'health_interpretation': 'Update execution, value fitting and exploration diagnostics; '
            'finite loss, completed epochs or lower entropy do not establish better dodging.',
        'training_diagnostics': diagnostics,
        'independent_evaluations': evaluations,
        'development_reports_found': len(developments),
        'development_reports_included_limit': latest_development,
        'errors': errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path, help='Exact trial folder')
    parser.add_argument('--output', type=Path, help='New report path outside the training run')
    parser.add_argument('--window', type=int, default=30)
    parser.add_argument('--latest-development', type=int, default=5)
    args = parser.parse_args()
    if not args.run.is_dir():
        parser.error('--run must be an existing trial folder')
    if not 1 <= args.latest_development <= 50:
        parser.error('--latest-development must be between 1 and 50')
    if args.output and args.output.resolve().is_relative_to(args.run.resolve()):
        parser.error('Export outside the training run; source results are read-only')
    report = build_report(args.run, window=args.window, latest_development=args.latest_development)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids accidentally replacing another file.
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(payload + '\n')
        print(str(args.output.resolve()))
    else:
        print(payload)


if __name__ == '__main__':
    main()
