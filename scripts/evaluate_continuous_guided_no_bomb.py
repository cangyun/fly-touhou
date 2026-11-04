"""Frozen native NoBomb screen for calibrated full-connectome DAgger models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('source', 'connectome', 'environment-config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--seed', type=int, default=2027432905)
    p.add_argument('--movement-readout',choices=('composed','raw','risk_only'),default='composed')
    p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--opening-activity-rows',type=int,default=0)
    p.add_argument('--history-reconstruction-probe',action='store_true')
    p.add_argument('--max-seconds', type=int, default=5400)
    a = p.parse_args()
    import psutil
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.owned_worker_affinity import worker_cpu_pool
    from fly_touhou.thrl_job_paths import configure_native_temp
    a.output.mkdir(parents=True, exist_ok=False)
    current = psutil.Process()
    cpus = worker_cpu_pool(current.cpu_affinity())
    current.cpu_affinity(current.cpu_affinity()[-2:])
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-guided-screen-' + str(current.pid))
    configure_native_temp(a.output)
    environment = read_shared_json(a.environment_config)
    exe = Path(environment['game_exe']).name.lower()
    if any((v.info['name'] or '').lower() == exe for v in psutil.process_iter(['name'])):
        raise RuntimeError('Original-game job still alive; do not overlap the frozen screen')
    start = time.monotonic()
    launch = dict(PID=current.pid, create_time=current.create_time(), status='running',
        started_UTC=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
        source=str(a.source), source_sha256=a.source_sha256,
        frozen=True, optimizer_updates=0, teacher_actions=0, episodes=a.episodes, native_workers=a.episodes,
        action_selection='greedy', Bomb_forbidden=True, native_frame_cap=None,
        movement_readout=a.movement_readout,
        app_goal_remains_active=True, native_started=False)
    write_json(a.output / 'launch.json', launch)

    def guard():
        if (a.output / 'STOP').exists():
            raise RuntimeError('Operator STOP')
        if time.monotonic() - start > a.max_seconds:
            raise RuntimeError('Frozen screen wall budget reached')

    try:
        import torch
        from fly_touhou.thrl_dodge_diagnostics import _shared_binary
        from fly_touhou.continuous_guided_dagger import sha, build_guided_model
        from fly_touhou.continuous_no_bomb_first_contact import make_no_bomb_policy, evaluate_no_bomb
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if sha(a.source) != a.source_sha256:
            raise ValueError('Frozen source SHA differs')
        with _shared_binary(a.source) as stream:
            payload = torch.load(stream, map_location='cpu', weights_only=True)
        launch['source_counters'] = payload['counters']
        write_json(a.output / 'progress.json', dict(phase='loading_frozen_full_connectome',
            costs=payload['counters'], app_goal_remains_paused=False))
        model = build_guided_model(payload, a.connectome, device='cuda:0').eval().requires_grad_(False)
        del payload
        policy = make_no_bomb_policy(torch, model, family='deep', episodes=a.episodes,movement_readout=a.movement_readout)
        launch.update(native_started=True, graph_neurons=model.core.n_neurons, graph_edges=model.core.n_edges)
        write_json(a.output / 'launch.json', launch)

        def progress(row):
            write_json(a.output / 'progress.json', dict(phase='frozen_NoBomb_evaluation',
                evaluation=row, costs=launch['source_counters'], app_goal_remains_paused=False))

        report = evaluate_no_bomb(torch, policy, environment, a.output / 'development20',
            checkpoint_sha256=a.source_sha256, episodes=a.episodes, workers=a.episodes, seed=a.seed,
            check_stop=guard, progress=progress, worker_cpus=cpus,opening_activity_rows=a.opening_activity_rows)
        if a.history_reconstruction_probe:
            from fly_touhou.continuous_history_reconstruction import audit_history_reconstruction
            write_json(a.output/'progress.json',dict(phase='offline_history_reconstruction',costs=launch['source_counters']))
            pointer=report['episode_results'][0]['first_contact_tail']['opening_activity_trace']
            audit=audit_history_reconstruction(model,pointer['path'],check_stop=guard)
            audit.update(source_SHA256=a.source_sha256,native_report=str(a.output/'development20/report.json'))
            write_json(a.output/'history-reconstruction.json',audit)
        write_json(a.output / 'result.json', dict(status='completed', report=report,
            costs=launch['source_counters'], app_goal_remains_active=True,
            elapsed_seconds=time.monotonic() - start,
            basic_gate_candidate=report['first_stage_without_observed_contact_count'] >= 8,
            independent_confirmation_still_required=True))
        launch['status'] = 'completed'
        write_json(a.output / 'launch.json', launch)
        write_json(a.output / 'progress.json', dict(phase='frozen_evaluation_completed',
            completed=a.episodes, costs=launch['source_counters'], app_goal_remains_paused=False,
            first_stage_clean=report['first_stage_without_observed_contact_count']))
    except BaseException as error:
        write_json(a.output / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic() - start, app_goal_remains_active=True))
        raise


if __name__ == '__main__':
    main()
