"""Exact numerical recovery of an interrupted, already initialized B1 PPO run.

This entry never repeats BC or resets Adam. The qualified native collector,
network, reward and PPO implementation are inherited unchanged in the snapshot.
"""
from pathlib import Path
import argparse
from dataclasses import asdict
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now


def validate_recovery(cfg):
    resume = cfg.get('baseline_resume') or {}
    if (cfg.get('branch') != 'B1' or cfg.get('workers') != 30
            or cfg.get('training_mode') != 'full-story-v1'
            or not cfg.get('headless') or cfg.get('native_frame_horizon') is not None
            or not cfg.get('finite_native_lives') or cfg.get('continues_allowed')
            or cfg.get('policy_action_count') != 19
            or cfg.get('reset_Adam_after_BC') is not False
            or not cfg.get('full_teacher_actor_transfer')
            or cfg.get('evaluation_after_training_shutdown') is not True
            or cfg.get('initial_avoidance_development')
            or cfg.get('ready_shared_observations', False)
            or not resume.get('checkpoint') or not resume.get('source_run')
            or resume.get('optimized_decisions', 0) <= 0):
        raise ValueError('Require exact B1 PPO recovery, same30 worlds and complete Story protocol; never BC/fresh Adam')
    return resume


def run(args):
    import psutil
    import torch
    from train_thrl_branch import verify_sources
    from train_thrl_native import train_trial, StopRequested, validate_evaluation_schedule
    from fly_touhou.thrl_branching import candidate_trial_config
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_job_paths import configure_native_temp
    from fly_touhou.thrl_ppo import PPOConfig
    from fly_touhou.th14_completion_protocol import learning_overrides

    cfg = read(args.config)
    validate_recovery(cfg)
    trial_cfg = candidate_trial_config(cfg, 0)
    if not trial_cfg['candidate_resume_applied']:
        raise ValueError('No numerical recovery was applied')
    validate_evaluation_schedule(cfg)
    verify_sources(trial_cfg)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    for identity in cfg['reboot_recovery']['previous_process_identities']:
        try:
            process = psutil.Process(identity['pid'])
            if abs(process.create_time() - identity['create_time']) < .01 and process.is_running():
                raise RuntimeError('Previous job is still running')
        except psutil.NoSuchProcess:
            pass
    if any((p.info['name'] or '').lower() == 'th14.exe' for p in psutil.process_iter(['name'])):
        raise RuntimeError('Prior original worlds still exist')

    args.output.mkdir(parents=True, exist_ok=False)
    args.output_created = True
    write(args.output / 'launch.json', dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
        utc=utc_now(), CUDA_required=True, native_worlds_max=30, desktop_input=False))
    configure_native_temp(args.output)
    os.environ['FLY_TOUHOU_DESKTOP_NAME'] = 'fly-touhou-job-B1-PPO-recovery-' + str(os.getpid())
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(cfg.get('torch_threads', 2))
    parameters = {**asdict(PPOConfig()), **cfg['parameters'], **learning_overrides(cfg)}
    write(args.output / 'study_config.json', cfg)
    write(args.output / 'effective_study_config.json', cfg)
    if cfg.get('goal_panel_registration'):
        from register_b1_goal_phase import register
        panel = cfg['goal_panel_registration']
        write(args.output / 'panel_registration.json', register(args.output, panel['job_id'],
            'training', control_root=Path(panel['control_root'])))
    started = time.monotonic()

    def progress(row):
        write(args.output / 'study_status.json', dict(status=row.get('status', 'training'),
            current_run=str(args.output / 'trial_000_both'), current_progress=row,
            updated_utc=utc_now(), wall_seconds=time.monotonic() - started,
            planned_trials=1, device='cuda:0', gpu=torch.cuda.get_device_name(0),
            optimizer_initialization='Exact existing PPO Adam and four sampling RNG states restored; no repeated BC'))

    try:
        result = train_trial(torch, trial_cfg, args.output, args.output / 'trial_000_both',
            parameters, trial=None, study_started=started, on_progress=progress,
            attention='both', model_factory=model_from_config)
        write(args.output / 'study_result.json', dict(status='finished', training=result,
            formal_acceptance=False, BC_repeated=False, PPO_optimizer_fresh=False,
            exact_PPO_recovery=True, full_shared_policy_optimized=True))
    except StopRequested as error:
        write(args.output / 'study_result.json', dict(status='operator_or_budget_stop', reason=str(error)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException:
        if getattr(args, 'output_created', False):
            write(args.output / 'failure.json', dict(utc=utc_now(), traceback=traceback.format_exc(),
                no_automatic_restart=True))
        raise
