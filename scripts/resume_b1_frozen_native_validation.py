"""Resume a dead fixed validation, retaining only verified completed episodes.

No training, optimizer or protocol selection. Incomplete original worlds start
from their unchanged original episode seeds; no in-game save state is claimed.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now
from fly_touhou.thrl_durable_records import write_json as write
from fly_touhou.thrl_full_story_recovery import digest, journal_rows, verify_episode_trace


def alive(identity):
    import psutil
    try:
        return abs(psutil.Process(identity['pid']).create_time()-identity['create_time']) < .01
    except psutil.Error:
        return False


def qualify(plan):
    import psutil
    from run_b1_entity_PPO_validation import evaluation_protocol, verify_probe_cohort
    if digest(ROOT / 'source_manifest.json') != plan['source_manifest_sha256']:
        raise ValueError('Recovery source manifest differs')
    for item in read(ROOT / 'source_manifest.json')['files']:
        if digest(ROOT / item['path']) != item['sha256']:
            raise ValueError('Recovery source file differs:' + item['path'])
    old = Path(plan['recovery_source'])
    original = read(plan['original_plan'])
    if digest(plan['original_plan']) != plan['original_plan_sha256']:
        raise ValueError('Original frozen protocol plan changed')
    if digest(original['scientific_protocol']) != original['scientific_protocol_sha256']:
        raise ValueError('Prospective scientific protocol changed')
    if original['expected_native_seeds'] != plan['expected_native_seeds']:
        raise ValueError('Recovery changed the original test cohort')
    if alive(read(old / 'launch.json')):
        raise RuntimeError('Original validation identity is still alive; recovery refused')
    if any((p.info['name'] or '').lower() == 'th14.exe' for p in psutil.process_iter(['name'])):
        raise RuntimeError('Another original-game pool is present')
    training = read(Path(original['training_output']) / 'trial_000_both' / 'training_result.json')
    if (training.get('status') != 'finished'
            or training.get('candidate_optimized_decisions') != original['training_decisions']
            or training.get('final_sha256') != plan['checkpoints']['final']['sha256']):
        raise ValueError('The already finished PPO budget and final identity must be preserved')
    for item in plan['checkpoints'].values():
        if digest(item['path']) != item['sha256']:
            raise ValueError('Frozen original model changed')
    config = read(old / 'evaluation_input' / 'config.json')
    if digest(old / 'evaluation_input' / 'config.json') != plan['input_config_sha256']:
        raise ValueError('Original evaluation configuration changed')
    if (digest(config['game_exe']) != plan['game_exe_sha256']
            or digest(config['profile']) != plan['profile_sha256']):
        raise ValueError('Original executable or native profile changed')
    runtime = Path(config['source_runtime'])
    for name in ('thrl_native_env.py','thrl_full_story_env.py','responsive_controller.py','isolated_desktop_process.py'):
        if digest(ROOT/'src/fly_touhou'/name) != digest(runtime/'src/fly_touhou'/name):
            raise ValueError('Qualified native execution path changed:' + name)
    if digest(ROOT/'native/th14_fake_d3d9.dll') != digest(runtime/'native/th14_fake_d3d9.dll'):
        raise ValueError('Qualified headless graphics DLL changed')
    probe_path = old / 'no_Bomb_fresh100' / 'report.json'
    probe = read(probe_path)
    if digest(probe_path) != plan['no_Bomb_report_sha256']:
        raise ValueError('Completed original noBomb report changed')
    verify_probe_cohort(probe, plan['expected_native_seeds'])
    for label, probe_name in (('initial','initial'),('final','trained')):
        pin = probe['pins'][probe_name]
        if (pin['sha256'] != plan['checkpoints'][label]['sha256']
                or digest(pin['copy']) != pin['sha256']):
            raise ValueError('Completed noBomb comparison model pin differs')
    if (probe.get('status') != 'completed' or probe.get('training_modified') is not False
            or read(probe_path.parent / 'cleanup.json') != dict(closed=True, close_errors=[])):
        raise ValueError('Only the complete closed100-game comparison can be inherited')
    trace_count = 0
    for row in probe['episode_results']:
        pointer = row.get('failure_trace')
        if not pointer:
            continue
        receipt = read(pointer['receipt'])
        if (digest(pointer['arrays']) != pointer['sha256']
                or receipt['sha256'] != pointer['sha256']
                or (receipt['policy'], receipt['game']) != (row['policy'], row['game'])):
            raise ValueError('Completed noBomb diagnostic provenance differs')
        trace_count += 1
    completed, receipts, journals = {}, {}, {}
    from fly_touhou.thrl_full_story_evaluation import remaining_jobs
    import numpy as np
    jobs = []
    for game in range(20):
        e, a = np.random.SeedSequence([original['seed'], game]).generate_state(2)
        jobs.append(dict(game=game, environment_seed=int(e), action_seed=int(a)))
    for label in ('initial', 'final'):
        folder = old / ('full_story_' + label)
        rows, journal = journal_rows(folder / 'episodes.jsonl')
        remaining_jobs(jobs, rows, 'test')
        if any(r['native_seed'] != plan['expected_native_seeds'][r['evaluation_game']-1] for r in rows):
            raise ValueError('Saved full-game cohort differs')
        receipts[label] = [verify_episode_trace(r, folder, plan['checkpoints'][label]['sha256']) for r in rows]
        completed[label], journals[label] = rows, journal
    return original, config, probe, completed, dict(utc=utc_now(), status='passed',
        inherited_noBomb=100, noBomb_trace_files_verified=trace_count,
        inherited_full_games={k:len(v) for k,v in completed.items()},
        full_game_receipts=receipts, journals=journals, original_job_plan=jobs,
        original_process_identity_dead=True, original_worlds=0, optimizer_updates=0)


def run(args):
    import psutil
    plan = read(args.plan)
    original, config, probe, completed, qualification = qualify(plan)
    if args.qualify_only:
        write(args.qualify_only, qualification)
        print(json.dumps(qualification, ensure_ascii=False), flush=True)
        return
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    args.created_output = output
    identity = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), utc=utc_now())
    write(output / 'launch.json', identity)
    write(output / 'qualification.json', qualification)
    write(output / 'study_config.json', config)
    write(output / 'recovery_protocol.json', plan)
    os.environ['FLY_TOUHOU_DESKTOP_NAME'] = 'fly-touhou-job-B1-fixed-recovery-' + str(os.getpid())
    from fly_touhou.thrl_job_paths import configure_native_temp
    configure_native_temp(output)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    def stop():
        if (output / 'STOP').exists():
            raise RuntimeError('Operator STOP; preserve all completed scores')
    def publish(phase, dest=None, detail=None):
        write(output / 'study_status.json', dict(status='finished' if phase=='completed' else 'validation',
            phase=phase, current_run=str(dest or output), updated_utc=utc_now(),
            native_instances_max=30, current_progress=dict(status='validation',stage='validation',
                actual_decisions=original['training_decisions'],target_decisions=original['training_decisions'],
                workers=30,device='cuda:0',evaluation_progress=detail,optimizer_updates=0)))
    write(output / 'initial_reference.json', read(Path(plan['recovery_source']) / 'initial_reference.json'))
    inherited_probe = output / 'no_Bomb_fresh100'
    inherited_probe.mkdir()
    write(inherited_probe / 'report.json', probe)
    write(inherited_probe / 'cleanup.json', dict(closed=True, close_errors=[]))
    write(inherited_probe / 'reused_results.json', dict(source=str(Path(plan['recovery_source']) / 'no_Bomb_fresh100'),
        original_report_sha256=plan['no_Bomb_report_sha256'],inherited_completed_games=100,
        newly_executed_games=0,original_trace_paths_retained=True))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no inference CPU fallback')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_full_story_evaluation import evaluate
    from fly_touhou.thrl_training_mode import episode_protocol
    from train_thrl_native import make_env_kwargs, development_summary
    from run_b1_entity_PPO_validation import full_story_configuration, evaluation_protocol
    cfg = full_story_configuration(config, original)
    try:
        from register_b1_goal_phase import register
        write(output / 'panel_registration.json', register(output,plan['panel_job_id'],'evaluation',
            control_root=Path(original['control_center_root'])))
    except Exception as error:
        write(output / 'panel_registration_error.json', dict(error=repr(error),native_actions_affected=False))
    reports = {}
    for label in ('initial', 'final'):
        stop()
        phase, dest = 'full_story_'+label, output / ('full_story_'+label)
        publish(phase, dest)
        item = plan['checkpoints'][label]
        payload = torch.load(item['path'], map_location='cpu', weights_only=False)
        if (payload.get('numerical_qualification_only') or payload.get('deployment_forbidden')
                or payload.get('optimizer_transaction_complete') is False
                or payload['model_config'].get('input_schema') != config['input_schema']):
            raise ValueError('Only the complete frozen production model may enter original-game evaluation')
        model = model_from_config(**payload['model_config']).cuda()
        model.load_state_dict(payload['model'], strict=True)
        if any(not torch.equal(v.detach().cpu(),payload['model'][k]) for k,v in model.state_dict().items()):
            raise ValueError('Actual resumed evaluation model weights differ')
        del payload
        workers = min(30, 20-len(completed[label]))
        reports[label] = evaluate(torch,model,make_env_kwargs(cfg,dest,workers,'test'),dest,
            episodes=20,seed=original['seed'],check_stop=stop,summary=development_summary,
            progress=lambda row,p=phase,d=dest:publish(p,d,row),native_split='test',
            episode_protocol=episode_protocol(cfg),action_selection=original['full_story_action_selection'],
            trace_config=dict(checkpoint_sha256=item['sha256'],history=128),
            precompleted_rows=completed[label])
        if any(r['native_seed'] != plan['expected_native_seeds'][r['evaluation_game']-1]
               for r in reports[label]['episode_results']):
            raise ValueError('Executed recovery games changed the fixed native cohort')
        del model
        torch.cuda.empty_cache()
    old = {r['native_seed']:r for r in reports['initial']['episode_results']}
    new = {r['native_seed']:r for r in reports['final']['episode_results']}
    if set(old) != set(new) or set(old) != set(plan['expected_native_seeds']):
        raise ValueError('Completed full-game cohorts do not match')
    pairs = []
    for seed,x in old.items():
        y = new[seed]
        if x['action_seed'] != y['action_seed']:
            raise ValueError('Original per-episode inference seeds differ')
        pairs.append(dict(native_seed=seed,initial_frames=x['native_frames'],final_frames=y['native_frames'],
            initial_stage=x['stage'],final_stage=y['stage'],initial_clear=x['outcome']=='clear',final_clear=y['outcome']=='clear'))
    summary = {label:{k:r[k] for k in ('episodes','clears','mean_native_frames','mean_max_stage','mean_misses','mean_bombs_used')}
               for label,r in reports.items()}
    write(output / 'report.json',dict(status='completed',utc=utc_now(),schema='B1-entity-PPO-frozen-native-comparison-v1',
        no_Bomb_report=str(Path(plan['recovery_source'])/'no_Bomb_fresh100'/'report.json'),
        inherited_noBomb_display_report=str(inherited_probe/'report.json'),full_story=summary,paired_full_games=pairs,
        comparison_scope='whole-full-teacher-BC-and-shared-policy-PPO-pipeline',
        initial_baseline_semantics=config['initial_baseline_semantics'],
        native_seed=original['seed'],initial_RNG_matched=True,same_future_hazards_guaranteed=False,
        native_games=140,unique_completed_native_games=140,new_optimizer_updates=0,formal_six_stage_acceptance=False,
        inherited_completed_native_games=100+sum(len(v) for v in completed.values()),
        newly_completed_native_games=sum(r['newly_completed_episodes'] for r in reports.values()),
        interrupted_prior_sampling_decisions_lower_bound=plan['prior_sampling_decisions_lower_bound'],
        newly_executed_decisions=sum(r['newly_executed_decisions'] for r in reports.values()),
        paid_sampling_cost_lower_bound=plan['prior_sampling_decisions_lower_bound']+sum(r['newly_executed_decisions'] for r in reports.values()),
        unrecoverable_game_prefixes_restarted_from_same_seed=True,
        evaluation_protocol=evaluation_protocol(original),
        full_story_trace_capture={label:r.get('full_game_trace_capture') for label,r in reports.items()}))
    publish('completed')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--qualify-only',type=Path)
    a = parser.parse_args()
    try:
        run(a)
    except BaseException:
        if getattr(a,'created_output',None):
            write(a.created_output/'failure.json',dict(utc=utc_now(),traceback=traceback.format_exc(),
                new_optimizer_updates=0,no_automatic_restart=True))
        raise
