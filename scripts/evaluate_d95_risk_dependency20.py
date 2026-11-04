"""Frozen D95 risk dependence, using the original15/20 evaluator.

Default is NoTorch inspection. Only --execute-cuda plus an explicit wall budget
can start native games. This file does not train, resume, or edit a student.
Explicit historical-beta8 reuse runs only beta0; the dual-arm plan stays valid.
"""
from pathlib import Path
import argparse
import copy
import datetime
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
import traceback

SCHEMA = 'D95-risk-dependency-original20-sequential-v1'
D95_SHA = 'd95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'
RUNTIME_SHA = '7678b3db16d0f54d0a1d0a7f765d8a253a73e00f7e0b5a483e1f8fe086c30a37'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def alive(identity):
    import psutil
    try:
        p = psutil.Process(identity['pid'])
        return p.is_running() and abs(p.create_time()-identity['create_time']) < .01
    except psutil.NoSuchProcess:
        return False


def originals():
    import psutil
    return [dict(pid=p.pid, create_time=p.create_time()) for p in psutil.process_iter(['name'])
            if (p.info['name'] or '').lower() == 'th14.exe']


def inspect(plan):
    """Reuse the existing reference entry's generic frozen-runtime verifier."""
    if Path(plan['entry']).resolve() != Path(__file__).resolve() or sha(__file__) != plan['entry_sha256']:
        raise ValueError('New entry differs from its declared source pin')
    reuse = plan.get('reuse_historical_beta8', False)
    expected_arms = ([{'name':'beta0','beta':0.}] if reuse is True
                     else [{'name':'beta8','beta':8.}, {'name':'beta0','beta':0.}])
    if (type(reuse) is not bool or plan.get('schema') != SCHEMA or plan.get('checkpoint_sha256') != D95_SHA
            or plan.get('source_manifest_sha256') != RUNTIME_SHA or plan.get('runtime_files') != 486
            or plan.get('seed') != 383021 or plan.get('workers') != 14 or plan.get('episodes_each') != 20
            or plan.get('arms') != expected_arms
            or plan.get('action_selection') != 'greedy' or plan.get('action_temperature') != 1.
            or plan.get('native_split') != 'development' or plan.get('continues') != 0
            or plan.get('native_frame_horizon', 'missing') is not None
            or type(plan.get('capture_failure_traces')) is not bool):
        raise ValueError('Require original D95 natural20/seed383021/14 workers;dual8,0 or explicit historical8 plus fresh0')
    helper_path = Path(plan['runtime_verifier'])
    if sha(helper_path) != plan['runtime_verifier_sha256']:
        raise ValueError('Existing NoTorch runtime helper changed')
    spec = importlib.util.spec_from_file_location('_sealed_D95_reference_tools', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    runtime, count = helper.verify_runtime(plan)
    for key in ('checkpoint','environment_config','original15_plan','original15_report','original15_episode_plan'):
        helper.pin(plan[key], plan[key+'_sha256'])
    config, old, episodes = (read(plan[k]) for k in ('environment_config','original15_report','original15_episode_plan'))
    prior = read(plan['original15_plan'])
    if (plan['original15_report_sha256'] != 'a287e30ca2081fa93abfd76b7ff5f11422cefe6585cfce511354613aae63a56c'
            or old.get('episodes') != 20 or old.get('clears') != 15 or old.get('checkpoint_sha256') != D95_SHA
            or old.get('policy_frozen') is not True or old.get('action_selection') != 'greedy'
            or old.get('native_seed_split') != 'development' or old.get('evaluation_workers') != 14
            or prior['seed'] != 383021 or prior['expected_native_seeds'] != plan['expected_native_seeds']
            or episodes.get('RNG_protocol') != 'episode-seeded-argmax-v1' or episodes.get('workers') != 14):
        raise ValueError('The sealed15/20 cohort is not the declared original reference')
    wrapper = config.get('model_config',{}).get('pair_risk_wrapper',{})
    if (config.get('input_schema') != 'semantic-separated-observed-laser-segments-v5'
            or config.get('headless') is not True or config.get('training_mode') != 'full-story-v1'
            or config.get('policy_action_count') != 19 or config.get('autonomous_bomb') is not True
            or config.get('continues_allowed') is not False or config.get('reward_schema') != 'th14-completion-reward-v3'
            or wrapper.get('beta') != 8. or wrapper.get('aggregation') != 'worst-known-margin-v2'
            or wrapper.get('laser_width') != 128):
        raise ValueError('Keep the original V5/source19/reward3/full-story environment and combined risk operator')
    helper.pin(config['game_exe'], plan['game_exe_sha256'])
    helper.pin(config['profile'], plan['profile_sha256'])
    helper.install_runtime(runtime)
    import numpy as np
    training_mode = importlib.import_module('fly_touhou.thrl_training_mode')
    helper.require_frozen_module(training_mode,runtime)
    jobs=[];native=[]
    for game in range(20):
        e,a=map(int,np.random.SeedSequence([383021,game]).generate_state(2))
        jobs.append(dict(game=game,environment_seed=e,action_seed=a))
        native.append(training_mode.draw_native_seed(np.random.default_rng(e),'development'))
    if jobs != episodes['jobs'] or jobs != plan['episode_jobs'] or native != plan['expected_native_seeds']:
        raise ValueError('Original environment/action/native seeds do not match')
    rows=old.get('episode_results',[])
    if len(rows)!=20 or any((r.get('evaluation_game'),r.get('evaluation_seed'),r.get('action_seed'),r.get('native_seed'))
            != (i+1,j['environment_seed'],j['action_seed'],native[i]) for i,(r,j) in enumerate(zip(rows,jobs))):
        raise ValueError('Historical beta8 actual completed games do not match the sealed20 cohort')
    if any(n == 'torch' or n.startswith('torch.') for n in sys.modules):
        raise RuntimeError('Default inspection must remain NoTorch')
    return helper,config,dict(schema=SCHEMA,status='prepared-NoTorch-inspect-passed',utc=now(),
        source_runtime=str(runtime),verified_runtime_files=count,checkpoint_sha256=D95_SHA,
        original15_report_sha256=plan['original15_report_sha256'],native_seeds=native,
        native_worlds_started=0,optimizer_updates=0,Torch_imported=False,checkpoint_deserialized=False,
        planned_new_native_games=20*len(expected_arms),historical_beta8_reused=reuse,
        historical_beta8_report=plan['original15_report'] if reuse else None,
        historical_beta8_report_sha256=plan['original15_report_sha256'] if reuse else None,
        currently_blocked_by_predecessor=alive(plan['predecessor']),current_original_worlds=len(originals()),
        execution_requires='explicit --execute-cuda --max-seconds; predecessor exact identity exited and training-cleanup closed; no original processes')


def execute(plan, helper, config, max_seconds):
    import psutil
    if alive(plan['predecessor']):
        raise RuntimeError('Exact student predecessor still running; leave it untouched')
    closed=read(plan['predecessor_cleanup'])
    if closed.get('training_closed') is not True or closed.get('close_errors') != []:
        raise RuntimeError('Student native cleanup is not verified closed')
    if originals():
        raise RuntimeError('Another original pool exists; no overlap or takeover')
    output=Path(plan['output'])
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    identity=dict(pid=os.getpid(),create_time=psutil.Process().create_time())
    write(output/'launch.json',dict(schema=SCHEMA,utc=now(),**identity,entry_sha256=sha(__file__),
        checkpoint_sha256=D95_SHA,workers_max=14,total_world_limit=30,arms_sequential=True,
        optimizer_updates=0,maximum_seconds=max_seconds,artificial_memory_caps=False,
        CUDA_memory_fraction=None,student_checkpoint_read_or_written=False,
        executed_arms=[a['name'] for a in plan['arms']],historical_beta8_reused=plan.get('reuse_historical_beta8',False)))
    write(output/'plan.json',plan)
    results={};arm_receipts=[];failure=None
    if plan.get('reuse_historical_beta8',False):
        results['beta8']=read(plan['original15_report'])
        write(output/'historical-beta8-reused.json',dict(
            report=plan['original15_report'],report_sha256=plan['original15_report_sha256'],
            checkpoint_sha256=D95_SHA,source_manifest_sha256=RUNTIME_SHA,seed=383021,
            workers=14,episodes=20,clears=15,newly_executed_native_games=0,
            historical_beta8_reused=True,contemporaneous_fresh_reproduction=False))
    def stop():
        if (output/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-started > max_seconds:raise TimeoutError('Explicit total wall budget reached; unfinished games are not scores')
    try:
        import torch
        factory=importlib.import_module('fly_touhou.thrl_pair_policy')
        evaluator=importlib.import_module('fly_touhou.thrl_full_story_evaluation')
        native=importlib.import_module('train_thrl_native')
        comparison=importlib.import_module('evaluate_b1_full_story_comparison')
        paths=importlib.import_module('fly_touhou.thrl_job_paths')
        for module in (factory,evaluator,native,comparison,paths):helper.require_frozen_module(module,Path(plan['source_runtime']).resolve())
        if not torch.cuda.is_available():raise RuntimeError('Explicit CUDA evaluation required; no fallback')
        torch.set_num_threads(2);torch.cuda.init()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        payload=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)
        if (payload.get('schema') != 'B1-risk-calibrated-frozen-candidate-v1'
                or payload.get('model_config') != config['model_config']):
            raise ValueError('The actual D95 checkpoint schema/config differs')
        model=factory.model_from_config(**payload['model_config']).cuda()
        model.load_state_dict(payload['model'],strict=True);model.requires_grad_(False);model.eval()
        original_model_sha=comparison.model_digest(payload['model'])
        if comparison.model_digest(model.state_dict()) != original_model_sha:raise ValueError('Loaded weights differ')
        paths.configure_native_temp(output)
        for arm in plan['arms']:
            stop()
            if originals():raise RuntimeError('Previous pool not fully gone; next arm not started')
            role=arm['name'];model.beta=float(arm['beta'])
            destination=output/role;destination.mkdir()
            os.environ['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-D95-risk-'+role+'-'+str(os.getpid())
            kwargs=native.make_env_kwargs(dict(config,workers=14,PPO_training_active=False,
                evaluation_checkpoint_sha256=D95_SHA),destination,14,'development')
            if any(k.get('policy_action_count') != 19 or k.get('frame_skip') != 4
                    or k.get('wall_time_limit_seconds') is not None or 'online_teacher_revision' in k
                    or not k.get('headless') or not k.get('hide_window') for k in kwargs):
                raise ValueError('Keep natural private19-action games and no teacher or frame horizon')
            arm_start=time.monotonic()
            write(destination/'launch.json',dict(**identity,utc=now(),arm=role,beta=model.beta,
                model_state_sha256=original_model_sha,policy_identity_sha256=comparison.policy_identity(model.state_dict(),model.model_config(),'greedy'),
                exact_same_frozen_state_tensors=True,hidden='own zero-at-new-game GRU;stage/Miss carry;no cross-arm hidden copying'))
            def progress(row):
                write(output/'progress.json',dict(schema=SCHEMA,status='evaluating',arm=role,beta=model.beta,
                    utc=now(),elapsed_seconds=time.monotonic()-started,evaluation_progress=row,
                    completed_new_arms=[r['arm'] for r in arm_receipts],available_reference_arms=list(results),
                    historical_beta8_reused=plan.get('reuse_historical_beta8',False),optimizer_updates=0))
            trace=dict(checkpoint_sha256=D95_SHA,history=128) if plan['capture_failure_traces'] else None
            report=evaluator.evaluate(torch,model,kwargs,destination/'games',episodes=20,seed=383021,
                check_stop=stop,progress=progress,summary=native.development_summary,native_split='development',
                episode_protocol=importlib.import_module('fly_touhou.thrl_training_mode').episode_protocol(config),
                action_selection='greedy',trace_config=trace)
            executed=read(destination/'games/episode_plan.json')
            if ([r['native_seed'] for r in report['episode_results']] != plan['expected_native_seeds']
                    or executed['jobs'] != plan['episode_jobs'] or executed['workers'] != 14
                    or executed['RNG_protocol'] != 'episode-seeded-argmax-v1'):
                raise ValueError('Executed cohort differs from the original20')
            cleanup=read(destination/'games/native_cleanup.json')
            if cleanup.get('close_errors') != [] or originals():
                raise RuntimeError('Current arm did not release every original before the next arm')
            if comparison.model_digest(model.state_dict()) != original_model_sha or sha(plan['checkpoint']) != D95_SHA:
                raise RuntimeError('Frozen weights/file changed')
            report.update(risk_dependency_arm=role,risk_beta=model.beta,checkpoint_sha256=D95_SHA,
                model_state_sha256=original_model_sha,optimizer_updates=0,
                hidden_same_weights_not_same_trajectory=True,risk0_is_posttraining_operator_ablation=True)
            write(destination/'report.json',report);results[role]=report
            receipt=dict(arm=role,beta=model.beta,**identity,wall_seconds=time.monotonic()-arm_start,
                native_games=20,report=str(destination/'report.json'),report_sha256=sha(destination/'report.json'),
                native_cleanup=cleanup,originals_remaining=[],next_arm_waits_for_verified_cleanup=True)
            arm_receipts.append(receipt);write(output/'arm-receipts.json',arm_receipts)
        summary=comparison.paired_summary(results,20,'beta8')
        write(output/'report.json',dict(schema=SCHEMA,status='completed',utc=now(),summary=summary,
            arms=arm_receipts,native_games=20*len(arm_receipts),
            newly_executed_native_games=20*len(arm_receipts),reused_native_games=20 if plan.get('reuse_historical_beta8',False) else 0,
            historical_beta8_reused=plan.get('reuse_historical_beta8',False),
            beta8_fresh_reproduction=not plan.get('reuse_historical_beta8',False),
            historical_beta8_report=plan['original15_report'] if plan.get('reuse_historical_beta8',False) else None,
            historical_beta8_report_sha256=plan['original15_report_sha256'] if plan.get('reuse_historical_beta8',False) else None,
            workers_max=14,pools_sequential=True,no_overlap=True,
            optimizer_updates=0,teacher_actions=0,safety_takeovers=0,continues=0,native_frame_horizon=None,
            elapsed_seconds=time.monotonic()-started,interpretation='trainedD95 risk dependence;not a separately trained risk-free GRU capacity test',
            original15_reproduction_not_assumed=True,student_checkpoint_read_or_written=False))
    except BaseException as error:
        failure=repr(error)
        write(output/'failure.json',dict(schema=SCHEMA,status='failed',utc=now(),error=failure,traceback=traceback.format_exc(),
            completed_new_arms=[r['arm'] for r in arm_receipts],historical_beta8_reused=plan.get('reuse_historical_beta8',False),
            partial_games_not_scores=True,optimizer_updates=0,no_automatic_retry=True))
        raise
    finally:
        write(output/'cleanup.json',dict(schema=SCHEMA,utc=now(),**identity,remaining_originals=originals(),
            completed_new_arms=[r['arm'] for r in arm_receipts],historical_beta8_reused=plan.get('reuse_historical_beta8',False),
            failure=failure,owned_cleanup_delegated_to_frozen_evaluator=True,
            unrelated_processes_terminated=False,student_auto_resume=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--execute-cuda',action='store_true')
    parser.add_argument('--max-seconds',type=float)
    parser.add_argument('--receipt',type=Path)
    args=parser.parse_args()
    if args.execute_cuda and (args.max_seconds is None or not 0 < args.max_seconds < float('inf')):
        parser.error('CUDA execution requires an explicit positive finite --max-seconds')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
    import psutil
    p=psutil.Process();p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    # Bound inspection CPU only. Do not silently confine all14 native children
    # to the learner's last two cores: keep the original native affinity policy.
    if not args.execute_cuda:p.cpu_affinity(p.cpu_affinity()[-2:])
    plan=read(args.plan)
    helper,config,receipt=inspect(plan)
    if args.receipt:write(args.receipt,receipt)
    if not args.execute_cuda:
        print(json.dumps(receipt))
        return
    execute(plan,helper,config,args.max_seconds)


if __name__=='__main__':main()
