"""Record current diagnostic evidence and verified live process handles."""
from pathlib import Path
import json
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fly_touhou.experiment import file_sha256
from fly_touhou.native_diagnostic import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bound(path):
    return {'path':str(path),'sha256':file_sha256(path)}


def process_record(pid, expected):
    try:
        process = psutil.Process(pid)
        command = process.cmdline()
        return {'pid':pid,'verified_live':process.is_running() and expected in ' '.join(command),
                'created_unix':process.create_time(),'command':command}
    except (psutil.NoSuchProcess,psutil.AccessDenied):
        return {'pid':pid,'verified_live':False}


def main():
    run = Path('./outputs/experiments/native-list-diagnostic-v1')
    protocol_path = ROOT/'artifacts/diagnostic_protocol/native-list-v1.json'
    protocol = read(protocol_path)
    qualification = read(run/'qualification/independent_audit_v2.json')
    seeds = {}
    for seed in protocol['training_seeds']:
        control = Path(f'./outputs/legacy-build/background_jobs/native-list-gru-seed{seed}-v1')
        status = read(control/'status.json')
        progress_path = run/f'seed{seed}/train/progress.json'
        progress = read(progress_path) if progress_path.exists() else None
        if progress:
            assert file_sha256(progress['latest']['path']) == progress['latest']['sha256']
        entry = {'control':str(control),'worker_status':status,
            'worker_process':process_record(status['worker_pid'],'background_job.py'),
            'child_process':process_record(status.get('child_pid',0),'run_native_diagnostic.py'),
            'progress':progress,'learning_audits':[bound(p) for p in sorted((run/f'seed{seed}/train').glob('learning_audit*.json'))]}
        evaluation = run/f'seed{seed}/evaluation/report.json'
        if evaluation.exists(): entry['evaluation_report']=bound(evaluation)
        seeds[str(seed)] = entry
    suite_control = Path('./outputs/legacy-build/background_jobs/native-list-diagnostic-suite-v1')
    suite_status = read(suite_control/'status.json')
    suite_pipeline = run/'suite/status.json'
    result = run/'suite/result.json'
    reference_control=Path('./outputs/legacy-build/background_jobs/sampled-fly49-reference20-v1')
    reference_run=run/'preserved_fly49_sampled_reference20'
    reference=None
    if (reference_control/'status.json').exists():
        reference_status=read(reference_control/'status.json')
        reference={'background':reference_status,
            'child_process':process_record(reference_status.get('child_pid',0),'evaluate_fly_diagnostic_reference.py'),
            'progress':read(reference_run/'progress.json') if (reference_run/'progress.json').exists() else None,
            'report':bound(reference_run/'report.json') if (reference_run/'report.json').exists() else None,
            'independent_audits':[bound(p) for p in reference_run.glob('independent_audit*.json')]}
    journal = {'schema':'native-diagnostic-route-current-v2','updated_unix':time.time(),
        'historical_progress_evidence':'fixedprotocol,newGRU,31checks,G1originalqualification318actions,and3CUDAtrainingbranches',
        'state_snapshot_alone_is_not_goal_turn_progress':True,
        'goal_status':'active;G2/G3andfinalLunatic6unmet',
        'historical_pause_preserved':bound(ROOT/'artifacts/runtime/PROJECT_PAUSED.json'),
        'explicit_resume':bound(ROOT/'artifacts/runtime/PROJECT_RESUMED.json'),
        'old_fly_checkpoint49_preserved':protocol['observation_parent'],
        'old_updates50through57_not_started':True,
        'protocol':bound(protocol_path),'documentation':bound(ROOT/'docs/diagnostic_comparison_v1.md'),
        'source_manifest':bound(Path(protocol['source_snapshot'])/'snapshot_manifest.json'),
        'source_files':180,'shared_protected_sources_unchanged_from85':24,
        'unit_checks_passed':31,'G1':{'status':'native qualification passed','independent_audit':bound(run/'qualification/independent_audit_v2.json'),
            'actions_verified':qualification['actions_verified'],'episodes':qualification['episodes_verified'],
            'GPU_CPU_logp_error':qualification['maximum_GPU_CPU_recurrent_logp_error'],
            'first_auditor_UTF8_failure_preserved':bound(run/'qualification/audit_v1_failure.json')},
        'G2':{'status':'training;notproven','seeds':seeds,
            'budget_per_seed':32768,'formal_test_per_policy_per_seed':20,
            'four_test_policies':['trainedGRU','same-seeduntrainedGRU','static','random'],
            'fixed_loss_or_representation_not_changed_during_course':True},
        'suite':{'background':suite_status,
            'worker_process':process_record(suite_status['worker_pid'],'background_job.py'),
            'child_process':process_record(suite_status.get('child_pid',0),'run_native_diagnostic_suite.py'),
            'pipeline':read(suite_pipeline) if suite_pipeline.exists() else None,
            'result':bound(result) if result.exists() else None},
        'user_priority':read(ROOT/'artifacts/runtime/GRU_FIRST_POLICY.json'),
        'G3':{'status':'deferred by user until GRU evidence;no new fly training/evaluation/development',
            'new_fly_training_started':False,'preserved_fly49_reference20':reference},
        'interim_training_diagnostics':{'documentation':bound(ROOT/'docs/diagnostic_interim_evidence.md')
            if (ROOT/'docs/diagnostic_interim_evidence.md').exists() else None,
            'input_analysis':[bound(p) for p in (run/'seed11/train').glob('input_direction_analysis*.json')],
            'private_gradient_probes':[bound(p) for p in (run/'seed11/train').glob('private_*0016_v1.json')]},
        'final_six_stage_acceptance':False,
        'next':'verify same livejobs;on terminal inspect suite audits and heldout result;never duplicate/restart from observationtimeout;follow failedphase evidence before next experiment'}
    path = ROOT/'artifacts/runtime/diagnostic_route_current.json'
    if result.exists():
        outcome=read(result)
        journal['G2']['status']='independent evidence passed' if outcome['G2_achieved'] else 'fixedcourse ended;G2notproven'
        journal['G2']['result']=bound(result)
    reuse_run=run.with_name('native-list-gru-reuse4-v1')
    preparation=reuse_run/'preparation.json'
    if preparation.exists():
        reuse={'preparation':bound(preparation),
            'research_review':bound(ROOT/'docs/native_open_source_review_20261019.md'),
            'private_training_only_comparison':bound(run/'private_reuse_probe_v1/report.json'),
            'formal_game_improvement_proven':False,'new_fly_work':False}
        transition=reuse_run/'transition_status.json'
        if transition.exists():reuse['transition']=read(transition)
        controls={}
        for seed in (11,29,47):
            control=Path(f'./outputs/legacy-build/background_jobs/native-list-gru-seed{seed}-reuse4-v1/status.json')
            if control.exists():
                status=read(control)
                entry={'background':status,
                    'child_process':process_record(status.get('child_pid',0),'run_native_diagnostic.py')}
                progress=reuse_run/f'seed{seed}/train/progress.json'
                if progress.exists():entry['progress']=read(progress)
                entry['learning_audits']=[bound(p) for p in sorted((reuse_run/f'seed{seed}/train').glob('learning_audit*.json'))]
                controls[str(seed)]=entry
        reuse['branches']=controls
        initialization=reuse_run/'initialization_comparison_v1.json'
        if initialization.exists():reuse['initialization_comparison']=bound(initialization)
        suite_control=Path('./outputs/legacy-build/background_jobs/native-list-gru-reuse4-suite-v1/status.json')
        if suite_control.exists():
            status=read(suite_control)
            reuse['suite']={'background':status,
                'child_process':process_record(status.get('child_pid',0),'run_native_diagnostic_suite.py'),
                'pipeline':read(reuse_run/'suite/status.json') if (reuse_run/'suite/status.json').exists() else None}
        reuse_result=reuse_run/'suite/result.json'
        if reuse_result.exists():
            reuse['result']=bound(reuse_result)
            reuse['formal_game_improvement_proven']=read(reuse_result)['G2_achieved']
            journal['G2']['status']=('independent GRU reuse4 evidence passed' if reuse['formal_game_improvement_proven']
                                     else 'GRU reuse4 fixedcourse ended;G2notproven')
            journal['G2']['current_experiment_result']=bound(reuse_result)
            if reuse['formal_game_improvement_proven']:
                journal['goal_status']='active;G2verified;G3andfinalLunatic6unmet'
        elif any(e['child_process']['verified_live'] for e in controls.values()):
            journal['G2']['status']='GRU reuse4 active training;notproven'
        journal['G2']['active_experiment']='native-list-gru-reuse4-v1'
        journal['GRU_reuse4_single_change_experiment']=reuse
    write_json(path,journal)
    print(json.dumps({'journal':str(path),'G1':'passed','G2':journal['G2']['status'],'G3':'deferred by user',
        'training':{seed:{'status':entry['worker_status']['status'],'child_live':entry['child_process']['verified_live'],
            'counters':entry['progress']['counters'] if entry['progress'] else None} for seed,entry in seeds.items()},
        'suite_live':journal['suite']['child_process']['verified_live'],
        'reuse4_training':{seed:{'status':entry['background']['status'],
            'live':entry['child_process']['verified_live'],
            'counters':entry.get('progress',{}).get('counters')}
            for seed,entry in journal.get('GRU_reuse4_single_change_experiment',{}).get('branches',{}).items()}},ensure_ascii=False))


if __name__ == '__main__':
    main()
