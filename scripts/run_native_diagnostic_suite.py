"""One finite pipeline: verify existing training jobs, audit, then held-out tests.

This is not a scheduler and never restarts a training process. A failed phase
keeps all evidence and prevents downstream execution for that training seed.
"""
from pathlib import Path
import argparse
import json
import os
from statistics import NormalDist
import subprocess
import sys
import time

import psutil
from scipy.stats import beta

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE/'src'))
from fly_touhou.experiment import file_sha256
from fly_touhou.native_diagnostic import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sealed(path):
    return {'path': str(path), 'sha256': file_sha256(path)}


def live_expected(pid, needle):
    try:
        process = psutil.Process(pid)
        return needle in ' '.join(process.cmdline()) and process.is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def start(command, directory, label):
    out = (directory/f'{label}.stdout.log').open('xb')
    err = (directory/f'{label}.stderr.log').open('xb')
    env = dict(os.environ, PYTHONUNBUFFERED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
        env=env, close_fds=True, creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
    return process, out, err


def audit(command, directory, label):
    process, out, err = start(command, directory, label)
    code = process.wait()
    out.close(); err.close()
    if code:
        raise RuntimeError(f'{label} failed with code{code}; inspect preserved logs')


def aggregate(root, seeds):
    policies = ('gru', 'untrained', 'static', 'random')
    reports = {seed: read(root/f'seed{seed}/evaluation/report.json') for seed in seeds}
    # Bonferroni intervals cover all seed/policy binomial rates together. Taking
    # means does not pretend the three trained models share one success rate.
    tail = .05/(2*len(seeds)*len(policies))
    cells, pooled = {}, {}
    for seed, report in reports.items():
        audit_report = read(root/f'seed{seed}/evaluation/independent_audit_v2.json')
        assert audit_report['status'] == 'passed'
        assert audit_report['report_sha256'] == file_sha256(root/f'seed{seed}/evaluation/report.json')
        cells[seed] = {}
        for policy in policies:
            summary = report['summary'][policy]
            x, n = summary['successes'], summary['episodes']
            assert n == 20
            lower = 0. if x == 0 else float(beta.ppf(tail, x, n-x+1))
            upper = 1. if x == n else float(beta.ppf(1-tail, x+1, n-x))
            cells[seed][policy] = {'successes': x, 'episodes': n,
                'rate': x/n, 'simultaneous95_interval': [lower, upper]}
    for policy in policies:
        values = [cells[seed][policy] for seed in seeds]
        pooled[policy] = {'successes': sum(v['successes'] for v in values),
            'episodes': sum(v['episodes'] for v in values),
            'mean_rate': sum(v['rate'] for v in values)/len(seeds),
            'mean_rate_simultaneous95_interval': [sum(v['simultaneous95_interval'][i] for v in values)/len(seeds) for i in (0,1)]}
    controls = policies[1:]
    gain = pooled['gru']['mean_rate']-max(pooled[k]['mean_rate'] for k in controls)
    interval = [pooled['gru']['mean_rate_simultaneous95_interval'][0]-max(pooled[k]['mean_rate_simultaneous95_interval'][1] for k in controls),
                pooled['gru']['mean_rate_simultaneous95_interval'][1]-max(pooled[k]['mean_rate_simultaneous95_interval'][0] for k in controls)]
    seed_gains = {seed: cells[seed]['gru']['rate']-max(cells[seed][k]['rate'] for k in controls) for seed in seeds}
    qualifies = gain >= .20-1e-12 and all(v > 0 for v in seed_gains.values()) and interval[0] > 0
    return {'schema': 'three-seed-independent-native-GRU-diagnostic-result-v1',
        'status': 'G2-evidence-passed' if qualifies else 'G2-not-proven',
        'per_seed': cells, 'policies': pooled, 'gain_vs_strongest_control': gain,
        'per_training_seed_gain': seed_gains, 'gain_simultaneous95_interval': interval,
        'interval_method': 'perseed/perpolicy exact ClopperPearson Bonferroni12cells;mean conditional on these3fixed trainedmodels;not arbitrary-new-training-seed population precision',
        'G1_native_interface_qualified': True, 'G2_achieved': qualifies,
        'G3_achieved': False, 'final_six_stage_acceptance': False,
        'next': 'matched full-fly diagnostic' if qualifies else 'inspect common pipeline,optimization budget,and recorded task learnability;do not conclude observations impossible from this negative result'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve previous suite execution')
    args.output.mkdir(parents=True)
    protocol = read(args.protocol)
    assert read(args.root/'qualification/independent_audit_v2.json')['status'] == 'passed'
    seeds = protocol['training_seeds']
    state = {'schema': 'finite-native-diagnostic-suite-v1', 'status': 'running',
        'suite_pid': os.getpid(), 'protocol': sealed(args.protocol),
        'runner': sealed(Path(__file__)), 'seeds': {str(s): {'phase':'waiting-for-existing-training'} for s in seeds},
        'training_processes_will_not_be_restarted': True, 'final_acceptance': False}
    processes = {}
    learning_auditor = Path('./outputs/legacy-build/source_snapshots/20261019-diagnostic-learning-audit-v1/scripts/audit_diagnostic_learning.py')
    native_auditor = Path('./outputs/legacy-build/source_snapshots/20261019-diagnostic-audit-v2/scripts/audit_native_diagnostic.py')
    try:
        while True:
            if (args.root/'stop_after_current_phase.json').exists():
                state['status'] = 'stopped-by-explicit-request-no-new-phases'
                break
            for seed in seeds:
                item = state['seeds'][str(seed)]
                if item['phase'] == 'finished': continue
                if seed in processes:
                    process, out, err = processes[seed]
                    if process.poll() is None:
                        item.update(phase='heldout-evaluation-running', evaluation_pid=process.pid)
                        continue
                    code = process.returncode; out.close(); err.close()
                    if code: raise RuntimeError(f'Seed{seed} heldout evaluation failed with code{code}')
                    audit_output = args.root/f'seed{seed}/evaluation/independent_audit_v2.json'
                    audit([str(args.python), str(native_auditor), '--run', str(audit_output.parent),
                           '--output', str(audit_output)], args.output, f'native_audit_seed{seed}')
                    item.update(phase='finished', native_audit=sealed(audit_output))
                    continue
                control = Path(f'./outputs/legacy-build/background_jobs/native-list-gru-seed{seed}-v1')
                job = read(control/'status.json')
                if job['status'] in ('starting', 'running'):
                    alive = live_expected(job.get('child_pid', 0), 'run_native_diagnostic.py')
                    worker_alive = live_expected(job['worker_pid'], 'background_job.py')
                    if not alive and not worker_alive:
                        raise RuntimeError(f'Seed{seed} training handles missing without terminal status;preserve and inspect,never restart')
                    item.update(phase='waiting-for-existing-training', child_pid=job.get('child_pid'),
                        child_verified_live=alive, worker_verified_live=worker_alive)
                    continue
                if job['status'] != 'finished' or job['exit_code'] != 0:
                    raise RuntimeError(f'Seed{seed} training terminal failure;downstream not authorized')
                train = args.root/f'seed{seed}/train'
                report = read(train/'report.json')
                assert report['status'] == 'finished'
                if report['counters']['training_decisions'] < protocol['training_decision_budget_per_seed']:
                    raise RuntimeError(f'Seed{seed} stopped before fixed budget;no test/automatic continuation')
                assert file_sha256(report['latest']['path']) == report['latest']['sha256']
                readback = train/'learning_audit_final_v1.json'
                audit([str(args.python), str(learning_auditor), '--run', str(train),
                       '--through', str(report['counters']['updates']), '--output', str(readback)],
                      args.output, f'learning_audit_seed{seed}')
                item['learning_audit'] = sealed(readback)
                command = [str(args.python), str(Path(protocol['source_snapshot'])/'scripts/run_native_diagnostic.py'),
                    '--protocol', str(args.protocol), '--output', str(args.root/f'seed{seed}/evaluation'),
                    '--mode', 'evaluate', '--seed', str(seed), '--checkpoint', report['latest']['path'],
                    '--initial-checkpoint', report['initial']['path']]
                processes[seed] = start(command, args.output, f'evaluate_seed{seed}')
                item.update(phase='heldout-evaluation-running', evaluation_pid=processes[seed][0].pid,
                    evaluation_command=command)
            state['heartbeat_unix'] = time.time()
            write_json(args.output/'status.json', state)
            if all(v['phase'] == 'finished' for v in state['seeds'].values()):
                result = aggregate(args.root, seeds)
                write_json(args.output/'result.json', result, exclusive=True)
                state.update(status='finished', result=sealed(args.output/'result.json'))
                print(json.dumps(result, ensure_ascii=False), flush=True)
                break
            time.sleep(15)
    except BaseException as error:
        state.update(status='failed-preserved', error=repr(error))
        raise
    finally:
        state['updated_unix'] = time.time()
        write_json(args.output/'status.json', state)


if __name__ == '__main__':
    main()
