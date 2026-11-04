"""Finite fresh-test queue for one fixed entity-risk candidate; no training/HPO."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write


def qualifies(report):
    if report.get('status')!='completed' or report.get('completed')!=100 or report.get('native_seed_split')!='development':
        raise ValueError('Expected complete100-game development assessment')
    policies=report['policies']
    for key in ['trained_sampled','initial_sampled','trained_greedy','initial_greedy','uniform18']:
        if policies[key]['episodes']!=20:raise ValueError('Incomplete condition')
    return all(policies['trained_'+mode]['mean_capped_frames']>policies['initial_'+mode]['mean_capped_frames']
               for mode in ['sampled','greedy'])


def run(a):
    import psutil
    plan=read(a.plan);control=Path(plan['control'])
    write(control/'queue_identity.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time()))
    def publish(phase,**kw):write(control/'pipeline_status.json',dict(phase=phase,utc=utc_now(),**kw))
    try:
        before=Path(plan['development_control']);source=Path(plan['development_output'])
        while True:
            if (control/'STOP').exists():raise RuntimeError('Operator stopped pending fresh test')
            status=read(before/'status.json')
            if status['status'] in ('failed','launcher-error'):raise RuntimeError('Required development assessment failed')
            if status['status']=='finished':
                if status.get('exit_code')!=0:raise RuntimeError('Development exit not successful')
                break
            publish('waiting_for_fixed_development',native_instances=0);time.sleep(10)
        development=read(source/'report.json');cleanup=read(source/'cleanup.json')
        if cleanup!=dict(closed=True,close_errors=[]):raise ValueError('Preceding native pool not closed cleanly')
        for label,filename in [('trained','latest.pt'),('initial','initial.pt')]:
            expected=plan['checkpoint_sha256'][label]
            if development['pins'][label]['sha256']!=expected or hashlib.sha256((Path(plan['input'])/filename).read_bytes()).hexdigest()!=expected:
                raise ValueError('Do not switch weights after development')
        promising=qualifies(development)
        write(control/'development_gate.json',dict(utc=utc_now(),promising=promising,
            criterion='Both sampled and greedy average capped survival exceed same existing B1 baseline; exploratory gate only',
            policies=development['policies'],no_native_acceptance_claim=True))
        if not promising:
            publish('candidate_not_promoted',native_instances=0,fresh_games_started=0);return
        command=[sys.executable,str(ROOT/'scripts/evaluate_b1_pair_policy.py'),*plan['test_arguments']]
        publish('running_fresh_test',native_instances_max=18,teacher_reserve=12,combined_native_max=30)
        with (control/'evaluation.stdout.log').open('xb') as stdout,(control/'evaluation.stderr.log').open('xb') as stderr:
            process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            write(control/'dispatch.json',dict(command=command,pid=process.pid,utc=utc_now(),one_shot=True))
            code=process.wait()
        if code:raise RuntimeError('Fresh entity-risk test failed; do not restart automatically')
        output=Path(plan['test_output']);result=read(output/'report.json')
        if result.get('status')!='completed' or result.get('native_seed_split')!='test' or read(output/'cleanup.json')!=dict(closed=True,close_errors=[]):
            raise ValueError('Fresh test result or native cleanup incomplete')
        publish('completed',native_instances=0,policies=result['policies'],paired=result['paired'],greedy_paired=result['greedy_paired'],
                formal_six_stage_acceptance=False)
    except BaseException:
        write(control/'failure.json',dict(utc=utc_now(),traceback=traceback.format_exc(),automatic_restart=False))
        publish('failed');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',type=Path,required=True);run(p.parse_args())
