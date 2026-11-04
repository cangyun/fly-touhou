"""One bounded handoff after an already-running full-game evaluation completes.

Finish its four games, request the trainer's graceful checkpointed stop, then
run one Bomb calibration and, if selected, one frozen four-game comparison.
No forced process termination, topology change or automatic retry.
"""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read
from fly_touhou.thrl_durable_records import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    import psutil
    plan=read(a.plan);control=a.plan.parent;current=Path(plan['current_job']);source=Path(plan['source_output'])
    process=psutil.Process(plan['source_executor']['PID'])
    if abs(process.create_time()-plan['source_executor']['create_time'])>.01:raise ValueError('Source PID identity differs')
    me=psutil.Process();state=dict(PID=me.pid,create_time=me.create_time(),phase='waiting_for_complete_frozen4',plan=str(a.plan))
    write_json(control/'handoff-status.json',state);started=time.monotonic()
    def status(phase,**extra):
        state.update(phase=phase,**extra);write_json(control/'handoff-status.json',state)
    def guard():
        if (control/'STOP').exists():raise RuntimeError('Operator cancelled repair handoff')
    def alive():
        try:return process.is_running() and abs(process.create_time()-plan['source_executor']['create_time'])<.01
        except psutil.NoSuchProcess:return False
    def run(command,output,label):
        guard();out=Path(output)
        with (control/(label+'-stdout.log')).open('xb') as log,(control/(label+'-stderr.log')).open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
        status(label,launcher_PID=child.pid,output=str(out))
        receipt=None
        while child.poll() is None:
            if receipt is None and (out/'launch.json').exists():
                receipt=read(out/'launch.json');j=read(current)
                j.update(active_output=str(out),actual_executor={k:receipt[k] for k in ('PID','create_time')},
                    active_stage=label,dispatch=str(control/(label+'-dispatch.json')),
                    pending_Bomb_repair=str(a.plan))
                dispatch=dict(argv=command,actual_executor=j['actual_executor'],launcher_PID=child.pid,
                    source_checkpoint=plan['checkpoint'],output=str(out),native_games=4 if label=='Bomb_repair_frozen4' else 0)
                write_json(control/(label+'-dispatch.json'),dispatch);write_json(current,j)
            time.sleep(1)
        if child.returncode:raise RuntimeError(f'{label} exited{child.returncode}; no automatic retry')
        return out
    try:
        # evaluations.json is written after evaluate() has closed its workers.
        while True:
            guard()
            if not alive():raise RuntimeError('Source executor ended before controlled handoff')
            f=source/'evaluations.json'
            if f.exists() and any(n.get('cycle')==8 for n in read(f).get('nodes',[])):break
            if time.monotonic()-started>14400:raise TimeoutError('Waiting for existing evaluation exceeded4 hours')
            time.sleep(1)
        report=read(source/'development-cycle008/report.json')
        if report.get('status')!='completed' or report.get('episodes')!=4:raise ValueError('Complete four-game baseline required')
        j=read(current)
        if j['actual_executor']!=plan['source_executor']:raise RuntimeError('Owned job changed before handoff')
        (source/'STOP').write_text('User requested premature Bomb repair after completed cycle8 frozen4. Preserve model and Adam.\n',encoding='utf-8')
        status('waiting_for_checkpointed_stop',baseline=report)
        stopped=time.monotonic()
        while alive():
            if time.monotonic()-stopped>600:raise TimeoutError('Graceful stop exceeded10 minutes; no forced termination')
            time.sleep(1)
        stop=read(source/'failure.json');cleanup=read(source/'native-training/cleanup.json')
        if not stop.get('operator_requested_stop') or not stop.get('operator_stop_checkpoint') or stop.get('checkpoint_save_error') or cleanup.get('errors'):
            raise RuntimeError('Source checkpoint/worker cleanup incomplete')
        j=read(current)
        j['historical_jobs'].append(dict(output=str(source),actual_executor=plan['source_executor'],
            status='controlled_stop_after_complete_cycle8_frozen4_for_premature_Bomb_repair',
            operator_stop_checkpoint=stop['operator_stop_checkpoint'],new_student_actions=65536,
            new_PPO_updates=256,model_lineage_PPO_updates=1216,all_models_tapes_logs_preserved=True,
            native_report=str(source/'development-cycle008/report.json'),clears=report['clears'],stage_counts=report['stage_counts']))
        j['latest_completed_frozen_report']=str(source/'development-cycle008/report.json')
        j['planned_Bomb_repair_native_episodes']=4;write_json(current,j)
        calibration=run(plan['calibration_argv'],plan['calibration_output'],'Bomb_readout_calibration')
        calibrated=read(calibration/'report.json')
        if not calibrated['candidate_selected']:
            status('completed_no_calibration_candidate',report=str(calibration/'report.json'));return
        evaluated=run(plan['evaluation_argv'],plan['evaluation_output'],'Bomb_repair_frozen4')
        status('completed_Bomb_repair_frozen4',report=str(evaluated/'evaluation-family_mode/report.json'),
            calibration_report=str(calibration/'report.json'),goal_complete=False)
    except BaseException as error:
        status('failed_no_automatic_retry',error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
