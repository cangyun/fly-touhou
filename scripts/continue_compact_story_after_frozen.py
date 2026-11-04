"""One owned continuation: finish frozen4, select a source, then run its PPO.

Uses completed native development evidence, keeps the paid-cost ledger, and
rebuilds TRAIN-only DN anchors when returning to an earlier brain. No retry,
forced termination, new schedule, or desktop input is introduced.
"""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.compact_native_budget import compare,metrics


def option(argv,name,value):
    argv[argv.index('--'+name)+1]=str(value)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    import psutil
    plan=read(a.plan);control=a.plan.parent;current=Path(plan['current_job'])
    source=Path(plan['current_evaluation']);prepared_path=Path(plan['prepared_training'])
    me=psutil.Process();state=dict(PID=me.pid,create_time=me.create_time(),phase='waiting_for_complete_current_frozen4',plan=str(a.plan))
    def status(phase,**values):
        state.update(phase=phase,**values);write_json(control/'status.json',state)
    def guard():
        if (control/'STOP').exists() or read(current).get('scheduling_suspended'):
            raise InterruptedError('Follow-up scheduling paused;existing owned work left running')
    def source_alive():
        try:
            proc=psutil.Process(plan['current_executor']['PID'])
            return proc.is_running() and abs(proc.create_time()-plan['current_executor']['create_time'])<.01
        except psutil.NoSuchProcess:return False
    def run(argv,output,label,**job_fields):
        guard();output=Path(output)
        with (control/(label+'-stdout.log')).open('xb') as out,(control/(label+'-stderr.log')).open('xb') as err:
            child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
        status(label,launcher_PID=child.pid,output=str(output));receipt=None
        while child.poll() is None:
            f=output/'launch.json'
            if receipt is None and f.exists():
                receipt=read(f);j=read(current)
                j.update(active_output=str(output),active_stage=label,
                    actual_executor={k:receipt[k] for k in ('PID','create_time')},
                    dispatch=str(control/(label+'-dispatch.json')),
                    last_verified_progress=dict(phase='starting_'+label,output=str(output)),**job_fields)
                write_json(control/(label+'-dispatch.json'),dict(argv=argv,output=str(output),
                    actual_executor=j['actual_executor'],launcher_PID=child.pid,controller=str(a.plan)))
                write_json(current,j)
            time.sleep(1)
        if child.returncode:raise RuntimeError(f'{label} exited{child.returncode};no automatic retry')
        return output
    try:
        status('waiting_for_complete_current_frozen4');started=time.monotonic()
        while source_alive():
            guard()
            if time.monotonic()-started>14400:raise TimeoutError('Existing frozen evaluation wait exceeded4 hours;no restart')
            time.sleep(1)
        guard()
        finished=read(source/'result.json')
        if finished.get('status')!='completed_full_story_decoding_evaluation':
            raise ValueError('Current native evaluation did not complete normally')
        report_path=source/'evaluation-family_mode/report.json';candidate=read(report_path)
        if candidate['episodes']!=4:raise ValueError('Complete current frozen4 required')
        reference=read(Path(plan['incumbent_report']));decision=compare(candidate,reference)
        write_json(control/'native-source-comparison.json',dict(decision,
            candidate_report=str(report_path),incumbent_report=plan['incumbent_report'],
            selection='current_calibrated_brain' if decision['improved'] else 'previous_best_native_brain'))
        j=read(current)
        j['historical_jobs'].append(dict(output=str(source),actual_executor=plan['current_executor'],
            status='completed_frozen_Bomb_repair4',clears=candidate['clears'],stage_counts=candidate['stage_counts'],
            report=str(report_path),new_student_actions=0,PPO_updates=0))
        j['latest_completed_frozen_report']=str(report_path)
        if decision['improved']:
            j.setdefault('native_reference_history',[]).append(dict(parameters=j['best_verified_parameters'],report=j['best_verified_native_report']))
            j['best_verified_parameters']=plan['candidate_checkpoint'];j['best_verified_native_report']=str(report_path)
        write_json(current,j)
        training=read(prepared_path);argv=list(training['argv'])
        if decision['improved']:
            checkpoint=Path(plan['candidate_checkpoint']);adam=Path(plan['candidate_training'])
            anchor=Path(plan['candidate_anchor']);source_costs=plan['candidate_costs']
        else:
            # The prior brain needs its own DN responses,not a newer brain's cache.
            calibration=list(plan['calibration_template'])
            option(calibration,'checkpoint',plan['incumbent_checkpoint'])
            option(calibration,'resume-training',plan['incumbent_training'])
            option(calibration,'output',plan['reanchor_output'])
            reanchor=run(calibration,plan['reanchor_output'],'best_brain_Bomb_training_reanchor',
                planned_new_decisions=0,planned_new_Adam_updates=0,planned_readout_updates=128,
                source_costs=plan['incumbent_costs'],native_source_comparison=str(control/'native-source-comparison.json'))
            fitted=read(reanchor/'report.json');anchor=reanchor/'training-Bomb-DN.pt'
            if fitted['candidate_selected']:
                checkpoint=reanchor/'model-parameters.pt';adam=reanchor/'resumable-training.pt'
            else:
                checkpoint=Path(plan['incumbent_checkpoint']);adam=Path(plan['incumbent_training'])
            source_costs=dict(plan['incumbent_costs'])
            if fitted['candidate_selected']:source_costs['bomb_readout_calibration_updates']=fitted['selected_update']
            j=read(current);j['historical_jobs'].append(dict(output=str(reanchor),
                status='completed_best_native_brain_training_only_reanchor',
                paid_readout_updates=fitted['paid_readout_updates'],candidate_selected=fitted['candidate_selected'],
                source_costs=source_costs,new_student_actions=0,PPO_updates=0,report=str(reanchor/'report.json')))
            write_json(current,j)
        guard()
        option(argv,'checkpoint',checkpoint);option(argv,'resume-training',adam)
        option(argv,'bomb-dagger-anchor-cache',anchor)
        j=read(current);option(argv,'budget-benefit-reference',j['best_verified_native_report'])
        training.update(status='selected_bound_and_dispatching',argv=argv,source_is_provisional_until_current_native4_completes=False,
            selected_checkpoint=str(checkpoint),selected_training=str(adam),selected_anchor=str(anchor),source_costs=source_costs,
            native_source_comparison=str(control/'native-source-comparison.json'),
            execution_condition='Explicit user resume;current frozen4 completed;source selected from complete native evidence;matching training-only DN anchor rebuilt when necessary.')
        write_json(prepared_path,training)
        spec_path=prepared_path.parent/'background-spec.json'
        spec=read(spec_path);spec['arguments']=argv[3:];write_json(spec_path,spec)
        output=run(argv,training['output'],'deployment_guided_continuous_full_story_PPO',
            planned_new_decisions=training['maximum_new_decisions'],initial_new_decisions=training['initial_new_decisions'],
            planned_new_Adam_updates=training['maximum_PPO_updates'],planned_evaluation_episodes=4,
            source_costs=source_costs,paid_v4_new_decisions_before_job=training['paid_v4_new_decisions_before_job'],
            paid_total_costs_before_job=training['paid_total_costs_before_job'],
            original_v4_new_decision_budget=training['original_total_v4_decision_budget'],
            conditional_extension_decisions=training['conditional_extension_decisions'],
            next_training_source_is_provisional=False,native_source_comparison=str(control/'native-source-comparison.json'))
        status('completed_bounded_training_job',output=str(output),result=str(output/'result.json'),goal_complete=False)
    except InterruptedError as error:
        status('suspended_no_further_launch',reason=str(error))
    except BaseException as error:
        status('failed_no_automatic_retry',error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
