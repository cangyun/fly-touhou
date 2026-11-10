"""Read a live NBNM run; do not confuse sampled episodes with frozen tests."""
from pathlib import Path
import argparse,json,statistics,sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json


def rows(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:break  # an append may still be writing its final line
    return result


def attempts(records):
    successful=[r for r in records if r['nbnm_success']]
    failed=[r for r in records if not r['nbnm_success']]
    mean=lambda group:statistics.mean([r['task_native_frames'] for r in group]) if group else None
    return dict(completed=len(records),successes=len(successful),failures=len(failed),
        successful_mean_native_frames=mean(successful),failed_mean_native_frames=mean(failed),
        completed_native_seeds=[r['native_seed'] for r in records],
        Bombs=sum(r['task_Bombs'] for r in records),
        spell_captures=sum(r.get('spell_captured',False) for r in records),
        survived_timeouts=sum(r.get('spell_survived_timeout',False) for r in records),
        unfinished_attempts_excluded=True)


def summarize(output):
    output=Path(output)
    if not (output/'progress.json').exists():return dict(status='not_started',output=str(output),network_training=False)
    progress=read_shared_json(output/'progress.json')
    episodes=[]
    for path in sorted(output.glob('native-training-*/training-episodes.jsonl')):episodes+=rows(path)
    segments=[];offset=0
    for record in rows(output/'cycles.jsonl'):
        collected=record['collection']
        count=collected['stage_coverage']['complete_games']
        part=episodes[offset:offset+count];offset+=count
        segments.append(dict(cycle=record['cycle'],sampled_decisions=collected['student_actions'],
            completed_optimizer_updates=record['optimizer_updates'],
            sampling_seconds=record['sample_seconds'],learning_seconds=record['learn_seconds'],
            contacts=collected['contacts'],safe_forecast_but_native_contact=collected['safe_forecast_but_native_contact'],
            task_group_decisions=collected['task_group_decisions'],
            task_decisions=collected.get('task_decisions',{}),
            route_guidance=collected.get('nbnm_route_guidance'),
            tasks={kind:attempts([r for r in part if r['task']['mode']==kind]) for kind in ('attack','chain','stage','full_story')},
            by_task={name:attempts([r for r in part if task_name(r)==name]) for name in sorted({task_name(r) for r in part})}))
    report=dict(schema='nbnm-training-progress-analysis-v1',observed_utc=datetime.now(timezone.utc).isoformat(),
        status={k:progress.get(k) for k in ('phase','cycle','completed_decisions','optimizer_updates','rollout_decisions','budget')},
        scope='Training attempts grouped by segment of completion; episodes can span policy updates. Not frozen evaluation.',
        completed_segments=segments,total_finished_training_attempts=len(episodes),
        pending_segment_finished_attempts=len(episodes)-offset,
        comparison='Avoidance improvement is not established by this training-only report; compare frozen checkpoints on matched development starts.')
    frozen=output/'frozen-card-evaluation/report.json'
    if frozen.exists():
        data=read_shared_json(frozen)
        report['final_frozen']={k:data[k] for k in ('task','attempts','successes','successful_mean_native_frames','opening_seeds','action_selection')}
    report['frozen_by_task']={}
    for path in output.glob('frozen-*/report.json'):
        data=read_shared_json(path)
        report['frozen_by_task'][path.parent.name]={k:data[k] for k in ('task','attempts','successes','successful_mean_native_frames')}
    report['current_workers']=progress.get('worlds',[])
    report['current_task_decisions']=progress.get('task_decisions',{})
    report['training_by_task']={name:attempts([r for r in episodes if task_name(r)==name])
        for name in sorted({task_name(r) for r in episodes})}
    report['current_route_guidance']=progress.get('nbnm_route_guidance')
    return report


def task_name(record):
    task=record['task']
    name=(f"stage{task['stage']}-{task.get('attack',task['mode'])}" if task['mode']!='full_story' else 'full_story')
    if task['mode']=='chain':name+=f"-through-stage{task['end_stage']}-{task.get('end_attack','end')}"
    return name+'-'+record.get('nbnm_entry_mode','story_prefix')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path)
    args=parser.parse_args();report=summarize(args.run)
    text=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text,encoding='utf-8')
    print(text)
