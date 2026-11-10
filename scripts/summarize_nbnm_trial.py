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
        unfinished_attempts_excluded=True)


def summarize(output):
    output=Path(output)
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
            tasks={kind:attempts([r for r in part if r['task']['mode']==kind]) for kind in ('attack','full_story')}))
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
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path)
    args=parser.parse_args();report=summarize(args.run)
    text=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text,encoding='utf-8')
    print(text)
