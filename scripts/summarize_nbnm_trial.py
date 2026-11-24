"""Read a live NBNM run; do not confuse sampled episodes with frozen tests."""
from pathlib import Path
import argparse,json,statistics,sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json, read_shared_text


def rows(path):
    if not path.exists():return []
    result=[]
    for line in read_shared_text(path).splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:break  # an append may still be writing its final line
    return result


def attempts(records):
    successful=[r for r in records if r['nbnm_success']]
    failed=[r for r in records if not r['nbnm_success']]
    contacts=[r for r in records if r.get('nb_nm_reason')=='first-native-contact']
    mean=lambda group:statistics.mean([r['task_native_frames'] for r in group]) if group else None
    return dict(completed=len(records),successes=len(successful),failures=len(failed),
        successful_mean_native_frames=mean(successful),failed_mean_native_frames=mean(failed),
        first_contact_count=len(contacts),first_contact_mean_native_frames=mean(contacts),
        first_contact_median_native_frames=(statistics.median(r['task_native_frames'] for r in contacts) if contacts else None),
        completed_native_seeds=[r['native_seed'] for r in records],
        Bombs=sum(r['task_Bombs'] for r in records),
        spell_captures=sum(r.get('spell_captured',False) for r in records),
        survived_timeouts=sum(r.get('spell_survived_timeout',False) for r in records),
        reached_stage_counts={str(s):sum(s in r.get('stage_history',[]) for r in records) for s in range(1,7)},
        failures_by_terminal_phase={phase:sum(r.get('terminal_phase')==phase for r in failed)
            for phase in sorted({r['terminal_phase'] for r in failed if r.get('terminal_phase')})},
        max_native_frames=max((r['task_native_frames'] for r in records),default=0),
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
            event_progress=collected.get('nbnm_event_progress'),
            tasks={kind:attempts([r for r in part if r['task']['mode']==kind]) for kind in ('attack','chain','stage','full_story')},
            by_task={name:attempts([r for r in part if task_name(r)==name]) for name in sorted({task_name(r) for r in part})}))
    report=dict(schema='nbnm-training-progress-analysis-v1',observed_utc=datetime.now(timezone.utc).isoformat(),
        status={k:progress.get(k) for k in ('phase','cycle','completed_decisions','optimizer_updates','rollout_decisions','budget')},
        scope='Training attempts grouped by segment of completion; episodes can span policy updates. Not frozen evaluation.',
        completed_segments=segments,total_finished_training_attempts=len(episodes),
        pending_segment_finished_attempts=len(episodes)-offset,
        comparison='Avoidance improvement is not established by this training-only report; compare frozen checkpoints on matched development starts.')
    resume=read_shared_json(output/'resume.json') if (output/'resume.json').exists() else {}
    report['run_progress']=dict(resume_decisions=resume.get('decisions',0),
        resume_optimizer_updates=resume.get('optimizer_updates',0),
        newly_committed_decisions=progress.get('completed_decisions',0)-resume.get('decisions',0),
        newly_committed_optimizer_updates=progress.get('optimizer_updates',0)-resume.get('optimizer_updates',0),
        restored_worlds=resume.get('restored_world_count'),
        elapsed_seconds=progress.get('elapsed_seconds'))
    if progress.get('phase')=='initializing_native_workers':
        # Restoration can take minutes without a learner heartbeat. Report
        # completed native receipts rather than pretending the saved16 worlds
        # are already restored, or mistaking the old heartbeat for a dead run.
        for directory in sorted(output.glob('native-training-*'),reverse=True):
            owner_path=directory/'workers/ownership.json'
            if not owner_path.exists():continue
            owned=read_shared_json(owner_path)
            if owned.get('owner',{}).get('pid')!=progress.get('PID'):continue
            receipts={p.parent.parent.parent.name:read_shared_json(p)
                for p in directory.glob('native/train/worker*/appdata/*/continuation.json')}
            report['native_restore']=dict(completed_worlds=len(receipts),target_worlds=len(owned['workers']),
                registered_native_workers=sum(bool(w.get('native_identity')) for w in owned['workers']),
                replayed_actions=sum(r['replayed_actions'] for r in receipts.values()),new_PPO_samples=0)
            break
    frozen=output/'frozen-card-evaluation/report.json'
    if frozen.exists():
        data=read_shared_json(frozen)
        report['final_frozen']={k:data[k] for k in ('task','attempts','successes','successful_mean_native_frames','opening_seeds','action_selection')}
    report['frozen_by_task']={}
    report['frozen_in_progress']={}
    for folder in output.glob('frozen-*'):
        if not (folder/'report.json').exists() and (folder/'episodes.jsonl').exists():
            partial=rows(folder/'episodes.jsonl')
            report['frozen_in_progress'][folder.name]=dict(completed=len(partial),
                results=[{k:r.get(k) for k in ('native_seed','stage','stage_history','task_native_frames','nbnm_success','end_reason')} for r in partial],
                outcomes=attempts(partial),not_final=True)
    report['evaluation_progress']={k:progress.get(k) for k in ('completed','target','successes','decisions','last_stage','last_stage_frame')} if progress.get('phase')=='frozen_NBNM_evaluation' else None
    for path in output.glob('frozen-*/report.json'):
        data=read_shared_json(path)
        item={k:data.get(k) for k in ('task','attempts','successes','successful_mean_native_frames','learner_cursor','action_selection')}
        item['report']=str(path)
        if 'episode_results' in data:item['outcomes']=attempts(data['episode_results'])
        report['frozen_by_task'][path.parent.name]=item
    report['current_workers']=progress.get('worlds',[])
    report['current_task_decisions']=progress.get('task_decisions',{})
    report['training_by_task']={name:attempts([r for r in episodes if task_name(r)==name])
        for name in sorted({task_name(r) for r in episodes})}
    report['current_route_guidance']=progress.get('nbnm_route_guidance')
    adaptive=output/'adaptive-course.json'
    if progress.get('adaptive_course'):report['adaptive_course']=progress['adaptive_course']
    elif adaptive.exists():report['adaptive_course']=read_shared_json(adaptive)
    stage_course=output/'adaptive-stage-chain.json'
    if stage_course.exists():report['adaptive_stage_chain']=read_shared_json(stage_course)
    report['training_by_scope']={kind:attempts([r for r in episodes if r['task']['mode']==kind])
        for kind in ('attack','chain','stage','full_story')}
    def is_spell(record):
        task=record['task']
        return task.get('spell_id') is not None or 'Card' in task.get('attack','')
    report['training_by_scope']['spell']=attempts([r for r in episodes if r['task']['mode']=='attack' and is_spell(r)])
    report['training_by_scope']['nonspell']=attempts([r for r in episodes if r['task']['mode']=='attack' and not is_spell(r)])
    report['periodic_frozen']=[]
    report['teacher_sources']=[]
    configuration=read_shared_json(output/'configuration.json') if (output/'configuration.json').exists() else {}
    for source in configuration.get('rehearsal',{}).get('additional_manifests',[]):
        path=Path(source);status_path=path.parent/'progress.json'
        if status_path.exists():
            state=read_shared_json(status_path)
            report['teacher_sources'].append(dict(source=str(path),scope='teacher data; not student performance',
                **{k:state.get(k) for k in ('phase','completed','target','successes','decisions','rolling_search_seconds',
                    'rolling_queries','rolling_replans','starts_from_actual_student_states')}))
        elif path.parent.exists():
            report['teacher_sources'].append(dict(source=str(path),phase='restoring_teacher_native_worlds',
                scope='teacher data; not student performance'))
    for path in sorted(output.glob('development-*/report.json')):
        data=read_shared_json(path)
        report['periodic_frozen'].append(dict(task=data['task'],learner_cursor=data.get('learner_cursor'),
            attempts=data['attempts'],successes=data['successes'],
            outcomes=attempts(data['episode_results']),report=str(path)))
    return report


def task_name(record):
    task=record['task']
    name=(f"stage{task['stage']}-{task.get('attack',task['mode'])}" if task['mode']!='full_story' else 'full_story')
    if task['mode']=='chain':name+=f"-through-stage{task['end_stage']}-{task.get('end_attack','end')}"
    return name+'-'+record.get('nbnm_entry_mode','story_prefix')


def compact(report):
    status=report['status']
    if not isinstance(status,dict):return str(status)
    lines=['NBNM 训练与通关进度',
        f"状态：{status.get('phase')}   采样段：{status.get('cycle')}",
        f"已保存决策：{status.get('completed_decisions'):,} / {status.get('budget'):,}   Adam：{status.get('optimizer_updates')}",
        f"当前段采样：{status.get('rollout_decisions') or 0:,}",
        f"本次续训新增已保存决策：{report['run_progress']['newly_committed_decisions']:,}；新增 Adam：{report['run_progress']['newly_committed_optimizer_updates']}",
        '', '训练局（策略变化中，不是冻结评测）：']
    for key,label in (('spell','符卡'),('nonspell','非符'),('stage','整面'),('chain','连段'),('full_story','开局六面连打')):
        r=report['training_by_scope'][key]
        lines.append(f"  {label}：{r['successes']} / {r['completed']} 完成；最长 {r['max_native_frames']} 原作帧")
        if key=='full_story':
            lines.append(f"  已结束连打到达各面：{r['reached_stage_counts']}；首次碰撞中位数：{r['first_contact_median_native_frames']} 原作帧")
    if report['completed_segments']:
        seg=report['completed_segments'][-1];n=seg['sampled_decisions'];g=seg.get('route_guidance') or {}
        lines+=['',f"最近完整采样段：采样 {seg['sampling_seconds']:.1f}s；学习 {seg['learning_seconds']:.1f}s；合计 {n/(seg['sampling_seconds']+seg['learning_seconds']):.2f} 决策/s",
            f"  人类路线指导采用 {g.get('demonstration_guided_rows',0)}/{n}；普通教师未覆盖 {g.get('unguided_rows',0)}/{n}"]
        if g.get('guided_choice_rows'):
            lines.append(f"  教师首选动作平均获学生概率 {g['teacher_top_probability_sum']/g['guided_choice_rows']:.1%}")
        if g.get('border_exit_rows'):
            lines.append(f"  贴边且仍有向内出口：教师向内概率 {g['border_teacher_inward_mass_sum']/g['border_exit_rows']:.1%}；学生 {g['border_student_inward_mass_sum']/g['border_exit_rows']:.1%}")
    if report.get('teacher_sources'):
        lines+=['','教师成功轨迹采集（不是学生成绩）：']
        for source in report['teacher_sources']:
            lines.append(f"  {Path(source['source']).parent.name}：{source['phase']}；已完成 {source.get('completed','?')}/{source.get('target','?')}，成功 {source.get('successes','?')}")
    lines+=['','冻结开发评测（固定参数，局数以各项报告为准）：']
    latest={}
    for r in report['periodic_frozen']:latest[str(r['task'])]=r
    if not latest and not report['frozen_by_task']:lines.append('  尚未完成，暂不能判断通关能力改善。')
    for r in latest.values():
        lines.append(f"  {r['task']}：{r['successes']} / {r['attempts']}；参数进度 {r['learner_cursor']}")
        lines.append(f"  到达各面：{r['outcomes']['reached_stage_counts']}")
    for name,r in report['frozen_by_task'].items():
        lines.append(f"  最终 {name}：{r['successes']} / {r['attempts']}；任务 {r['task']}")
        if r.get('outcomes'):lines.append(f"  到达各面：{r['outcomes']['reached_stage_counts']}")
    for name,r in report['frozen_in_progress'].items():
        state='评估中断' if status.get('phase')=='failed' else '正在评估'
        lines.append(f"  {state} {name}：已结束 {r['completed']} 局，以下不是最终成绩")
        for e in r['results']:
            lines.append(f"    种子 {e['native_seed']}：第{e['stage']}面，{e['task_native_frames']}帧，{e['end_reason']}")
    if report.get('evaluation_progress'):
        e=report['evaluation_progress']
        lines.append(f"  评估进度：{e.get('completed')}/{e.get('target')}；最近活动位于第{e.get('last_stage')}面，阶段帧{e.get('last_stage_frame')}")
    lines+=['','当前实例：']
    for w in ([] if report.get('evaluation_progress') or status.get('phase') in ('failed','complete','stopped') else report.get('current_workers',[])):
        lines.append(f"  {w['world']:02}  {w.get('task_key')}  第 {w['stage']} 面  任务帧 {w.get('task_native_frames', '?')}")
    adaptive=report.get('adaptive_course')
    if adaptive:
        lines+=['','动态局部训练配额（按决策量；训练通过率不是冻结成绩）：']
        for key,row in sorted(adaptive['tasks'].items(),key=lambda kv:kv[1]['target_local_decision_fraction'],reverse=True)[:8]:
            lines.append(f"  {key}: 近期 {row['successes']}/{row['attempts']}，局部目标 {row['target_local_decision_fraction']:.1%}，已采 {row['actual_decisions']} 决策")
    return '\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path)
    parser.add_argument('--compact',action='store_true')
    args=parser.parse_args();report=summarize(args.run)
    text=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text,encoding='utf-8')
    print(compact(report) if args.compact else text)
