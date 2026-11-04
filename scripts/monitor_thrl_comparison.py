"""Native Tk comparison monitor: shared JSON reads, explicit branch STOP only.

Launch: pythonw scripts/monitor_thrl_comparison.py --comparison-output PATH
Headless read: python scripts/monitor_thrl_comparison.py --comparison-output PATH --read-once
No browser, Torch/model import, global keyboard/mouse, or direct game operation.
Optional control center submits versioned jobs to the existing detached launcher.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import queue
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fly_touhou.thrl_study_io import read_shared_json
from fly_touhou.thrl_limits import MAX_NATIVE_WORKERS
from fly_touhou.thrl_training_mode import is_full_story

DEFAULT_CONFIG = {'trial_decisions': 262144,
    'development_decisions': [32768, 65536, 131072, 262144],
    'development_episodes': 6, 'validation_episodes_per_policy': 20}
DEFAULT_BRANCHES = {'A': {'attention': 'none'}, 'B': {'attention': 'both'},
                    'C': {'attention': 'both', 'teacher_pretraining': True}}
DEFAULT_PASSIVE_DIAGNOSTICS_POINTER = (Path(__file__).resolve().parents[1]
    / 'artifacts/engineering/active_thrl_passive_diagnostics.json')
LABELS = {'none': '无注意力', 'both': '空间＋对象注意力', 'spatial': '空间注意力',
    'object': '对象注意力', 'teacher_collection': '教师示范采集中',
    'teacher_pretraining': '教师示范预训练中', 'collecting': '原作采集中',
    'updating': 'GPU 更新中', 'development': '开发评估中',
    'curriculum_gate':'课程晋级复核中',
    'validation20': '固定模型终局验证中', 'validation': '固定模型终局验证中',
    'starting': '启动中', 'initializing': '初始化中', 'training': 'PPO 训练中',
    'running': '运行中', 'preparing': '准备中', 'waiting': '等待中',
    'finished': '已结束', 'completed': '已完成', 'failed': '失败',
    'stopped': '已停止', 'stopping': '正在收尾', 'pruned': '提前结束',
    'operator_or_budget_stop': '已停止', 'error': '需要检查',
    'control_center':'本地控制面板','interrupted':'作业已中断，可继续','ready':'已配置，待手动启动'}


def _dict(value):
    return value if isinstance(value, dict) else {}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def tensorboard_viewer_command(root=None):
    root=Path(root) if root is not None else Path(__file__).resolve().parents[1]
    pointer=read_shared_json(root/'artifacts/engineering/active_tensorboard.json')
    config=Path(pointer['configuration']);python=Path(pointer['pythonw'])
    script=root/'scripts/tensorboard_live.py'
    if not all(p.is_file() for p in (config,python,script)):
        raise FileNotFoundError('TensorBoard入口尚未配置完整')
    return [str(python),'-B',str(script),'--config',str(config),'--open-browser']


RUNNING_PHASES=frozenset(('starting','initializing','preparing','running','training',
    'collecting','updating','teacher_collection','teacher_pretraining','development',
    'curriculum_gate','validation20','validation','stopping'))
GROUP_DISPLAY_NAMES={
    'A0':'A0　原输入＋GRU，无教师预训练', 'A1':'A1　原输入＋GRU，有教师预训练',
    'B0':'B0　整理输入＋GRU，无教师预训练（M0 的 GRU 对照）',
    'B1':'B1　整理输入＋GRU，有教师预训练（M1 的 GRU 对照）',
    'M0':'M0　整理输入＋MLP，无教师预训练', 'M1':'M1　整理输入＋MLP，有教师预训练'}


def running_branches(snapshot):
    """UI-only selection. Verified controller liveness overrides stale phases.

    Keep all branch records in the full snapshot for job control/history. For
    generic comparison files without a controller, use their active phase.
    This pure function does not inspect files, PIDs or models.
    """
    result={}
    for key,branch in _dict(snapshot.get('branches')).items():
        if not isinstance(branch,dict):continue
        control=_dict(branch.get('control'))
        running=(control['running'] is True if 'running' in control
            else branch.get('phase') in RUNNING_PHASES)
        if running:result[key]=branch
    return result


def number(value, digits=2, integer=False):
    if not _finite(value):
        return '—'
    return f'{value:,.0f}' if integer else f'{value:,.{digits}f}'


def teacher_quality_text(teacher,progress):
    result={**_dict(teacher.get('result')),**_dict(progress)}
    validation=_dict(result.get('teacher_validation'))
    status=result.get('teacher_fit_status')
    if not status:return '旧记录未验证示范拟合质量'
    label={'fit-diagnostic-passed':'示范验证达标（还需原作测试）',
        'underfit':'示范拟合不足','underfit-or-unhelpful':'示范拟合不足或未改善',
        'validation-unavailable':'缺少可分离验证集，未预训练',
        'improved-but-underfit':'有改善，拟合仍不足'}.get(status,status)
    return (f'{label}；教师KL {number(validation.get("teacher_KL"),3)}；'
        f'首选动作一致 {number(100*validation["argmax_agreement"],1) if _finite(validation.get("argmax_agreement")) else "—"}%')


def behavior_diagnostics_text(summary):
    data=_dict(summary.get('reward_diagnostics'))
    if not data:return '旧记录未分解决死Bomb和受保护时间'
    if 'deathbomb_usable_decision_count' in data:
        opportunities=(f'\n可见决死窗口 {number(data.get("deathbomb_observed_window_count"),integer=True)}；'
            f'有库存可用决策 {number(data.get("deathbomb_usable_decision_count"),integer=True)}；'
            f'其中Bomb请求 {number(data.get("deathbomb_usable_bomb_requests"),integer=True)} / '
            f'实际消耗 {number(data.get("deathbomb_requested_consumed_decision_count"),integer=True)} / '
            f'消耗且无Miss {number(data.get("deathbomb_requested_consumed_without_miss_decision_count"),integer=True)}'
            '\n以上仅统计动作前后可见机会，不能当作全部决死成功率')
    else:
        opportunities='\n决死机会统计：此运行版本未记录，不能据此计算决死成功率'
    return (f'最近训练回合：实际Bomb {number(data.get("actual_bombs_used"),integer=True)}；'
        f'确认决死 {number(data.get("confirmed_deathbombs"),integer=True)}；普通/未确认 {number(data.get("ordinary_or_unconfirmed_bombs"),integer=True)}\n'
        f'记录的无保护且无中弹/Bomb帧 {number(data.get("vulnerable_frames"),integer=True)}；'
        f'受保护决策 {number(data.get("protected_decisions"),integer=True)}；'
        f'无库存Bomb请求 {number(data.get("empty_inventory_bomb_requests"),integer=True)}'
        f'{opportunities}')


def rescue_learning_text(metrics):
    data=_dict(metrics.get('deathbomb_auxiliary'))
    if not data.get('enabled'):return '决死辅助监督：此运行版本未启用'
    after=_dict(data.get('post_update_policy'))
    chosen=data.get('behavior_Bomb_selection_rate');probability=after.get('Bomb_probability')
    return (f'PPO＋决死辅助监督：本批有效机会 {number(data.get("unique_usable_decisions"),integer=True)}；'
        f'实际选择Bomb {number(chosen*100,1) if _finite(chosen) else "—"}%\n'
        f'更新后这些状态的平均Bomb概率 {number(probability*100,1) if _finite(probability) else "—"}%；'
        '网络自主执行，无自动代放（概率不等于实机救回率）')


def movement_learning_text(metrics,recent=None):
    data=_dict(metrics.get('movement_guidance'))
    if not data:return '移动指导：此运行版本未启用（不是已训练有效的结论）'
    teacher=_dict(data.get('teacher'));reasons=_dict(teacher.get('reasons'))
    names={'clipped_collision_geometry':'碰撞尺寸/速度裁剪','growth_motion_unknown':'子弹变化阶段',
        'nearby_unrepresented_map_hazard':'附近地图威胁未完整入对象','laser_motion_unknown':'激光运动未知',
        'no_feasible_first_action':'暂无可行首动作','no_directional_warning':'无需方向指导',
        'all_first_actions_feasible':'首动作均可继续','label_budget':'达到本批教师配额'}
    abstentions='、'.join(f'{names.get(k,k)} {number(v,integer=True)}' for k,v in
        sorted(reasons.items(),key=lambda row:row[1],reverse=True)[:3])
    evidence=_dict(_dict(recent).get('movement_diagnostics'))
    first=evidence.get('first_collision_observed_frames')
    return (f'移动指导：本批可行集合 {number(data.get("usable_safe_sets"),integer=True)}；'
        f'原作碰撞动作 {number(data.get("observed_collision_actions"),integer=True)} / '
        f'有效四帧转移 {number(data.get("observed_four_tick_transitions"),integer=True)}\n'
        f'教师与实际碰撞冲突 {number(data.get("teacher_collision_contradictions"),integer=True)}（事实优先）；'
        f'教师耗时 {number(data.get("teacher_seconds"),3)} 秒\n'
        f'跳过原因：{abstentions or "—"}\n'
        f'最近训练局首次观测碰撞 {number(first,integer=True)} 原作帧（包含决死救回；非独立测试）\n'
        '教师仅参与训练；PPO长期回报继续，网络自主执行')


def hardware_text(hardware):
    """Format supplied global readings only; never query CPU/CUDA ourselves."""
    h = _dict(hardware)
    return (f'硬件：CPU {number(h.get("cpu_percent"), 1)}%　'
        f'RAM 可用 {number(h.get("available_ram_gib"), 1)} / 总 {number(h.get("total_ram_gib"), 1)} GiB　'
        f'GPU {number(h.get("gpu_utilization_percent"), 1)}%　'
        f'显存 {number(h.get("gpu_memory_used_mib"), integer=True)} / '
        f'{number(h.get("gpu_memory_total_mib"), integer=True)} MiB')


def attach_dodge_diagnostics(snapshot,readers,reader_factory=None):
    """Bounded incremental log reads in the reader thread, never the Tk renderer."""
    if reader_factory is None:
        from fly_touhou.thrl_dodge_diagnostics import DodgeDiagnosticsReader
        reader_factory=DodgeDiagnosticsReader
    active_paths=set()
    for branch in snapshot.get('branches',{}).values():
        run=branch.get('training_run')
        if not run:
            branch['dodge_diagnostics']={'status':'waiting','reason':'尚无当前训练日志；未开启额外游戏'}
            continue
        run=str(Path(run).resolve())
        metadata=dict(_dict(branch.get('diagnostic_protocol')))
        if metadata.get('training_mode')!='full-story-v1':
            branch['dodge_diagnostics']={'status':'unsupported-protocol',
                'reason':'此运行是旧课程或协议未标记；保留原显示，不读取旧日志计算完整新局趋势'}
            continue
        active_paths.add(run)
        metadata['source_run']=run
        job=_dict(_dict(branch.get('control')).get('job'))
        metadata['job_submitted_utc']=job.get('submitted_utc')
        progress=branch.get('progress') if branch.get('display_run')==branch.get('training_run') else None
        if progress and _finite(progress.get('wall_seconds')):
            try:
                updated=datetime.fromisoformat(progress['updated_utc'].replace('Z','+00:00'))
                metadata['run_started_utc_estimate']=(updated-timedelta(seconds=progress['wall_seconds'])).isoformat()
                metadata['run_start_basis']='progress.updated_utc minus wall_seconds; estimated, not a native event'
            except (KeyError,TypeError,ValueError,OverflowError):pass
        try:
            if run not in readers:
                readers[run]=reader_factory(Path(run),window_episodes=30,read_budget_bytes=262144)
            data=readers[run].poll(progress=progress,protocol_metadata=metadata)
            data['monitor_source']={'training_run':run,**metadata}
            branch['dodge_diagnostics']=data
        except (OSError,ValueError) as error:
            branch['dodge_diagnostics']={'status':'unavailable','reason':str(error),'source_run':run}
    for run in set(readers)-active_paths:readers.pop(run)
    return snapshot


def dodge_diagnostics_text(data,branch=None):
    """Render existing statistics only. This function performs no file reads."""
    data=_dict(data);branch=_dict(branch)
    if data.get('status') in ('waiting','unavailable','unsupported-protocol'):
        return '躲避诊断：'+str(data.get('reason') or '等待完整训练日志')
    if not data:return '等待完整训练局日志；不会为诊断另开游戏'
    lines=['训练采样诊断：仅统计已结束回合；随机策略持续更新，不是独立固定模型测试。']
    full_groups=[]
    for group in data.get('protocol_groups') or []:
        group=_dict(group)
        if group.get('protocol')!='full-story-v1':
            lines.append(f'{group.get("label","旧协议/未知协议")}：单独保留，不计算完整新局趋势。')
            continue
        full_groups.append(group)
        initial,recent=_dict(group.get('initial')),_dict(group.get('recent'))
        lines.append(f'完整开局窗口：已读较早 {number(initial.get("episodes"),integer=True)} 局 → 最近 {number(recent.get("episodes"),integer=True)} 局（窗口上限30局）。')
        for label,window in (('较早',initial),('最近',recent)):
            frames=_dict(window.get('native_frames'));stages=_dict(window.get('stage_reach_counts'))
            reached=' / '.join(f'{stage}面 {number(count,integer=True)}局' for stage,count in sorted(stages.items())
                if _finite(count) and count>0) or '关卡覆盖缺失'
            lines.append(f'{label}：推进帧均值 {number(frames.get("mean"),1)} / 中位 {number(frames.get("median"),1)}'
                f'（有效 {number(frames.get("coverage_episodes"),integer=True)}/{number(window.get("episodes"),integer=True)} 局）；'
                f'{reached}（关卡记录 {number(window.get("stage_history_coverage_episodes"),integer=True)}/{number(window.get("episodes"),integer=True)} 局）')
            lines.append(f'　Miss {number(window.get("misses"),integer=True)}；实际Bomb {number(window.get("actual_bombs_used"),integer=True)}；'
                f'普通/未确认 {number(window.get("ordinary_or_unconfirmed_bombs"),integer=True)}；确认决死 {number(window.get("confirmed_deathbombs"),integer=True)}')
        coverage=_dict(recent.get('coverage'));count=recent.get('episodes')
        lines.append(f'最近窗口覆盖：Miss {number(coverage.get("misses"),integer=True)}/{number(count,integer=True)} 局；'
            f'Bomb {number(coverage.get("actual_bombs_used"),integer=True)}/{number(count,integer=True)} 局；缺字段显示“—”，不当作0。')
        protected=recent.get('protected_decision_fraction')
        lines.append(f'最近：每1000推进帧 Miss {number(recent.get("misses_per_1000_native_frames"),3)}；'
            f'受保护决策占比 {number(protected*100,1) if _finite(protected) else "—"}%（均非纯躲避评分）。')
        deathbomb=_dict(recent.get('deathbomb'));rate=deathbomb.get('conditional_request_rate')
        consumed=deathbomb.get('requested_consumed_without_miss_fraction')
        lines.append(f'可见决死：有库存决策 {number(deathbomb.get("usable_decision_opportunities"),integer=True)}；'
            f'其中请求 {number(deathbomb.get("requests"),integer=True)}，请求率 {number(rate*100,1) if _finite(rate) else "—"}%；'
            f'请求后消耗且无Miss {number(consumed*100,1) if _finite(consumed) else "—"}%。')
        lines.append('真实决死成功率不可得：决策计数不是全部独立决死窗口，未观测窗口仍未知。')
        resources=_dict(recent.get('resources'))
        if resources:
            lines.append('最近终局资源：'+ ' / '.join(f'{label} {number(_dict(resources.get(field)).get("mean"),2)}'
                for field,label in (('end_power','原始P值'),('end_bombs','Bomb'),('end_lives','残机')))+'；终局库存不等于整局资源收益。')
        else:
            lines.append('完整局资源窗口尚无数据；实时残机/Bomb/P见“当前原作实例”。')
        comparison=_dict(group.get('comparison'))
        if comparison.get('available'):
            delta=_dict(comparison.get('delta'));delta_coverage=_dict(comparison.get('delta_coverage'))
            frame_coverage=_dict(delta_coverage.get('median_native_frames'))
            bomb_coverage=_dict(delta_coverage.get('ordinary_or_unconfirmed_bombs_per_episode'))
            def covered_delta(name,coverage):
                return (delta.get(name) if coverage.get('available') is True and all(
                    _finite(coverage.get(key)) and coverage[key]>=20 for key in ('initial','recent')) else None)
            lines.append(f'近期－较早（描述性变化）：推进帧中位数 {number(covered_delta("median_native_frames",frame_coverage),1)}；'
                f'普通Bomb/局 {number(covered_delta("ordinary_or_unconfirmed_bombs_per_episode",bomb_coverage),2)}。')
            lines.append(f'比较有效局（较早/近期）：推进帧 {number(frame_coverage.get("initial"),integer=True)}/{number(frame_coverage.get("recent"),integer=True)}；'
                f'普通Bomb {number(bomb_coverage.get("initial"),integer=True)}/{number(bomb_coverage.get("recent"),integer=True)}。')
            lines.append('“—”表示该指标缺数据或任一窗口不足20个有效局；总回合数不替代指标覆盖数。')
            lines.append('窗口变化不构成显著改善结论；必须同时看推进、Miss、Bomb及独立测试。')
        else:
            lines.append(f'趋势暂不足判断：需要两个互不重叠的同协议窗口，且对应指标在两窗各至少{number(comparison.get("minimum_episodes",20),integer=True)}局有效数据。')
    if not full_groups:lines.append('尚无可识别的完整开局训练窗口；旧课程局不能替代。')
    reader=_dict(data.get('reader'))
    if reader.get('error'):lines.append('日志读取暂未成功：'+str(reader['error']))
    if reader.get('file_available') is False:lines.append('等待本次运行写出完整训练局日志。')
    elif reader.get('caught_up') is False:lines.append('新增日志尚未读完，将在后续刷新补读；本次窗口可能不完整。')
    if reader.get('history_complete') is False:lines.append('为限制读取开销，仅保留已读历史窗口；这不是全历史汇总。')
    if reader.get('malformed_lines'):lines.append(f'跳过无法解析的日志 {number(reader["malformed_lines"],integer=True)} 行。')
    lines.append('首次中弹时间、最长连续无伤段：现有日志无法重建；无保护帧是筛选记录，不是精确风险暴露时间。')
    development=[row for row in branch.get('development',[]) if row.get('complete')]
    validation=[row for row in branch.get('validation',[]) if row.get('complete')]
    lines.append(f'独立评估另列：完整开发节点 {len(development)}，最终测试报告 {len(validation)}；均未混入上方训练窗口。')
    return '\n'.join(lines)


def _bounded_shared_json(path, max_bytes):
    """One bounded shared read. Never hold a handle while parsing or rendering."""
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    with _shared_binary(path) as stream:
        raw=stream.read(max_bytes+1)
    if len(raw)>max_bytes:raise ValueError('旁读报告超过监控读取大小上限')
    value=json.loads(raw.decode('utf-8-sig'))
    if not isinstance(value,dict):raise ValueError('旁读报告必须是JSON对象')
    return value


def _absolute_report_path(value):
    if not isinstance(value,str) or not value or not Path(value).is_absolute():
        raise ValueError('旁读来源和报告必须使用绝对路径')
    return Path(value).resolve()


def attach_passive_diagnostics(snapshot,pointer=DEFAULT_PASSIVE_DIAGNOSTICS_POINTER):
    """Attach only an existing report for the exact current run; no game access."""
    branches=list(_dict(snapshot.get('branches')).values())
    for branch in branches:
        branch['passive_diagnostics']={'status':'waiting','reason':'当前训练尚无匹配的旁读采样'}
    try:
        document=_bounded_shared_json(Path(pointer),16384)
    except FileNotFoundError:
        return snapshot
    except (OSError,ValueError,UnicodeError) as error:
        for branch in branches:
            branch['passive_diagnostics']={'status':'unavailable','reason':str(error)}
        return snapshot
    targets=[]
    try:
        if document.get('schema')!='th14-passive-diagnostics-pointer-v1':
            raise ValueError('旁读指针协议不匹配')
        source=_absolute_report_path(document.get('source_run'))
        report_path=_absolute_report_path(document.get('report_path'))
        targets=[branch for branch in branches if branch.get('training_run')
            and _absolute_report_path(branch['training_run'])==source]
        if not targets:return snapshot
        report=_bounded_shared_json(report_path,1048576)
        if report.get('schema')!='th14-passive-diagnostics-v1':
            raise ValueError('旁读报告协议不匹配')
        if _absolute_report_path(report.get('source_run'))!=source:
            raise ValueError('旁读报告来源与指针不一致，已隔离')
        data={**report,'monitor_source':{'source_run':str(source),'report_path':str(report_path)}}
        for branch in targets:branch['passive_diagnostics']=data
    except (OSError,ValueError,UnicodeError) as error:
        for branch in targets or branches:
            branch['passive_diagnostics']={'status':'unavailable','reason':str(error)}
    return snapshot


def passive_diagnostics_text(data):
    """Format supplied sparse telemetry only, separate from training/evaluation."""
    data=_dict(data)
    if not data or data.get('status')=='waiting':
        return '独立旁读诊断：未采集当前训练的匹配报告；不会自动开启采样。'
    if data.get('status')=='unavailable':
        return '独立旁读诊断：暂不可读；'+str(data.get('reason') or '等待一致的报告')
    status={'sampling':'采样中','completed':'采样已结束','failed':'采样失败'}.get(
        data.get('status'),str(data.get('status') or '—'))
    lines=['独立旁读诊断（现有单个实例；不计入训练窗口、开发或正式测试）：',
        f'{status}；更新时间 {data.get("updated_utc") or "—"}；已采样 {number(data.get("wall_seconds"),1)} 秒；'
        f'有效样本 {number(data.get("sample_count"),integer=True)}；实例 {number(data.get("worker"),integer=True)}；'
        f'采样频率上限 {number(data.get("frequency_hz_limit"),1)} Hz。']
    alerts=data.get('selection_alerts')
    if isinstance(alerts,list):
        for alert in [row for row in alerts if isinstance(row,dict)][:3]:
            lines.append(f'对象名额挤占：{number(alert.get("stage"),integer=True)}面 '
                f'{number(alert.get("stage_frame"),integer=True)}帧；弹幕候选 '
                f'{number(alert.get("bullet_candidates"),integer=True)}，选中弹幕 '
                f'{number(alert.get("selected_bullets"),integer=True)}；选中道具 '
                f'{number(alert.get("selected_items"),integer=True)} / 敌机 '
                f'{number(alert.get("selected_enemies"),integer=True)} / Boss '
                f'{number(alert.get("selected_bosses"),integer=True)} / 激光点 '
                f'{number(alert.get("selected_lasers"),integer=True)}；人物无敌剩余 '
                f'{number(alert.get("player_invulnerability_frames"),integer=True)}帧；'
                f'危险弹幕代理遗漏 {number(alert.get("danger_bullets_omitted"),integer=True)}。'
                '地图仍可能保留弹幕，非实测碰撞。')
    audit=_dict(data.get('observation_audit'));groups=_dict(audit.get('selection_groups'))
    for name,label in (('bullet','子弹'),('laser','激光采样点'),('enemy','敌机'),('boss','Boss'),('item','道具')):
        group=_dict(groups.get(name))
        if not group and name not in ('bullet','laser'):continue
        lines.append(f'{label}：对象候选 {number(group.get("candidate_tokens"),integer=True)} / '
            f'保留 {number(group.get("selected_tokens"),integer=True)}；危险候选 '
            f'{number(group.get("danger_tokens"),integer=True)} / 保留 '
            f'{number(group.get("selected_danger_tokens"),integer=True)} / 遗漏 '
            f'{number(group.get("omitted_danger_tokens"),integer=True)}。')
    lines.append('危险是线性运动代理，非真实碰撞；重复快照会重复计数，激光点不是独立激光。对象列表遗漏不代表地图也遗漏。')
    selected=_dict(audit.get('object_fields_selected'));overflows=[];incomplete=0
    for field,counts in selected.items():
        counts=_dict(counts);below=counts.get('below');above=counts.get('above')
        if not all(_finite(value) and value>=0 for value in (below,above)):
            incomplete+=1;continue
        if below+above:overflows.append((str(field),below+above))
    overflows.sort(key=lambda row:(-row[1],row[0]))
    overflow_text=('；'.join(f'{field} {number(count,integer=True)} 次' for field,count in overflows[:8])
        if overflows else ('0 次' if selected and not incomplete else '—'))
    if len(overflows)>8:overflow_text+=f'；另有 {len(overflows)-8} 个字段'
    if incomplete:overflow_text+=f'；{incomplete} 个字段记录不完整'
    lines.append('已选对象输入严格超界：'+overflow_text+'（仅 below＋above，等于边界不计；非地图裁剪统计）。')
    episodes=[row for row in data.get('episodes',[]) if isinstance(row,dict)]
    def total(*keys):
        values=[]
        for episode in episodes:
            value=episode
            for key in keys:value=_dict(value).get(key)
            if not _finite(value) or value<0:return None
            values.append(value)
        return sum(values) if values else None
    protected=total('protection','observation_counts','protected')
    unprotected=total('protection','observation_counts','unprotected')
    denominator=(protected+unprotected if _finite(protected) and _finite(unprotected) else None)
    ratio=100*protected/denominator if _finite(denominator) and denominator>0 else None
    lines.append(f'快照受保护占比 {number(ratio,1)}%（受保护 {number(protected,integer=True)} / '
        f'可分类 {number(denominator,integer=True)}）；不是原作时间占比，保护可能来自Bomb、复活或其他原因。')
    visible=total('collision','state4_observations')
    lower=total('collision','total_collision_events_lower_bound')
    first=[row.get('first_observed_collision') for row in episodes if isinstance(row.get('first_observed_collision'),dict)]
    first_label=('；'.join(f'{row.get("stage","—")}面帧 {number(row.get("stage_frame"),integer=True)}'
        for row in first[:3]) or '—')
    lines.append(f'可见碰撞证据：state4 快照 {number(visible,integer=True)}；碰撞事件下界 '
        f'{number(lower,integer=True)}；各记录段首次观测位置 {first_label}。')
    gaps=[_dict(row.get('coverage')).get('max_frame_interval') for row in episodes]
    max_gap=max(gaps) if gaps and all(_finite(value) and value>=0 for value in gaps) else None
    lines.append(f'盲区：最大观测间隔 {number(max_gap,integer=True)} 原作帧；未采到的帧点 '
        f'{number(total("coverage","unsampled_native_frame_checkpoints"),integer=True)}；'
        '真实首次中弹帧、全部碰撞、最长连续无伤仍为未知；未观察到碰撞不等于没有碰撞。')
    lines.append('单实例稀疏旁读会漏过决死和保护窗口，不能据此判定已经学会躲避。')
    return '\n'.join(lines)


def curriculum_text(config):
    if is_full_story(config):
        return '单策略全程连打：每局从第一面开局，到原作 Game Over 或通关；允许自主 Bomb，禁止续关'
    if config.get('curriculum_manifest'):
        return '渐进课程：按顺序解锁；段内扩大随机起点；旧段复习与开局连打；原作种子分区'
    fraction=config.get('full_story_fraction')
    if not _finite(fraction):return '训练起点计划待上报；开发和最终测试从第一面开始'
    return f'训练起点：{100*fraction:g}% 完整新局，其余为已验证片段起点；评估从第一面开始'


def curriculum_progress_text(progress,config=None):
    if is_full_story(config or {}) or is_full_story(progress):
        history=_dict(progress.get('recurrent_history'))
        model=_dict((config or {}).get('model_config'))
        memory=('MLP 前馈：无隐藏状态'
            if model.get('temporal_core')=='mlp' else
            'GRU 跨更新重算近期历史；每个新局清空隐藏状态')
        refresh=(f'；已重算 {number(history.get("current_parameter_refreshes"),integer=True)} 次'
            if history.get('current_parameter_refreshes') is not None else '')
        return ('单一网络覆盖全程；所有训练、开发和最终测试均从开局开始。\n'
            '回合终点：原作 Game Over 或通关；无分段截断、倒序起点或课程晋级复核。\n'
            f'{memory}{refresh}。PPO 按完整采样批次更新，回合可跨多个批次。')
    c=_dict(progress.get('curriculum'))
    if not c:return '此作业没有渐进课程记录；旧实验不会改写成新课程'
    store=_dict(c.get('store'));history=_dict(c.get('recurrent_history'))
    fractions=_dict(c.get('observed_fractions'));gate=_dict(c.get('gate'))
    memory_text=(f'GRU 历史预热最多 {number(c.get("burn_in"),integer=True)} 步；参数更新后已重算 {number(history.get("current_parameter_refreshes"),integer=True)} 次'
        if c.get('temporal_core','gru')=='gru' else 'MLP 前馈：无隐藏状态，不采集历史预热，不重算GRU历史')
    priority={'normal':'常规','review':'旧段保持未达标','chain':'连打或资源仍需练习'}.get(c.get('practice_priority'),'常规')
    chain=number(100*gate['chain_pass_rate'],1)+'%' if _finite(gate.get('chain_pass_rate')) else '待独立复核'
    reserve=number(100*gate['chain_resource_pass_rate'],1)+'%' if _finite(gate.get('chain_resource_pass_rate')) else '—'
    target=_dict(c.get('target_fractions'))
    evaluation=_dict(progress.get('evaluation_progress'))
    last_timing=_dict(gate.get('evaluation'))
    last_speed=(f'上次复核耗时 {number(gate.get("wall_seconds"),1)}s；'
        f'平均推理批量 {number(last_timing.get("mean_policy_batch_size"),2)}；'
        f'推理 {number(last_timing.get("policy_seconds"),1)}s / 等待 {number(last_timing.get("ready_wait_seconds"),1)}s\n'
        if last_timing else '')
    evaluation_text=(f'复核执行：{number(evaluation.get("workers",1),integer=True)} 并行实例；'
        f'总完成 {number(evaluation.get("total_completed"),integer=True)} / {number(evaluation.get("total_target"),integer=True)} 局；'
        f'当前类 {number(evaluation.get("completed"),integer=True)} / {number(evaluation.get("target"),integer=True)} 局；'
        f'已用 {number(evaluation.get("elapsed_seconds"),1)}s；平均批量 {number(evaluation.get("mean_policy_batch_size"),2)}\n'
        if evaluation.get('status')=='curriculum_gate' else '')
    retention=' / '.join(f'{key} {100*value:.0f}%' for key,value in _dict(gate.get('retention')).items() if _finite(value)) or '待进行'
    chain_scope=('训练连打：到Game Over或通关；晋级复核仍结算目标段\n'
        if c.get('curriculum_chain_horizon')=='native-game-v1' else '训练连打：到当前目标段末\n')
    inherited_gate='（升级前继承成绩）' if gate.get('inherited') else ''
    return (f'{chain_scope}当前：{c.get("label",c.get("lesson","—"))} · 已解锁 {number(c.get("unlocked"),integer=True)} / {number(c.get("catalogue_lessons"),integer=True)}\n'
        f'起点范围：末尾 {number(c.get("depth"),integer=True)} / 全段 {number(c.get("max_depth"),integer=True)} 原作帧\n'
        f'最近 {number(c.get("recent_attempts"),integer=True)} 次有效尝试，通过率 {number(100*c["recent_pass_rate"],1) if _finite(c.get("recent_pass_rate")) else "—"}% / 晋级门槛 {number(100*c.get("threshold",.8),0)}%\n'
        f'实际决策分配：新段 {number(100*fractions.get("current",0),1)}% / 复习 {number(100*fractions.get("review",0),1)}% / 连打 {number(100*fractions.get("chain",0),1)}%\n'
        f'目标配额：当前 {number(100*target.get("current",0),0)}% / 复习 {number(100*target.get("review",0),0)}% / 连打 {number(100*target.get("chain",0),0)}%\n'
        f'原作轨迹 {number(store.get("trajectories"),integer=True)} 条 / {number(store.get("distinct_native_seeds"),integer=True)} 个种子 / 已核验 {number(store.get("verified_checkpoints"),integer=True)} 个状态\n'
        f'{memory_text}\n'
        f'最近课程复核{inherited_gate}：{number(100*gate["pass_rate"],1)+"%" if _finite(gate.get("pass_rate")) else "待进行"}；连打 {chain} / 保留资源 {reserve}；旧段 {retention}\n'
        f'{evaluation_text}'
        f'{last_speed}'
        f'当前配额：{priority}；训练回合含片段截断；晋级门槛只调整课程，不拒绝 PPO 更新')


def refresh_idle_hardware(snapshot,cache):
    """Keep hardware live while a stopped/repaired supervisor has stale readings."""
    import os
    import shutil
    import subprocess
    import time
    import psutil
    now=time.monotonic();hardware=_dict(snapshot.get('hardware'))
    try:
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(hardware['gpu_observed_utc'])).total_seconds()
    except (KeyError,TypeError,ValueError):age=9999
    if age<=20:return
    if now-cache.get('time',-9999)>=15:
        memory=psutil.virtual_memory()
        value={'cpu_percent':psutil.cpu_percent(None),'available_ram_gib':memory.available/2**30,
            'total_ram_gib':memory.total/2**30,'source':'monitor-read-only-fallback'}
        executable=shutil.which('nvidia-smi')
        if executable:
            try:
                result=subprocess.run([executable,'--query-gpu=utilization.gpu,memory.used,memory.total',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                gpu=[float(item.strip()) for item in result.stdout.splitlines()[0].split(',')]
                value.update(gpu_utilization_percent=gpu[0],gpu_memory_used_mib=gpu[1],gpu_memory_total_mib=gpu[2])
            except (OSError,ValueError,IndexError,subprocess.TimeoutExpired):pass
        cache.update(time=now,hardware=value)
    snapshot['hardware']=cache.get('hardware',hardware)


def candidate_target(branch):
    """Candidate/development axis is independent of a resumed segment budget."""
    progress, config = _dict(branch.get('progress')), _dict(branch.get('configuration'))
    return _first(progress.get('candidate_target_decisions'), config.get('candidate_trial_decisions'),
                  config.get('trial_decisions'))


def development_axis_max(branches):
    steps = [row['decisions'] for branch in branches.values() for row in branch['development']
             if row['complete'] and _finite(row.get('decisions'))]
    targets = [candidate_target(branch) for branch in branches.values()]
    return max(steps + [value for value in targets if _finite(value)] + [1])


def _first(*values):
    return next((value for value in values if value is not None), None)


def _path(value, base):
    if not isinstance(value, (str, Path)) or not str(value):
        return None
    result = Path(value)
    return (result if result.is_absolute() else base / result).resolve()


def _read(path, errors, label):
    if path is None or not path.is_file():
        return {}
    try:
        value = read_shared_json(path)
        if not isinstance(value, dict):
            raise ValueError('Expected JSON object')
        return value
    except (OSError, ValueError) as error:
        errors.append(f'{label}暂未就绪（{type(error).__name__}）')
        return {}


def failure_summary(document):
    text=str(document.get('traceback') or document.get('error') or '')
    for line in reversed(text.replace('\\n','\n').splitlines()):
        if 'Error:' in line or 'Exception:' in line:
            return line.strip()[:360]
    return text[:360]


def active_study_snapshot(pointer,last_runs=None):
    document=read_shared_json(pointer)
    root=Path(document['comparison_output']).resolve()
    last_runs={key:value for key,value in (last_runs or {}).items()
               if value and Path(value).resolve().is_relative_to(root)}
    snapshot=load_comparison_snapshot(document['comparison_output'],last_runs)
    snapshot['engineering_note']=document.get('note','')
    return snapshot


def _report(document, path, *, expected_episodes=None, name=None, decisions=None):
    row = {key: document.get(key) for key in ('episodes', 'clears', 'score',
        'mean_native_frames', 'mean_max_stage', 'status')}
    row.update(path=str(path), name=name, decisions=decisions)
    count = row['episodes']
    row['complete'] = (_finite(count) and count > 0
        and (expected_episodes is None or count == expected_episodes)
        and row['status'] not in ('failed', 'error', 'partial', 'stopped'))
    return row


def _branch_snapshot(key, descriptor, root, common_config, last_run):
    errors = []
    output = _path(descriptor.get('output'), root) or (root / f'branch_{key}').resolve()
    study = _read(output / 'study_status.json', errors, f'{key}分支状态')
    root_progress = _read(output / 'progress.json', errors, f'{key}分支进度')
    attention = descriptor.get('attention', DEFAULT_BRANCHES.get(key, {}).get('attention'))
    candidates = sorted(output.glob('trial_*')) if output.is_dir() else []
    candidates = [path for path in candidates if path.is_dir()]
    matching = [path for path in candidates if attention and path.name.endswith('_'+str(attention))]
    training_run = _path(_first(descriptor.get('training_run'), study.get('training_run'),
        study.get('last_training_run')), output)
    if training_run is None:
        training_run = _path(last_run, output)
    if training_run is None and (matching or candidates):
        training_run = (matching or candidates)[-1]
    current = _path(_first(study.get('current_run'), descriptor.get('current_run'),
                         root_progress.get('current_run')), output)
    if current is not None and current.name.startswith('trial_'):
        training_run = current
    display_run = current or training_run
    inherited_fields = ('actual_decisions', 'adam_minibatch_steps',
        'inherited_decisions', 'lifetime_decisions', 'inherited_adam_minibatch_steps',
        'lifetime_adam_minibatch_steps', 'candidate_decisions', 'candidate_target_decisions',
        'candidate_decision_offset', 'teacher_samples', 'teacher_pretrain_updates', 'pretrain_updates')
    progress = {**{field: descriptor[field] for field in inherited_fields if field in descriptor},
                **_dict(study.get('current_progress')), **_dict(descriptor.get('progress')),
                **root_progress, **_read(display_run / 'progress.json' if display_run else None,
                                       errors, f'{key}当前实例进度')}
    study_config = _read(output / 'study_config.json', errors, f'{key}分支配置')
    run_config = _read(training_run / 'config.json' if training_run else None, errors, f'{key}模型配置')
    actual_config = {**common_config, **study_config, **_dict(descriptor.get('config')),
                     **{field: descriptor[field] for field in (*DEFAULT_CONFIG, 'candidate_trial_decisions') if field in descriptor},
                     **run_config}
    if 'workers' in descriptor:
        actual_config['workers'] = descriptor['workers']
    if not actual_config.get('model_config') and isinstance(actual_config.get('model_overrides'),dict):
        actual_config['model_config']={**actual_config['model_overrides'],
            'attention':attention,'action_dim':actual_config.get('policy_action_count')}
    config = {**DEFAULT_CONFIG, **actual_config}
    phase = _first(progress.get('stage'), progress.get('phase'), progress.get('status'),
                   descriptor.get('phase'), descriptor.get('status'), study.get('status'), 'waiting')
    terminal={'failed','finished','operator_or_budget_stop','stopped','pruned'}
    phase=next((value for value in (study.get('status'),descriptor.get('status'),progress.get('status'))
                if value in terminal),phase)
    failures=[]
    for path in (output/'failure.json',display_run/'failure.json' if display_run else None):
        if path is not None and path.is_file():
            summary=failure_summary(_read(path,errors,f'{key}失败记录'))
            if summary:failures.append({'summary':summary,'path':str(path)})
    attention = _first(progress.get('attention'), run_config.get('attention'), attention)
    development = []
    if training_run and training_run.is_dir():
        for path in sorted(training_run.glob('development_*/report.json')):
            document = _read(path, errors, f'{key}开发报告')
            suffix = path.parent.name.removeprefix('development_')
            decisions = int(suffix) if suffix.isdecimal() else document.get('actual_decisions')
            if document and _finite(decisions):
                development.append(_report(document, path, decisions=int(decisions),
                    expected_episodes=config.get('development_episodes')))
    development.sort(key=lambda row: row['decisions'])
    validation = {}
    for kind, value in _dict(study.get('validation20')).items():
        if isinstance(value, dict):
            validation[kind] = _report(value, output / f'validation20_{kind}' / 'report.json',
                name=kind, expected_episodes=config.get('validation_episodes_per_policy'))
    for folder in dict.fromkeys(path for path in (output, training_run) if path):
        for pattern in ('validation*/report.json', 'final_evaluation*/report.json'):
            for path in sorted(folder.glob(pattern)):
                name = path.parent.name.removeprefix('validation20_')
                document = _read(path, errors, f'{key}终局验证报告')
                if document:
                    validation[name] = _report(document, path, name=name,
                        expected_episodes=config.get('validation_episodes_per_policy'))
    timing = _dict(progress.get('timing'))
    decisions = progress.get('actual_decisions')
    collect = timing.get('collect_seconds')
    sample_rate = _first(progress.get('sampling_decisions_per_second'),
                         _dict(progress.get('latest_rollout_timing')).get('sampling_decisions_per_second'),
                         timing.get('sampling_decisions_per_second'))
    # Producer collect_seconds excludes its active unfinished rollout. Do not
    # divide current decisions by stale completed-rollout time while collecting.
    if sample_rate is None and phase in ('updating', 'development', 'finished', 'completed'):
        if _finite(decisions) and _finite(collect) and collect > 0:
            sample_rate = decisions / collect
    teacher = {**_dict(descriptor.get('teacher')), **_dict(progress.get('teacher'))}
    teacher_result = _dict(study.get('teacher_pretraining_result'))
    teacher_samples = _first(progress.get('teacher_samples'), teacher.get('samples'), teacher_result.get('teacher_samples'))
    pretrain_updates = _first(progress.get('teacher_pretrain_updates'), teacher.get('teacher_pretrain_updates'),
                             progress.get('pretrain_updates'), teacher.get('pretrain_updates'), teacher.get('updates'),
                             teacher_result.get('BC_adam_minibatch_steps'))
    teacher_enabled = bool(descriptor.get('teacher_pretraining') or teacher or teacher_result
        or phase in ('teacher_collection', 'teacher_pretraining') or key == 'C')
    return {'key': key, 'output': str(output), 'current_run': str(current) if current else None,
        'training_run': str(training_run) if training_run else None,
        'display_run': str(display_run) if display_run else None, 'phase': phase,
        'label': descriptor.get('label') or (('教师预训练＋' if teacher_enabled else '')
            + LABELS.get(attention, str(attention or '模型待上报'))),
        'attention': attention, 'progress': progress, 'study': study,
        'configuration': {key: config.get(key) for key in (*DEFAULT_CONFIG, 'workers', 'seed', 'input_schema',
            'headless','collector_mode','full_story_fraction','native_frame_horizon','optimization_trials','teacher_pretraining',
            'curriculum_manifest','curriculum_protocol','curriculum_options','training_recipe','reward_schema',
            'training_mode','learning_revision','first_hit_terminal','continues_allowed',
            'policy_action_count', 'candidate_trial_decisions', 'parameters', 'model_config')},
        'diagnostic_protocol':{name:config.get(name) for name in ('training_mode','episode_protocol',
            'learning_revision','learning_migration','training_recipe','reward_schema','input_schema',
            'curriculum_protocol','native_frame_horizon','first_hit_terminal','continues_allowed',
            'source_runtime','source_manifest_sha256','candidate_decision_offset')},
        'budget_is_default': 'trial_decisions' not in actual_config,
        'parameter_counts': _dict(_first(progress.get('parameter_counts'), run_config.get('parameter_counts'),
                                       descriptor.get('parameter_counts'))),
        'workers': _first(config.get('workers'),progress.get('workers')),
        'active_workers':_first(progress.get('workers'),config.get('workers')),
        'sampling_decisions_per_second': sample_rate,
        'teacher': {'enabled': teacher_enabled, 'samples': teacher_samples, 'pretrain_updates': pretrain_updates,
                    'result':teacher_result},
        'development': development, 'latest_development': _dict(progress.get('latest_development')),
        'validation': list(validation.values()), 'failures':failures,
        'stop_requested': (output / 'STOP').is_file(), 'errors': errors}


def load_comparison_snapshot(comparison_output, last_runs=None):
    """Read small local JSON only; never create directories, load models or UI.

    Manifest/status schema: thrl-native-comparison-v1 with config and branches
    {A:{output,attention,workers,...}, B:{...}, C:{...}}. Paths are relative to
    comparison root. Any number of named branches is supported. Last runs are
    optional branch->training-run paths retained after current_run becomes null.
    """
    supplied = Path(comparison_output).resolve()
    explicit = supplied if supplied.suffix.lower() == '.json' else None
    root = supplied.parent if explicit else supplied
    errors, manifest = [], {}
    paths = [explicit] if explicit else [root / 'manifest.json', root / 'comparison_manifest.json']
    if root / 'comparison_status.json' not in paths:
        paths.append(root / 'comparison_status.json')
    documents = [_read(path, errors, '对照状态') for path in paths]
    branches, config = {}, {}
    for document in documents:
        if document:
            manifest.update({key: value for key, value in document.items() if key not in ('branches', 'config')})
            config.update(_dict(document.get('config')))
            for key, descriptor in _dict(document.get('branches')).items():
                if isinstance(descriptor, str):
                    descriptor = {'output': descriptor}
                if isinstance(descriptor, dict):
                    branches[str(key)] = {**branches.get(str(key), {}), **descriptor}
    if not branches:
        branches = DEFAULT_BRANCHES
    observed = {}
    for key, descriptor in branches.items():
        observed[key] = _branch_snapshot(key, descriptor, root, config, (last_runs or {}).get(key))
    return {'schema': 'thrl-native-comparison-monitor-v1', 'observed_utc': datetime.now(timezone.utc).isoformat(),
        'output': str(root), 'status': manifest.get('status', 'waiting'), 'config': {**DEFAULT_CONFIG, **config},
        'hardware': _dict(manifest.get('hardware')), 'branches': observed, 'errors': errors}


def request_stop(comparison_output, branches=None):
    """Only explicit button/API calls write each selected branch's STOP file.

    Returns per-branch receipts, including failures; never kills processes or
    creates missing output directories. Repeated requests are idempotent.
    """
    snapshot = load_comparison_snapshot(comparison_output)
    keys = list(snapshot['branches']) if branches is None else list(branches)
    result = {}
    for key in keys:
        if key not in snapshot['branches']:
            result[key] = {'requested': False, 'error': '未配置此分支'}
            continue
        output = Path(snapshot['branches'][key]['output'])
        try:
            if not output.is_dir():
                raise FileNotFoundError('分支输出目录尚不存在')
            with (output / 'STOP').open('x', encoding='utf-8') as stream:
                json.dump({'requested_utc': datetime.now(timezone.utc).isoformat(), 'branch': key,
                    'request': 'stop_current_and_later_trials', 'source': 'native_comparison_monitor'},
                    stream, ensure_ascii=False)
            result[key] = {'requested': True, 'already_requested': False, 'path': str(output / 'STOP')}
        except FileExistsError:
            result[key] = {'requested': True, 'already_requested': True, 'path': str(output / 'STOP')}
        except OSError as error:
            result[key] = {'requested': False, 'error': str(error)}
    return result


def _triple(value):
    values = value if isinstance(value, (list, tuple)) else []
    return ' / '.join(number(values[i] if i < len(values) else None, 4) for i in range(3))


class ComparisonWindow:
    def __init__(self, comparison_output, refresh_seconds=2., active_study=None,ui_state=None,control_root=None):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.output, self.refresh_seconds = Path(comparison_output), refresh_seconds
        self.active_study=active_study
        self.ui_state=ui_state
        if control_root:
            from fly_touhou.thrl_control import ControlCenter
            self.controller=ControlCenter(control_root)
        else:self.controller=None
        self.action_messages=queue.Queue();self.action_busy=False
        self.view_selection=('A0','错误日志')
        self.root = tk.Tk()
        self.root.title('东方原作 · 训练控制面板')
        self.root.geometry('1460x940'); self.root.minsize(1040, 680)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        style = ttk.Style(self.root)
        if 'vista' in style.theme_names(): style.theme_use('vista')
        style.configure('Title.TLabel', font=('Microsoft YaHei UI', 16, 'bold'))
        frame = ttk.Frame(self.root, padding=12); frame.pack(fill='both', expand=True)
        self.title = tk.StringVar(value='等待训练对照状态')
        heading=ttk.Frame(frame);heading.pack(fill='x')
        ttk.Label(heading, textvariable=self.title, style='Title.TLabel').pack(side='left')
        ttk.Button(heading,text='TensorBoard 曲线',command=self.open_tensorboard).pack(side='right')
        self.hardware = tk.StringVar(value=hardware_text({}))
        ttk.Label(frame, textvariable=self.hardware).pack(anchor='w', pady=(5, 0))
        self.engineering_note=tk.StringVar()
        self.notebook=ttk.Notebook(frame);self.notebook.pack(fill='both',expand=True,pady=(8,0))
        live=ttk.Frame(self.notebook,padding=6);self.notebook.add(live,text='实时训练与评估')
        if self.controller:self._make_control_tab()
        from fly_touhou.thrl_demo_ui import DemoTab
        self.demo_tab=DemoTab(self.notebook,Path(__file__).resolve().parents[1]/'artifacts/engineering/real-model-demonstration-v1')
        self._make_log_tab()
        frame=live
        columns=('group','input','phase','decisions','adam','demos','bc','development')
        self.overview=ttk.Treeview(frame,columns=columns,show='headings',height=4)
        headings=('组别','输入方案','当前阶段','PPO 决策','PPO Adam','教师示范','BC Adam','最近完整开发评估')
        widths=(90,180,180,135,100,120,90,240)
        for name,label,width in zip(columns,headings,widths):
            self.overview.heading(name,text=label);self.overview.column(name,width=width,minwidth=60)
        self.overview.tag_configure('failed',foreground='#b32125')
        self.overview.pack(fill='x',pady=(6,3))
        ttk.Label(frame, text='教师预训练、PPO 训练、开发评估、终局验证分开显示；关闭窗口不会停止训练。').pack(anchor='w', pady=(5, 8))
        area = ttk.Frame(frame); area.pack(fill='both', expand=True)
        area.rowconfigure(0, weight=1); area.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(area, highlightthickness=0)
        sy = ttk.Scrollbar(area, orient='vertical', command=self.canvas.yview)
        sx = ttk.Scrollbar(area, orient='horizontal', command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        self.canvas.grid(row=0, column=0, sticky='nsew'); sy.grid(row=0, column=1, sticky='ns'); sx.grid(row=1, column=0, sticky='ew')
        self.cards = ttk.Frame(self.canvas); self.canvas.create_window(0, 0, window=self.cards, anchor='nw')
        self.cards.bind('<Configure>', lambda _: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        footer = ttk.Frame(frame); footer.pack(fill='x', pady=(8, 0))
        self.footer = tk.StringVar(value='每2秒读取本地状态；停止按钮写各分支 STOP，等待训练器安全收尾。')
        ttk.Label(footer, textvariable=self.footer, wraplength=1120).pack(side='left')
        ttk.Button(footer, text='停止全部分支', command=lambda: self.stop(None)).pack(side='right')
        self.closed, self.messages = threading.Event(), queue.Queue(maxsize=1)
        self.snapshot, self.widgets = {}, {}
        threading.Thread(target=self.read_loop, name='comparison-json-reader', daemon=True).start()
        self.root.after(100, self.poll)

    def _make_control_tab(self):
        tk,ttk=self.tk,self.ttk
        from fly_touhou.thrl_control import ALL_GROUPS
        tab=ttk.Frame(self.notebook,padding=18);self.notebook.add(tab,text='训练设置与启动')
        self.control_tab=tab;self.settings_vars={};self.group_vars={}
        settings=self.controller.settings()
        fields=(('workers',f'每个模型的原作实例数（1–{MAX_NATIVE_WORKERS}）'),
            ('decisions','每个候选的 PPO 总决策目标'),('candidates','Optuna 候选数（包含首个固定参数对照）'),
            ('development_interval','开发评估间隔（决策）'),('development_episodes','每次开发评估局数'),
            ('validation_episodes','最终固定模型测试局数'))
        for i,(key,label) in enumerate(fields):
            ttk.Label(tab,text=label).grid(row=i,column=0,sticky='w',pady=7)
            value=tk.StringVar(value=str(settings[key]));self.settings_vars[key]=value
            ttk.Entry(tab,textvariable=value,width=24).grid(row=i,column=1,sticky='w',padx=12)
            value.trace_add('write',lambda *_:self._update_plan())
        self.settings_vars['optuna']=tk.BooleanVar(value=settings['optuna'])
        ttk.Checkbutton(tab,text='启用 Optuna：后续候选自动调整学习率、γ、λ、熵；首候选保持固定参数',
            variable=self.settings_vars['optuna'],command=self._update_plan).grid(row=6,column=0,columnspan=3,sticky='w',pady=9)
        self.plan_text=tk.StringVar()
        ttk.Label(tab,textvariable=self.plan_text,wraplength=1100,justify='left').grid(row=7,column=0,columnspan=4,sticky='w',pady=10)
        groups=ttk.LabelFrame(tab,text='选择实验组',padding=12);groups.grid(row=8,column=0,columnspan=4,sticky='ew',pady=8)
        names=GROUP_DISPLAY_NAMES
        available=tuple(self.controller.config['template_configs'])
        for i,key in enumerate(available):
            variable=tk.BooleanVar(value=key in ALL_GROUPS);self.group_vars[key]=variable
            ttk.Checkbutton(groups,text=names.get(key,key),variable=variable,command=self._update_plan).grid(row=i//2,column=i%2,sticky='w',padx=12,pady=7)
        actions=ttk.Frame(tab);actions.grid(row=9,column=0,columnspan=4,sticky='w',pady=12)
        self.action_buttons=[]
        for title,action in (('保存设置',self.save_control_settings),('从头启动所选组',lambda:self.launch_groups()),
                ('继续所选组的主对照',lambda:self.launch_groups(mode='resume')),
                ('停止所选组',lambda:self.stop([key for key,var in self.group_vars.items() if var.get()]))):
            button=ttk.Button(actions,text=title,command=action);button.pack(side='left',padx=(0,12));self.action_buttons.append(button)
        self.action_status=tk.StringVar(value='设置只用于下一次启动；已有训练的实例数和预算保持原值。')
        ttk.Label(tab,textvariable=self.action_status,wraplength=1150,foreground='#215b8c').grid(row=10,column=0,columnspan=4,sticky='w',pady=8)
        ttk.Label(tab,wraplength=1150,justify='left',text=(
            '从头启动：创建新目录，保留历史结果；有教师组重新采集当前输入的示范。\n'
            '继续主对照：恢复 trial000 的权重、Adam 和抽样 RNG，使用新的总决策目标；保持原实例数。'
            '新连打方案每局从第一面开局；GRU每局重置隐藏状态，MLP无隐藏状态。未完成的教师采集重新开始。\n'
            '旧课程作业仅在存在明确迁移记录时可继续为新方案；历史成绩保留原协议标记。\n'
            '每组共用一个 CUDA 策略，组间独立。允许自主 Bomb、用完残机才结束、禁止续关；不限制1024帧。\n'
            '停止会等待采样/保存收尾；关闭面板不会停止训练。新训练不设小时上限，达到决策目标及评估配额后结束。'
        )).grid(row=11,column=0,columnspan=4,sticky='w',pady=10)
        ttk.Label(tab,textvariable=self.engineering_note,wraplength=1150).grid(
            row=12,column=0,columnspan=4,sticky='w',pady=8)
        self._update_plan()

    def _control_values(self):
        return {key:(value.get() if key=='optuna' else int(value.get().replace(',','').strip()))
                for key,value in self.settings_vars.items()}

    def _update_plan(self):
        if not hasattr(self,'plan_text'):return
        try:
            from fly_touhou.thrl_control import settings_plan
            plan=(self.controller.plan(self._control_values()) if self.controller else settings_plan(self._control_values()))
            groups=sum(v.get() for v in self.group_vars.values())
            self.plan_text.set(f'每个候选实际目标：{plan["target_decisions"]:,} 次决策（按完整 {plan["rollout"]:,} 次采样对齐）\n'
                f'所选 {groups} 组 × 每组 {plan["settings"]["workers"]} 实例 = {groups*plan["settings"]["workers"]} 个训练实例；'
                f'每组 {plan["candidates"]} 个候选，最大 PPO 总量 {groups*plan["max_PPO_decisions_per_group"]:,} 次。\n'
                f'每候选 Adam {"上限" if plan.get("Adam_estimate_is_upper_bound") else "约"} {plan["expected_Adam_per_candidate"]:,} 次；每组仅地图采样缓冲约 '
                f'{plan["rollout_ram_gib_per_group"]:.2f} GiB。教师采集、开发和终测另计。')
        except (ValueError,TypeError,KeyError):self.plan_text.set('请填写有效的整数；默认每组8实例、每候选5,000,000次决策。')

    def _submit_action(self,function):
        if self.action_busy:return
        self.action_busy=True
        for button in getattr(self,'action_buttons',[]):button.state(['disabled'])
        def work():
            try:self.action_messages.put((True,function()))
            except Exception as error:self.action_messages.put((False,str(error)))
        threading.Thread(target=work,name='local-control-action',daemon=False).start()

    def save_control_settings(self):
        try:values=self._control_values()
        except ValueError:self.action_status.set('设置无效：请输入整数');return
        self._submit_action(lambda:(self.controller.save_settings(values),'设置已保存；已有作业不受影响。')[1])

    def launch_groups(self,keys=None,mode='new'):
        if not self.controller:return
        keys=keys or [key for key,value in self.group_vars.items() if value.get()]
        try:values=self._control_values()
        except ValueError:self.action_status.set('设置无效：请输入整数');return
        self.action_status.set('正在创建独立作业；界面仍会刷新。')
        def launch():
            receipt=self.controller.launch(keys,values,mode)
            return ('已提交 '+', '.join(keys)+'；实际目标 '+f'{receipt["plan"]["target_decisions"]:,}'+
                ' 次/候选。启动进度和错误见实时页与日志页。目录：'+receipt['output'])
        self._submit_action(launch)

    def _make_log_tab(self):
        tk,ttk=self.tk,self.ttk
        tab=ttk.Frame(self.notebook,padding=10);self.notebook.add(tab,text='日志、配置与报告')
        bar=ttk.Frame(tab);bar.pack(fill='x')
        self.log_group=tk.StringVar(value='');self.log_kind=tk.StringVar(value='错误日志')
        group=ttk.Combobox(bar,textvariable=self.log_group,values=(),state='readonly',width=8)
        self.log_group_selector=group
        group.pack(side='left',padx=(0,12))
        kind=ttk.Combobox(bar,textvariable=self.log_kind,values=('错误日志','运行日志','当前配置','开发报告','最终报告'),state='readonly',width=18)
        kind.pack(side='left')
        for control in (group,kind):control.bind('<<ComboboxSelected>>',lambda _:self.select_log())
        ttk.Button(bar,text='打开该组结果文件夹',command=self.open_selected_folder).pack(side='right')
        ttk.Label(tab,text='每2秒刷新；日志显示末尾64 KiB。开发评估和最终测试分别列出。').pack(anchor='w',pady=7)
        body=ttk.Frame(tab);body.pack(fill='both',expand=True)
        self.log_text=tk.Text(body,font=('Consolas',10),wrap='none',state='disabled')
        sy=ttk.Scrollbar(body,orient='vertical',command=self.log_text.yview)
        sx=ttk.Scrollbar(body,orient='horizontal',command=self.log_text.xview)
        self.log_text.configure(yscrollcommand=sy.set,xscrollcommand=sx.set)
        self.log_text.grid(row=0,column=0,sticky='nsew');sy.grid(row=0,column=1,sticky='ns');sx.grid(row=1,column=0,sticky='ew')
        body.rowconfigure(0,weight=1);body.columnconfigure(0,weight=1)

    def select_log(self):self.view_selection=(self.log_group.get(),self.log_kind.get())

    def open_selected_folder(self):
        import os
        branch=self.snapshot.get('branches',{}).get(self.log_group.get())
        if branch and Path(branch['output']).is_dir():os.startfile(branch['output'])

    def open_tensorboard(self):
        import os,subprocess
        try:
            command=tensorboard_viewer_command()
            pointer=read_shared_json(Path(__file__).resolve().parents[1]/'artifacts/engineering/active_tensorboard.json')
            environment=dict(os.environ)
            if pointer.get('temporary_directory'):
                environment.update({key:pointer['temporary_directory'] for key in ('TEMP','TMP','TMPDIR')})
            subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                env=environment,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.footer.set('已打开TensorBoard曲线；曲线服务独立运行，训练继续。')
        except (OSError,ValueError,KeyError) as error:
            self.footer.set('TensorBoard入口暂不可用：'+str(error))

    def _log_view(self,snapshot):
        from fly_touhou.thrl_dodge_diagnostics import _shared_binary
        branches=running_branches(snapshot)
        key,kind=self.view_selection
        if key not in branches:key=next(iter(branches),None)
        branch=branches.get(key)
        if not branch:return '当前没有正在训练的项目。'
        output=Path(branch['output']);job=_dict(branch.get('control',{}).get('job'))
        control=Path(job['control']) if job else None
        run=Path(branch['display_run']) if branch.get('display_run') else output
        if kind=='错误日志':paths=[output.parent/f'{key}_stderr.log',output/'failure.json',run/'failure.json']+([control/'action_failure.json',control/'stderr.log'] if control else [])
        elif kind=='运行日志':paths=[output.parent/f'{key}_stdout.log']+([control/'stdout.log'] if control else [])
        elif kind=='当前配置':paths=[output/'study_config.json',run/'config.json']
        elif kind=='开发报告':paths=[Path(row['path']) for row in branch['development'][-3:]]
        else:paths=[output/'study_result.json']+[Path(row['path']) for row in branch['validation']]
        texts=[]
        for path in paths:
            if not path.is_file():continue
            try:
                with _shared_binary(path) as stream:
                    stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-65536));value=stream.read().decode('utf-8','replace')
                if value.strip():texts.append(str(path)+'\n'+value)
            except OSError:pass
        return '\n\n'.join(texts) or '尚无记录。'

    def read_loop(self):
        last_runs = {}
        hardware_cache={}
        dodge_readers={}
        while not self.closed.is_set():
            try:
                snapshot = (active_study_snapshot(self.active_study,last_runs) if self.active_study
                            else load_comparison_snapshot(self.output, last_runs))
                if self.controller:self.controller.decorate(snapshot)
                live_snapshot={**snapshot,'branches':running_branches(snapshot)}
                attach_dodge_diagnostics(live_snapshot,dodge_readers)
                attach_passive_diagnostics(live_snapshot)
                refresh_idle_hardware(snapshot,hardware_cache)
                snapshot['log_view']=self._log_view(snapshot)
                last_runs = {key: row['training_run'] for key, row in snapshot['branches'].items()}
            except (OSError, ValueError) as error:
                snapshot = {'branches': {}, 'errors': [f'状态暂未就绪（{type(error).__name__}）']}
            try: self.messages.put_nowait(snapshot)
            except queue.Full:
                try: self.messages.get_nowait()
                except queue.Empty: pass
                self.messages.put_nowait(snapshot)
            self.closed.wait(self.refresh_seconds)

    def poll(self):
        try:
            ok,message=self.action_messages.get_nowait();self.action_busy=False
            if hasattr(self,'action_status'):self.action_status.set(('' if ok else '操作未执行：')+message)
            for button in getattr(self,'action_buttons',[]):button.state(['!disabled'])
        except queue.Empty:pass
        try:
            self.snapshot = self.messages.get_nowait(); self.render()
        except queue.Empty: pass
        if not self.closed.is_set(): self.root.after(150, self.poll)

    def _make_cards(self, keys):
        for widget in self.cards.winfo_children(): widget.destroy()
        self.widgets = {}
        for column in range(2):self.cards.columnconfigure(column,weight=0)
        if not keys:
            self.ttk.Label(self.cards,text='当前没有正在训练的项目。请到“训练设置与启动”选择实验组。',
                padding=20).grid(row=0,column=0,sticky='w')
            return
        columns=2 if len(keys)>3 else max(1,len(keys))
        for index, key in enumerate(keys):
            row,column=divmod(index,columns)
            card = self.ttk.LabelFrame(self.cards, text=str(key), padding=10)
            card.grid(row=row, column=column, sticky='nsew', padx=5, pady=3)
            self.cards.columnconfigure(column, weight=1)
            parts = {'card': card}
            for name, title in (('model', '模型与世界'), ('teacher', '教师示范预训练'),
                                ('curriculum','训练路线与回合终点'),
                                ('worlds', '当前原作实例'),
                                ('train', 'PPO 训练（训练回合）'), ('metrics', '最近采样 / GPU 更新指标'),
                                ('dodge','躲避能力诊断（只读训练日志）'),
                                ('dev', '开发评估（独立新局）'), ('validation', '终局固定模型验证')):
                part = self.ttk.LabelFrame(card, text=title, padding=8); part.pack(fill='x', pady=(0, 7))
                variable = self.tk.StringVar(value='等待记录')
                self.ttk.Label(part, textvariable=variable, justify='left', wraplength=620).pack(anchor='w', fill='x')
                parts[name] = variable
                if name == 'train':
                    parts['bar'] = self.ttk.Progressbar(part, length=399, mode='determinate')
                    parts['bar'].pack(fill='x', pady=(7, 0))
                if name == 'dev':
                    chart = self.tk.Canvas(part, width=620, height=165, background='white', highlightthickness=0)
                    chart.pack(fill='x', pady=(8, 0)); parts['chart'] = chart
                    chart.bind('<Configure>', lambda _, k=key: self.draw_chart(k))
            actions=self.ttk.Frame(card);actions.pack(fill='x',before=card.winfo_children()[0],pady=(0,7))
            parts['stop'] = self.ttk.Button(actions, text=f'停止 {key}', command=lambda k=key: self.stop([k]))
            parts['stop'].pack(side='right')
            self.widgets[key] = parts

    def render(self):
        if self.snapshot.get('output'):self.output=Path(self.snapshot['output'])
        branches = running_branches(self.snapshot)
        display_keys=[key for key in ('A0','B0','M0','A1','B1','M1') if key in branches]
        display_keys.extend(key for key in branches if key not in display_keys)
        if display_keys != list(self.widgets): self._make_cards(display_keys)
        self.title.set(' / '.join(branches)+' 原作训练' if branches else '当前没有正在训练的项目')
        if hasattr(self,'log_group_selector'):
            choices=tuple(branches)
            self.log_group_selector.configure(values=choices)
            if self.log_group.get() not in choices:
                self.log_group.set(next(iter(choices),''))
                self.view_selection=(self.log_group.get(),self.log_kind.get())
        self.hardware.set(hardware_text(self.snapshot.get('hardware')))
        self.engineering_note.set(self.snapshot.get('engineering_note',''))
        text=self.snapshot.get('log_view','')
        if getattr(self,'last_log_view',None)!=text:
            self.last_log_view=text;self.log_text.configure(state='normal');self.log_text.delete('1.0','end')
            self.log_text.insert('1.0',text);self.log_text.configure(state='disabled')
        self.overview.delete(*self.overview.get_children())
        for key,branch in branches.items():
            p=branch['progress'];teacher=branch['teacher'];cfg=branch['configuration']
            schema=p.get('input_schema') or cfg.get('input_schema')
            input_text={'original273-v1':'273＋地图','semantic-separated-v1':'20＋96对象＋地图',
                'original273-timers-v2':'279（含计时）＋地图','semantic-separated-timers-v2':'25（含计时）＋96对象＋地图',
                'semantic-separated-timers-object-priority-v3':'25＋96对象＋地图 · 弹幕名额修正v3',
                'semantic-separated-timers-hp-independent-contact-v4':'25＋96对象＋地图 · 接触判定修正v4',
                'semantic-separated-observed-laser-segments-v5':'25＋96×29对象＋地图 · 激光运动v5'}.get(schema,branch['attention'])
            dev=branch.get('latest_development',{})
            latest=[row for row in branch['development'] if row['complete']]
            if latest:dev=latest[-1]
            dev_text=(f'{number(dev.get("clears"),integer=True)}/{number(dev.get("episodes"),integer=True)}通关 · '
                      f'平均最远{number(dev.get("mean_max_stage"),2)}面') if dev else '待评估'
            self.overview.insert('', 'end',values=(key,input_text,LABELS.get(branch['phase'],branch['phase']),
                number(p.get('candidate_decisions',p.get('actual_decisions')),integer=True),
                number(p.get('adam_minibatch_steps'),integer=True),number(teacher.get('samples'),integer=True),
                number(teacher.get('pretrain_updates'),integer=True),dev_text),tags=(branch['phase'],))
        all_errors = list(self.snapshot.get('errors', []))
        for key, branch in branches.items():
            w, p, cfg = self.widgets[key], branch['progress'], branch['configuration']
            w['card'].configure(text=f'{key} · {branch["label"]} · {LABELS.get(branch["phase"], branch["phase"])}')
            if branch['phase']=='ready' and self.controller:
                from fly_touhou.thrl_control import settings_plan
                plan=self.controller.plan(self.controller.settings())
                cfg={**cfg,'workers':plan['settings']['workers'],'trial_decisions':plan['target_decisions'],
                    'development_decisions':plan['development_decisions'],
                    'development_episodes':plan['settings']['development_episodes'],
                    'validation_episodes_per_policy':plan['settings']['validation_episodes'],
                    'optimization_trials':plan['candidates']}
                branch={**branch,'configuration':cfg,'workers':plan['settings']['workers'],'active_workers':None}
            model, params, counts = _dict(cfg.get('model_config')), _dict(cfg.get('parameters')), branch['parameter_counts']
            gpu = p.get('gpu') or branch['study'].get('gpu') or '—'
            if isinstance(gpu, dict): gpu = gpu.get('name', '—')
            workers = p.get('current_environments')
            recorded = len(workers) if isinstance(workers, list) else None
            schema=p.get('input_schema') or cfg.get('input_schema')
            input_text=({'original273-v1':'原 273 维＋地图；原对象独立注意力',
                         'semantic-separated-v1':'全局20＋对象96×21＋地图；三路独立编码',
                         'original273-timers-v2':'原273维＋6项状态/计时＋地图；对象独立注意力',
                         'semantic-separated-timers-v2':'全局25（含状态/计时）＋对象96×21＋地图；三路独立编码',
                         'semantic-separated-timers-object-priority-v3':'全局25＋对象96×21＋地图；保底后先补非道具，防止额外道具挤占弹幕',
                         'semantic-separated-timers-hp-independent-contact-v4':'全局25＋对象96×21＋地图；按原作接触标志保留零血量实体',
                         'semantic-separated-observed-laser-segments-v5':'全局25＋对象96×29＋地图；完整激光段、实测两端运动与未知标志；三路独立编码'}.get(schema,'输入待上报'))
            core_text=(f'MLP：384→{model.get("mlp_hidden_size",768)}→{model.get("gru_hidden_size",256)}，无循环状态'
                if model.get('temporal_core','gru')=='mlp' else f'GRU：{number(model.get("gru_hidden_size"),integer=True)}')
            w['model'].set(f'PPO 实例计划 {number(branch["workers"], integer=True)}；阶段记录 {number(branch["active_workers"],integer=True)}；最近实例快照 {number(recorded, integer=True)}\n'
                f'参数总量：{number(counts.get("total"), integer=True)}　{core_text}\n'
                f'注意力：对象 {number(model.get("object_attention_layers"),integer=True)}层 / {number(model.get("object_embed_dim"),integer=True)}维 / {number(model.get("object_attention_heads"),integer=True)}头；空间 {number(model.get("spatial_attention_layers"),integer=True)}层\n'
                f'动作：{number(model.get("action_dim", cfg.get("policy_action_count")), integer=True)}　设备：{p.get("device", "—")} / {gpu}\n'
                f'当前学习率 {number(params.get("learning_rate"), 6)}　γ {number(params.get("gamma"), 4)}　λ {number(params.get("gae_lambda"), 3)}\n'
                f'随机采样温度 {number(p.get("action_temperature", params.get("action_temperature", 1.)), 3)}；数值越低，动作越集中\n'
                f'候选数 {number(cfg.get("optimization_trials"),integer=True)}；当前 {Path(branch["display_run"]).name if branch.get("display_run") else "待启动"}\n'
                f'输入：{input_text}\n'
                f'{curriculum_text(cfg)}\n'
                f'当前记录协议：{cfg.get("training_recipe","旧版")}；下次新训练：{branch.get("next_training_recipe","—")}\n'
                f'下次新训练输入：{"弹幕名额修正v3" if branch.get("next_input_schema")=="semantic-separated-timers-object-priority-v3" else branch.get("next_input_schema","—")}\n'
                f'下次启动路线：{"单策略全程连打" if branch.get("next_training_mode")=="full-story-v1" else "按对应模板"}\n'
                f'{branch.get("resume_blocked_reason") or ""}')
            if 'curriculum' in w:
                w['curriculum'].set(('单策略全程连打已配置，尚未启动。\n'+curriculum_progress_text(p,cfg))
                    if branch['phase']=='ready' and is_full_story(cfg) else
                    ('课程已配置，尚未启动：第一面道中前半，起点从段末300帧范围开始。\n'
                    f'{key}采用前馈MLP，无历史预热或隐藏状态；点击“新训练 {key}”才会创建作业。')
                    if branch['phase']=='ready' and model.get('temporal_core')=='mlp' else curriculum_progress_text(p,cfg))
            teacher = branch['teacher']
            bc=_dict(cfg.get('teacher_pretraining'))
            bc_target=(bc.get('samples',0)*bc.get('epochs',0)//(bc.get('seq_len',16)*bc.get('seqs_per_batch',4))) if bc else None
            w['teacher'].set((f'已采示范 {number(teacher["samples"], integer=True)} / {number(bc.get("samples"),integer=True)} 条\n'
                f'预训练更新 {number(teacher["pretrain_updates"], integer=True)} / 上限 {number(bc_target,integer=True)} 次；与 PPO Adam 分开\n'
                f'{teacher_quality_text(teacher,p)}')
                if teacher['enabled'] else '本分支未配置教师示范预训练')
            stopped=branch['phase'] in ('failed','finished','operator_or_budget_stop','stopped','pruned')
            w['worlds'].set(('作业已停止；以下为最后快照\n' if stopped and workers else '')+'\n'.join(f'{i+1}: {number(row.get("stage"),integer=True)}面 @ {number(row.get("stage_frame"),integer=True)}帧 · '
                f'残{number(row.get("lives"),integer=True)} / Bomb {number(row.get("bombs"),integer=True)} / P {number(row.get("power"),integer=True)}'
                for i,row in enumerate(workers or [])) or '暂无当前实例记录')
            decisions = p.get('actual_decisions'); target = _first(p.get('target_decisions'), cfg.get('trial_decisions'))
            timing = _dict(p.get('timing'))
            plan_tag = '（默认计划，待配置确认）' if branch['budget_is_default'] else ''
            w['train'].set(f'本次运行决策 {number(decisions, integer=True)} / {number(target, integer=True)} {plan_tag}\n'
                f'继承决策 {number(p.get("inherited_decisions"), integer=True)}；累计 {number(p.get("lifetime_decisions"), integer=True)}\n'
                f'本候选累计 {number(p.get("candidate_decisions"), integer=True)} / {number(candidate_target(branch), integer=True)}\n'
                f'本次运行 PPO Adam {number(p.get("adam_minibatch_steps"), integer=True)}　更新 {number(p.get("rollout_updates"), integer=True)}\n'
                f'继承 Adam {number(p.get("inherited_adam_minibatch_steps"), integer=True)}；累计 {number(p.get("lifetime_adam_minibatch_steps"), integer=True)}\n'
                f'训练终局 {number(p.get("episodes"), integer=True)}（{"均从完整开局" if is_full_story(cfg) else "历史起点方案"}）；通关 {number(p.get("clear_episodes"), integer=True)}\n'
                f'本组总均速 {number(p.get("decisions_per_second"), 1)}；最近采样＋更新 {number(_dict(p.get("latest_rollout_timing")).get("sampling_and_update_decisions_per_second"), 1)} 决策/秒\n'
                f'最近仅采样 {number(branch["sampling_decisions_per_second"], 1)} 决策/秒（均为本组全部实例合计）\n'
                f'累计采样 {number(timing.get("collect_seconds"), 1)}s / GPU更新 {number(timing.get("learn_seconds"), 1)}s\n'
                f'评估 {number(timing.get("evaluation_seconds"), 1)}s / 全程 {number(p.get("wall_seconds"), 1)}s')
            w['bar']['value'] = max(0, min(100, decisions / target * 100)) if _finite(decisions) and _finite(target) and target > 0 else 0
            metrics = _dict(p.get('metrics'))
            w['metrics'].set('三头奖励：生存 / 战斗 / 资源\n'+_triple(p.get('reward_components_recent'))+'\n'
                f'策略损失 {number(metrics.get("policy_loss"), 5)}　总损失 {number(metrics.get("total_loss"), 5)}\n'
                '价值损失 '+_triple(metrics.get('value_loss'))+'\n'
                f'熵 {number(metrics.get("entropy"), 4)}　系数 {number(metrics.get("entropy_coeff"), 5)}　采样KL {number(metrics.get("approx_kl"), 6)}\n'
                f'更新后全批最大KL {number(metrics.get("max_post_epoch_kl"),5)}；完成 {number(metrics.get("epochs_completed"),integer=True)}/{number(metrics.get("epochs_planned"),integer=True)} 遍\n'
                '价值解释率 '+_triple(metrics.get('explained_variance_before'))+'\n'
                f'{rescue_learning_text(metrics)}\n'
                f'{movement_learning_text(metrics,p.get("recent_episode_summary"))}\n'
                f'{behavior_diagnostics_text(_dict(p.get("recent_episode_summary")))}')
            w['dodge'].set(dodge_diagnostics_text(branch.get('dodge_diagnostics'),branch)+'\n\n'
                +passive_diagnostics_text(branch.get('passive_diagnostics')))
            reports = [row for row in branch['development'] if row['complete']]
            latest = reports[-1] if reports else branch['latest_development']
            evaluation = _dict(p.get('evaluation_progress'))
            active_dev = branch['phase'] == 'development'
            active = (f'当前开发节点：{number(evaluation.get("completed"), integer=True)} / {number(evaluation.get("target"), integer=True)} 完整局\n') if active_dev else ''
            nodes = cfg.get('development_decisions') or []
            w['dev'].set(active+f'计划每节点 {number(cfg.get("development_episodes"), integer=True)} 局；完成 {len(reports)} / {len(nodes)} 节点\n'
                f'最近节点 {number(latest.get("decisions"), integer=True)}：通关 {number(latest.get("clears"), integer=True)} / {number(latest.get("episodes"), integer=True)}\n'
                f'开发分 {number(latest.get("score"), 4)}　平均最远面 {number(latest.get("mean_max_stage"), 2)}\n'
                f'平均原作帧 {number(latest.get("mean_native_frames"), 1)}')
            lines = [f'计划每配置 {number(cfg.get("validation_episodes_per_policy"), integer=True)} 个完整局；不计入开发曲线']
            if branch['phase'] in ('validation20', 'validation'):
                lines.append(f'当前完成 {number(evaluation.get("completed"), integer=True)} / {number(evaluation.get("target"), integer=True)} 局')
            for row in branch['validation']:
                name={'primary':'固定参数主对照','selected_HPO':'Optuna 选中模型',
                    'same_initial':'本组 PPO 起点模型','selected':'开发集选中模型'}.get(row['name'],row['name'])
                lines.append(f'{name}：通关 {number(row["clears"], integer=True)} / {number(row["episodes"], integer=True)}；均面 {number(row["mean_max_stage"], 2)}'+('' if row['complete'] else '（尚未完整）'))
            w['validation'].set('\n'.join(lines))
            w['stop'].state(['disabled'] if branch['stop_requested'] or stopped else ['!disabled'])
            if self.controller:
                running=branch.get('control',{}).get('running',False)
                for field in ('start','resume'):
                    if field in w:w[field].state(['disabled'] if running or self.action_busy else ['!disabled'])
                if (branch['phase']=='ready' or branch.get('resume_blocked_reason')) and 'resume' in w:
                    w['resume'].state(['disabled'])
                w['stop'].state(['!disabled'] if running else ['disabled'])
            self.draw_chart(key); all_errors.extend(branch['errors'])
        failure_lines=[f'{key}：{row["summary"]}' for key,branch in branches.items() for row in branch.get('failures',[])]
        self.footer.set('；'.join(all_errors[:5]+failure_lines)[-1400:] if all_errors or failure_lines else
            f'状态每{self.refresh_seconds:g}秒、硬件每15秒读取 · {self.snapshot.get("observed_utc", "—")} · 关闭窗口不停止训练。当前：{self.output}')
        if self.ui_state:
            import os
            from fly_touhou.atomic_json_io import atomic_write_json
            atomic_write_json(self.ui_state,{'pid':os.getpid(),'title':self.root.title(),
                'visible':bool(self.root.winfo_viewable()),'output':str(self.output),
                'engineering_note':self.engineering_note.get(),'hardware':self.hardware.get(),
                'control_root':str(self.controller.root) if self.controller else None,
                'max_native_workers_per_group':MAX_NATIVE_WORKERS,
                'curriculum_protocol_supported':'th14-progressive-rehearsal-curriculum-v1',
                'training_modes_supported':['full-story-v1','legacy-records'],
                'torch_imported':'torch' in sys.modules,
                'settings':{key:value.get() for key,value in self.settings_vars.items()} if self.controller else None,
                'actions':{key:{field:not parts[field].instate(['disabled']) for field in ('start','resume','stop') if field in parts}
                    for key,parts in self.widgets.items()},
                'cards':{key:{field:value.get() for field,value in parts.items()
                    if field in ('model','worlds','teacher','curriculum','train','metrics','dodge','dev','validation')}
                    for key,parts in self.widgets.items()},'read_errors':all_errors})

    def draw_chart(self, key):
        if key not in self.widgets: return
        canvas = self.widgets[key]['chart']; canvas.delete('all')
        branches = running_branches(self.snapshot)
        all_rows = [row for branch in branches.values() for row in branch['development'] if row['complete']]
        metric = 'score' if any(_finite(row.get('score')) for row in all_rows) else 'mean_max_stage'
        points = [(row['decisions'], row[metric]) for row in branches.get(key, {}).get('development', [])
                  if row['complete'] and _finite(row.get(metric))]
        width, height = max(canvas.winfo_width(), 360), 165
        if not points:
            canvas.create_text(width / 2, height / 2, text='等待完整开发节点\n训练局与终局验证不进入此曲线', fill='#657381', justify='center'); return
        ys = [row[metric] for row in all_rows if _finite(row.get(metric))]
        ymin, ymax = min(ys), max(ys); margin = max((ymax-ymin)*.12, abs(ymax)*.01, .05)
        ymin, ymax = ymin-margin, ymax+margin
        xmax = development_axis_max(branches)
        left, right, top, bottom = 52, width-15, 25, height-33
        canvas.create_text(left, 5, anchor='nw', text='开发分（各分支相同坐标轴）' if metric=='score' else '平均最远面（各分支相同坐标轴）', fill='#263545')
        for i in range(3):
            y = top+i*(bottom-top)/2
            canvas.create_line(left, y, right, y, fill='#e5e9ee')
            canvas.create_text(left-6, y, anchor='e', text=f'{ymax-i*(ymax-ymin)/2:.3g}', fill='#657381')
        positions = []
        for x, y in points:
            px = left+x/xmax*(right-left); py = bottom-(y-ymin)/(ymax-ymin)*(bottom-top)
            positions += [px, py]; canvas.create_oval(px-3, py-3, px+3, py+3, fill='#2867ad', outline='')
            canvas.create_text(px, bottom+12, text=f'{x/1000:g}k', fill='#657381')
        if len(points)>1: canvas.create_line(*positions, fill='#2867ad', width=2)

    def stop(self, branches):
        try:results = self.controller.stop(branches) if self.controller else request_stop(self.output, branches)
        except (OSError,ValueError,RuntimeError) as error:
            self.footer.set('停止尚未提交：'+str(error));return
        self.footer.set('；'.join(f'{key}：'+('已请求停止，等待安全收尾' if row['requested'] else row['error'])
                                  for key, row in results.items()))

    def close(self):
        self.closed.set()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--comparison-output', type=Path, help='对照输出目录或 manifest JSON')
    source.add_argument('--active-study',type=Path,help='自动跟随当前作业指针')
    parser.add_argument('--refresh-seconds', type=float, default=2.)
    parser.add_argument('--read-once', action='store_true', help='只读输出一次JSON，不创建Tk窗口')
    parser.add_argument('--ui-state',type=Path,help='可选：导出本监控窗口的渲染回执')
    parser.add_argument('--control-root',type=Path,help='可选：启用本地设置与启动控制')
    args = parser.parse_args(argv)
    if not math.isfinite(args.refresh_seconds) or not .5 <= args.refresh_seconds <= 60:
        parser.error('--refresh-seconds must be between0.5 and60')
    if args.read_once:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        snapshot=active_study_snapshot(args.active_study) if args.active_study else load_comparison_snapshot(args.comparison_output)
        attach_dodge_diagnostics(snapshot,{})
        attach_passive_diagnostics(snapshot)
        print(json.dumps(snapshot, ensure_ascii=False, allow_nan=False))
    else:
        output=(read_shared_json(args.active_study)['comparison_output'] if args.active_study else args.comparison_output)
        ComparisonWindow(output,args.refresh_seconds,args.active_study,args.ui_state,args.control_root).run()


if __name__ == '__main__':
    main()
