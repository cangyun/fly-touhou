"""Bounded, read-only descriptions of observed training behavior.

No game, learner, model, or checkpoint imports. Windows handles share read,
write AND delete so append/replace writers are never denied by this reader.
These statistics do not identify pure unaided dodging or independent skill.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import statistics
import threading

SCHEMA = 'th14-dodge-diagnostics-v1'
FULL_STORY = 'full-story-v1'
TRAINING_SCOPE = 'completed-training-episodes-not-independent-evaluation'
VULNERABLE_LABEL = '记录的无保护且无中弹/Bomb帧；不是全部可受伤暴露时间'
LIMITATIONS = [
    '训练窗口包含变化中的策略，不是独立冻结策略测试；20局仅为描述稳定性门槛。',
    '完整局时长包含无敌、Bomb保护和过场；Bomb减少不能证明躲避进步。',
    'vulnerable_frames只计前后均正常且无保护、无中弹、无Bomb的筛选帧；相关Miss率仅为有选择偏差的代理指标。',
    '有限残机Game Over通常消耗固定Miss数，单看每局Miss总量区分力有限。',
    '决死机会按可见决策计数，同一窗口可见多次；隐藏在采样间隔中的窗口数未知。',
    '当前日志未记录首次中弹时间、最长不靠Bomb的连续躲避、各关单独的受伤暴露时间。',
]


@contextmanager
def _shared_binary(path):
    """Short-lived binary stream whose Win32 sharing includes deletion."""
    if os.name != 'nt':
        with Path(path).open('rb') as stream:
            yield stream
        return
    import ctypes
    from ctypes import wintypes
    import msvcrt
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(Path(path).resolve()), 0x80000000, 1 | 2 | 4,
                                None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(int(handle), os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    with os.fdopen(descriptor, 'rb') as stream:
        yield stream


def _number(value, *, signed=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) and (signed or value >= 0) else None


def _protocol(row):
    if any(key.startswith('curriculum_') for key in row) or row.get('lesson') is not None:
        return 'legacy-course'
    return FULL_STORY if row.get('training_mode') == FULL_STORY else 'legacy-unidentified'


def _normalized(row, offset=0):
    diagnostics = row.get('reward_diagnostics')
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    result = {'_offset': offset, 'protocol': _protocol(row), 'outcome': row.get('outcome')}
    for key in ('native_frames', 'at_decisions', 'policy_decisions'):
        result[key] = _number(row.get(key))
    for key in ('native_misses', 'vulnerable_frames', 'actual_bombs_used',
                'ordinary_or_unconfirmed_bombs', 'confirmed_deathbombs', 'protected_decisions',
                'deathbomb_decision_count', 'deathbomb_usable_decision_count',
                'deathbomb_observed_window_count', 'deathbomb_usable_bomb_requests',
                'deathbomb_requested_consumed_without_miss_decision_count',
                'deathbomb_timer1_usable_decision_count'):
        result[key] = _number(diagnostics.get(key))
    result['misses'] = result['native_misses']
    if result['misses'] is None:
        result['misses'] = _number(row.get('misses'))
    if result['actual_bombs_used'] is None:
        result['actual_bombs_used'] = _number(row.get('bombs_used'))
    if result['policy_decisions'] is None:
        result['policy_decisions'] = _number(diagnostics.get('policy_decisions'))
    stages = row.get('stage_history')
    result['stages'] = (sorted({value for value in stages if type(value) is int and 1 <= value <= 6})
                        if isinstance(stages, list) else [])
    result['max_stage'] = max(result['stages'], default=None)
    endpoints=row.get('endpoints');end=endpoints.get('end') if isinstance(endpoints,dict) else None
    end=end if isinstance(end,dict) else {}
    result.update(end_power=_number(end.get('power')),
        end_bombs=_number(end.get('bombs_remaining',row.get('bombs_remaining'))),
        end_lives=_number(end.get('lives'),signed=True),point_items=_number(row.get('point_items')))
    return result


def _sum(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return (sum(values) if values else None), len(values)


def _ratio(rows, numerator, denominator, scale=1.):
    pairs = [(row[numerator], row[denominator]) for row in rows
             if row.get(numerator) is not None and row.get(denominator) is not None
             and row[denominator] > 0]
    denom = sum(pair[1] for pair in pairs)
    return (scale * sum(pair[0] for pair in pairs) / denom if denom > 0 else None), len(pairs)


def _summary(rows, *, scope):
    count = len(rows)
    frames = [row['native_frames'] for row in rows if row['native_frames'] is not None]
    stages = [row['max_stage'] for row in rows if row['max_stage'] is not None]
    coverage = {}
    result = dict(scope=scope, episodes=count,
        native_frames=dict(mean=statistics.mean(frames) if frames else None,
            median=statistics.median(frames) if frames else None,
            min=min(frames) if frames else None, max=max(frames) if frames else None,
            coverage_episodes=len(frames)),
        clears=sum(row['outcome'] == 'clear' for row in rows),
        mean_max_stage=statistics.mean(stages) if stages else None,
        stage_history_coverage_episodes=len(stages),
        stage_reach_counts={str(stage):sum(stage in row['stages'] for row in rows) for stage in range(1,7)},
        # Missing stage history is not evidence of failure to reach that stage.
        stage_reach_rates={str(stage):(sum(stage in row['stages'] for row in rows)/len(stages)
                            if stages else None) for stage in range(1,7)},
        vulnerable_frames_label=VULNERABLE_LABEL,
        vulnerable_rate_is_biased_proxy=True,
        first_hit={'available':False, 'reason':'首次中弹时间未由当前episode日志记录'},
        longest_unassisted_dodge={'available':False,'reason':'无Bomb连续躲避区间未记录'},
        coverage=coverage)
    keys = ('misses','vulnerable_frames','actual_bombs_used','ordinary_or_unconfirmed_bombs',
            'confirmed_deathbombs','policy_decisions','protected_decisions')
    for key in keys:
        result[key], coverage[key] = _sum(rows,key)
    for key in ('confirmed_deathbombs','ordinary_or_unconfirmed_bombs','misses'):
        result[key+'_per_episode'] = result[key]/coverage[key] if coverage[key] else None
    for name, numerator, denominator, scale in (
            ('misses_per_1000_native_frames','misses','native_frames',1000.),
            ('misses_per_1000_vulnerable_frames','misses','vulnerable_frames',1000.),
            ('protected_decision_fraction','protected_decisions','policy_decisions',1.)):
        result[name], coverage[name] = _ratio(rows,numerator,denominator,scale)
    deathbomb = {'coverage_complete':False, 'unit':'observed-policy-decisions-not-independent-windows',
                'unobserved_windows_unknown':True, 'coverage':{}}
    for name,key in (
            ('decision_opportunities','deathbomb_decision_count'),
            ('usable_decision_opportunities','deathbomb_usable_decision_count'),
            ('observed_windows','deathbomb_observed_window_count'),
            ('requests','deathbomb_usable_bomb_requests'),
            ('requested_consumed_without_miss','deathbomb_requested_consumed_without_miss_decision_count'),
            ('timer1_usable_decisions','deathbomb_timer1_usable_decision_count')):
        deathbomb[name], deathbomb['coverage'][name] = _sum(rows,key)
    for name,numerator,denominator in (
            ('conditional_request_rate','deathbomb_usable_bomb_requests','deathbomb_usable_decision_count'),
            ('requested_consumed_without_miss_fraction',
             'deathbomb_requested_consumed_without_miss_decision_count','deathbomb_usable_bomb_requests')):
        deathbomb[name],deathbomb['coverage'][name] = _ratio(rows,numerator,denominator)
    # This bound is available only where BOTH observed timer counts were logged.
    robust = [row['deathbomb_usable_decision_count']-row['deathbomb_timer1_usable_decision_count']
              for row in rows if row.get('deathbomb_usable_decision_count') is not None
              and row.get('deathbomb_timer1_usable_decision_count') is not None
              and row['deathbomb_timer1_usable_decision_count'] <= row['deathbomb_usable_decision_count']]
    deathbomb['at_least_2_frames_decisions'] = sum(robust) if robust else None
    deathbomb['coverage']['at_least_2_frames_decisions'] = len(robust)
    deathbomb['true_native_deathbomb_success_rate'] = None
    result['deathbomb'] = deathbomb
    result['resources']={'scope':'native-terminal-resource-snapshot-not-dodge-score'}
    for key in ('end_power','end_bombs','end_lives','point_items'):
        total,covered=_sum(rows,key)
        result['resources'][key]={'mean':total/covered if covered else None,'coverage_episodes':covered}
    return result


def summary_rows(rows, *, scope):
    """Summarize already-read native-terminal rows with an explicit provenance.

    Intended for existing development/test reports. It does not open files or
    borrow training samples. Do not call a developing-policy window a test.
    """
    if not isinstance(scope,str) or not scope.strip():
        raise ValueError('Explicit diagnostic scope required')
    rows = list(rows)
    valid = [_normalized(row,i) for i,row in enumerate(rows)
             if isinstance(row,dict) and row.get('outcome') in ('gameover','clear')]
    protocols=sorted({row['protocol'] for row in valid})
    if len(protocols)>1:
        raise ValueError('Cannot combine full-story, old course and unidentified protocols')
    result = _summary(valid,scope=scope)
    result['excluded_nonterminal_or_invalid_rows'] = len(rows)-len(valid)
    result['protocols'] = protocols
    result['mixed_protocols'] = len(result['protocols']) > 1
    result['statistical_significance_claimed'] = False
    result['limitations'] = LIMITATIONS.copy()
    return result


summarize_rows = summary_rows


def _comparison(initial, recent, initial_rows, recent_rows, protocol):
    disjoint = not ({row['_offset'] for row in initial_rows} & {row['_offset'] for row in recent_rows})
    threshold = initial['episodes'] >= 20 and recent['episodes'] >= 20
    blocked = ('旧课程或未识别协议不作全程技能趋势比较' if protocol != FULL_STORY else
               '初期与最近窗口有重复局' if not disjoint else None)
    delta = {};delta_coverage = {}
    metric_values = [
        (key,initial['native_frames'][field],recent['native_frames'][field],
         initial['native_frames']['coverage_episodes'],recent['native_frames']['coverage_episodes'])
        for key,field in (('mean_native_frames','mean'),('median_native_frames','median'))]
    metric_values.append(('mean_max_stage',initial['mean_max_stage'],recent['mean_max_stage'],
        initial['stage_history_coverage_episodes'],recent['stage_history_coverage_episodes']))
    for key,coverage_key in (
            ('confirmed_deathbombs_per_episode','confirmed_deathbombs'),
            ('ordinary_or_unconfirmed_bombs_per_episode','ordinary_or_unconfirmed_bombs'),
            ('protected_decision_fraction','protected_decision_fraction')):
        metric_values.append((key,initial[key],recent[key],initial['coverage'][coverage_key],
                              recent['coverage'][coverage_key]))
    for key,a,b,first_count,last_count in metric_values:
        usable = bool(blocked is None and first_count >= 20 and last_count >= 20
                      and a is not None and b is not None)
        reason = (blocked or ('该指标两窗均至少20个有效观测局；仅作描述比较' if usable else
            f'该指标有效观测不足：初期{first_count}局、最近{last_count}局，各需20局'
            if first_count < 20 or last_count < 20 else '该指标数值不可用'))
        delta[key] = b-a if usable else None
        delta_coverage[key] = dict(initial=first_count,recent=last_count,
            minimum_episodes=20,available=usable,reason=reason)
    available = any(row['available'] for row in delta_coverage.values())
    reason = (blocked or ('至少一项指标的两窗有效观测均达20局；逐项检查覆盖，仅描述变化'
                          if available else '没有指标在两窗都达到20个有效观测局'))
    return dict(available=available,reason=reason,nonoverlapping=disjoint,
        minimum_episodes=20,descriptive_sample_threshold_met=threshold,
        availability_semantics='at-least-one-delta-has-20-valid-episodes-per-window',
        delta=delta,delta_coverage=delta_coverage,
        statistical_significance_claimed=False,causal_improvement_claimed=False)


class DodgeDiagnosticsReader:
    """One cached reader per actual run folder. Every poll has bounded byte I/O.

    On a large pre-existing log read its head and tail, not its whole history.
    The gap is explicit; totals refer only to read rows, never the whole file.
    Thereafter read only appended bytes. A tail without '\n' stays pending.
    """
    def __init__(self,run_dir,*,window_episodes=30,read_budget_bytes=262144,max_line_bytes=65536):
        if type(window_episodes) is not int or not 1 <= window_episodes <= 200:
            raise ValueError('window_episodes must be 1..200')
        if type(read_budget_bytes) is not int or not 1024 <= read_budget_bytes <= 4*1024*1024:
            raise ValueError('read_budget_bytes must be 1KiB..4MiB')
        if type(max_line_bytes) is not int or not 256 <= max_line_bytes <= read_budget_bytes:
            raise ValueError('max_line_bytes must fit inside read_budget_bytes')
        self.run_dir = Path(run_dir).resolve()
        self.path = self.run_dir/'training_episodes.jsonl'
        self.window = window_episodes; self.budget = read_budget_bytes; self.max_line = max_line_bytes
        self._lock = threading.Lock(); self._generation = 0
        self._reset()

    def _reset(self):
        self._identity = None; self._offset = 0; self._size = 0; self._prefix = b''
        self._pending = b''; self._pending_offset = 0; self._discard_line = False
        self._initial = {}; self._recent = {}; self._counts = {}
        self._malformed = self._oversized = self._nonterminal = self._skipped = self._rows = 0
        self._history_complete = True

    def _accept(self,line,offset,baseline):
        if not line.strip():return
        if len(line)>self.max_line:
            self._oversized += 1;return
        try:
            row = json.loads(line.decode('utf-8-sig'))
        except (ValueError,UnicodeError):
            self._malformed += 1;return
        if not isinstance(row,dict):
            self._malformed += 1;return
        if row.get('outcome') not in ('gameover','clear'):
            self._nonterminal += 1;return
        value = _normalized(row,offset);protocol = value['protocol']
        first = self._initial.setdefault(protocol,[])
        recent = self._recent.setdefault(protocol,deque(maxlen=self.window))
        if baseline and len(first)<self.window:first.append(value)
        recent.append(value);self._counts[protocol]=self._counts.get(protocol,0)+1;self._rows+=1

    def _feed(self,data,start,*,baseline):
        origin = self._pending_offset if self._pending else start
        blob = self._pending+data;cursor = 0
        while True:
            end = blob.find(b'\n',cursor)
            if end<0:break
            if self._discard_line:self._discard_line=False
            else:self._accept(blob[cursor:end],origin+cursor,baseline)
            cursor=end+1
        remainder=blob[cursor:]
        self._pending_offset=origin+cursor
        if self._discard_line:
            self._pending=b''
        elif len(remainder)>self.max_line:
            self._oversized+=1;self._discard_line=True;self._pending=b''
        else:self._pending=remainder

    def _read(self):
        used=0;error=None;available=False
        try:
            with _shared_binary(self.path) as stream:
                available=True;status=os.fstat(stream.fileno());size=status.st_size
                identity=(status.st_dev,status.st_ino)
                # Detect same-inode rewrites too. This small read is included in
                # the byte budget, and never rereads the complete log.
                prefix=stream.read(min(128,size));used+=len(prefix)
                prefix_changed=bool(self._prefix and not prefix.startswith(self._prefix))
                if self._identity is not None and (identity!=self._identity or size<self._offset or prefix_changed):
                    self._generation+=1;self._reset()
                initial=self._identity is None
                self._identity=identity;self._size=size
                if not self._prefix:self._prefix=prefix
                if initial:
                    self._prefix=prefix
                    budget=self.budget-used
                    if size<=budget:
                        stream.seek(0);data=stream.read(size);used+=len(data)
                        self._feed(data,0,baseline=True);self._offset=len(data)
                    else:
                        head_size=budget//2;tail_size=budget-head_size
                        stream.seek(0);head=stream.read(head_size);used+=len(head)
                        self._feed(head,0,baseline=True)
                        tail_start=max(len(head),size-tail_size)
                        # Do not treat the middle of a JSON record as a row.
                        stream.seek(tail_start);tail=stream.read(tail_size);used+=len(tail)
                        newline=tail.find(b'\n')
                        skip=len(tail) if newline<0 else newline+1
                        tail_content_start=tail_start+skip
                        self._skipped=max(0,tail_content_start-self._pending_offset)
                        self._pending=b'';self._discard_line=False
                        self._history_complete=False
                        self._feed(tail[skip:],tail_content_start,baseline=False)
                        self._offset=tail_start+len(tail)
                else:
                    stream.seek(self._offset);data=stream.read(min(size-self._offset,self.budget-used))
                    used+=len(data);self._feed(data,self._offset,baseline=self._history_complete)
                    self._offset+=len(data)
        except (OSError,ValueError) as exc:
            error=type(exc).__name__+': '+str(exc)
        return dict(file_available=available,bytes_read_this_poll=used,
            byte_budget=self.budget,file_bytes=self._size,offset=self._offset,
            caught_up=available and error is None and self._offset>=self._size,
            pending_line_bytes=len(self._pending),history_complete=self._history_complete,
            skipped_history_bytes=self._skipped,complete_rows_observed=self._rows,
            malformed_lines=self._malformed,oversized_lines=self._oversized,
            excluded_nonterminal_rows=self._nonterminal,generation=self._generation,
            error=error,partial_tail_not_counted=bool(self._pending or self._discard_line))

    def poll(self,progress=None,protocol_metadata=None):
        """Return JSON-safe data. Optional inputs must belong to this run.

        Existing monitor readers supply progress/config; this method does not
        perform another JSON read or infer a protocol from next-run templates.
        """
        with self._lock:
            reader=self._read();groups=[]
            labels={FULL_STORY:'原作全程连打','legacy-course':'旧课程回合（可能为片段）',
                    'legacy-unidentified':'旧版或协议未标记回合'}
            for protocol in sorted(self._recent):
                first=list(self._initial[protocol]);last=list(self._recent[protocol])
                initial=_summary(first,scope=TRAINING_SCOPE);recent=_summary(last,scope=TRAINING_SCOPE)
                groups.append(dict(protocol=protocol,label=labels[protocol],initial=initial,recent=recent,
                    complete_rows_observed=self._counts[protocol],
                    initial_window_requested=self.window,recent_window_requested=self.window,
                    initial_window_complete=len(first)>=self.window,
                    comparison=_comparison(initial,recent,first,last,protocol)))
            progress=progress if isinstance(progress,dict) else {}
            metadata=protocol_metadata if isinstance(protocol_metadata,dict) else {}
            worlds=progress.get('current_environments',[])
            live=dict(scope='ongoing-training-prefixes-excluded-from-completed-episode-statistics',
                progress_available=bool(progress),
                actual_decisions=_number(progress.get('actual_decisions')),
                optimized_decisions=_number(progress.get('candidate_optimized_decisions')),
                reported_training_episode_count=_number(progress.get('episodes')),
                workers=[{key:row.get(key) for key in ('worker','stage','stage_frame','lives','bombs','power',
                        'misses','bombs_used','episode_frames','policy_steps')}
                         for row in worlds[:100] if isinstance(row,dict)] if isinstance(worlds,list) else [])
            return dict(schema=SCHEMA,run_directory=str(self.run_dir),scope=TRAINING_SCOPE,
                independent_evaluation_included=False,training_policy_changes_within_windows=True,
                declared_training_mode=metadata.get('training_mode',progress.get('training_mode')),
                row_protocol_is_authoritative=True,protocol_groups=groups,reader=reader,live=live,
                limitations=LIMITATIONS.copy())
