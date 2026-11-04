"""Native, read-only Tkinter training monitor; only its Stop button writes STOP.

No browser, Torch, model loading, network request, or game operation is used.
Tk is imported only when main creates the GUI, so readers can be checked cold.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import queue
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fly_touhou.thrl_study_io import read_shared_json

REFRESH_SECONDS = 2.
STATE_LABELS = {
    'collecting': '采集中', 'updating': '参数更新中', 'development': '开发评估中',
    'starting': '启动中', 'preparing': '准备中', 'running': '运行中',
    'finished': '已结束', 'completed': '已完成', 'stopped': '已停止',
    'stopping': '正在收尾', 'failed': '失败', 'error': '需要检查',
    'training': '训练中', 'initializing': '初始化中', 'pruned': '已提前结束',
    'attention_comparison': '独立注意力对照', 'operator_or_budget_stop': '停止收尾',
    'validation20': '冻结模型扩展验证（每组20局）',
    'RUNNING': '进行中', 'COMPLETE': '已完成', 'PRUNED': '提前结束',
    'FAIL': '失败', 'WAITING': '等待中',
}


def number(value, digits=3, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return '—'
    return f'{value:,.0f}' if integer else f'{value:,.{digits}f}'


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def load_snapshot(study_output, last_run=None):
    """Read only small status/development JSON with shared-read/write/delete.

    Reports are display data, not a result audit or formal-test selector. A
    replacement/partial-write error is retained as an error until next refresh.
    No checkpoints, arrays, compressed logs, or HTTP endpoint are opened.
    """
    output = Path(study_output)
    errors = []

    def read(path, description):
        try:
            document = read_shared_json(path)
            if not isinstance(document, dict):
                raise ValueError('Expected a JSON object')
            return document
        except (OSError, ValueError) as error:
            errors.append(f'{description}暂未就绪（{type(error).__name__}）')
            return {}

    study = read(output / 'study_status.json', '调参记录')
    current = study.get('current_run')
    display_run = current or last_run
    run = Path(display_run) if isinstance(display_run, str) and display_run else None
    progress = read(run / 'progress.json', '训练记录') if run else {}
    reports = []
    if run:
        try:
            paths = sorted(run.glob('development_*/report.json'))
        except OSError as error:
            paths = []
            errors.append(f'开发目录暂未就绪（{type(error).__name__}）')
        for path in paths:
            report = read(path, '开发报告')
            if not report:
                continue
            suffix = path.parent.name.removeprefix('development_')
            step = int(suffix) if suffix.isdecimal() else report.get('actual_decisions', report.get('decisions'))
            if not _finite(step):
                continue
            reports.append({'decisions': int(step), 'path': str(path),
                **{key: report.get(key) for key in ('episodes', 'clears', 'score',
                    'mean_native_frames', 'mean_max_stage', 'status')}})
    reports.sort(key=lambda item: item['decisions'])
    return {'observed_utc': datetime.now(timezone.utc).isoformat(),
            'study': study, 'progress': progress, 'development': reports,
            'current_run': current, 'display_run': str(run) if run else None,
            'stop_requested': (output / 'STOP').is_file(), 'errors': errors}


def request_stop(study_output):
    """Idempotent explicit user request. Never terminate a process directly."""
    output = Path(study_output)
    if not output.is_dir():
        raise FileNotFoundError('调参输出目录尚不存在')
    try:
        with (output / 'STOP').open('x', encoding='utf-8') as stream:
            json.dump({'requested_utc': datetime.now(timezone.utc).isoformat(),
                       'request': 'stop_current_and_later_trials',
                       'source': 'native_training_monitor'}, stream, ensure_ascii=False)
    except FileExistsError:
        return False
    return True


def _triple(value):
    values = value if isinstance(value, (list, tuple)) else []
    return ' / '.join(number(values[index] if index < len(values) else None) for index in range(3))


class MonitorWindow:
    def __init__(self, study_output):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.output = Path(study_output).resolve()
        self.root = tk.Tk()
        self.root.title('东方训练监控 · 开发过程')
        self.root.geometry('1120x820')
        self.root.minsize(940, 720)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        style = ttk.Style(self.root)
        if 'vista' in style.theme_names():
            style.theme_use('vista')
        style.configure('Title.TLabel', font=('Microsoft YaHei UI', 17, 'bold'))
        style.configure('Card.TLabel', font=('Microsoft YaHei UI', 11))
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill='both', expand=True)
        self.title = tk.StringVar(value='等待训练记录')
        ttk.Label(frame, textvariable=self.title, style='Title.TLabel').pack(anchor='w')
        self.subtitle = tk.StringVar(value='训练与开发记录分开显示；本窗口没有正式测试结果。')
        ttk.Label(frame, textvariable=self.subtitle).pack(anchor='w', pady=(6, 12))
        cards = ttk.Frame(frame)
        cards.pack(fill='both', expand=True)
        for col in range(2):
            cards.columnconfigure(col, weight=1, uniform='cards')
        cards.rowconfigure(0, weight=2)
        cards.rowconfigure(1, weight=3)
        self.training = tk.StringVar(value='等待原作采样……')
        self.metrics = tk.StringVar(value='等待首次更新……')
        self.development = tk.StringVar(value='等待开发节点，每节点计划6局。')
        titles = ('训练进度（原作采样）', '训练指标（最近采样 / 更新）',
                  '开发评估（每节点6局；非正式测试）', 'Optuna 调参试次（开发）')
        containers = []
        for index, title in enumerate(titles):
            card = ttk.LabelFrame(cards, text=title, padding=12)
            card.grid(row=index // 2, column=index % 2, sticky='nsew',
                      padx=(0, 8) if index % 2 == 0 else (8, 0), pady=(0, 12))
            containers.append(card)
        ttk.Label(containers[0], textvariable=self.training, style='Card.TLabel',
                  justify='left', wraplength=490).pack(anchor='nw', fill='x')
        ttk.Label(containers[1], textvariable=self.metrics, style='Card.TLabel',
                  justify='left', wraplength=490).pack(anchor='nw', fill='x')
        ttk.Label(containers[2], textvariable=self.development, justify='left',
                  wraplength=490).pack(anchor='nw', fill='x')
        self.chart = tk.Canvas(containers[2], background='white', highlightthickness=0, height=250)
        self.chart.pack(fill='both', expand=True, pady=(8, 0))
        self.chart.bind('<Configure>', lambda _: self.draw_chart())
        columns = ('number', 'variant', 'state', 'decisions', 'score', 'stage')
        self.trials = ttk.Treeview(containers[3], columns=columns, show='headings', height=10)
        for key, label, width in zip(columns,
            ('编号', '模型', '状态', '采样步', '开发得分', '最远面'), (45, 85, 75, 85, 80, 65)):
            self.trials.heading(key, text=label)
            self.trials.column(key, width=width, minwidth=40, anchor='center', stretch=True)
        scrollbar = ttk.Scrollbar(containers[3], orient='vertical', command=self.trials.yview)
        self.trials.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.trials.pack(fill='both', expand=True)
        footer = ttk.Frame(frame)
        footer.pack(fill='x', pady=(4, 0))
        self.footer = tk.StringVar(value='每2秒刷新；关闭监控窗口不会停止训练。')
        ttk.Label(footer, textvariable=self.footer, wraplength=760).pack(side='left', anchor='w')
        self.stop_button = ttk.Button(footer, text='停止当前训练及后续调参', command=self.stop)
        self.stop_button.pack(side='right', padx=(8, 0))
        self.closed = threading.Event()
        self.messages = queue.Queue(maxsize=1)
        self.snapshot = {}
        self.stop_clicked = False
        self.reader = threading.Thread(target=self.read_loop, name='training-status-reader', daemon=True)
        self.reader.start()
        self.root.after(100, self.poll)

    def read_loop(self):
        last_run = None
        while not self.closed.is_set():
            snapshot = load_snapshot(self.output, last_run=last_run)
            last_run = snapshot.get('display_run')
            try:
                self.messages.put_nowait(snapshot)
            except queue.Full:
                try:
                    self.messages.get_nowait()
                except queue.Empty:
                    pass
                self.messages.put_nowait(snapshot)
            self.closed.wait(REFRESH_SECONDS)

    def poll(self):
        try:
            self.snapshot = self.messages.get_nowait()
            self.render()
        except queue.Empty:
            pass
        if not self.closed.is_set():
            self.root.after(150, self.poll)

    def render(self):
        study, progress = self.snapshot.get('study', {}), self.snapshot.get('progress', {})
        name = study.get('study_name') or '东方训练'
        status = ((progress.get('stage') or progress.get('status')) if self.snapshot.get('current_run')
                  else study.get('status')) or study.get('status')
        label = STATE_LABELS.get(status, status or '等待中')
        self.title.set(f'{name} · {label}')
        attention = progress.get('attention', 'none')
        attention_label = {'none': '无注意力', 'spatial': '空间注意力'}.get(attention, str(attention))
        gpu = progress.get('gpu') or study.get('gpu')
        gpu = gpu.get('name', '—') if isinstance(gpu, dict) else (gpu or '—')
        retained = '　保留最后实例记录' if not self.snapshot.get('current_run') and progress else ''
        self.subtitle.set(f'模型：{attention_label}　设备：{progress.get("device", study.get("device", "—"))} / {gpu}'
                          f'　调参完成 {number(study.get("completed_trials"), integer=True)} / '
                          f'{number(study.get("planned_trials"), integer=True)}{retained}；以下均为训练或开发记录。')
        environments = progress.get('current_environments') or []
        worker_lines = []
        for env in environments:
            if not isinstance(env, dict):
                continue
            worker_lines.append(f'实例{number(env.get("worker"), integer=True)}：第{number(env.get("stage"), integer=True)}面 '
                f'{number(env.get("stage_frame"), integer=True)}帧　残机{number(env.get("lives"), integer=True)} / '
                f'Bomb{number(env.get("bombs"), integer=True)} / Miss{number(env.get("misses"), integer=True)}')
        self.training.set(f'采样决策：{number(progress.get("actual_decisions"), integer=True)}'
            f'　Adam：{number(progress.get("adam_minibatch_steps"), integer=True)}\n'
            f'更新轮次：{number(progress.get("rollout_updates"), integer=True)}'
            f'　回合：{number(progress.get("episodes"), integer=True)}'
            f'　通关：{number(progress.get("clear_episodes"), integer=True)}\n'
            f'速度：{number(progress.get("decisions_per_second"), 1)} 决策/秒'
            f'　运行：{number(progress.get("wall_seconds"), 0)}秒\n' + '\n'.join(worker_lines))
        metrics = progress.get('metrics') or {}
        self.metrics.set(f'三目标奖励（生存 / 战斗 / 资源）\n{_triple(progress.get("reward_components_recent"))}\n\n'
            f'策略熵：{number(metrics.get("entropy"))}　熵系数：{number(metrics.get("entropy_coeff"), 4)}\n'
            f'策略损失：{number(metrics.get("policy_loss"))}　总损失：{number(metrics.get("total_loss"))}\n'
            f'三头价值损失：{_triple(metrics.get("value_loss"))}\n'
            f'KL：{number(metrics.get("approx_kl"), 5)}')
        latest = progress.get('latest_development') or {}
        reports = self.snapshot.get('development', [])
        if not latest and reports:
            latest = reports[-1]
        evaluation = progress.get('evaluation_progress') or {}
        active = (f'当前节点：完成 {number(evaluation.get("completed"), integer=True)} / '
                  f'{number(evaluation.get("target"), integer=True)} 局\n') if evaluation else ''
        self.development.set(active + '最近完成开发：通关 '
            f'{number(latest.get("clears"), integer=True)} / {number(latest.get("episodes"), integer=True)} 局'
            f'　得分 {number(latest.get("score"))}\n'
            f'平均最远面数 {number(latest.get("mean_max_stage"), 2)}'
            f'　平均原作帧 {number(latest.get("mean_native_frames"), 1)}')
        self.draw_chart()
        current_selection = self.trials.selection()
        self.trials.delete(*self.trials.get_children())
        for index, trial in enumerate(study.get('trials') or []):
            if not isinstance(trial, dict):
                continue
            variant = trial.get('variant', '—')
            variant = {'none': '无注意力', 'spatial': '空间注意力'}.get(variant, str(variant))
            state = trial.get('state')
            self.trials.insert('', 'end', iid=str(index), values=(trial.get('number', '—'), variant,
                STATE_LABELS.get(state, state or '—'), number(trial.get('decisions'), integer=True),
                number(trial.get('score')), number(trial.get('best_stage'), 1)))
        if current_selection and self.trials.exists(current_selection[0]):
            self.trials.selection_set(current_selection[0])
        if self.stop_clicked or self.snapshot.get('stop_requested'):
            self.stop_button.state(['disabled'])
            self.footer.set('已请求停止当前训练及后续调参，等待训练器安全收尾。')
        else:
            self.stop_button.state(['!disabled'])
            errors = self.snapshot.get('errors') or []
            self.footer.set('；'.join(errors) if errors else
                f'每2秒刷新 · 记录更新：{study.get("updated_utc", "—")} · 关闭窗口不停止训练。')

    def draw_chart(self):
        canvas = self.chart
        canvas.delete('all')
        width, height = max(canvas.winfo_width(), 200), max(canvas.winfo_height(), 150)
        reports = self.snapshot.get('development', [])
        key, title = 'score', '开发得分'
        if not any(_finite(row.get(key)) for row in reports):
            key, title = 'mean_max_stage', '开发平均最远面数'
        points = [(row['decisions'], row[key]) for row in reports if _finite(row.get(key))]
        if not points:
            canvas.create_text(width / 2, height / 2, text='等待首个开发节点\n开发每节点6局；没有正式测试结果。', fill='#687481', justify='center')
            return
        left, top, right, bottom = 64, 24, width - 24, height - 42
        xs, ys = zip(*points)
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        if xmin == xmax:
            xmin, xmax = 0, max(xmax, 1)
        margin = max((ymax - ymin) * .12, abs(ymax) * .03, .05)
        ymin, ymax = ymin - margin, ymax + margin
        canvas.create_text(left, 7, anchor='nw', text=title, fill='#263545')
        for index in range(4):
            y = top + index * (bottom - top) / 3
            value = ymax - index * (ymax - ymin) / 3
            canvas.create_line(left, y, right, y, fill='#e4e8ed')
            canvas.create_text(left - 8, y, anchor='e', text=f'{value:.2g}', fill='#687481')
        positions = []
        for x, y in points:
            px = left + (x - xmin) * (right - left) / (xmax - xmin)
            py = bottom - (y - ymin) * (bottom - top) / (ymax - ymin)
            positions.extend((px, py))
            canvas.create_oval(px - 3, py - 3, px + 3, py + 3, fill='#2867ad', outline='')
            canvas.create_text(px, bottom + 13, text=f'{x / 1000:.1f}k', fill='#687481')
        if len(points) > 1:
            canvas.create_line(*positions, fill='#2867ad', width=2)
        canvas.create_text((left + right) / 2, height - 5, text='训练采样决策数（开发节点位置）', anchor='s', fill='#687481')

    def stop(self):
        try:
            request_stop(self.output)
            self.stop_clicked = True
            self.stop_button.state(['disabled'])
            self.footer.set('已请求停止当前训练及后续调参，等待训练器安全收尾。')
        except OSError as error:
            self.footer.set(f'停止请求未写入：{error}；可重试按钮。')

    def close(self):
        self.closed.set()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-output', type=Path, required=True)
    args = parser.parse_args(argv)
    MonitorWindow(args.study_output).run()


if __name__ == '__main__':
    main()
