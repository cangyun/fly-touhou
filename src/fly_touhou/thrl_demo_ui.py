"""The original native control panel's real-game demonstration tab."""
from pathlib import Path
import queue
import threading

from .thrl_demo_control import DemoControl, selection_description


class DemoTab:
    def __init__(self, notebook, control_root):
        import tkinter as tk
        from tkinter import ttk
        self.control=DemoControl(control_root);self.events=queue.Queue();self.busy=False
        self.tab=ttk.Frame(notebook,padding=20);notebook.add(self.tab,text='模型实机演示')
        self.message=tk.StringVar(value='载入已验证的模型…')
        self.evidence=tk.StringVar(value='已知开局展示，不计入新的独立测试。')
        self.state=tk.StringVar(value='尚未开始演示')
        self.details=tk.StringVar(value='')
        ttk.Label(self.tab,text='神经网络实时操作原作《辉针城》',font=('',15,'bold')).pack(anchor='w')
        ttk.Label(self.tab,text=('Lunatic · 灵梦A · 19动作 · 自主 Bomb · 禁止续关\n'
            '每个模型使用其记录中已通关的开局；评估类型与种子随所选模型显示。战斗动作由网络实时推理，菜单由接口处理。\n'
            '真实原作窗口保留游戏帧率上限；实际速度受推理耗时影响。窗口以不激活方式显示，不发送系统键鼠输入。'),
            justify='left',wraplength=1120).pack(anchor='w',pady=14)
        self.labels={};self.catalogue_config=None;self.selection=tk.StringVar()
        try:
            catalogue=self.control.catalogue()
            self.catalogue_config=catalogue
            self.labels={entry['label']:key for key,entry in catalogue['models'].items()}
            self.selection.set(catalogue['models'][catalogue['default_model']]['label'])
            self.evidence.set(selection_description(catalogue,catalogue['default_model']))
            self.message.set(catalogue['selection_note'])
        except Exception as error:self.message.set('演示配置暂不可用：'+str(error))
        line=ttk.Frame(self.tab);line.pack(anchor='w',fill='x',pady=8)
        ttk.Label(line,text='选择模型：').pack(side='left')
        self.combo=ttk.Combobox(line,textvariable=self.selection,values=list(self.labels),state='readonly',width=75)
        self.combo.pack(side='left',padx=12)
        self.combo.bind('<<ComboboxSelected>>',self.update_selection_evidence)
        ttk.Label(self.tab,textvariable=self.evidence,wraplength=1120,justify='left',foreground='#915600').pack(anchor='w',pady=6)
        actions=ttk.Frame(self.tab);actions.pack(anchor='w',pady=12)
        self.start_button=ttk.Button(actions,text='开始实机演示',command=lambda:self.submit('start'))
        self.start_button.pack(side='left',padx=(0,12))
        self.pause_button=ttk.Button(actions,text='暂停 / 继续',command=lambda:self.submit('pause'))
        self.pause_button.pack(side='left',padx=(0,12))
        self.stop_button=ttk.Button(actions,text='停止演示',command=lambda:self.submit('stop'))
        self.stop_button.pack(side='left',padx=(0,12))
        ttk.Button(actions,text='打开本次记录',command=self.open_record).pack(side='left')
        ttk.Button(actions,text='本轮奖励与失败复盘',command=self.open_review).pack(side='left',padx=12)
        ttk.Button(actions,text='查看终局画面',command=self.open_terminal).pack(side='left')
        ttk.Label(self.tab,textvariable=self.message,wraplength=1120,justify='left',foreground='#215b8c').pack(anchor='w',pady=10)
        box=ttk.LabelFrame(self.tab,text='当前实机',padding=16);box.pack(fill='x',pady=10)
        ttk.Label(box,textvariable=self.state,font=('',14,'bold')).pack(anchor='w')
        ttk.Label(box,textvariable=self.details,wraplength=1100,justify='left').pack(anchor='w',pady=12)
        ttk.Label(self.tab,text=('暂停只暂停演示的原作帧，不影响训练和评估；停止只关闭该演示拥有的原作进程。\n'
            '通关后保留终局画面10分钟，也可点击停止关闭。\n'
            '关闭控制面板不会停止演示。若发生重启或异常，保留记录，并由你再次点击开始。'),
            justify='left',wraplength=1120).pack(anchor='w',pady=12)
        self.tab.after(250,self.poll)

    def update_selection_evidence(self, _event=None):
        key=self.labels.get(self.selection.get())
        if self.catalogue_config is not None and key is not None:
            self.evidence.set(selection_description(self.catalogue_config,key))

    def submit(self, action):
        if self.busy:return
        self.busy=True
        key=self.labels.get(self.selection.get())
        def work():
            try:
                value=self.control.start(key) if action=='start' else getattr(self.control,action)()
                self.events.put(('action',action,value))
            except Exception as error:self.events.put(('error',str(error),None))
        threading.Thread(target=work,name='real-demo-control',daemon=True).start()

    def open_record(self):
        import os
        state=self.control.status()
        if state.get('output') and Path(state['output']).is_dir():os.startfile(state['output'])

    def open_review(self):
        import os
        path=self.control.root.parent/'B1-autonomous-dodge-20261024/stable-clear-wrapup-review-v1.html'
        if path.is_file():os.startfile(path)

    def open_terminal(self):
        import os
        state=self.control.status()
        if not state.get('output'):return
        for name in ('terminal-renderer.png','terminal-observed-by-host.png'):
            path=Path(state['output'])/name
            if path.is_file():os.startfile(path);return

    def poll(self):
        try:
            while True:
                kind,value,_=self.events.get_nowait();self.busy=False
                self.message.set(value if kind=='error' else {'start':'演示已派发，正在载入 CUDA 模型和原作。',
                    'pause':'已切换暂停状态。','stop':'已请求停止演示，等待原作安全关闭。'}[value])
        except queue.Empty:pass
        try:
            state=self.control.status();running=state['running'];game=state.get('game',{})
            names={'idle':'尚未开始','starting':'正在启动','running':'正在实机演示','paused':'已暂停',
                'menu_resume':'正在解除原作暂停菜单',
                'stopped':'演示已停止','completed':'演示完成','failed':'演示失败','interrupted':'演示被中断'}
            outcome={'clear':'六面通关','gameover':'Game Over'}.get(state.get('outcome'),'')
            self.state.set(names.get(state.get('status'),state.get('status',''))+(' · '+outcome if outcome else ''))
            self.details.set(f"模型：{state.get('model_label','—')}\n"
                f"第 {game.get('stage','—')} 面 · 残机 {game.get('lives','—')} · Bomb {game.get('bombs','—')} · "
                f"Miss {game.get('misses','—')} · 已用 Bomb {game.get('bombs_used','—')}\n"
                f"决策 {state.get('policy_decisions',0):,} · 原作帧 {game.get('episode_frames','—')} · "
                f"运行 {state.get('wall_seconds',0):.0f} 秒 · "
                f"平均 {(game.get('episode_frames') or 0)/max(state.get('wall_seconds',0),.001):.1f} 原作帧/秒\n"
                '已知种子展示 · 实时 CUDA 推理 · 无教师接管 · 不计入新的测试成绩')
            self.start_button.configure(state='disabled' if running or self.busy or not self.labels else 'normal')
            for button in (self.pause_button,self.stop_button):button.configure(state='normal' if running and not self.busy else 'disabled')
            self.combo.configure(state='disabled' if running else 'readonly')
        except Exception as error:self.message.set('读取演示状态失败：'+str(error))
        if self.tab.winfo_exists():self.tab.after(1000,self.poll)
