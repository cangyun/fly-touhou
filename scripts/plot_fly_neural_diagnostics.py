"""CPU-only exploratory plots from verified, existing neural diagnostic records."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['MPLCONFIGDIR'] = './outputs/tmp/matplotlib-neural-diagnostics'
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.thrl_study_io import read_shared_json

COLORS = ('#0072B2', '#D55E00', '#777777')
LABELS = ('选中VP输入端', 'DN输出端', '其余细胞')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(fig, output, stem):
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'{stem}.{suffix}', dpi=160, facecolor='white')
    plt.close(fig)


def plot_activity(report, output):
    activity = report['activity']
    replay = activity['replay']
    if replay['status'] != 'replay_verified':
        raise ValueError('Unverified replay cannot be presented as a verified neural trace')
    source = Path(replay['NPZ'])
    if digest(source) != replay['NPZ_sha256']:
        raise ValueError('Recorded neural trace SHA differs')
    with np.load(source, allow_pickle=False) as archive:
        spikes = archive['binary_spikes'].copy()
        voltage = archive['prethreshold_mv'].copy()
        population = archive['population_spikes'].copy()
    if (spikes.shape != (400, 64) or voltage.shape != spikes.shape
            or population.shape != (400, 3) or not np.isfinite(voltage).all()
            or not np.isin(spikes, [0, 1]).all() or (population < 0).any()):
        raise ValueError('Expected the complete verified400-tick/64-cell trace')
    dt = float(replay['dt_ms'])
    times = (np.arange(400) + 1) * dt
    groups = (np.arange(16), np.arange(16, 32), np.arange(32, 64))
    names = ('VP_input_ports', 'DN_output_ports', 'remaining_intermediate')
    batch = next(x for x in activity['snapshot']['batches'] if replay['worker'] in x['workers'])
    sizes = np.asarray([batch['populations'][name]['neurons'] for name in names])
    rates = population.reshape(40, 10, 3).sum(axis=1) / (sizes * dt * 10 / 1000)
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9), sharex=True,
                             gridspec_kw={'height_ratios': [1.5, 1, 1.1]})
    fig.subplots_adjust(left=.13, right=.96, bottom=.17, top=.87, hspace=.28)
    fig.suptitle('果蝇学生的脑内响应：已核验的单实例片段', y=.975, fontsize=15)
    fig.text(.13, .92, '第8批 · 实例0 · 4次决策 · 80 ms神经时间；64个细胞为确定性抽样', fontsize=10)
    for indices, label, color in zip(groups, LABELS, COLORS):
        tick, cell = np.nonzero(spikes[:, indices])
        axes[0].scatter(times[tick], indices[cell], marker='|', s=17, linewidths=.8, color=color)
    axes[0].set(ylim=(-1, 64), yticks=[7.5, 23.5, 47.5],
                yticklabels=['输入端16个', '输出端16个', '其余32个'], ylabel='抽样细胞')
    axes[0].axhline(15.5, color='#cccccc', linewidth=.6)
    axes[0].axhline(31.5, color='#cccccc', linewidth=.6)
    axes[0].set_title('a  二值脉冲时序（展示全部64个抽样细胞）', loc='left', fontsize=11)
    for i, (label, color) in enumerate(zip(LABELS, COLORS)):
        axes[1].plot(np.arange(40) * 2 + 1, rates[:, i], color=color, linewidth=1.5, label=label)
    axes[1].set_yscale('symlog', linthresh=1)
    axes[1].set(ylabel='Hz / 细胞', yticks=[0, 1, 10, 100, 1000], yticklabels=['0', '1', '10', '100', '1000'])
    axes[1].legend(frameon=False, ncol=3, loc='upper right', fontsize=9)
    axes[1].set_title('b  完整群体发放率（2 ms分箱；纵轴1 Hz以上取对数）', loc='left', fontsize=11)
    output_cells = groups[1]
    for cell in output_cells:
        axes[2].plot(times, voltage[:, cell], color='#aaaaaa', alpha=.42, linewidth=.7)
    highlighted = int(output_cells[np.argmax(spikes[:, output_cells].sum(axis=0))])
    axes[2].plot(times, voltage[:, highlighted], color=COLORS[1], linewidth=1.5)
    axes[2].axhline(-45, color='#333333', linestyle=':', linewidth=.8, label='发放阈值')
    axes[2].set(ylabel='膜电位 / mV', xlabel='相对神经时间 / ms（每20 ms一次决策）')
    axes[2].set_title('c  抽样DN膜电位：全部16个（灰）及本段放电最多者（橙）', loc='left', fontsize=11)
    for ax in axes:
        ax.set_xlim(0, 80)
        ax.set_xticks([0, 20, 40, 60, 80])
        for x in (20, 40, 60):
            ax.axvline(x, color='#dddddd', linestyle='--', linewidth=.6, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
    fig.text(.13, .065, '工程端口分组，非解剖脑区。重放的动作logits及末端全部状态与真实采样精确一致。', fontsize=9)
    fig.text(.13, .038, '膜电位为阈值比较前的值；虚线为发放阈值。高发放细胞仅用于示例，不代表群体典型值。', fontsize=9)
    fig.text(.13, .011, '这些图表示神经活动，不能单独证明信息被用于有效避弹。', fontsize=9)
    export(fig, output, 'neural_activity')
    return dict(trace_source=str(source), trace_sha256=digest(source), sampled_cells=64,
                neural_ticks=400, population_sizes=sizes.tolist(), rate_bin_ms=2,
                all_observed_spikes_plotted=int(spikes.sum()), dropped_ticks=0, dropped_cells=0,
                membrane_panel='all16 sampledDN;input/rest voltages remain in source NPZ',
                highlighted_output_root_id=str(replay['cell_root_ids'][highlighted]),
                highlighted_output_sample_column=highlighted,
                highlight_selection='maximum recorded spikes within the16 sampledDN,descriptive only')


def plot_horizon(reports, output):
    parameters = ('brain.input_encoder.weight', 'brain.recurrent_gain_logits', 'brain.action_readout.weight')
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(left=.13, right=.96, bottom=.22, top=.79)
    rows = []
    handles = []
    for i, (number, report) in enumerate(reports):
        h = report['credit_horizon']
        if h['status'] != 'verified_readonly_comparison' or h['pair']['loss_terms_max_abs_error'] != 0:
            raise ValueError('Only paired verified same-loss gradient comparisons are plotted')
        positions = np.arange(3) + (-.13 if i == 0 else .13)
        for j, name in enumerate(parameters):
            g = h['parameter_gradients'][name]
            y = [g['T4_norm'], g['T16_norm']]
            if not all(np.isfinite(y)) or min(y) <= 0:
                raise ValueError('Positive finite actual gradient norms required')
            ax.plot([positions[j]] * 2, y, color='#cccccc', linewidth=1, zorder=1)
            rows.append(dict(rollout=number, parameter=name, T4=y[0], T16=y[1]))
        for horizon, color in ((4, COLORS[0]), (16, COLORS[1])):
            values = [h['parameter_gradients'][name][f'T{horizon}_norm'] for name in parameters]
            ax.scatter(positions, values, marker=('o' if i == 0 else 's'), color=color,
                       facecolors=('none' if horizon == 4 else color), s=(62 if horizon == 4 else 25),
                       linewidths=1.3, zorder=3)
            handles.append(Line2D([], [], color=color, marker=('o' if i == 0 else 's'), linestyle='none',
                                  markerfacecolor=('none' if horizon == 4 else color),
                                  label=f'{horizon}步 / 第{number}批'))
    ax.set(yscale='log', ylim=(.01, 1e22), xlim=(-.5, 2.5),
           xticks=[0, 1, 2], xticklabels=['感觉输入权重', '脑内神经元增益', '动作读出权重'],
           ylabel='梯度 L2 范数（对数轴）')
    ax.set_yticks([1e-2, 1e2, 1e6, 1e10, 1e14, 1e18, 1e22])
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#eeeeee', linewidth=.7)
    fig.suptitle('前向结果一致，长链代理梯度严重放大', y=.97, fontsize=15)
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.53, .9), fontsize=9)
    fig.text(.13, .105, '每段比较相同输入、参数、随机键和最后4步损失；16步多反传12步历史。', fontsize=9)
    fig.text(.13, .06, '两段示例不是独立统计重复。只读对照未进入Adam；同一片段的动作读出梯度重合。', fontsize=9)
    export(fig, output, 'gradient_horizon')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import psutil
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[-2:])
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(font)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'svg.fonttype': 'none', 'font.size': 10})
    args.output.mkdir(parents=True, exist_ok=True)
    reports = [(n, read_shared_json(args.run / 'neural-diagnostics' / f'rollout-{n:06}' / 'report.json'))
               for n in (8, 32)]
    activity = plot_activity(reports[0][1], args.output)
    gradients = plot_horizon(reports, args.output)
    assert 'torch' not in sys.modules
    manifest = dict(schema='fly-neural-exploratory-plots-v1', Torch_imported=False,
                    GPU=False, native_worlds=0, optimizer_updates=0, activity=activity,
                    gradient_pairs=gradients, source_sha256=digest(__file__),
                    exclusions='none within the declared64-cell/400-tick sample',
                    uncertainty='individual traces,not confidence intervals;two gradient examples,no sampling inference',
                    source_reports={str(n):digest(args.run/'neural-diagnostics'/f'rollout-{n:06}'/'report.json')
                                    for n,_ in reports})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),figures=['neural_activity.png','gradient_horizon.png'],
                         GPU=False,Torch_imported=False),ensure_ascii=False))


if __name__ == '__main__':
    main()
