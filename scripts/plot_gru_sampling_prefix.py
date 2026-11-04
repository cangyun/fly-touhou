"""Plot an engineering diagnostic at a fixed native sampling milestone.

Only sealed, complete training batches through the first checkpoint reaching the
requested milestone are used. These are on-policy training curves, not test
outcomes or evidence for G2 acceptance. No training process is modified.
"""
from pathlib import Path
import argparse
import csv
import json
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.experiment import file_sha256
from fly_touhou.native_diagnostic import write_json


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def seal(path):return {'path':str(path),'sha256':file_sha256(path)}


def collect(run,seed,label,milestone):
    training=run/f'seed{seed}/train'
    progress=read(training/'progress.json')
    if progress['counters']['training_decisions']<milestone:
        raise ValueError(f'{label} seed{seed} has not reached fixed milestone{milestone}')
    assert file_sha256(progress['latest']['path'])==progress['latest']['sha256']
    rows=[];sources=[];recent=[];fit_seconds=0.;no_hit=0
    for update in range(1,progress['counters']['updates']+1):
        path=training/f'update_{update:04d}.json'
        data=read(path);sources.append(seal(path))
        counters=data['counters']
        assert counters['updates']==update
        assert len(data['episodes'])==4
        recent.extend(e['frames'] for e in data['episodes'])
        no_hit+=sum(bool(e['horizon_completed_without_hit']) for e in data['episodes'])
        fit_seconds+=data['fit_wall_seconds']
        window=recent[-64:]
        rows.append({'experiment':label,'seed':seed,'update':update,
            'actual_decisions':counters['training_decisions'],
            'actual_native_frames':counters['training_native_frames'],
            'training_episodes':counters['training_episodes'],
            'actor_accepted_batches':counters['accepted_actor_updates'],
            'actor_optimizer_steps':counters.get('actor_optimizer_steps',counters['accepted_actor_updates']),
            'mean_last_at_most64_training_episodes':statistics.mean(window),
            'median_last_at_most64_training_episodes':statistics.median(window),
            'window_episodes':len(window),'training_nohit_1024_so_far':no_hit,
            'cumulative_reported_fit_seconds':fit_seconds})
        if counters['training_decisions']>=milestone:break
    if rows[-1]['actual_decisions']<milestone:raise ValueError('No sealed complete batch reaches milestone')
    return rows,sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--milestone',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Keep each diagnostic snapshot')
    if args.milestone not in (4096,8192,16384,32768):
        raise ValueError('Use a fixed power-of-two diagnostic sampling milestone')
    results={};sources=[];flat=[]
    for seed in (11,29,47):
        results[seed]={}
        for label,run in (('one-pass',args.base),('four-pass',args.candidate)):
            rows,bindings=collect(run,seed,label,args.milestone)
            results[seed][label]=rows;flat.extend(rows);sources.extend(bindings)
    args.output.mkdir(parents=True)
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['font.size']=10
    fig,axes=plt.subplots(2,3,figsize=(14,7.4),sharex=True,sharey='row')
    colors={'one-pass':'#777777','four-pass':'#0072B2'}
    labels={'one-pass':'每批最多1遍','four-pass':'每批最多4遍'}
    extent=max(r['actual_decisions'] for r in flat)
    for column,seed in enumerate((11,29,47)):
        for label,rows in results[seed].items():
            x=[r['actual_decisions'] for r in rows]
            axes[0,column].plot(x,[r['mean_last_at_most64_training_episodes'] for r in rows],
                color=colors[label],label=labels[label],linewidth=2,marker='.',markersize=4)
            axes[1,column].plot(x,[r['actor_optimizer_steps'] for r in rows],
                color=colors[label],label=labels[label],linewidth=2,marker='.',markersize=4)
        axes[0,column].set_title(f'训练种子 {seed}')
        axes[0,column].axhline(1024,linestyle=':',color='#BA4545',linewidth=1)
        axes[0,column].set_ylim(0,1080)
        axes[1,column].set_xlabel('实际原作决策数（每次4原生帧）')
        for row in range(2):
            ax=axes[row,column]
            ax.set_xlim(0,extent*1.025)
            ax.axvline(args.milestone,color='#BBBBBB',linestyle='--',linewidth=.8)
            ax.grid(alpha=.18)
            ax.spines[['top','right']].set_visible(False)
    axes[0,0].set_ylabel('训练局首次中弹/截止帧数\n最近至多64局均值')
    axes[1,0].set_ylabel('累计实际策略优化步数')
    axes[0,0].legend(loc='upper left',frameon=False)
    fig.suptitle(f'普通GRU训练诊断｜固定 {args.milestone:,} 次采样节点',fontsize=16,y=.99)
    fig.text(.5,.035,'仅训练曲线，不代表独立测试。原作RNG未控制；每条曲线保留首次达到节点的完整4局批次。',
             ha='center',fontsize=10)
    fig.text(.5,.008,'上排：原作生存表现；下排：优化工作量，不是成绩。无Bomb、持续射击、首次中弹结束。',
             ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.065,1,.95))
    png=args.output/'training_prefix.png'
    fig.savefig(png,dpi=150,facecolor='white')
    plt.close(fig)
    csv_path=args.output/'training_prefix.csv'
    with csv_path.open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    summary={str(seed):{label:rows[-1] for label,rows in data.items()} for seed,data in results.items()}
    result={'schema':'native-GRU-fixed-sampling-prefix-training-plot-v1','milestone':args.milestone,
        'selection':'first sealed complete batch at or above fixed milestone;no best-checkpoint selection',
        'snapshot_endpoints':summary,'source_metrics':sources,'script':seal(Path(__file__)),
        'figure':seal(png),'csv':seal(csv_path),'heldout_data_read':False,'new_native_actions':0,
        'scope':'training trajectories and optimizer work only;different visited states and originalRNG;not a paired causal estimate',
        'G2_achieved':False,'final_acceptance':False}
    write_json(args.output/'report.json',result,exclusive=True)
    print(json.dumps({'figure':str(png),'endpoints':summary,'G2_achieved':False},ensure_ascii=False))


if __name__=='__main__':main()
