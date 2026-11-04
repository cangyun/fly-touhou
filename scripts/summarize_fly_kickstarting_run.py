"""NoTorch read-only summary of canonically committed Kickstarting rollouts."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import tensorboard_live as tb  # Definitions only; its main/service is guarded.
from fly_touhou.thrl_dodge_diagnostics import _shared_binary
if sys.modules.get('torch') is not None:raise RuntimeError('Summary must not import Torch')
PILOT=Path('./outputs/distillation/D95-fly-visual-Kickstarting-PPO4-pilot-v1')

def now():return datetime.now(timezone.utc).isoformat()
def sha(data):return hashlib.sha256(data).hexdigest()
def numeric(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def mean(values):
    values=[x for x in values if numeric(x)]
    return float(np.mean(values)) if values else None
def fmt(x):return f'{x:.5g}' if numeric(x) else '—'

def case(item):
    g=item.get('summary',item);m=g.get('movement_diagnostics',{});d=g.get('reward_diagnostics',{})
    hit=m.get('first_collision_observed_frames',g.get('first_collision_observed_frames'))
    return dict(worker=item.get('worker'),game_id=item.get('game_id'),native_seed=g.get('native_seed',g.get('full_story_native_seed')),
        outcome=g.get('outcome'),frames=g.get('native_frames',g.get('total_frames')),first_hit=hit,
        first_hit_status='observed' if numeric(hit) else 'unknown_or_censored',
        stage=max(g['stage_history']) if g.get('stage_history') else g.get('stage'),Bomb=g.get('bombs_used'),Miss=g.get('misses'),
        confirmed_DB=d.get('confirmed_deathbombs'),ordinary_or_unconfirmed_Bomb=d.get('ordinary_or_unconfirmed_bombs'))

def episode_stats(cases):
    known=[c['first_hit'] for c in cases if numeric(c['first_hit'])]
    return dict(complete_n=len(cases),outcomes=dict(Counter(c['outcome'] for c in cases)),
        stage_counts=dict(Counter(str(c['stage']) for c in cases)),frames_mean=mean([c['frames'] for c in cases]),
        Bomb_mean=mean([c['Bomb'] for c in cases]),first_hit_observed_n=len(known),
        first_hit_unknown_or_censored_n=len(cases)-len(known),first_hit_observed_mean=mean(known))

def fixed_development(result,launch):
    d=result.get('development') or {};games=d.get('episode_results') or []
    expected=launch.get('training_then_shutdown_then_fixed_last_development_games',d.get('episodes'))
    complete=(result.get('status')=='completed' and len(games)==expected==d.get('episodes')
        and d.get('policy_frozen') is True and d.get('native_seed_split')=='development'
        and all(d.get(k)==0 for k in ('PPO_updates','teacher_actions','safety_takeovers','continues'))
        and all(g.get('outcome') in ('gameover','clear') for g in games))
    cases=[case(g) for g in games] if complete else []
    return dict(status='complete_fixed_development' if complete else 'not_complete_or_unverified',
        seed_master=launch.get('development_seed'),expected_games=expected,formal_stability_test=False,
        stats=episode_stats(cases),cases=cases)

def summarize(run, *, max_rollouts=None):
    if max_rollouts is not None and (type(max_rollouts) is not int or max_rollouts<1):
        raise ValueError('Positive integer fixed review boundary required')
    started=now();refs={};warnings=[]
    def raw(path):
        with _shared_binary(path) as stream:data=stream.read()
        refs[str(path.resolve())]=dict(sha256=sha(data),bytes=len(data),read_utc=now())
        return data
    def read(path):return json.loads(raw(path).decode('utf-8-sig'))  # shared read/write/delete handle
    launch=read(run/'launch.json');has_result=(run/'result.json').exists()
    canonical_path=run/('result.json' if has_result else 'progress.json')
    canonical=read(canonical_path);canonical_limit=int(canonical.get('costs',{}).get('completed_rollouts',0))
    limit=canonical_limit if max_rollouts is None else min(max_rollouts,canonical_limit)
    rows=[];games={};unscored=0;pending=[];adam=0;review_costs={}
    for path in sorted((run/'traces').glob('update-*.json'),key=lambda p:int(p.stem.split('-')[-1])):
        number=int(path.stem.split('-')[-1])
        if number>limit:continue
        try:
            u=read(path)
            if u.get('costs',{}).get('completed_rollouts')!=number:raise ValueError('update ordinal differs')
            observation=run/'traces'/f'rollout-{number-1:03d}.json';o=read(observation)
            payload=raw(observation.with_suffix('.npz'))
            if o.get('trace_sha256') and sha(payload)!=o['trace_sha256']:raise ValueError('trace SHA differs')
            with np.load(io.BytesIO(payload),allow_pickle=False) as a:
                p=a['teacher_probs'];q=a['action_probs'] if 'action_probs' in a.files else None;legal=a['legal_actions']
            n=p.reshape(-1,19).shape[0];overall=tb.fly_policy_probability_metrics(p,q,legal)
            infos=[e for step in o.get('step_info',[]) for e in step]
            known=np.zeros(n,bool);vulnerable=np.zeros(n,bool)
            if len(infos)==n:
                known=np.asarray([e.get('protected_decision') in (0,1) for e in infos])
                vulnerable=np.asarray([e.get('protected_decision')==0 for e in infos])
            split=tb.fly_policy_probability_metrics(p.reshape(-1,19)[vulnerable],
                q.reshape(-1,19)[vulnerable] if q is not None else None,legal.reshape(-1,19)[vulnerable]) if vulnerable.any() else {}
            greedy=np.bincount(q.reshape(-1,19).argmax(-1),minlength=19).tolist() if q is not None else None
            terms=u.get('learning',{}).get('terms',[])
            weights=np.asarray([t.get('real_rows',n/len(terms)) for t in terms]) if terms else []
            loss={k:float(np.average([t[k] for t in terms],weights=weights)) for k in
                ('total_loss','policy_loss','value_loss','teacher_KL','teacher_CE','entropy','approximate_KL','clip_fraction','brain_regularization')
                if terms and all(numeric(t.get(k)) for t in terms)}
            components=[e for step in o.get('reward_components',[]) for e in step]
            rows.append(dict(rollout=number,rows=n,decisions=u.get('costs',{}).get('student_actions_executed'),
                overall=overall,vulnerable=split,behavior_probability_vectors_saved=q is not None,vulnerable_rows=int(vulnerable.sum()),protection_known_rows=int(known.sum()),
                protected_fraction=1-float(vulnerable.sum()/known.sum()) if known.any() else None,
                full_vulnerable_reward_rows=sum(e.get('vulnerable_progress',0)>=.0009999 for e in components),
                greedy_counts=greedy,greedy_majority_action=int(np.argmax(greedy)) if greedy else None,
                greedy_concentration=max(greedy)/n if greedy else None,weighted_learning_terms=loss,
                learning_metrics_note='real-row weighted;successive minibatch parameters,not one frozen-policy KL'))
            adam+=int(u.get('learning',{}).get('optimizer_updates',0))
            review_costs=u.get('costs',{})
            for item in o.get('completed_episodes',u.get('completed_native_episodes',[])):
                g=item.get('summary',item)
                if g.get('outcome') not in ('gameover','clear') or g.get('full_story_episode') is not True:
                    unscored+=1;continue
                c=case(item);c['completed_rollout']=number;key=(c['worker'],c['game_id'],g.get('native_episode'))
                games[key]=c
        except Exception as exc:warnings.append(dict(file=str(path),error=repr(exc)))
    for path in sorted((run/'traces').glob('rollout-*.json'),key=lambda p:int(p.stem.split('-')[-1])):
        number=int(path.stem.split('-')[-1])+1
        if number>canonical_limit:
            try:
                o=read(path);pending.append(dict(rollout=number,sampled_rows=o.get('metadata',{}).get('student_actions_executed'),
                    has_update_file=(run/'traces'/f'update-{number:03d}.json').exists()))
            except Exception as exc:warnings.append(dict(file=str(path),error=repr(exc)))
    fixed=fixed_development(canonical if has_result and limit==canonical_limit else {},launch);comparison=None
    if fixed['status']=='complete_fixed_development' and fixed['seed_master']==2027431601 and run.resolve()!=PILOT.resolve():
        try:
            pilot=read(PILOT/'result.json');reference=fixed_development(pilot,pilot['launch'])
            if reference['status']=='complete_fixed_development' and reference['seed_master']==fixed['seed_master']:
                comparison=dict(pilot4=reference,current=fixed,descriptive_only=True,significance_claimed=False,
                    same_native_seed_set=sorted(c['native_seed'] for c in reference['cases'])==sorted(c['native_seed'] for c in fixed['cases']))
        except Exception as exc:warnings.append(dict(file=str(PILOT/'result.json'),error=repr(exc)))
    training=list(games.values());source_sha={str(Path(__file__).resolve()):sha(raw(Path(__file__))),str(Path(tb.__file__).resolve()):sha(raw(Path(tb.__file__)))}
    return dict(schema='fly-kickstarting-readonly-run-summary-v1',status='final' if fixed['status']=='complete_fixed_development' else 'interim',
        run=str(run.resolve()),read_started_utc=started,read_finished_utc=now(),canonical_source=str(canonical_path),
        canonical_completed_rollouts=canonical_limit,canonical_costs=canonical.get('costs',{}),phase=canonical.get('phase',canonical.get('status')),
        requested_review_rollouts=max_rollouts,review_completed_rollout_boundary=limit,review_costs=review_costs,
        committed_rollouts_outside_review_scope=canonical_limit-limit,
        review_scope_complete=len(rows)==limit and (max_rollouts is None or canonical_limit>=max_rollouts),
        summarized_committed_rollouts=len(rows),committed_trace_Adam=adam,trace_summary_complete=len(rows)==canonical_limit,
        sampled_not_canonically_committed=pending,rollouts=rows,
        changing_policy_training=dict(stats=episode_stats(training),cases=training,unscored_records=unscored,
            fixed_model_test=False,unfinished_worlds_never_counted_as_gameover=True),fixed_development=fixed,pilot4_comparison=comparison,
        source_sha256=source_sha,metric_function_source_sha256=sha(inspect.getsource(tb.fly_policy_probability_metrics).encode()),
        metric_definition='same TensorBoard inventory-conditioned constant/KL math;log floor1e-30;vulnerable=known protected_decision0,not identical to positive survival reward',
        first_hit_None_policy='unknown_or_censored;never zero-filled;means use observed subset only',read_files=refs,warnings=warnings,
        Torch_imported=sys.modules.get('torch') is not None,model_forwards=0,GPU=False,native_worlds=0,optimizer_updates=0)

def markdown(s):
    t=s['changing_policy_training']['stats'];f=s['fixed_development'];lines=[f"# {s['status'].upper()}：Kickstarting运行汇总",'',
        f"读取边界 {s['read_started_utc']} → {s['read_finished_utc']}；canonical={Path(s['canonical_source']).name}，完成{s['canonical_completed_rollouts']} rollout，汇总{s['summarized_committed_rollouts']}。",
        f"本报告固定到rollout {s['review_completed_rollout_boundary']}；范围完整={s['review_scope_complete']}；另有{s['committed_rollouts_outside_review_scope']}个已提交rollout在报告范围之外（不标为未提交）。",
        f"已采样未canonical提交：{s['sampled_not_canonically_committed']}（不计训练/Adam表）；读取警告{len(s['warnings'])}。",'',
        f"训练完整自然局（持续变参，非固定模型测试）：{t['complete_n']}局，{t['outcomes']}，stage={t['stage_counts']}；平均帧{fmt(t['frames_mean'])}/Bomb{fmt(t['Bomb_mean'])}；首碰观测{t['first_hit_observed_n']}，未知/截尾{t['first_hit_unknown_or_censored_n']}，仅已观测均值{fmt(t['first_hit_observed_mean'])}。",'',
        '|rollout|新决策|overall ΔKL|易伤 ΔKL(n)|greedy最集中动作/比例|保护比例(已知n)|clip/approxKL|P/V/teacherKL/H|',
        '|---:|---:|---:|---:|---|---|---|---|']
    for r in s['rollouts']:
        l=r['weighted_learning_terms'];lines.append(f"|{r['rollout']}|{r['decisions']}|{fmt(r['overall'].get('KL_minus_same_window_constant'))}|{fmt(r['vulnerable'].get('KL_minus_same_window_constant'))}({r['vulnerable_rows']})|{r['greedy_majority_action']}/{fmt(r['greedy_concentration'])}|{fmt(r['protected_fraction'])}({r['protection_known_rows']})|{fmt(l.get('clip_fraction'))}/{fmt(l.get('approximate_KL'))}|"+'/'.join(fmt(l.get(k)) for k in ('policy_loss','value_loss','teacher_KL','entropy'))+'|')
    lines+=['',f"固定末模型开发评估：{f['status']}；未完成/未验证时不公布GO或clear成绩。"]
    if f['status']=='complete_fixed_development':
        lines+=['','|固定策略|局数/结果|平均帧|首碰观测均值(未知数)|平均Bomb|stage|','|---|---|---:|---|---:|---|']
        variants=[('本次',f)]+([('pilot4同master',s['pilot4_comparison']['pilot4'])] if s['pilot4_comparison'] else [])
        for name,v in variants:
            a=v['stats'];lines.append(f"|{name}|{a['complete_n']}/{a['outcomes']}|{fmt(a['frames_mean'])}|{fmt(a['first_hit_observed_mean'])}({a['first_hit_unknown_or_censored_n']})|{fmt(a['Bomb_mean'])}|{a['stage_counts']}|")
        lines+=['','本次逐局：'+str(f['cases']),'仅描述性并排，无小样本显著性承诺；同种子不保证动作分叉后轨迹逐帧相同。']
    lines+=['','ΔKL=teacher→实际behavior KL−同窗最佳库存条件常数；负数仅说明条件拟合胜基线，不等于避弹。易伤mask与完整正存活奖励条件不同。loss/clip为更新期混合θ的真实行加权均值，不能求和当最终KL。',
        'None首碰保留未知/截尾；未完局不填GO。缺action_probs时不计算behavior ΔKL/greedy，不用old_logp或采样动作冒充。完整每局与数据/源码SHA见配套JSON；未读取checkpoint或运行Torch/GPU/模型/原作/训练。']
    return '\n'.join(lines)+'\n'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-rollouts',type=int);a=p.parse_args()
    s=summarize(a.run,max_rollouts=a.max_rollouts);base=a.output.with_suffix('') if a.output.suffix.lower() in ('.json','.md') else a.output/'summary'
    paths=[base.with_suffix('.json'),base.with_suffix('.md')]
    if any(str(path.resolve()) in s['read_files'] or path.resolve().is_relative_to((a.run/'traces').resolve()) or (path.resolve().is_relative_to(a.run.resolve()) and base.name!='summary') for path in paths):raise ValueError('Output cannot overwrite input/active records;use an external folder or summary.json/.md')
    for path,data in zip(paths,(json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)+'\n',markdown(s))):
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(data,encoding='utf-8');os.replace(tmp,path)
    print(json.dumps(dict(status=s['status'],completed_rollouts=s['canonical_completed_rollouts'],summary_rollouts=s['summarized_committed_rollouts'],fixed_development=s['fixed_development']['status'],output=[str(p) for p in paths]),ensure_ascii=False))

if __name__=='__main__':main()
