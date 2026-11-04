"""Read a completed avoidance-guidance pilot and paired frozen baselines only."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json

GUIDANCE_SCHEMA='continuous-object-NoBomb-guidance-entry-v1'
PPO_SCHEMA='continuous-object-NoBomb-survival-PPO-entry-v1'
ACTION_QUERY_SCHEMA='continuous-action-query-object-NoBomb-survival-PPO-entry-v1'
FULL_SENSORY_SCHEMA='continuous-full-sensory-NoBomb-survival-PPO-entry-v1'


def quantile(values, p):
    values = sorted(values)
    k = (len(values)-1)*p
    a, b = math.floor(k), math.ceil(k)
    return values[a]+(values[b]-values[a])*(k-a)


def completed_avoidance_report(candidate):
    """PPO families retain the original three-value/survival-only checks."""
    report=read_shared_json(candidate/'result.json')
    assert report['status'] == 'completed'
    assert report['schema'] in (GUIDANCE_SCHEMA,PPO_SCHEMA,ACTION_QUERY_SCHEMA,FULL_SENSORY_SCHEMA)
    if report['schema'] == GUIDANCE_SCHEMA:
        assert report['PPO_updates'] == report['value_updates'] == 0
    else:
        assert report['full_story_PPO_resumed'] is False
        assert report['algorithm']['PPO']['objective_weights'] == [1.,0.,0.]
        assert report['algorithm']['all_three_value_heads_retained_and_updated'] is True
    e = report['development']
    assert e['episodes'] == 20 and e['actual_Bombs'] == 0 and e['no_frame_cap']
    assert e['optimizer_updates'] == e['teacher_actions'] == 0
    assert e['action_selection'] == 'greedy' and e['action_temperature'] == 1.
    rows = sorted(e['episode_results'], key=lambda x:x['evaluation_game'])
    assert len(rows) == 20 and len({r['evaluation_game'] for r in rows}) == 20
    assert all(r['first_collision_observed'] != r['right_censored_at_natural_clear'] for r in rows)
    return report,e,rows


def paired_frozen_evaluations(e,ref,name):
    rows=sorted(e['episode_results'],key=lambda r:r['evaluation_game'])
    base=sorted(ref['episode_results'],key=lambda r:r['evaluation_game'])
    assert ref['episode_plan']==e['episode_plan'] and len(base)==20
    assert ref['actual_Bombs']==0 and ref['no_frame_cap']
    wins=losses=ties=ambiguous=0;deltas=[];details=[]
    for a,b in zip(rows,base):
        for key in ('evaluation_game','evaluation_seed','action_seed','native_seed'):
            assert a[key]==b[key],(name,key)
        assert a['seed_applied'] and b['seed_applied']
        assert a['actual_bombs_used']==b['actual_bombs_used']==0
        alo=a['native_frames'];blo=b['native_frames']
        ahi=math.inf if a['right_censored_at_natural_clear'] else alo
        bhi=math.inf if b['right_censored_at_natural_clear'] else blo
        if alo>bhi:wins+=1;order='known_longer'
        elif ahi<blo:losses+=1;order='known_earlier'
        elif ahi==alo==bhi==blo:ties+=1;order='tied'
        else:ambiguous+=1;order='censoring_ambiguous'
        exact=math.isfinite(ahi) and math.isfinite(bhi)
        if exact:deltas.append(alo-blo)
        details.append(dict(game=a['evaluation_game'],native_seed=a['native_seed'],candidate_endpoint=alo,
            reference_endpoint=blo,candidate_censored=a['right_censored_at_natural_clear'],
            reference_censored=b['right_censored_at_natural_clear'],order=order,
            contact_delta_if_both_observed=alo-blo if exact else None))
    vals=[r['native_frames'] for r in base]
    reference=dict(endpoint_median_lower_bound=statistics.median(vals),endpoint_q25_lower_bound=quantile(vals,.25),
        censored_n=sum(r['right_censored_at_natural_clear'] for r in base),
        stage1_no_contact=sum(r['first_stage_without_observed_contact'] for r in base))
    paired=dict(known_longer=wins,known_earlier=losses,tied=ties,ambiguous=ambiguous,
        contact_delta_mean_if_both_observed=statistics.mean(deltas) if deltas else None,both_observed_count=len(deltas))
    return paired,reference,details


def summarize(candidate, baselines, output, parent_evaluation=None):
    report,e,rows=completed_avoidance_report(candidate)
    endpoint_lower = [r['native_frames'] for r in rows]
    censor = sum(r['right_censored_at_natural_clear'] for r in rows)
    references = {}
    pairs = {}
    evaluations = [(name,read_shared_json(baselines/name/'result.json')['evaluation'])
        for name in ('initial','current','D95')]
    if parent_evaluation is not None:
        parent = read_shared_json(parent_evaluation)
        assert parent['status'] == 'completed'
        evaluations.append(('training_parent',parent.get('development') or parent['evaluation']))
    for name,ref in evaluations:
        pairs[name],references[name],_=paired_frozen_evaluations(e,ref,name)
    median = statistics.median(endpoint_lower)
    q25 = quantile(endpoint_lower,.25)
    stage1 = sum(r['first_stage_without_observed_contact'] for r in rows)
    d95 = references['D95']
    # Natural clears make these conservative lower bounds on contact-free time.
    # This is an engineering screen, not statistical equivalence or independent
    # confirmation; development seeds were already used for model selection.
    near = (median >= .9*d95['endpoint_median_lower_bound']
        and q25 >= .9*d95['endpoint_q25_lower_bound']
        and (d95['stage1_no_contact']-stage1)/20 <= .10)
    result = dict(schema='object-avoidance-guidance-actual20-review-v1',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        candidate_path=str(candidate), costs=report['costs'],learning_method=report['schema'],
        training_parent_evaluation=None if parent_evaluation is None else str(parent_evaluation),
        candidate_source_SHA=report['provenance']['parent']['sha256'],
        candidate=dict(episodes=20, natural_clear_right_censored=censor,
            first_contact_median=median if not censor else None,
            first_contact_q25=q25 if not censor else None,
            event_free_median_lower_bound=median, event_free_q25_lower_bound=q25,
            first_stage_without_contact=stage1, actual_Bombs=0),
        baselines=references, paired=pairs, actual_seeds_matched=True,
        preregistered_near_D95_development_screen=near,
        independent_new20_confirmation_completed=False,
        fullstory_PPO_resume_allowed=False,
        scope='Frozen development20;NOT training episodes or formal stability. Natural clear is censored,never an invented hit. Near-D95 is a conservative engineering screen;new20 matched D95 confirmation still required.',
        model_forwards=0, native_actions=0, optimizer_updates=0,
        source_script_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True,exist_ok=False)
    (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    text = ['# 对象感知避弹专项：冻结20局', '',
        f'首次无触弹时间中位下界：{median}帧；P25下界：{q25}帧；第一面无伤：{stage1}/20；无伤通关删失：{censor}。',
        '', '| 比较对象 | 已知更晚触弹 | 已知更早触弹 | 相同 | 删失后不能排序 |',
        '|---|---:|---:|---:|---:|']
    for name,p in pairs.items():
        text.append(f"| {name} | {p['known_longer']} | {p['known_earlier']} | {p['tied']} | {p['ambiguous']} |")
    text += ['',f'预设接近D95的开发筛选：{near}。独立新20局确认尚未完成，通关PPO仍不恢复。',
        '这是无Bomb首次原生接触开发评测，与允许Bomb的完整通关率不同；不作小样本显著性或统计等价主张。']
    (output/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return result


def summarize_matched_candidates(control,candidate,output):
    """One object control and one query candidate;no ranking/selection of runs."""
    cr,ce,crows=completed_avoidance_report(control)
    qr,qe,qrows=completed_avoidance_report(candidate)
    assert cr['schema']==PPO_SCHEMA and qr['schema']==ACTION_QUERY_SCHEMA
    cl=read_shared_json(control/'launch.json');ql=read_shared_json(candidate/'launch.json')
    assert cl['schema']==cr['schema'] and ql['schema']==qr['schema']
    parent=cl['training_source_sha256'];assert parent==ql['training_source_sha256']
    assert cr['provenance']['parent']['sha256']==qr['provenance']['parent']['sha256']==parent
    fields=('environment_config_sha256','student_decisions','rollouts','worlds','steps_per_world',
        'neural_groups','K','brain_BPTT','PPO','training_seed','evaluation_seed','action_support',
        'task_end','task_reward','guidance','policy_cost_gradient')
    matched={}
    for key in fields:
        assert key in cl and key in ql and cl[key]==ql[key],('matched control differs',key)
        matched[key]=cl[key]
    for launch,report in ((cl,cr),(ql,qr)):
        assert report['costs']['student_actions']==launch['student_decisions']
        assert report['costs']['completed_rollouts']==launch['rollouts']
    pairs,_,details=paired_frozen_evaluations(qe,ce,'query_minus_same_parent_control')
    def stats(report,rows):
        values=[r['native_frames'] for r in rows];censor=sum(r['right_censored_at_natural_clear'] for r in rows)
        return dict(learning_method=report['schema'],costs=report['costs'],episodes=20,
            event_free_mean_lower_bound=statistics.mean(values),event_free_median_lower_bound=statistics.median(values),
            event_free_q25_lower_bound=quantile(values,.25),natural_clear_right_censored=censor,
            first_contact_median=None if censor else statistics.median(values),
            stage1_no_contact=sum(r['first_stage_without_observed_contact'] for r in rows),actual_Bombs=0)
    refs=[]
    for folder in (control,candidate):
        for name in ('launch.json','result.json'):
            path=folder/name;refs.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    missing=[key for key in ('seed','action_seed') if key not in cl or key not in ql]
    for key in ('seed','action_seed'):
        if key not in missing:assert cl[key]==ql[key],key;matched[key]=cl[key]
    result=dict(schema='same-parent-action-query-NoBomb-paired-development20-review-v1',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),common_parent_checkpoint_sha256=parent,
        control_path=str(control),candidate_path=str(candidate),matched_training_controls=matched,
        training_RNG_fields_not_recorded_by_launch=missing,actual_evaluation_seeds_matched=True,
        control=stats(cr,crows),candidate=stats(qr,qrows),paired_query_minus_control=pairs,paired_games=details,
        paid_Adam_may_differ_with_same_KL_early_stop=True,initial_function_equivalence_tested_here=False,
        source_refs=refs,source_script_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Same-parent/equal-decision-budget frozen NoBomb20 development comparison. Natural clear censored;not historical Bomb-allowed75percent or statistical equivalence. Training action/minibatch seeds are not claimed matched unless launch records them. No candidate selection or formal confirmation.',
        model_forwards=0,native_actions=0,optimizer_updates=0)
    output.mkdir(parents=True,exist_ok=False)
    (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    text=['# 同父模型：ActionQuery与无Query control配对开发20局','',
        '| 模型 | 实付PPO Adam | 决策 | 无触弹中位下界 | P25下界 | 第一面无伤 | 自然clear删失 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in ('control','candidate'):
        r=result[name];text.append(f"| {name} | {r['costs']['PPO_Adam']} | {r['costs']['student_actions']} | {r['event_free_median_lower_bound']} | {r['event_free_q25_lower_bound']} | {r['stage1_no_contact']}/20 | {r['natural_clear_right_censored']} |")
    text+=['',f"Query相对control：已知更晚{pairs['known_longer']}、更早{pairs['known_earlier']}、相同{pairs['tied']}、删失不能排序{pairs['ambiguous']}。", 
        '父checkpoint SHA、预算、PPO设置及实际评测种子严格核对。实际Adam可因同一KL规则提前停止而不同。',
        '自然clear不伪造触弹；未记录的训练随机字段不声称相同。此比较不是统计等价、独立稳定验收或自动部署依据。']
    (output/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--baselines',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parent-evaluation',type=Path)
    p.add_argument('--matched-control',type=Path)
    a = p.parse_args()
    if a.matched_control:
        if a.baselines or a.parent_evaluation:p.error('Matched-control mode uses only the two completed candidates')
        result=summarize_matched_candidates(a.matched_control,a.candidate,a.output)
        print(json.dumps({k:result[k] for k in ('control','candidate','paired_query_minus_control')},ensure_ascii=False))
    else:
        if a.baselines is None:p.error('--baselines is required for the original baseline review')
        result = summarize(a.candidate,a.baselines,a.output,a.parent_evaluation)
        print(json.dumps({k:result[k] for k in ('costs','candidate','paired','preregistered_near_D95_development_screen')},ensure_ascii=False))
