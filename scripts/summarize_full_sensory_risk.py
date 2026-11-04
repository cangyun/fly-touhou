"""Prepared JSON-only review of one completed risk-only stage and frozen20.

No checkpoint/model is loaded. Hash receipts describe producer-verified tensor
and Adam identity;this consumer verifies those JSON receipts,not the PT again.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.thrl_study_io import read_shared_json
from fly_touhou.continuous_risk_optimizer_transition import validate_risk_optimizer_transition
from summarize_object_avoidance import paired_frozen_evaluations,quantile

ENTRY_SCHEMA='continuous-full-sensory-risk-only-NoBomb-stage-entry-v1'
UPDATE_SCHEMA='continuous-full-sensory-risk-only-signed4f-update-v1'


def require(condition,message):
    if not condition:raise ValueError(message)


def validate_risk_stage(report,launch,updates,before,after):
    """Validate this new phase only;the existing PPO checker is untouched."""
    require(report.get('schema')==launch.get('schema')==ENTRY_SCHEMA,'risk-only entry schema')
    require(report.get('status')=='completed','risk stage is not completed')
    require(report.get('PPO_updates')==report.get('value_updates')==0,'PPO/value must be zero')
    require(report.get('teacher_actions_executed')==0,'teacher executed an action or is unknown')
    require(report.get('actor_critic_parameters_and_Adam_exact') is True,'actor/three-critic identity not verified')
    require(bool(before) and before==after,'actor/critic before/after hash receipts differ')
    require(set(before)=={'actor','critic'},'identity receipt must cover actor and all three critics')
    critics=before['critic']['state'];critic_Adam=before['critic']['Adam']
    require(len(critic_Adam)==18 and set(critic_Adam)==set(critics)
        and all('heads.'+str(i)+'.'+key in critics for i in range(3) for key in ('weight','bias')),
        'all three independent critic heads and their complete Adam entries required')
    require(bool(before['actor']['state']) and set(before['actor']['state'])==set(before['actor']['Adam']),
        'complete actor parameter/Adam entries required')
    branches=launch.get('perception_branches') or {}
    require(all(branches.get(k) is True for k in ('map','spatial','numeric','object','query')),
        'all five perception branches must be trainable')
    algorithm=report.get('algorithm') or {}
    transition=validate_risk_optimizer_transition(report.get('actual_transition'),expected_mode=algorithm.get('optimizer_mode'))
    require(transition==launch.get('actual_transition')==algorithm.get('optimizer_transition'),
        'actual optimizer transition differs across launch/algorithm/result')
    clip=algorithm.get('max_grad_norm')
    require(type(clip) in (int,float) and clip in (.5,5.),'explicit supported clip required')
    loss=algorithm.get('loss') or {}
    require(loss.get('field')=='loss' and loss.get('coefficient')==1.,'unscaled physical risk objective required')
    require(all(loss.get(k) is False for k in ('PPO','guide','value','ranking','GAE')),
        'risk phase must exclude PPO/guide/value/ranking/GAE')
    costs=report['costs'];paid=0
    require(type(costs.get('supervised_Adam')) is int and costs['supervised_Adam']>=0,'actual supervised Adam count')
    require('PPO_Adam' not in costs,'inherited PPO cost belongs in cumulative_counters,not this phase')
    require(len(updates)==costs.get('completed_rollouts'),'committed rollout count mismatch')
    for ordinal,update in enumerate(updates,1):
        learning=update['learning']
        require(learning.get('schema')==UPDATE_SCHEMA,'risk-only updater schema')
        require(learning.get('epochs_completed')==learning.get('epochs_requested')==3,'three physical epochs required')
        require(learning.get('max_grad_norm')==clip,'update clip differs from declared stage')
        require(learning.get('objective_field')=='signed_clearance4f_terms.loss'
            and learning.get('objective_coefficient')==1.,'risk objective mismatch')
        require(all(learning.get(k) is False for k in
            ('PPO_called','guide_called','value_loss_called','GAE_called','KL_gate','new_optimizer_created')),
            'unexpected optimization path')
        require(learning.get('actor_gradient_tensors')==learning.get('critic_gradient_tensors')==0,
            'actor/critic received gradients')
        require(learning.get('actor_critic_parameters_and_Adam_exact') is True
            and learning.get('existing_requires_grad_retained') is True,'update identity not exact')
        n=learning.get('optimizer_updates')
        require(type(n) is int and 0<=n<=90,'invalid paid physical Adam count')
        require(learning.get('optimizer_step_attempts')==n
            and learning.get('optimizer_transaction_complete') is True,'incomplete optimizer transaction')
        paid+=n
        require(update['costs']['supervised_Adam']==paid and update['costs']['completed_rollouts']==ordinal,
            'paid update cumulative counters differ')
    require(paid==costs['supervised_Adam'],'result and committed paid Adam differ')
    require(costs.get('student_actions')==7680*len(updates),'complete collected-action count mismatch')
    return dict(costs=costs,perception_branches=branches,actor_critic_identity_exact=True,
        actual_transition=transition,max_grad_norm=clip,
        committed_updates=len(updates),physical_risk_Adam=paid,PPO_updates=0,value_updates=0,
        teacher_actions_executed=0,identity_evidence='Producer per-tensor weights/buffers/Adam hashes;JSON verified,PT not reloaded')


def validate_evaluation(e):
    require(e.get('episodes')==20 and e.get('actual_Bombs')==0 and e.get('no_frame_cap') is True,
        'complete uncapped NoBomb20 required')
    require(e.get('optimizer_updates')==e.get('teacher_actions')==0,'frozen evaluation modified policy/used teacher')
    require(e.get('action_selection')=='greedy' and e.get('action_temperature')==1.,'frozen greedy T1 required')
    rows=e.get('episode_results') or []
    require(len(rows)==20 and len({r['evaluation_game'] for r in rows})==20,'twenty unique games required')
    require(all(type(r.get('first_collision_observed')) is bool
        and type(r.get('right_censored_at_natural_clear')) is bool
        and r['first_collision_observed']!=r['right_censored_at_natural_clear'] for r in rows),
        'each game needs contact or natural-clear censoring')
    return rows


def summarize(candidate,baselines,output,*,parent_evaluation=None):
    refs=[]
    def load(path):
        value=read_shared_json(path)
        refs.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        return value
    report=load(candidate/'result.json');launch=load(candidate/'launch.json')
    identities=report['actor_critic_identity_files']
    def identity(which):
        spec=identities[which];path=Path(spec['path'])
        if not path.is_absolute():path=candidate/path
        value=load(path)
        require(refs[-1]['sha256']==spec['sha256'],'identity file SHA mismatch')
        return value
    before,after=identity('before'),identity('after')
    paths=sorted((candidate/'traces').glob('update-*.json'))
    updates=[load(p) for p in paths]
    stage=validate_risk_stage(report,launch,updates,before,after)
    e=report['development'];rows=validate_evaluation(e)
    pairs={};references={};details={}
    evaluations=[]
    for name in ('initial','current','D95'):
        base=load(baselines/name/'result.json')
        require(base.get('status')=='completed','baseline incomplete')
        evaluations.append((name,base['evaluation']))
    if parent_evaluation is not None:
        parent=load(parent_evaluation);require(parent.get('status')=='completed','parent incomplete')
        evaluations.append(('training_parent',parent.get('development') or parent['evaluation']))
    for name,ref in evaluations:
        validate_evaluation(ref)
        pairs[name],references[name],details[name]=paired_frozen_evaluations(e,ref,name)
    values=[r['native_frames'] for r in rows];censored=sum(r['right_censored_at_natural_clear'] for r in rows)
    median=statistics.median(values);q25=quantile(values,.25)
    stage1=sum(r['first_stage_without_observed_contact'] for r in rows)
    d95=references['D95']
    near=(median>=.9*d95['endpoint_median_lower_bound'] and q25>=.9*d95['endpoint_q25_lower_bound']
        and (d95['stage1_no_contact']-stage1)/20<=.10)
    result=dict(schema='full-sensory-risk-only-frozen20-JSON-review-v1',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidate_path=str(candidate.resolve()),
        risk_stage=stage,candidate=dict(episodes=20,event_free_mean_lower_bound=statistics.mean(values),
            event_free_median_lower_bound=median,event_free_q25_lower_bound=q25,
            first_contact_median=None if censored else median,first_contact_q25=None if censored else q25,
            natural_clear_right_censored=censored,first_stage_without_contact=stage1,actual_Bombs=0),
        baselines=references,paired=pairs,paired_games=details,actual_seeds_matched=True,
        preregistered_near_D95_development_screen=near,independent_new20_confirmation_completed=False,
        fullstory_PPO_resume_allowed=False,source_refs=refs,
        source_script_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        parent_provenance=report.get('provenance'),cumulative_counters=report.get('cumulative_counters'),
        model_forwards=0,native_actions=0,optimizer_updates=0,
        scope='Frozen NoBomb first-observed-contact development20;natural clear is censored. Not Bomb-allowed75%,formal stability,statistical equivalence or deployment. Risk-stage collection is changing-policy data,not these frozen results.')
    output.mkdir(parents=True,exist_ok=False)
    (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    text=['# 全感觉风险独立预训练：冻结20局开发评审','',
        f"实付风险Adam {stage['physical_risk_Adam']}；采集动作{stage['costs']['student_actions']}；PPO/value/教师动作均0。",
        'actor与三critic参数、buffer及Adam由生产者逐张量哈希核验不变；本脚本仅校验JSON凭据，不重读模型。','',
        f'无触弹中位下界{median}帧；P25下界{q25}；第一面无接触{stage1}/20；自然clear删失{censored}。','',
        '| 对照 | 已知更晚接触 | 更早 | 相同 | 删失无法排序 |','|---|---:|---:|---:|---:|']
    for name,p in pairs.items():text.append(f"| {name} | {p['known_longer']} | {p['known_earlier']} | {p['tied']} | {p['ambiguous']} |")
    text+=['',f'预设接近D95开发筛选：{near}；独立新20局确认未完成。',
        '首次观察原生接触包含可决死挽回的接触，通常为4tick端点；不是Miss，也不是精确原生事件帧。',
        '自然clear作右删失，不伪造触弹。此协议不同于允许Bomb的历史15/20，不作统计等价或自动部署判断。']
    (output/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',required=True,type=Path);p.add_argument('--baselines',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--parent-evaluation',type=Path)
    a=p.parse_args(argv);result=summarize(a.candidate,a.baselines,a.output,parent_evaluation=a.parent_evaluation)
    print(json.dumps({k:result[k] for k in ('risk_stage','candidate','paired','preregistered_near_D95_development_screen')},ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
