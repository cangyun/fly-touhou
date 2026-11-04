"""Read completed same-checkpoint beta8/beta0 NoBomb20 behavior reports."""
from pathlib import Path
import argparse
import datetime
import json
import math
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json


def quantile(values,p):
    values=sorted(values);n=(len(values)-1)*p;a=math.floor(n);b=math.ceil(n)
    return values[a]+(values[b]-values[a])*(n-a)


def compare(beta8,beta0,output):
    r8=read_shared_json(beta8/'result.json');r0=read_shared_json(beta0/'result.json')
    assert r8['status']==r0['status']=='completed'
    assert r0['schema']=='continuous-object-learning-runtime-risk-beta0-NoBomb-first-contact-entry-v1'
    assert r0['source_beta']==8. and r0['effective_risk_beta']==0.
    override=r0['runtime_policy_override']
    assert override['runtime_only'] and not override['model_parameters_changed']
    assert not override['source_checkpoint_changed'] and not override['modified_checkpoint_saved']
    e8=r8['development'];e0=r0['evaluation']
    assert e8['checkpoint_sha256']==e0['checkpoint_sha256']==r0['checkpoint_sha256']
    assert e8['episode_plan']==e0['episode_plan'] and e8['seed']==e0['seed']
    l8=read_shared_json(beta8/'launch.json');l0=read_shared_json(beta0/'launch.json')
    assert l8['environment_config_sha256']==l0['environment_config_sha256']
    rows={};summary={}
    for name,e in [('beta8',e8),('beta0',e0)]:
        assert e['episodes']==20 and e['actual_Bombs']==0 and e['no_frame_cap']
        assert e['action_selection']=='greedy' and e['action_temperature']==1.
        assert e['optimizer_updates']==e['teacher_actions']==0
        selected=sorted(e['episode_results'],key=lambda x:x['evaluation_game'])
        assert len(selected)==20 and len({x['evaluation_game'] for x in selected})==20
        assert all(x['seed_applied'] and x['actual_bombs_used']==0 for x in selected)
        assert all(x['first_collision_observed']!=x['right_censored_at_natural_clear'] for x in selected)
        endpoint=[x['native_frames'] for x in selected]
        summary[name]=dict(event_free_median_lower_bound=statistics.median(endpoint),
            event_free_q25_lower_bound=quantile(endpoint,.25),endpoint_mean=statistics.mean(endpoint),
            natural_clear_censored=sum(x['right_censored_at_natural_clear'] for x in selected),
            first_stage_without_contact=sum(x['first_stage_without_observed_contact'] for x in selected))
        rows[name]=selected
    wins=losses=ties=unknown=0;paired=[]
    for a,b in zip(rows['beta0'],rows['beta8']):
        for k in ('evaluation_game','evaluation_seed','action_seed','native_seed'):
            assert a[k]==b[k],k
        alo=a['native_frames'];blo=b['native_frames']
        ahi=math.inf if a['right_censored_at_natural_clear'] else alo
        bhi=math.inf if b['right_censored_at_natural_clear'] else blo
        status='longer' if alo>bhi else 'earlier' if ahi<blo else 'tied' if ahi==alo==bhi==blo else 'ambiguous'
        wins+=status=='longer';losses+=status=='earlier';ties+=status=='tied';unknown+=status=='ambiguous'
        paired.append(dict(game=a['evaluation_game'],native_seed=a['native_seed'],beta0_endpoint=alo,
            beta8_endpoint=blo,beta0_censored=a['right_censored_at_natural_clear'],
            beta8_censored=b['right_censored_at_natural_clear'],status=status))
    a=summary['beta0'];b=summary['beta8']
    prefer=(a['event_free_median_lower_bound']>b['event_free_median_lower_bound']
        and a['event_free_q25_lower_bound']>=b['event_free_q25_lower_bound']
        and a['first_stage_without_contact']>=b['first_stage_without_contact'] and wins>losses)
    result=dict(schema='same-source-NoBomb20-risk-beta0-vs-beta8-behavior-v1',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),summary=summary,
        beta0_known_longer=wins,beta0_known_earlier=losses,tied=ties,ambiguous=unknown,pairs=paired,
        provisional_beta0_preferable=prefer,
        source_parameters_sha256=e8['checkpoint_sha256'],actual_seed_alignment_verified=True,
        full_story_PPO_resume_allowed=False,independent_new20_confirmation_completed=False,
        caution='Development model selection only. Natural clears are lower bounds,not invented contact times. Runtime beta0 has no deployable modified checkpoint;any future adoption must explicitly persist/load the effective policy configuration.',
        model_forwards=0,native_actions=0,optimizer_updates=0)
    output.mkdir(parents=True,exist_ok=False)
    (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 同一果蝇存档：风险修正β8与β0实际行为对照','',
        '| 执行系数 | 首触时间中位下界 | P25下界 | 观察终点均值 | 第一面无触弹通过 | 无伤通关删失 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,s in summary.items():
        lines.append(f"| {name} | {s['event_free_median_lower_bound']} | {s['event_free_q25_lower_bound']} | {s['endpoint_mean']} | {s['first_stage_without_contact']}/20 | {s['natural_clear_censored']} |")
    lines+=['',f'β0同种子更晚{wins}、更早{losses}、相同{ties}、删失无法排序{unknown}。',
        f'开发选择条件是否偏向β0：{prefer}。未完成新20局独立确认，通关PPO继续停止。',
        '仅执行系数变化；源参数和文件未改、无教师控制/优化器更新。β0未保存为可部署新存档。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('beta8','beta0','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();r=compare(a.beta8,a.beta0,a.output)
    print(json.dumps({k:r[k] for k in ('summary','beta0_known_longer','beta0_known_earlier','tied','provisional_beta0_preferable')},ensure_ascii=False))
