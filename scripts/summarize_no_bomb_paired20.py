"""Read three completed immutable NoBomb JSON reports; no model/native execution."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,statistics,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json

ENTRY='continuous-and-D95-NoBomb-first-contact-entry-v1'
PROBE='th14-NoBomb-uncapped-first-observed-contact-v1'
GROUPS=('initial','current','D95')


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def distribution(values):
    values=sorted(values)
    if not values:return dict(count=0,mean=None,median=None,min=None,max=None)
    return dict(count=len(values),mean=statistics.mean(values),median=statistics.median(values),
        min=min(values),max=max(values),values=values)


def summarize(root,output):
    reports={};launches={};rows={};sources=[]
    for name in GROUPS:
        folder=root/name
        launch=read_shared_json(folder/'launch.json');report=read_shared_json(folder/'result.json')
        assert launch['schema']==report['schema']==ENTRY and report['status']=='completed',name
        evaluation=report['evaluation'];assert evaluation['schema']==PROBE
        assert evaluation['Bomb_forbidden'] is True and evaluation['action_selection']=='greedy'
        assert evaluation['action_temperature']==1 and evaluation['no_frame_cap'] is True
        selected=sorted(evaluation['episode_results'],key=lambda row:row['evaluation_game'])
        assert len(selected)==20 and len({row['evaluation_game'] for row in selected})==20,name
        for row in selected:
            assert row['actual_bombs_used']==0 and row['seed_applied'] is True
            assert row['first_collision_observed'] != row['right_censored_at_natural_clear']
            if row['right_censored_at_natural_clear']:
                assert row['natural_clear_claim'] is True and row['first_contact_frames'] is None
            else:assert row['first_contact_frames']==row['native_frames']
        reports[name]=report;launches[name]=launch;rows[name]=selected
        for filename in ('launch.json','result.json','evaluation/episode_plan.json'):
            path=folder/filename;sources.append(dict(path=str(path),sha256=digest(path)))
    reference=launches['initial'];plan=reports['initial']['evaluation']['episode_plan']
    for name in GROUPS:
        assert launches[name]['environment_config_sha256']==reference['environment_config_sha256']
        assert launches[name]['master_seed']==reference['master_seed']
        assert reports[name]['evaluation']['episode_plan']==plan
        for a,b in zip(rows['initial'],rows[name]):
            for key in ('evaluation_game','evaluation_seed','action_seed','native_seed'):
                assert a[key]==b[key],(name,key)
    # A descriptive finite horizon, not an evaluation cap. Observed events remain
    # exact; censored natural clears receive bounds, not extrapolated event times.
    tau=max(row['native_frames'] for name in GROUPS for row in rows[name])
    summaries={}
    for name in GROUPS:
        selected=rows[name];contacts=[r['first_contact_frames'] for r in selected if r['first_collision_observed']]
        lower=[min(r['native_frames'],tau) for r in selected]
        upper=[min(r['native_frames'],tau) if r['first_collision_observed'] else tau for r in selected]
        summaries[name]=dict(checkpoint_sha256=reports[name]['checkpoint_sha256'],episodes=20,
            observed_contacts=len(contacts),natural_clear_right_censored=20-len(contacts),actual_Bombs=0,
            contact_only_frames=distribution(contacts),observed_endpoint_frames=distribution([r['native_frames'] for r in selected]),
            first_stage_without_contact=sum(r['first_stage_without_observed_contact'] for r in selected),
            restricted_event_free_mean_bounds=dict(tau=tau,lower=statistics.mean(lower),upper=statistics.mean(upper)))
    paired={}
    for a,b in (('current','initial'),('D95','initial'),('D95','current')):
        records=[]
        for x,y in zip(rows[a],rows[b]):
            def bounds(row):
                lo=min(row['native_frames'],tau)
                return lo,lo if row['first_collision_observed'] else tau
            xl,xu=bounds(x);yl,yu=bounds(y)
            exact=x['first_collision_observed'] and y['first_collision_observed']
            records.append(dict(game=x['evaluation_game'],native_seed=x['native_seed'],
                a_endpoint=x['native_frames'],b_endpoint=y['native_frames'],
                a_censored=x['right_censored_at_natural_clear'],b_censored=y['right_censored_at_natural_clear'],
                observed_contact_delta=x['native_frames']-y['native_frames'] if exact else None,
                restricted_difference_lower=xl-yu,restricted_difference_upper=xu-yl))
        exact=[r['observed_contact_delta'] for r in records if r['observed_contact_delta'] is not None]
        paired[a+'_minus_'+b]=dict(both_observed_contact_pairs=len(exact),conditional_contact_delta=distribution(exact),
            known_larger_restricted_contact_free_time=sum(r['restricted_difference_lower']>0 for r in records),
            known_smaller_restricted_contact_free_time=sum(r['restricted_difference_upper']<0 for r in records),
            restricted_mean_difference_bounds=[statistics.mean(r[k] for r in records) for k in
                ('restricted_difference_lower','restricted_difference_upper')],pairs=records)
    result=dict(schema='three-frozen-NoBomb-same-native20-descriptive-comparison-v1',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='completed',groups=summaries,paired=paired,
        actual_seed_alignment_verified=True,source_refs=sources,source_script_sha256=digest(__file__),
        scope='NoBomb greedyT1/native first state4 or Miss;observed4tick endpoint. Natural clears are right-censored;contact-only median is conditional. Bounds avoid assuming noninformative censoring. Same-seed action divergence is expected. No significance/robustness claim or historical Bomb-allowed75percent comparison.',
        Torch_imported='torch' in sys.modules,model_forwards=0,optimizer_updates=0,native_actions=0)
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    text=['# 三组冻结NoBomb同种子比较','',
        '自然无伤通关单列右删失；触弹均值/中位仅描述已观察触弹的局。限制时间的区间不假设删失独立。',
        '', '| 模型 | 观察触弹 | 无伤通关删失 | 触弹均值/中位帧 | 第一面无伤通过 |',
        '|---|---:|---:|---:|---:|']
    for name,s in summaries.items():
        d=s['contact_only_frames'];text.append(f"| {name} | {s['observed_contacts']} | {s['natural_clear_right_censored']} | {d['mean']} / {d['median']} | {s['first_stage_without_contact']}/20 |")
    text+=['',f'共同观察限制时间τ={tau}帧；配对明细及受限无触弹时间上下界见JSON。',
        '三组实际原生/环境/动作种子均核对一致。动作分叉不是配对失败。端点一般为4tick观察分辨率，非精确原生事件帧。',
        '这是无Bomb的首次接触评测，与历史允许Bomb的15/20开发通关属于不同目标；本报告不作小样本显著性或稳定通关结论。']
    (output/'summary.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    import psutil
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);process.cpu_affinity(process.cpu_affinity()[-2:])
    print(json.dumps(summarize(args.root,args.output)['groups'],ensure_ascii=False))
