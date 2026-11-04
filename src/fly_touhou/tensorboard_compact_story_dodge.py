"""Cached CPU diagnostics inside the existing observer, never a second poller.

The monitor itself stays JSON-only. One short CPU subprocess per committed
rollout/report reduces saved observations; it never loads a policy or a game.
Derived files live in the monitor cache, not in the producer's directory.
"""
from pathlib import Path
import hashlib
import os
import subprocess
import time
from . import tensorboard_continuous as common


def _numeric_tree(exporter, prefix, values, step):
    for name, value in values.items():
        if isinstance(value, dict):
            _numeric_tree(exporter, prefix + name + '/', value, step)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            common._scalar(exporter, prefix + name, value, step)


def _extended(exporter, prefix, report, step):
    for horizon, raw in report.get('forecast_supervision_coverage', {}).items():
        values = dict(raw)
        n = values.get('unprotected_rows', 0)
        if n:
            for key in ('pre_action_forecast_known_rows', 'post_contact_consistency_accepted_rows',
                        'forecast_unknown_rows', 'native_fact_veto_rows'):
                values[key.removesuffix('_rows') + '_fraction'] = values[key] / n
        _numeric_tree(exporter, prefix + 'coverage/' + horizon + 'f/', values, step)
    for horizon, values in report.get('horizon_neural_risk_diagnosis', {}).items():
        _numeric_tree(exporter, prefix + 'risk_horizons/' + horizon + 'f/', values, step)
    _numeric_tree(exporter, prefix + 'behavior/', report.get('behavior_summary', {}), step)
    common._text(exporter, prefix + 'extended_scope', dict(
        changing_policy=True, formal_test=False,
        supervision=report.get('forecast_supervision_scope'),
        behavior=report.get('behavior_summary', {}).get('scope'),
        risk='Neural outputs against conditional geometric labels; not observed outcomes for unexecuted actions.'), step)


def export_dodge(exporter,label,root,*,settings,read_json):
    if not settings.get('enabled'):
        return
    root=Path(root)
    key=hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]
    cache=Path(settings['cache_root'])/key
    cache.mkdir(parents=True,exist_ok=True)
    cursor=exporter.state.setdefault('compact_story_dodge',{}).setdefault(label,{})
    prefix='fly_iterations/'+label+'/full_story_PPO/'
    scripts=Path(__file__).resolve().parents[2]/'scripts'
    ran=False

    def obtain(source,destination,script,argument):
        nonlocal ran
        if destination.exists():
            return read_json(destination)
        if ran or time.time()<cursor.get('retry_after',0):
            return None
        ran=True
        argv=[settings['python'],'-B',str(scripts/script),argument,str(source),'--output',str(destination)]
        try:
            with (cache/'analysis.stderr.log').open('ab') as errors:
                subprocess.run(argv,stdout=subprocess.DEVNULL,stderr=errors,check=True,timeout=120,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            cursor.pop('last_error',None);cursor.pop('retry_after',None)
            return read_json(destination)
        except (OSError,subprocess.SubprocessError) as problem:
            cursor['last_error']=str(problem);cursor['retry_after']=time.time()+300
            return None

    decisions=0
    for marker in sorted(root.glob('rollout-*.json')):
        number=marker.stem.removeprefix('rollout-')
        if not number.isdigit():
            continue
        metadata=read_json(marker)
        decisions+=metadata['student_actions']
        core_done=number in cursor.get('exported_rollouts',[])
        extended_done=number in cursor.get('exported_extended_v2',[])
        if core_done and extended_done:
            continue
        receipt=root/f'rollout-{number}-path.json'
        tape=Path(read_json(receipt)['path']) if receipt.exists() else root/f'rollout-{number}.pt'
        derived=cache/f'rollout-{number}-v2.json'
        if not tape.exists() and not derived.exists():
            continue
        report=obtain(tape,derived,'analyze_compact_story_rollout.py','--rollout')
        if report is None:
            continue
        if not extended_done:
            _extended(exporter,prefix+'rollout_training/',report,decisions)
            cursor.setdefault('exported_extended_v2',[]).append(number)
            cursor['latest_extended']=dict(cycle=int(number),decisions=decisions,
                derived=str(derived),coverage=report.get('forecast_supervision_coverage'),
                risk_horizons=report.get('horizon_neural_risk_diagnosis'),
                behavior=report.get('behavior_summary'))
        if core_done:
            continue
        native=report['actual_next_step'];n=decisions
        values=dict(student_decisions=report['student_actions'],
            observed_unprotected_contacts=native['contacts'],
            observed_unprotected_movement_rows=native['observed_unprotected_movement_rows'],
            forecast_safe_but_observed_contact=native['forecast_safe_but_observed_contact'],
            actual_Bombs=metadata['actual_bombs'],Bomb_requests=metadata['Bomb_requests'],
            Bomb_in_observed_deathbomb_window=metadata['deathbombs'],
            mixed_forecast_rows=report['mixed_immediate_forecast_rows'],
            sampled_unsafe_in_mixed=report['mixed_chosen_unsafe'])
        denominator=native['observed_unprotected_movement_rows']
        if denominator:values['contacts_per1000_unprotected_moves']=1000*native['contacts']/denominator
        for k,v in values.items():common._scalar(exporter,prefix+'rollout_training/'+k,v,n)
        for stage,count in report['stage_decisions'].items():common._scalar(exporter,prefix+'rollout_training/stage'+stage+'_decisions',count,n)
        risk=report.get('neural_risk_diagnosis',{})
        for k,v in risk.get('0.2',{}).items():common._scalar(exporter,prefix+'rollout_training/risk4/'+k,v,n)
        sampling=report.get('same_state_sampling_diagnosis',{})
        for k in ('sampled_contacts_where_family_forecast_safe','family_unsafe_in_mixed'):
            if k in sampling:common._scalar(exporter,prefix+'rollout_training/sampling/'+k,sampling[k],n)
        common._text(exporter,prefix+'rollout_training/scope',dict(
            changing_policy=True,formal_test=False,source=str(tape),derived=str(derived),
            collision_scope=report['collection_contact_counter_scope'],
            forecast_scope='Unexecuted alternatives are geometric forecasts,not observed native outcomes.'),n)
        cursor.setdefault('exported_rollouts',[]).append(number)
        cursor['latest_training']=dict(cycle=int(number),decisions=n,**values,stage_decisions=report['stage_decisions'])

    for marker in sorted(root.glob('development-cycle*/report.json')):
        role=marker.parent.name
        if role in cursor.get('exported_development',[]):continue
        if not (marker.parent/'diagnostics').is_dir():continue
        report=obtain(marker.parent,cache/(role+'-contacts.json'),
            'analyze_compact_story_contact_choices.py','--evaluation')
        if report is None:continue
        n=int(role.removeprefix('development-cycle'))
        for k,v in report['summary'].items():
            if isinstance(v,(int,float)):common._scalar(exporter,prefix+'frozen_contact_analysis/'+k,v,n)
        for h,values in report['summary']['route_warning_contacts_by_horizon'].items():
            for k,v in values.items():common._scalar(exporter,prefix+'frozen_contact_analysis/'+h+'f/'+k,v,n)
        common._text(exporter,prefix+'frozen_contact_analysis/scope',report['scope'],n)
        cursor.setdefault('exported_development',[]).append(role)
        cursor['latest_development']=dict(cycle=n,source=str(marker),summary=report['summary'])
    return cursor
