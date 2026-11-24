"""Apply native-validated teacher repairs on the next committed resume only."""
from copy import deepcopy
from pathlib import Path

SCHEMA='nbnm-next-resume-native-guidance-v1'


def pending(config_path):
    from .thrl_study_io import read_shared_json
    path=Path(config_path).with_suffix('.next-resume-guidance.json')
    if not path.exists():return None
    patch=read_shared_json(path)
    if patch.get('schema')!=SCHEMA or not patch.get('native_validation_passed'):
        raise ValueError('Next-resume guidance needs completed native validation')
    if (not set(patch['nbnm']) or
            set(patch['nbnm'])-{'teacher_outcome_revision','live_route_stage_beams','demonstration_route_teacher','live_route_planner','native_warning48','corridor_risk'}):
        raise ValueError('Unknown next-resume guidance option')
    if 'native_warning48' in patch['nbnm']:
        from .nbnm_native_warning import validate
        validate(patch['nbnm']['native_warning48'])
    if 'corridor_risk' in patch['nbnm']:
        from .nbnm_corridor import validate
        validate(patch['nbnm']['corridor_risk'])
    if 'live_route_stage_beams' in patch['nbnm']:
        from .nbnm_live_routes import configured_beam
        for stage in patch['nbnm']['live_route_stage_beams']:
            if stage not in ('1','2','3','4','5','6'):raise ValueError('Original stage number required')
            configured_beam(int(stage),{'stage_beams':patch['nbnm']['live_route_stage_beams']})
    if set(patch['rehearsal'])-{'native_short_risk_coefficient','native_correction','additional_manifests','every_updates','phase_priorities','reserved_task'}:
        raise ValueError('Only verified native replay changes are supported by this patch')
    if set(patch.get('adaptive_course',{}))-{'priority_multipliers','stage_chain_priority_multipliers'}:
        raise ValueError('Adjust only declared native-task priorities on this resume')
    return patch


def apply(spec,patch):
    result=deepcopy(spec)
    if patch:
        settings=deepcopy(patch['nbnm'])
        stage_beams=settings.pop('live_route_stage_beams',None)
        result['nbnm'].update(settings)
        if stage_beams is not None:
            result['nbnm'].setdefault('live_route_planner',{}).setdefault('stage_beams',{}).update(stage_beams)
        result.setdefault('rehearsal',{}).update(deepcopy(patch['rehearsal']))
        if 'route_bank_refresh' in patch:
            result['route_bank_refresh']=deepcopy(patch['route_bank_refresh'])
        if 'adaptive_course' in patch:
            for key,values in patch['adaptive_course'].items():
                result['adaptive_course'].setdefault(key,{}).update(deepcopy(values))
        if 'evaluation_tasks' in patch:
            result['evaluation_tasks']=list(dict.fromkeys(result.get('evaluation_tasks',[])+patch['evaluation_tasks']))
    return result


def for_saved_rollout(requested,saved):
    old=saved['spec']
    differs=(requested['nbnm']!=old['nbnm'] or requested.get('rehearsal')!=old.get('rehearsal')
        or any(requested.get(k)!=old.get(k) for k in ('route_bank_refresh','adaptive_course','evaluation_tasks')))
    if saved.get('in_progress') is None or not differs:return requested,False
    current=deepcopy(requested)
    current['nbnm']=deepcopy(old['nbnm']);current['rehearsal']=deepcopy(old.get('rehearsal',{}))
    for key in ('route_bank_refresh','adaptive_course','evaluation_tasks'):
        if key in old:current[key]=deepcopy(old[key])
        else:current.pop(key,None)
    return current,True


def rehearsal_for(spec):
    from .nbnm_rehearsal import SuccessRehearsal
    r=spec.get('rehearsal',{})
    if not r.get('coefficient',0):return None
    return SuccessRehearsal(r['manifest'],scope_weights=r.get('scope_weights'),
        additional_manifests=r.get('additional_manifests',()),attack_spell_fraction=r.get('attack_spell_fraction'),
        challenging_fraction=r.get('challenging_fraction',0.),phase_priorities=r.get('phase_priorities'),
        priority_fraction=r.get('priority_fraction',.5),native_short_risk_coefficient=r.get('native_short_risk_coefficient',0.),
        native_correction=r.get('native_correction'))


def record(config_path,spec,patch,output,decisions,updates,migration):
    from .thrl_study_io import read_shared_json
    from .thrl_durable_records import write_json
    write_json(config_path,apply(read_shared_json(config_path),patch))
    write_json(Path(output)/'configuration.json',spec)
    write_json(Path(output)/'native-guidance-activation.json',dict(schema=SCHEMA,
        decisions=decisions,optimizer_updates=updates,patch=patch,migration=migration,
        old_rollout_relabelled=False,optimizer_reset=False,brain_topology_changed=False,
        native_worlds_preserved=True,rule_teacher_takes_actions=False))
