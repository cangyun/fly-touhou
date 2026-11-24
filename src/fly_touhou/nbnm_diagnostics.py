"""Record teacher, geometric and neural choices without changing execution."""
import numpy as np


def decision_record(info, features, output, index, *, policy):
    def array(name):return output[name][index].detach().float().cpu().numpy()
    logits=array('logits');route=array('route_scores');scores=array('horizon_scores')
    values=array('values')
    from .nbnm_numerics import validate_network_arrays
    validate_network_arrays(dict(logits=logits,route_scores=route,horizon_scores=scores,values=values),features=features)
    risk=1/(1+np.exp(-np.clip(8*scores-1,-60,60)))
    f=np.asarray(features,float);geo=info.get('compact_targets',{})
    teacher=info.get('teacher_label',{})
    finite=logits[:18]-np.max(logits[:18]);prob=np.exp(finite);prob/=prob.sum()
    warning = {}
    if 'settled_safe24' in output:
        warning['settled_safe24']=bool(output['settled_safe24'][index])
        warning['unadjusted_action']=int(output['unadjusted_action'][index])
    if 'wait_eligible' in output:
        warning['wait_eligible']=bool(output['wait_eligible'][index])
        warning['wait_action']=int(output['wait_action'][index])
    if 'heading_eligible' in output:
        warning['heading_eligible']=bool(output['heading_eligible'][index])
        warning['pre_heading_action']=int(output['pre_heading_action'][index])
        warning['previous_executed_action']=int(output['previous_executed_action'][index])
    if 'wall_escape_rule' in output:
        warning['wall_escape_rule']=int(output['wall_escape_rule'][index])
        warning['pre_wall_action']=int(output['pre_wall_action'][index])
        if warning['wall_escape_rule']:
            selected=int(output['actions'][index])
            warning['pre_wall_selected_score']=float(output['pre_wall_logits'][index,selected])
    if 'issun_lure_guidance' in info:
        warning['issun_lure_guidance'] = np.flatnonzero(info['issun_lure_guidance']).tolist()
        warning['issun_lure_lesson'] = info['issun_lure_lesson']
    if 'nbnm_calm_movement' in info:
        warning['previous_calm_movement'] = info['nbnm_calm_movement']
    if 'issun_lure_tail_turn' in info:
        warning['issun_lure_tail_turn']=info['issun_lure_tail_turn']
    if 'issun_lure' in info:
        lure=info['issun_lure']
        selected=int(output['actions'][index])
        warning['issun_lure']=dict(lure,selected_published_course=any(
            row['action']==selected for row in lure['published']))
    if 'corridor_endpoint_cost' in output:
        cost=array('corridor_endpoint_cost')
        validate_network_arrays({'corridor_endpoint_cost':cost})
        warning['corridor_endpoint_cost']=cost.tolist()
    if 'corridor_logits' in output:
        raw=array('corridor_logits')
        validate_network_arrays({'corridor_logits':raw})
        warning['corridor_neural_risk48_96']=(1/(1+np.exp(-np.clip(raw,-60,60)))).tolist()
    if 'corridor_witness_logits' in output:
        raw=array('corridor_witness_logits')
        validate_network_arrays({'corridor_witness_logits':raw})
        warning['corridor_neural_exit48_96']=(1/(1+np.exp(-np.clip(raw,-60,60)))).tolist()
    if 'corridor_course_logits' in output:
        raw=array('corridor_course_logits')
        validate_network_arrays({'corridor_course_logits':raw})
        warning['corridor_neural_course']=(1/(1+np.exp(-np.clip(raw,-60,60)))).tolist()
    if 'warning48_logits' in output:
        raw = array('warning48_logits')
        validate_network_arrays({'warning48_logits': raw})
        warning['native_warning48'] = (1 / (1 + np.exp(-np.clip(raw,-60,60)))).tolist()
    return dict(native_seed=info.get('native_seed'),native_episode=info.get('native_episode'),
        episode_frame=info.get('episode_frames'),stage=info.get('stage'),stage_frame=info.get('stage_frame'),
        action=int(output['actions'][index]),position=[float(f[0]*368-184),float(f[1]*400+32)],
        normal=bool(geo.get('normal',False)),trusted=geo.get('trusted'),margins=geo.get('margins'),
        uncertainty=geo.get('uncertainty'),phase=info.get('nb_nm_phase'),
        policy=policy,policy_probabilities=prob.tolist(),policy_logits=logits[:18].tolist(),
        route_scores=route.tolist(),neural_risk=risk.tolist(),value_heads=values.tolist(),
        teacher_action=teacher.get('action'),teacher_probabilities=teacher.get('probabilities'),
        teacher_weight=teacher.get('guidance_weight',0),teacher_route=teacher.get('nb_nm_route'),
        teacher_seconds=teacher.get('query_seconds'),features=f.tolist(),**warning)
