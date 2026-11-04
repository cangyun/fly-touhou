"""Change only a proved teacher loss coefficient, preserving the entire learner."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from fly_touhou.experiment import file_sha256
from fly_touhou.geometry_migration import (verify_previous_geometry,PROOF_SOURCE_FILES,
    PROOF_BINDING_SCHEMA,_check_proof,validate_geometry_deployment)
from fly_touhou.native_teacher_feedback import (NativeV6TrainingFeedback,verify_closed_teacher_reference,
    TEACHER_FEEDBACK_SCHEMA,TEACHER_TRAINING_KIND)
from requalify_geometry_policy import equal


def coefficient_only(previous,current):
    """Reject any coupled change to label definitions, experts, or observations."""
    if not isinstance(previous,dict) or not isinstance(current,dict):
        raise ValueError('Both complete teacher specifications are required')
    old,new=deepcopy(previous),deepcopy(current)
    a,b=old.pop('coefficient',None),new.pop('coefficient',None)
    if (any(isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) or x<=0
            for x in (a,b)) or a==b or old!=new
            or old.get('schema')!=TEACHER_FEEDBACK_SCHEMA):
        raise ValueError('Only a changed finite positive teacher coefficient is allowed')
    return float(a),float(b)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','previous-source','proof','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-sha256',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    assert not a.output.exists() and not a.output.with_suffix('.report.json').exists()
    assert file_sha256(a.checkpoint)==a.expected_sha256
    data=torch.load(a.checkpoint,map_location='cpu',weights_only=True);before=deepcopy(data)
    old=verify_previous_geometry(data,a.previous_source);cfg=data['training_config']
    assert cfg['training_kind']==TEACHER_TRAINING_KIND
    assert cfg['action_feedback']['schema']=='native-geometry-action-set-v2'
    expert=NativeV6TrainingFeedback();new_spec=expert.specification()
    old_coefficient,new_coefficient=coefficient_only(cfg['teacher_feedback'],new_spec)
    state=data['rl_state']['teacher_feedback_state'];assert state['schema']==TEACHER_FEEDBACK_SCHEMA
    expert.load_state_dict(state['worlds']);assert equal(expert.state_dict(),state['worlds'])
    proof=json.loads(a.proof.read_text(encoding='utf-8'))
    assert proof['status']=='passed' and proof['learning_checkpoint_sha256']==a.expected_sha256
    assert proof['lineage_id']==old['proof']['lineage_id']
    assert proof['migrated_checkpoint_sha256']==old['proof']['migrated_checkpoint_sha256']
    assert proof['action_feedback']==cfg['action_feedback']
    assert proof['inherited_action_feedback_transition']==cfg['action_feedback_transition']
    assert proof['teacher_feedback']==new_spec
    reference=verify_closed_teacher_reference(proof['teacher_reference']['path'])
    assert reference==proof['teacher_reference']==cfg['teacher_reference']
    # Numerical changes are confined to this constant. Source guards prevent a
    # coefficient migration from silently also changing the learner or labels.
    allowed={'src/fly_touhou/native_teacher_feedback.py'}
    changed=[]
    for name in PROOF_SOURCE_FILES:
        if file_sha256(ROOT/name)!=file_sha256(a.previous_source/name):
            assert name in allowed,name
            changed.append(name)
    assert changed==list(allowed)
    transition={'schema':'explicit-native-v6-teacher-feedback-transition-v1',
        'change':'teacher coefficient only',
        'parent_checkpoint_sha256':a.expected_sha256,'parent_checkpoint_path':str(a.checkpoint.resolve()),
        'previous_proof_binding':old['binding'],
        'previous_teacher_feedback_transition':deepcopy(cfg['teacher_feedback_transition']),
        'previous_teacher_feedback':deepcopy(cfg['teacher_feedback']),
        'from_training_kind':cfg['training_kind'],'to_training_kind':cfg['training_kind'],
        'teacher_feedback':new_spec,'teacher_reference':reference,
        'teacher_mode_state':'bitwise preserved, reset by existing logic only for a new native episode',
        'learning_performed':False,'all_old_actor_critic_Adam_sampler_shuffle_counters_preserved':True,
        'observation_actions_neural_clock_and_native_control_changed':False}
    binding={'schema':PROOF_BINDING_SCHEMA,'path':str(a.proof.resolve()),'sha256':file_sha256(a.proof),
        'lineage_id':proof['lineage_id'],'teacher_feedback_validation':{'previous_source':old['source'],
            'previous_manifest_sha256':old['manifest_sha256'],'qualification_script_sha256':file_sha256(__file__)}}
    data['geometry_validation_proof']=binding;cfg['geometry_validation_proof']=binding
    data['teacher_feedback_transition']=transition;cfg['teacher_feedback_transition']=transition
    cfg['teacher_feedback']=new_spec
    _check_proof(data,proof);validate_geometry_deployment(data)
    x,y=deepcopy(before),deepcopy(data)
    for record in (x,y):
        for key in ('geometry_validation_proof','teacher_feedback_transition'):record.pop(key,None)
        for key in ('geometry_validation_proof','teacher_feedback_transition','teacher_feedback'):
            record['training_config'].pop(key,None)
    assert equal(x,y),'A coefficient change cannot replace learned or teacher-generator state'
    assert equal(before['rl_state']['teacher_feedback_state'],data['rl_state']['teacher_feedback_state'])
    assert file_sha256(a.checkpoint)==a.expected_sha256
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as out:torch.save(data,out)
    assert equal(data,torch.load(a.output,map_location='cpu',weights_only=True))
    report={'schema':'qualified-native-v6-teacher-coefficient-initializer-v1','status':'passed',
        'input_sha256':a.expected_sha256,'output':str(a.output.resolve()),'output_sha256':file_sha256(a.output),
        'coefficient_before':old_coefficient,'coefficient_after':new_coefficient,
        'changed_proof_source_files':changed,'transition':transition,'binding':binding,
        'all_old_learning_state_bitwise_preserved':True,'teacher_mode_state_bitwise_preserved':True,
        'model_trained':False,'native_game_started':False}
    with a.output.with_suffix('.report.json').open('x',encoding='utf-8') as out:
        json.dump(report,out,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':'passed','output':report['output'],'sha256':report['output_sha256'],
        'coefficient':[old_coefficient,new_coefficient]}),flush=True)


if __name__=='__main__':main()
