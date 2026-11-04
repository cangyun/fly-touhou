"""Enable proved current-rollout critical coverage with the full learner intact."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from fly_touhou.experiment import file_sha256
from fly_touhou.geometry_migration import (verify_previous_geometry,PROOF_SOURCE_FILES,
    PROOF_BINDING_SCHEMA,_check_proof,validate_geometry_deployment)
from fly_touhou.native_teacher_feedback import NativeV6TrainingFeedback,verify_closed_teacher_reference,TEACHER_TRAINING_KIND
from fly_touhou.critical_full_rollout import specification,CRITICAL_TRANSITION_SCHEMA
from requalify_geometry_policy import equal


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
    assert cfg['training_kind']==TEACHER_TRAINING_KIND and cfg['minibatch_chunks']==1
    assert cfg['teacher_feedback']==NativeV6TrainingFeedback().specification()
    assert 'critical_full_rollout_first_step' not in cfg and 'critical_first_step_transition' not in cfg
    proof=json.loads(a.proof.read_text(encoding='utf-8'))
    assert proof['status']=='passed' and proof['learning_checkpoint_sha256']==a.expected_sha256
    assert proof['lineage_id']==old['proof']['lineage_id']
    assert proof['migrated_checkpoint_sha256']==old['proof']['migrated_checkpoint_sha256']
    assert proof['action_feedback']==cfg['action_feedback']
    assert proof['inherited_action_feedback_transition']==cfg['action_feedback_transition']
    assert proof['teacher_feedback']==cfg['teacher_feedback']
    assert proof['critical_full_rollout_first_step']==specification()
    reference=verify_closed_teacher_reference(proof['teacher_reference']['path'])
    assert reference==proof['teacher_reference']==cfg['teacher_reference']
    allowed={'src/fly_touhou/whole_brain_ppo.py','src/fly_touhou/geometry_migration.py',
        'src/fly_touhou/serial_gradient_accumulation.py','src/fly_touhou/critical_full_rollout.py'}
    changed=[]
    for name in PROOF_SOURCE_FILES:
        previous=a.previous_source/name
        if not previous.is_file() or file_sha256(ROOT/name)!=file_sha256(previous):
            assert name in allowed,name
            changed.append(name)
    assert set(changed)==allowed
    critical={'schema':CRITICAL_TRANSITION_SCHEMA,'parent_checkpoint_path':str(a.checkpoint.resolve()),
        'parent_checkpoint_sha256':a.expected_sha256,'previous_proof_binding':old['binding'],
        'specification':specification(),'learning_performed':False,
        'all_actor_critic_Adam_samplers_shuffle_counters_teacher_modes_preserved':True,
        'native_control_observation_neural_clock_and_labels_changed':False}
    teacher={'schema':'explicit-native-v6-teacher-feedback-transition-v1',
        'change':'current on-policy critical full first-step schedule; expert targets and global weights unchanged',
        'parent_checkpoint_path':str(a.checkpoint.resolve()),'parent_checkpoint_sha256':a.expected_sha256,
        'previous_proof_binding':old['binding'],
        'previous_teacher_feedback_transition':deepcopy(cfg['teacher_feedback_transition']),
        'teacher_feedback':deepcopy(cfg['teacher_feedback']),'teacher_reference':reference,
        'from_training_kind':cfg['training_kind'],'to_training_kind':cfg['training_kind'],
        'learning_performed':False,'all_old_actor_critic_Adam_sampler_shuffle_counters_preserved':True}
    binding={'schema':PROOF_BINDING_SCHEMA,'path':str(a.proof.resolve()),'sha256':file_sha256(a.proof),
        'lineage_id':proof['lineage_id'],'critical_first_step_validation':{'previous_source':old['source'],
            'previous_manifest_sha256':old['manifest_sha256'],'qualification_script_sha256':file_sha256(__file__)}}
    data['geometry_validation_proof']=binding;cfg['geometry_validation_proof']=binding
    data['teacher_feedback_transition']=teacher;cfg['teacher_feedback_transition']=teacher
    data['critical_first_step_transition']=critical;cfg['critical_first_step_transition']=critical
    cfg['critical_full_rollout_first_step']=specification()
    _check_proof(data,proof);validate_geometry_deployment(data)
    x,y=deepcopy(before),deepcopy(data)
    for record in (x,y):
        for key in ('geometry_validation_proof','teacher_feedback_transition','critical_first_step_transition'):
            record.pop(key,None);record['training_config'].pop(key,None)
        record['training_config'].pop('critical_full_rollout_first_step',None)
    assert equal(x,y),'Critical coverage qualification cannot change any learned state'
    assert file_sha256(a.checkpoint)==a.expected_sha256
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as out:torch.save(data,out)
    assert equal(data,torch.load(a.output,map_location='cpu',weights_only=True))
    report={'schema':'qualified-onpolicy-critical-first-step-initializer-v1','status':'passed',
        'input_sha256':a.expected_sha256,'output':str(a.output.resolve()),'output_sha256':file_sha256(a.output),
        'critical_transition':critical,'teacher_transition':teacher,'binding':binding,
        'changed_proof_source_files':changed,'all_learning_and_teacher_mode_state_bitwise_preserved':True,
        'model_trained':False,'native_game_started':False}
    with a.output.with_suffix('.report.json').open('x',encoding='utf-8') as out:
        json.dump(report,out,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':'passed','output':report['output'],'sha256':report['output_sha256']}),flush=True)


if __name__=='__main__':main()
