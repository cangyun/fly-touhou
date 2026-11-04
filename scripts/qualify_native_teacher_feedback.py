"""Enable proved V6 training guidance without replacing any prior learner state."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from fly_touhou.experiment import file_sha256
from fly_touhou.geometry_action_feedback import ACTION_FEEDBACK_TRAINING_KIND
from fly_touhou.geometry_migration import (verify_previous_geometry,PROOF_SOURCE_FILES,
    PROOF_BINDING_SCHEMA,_check_proof,validate_geometry_deployment)
from fly_touhou.native_teacher_feedback import (NativeV6TrainingFeedback,verify_closed_teacher_reference,
    TEACHER_FEEDBACK_SCHEMA,TEACHER_TRAINING_KIND)
from requalify_geometry_policy import equal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','previous-source','proof','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-sha256',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    assert not a.output.exists() and not a.output.with_suffix('.report.json').exists()
    assert file_sha256(a.checkpoint)==a.expected_sha256
    data=torch.load(a.checkpoint,map_location='cpu',weights_only=True);before=deepcopy(data)
    old=verify_previous_geometry(data,a.previous_source);cfg=data['training_config']
    assert cfg['training_kind']==ACTION_FEEDBACK_TRAINING_KIND and cfg.get('teacher_feedback') is None
    assert cfg['action_feedback']['schema']=='native-geometry-action-set-v2'
    assert 'teacher_feedback_state' not in data['rl_state']
    proof=json.loads(a.proof.read_text(encoding='utf-8'))
    assert proof['status']=='passed' and proof['learning_checkpoint_sha256']==a.expected_sha256
    assert proof['lineage_id']==old['proof']['lineage_id']
    assert proof['migrated_checkpoint_sha256']==old['proof']['migrated_checkpoint_sha256']
    assert proof['action_feedback']==cfg['action_feedback']
    assert proof['inherited_action_feedback_transition']==cfg['action_feedback_transition']
    expert=NativeV6TrainingFeedback();assert proof['teacher_feedback']==expert.specification()
    reference=verify_closed_teacher_reference(proof['teacher_reference']['path'])
    assert reference==proof['teacher_reference']
    allowed={'src/fly_touhou/whole_brain_ppo.py','src/fly_touhou/ppo_cuda_capture.py',
        'src/fly_touhou/geometry_migration.py','src/fly_touhou/native_teacher_feedback.py'}
    for name in PROOF_SOURCE_FILES:
        if name not in allowed:assert file_sha256(ROOT/name)==file_sha256(a.previous_source/name),name
    transition={'schema':'explicit-native-v6-teacher-feedback-transition-v1',
        'parent_checkpoint_sha256':a.expected_sha256,'parent_checkpoint_path':str(a.checkpoint.resolve()),
        'previous_proof_binding':old['binding'],'from_training_kind':cfg['training_kind'],
        'to_training_kind':TEACHER_TRAINING_KIND,'teacher_feedback':expert.specification(),
        'teacher_reference':reference,'new_teacher_mode_state':'empty, then one independent V6 state per new native episode',
        'learning_performed':False,'all_old_actor_critic_Adam_sampler_shuffle_counters_preserved':True,
        'observation_actions_neural_clock_and_native_control_changed':False}
    binding={'schema':PROOF_BINDING_SCHEMA,'path':str(a.proof.resolve()),'sha256':file_sha256(a.proof),
        'lineage_id':proof['lineage_id'],'teacher_feedback_validation':{'previous_source':old['source'],
            'previous_manifest_sha256':old['manifest_sha256'],'qualification_script_sha256':file_sha256(__file__)}}
    data['geometry_validation_proof']=binding;cfg['geometry_validation_proof']=binding
    data['teacher_feedback_transition']=transition;cfg['teacher_feedback_transition']=transition
    cfg['teacher_feedback']=expert.specification();cfg['teacher_reference']=reference
    cfg['training_kind']=TEACHER_TRAINING_KIND
    data['rl_state']['teacher_feedback_state']={'schema':TEACHER_FEEDBACK_SCHEMA,'worlds':{}}
    _check_proof(data,proof);validate_geometry_deployment(data)
    x,y=deepcopy(before),deepcopy(data)
    for record in (x,y):
        for k in ('geometry_validation_proof','teacher_feedback_transition'):record.pop(k,None)
        for k in ('geometry_validation_proof','teacher_feedback_transition','teacher_feedback','teacher_reference'):
            record['training_config'].pop(k,None)
        record['training_config']['training_kind']=before['training_config']['training_kind']
        record['rl_state'].pop('teacher_feedback_state',None)
    assert equal(x,y),'Existing learned state cannot change when enabling expert supervision'
    assert file_sha256(a.checkpoint)==a.expected_sha256
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as out:torch.save(data,out)
    assert equal(data,torch.load(a.output,map_location='cpu',weights_only=True))
    report={'schema':'qualified-native-v6-training-feedback-initializer-v1','status':'passed',
        'input_sha256':a.expected_sha256,'output':str(a.output.resolve()),'output_sha256':file_sha256(a.output),
        'transition':transition,'binding':binding,'all_old_learning_state_bitwise_preserved':True,
        'model_trained':False,'native_game_started':False}
    with a.output.with_suffix('.report.json').open('x',encoding='utf-8') as out:json.dump(report,out,ensure_ascii=False,indent=2)
    print(json.dumps({'status':'passed','output':report['output'],'sha256':report['output_sha256']}))


if __name__=='__main__':main()
