"""Enable a separately proved training objective with complete learner history."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from fly_touhou.experiment import file_sha256
from fly_touhou.geometry_action_feedback import (GeometryActionFeedback,ACTION_FEEDBACK_TRAINING_KIND)
from fly_touhou.geometry_migration import (verify_previous_geometry,_check_proof,
    validate_geometry_deployment,PROOF_SOURCE_FILES,PROOF_BINDING_SCHEMA)
from requalify_geometry_policy import equal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','previous-source','proof','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    if args.output.exists() or args.output.with_suffix('.report.json').exists():
        raise FileExistsError('Preserve every learner and proof')
    assert file_sha256(args.checkpoint)==args.expected_sha256
    data=torch.load(args.checkpoint,map_location='cpu',weights_only=True);before=deepcopy(data)
    old=verify_previous_geometry(data,args.previous_source)
    cfg=data['training_config']
    previous_feedback=deepcopy(cfg.get('action_feedback'))
    if previous_feedback is None:
        assert cfg['training_kind']=='native-reward-complete-brain-recurrent-PPO'
    else:
        assert cfg['training_kind']==ACTION_FEEDBACK_TRAINING_KIND
        assert previous_feedback==old['proof']['action_feedback']
        assert previous_feedback['schema']=='native-geometry-action-set-v1'
    assert cfg.get('reward_profile',{}).get('schema')=='native-life-emergency-geometry-v2'
    feedback=GeometryActionFeedback(frame_skip=cfg['frame_skip']);specification=feedback.specification()
    proof=json.loads(args.proof.read_text(encoding='utf-8'))
    assert proof['status']=='passed' and proof['learning_checkpoint_sha256']==args.expected_sha256
    assert proof['action_feedback']==specification
    assert proof['migrated_checkpoint_sha256']==old['proof']['migrated_checkpoint_sha256']
    assert proof['lineage_id']==old['proof']['lineage_id']
    allowed={'src/fly_touhou/whole_brain_ppo.py','src/fly_touhou/ppo_cuda_capture.py',
        'src/fly_touhou/geometry_migration.py','src/fly_touhou/geometry_action_feedback.py'}
    for name in PROOF_SOURCE_FILES:
        if name not in allowed:
            assert file_sha256(ROOT/name)==file_sha256(Path(old['source'])/name),name
    transition={'schema':'explicit-native-action-feedback-transition-v1',
        'parent_checkpoint_sha256':args.expected_sha256,'parent_checkpoint_path':str(args.checkpoint.resolve()),
        'action_feedback':specification,'from_training_kind':cfg['training_kind'],
        'to_training_kind':ACTION_FEEDBACK_TRAINING_KIND,'learning_performed_in_transition':False,
        'all_weights_critic_Adam_RNG_and_counters_preserved':True,
        'inference_actor_action_space_observation_and_neural_clock_changed':False}
    if previous_feedback is not None:
        transition['previous_action_feedback']=previous_feedback
        transition['previous_action_feedback_transition']=deepcopy(cfg['action_feedback_transition'])
    binding={'schema':PROOF_BINDING_SCHEMA,'path':str(args.proof.resolve()),'sha256':file_sha256(args.proof),
        'lineage_id':proof['lineage_id'],'action_feedback_validation':{
            'parent_checkpoint_sha256':args.expected_sha256,'previous_binding':old['binding'],
            'previous_source':old['source'],'previous_manifest_sha256':old['manifest_sha256'],
            'qualification_script_sha256':file_sha256(__file__),
            'comparison_helper_sha256':file_sha256(Path(__file__).with_name('requalify_geometry_policy.py'))}}
    data['geometry_validation_proof']=binding;cfg['geometry_validation_proof']=binding
    data['action_feedback_transition']=transition;cfg['action_feedback_transition']=transition
    cfg['action_feedback']=specification;cfg['training_kind']=ACTION_FEEDBACK_TRAINING_KIND
    _check_proof(data,proof);validate_geometry_deployment(data)
    a,b=deepcopy(before),deepcopy(data)
    for record in (a,b):
        record.pop('geometry_validation_proof',None);record.pop('action_feedback_transition',None)
        for name in ('geometry_validation_proof','action_feedback_transition','action_feedback'):
            record['training_config'].pop(name,None)
        record['training_config']['training_kind']=before['training_config']['training_kind']
    assert equal(a,b),'No learned tensor/state may be reset or replaced when adding supervision'
    assert file_sha256(args.checkpoint)==args.expected_sha256
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:torch.save(data,stream)
    assert equal(data,torch.load(args.output,map_location='cpu',weights_only=True))
    report={'schema':'qualified-native-action-feedback-initializer-v1','status':'passed',
        'input_sha256':args.expected_sha256,'output':str(args.output.resolve()),'output_sha256':file_sha256(args.output),
        'transition':transition,'binding':binding,'all_old_learning_state_bitwise_preserved':True,
        'model_trained':False,'native_game_started':False}
    with args.output.with_suffix('.report.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'status':'passed','output':report['output'],'sha256':report['output_sha256']},ensure_ascii=False))


if __name__=='__main__':main()
