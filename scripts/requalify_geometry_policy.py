"""Revalidate the exact two lossless-log edits, preserving a learned checkpoint.

This deliberately cannot qualify changed training rules. The old runtime must
validate its original proof, the new full CUDA proof must cover the identical
zero-extension ancestor, and every covered source except two I/O lines must be
identical. No game, CUDA model, native action or optimizer is started here.
"""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from fly_touhou.experiment import file_sha256
from fly_touhou.geometry_migration import (_check_proof,validate_geometry_deployment,
    PROOF_BINDING_SCHEMA,PROOF_SOURCE_FILES)


def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a,b)
    if type(a)!=type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--previous-source',type=Path,required=True)
    parser.add_argument('--proof',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    if args.output.exists() or args.output.with_suffix('.report.json').exists():
        raise FileExistsError('All checkpoints and qualification evidence remain immutable')
    assert file_sha256(args.checkpoint)==args.expected_sha256,'Exact sealed learned checkpoint required'
    previous=args.previous_source.resolve(strict=True)
    manifest=json.loads((previous/'snapshot_manifest.json').read_text(encoding='utf-8'))
    assert manifest['schema']=='fly-touhou-frozen-source-v1'
    for row in manifest['files']:
        path=(previous/row['path']).resolve()
        assert path.is_relative_to(previous) and file_sha256(path)==row['sha256']
    for name in PROOF_SOURCE_FILES:
        old=(previous/name).read_text(encoding='utf-8')
        new=(ROOT/name).read_text(encoding='utf-8')
        if name=='src/fly_touhou/whole_brain_ppo.py':
            needle='with gzip.open(trajectory, "xt", encoding="utf-8") as stream:'
            changed='with gzip.open(trajectory, "xt", encoding="utf-8", compresslevel=3) as stream:'
            anchor="    if rollout_compression != 'none':\n"
            addition="    config['campaign_runtime']['native_trace_gzip_level'] = 3\n"
            assert old.count(needle)==old.count(anchor)==1
            old=old.replace(needle,changed).replace(anchor,addition+anchor)
        assert old==new,'Only the exact two native-log edits may be requalified: '+name
    data=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    before=deepcopy(data)
    module_path=previous/'src/fly_touhou/geometry_migration.py'
    spec=importlib.util.spec_from_file_location('fly_touhou._previous_geometry_log_qualification',module_path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.validate_geometry_deployment(data)
    old_binding=data.get('geometry_validation_proof',data['training_config'].get('geometry_validation_proof'))
    old_proof=json.loads(Path(old_binding['path']).read_text(encoding='utf-8'))
    proof=json.loads(args.proof.read_text(encoding='utf-8'));_check_proof(data,proof)
    assert proof['migrated_checkpoint_sha256']==old_proof['migrated_checkpoint_sha256']
    assert proof['lineage_id']==old_proof['lineage_id']
    ancestor=Path(proof['inputs']['migrated']['path'])
    assert file_sha256(ancestor)==proof['migrated_checkpoint_sha256']
    binding={'schema':PROOF_BINDING_SCHEMA,'path':str(args.proof.resolve()),
        'sha256':file_sha256(args.proof),'lineage_id':proof['lineage_id'],
        'lossless_log_requalification':{'schema':'exact-two-native-log-edits-v1',
            'previous_checkpoint_sha256':args.expected_sha256,'previous_binding':old_binding,
            'previous_source':str(previous),'previous_manifest_sha256':file_sha256(previous/'snapshot_manifest.json'),
            'script':str(Path(__file__).resolve()),'script_sha256':file_sha256(__file__),
            'learning_rules_changed':False,'tensor_state_changed':False}}
    data['geometry_validation_proof']=binding
    data['training_config']['geometry_validation_proof']=binding
    validate_geometry_deployment(data)
    for record in (before,data):
        record.pop('geometry_validation_proof',None)
        record['training_config'].pop('geometry_validation_proof',None)
    assert equal(before,data),'Any non-proof learner change is forbidden'
    data['geometry_validation_proof']=binding;data['training_config']['geometry_validation_proof']=binding
    assert file_sha256(args.checkpoint)==args.expected_sha256
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:torch.save(data,stream)
    restored=torch.load(args.output,map_location='cpu',weights_only=True)
    assert equal(data,restored)
    report={'schema':'learned-geometry-lossless-log-qualification-v1','status':'passed',
        'input_sha256':args.expected_sha256,'output_sha256':file_sha256(args.output),
        'output':str(args.output.resolve()),'binding':binding,
        'all_parameters_critic_Adam_RNG_counters_bitwise_preserved':True,
        'native_game_started':False,'training_performed':False,
        'counters':{k:data['rl_state'][k] for k in ('completed_timesteps','native_frames','updates','episodes_started')}}
    with args.output.with_suffix('.report.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'status':report['status'],'sha256':report['output_sha256'],'counters':report['counters']},ensure_ascii=False))


if __name__=='__main__':main()
