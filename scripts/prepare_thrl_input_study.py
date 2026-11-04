"""Freeze reviewed input schemas and prepare the authorized 2x2 input/teacher study."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from prepare_thrl_geometry_study import sha,write


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('runtime','prepared','output','evidence','tests','control','bank'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    if any(path.exists() for path in (a.runtime,a.prepared,a.output,a.control)):
        raise FileExistsError('Use fresh versioned preparation/runtime/control/output; never replace earlier results')
    suites=list(ET.parse(a.tests).getroot().iter('testsuite'))
    if not suites or sum(int(row.get('tests',0)) for row in suites)<320 or any(
            int(row.get('failures',0)) or int(row.get('errors',0)) for row in suites):
        raise ValueError('Current required regression qualification missing/failed')
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_segment_env import load_start_bank
    from fly_touhou.thrl_inputs import ORIGINAL_INPUT,SEPARATED_INPUT,make_encoder
    from audit_thrl_input_semantics import audit
    semantic=audit()
    qualification={}
    for name in ('trajectory-v4','prefix-reset-v2','segment-env-v3','aligned-prefix-bank-v1',
                 'cuda-original-prefix-pipeline-v1','cuda-separated-prefix-pipeline-v1',
                 'cuda-original-prefix-teacher-pipeline-v1','cuda-separated-prefix-teacher-pipeline-v1'):
        path=a.evidence/name/'report.json';report=read_shared_json(path)
        if report['status']!='passed':raise ValueError('Missing current qualification: '+name)
        if name.startswith('cuda-'):
            expected=ORIGINAL_INPUT if 'original-' in name else SEPARATED_INPUT
            if report.get('input_schema')!=expected or report.get('adam_tensors_on_cuda',0)<1:
                raise ValueError('Current input/GPU qualification mismatch: '+name)
            if 'teacher-' in name and report.get('teacher',{}).get('BC_adam_minibatch_steps',0)<1:
                raise ValueError('Missing actual BC/GPU transfer: '+name)
            if 'teacher-' not in name and report.get('workers')!=8:
                raise ValueError('Eight-world functional qualification required')
            if expected==SEPARATED_INPUT and report.get('observation_shapes',{}).get('objects')!=[96,21]:
                raise ValueError('Obsolete stacked/20-field input qualification')
        qualification[name]={'path':str(path.resolve()),'sha256':sha(path)}
    entries=[]
    entries=load_start_bank(a.bank)
    for seed in (12345,23023,37037):
        rows=[row for row in entries if row['native_seed']==seed]
        if {row['target'] for row in rows}!={'opening','midboss','boss'} or any(
                len(row.get('aligned_verification_receipts',[]))!=2 for row in rows):
            raise ValueError('Three independently verified targets required per source seed')

    import numpy as np
    import torch
    from fly_touhou.thrl_model import MOActorCritic
    from fly_touhou.thrl_ppo import PPOConfig,make_optimizer
    from fly_touhou.thrl_branching import validate_parallel_config
    from train_thrl_native import save_checkpoint
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no fallback')
    torch.set_num_threads(2)
    a.prepared.mkdir(parents=True,exist_ok=False);a.runtime.mkdir(parents=True,exist_ok=False)
    records=[]
    for folder in ('src','scripts','profiles'):
        for path in sorted((ROOT/folder).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix not in ('.py','.js','.cjs','.json'):continue
            content=path.read_bytes()
            if path.suffix=='.py':compile(content,str(path),'exec')
            relative=path.relative_to(ROOT);target=a.runtime/relative;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(content);records.append({'path':relative.as_posix(),'bytes':len(content),'sha256':sha(target)})
    for relative in ('COPYING','README.md','pyproject.toml','data/connectome/model.py'):
        source=ROOT/relative;target=a.runtime/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);records.append({'path':relative,'bytes':target.stat().st_size,'sha256':sha(target)})
    write(a.runtime/'source_manifest.json',{'schema':'thrl-semantic-input-frozen-v1','origin':str(ROOT),
        'files':records,'scientific_core':semantic['common_training']})
    write(a.prepared/'input_semantics.json',semantic)
    write(a.prepared/'start_bank.json',{'schema':'th14-native-start-bank-v2','entries':entries,
        'training_seeds':[12345,23023,37037],'formal_test':False,'resources_injected':False})

    initial={};shared=None;shared_audit={}
    for schema in (ORIGINAL_INPUT,SEPARATED_INPUT):
        random.seed(11);np.random.seed(11);torch.manual_seed(11);torch.cuda.manual_seed_all(11)
        model=MOActorCritic(**semantic['models'][schema]['configuration']).cuda()
        if shared is None:
            shared={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        else:
            target=model.state_dict()
            common=[key for key,value in shared.items() if key in target and target[key].shape==value.shape]
            merged={**target,**{key:shared[key] for key in common}}
            model.load_state_dict(merged,strict=True)
            shared_audit={'equal_shared_state_tensors':common,
                'different_input_state_keys':sorted((set(shared)|set(target))-set(common)),
                'shared_state_exact':all(torch.equal(model.state_dict()[key].cpu(),shared[key]) for key in common)}
            if not shared_audit['shared_state_exact']:raise AssertionError('Mismatched common initialization')
        # Both encodings and both teacher factors begin PPO's sampler from the
        # same declared seed. Input-module constructor order cannot shift it.
        random.seed(11);np.random.seed(11);torch.manual_seed(11);torch.cuda.manual_seed_all(11)
        path=a.prepared/f'initial_{schema}.pt'
        save_checkpoint(torch,path,model,make_optimizer(model),decisions=0,updates=0,adam=0,
            config={'parameters':asdict(PPOConfig()),'rollout_total_decisions':2048,
                'observation':make_encoder(schema).specification(),'new_initialization':True})
        initial[schema]={'checkpoint':str(path.resolve()),'sha256':sha(path),'model_config':model.model_config()}
        del model
    torch.cuda.empty_cache()
    branches={}
    cases=(('A0',ORIGINAL_INPUT,False,'原输入'),('B0',SEPARATED_INPUT,False,'整理输入'),
           ('A1',ORIGINAL_INPUT,True,'原输入＋教师'),('B1',SEPARATED_INPUT,True,'整理输入＋教师'))
    for index,(name,schema,teacher,label) in enumerate(cases):
        overrides={key:value for key,value in initial[schema]['model_config'].items() if key not in ('attention','action_dim')}
        config={'schema':'thrl-input-teacher-factorial-config-v1','branch':name,'display_name':label,
            'input_schema':schema,'game_exe':'games/th14/th14.exe',
            'profile':str((a.runtime/'profiles/th14_1.00b_original.json').resolve()),
            'source_runtime':str(a.runtime.resolve()),'source_manifest_sha256':sha(a.runtime/'source_manifest.json'),
            'warm_start_checkpoint':initial[schema]['checkpoint'],'warm_start_sha256':initial[schema]['sha256'],
            'seed':11,'workers':8,'attention':'both','model_overrides':overrides,
            'headless':True,'collector_mode':'ready','start_bank':str((a.prepared/'start_bank.json').resolve()),
            'start_bank_sha256':sha(a.prepared/'start_bank.json'),'full_story_fraction':.25,'segment_end':None,
            'policy_action_count':19,'native_frame_horizon':None,'finite_native_lives':True,
            'autonomous_bomb':True,'continues_allowed':False,'rollout_total_decisions':2048,
            'trial_decisions':262144,'development_decisions':[32768,65536,131072,262144],
            'development_episodes':6,'validation_episodes_per_policy':20,'checkpoint_every_updates':4,
            'optimization_trials':2,'trial_wall_cap_seconds':8*3600,'study_wall_cap_seconds':18*3600,
            'torch_threads':2,'api_port':18770+index,'comparison_stop_path':str((a.output/'STOP').resolve()),
            'study_name':f'TH14-{a.output.name}-{name}-20261022','first_trial_protected':True,
            'teacher_pretraining':({'samples':8192,'workers':4,'epochs':4,'seq_len':16,'seqs_per_batch':4,
                'learning_rate':.0001,'max_grad_norm':.5} if teacher else None),
            'match_PPO_initial_RNG':True,'original_game_RNG_controlled':False,
            'prefix_rng_controlled':True,'full_story_rng_controlled':False,'formal_acceptance':False,
            'primary_comparison_trial':0,'HPO_comparison_is_exploratory':True,'new_initialization':True,
            'old_checkpoints_used':False,
            'user_authorization':'Two inputs each with/without teacher; reviewed semantic inputs; original TH14 headless/uncapped; eight instances per shared CUDA policy; finite lives/autonomous Bomb/no continue'}
        config['game_exe_sha256']=sha(config['game_exe']);config['profile_sha256']=sha(config['profile'])
        config['capacity']=validate_parallel_config(config)
        path=a.prepared/f'branch_{name}_config.json';write(path,config)
        branches[name]={'directory':f'branch_{name}','label':label,'config':str(path.resolve()),'workers':8,
            'attention':'both','teacher_pretraining':teacher,'input_schema':schema,
            'script':str((a.runtime/'scripts/train_thrl_branch.py').resolve()),'source_runtime':str(a.runtime.resolve())}
    comparison={'schema':'thrl-input-teacher-comparison-config-v1','branches':branches,'python':sys.executable,
        'cwd':str(a.runtime.resolve()),'api_port':18769,'branch_launch_stagger_seconds':10,
        'source_runtime':str(a.runtime.resolve()),'source_manifest_sha256':sha(a.runtime/'source_manifest.json'),
        'trial_decisions':262144,'optimization_trials_per_branch':2,'native_frame_horizon':None,
        'primary_comparison':'trial000 across four groups; HPO maxima are exploratory',
        'qualification':qualification,'formal_acceptance':False}
    write(a.prepared/'comparison_config.json',comparison)
    write(a.prepared/'background_spec.json',{'schema':'fly-touhou-background-job-v1','python':sys.executable,
        'script':str((a.runtime/'scripts/run_thrl_comparison.py').resolve()),
        'arguments':['--config',str((a.prepared/'comparison_config.json').resolve()),'--output',str(a.output.resolve())],
        'cwd':str(a.runtime.resolve()),'control_directory':str(a.control.resolve()),
        'environment':{'PYTHONUTF8':'1','CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}})
    result={'status':'prepared','runtime':str(a.runtime.resolve()),'source_files':len(records),
        'manifest_sha256':sha(a.runtime/'source_manifest.json'),'start_entries':len(entries),
        'independent_landings':sum(row['verified_independent_resets'] for row in entries),
        'initializations':initial,'matched_initialization':shared_audit,
        'qualification':qualification,'primary_ppo_decisions_per_group':262144,
        'maximum_total_PPO_decisions':4*2*262144,'teacher_samples_per_teacher_group':8192,
        'expected_primary_PPO_Adam_per_group':12288,'BC_Adam_per_teacher_group':512,
        'old_stacked_preparation_dispatched':False,'formal_acceptance':False}
    write(a.prepared/'preparation_report.json',result)
    print(json.dumps({'status':'prepared','runtime':str(a.runtime),'cases':list(branches),'start_entries':len(entries)}))


if __name__=='__main__':main()
