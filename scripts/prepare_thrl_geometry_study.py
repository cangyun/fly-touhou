"""Freeze the qualified geometry/headless runtime and prepare fresh A/B/C.

Never resumes a 273-feature checkpoint into the new sensor. All branches start
from one NEW common initialization, with separate attention/teacher parameters.
"""
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


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(data,stream,indent=2,ensure_ascii=False,allow_nan=False)


def main():
    raise RuntimeError('Stacked2299 preparation superseded. Use prepare_thrl_input_study.py for the reviewed four-group study.')
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--runtime',required=True,type=Path);p.add_argument('--prepared',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--evidence',required=True,type=Path)
    p.add_argument('--tests',required=True,type=Path)
    a=p.parse_args()
    if any(path.exists() for path in (a.runtime,a.prepared,a.output)):
        raise FileExistsError('Fresh runtime/preparation/training output required; preserve old results')
    suites=list(ET.parse(a.tests).getroot().iter('testsuite'))
    if not suites or any(int(s.get('failures',0)) or int(s.get('errors',0)) for s in suites):
        raise ValueError('Required regression qualification failed')
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_segment_env import load_start_bank
    entries=[]
    for seed in (12345,23023,37037):
        bank=a.evidence/f'curriculum-bank-seed{seed}-v2'/'bank.json'
        rows=load_start_bank(bank)
        if {row['target'] for row in rows}!={'opening','midboss','boss'}:
            raise ValueError('Three independently verified targets required for every training seed')
        entries.extend(rows)
    for relative in ('trajectory-v4/report.json','prefix-reset-v2/report.json',
                     'segment-env-v3/report.json','cuda-ready-pipeline-v1/report.json'):
        report=read_shared_json(a.evidence/relative)
        if report['status']!='passed':raise ValueError('Missing qualification: '+relative)
        if relative.startswith('segment-env') and not report.get('second_reset'):
            raise ValueError('Repeated segment reset qualification required')
        if relative.startswith('cuda-ready') and (report.get('workers')!=8 or report.get('adam_tensors_on_cuda',0)<1):
            raise ValueError('Actual eight-world CUDA Adam qualification required')
    protected={}
    old=Path('./outputs/source_snapshots/20261021-thrl-parallel-abc-v3')
    for name in ('thrl_ppo.py','thrl_rewards.py'):
        current=ROOT/'src/fly_touhou'/name
        if sha(current)!=sha(old/'src/fly_touhou'/name):raise ValueError('PPO/reward changed without authorization')
        protected[name]=sha(current)
    import numpy as np
    import torch
    from fly_touhou.thrl_geometry_observation import FEATURE_DIM,THRLGeometryObservationEncoder
    from fly_touhou.thrl_model import MOActorCritic
    from fly_touhou.thrl_ppo import PPOConfig,make_optimizer
    from fly_touhou.thrl_branching import validate_parallel_config
    from train_thrl_native import save_checkpoint
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(2)
    random.seed(11);np.random.seed(11);torch.manual_seed(11);torch.cuda.manual_seed_all(11)
    a.prepared.mkdir(parents=True,exist_ok=False)
    a.runtime.mkdir(parents=True,exist_ok=False)
    records=[]
    for folder in ('src','scripts','profiles'):
        for path in sorted((ROOT/folder).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix not in ('.py','.js','.cjs','.json'):continue
            content=path.read_bytes()
            if path.suffix=='.py':compile(content,str(path),'exec')
            relative=path.relative_to(ROOT);target=a.runtime/relative;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(content);records.append({'path':relative.as_posix(),'bytes':len(content),'sha256':sha(target)})
    for relative in ('COPYING','README.md','pyproject.toml','data/connectome/model.py'):
        path=ROOT/relative;target=a.runtime/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target);records.append({'path':relative,'bytes':target.stat().st_size,'sha256':sha(target)})
    write(a.runtime/'source_manifest.json',{'schema':'fly-touhou-geometry-headless-frozen-v1',
        'origin':str(ROOT),'files':records,'protected_scientific_core':protected})
    write(a.prepared/'start_bank.json',{'schema':'th14-native-start-bank-v2','entries':entries,
        'training_seeds':[12345,23023,37037],'formal_test':False,'resources_injected':False})
    model=MOActorCritic(feature_dim=FEATURE_DIM,action_dim=19).cuda()
    optimizer=make_optimizer(model)
    common=a.prepared/'common_initial.pt'
    save_checkpoint(torch,common,model,optimizer,decisions=0,updates=0,adam=0,
        config={'parameters':asdict(PPOConfig()),'rollout_total_decisions':2048,
                'observation':THRLGeometryObservationEncoder().specification(),
                'new_initialization':True,'prior_v3_weights_used':False})
    branches={}
    for name,attention in (('A','none'),('B','both'),('C','both')):
        overrides={'feature_dim':FEATURE_DIM,'attention_initialization':'identity'}
        if name!='A':overrides.update(object_embed_dim=256,object_attention_heads=8,
            object_attention_layers=4,spatial_attention_layers=2)
        config={'schema':'thrl-native-geometry-architecture-config-v2','branch':name,
            'game_exe':'games/th14/th14.exe',
            'profile':str((a.runtime/'profiles/th14_1.00b_original.json').resolve()),
            'source_runtime':str(a.runtime.resolve()),'source_manifest_sha256':sha(a.runtime/'source_manifest.json'),
            'warm_start_checkpoint':str(common.resolve()),'warm_start_sha256':sha(common),
            'seed':11,'workers':8,'attention':attention,'model_overrides':overrides,
            'headless':True,'collector_mode':'ready','start_bank':str((a.prepared/'start_bank.json').resolve()),
            'start_bank_sha256':sha(a.prepared/'start_bank.json'),
            'full_story_fraction':.25,'segment_end':None,'policy_action_count':19,
            'native_frame_horizon':None,'finite_native_lives':True,'autonomous_bomb':True,'continues_allowed':False,
            'rollout_total_decisions':2048,'trial_decisions':262144,'development_decisions':[32768,65536,131072,262144],
            'development_episodes':6,'validation_episodes_per_policy':20,'checkpoint_every_updates':4,
            'optimization_trials':2,'trial_wall_cap_seconds':8*3600,'study_wall_cap_seconds':18*3600,
            'torch_threads':2,'api_port':18740+ord(name)-ord('A'),
            'comparison_stop_path':str((a.output/'STOP').resolve()),
            'study_name':f'TH14-geometry-headless-{name}-20261022-v1',
            'teacher_pretraining':({'samples':8192,'workers':4,'epochs':4,'seq_len':16,
                'seqs_per_batch':4,'learning_rate':.0001,'max_grad_norm':.5} if name=='C' else None),
            'first_trial_protected':True,'original_game_RNG_controlled':False,
            'prefix_rng_controlled':True,'full_story_rng_controlled':False,
            'formal_acceptance':False,'new_initialization':True,'prior_v3_weights_used':False,
            'user_authorization':'TH14 headless/uncapped, measured collision geometry, multi-layer spatial/object attention, then ready collection and segment curriculum; finite lives/autonomous Bomb/no continue; CUDA A/B/C pools'}
        config['game_exe_sha256']=sha(config['game_exe']);config['profile_sha256']=sha(config['profile'])
        config['capacity']=validate_parallel_config(config)
        path=a.prepared/f'branch_{name}_config.json';write(path,config)
        branches[name]={'directory':f'branch_{name}','config':str(path.resolve()),'workers':8,
            'attention':attention,'teacher_pretraining':name=='C','source_runtime':str(a.runtime.resolve())}
    comparison={'schema':'thrl-native-geometry-comparison-config-v2','branches':branches,
        'python':sys.executable,'cwd':str(ROOT),'api_port':18739,'branch_launch_stagger_seconds':20,
        'source_runtime':str(a.runtime.resolve()),'source_manifest_sha256':sha(a.runtime/'source_manifest.json'),
        'observation_schema':THRLGeometryObservationEncoder().specification()['schema'],
        'trial_decisions':262144,'optimization_trials_per_branch':2,'native_frame_horizon':None,
        'PPO_reward_unchanged':protected,'formal_acceptance':False}
    write(a.prepared/'comparison_config.json',comparison)
    write(a.prepared/'preparation_report.json',{'status':'passed','runtime':str(a.runtime),
        'manifest_sha256':sha(a.runtime/'source_manifest.json'),'source_files':len(records),
        'start_entries':len(entries),'independent_verified_landings':sum(r['verified_independent_resets'] for r in entries),
        'common_initial_sha256':sha(common),'old_checkpoints_used':False,
        'baseline_parameters':model.parameter_counts(),'protected_scientific_core':protected})
    print(json.dumps({'status':'prepared','runtime':str(a.runtime),'config':str(a.prepared/'comparison_config.json'),
                      'source_files':len(records),'start_entries':len(entries)},ensure_ascii=False))


if __name__=='__main__':main()
