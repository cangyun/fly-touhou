"""Validate the published manual-start path without launching any world/job."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))


def main():
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
    from fly_touhou.thrl_control import ControlCenter,prepare_execution,digest
    from fly_touhou.thrl_branching import validate_parallel_config
    from fly_touhou.thrl_inputs import observation_shapes
    from fly_touhou.thrl_teacher_training import TeacherPretrainingConfig
    from fly_touhou.thrl_ppo import PPOConfig
    from fly_touhou.th14_completion_protocol import baseline_parameters,learning_overrides,exploration_parameters
    from monitor_thrl_comparison import load_comparison_snapshot
    import psutil

    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tests',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    center=ROOT/'artifacts/engineering/training-control-center-v1';c=ControlCenter(center)
    protected={name:(center/name).read_bytes() for name in ('configuration.json','registry.json','settings.json')}
    registry=c.registry();groups=list(c.config['template_configs'])
    assert not any(c.branch_activity(k,registry)['running'] for k in groups)
    runtime=Path(c.config['engine']['runtime'])
    manifest=read(runtime/'source_manifest.json')
    assert digest(runtime/'source_manifest.json')==c.config['engine']['manifest_sha256']
    for row in manifest['files']:
        assert digest(runtime/row['path'])==row['sha256'],row['path']
    prepared=output/'prepared';prepared.mkdir()
    request=dict(job_id='offline-audit-no-launch',engine=c.config['engine'],control=str(prepared),
        output=str(output/'not-launched'),keys=groups,mode='new',settings=c.settings(),
        template_configs=c.config['template_configs'],previous={k:None for k in groups},python=sys.executable)
    comparison=prepare_execution(request)
    records={}
    for key,branch in comparison['branches'].items():
        cfg=read(branch['config']);capacity=validate_parallel_config(cfg)
        assert cfg['training_recipe']=='th14-completion-v3'
        assert cfg['reward_schema']=='th14-completion-reward-v3'
        assert cfg['input_schema'] in ('original273-timers-v2','semantic-separated-timers-v2')
        assert observation_shapes(cfg['input_schema'])['features'][0]==cfg['model_overrides']['feature_dim']
        assert digest(cfg['warm_start_checkpoint'])==cfg['warm_start_sha256']
        params={**asdict(PPOConfig()),**baseline_parameters(cfg),**learning_overrides(cfg),
            **exploration_parameters(cfg,baseline_parameters(cfg)['entropy_coeff_start'])}
        ppo=PPOConfig(**params)
        assert ppo.seqs_per_batch==16 and ppo.gae_chunk_size==0 and ppo.target_kl==.03
        assert cfg['curriculum_eval_workers']==16 and cfg['curriculum_eval_batch_wait_ms']==2.
        assert cfg['curriculum_eval_min_batch_size']==4
        assert cfg['curriculum_options']['gate_episodes']==20
        assert not cfg.get('baseline_resume') and not cfg.get('teacher_pretraining_result')
        if key.endswith('1'):
            bc=TeacherPretrainingConfig(**cfg['teacher_pretraining'])
            assert bc.validation_enabled and bc.minimum_epochs==4 and bc.epochs==20
        else:assert not cfg.get('teacher_pretraining')
        records[key]=dict(config=branch['config'],capacity=capacity,parameters=params,
            input_schema=cfg['input_schema'],model_config=cfg['model_overrides'],
            checkpoint=cfg['warm_start_checkpoint'],checkpoint_sha256=cfg['warm_start_sha256'],
            teacher_pretraining=cfg.get('teacher_pretraining'),
            effective_gate_workers=min(cfg['workers'],cfg['curriculum_eval_workers']),
            gate_batch_wait_ms=cfg['curriculum_eval_batch_wait_ms'])
    for first,second in (('A0','A1'),('B0','B1'),('M0','M1')):
        assert records[first]['checkpoint_sha256']==records[second]['checkpoint_sha256']
        assert records[first]['parameters']==records[second]['parameters']
    snapshot=load_comparison_snapshot(center);c.decorate(snapshot)
    read_errors=list(snapshot['errors'])
    read_errors.extend(error for branch in snapshot['branches'].values() for error in branch['errors'])
    assert not read_errors,read_errors
    for key,branch in snapshot['branches'].items():
        assert branch['next_training_recipe']=='th14-completion-v3'
        assert not branch['control']['running']
    write(output/'panel_snapshot.json',snapshot)
    forbidden=[]
    for pr in psutil.process_iter(['pid','create_time','name','cmdline']):
        info=pr.info;names={Path(arg).name.lower() for arg in info['cmdline'] or []}
        if ((info['name'] or '').lower()=='th14.exe' or
            names.intersection({'train_thrl_native.py','train_thrl_branch.py','control_thrl_training.py'})):
            forbidden.append(info)
    assert not forbidden,forbidden
    assert all((center/name).read_bytes()==content for name,content in protected.items())
    suites=list(ET.parse(args.tests).getroot().iter('testsuite'))
    assert not any(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites)
    write(output/'report.json',dict(status='passed',created_utc=utc_now(),groups=records,
        runtime=str(runtime),source_manifest_sha256=digest(runtime/'source_manifest.json'),
        frozen_files_verified=len(manifest['files']),manual_start_preparation_only=True,
        training_jobs_started=0,original_game_processes_started=0,current_training_or_game_processes=forbidden,
        configuration_registry_settings_unchanged_by_audit=True,panel_read_errors=[],
        CPU_tests=sum(int(s.get('tests',0)) for s in suites),CPU_report=str(args.tests.resolve())))
    print(str(output/'report.json'))


if __name__=='__main__':main()
