"""Prepare the next committed NBNM run with supported-continuation targets.

This only writes a configuration. It never launches training or changes the
active control pointer. Use the chosen parent's frozen report before calling.
"""
from pathlib import Path
import argparse
import copy
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'native-validation', 'config-directory', 'run-output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--additional-segments', type=int, default=8)
    parser.add_argument('--geometry-validation',type=Path)
    parser.add_argument('--geometry-resume-validation',type=Path)
    parser.add_argument('--geometry-horizon',type=int,choices=(24,96),default=24)
    args = parser.parse_args()
    if args.additional_segments < 1:
        parser.error('Use at least one segment')

    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.nbnm_continuation_risk import REVISION
    # A running v10 may be resumed after a crash, with its original targets.
    # Only a separate completed round can be the parent of this new experiment.
    parent = args.parent.resolve()
    progress = read_shared_json(parent/'progress.json')
    final = read_shared_json(parent/'final-report.json') if (parent/'final-report.json').exists() else {}
    if progress.get('phase') != 'complete' or not final.get('training_complete'):
        raise ValueError('Parent training and its frozen evaluation must finish first; live run unchanged')
    evidence = read_shared_json(args.native_validation)
    if not evidence.get('supported_turn_validation_passed'):
        raise ValueError('Finish the native turn-path validation before preparing this trial')

    import torch
    from fly_touhou.nbnm_run import validate_spec
    from fly_touhou.nbnm_resume import validate_resume
    checkpoint = parent/'latest-training.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    spec = copy.deepcopy(saved['spec'])
    spec['nbnm']['risk_target_revision'] = REVISION
    geometry=None
    if args.geometry_validation:
        from fly_touhou.nbnm_geometry_revision import REVISION as GEOMETRY_REVISION
        geometry_evidence=read_shared_json(args.geometry_validation)
        errors=sum(v for k,v in geometry_evidence['counts'].items() if k.startswith(('new_outside','new_size_under')))
        errors+=sum(v for k,v in geometry_evidence.get('laser_checks',{}).items() if '_outside' in k)
        errors+=geometry_evidence.get('fixture_failures',0)
        if geometry_evidence.get('status')!='complete' or errors or geometry_evidence['counts'].get('new_known24',0)<100:
            raise ValueError('Finish the original-game rule/size validation before preparing geometry')
        if args.geometry_resume_validation is None:raise ValueError('Supply the completed native continuation probe')
        continuation=read_shared_json(args.geometry_resume_validation)
        if continuation.get('status')!='complete' or not continuation['migration']['geometry_migration']['legacy_input_reproduced']:
            raise ValueError('Finish the native world/input migration probe')
        if args.geometry_horizon!=24:
            raise ValueError('The measured next-round default is24;96 remains an isolated performance experiment')
        args.config_directory.mkdir(parents=True,exist_ok=True)
        cfg=read_shared_json(spec['environment_config']);profile=read_shared_json(cfg['profile'])
        profile.setdefault('hooks',{}).update(native_ex_program=GEOMETRY_REVISION,native_ex_program_horizon=args.geometry_horizon)
        profile_path=args.config_directory.resolve()/'native-profile.json'
        env_path=args.config_directory.resolve()/'environment.json'
        write_json(profile_path,profile);cfg['profile']=str(profile_path);write_json(env_path,cfg)
        geometry=dict(revision=GEOMETRY_REVISION,horizon=args.geometry_horizon)
        spec['nbnm']['native_rule_geometry']=geometry;spec['environment_config']=str(env_path)
    migration = validate_resume(saved, spec)
    spec['resume_checkpoint'] = str(checkpoint)
    spec['reference_checkpoint'] = str(checkpoint)
    spec['output'] = str(args.run_output.resolve())
    spec['tensorboard_directory'] = str(args.run_output.resolve()/'tensorboard')
    spec['decision_budget'] = saved['completed_decisions'] + args.additional_segments*spec['segment_decisions']
    spec['resume_extension_segments'] = args.additional_segments
    spec['evaluate_at_end'] = True
    spec['evaluation_tasks'] = ['full_story']
    validation = validate_spec(spec)
    args.config_directory.mkdir(parents=True, exist_ok=True)
    write_json(args.config_directory/'route.json', spec)
    report = dict(status='prepared_not_started', parent=str(parent), source_checkpoint=str(checkpoint),
        source_decisions=saved['completed_decisions'], source_updates=saved['optimizer_updates'],
        additional_decisions=args.additional_segments*spec['segment_decisions'],
        additional_updates=args.additional_segments*(spec['segment_decisions']//256),
        changed_learning_options=['nbnm.risk_target_revision']+(['nbnm.native_rule_geometry'] if geometry else []),
        risk_target_revision=REVISION,native_rule_geometry=geometry,
        migration=migration, validation=validation,
        learning_rates_sampling_rewards_and_topology_preserved=True,
        active_control_pointer_changed=False, new_training_started=False)
    write_json(args.config_directory/'preparation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
