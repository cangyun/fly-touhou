"""Prepare a real M1 resume and inspect weights on CPU; never dispatch or run a game."""
from pathlib import Path
from dataclasses import asdict
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--center',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from fly_touhou.thrl_control import ControlCenter,prepare_execution,digest
    from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.thrl_branching import (reuse_teacher_initialization,candidate_trial_config,
        inherit_candidate_history,warm_start)
    from fly_touhou.thrl_training_mode import validate_training_mode
    from fly_touhou.thrl_learning_revision import verify_ppo_resume
    from fly_touhou.th14_completion_protocol import baseline_parameters,exploration_parameters,learning_overrides
    from fly_touhou.thrl_model import MOActorCritic
    from fly_touhou.thrl_ppo import PPOConfig,make_optimizer
    import torch
    torch.set_num_threads(2)
    center=ControlCenter(a.center);before={name:digest(a.center/name) for name in
        ('configuration.json','registry.json','settings.json')}
    control=a.output/'prepared-only';control.mkdir()
    request=dict(schema='thrl-control-request-v1',job_id='offline-M1-full-story-resume-audit',
        keys=['M1'],mode='resume',settings=center.settings(),control_root=str(a.center.resolve()),
        control=str(control),output=str(a.output/'not-dispatched'),engine=center.config['engine'],
        template_configs=center.config['template_configs'],python=center.config['python'],
        previous={'M1':center.recovery_source('M1',center.registry())})
    write(a.output/'request-not-dispatched.json',request)
    prepare_execution(request)
    cfg=read(control/'branch_M1_config.json');validate_training_mode(cfg)
    cfg,reused=reuse_teacher_initialization(cfg)
    cfg=candidate_trial_config(cfg,0);validate_training_mode(cfg)
    source=Path(cfg['warm_start_checkpoint']);before_sha=digest(source)
    payload=torch.load(source,map_location='cpu',weights_only=False)
    model=MOActorCritic(**payload['model_config'])
    baseline=baseline_parameters(cfg)
    params=PPOConfig(**baseline,**exploration_parameters(cfg,baseline['entropy_coeff_start']),
                     **learning_overrides(cfg))
    prior=asdict(PPOConfig(**payload['config']['parameters']))
    verify_ppo_resume(prior,asdict(params),cfg)
    optimizer=make_optimizer(model,params)
    audit=warm_start(torch,model,optimizer,source,before_sha,exact_optimizer=True)
    assert audit['common_weights_exact']
    state=optimizer.state_dict()
    assert state['param_groups']==payload['optimizer']['param_groups']
    for key,row in state['state'].items():
        for name,value in row.items():
            expected=payload['optimizer']['state'][key][name]
            if torch.is_tensor(value):assert torch.equal(value,expected)
            else:assert value==expected
    history_dir=a.output/'history-only';history_dir.mkdir()
    history=inherit_candidate_history(cfg,history_dir)
    assert history['best'] is None and not history['completed_development']
    assert history['pending_development_at_start'] is None
    assert history['candidate_initial']['sha256']==before_sha
    assert digest(source)==before_sha
    assert before=={name:digest(a.center/name) for name in before}
    write(a.output/'report.json',dict(status='passed',utc=utc_now(),
        training_mode=cfg['training_mode'],learning_revision=cfg['learning_revision'],
        source_checkpoint=str(source),source_sha256=before_sha,
        source_optimized_decisions=cfg['candidate_decision_offset'],
        exact_model_tensors=len(model.state_dict()),exact_Adam_states=len(state['state']),
        model_and_Adam_unchanged=True,source_files_unchanged=True,
        historical_teacher_reused=reused,teacher_resume_audit=cfg.get('teacher_resume_audit'),
        old_course_scores_selected=False,old_course_state_restored=False,
        retired_score_runs=len(history['retired_score_runs']),
        old_pending_gate_run=False,operator_registry_unchanged=True,
        native_processes_started=0,production_training_jobs_started=0,optimizer_updates=0,
        training_started=False))
    print(str(a.output/'report.json'))


if __name__=='__main__':main()
