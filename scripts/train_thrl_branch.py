"""One multi-instance CUDA policy branch, with a protected baseline and Optuna.

The independent branch owns its worlds, optimizer, recurrent states and output.
Initial A/B/C architecture comparisons use identical source PPO parameters.
Only later tuning candidates can be pruned, after two complete dev nodes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.atomic_json_io import atomic_write_json as write_json
from fly_touhou.thrl_study_io import StatusAPI, read_shared_json, utc_now
from train_thrl_native import (StopRequested, digest, evaluate, train_trial)
from fly_touhou.thrl_training_mode import validate_training_mode, is_full_story


class CandidateExecutionFailed(RuntimeError):
    """Recorded as FAIL, never converted into a pruned/complete game result."""


def optimize_candidates(study, objective, count, refresh):
    study.optimize(objective,n_trials=count,n_jobs=1,
        callbacks=[lambda *_:refresh()],gc_after_trial=True,catch=(CandidateExecutionFailed,))
    if not any(trial.state.name=='COMPLETE' for trial in study.trials):
        raise RuntimeError('No completed candidate; failed infrastructure is not a selection score')
    return study.best_trial


def candidate_initial_reference(run):
    config=read_shared_json(Path(run)/'config.json')
    checkpoint=Path(config.get('candidate_initial_checkpoint',Path(run)/'initial.pt'))
    if config.get('candidate_initial_sha256') and digest(checkpoint)!=config['candidate_initial_sha256']:
        raise ValueError('Candidate original initial checkpoint identity changed')
    return checkpoint


def final_validation_plan(study,best):
    """Always test protected trial0, even if an HPO candidate scores higher."""
    primary=next((row for row in study.trials if row.number==0 and row.state.name=='COMPLETE'),None)
    if primary is None:
        selected_run=Path(best.user_attrs['run_output'])
        return [('selected_HPO',selected_run/'best_development.pt',best),
                ('same_initial',candidate_initial_reference(selected_run),best)]
    primary_run=Path(primary.user_attrs['run_output'])
    plan=[('primary',primary_run/'best_development.pt',primary)]
    if best.number!=0:
        plan.append(('selected_HPO',Path(best.user_attrs['run_output'])/'best_development.pt',best))
    plan.append(('same_initial',candidate_initial_reference(primary_run),primary))
    return plan


def verify_sources(config):
    validate_training_mode(config)
    from fly_touhou.thrl_branching import validate_parallel_config
    capacity = validate_parallel_config(config)
    if config.get('start_bank') and digest(config['start_bank'])!=config.get('start_bank_sha256'):
        raise ValueError('Pinned native starting-point bank changed')
    if config.get('curriculum_manifest'):
        from fly_touhou.thrl_course_store import CourseStore
        from fly_touhou.thrl_curriculum import configuration
        CourseStore(config['curriculum_manifest'],config['curriculum_manifest_sha256'])
        configuration(config.get('curriculum_options'))
    for key in ('game_exe','profile','warm_start_checkpoint'):
        hash_key = {'game_exe':'game_exe_sha256','profile':'profile_sha256',
                    'warm_start_checkpoint':'warm_start_sha256'}[key]
        if digest(config[key]) != config[hash_key]:
            raise ValueError('Pinned identity changed: '+key)
    if config.get('baseline_resume'):
        resume=config['baseline_resume']
        if digest(resume['checkpoint'])!=resume['sha256']:
            raise ValueError('Pinned candidate recovery checkpoint changed')
    manifest = ROOT/'source_manifest.json'
    if digest(manifest) != config['source_manifest_sha256']:
        raise ValueError('Frozen branch manifest changed')
    for item in read_shared_json(manifest)['files']:
        path = (ROOT/item['path']).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != item['sha256']:
            raise ValueError('Frozen runtime file differs: '+item['path'])
    return capacity


def run(args):
    config = read_shared_json(args.config)
    capacity = verify_sources(config)
    # Windows spawn must never import Torch in the native workers.
    import torch
    import optuna
    from fly_touhou.thrl_ppo import PPOConfig
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU fallback is forbidden')
    torch.set_num_threads(config.get('torch_threads',2))
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'study_config.json',config)
    state = {'schema':'thrl-native-architecture-branch-v1','status':'starting',
        'branch':config['branch'],'attention':config['attention'],'workers':config['workers'],
        'parameter_counts':None,'current_run':None,'trials':[],'updated_utc':utc_now(),
        'capacity':capacity,'CUDA_required':True,'device':'cuda:0',
        'gpu':torch.cuda.get_device_name(0),'formal_acceptance':False}
    write_json(output/'study_status.json',state)
    api = StatusAPI(output,config['api_port'])
    started = time.monotonic()
    study = None

    def check_stop():
        parent_stop=config.get('comparison_stop_path')
        operator_stop=config.get('operator_stop_path')
        if ((output/'STOP').exists() or parent_stop and Path(parent_stop).exists()
                or operator_stop and Path(operator_stop).exists()):
            raise StopRequested('operator STOP; branch closes all owned native worlds')
        if config.get('study_wall_cap_seconds') is not None and time.monotonic()-started >= config['study_wall_cap_seconds']:
            raise StopRequested('branch wall budget reached')

    def refresh(progress=None):
        state.update(updated_utc=utc_now(),wall_seconds=time.monotonic()-started,
                     dashboard_api=api.url)
        if progress:
            state['current_progress'] = progress
            state['parameter_counts'] = progress.get('parameter_counts')
        if study is not None:
            state['trials'] = [{'number':t.number,'state':t.state.name,'score':t.value,
                'parameters':t.params,**t.user_attrs} for t in study.trials]
        write_json(output/'study_status.json',state)

    try:
        from fly_touhou.thrl_branching import inherit_study_lineage, reuse_teacher_initialization
        lineage=inherit_study_lineage(config,output)
        if lineage:
            state['historical_study_lineage']=lineage
            state['historical_trials']=lineage['trials']
        if config.get('baseline_resume'):
            state['candidate_recovery']={**config['baseline_resume'],
                'status':'protected baseline continuation, fresh native worlds',
                'candidate_target_decisions':config['trial_decisions'],
                'remaining_optimized_decisions':config['trial_decisions']-config['baseline_resume']['optimized_decisions'],
                'paid_cost_is_lower_bound':True,'old_FAILED_trials_remain_FAILED':True}
        config,teacher_reused=reuse_teacher_initialization(config)
        if config.get('teacher_pretraining') and not teacher_reused:
            from fly_touhou.thrl_teacher_training import pretrain_teacher_branch
            state['status']='teacher_pretraining'
            refresh()
            teacher_result = pretrain_teacher_branch(torch,config,output,
                on_progress=lambda row:(state.update(current_run=str(output/'teacher_pretraining')),
                                        refresh(row)), check_stop=check_stop)
            config = {**config,'warm_start_checkpoint':teacher_result['checkpoint'],
                      'warm_start_sha256':teacher_result['checkpoint_sha256'],
                      'teacher_pretraining_result':teacher_result}
            state['teacher_pretraining_result']=teacher_result
        elif teacher_reused:
            state.update(teacher_pretraining_reused=True,
                         teacher_pretraining_result=config['teacher_pretraining_result'])
        write_json(output/'effective_study_config.json',config)
        study = optuna.create_study(study_name=config['study_name'],direction='maximize',
            storage='sqlite:///'+str(output/'optuna.sqlite3'),load_if_exists=False,
            sampler=optuna.samplers.TPESampler(seed=config['seed'],n_startup_trials=1),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=1,n_warmup_steps=2,
                                              interval_steps=1,n_min_trials=1))
        from fly_touhou.th14_completion_protocol import baseline_parameters,exploration_parameters,learning_overrides,search_space,RECIPE
        baseline=baseline_parameters(config)
        study.enqueue_trial(baseline,user_attrs={'role':'protected architecture baseline'})
        refresh()

        def objective(trial):
            check_stop()
            parameters=asdict(PPOConfig())
            space=search_space(config)
            parameters.update(learning_rate=trial.suggest_float('learning_rate',*space['learning_rate'],log=True),
                gamma=trial.suggest_categorical('gamma',space['gamma']),
                gae_lambda=trial.suggest_categorical('gae_lambda',space['gae_lambda']),
                entropy_coeff_start=trial.suggest_categorical('entropy_coeff_start',space['entropy_coeff_start']))
            parameters.update(learning_overrides(config))
            parameters.update(exploration_parameters(config,parameters['entropy_coeff_start']))
            run_output=output/f'trial_{trial.number:03d}_{config["attention"]}'
            trial.set_user_attr('run_output',str(run_output))
            trial.set_user_attr('attention',config['attention'])
            from fly_touhou.thrl_branching import candidate_trial_config
            trial_config=candidate_trial_config(config,trial.number)
            state.update(status='training',current_run=str(run_output))
            refresh()
            trial.set_user_attr('candidate_recovery',trial_config['candidate_resume_applied'])
            trial.set_user_attr('candidate_optimized_offset',trial_config['candidate_decision_offset'])
            try:
                result=train_trial(torch,trial_config,output,run_output,parameters,trial=trial,
                    study_started=started,on_progress=refresh,attention=config['attention'])
            except (StopRequested,optuna.TrialPruned):
                raise
            except Exception as error:
                from fly_touhou.thrl_vector import THRLVectorError
                from fly_touhou.native import NativeError
                trial.set_user_attr('failure_kind',
                    'infrastructure_error' if isinstance(error,(THRLVectorError,NativeError)) else 'candidate_execution_error')
                trial.set_user_attr('exception_type',type(error).__name__)
                trial.set_user_attr('exception',str(error))
                trial.set_user_attr('failure_record',str(run_output/'failure.json'))
                if isinstance(error,(THRLVectorError,NativeError)):
                    # A broken native environment is not an HPO candidate.
                    # Preserve FAIL and stop this branch instead of launching
                    # the same infrastructure again with another learning rate.
                    raise
                raise CandidateExecutionFailed(str(error)) from error
            trial.set_user_attr('segment_decisions',result['decisions'])
            trial.set_user_attr('decisions',result.get('candidate_decisions',result['decisions']))
            trial.set_user_attr('adam_minibatch_steps',result['adam_minibatch_steps'])
            trial.set_user_attr('best_checkpoint',result['best_checkpoint'])
            trial.set_user_attr('best_development_score',result['best_development_score'])
            return result['latest_development']['score']

        best=optimize_candidates(study,objective,config.get('optimization_trials',2),refresh)
        selected_run=Path(best.user_attrs['run_output'])
        selected=selected_run/'best_development.pt'
        selection={'trial':best.number,'architecture':config['attention'],
                   'checkpoint':str(selected),'sha256':digest(selected),
                   'development':read_shared_json(selected_run/'best_development.json')}
        state.update(status='validation',selection_before_validation=selection)
        refresh()
        validation={}
        # Independent final games never feed trial/architecture selection.
        for label,checkpoint,validated_trial in final_validation_plan(study,best):
            if not config.get('validation_episodes_per_policy',20):
                break
            check_stop()
            from fly_touhou.thrl_model import MOActorCritic
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model=MOActorCritic(**payload['model_config']).cuda()
            model.load_state_dict(payload['model'],strict=True)
            validation_config={**config,'parameters':payload.get('config',{}).get('parameters',config.get('ppo_baseline',{}))}
            del payload
            destination=output/f'validation20_{label}'
            state.update(status='validation',current_run=str(destination))
            refresh()
            def validation_progress(row):
                progress={'stage':'validation','status':'validation',
                    'actual_decisions':validated_trial.user_attrs['decisions'],
                    'target_decisions':config['trial_decisions'],'evaluation_progress':row,
                    'workers':1,'attention':config['attention'],'teacher_actions':0,
                    'parameter_counts':model.parameter_counts(),'updated_utc':utc_now()}
                write_json(destination/'progress.json',progress)
                refresh(progress)
            validation[label]=evaluate(torch,model,validation_config,destination,
                episodes=config['validation_episodes_per_policy'],seed=config['seed']+900000,
                check_stop=check_stop,progress=validation_progress,
                **({'native_split':'test'} if config.get('curriculum_manifest') or is_full_story(config) else {}))
            state['validation20']=validation
            del model
            torch.cuda.empty_cache()
        state.update(status='finished',current_run=None)
        refresh()
        write_json(output/'study_result.json',{'status':'finished','branch':config['branch'],
            'selection_before_validation':selection,'trials':state['trials'],
            'validation20':validation,'primary_comparison':'validation20_primary' if 'primary' in validation else None,
            'HPO_results_are_exploratory':True,'teacher_in_evaluation':False,'formal_acceptance':False})
        return 0
    except StopRequested as error:
        state.update(status='operator_or_budget_stop',reason=str(error))
        refresh()
        return 0
    except BaseException:
        state.update(status='failed')
        refresh()
        write_json(output/'failure.json',{'traceback':traceback.format_exc(),
            'current_run':state.get('current_run'),'branch':config['branch']})
        raise
    finally:
        api.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
