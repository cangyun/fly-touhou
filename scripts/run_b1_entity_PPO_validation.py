"""Finite post-training native comparisons; no scheduler and no new optimizer."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write


def alive(identity):
    import psutil
    try:return abs(psutil.Process(identity['pid']).create_time()-identity['create_time'])<.01
    except psutil.Error:return False


def dependency(plan):
    control=Path(plan['training_control']);output=Path(plan['training_output'])
    status=read(control/'status.json')
    if status.get('status') in ('failed','launcher-error'):
        return dict(phase='failed',reason='upstream background job failed')
    if status.get('status')!='finished':
        return dict(phase='waiting',upstream_live=alive(plan['training_identity']))
    if status.get('exit_code')!=0:return dict(phase='failed',reason='upstream exit code is not0')
    result=read(output/'trial_000_both/training_result.json')
    if result.get('status')!='finished' or result.get('candidate_optimized_decisions')!=plan['training_decisions']:
        return dict(phase='failed',reason='upstream did not finish declared optimized budget')
    clean=read(output/'trial_000_both/native_cleanup.json')
    ownership=clean.get('ownership',{})
    if clean.get('close_errors')!=[] or ownership.get('status')!='closed':
        return dict(phase='failed',reason='upstream native pool not cleanly closed')
    workers=ownership.get('workers',[])
    if isinstance(workers,dict):workers=list(workers.values())
    if len(workers)!=30 or any(w.get('worker_alive_after_close') is not False for w in workers):
        return dict(phase='failed',reason='not all30 training workers were closed')
    if alive(plan['training_identity']):return dict(phase='waiting',upstream_live=True)
    return dict(phase='completed',upstream_live=False)


def panel_root(plan):
    """Mutable UI state belongs to the original workspace, not frozen source."""
    if not plan.get('control_center_root'):
        raise ValueError('Frozen validation requires explicit existing control_center_root')
    root=Path(plan['control_center_root']).resolve()
    config=read(root/'configuration.json')
    if config.get('schema')!='thrl-control-center-v1':
        raise ValueError('Existing native control-panel configuration required')
    return root


def baseline_semantics(config,plan=None):
    if plan and plan.get('pipeline_ancestor'):
        return ('Pinned ancestral policy before complete teacher-trajectory BC and following shared-policy PPO;'
            'input='+config['input_schema']+';not the post-BC PPO initial policy;fresh native test cohort')
    return ('Same frozen initial policy;input='+config['input_schema']+
        ';zero new candidate PPO updates;original candidate initial retained across recovery;'
        'initial native baseline is measured separately')


def evaluation_protocol(plan):
    """Declare inference and diagnostic horizon before the training result exists."""
    from fly_touhou.thrl_avoidance_probe import MAX_DIAGNOSTIC_FRAMES
    selection=plan.get('full_story_action_selection','sampled')
    cap=plan.get('avoidance_probe_max_frames',16384)
    if selection not in ('sampled','greedy'):raise ValueError('Explicit sampled or greedy full-story selection required')
    if type(cap) is not int or not 256<=cap<=MAX_DIAGNOSTIC_FRAMES:
        raise ValueError('Explicit supported diagnostic frame cap required')
    return dict(full_story_action_selection=selection,avoidance_probe_max_frames=cap,
        training_episode_rule_changed=False,training_Bomb_rule_changed=False)


def full_story_configuration(config,plan):
    protocol=evaluation_protocol(plan)
    return dict(config,full_story_evaluation_backend='full-story-ready-parallel-v1',
        full_story_evaluation_workers=plan['workers'],
        full_story_evaluation_action_selection=protocol['full_story_action_selection'])


def candidate_initial_reference(trial,config,plan=None):
    """A recovery's local initial.pt already contains candidate PPO updates."""
    if plan and plan.get('pipeline_ancestor'):
        reference=plan['pipeline_ancestor'];provenance=config.get('full_teacher_actor_transfer')
        if not provenance or reference.get('role')!='before-full-natural-teacher-BC-and-PPO':
            raise ValueError('Whole-pipeline baseline requires explicit real actor-transfer ancestry')
        path=Path(reference['checkpoint']).resolve()
        if (path!=Path(provenance['parent_checkpoint']).resolve()
                or reference['sha256']!=provenance['parent_sha256']
                or hashlib.sha256(path.read_bytes()).hexdigest()!=reference['sha256']):
            raise ValueError('Whole-pipeline baseline differs from the actual pre-BC ancestor')
        return path
    if config.get('candidate_resume_applied') and not (
            config.get('candidate_initial_checkpoint') and config.get('candidate_initial_sha256')):
        raise ValueError('Recovery validation requires its pinned original candidate initial')
    path=Path(config.get('candidate_initial_checkpoint',trial/'initial.pt'))
    if not path.is_absolute():path=trial/path
    expected=config.get('candidate_initial_sha256')
    if expected and hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('Original candidate initial checkpoint identity differs')
    return path


def verify_probe_cohort(report,expected):
    """Every condition must actually execute the predeclared new test seeds."""
    if (report.get('native_seed_split')!='test' or len(expected)!=20 or len(set(expected))!=20):
        raise ValueError('An explicit20-seed native test cohort is required')
    names=('trained_sampled','initial_sampled','uniform18','trained_greedy','initial_greedy')
    rows=report['episode_results']
    if len(rows)!=100 or {r['policy'] for r in rows}!=set(names):
        raise ValueError('Five complete20-game conditions are required')
    for name in names:
        group=sorted((r for r in rows if r['policy']==name),key=lambda r:r['game'])
        if ([r['game'] for r in group]!=list(range(20))
                or [r['native_seed'] for r in group]!=expected
                or any(r.get('seed_applied') is not True or r['actual_bombs_used']!=0 for r in group)):
            raise ValueError('Original test seed provenance or movement-only condition differs: '+name)
    return True


def run(a):
    import psutil,shutil
    plan=read(a.plan);protocol=evaluation_protocol(plan);center=panel_root(plan)
    for flag in ('capture_failure_history_maps','capture_full_story_failure_traces'):
        if type(plan.get(flag,False)) is not bool:raise ValueError('Explicit diagnostic capture flags required')
    output=Path(plan['output']);output.mkdir(parents=True,exist_ok=False)
    if not 1<=plan['workers']<=30:raise ValueError('Validation must stay within the30-world budget')
    identity=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now())
    write(output/'launch.json',identity)
    cfg=read(Path(plan['training_control'])/'config.json');write(output/'study_config.json',cfg)
    phase='waiting_for_training_and_owned_cleanup'
    def stop():
        if (output/'STOP').exists():raise RuntimeError('Operator requested validation STOP')
    def publish(phase,current=None,detail=None):
        write(output/'study_status.json',dict(status='finished' if phase=='completed' else 'validation',
            phase=phase,current_run=str(current or output),updated_utc=utc_now(),
            current_progress=dict(status='validation',stage='validation',actual_decisions=0 if phase.startswith('waiting') else plan['training_decisions'],
                target_decisions=plan['training_decisions'],workers=plan['workers'],device='cuda:0',
                evaluation_progress=detail,optimizer_updates=0),native_instances_max=0 if phase.startswith('waiting') else plan['workers']))
    try:
        while True:
            stop();state=dependency(plan)
            if state['phase']=='failed':raise RuntimeError(state['reason'])
            if state['phase']=='completed':break
            if not state.get('upstream_live',False) and time.time()-identity['create_time']>30:
                raise RuntimeError('Pinned training process is absent before a successful terminal record; no restart performed')
            publish(phase,detail=state);time.sleep(10)
        import torch
        from fly_touhou.thrl_pair_policy import model_from_config
        from evaluate_thrl_avoidance_probe import run as probe
        from fly_touhou.thrl_failure_trace_audit import audit
        from train_thrl_native import evaluate
        from register_b1_goal_phase import register
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
        torch.set_num_threads(2)
        trial=Path(plan['training_output'])/'trial_000_both'
        training=read(trial/'training_result.json')
        sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        if sha(trial/'final.pt')!=training['final_sha256']:raise ValueError('Final checkpoint hash differs')
        config=read(trial/'config.json')
        original_initial=candidate_initial_reference(trial,config,plan)
        inputs=output/'evaluation_input';inputs.mkdir()
        shutil.copyfile(original_initial,inputs/'initial.pt');shutil.copyfile(trial/'final.pt',inputs/'latest.pt')
        config['initial_baseline_semantics']=baseline_semantics(config,plan)
        if plan.get('pipeline_ancestor'):
            ancestor_header=torch.load(original_initial,map_location='cpu',weights_only=False)
            config['initial_baseline_checkpoint_identity']=dict(scope='before-full-natural-teacher-BC-and-PPO',
                sha256=sha(original_initial),optimized_decisions=ancestor_header.get('candidate_optimized_decisions',0))
            del ancestor_header
        write(output/'initial_reference.json',dict(source=str(original_initial),sha256=sha(original_initial),
            recovered_candidate=bool(config.get('candidate_resume_applied')),local_recovery_initial_not_used=True,
            whole_teacher_BC_and_PPO_pipeline_compared=bool(plan.get('pipeline_ancestor')),
            post_BC_PPO_initial_used=bool(config.get('full_teacher_actor_transfer')) and not bool(plan.get('pipeline_ancestor')),
            semantics=config['initial_baseline_semantics']))
        write(inputs/'config.json',config)
        phase='no_Bomb_fresh100';destination=output/phase;publish(phase,destination)
        register(output,plan['panel_job_id'],'evaluation',control_root=center)
        args=SimpleNamespace(run=inputs,output=destination,episodes=20,workers=plan['workers'],seed=plan['seed'],
            native_split='test',device='cuda',max_frames=protocol['avoidance_probe_max_frames'],wall_seconds=7200.,
            include_greedy=True,include_bullet_ablation=False,capture_failures=True,
            capture_history_maps=plan.get('capture_failure_history_maps',False))
        if plan.get('recovery_source'):
            source=Path(plan['recovery_source'])
            if alive(read(source/'launch.json')):
                raise RuntimeError('Original frozen validation process is still alive; recovery refused')
            original=read(Path(plan['recovery_original_plan']))
            invariant=('training_control','training_output','training_identity','training_decisions','workers',
                'seed','expected_native_seeds','capture_failure_history_maps','capture_full_story_failure_traces')
            if any(plan.get(k)!=original.get(k) for k in invariant) or source!=Path(original['output']):
                raise ValueError('Recovery changed the original frozen comparison protocol')
            args.recovery_source=source/'no_Bomb_fresh100'
            args.expected_native_seeds=plan['expected_native_seeds']
        probe(args,model_factory=model_from_config)
        verify_probe_cohort(read(destination/'report.json'),plan['expected_native_seeds'])
        if read(destination/'cleanup.json')!=dict(closed=True,close_errors=[]):raise RuntimeError('No-Bomb pool cleanup incomplete')
        audit(destination/'report.json')
        cfg=full_story_configuration(config,plan)
        reports={}
        for label,checkpoint in [('initial',inputs/'initial.pt'),('final',inputs/'latest.pt')]:
            stop();phase='full_story_'+label;dest=output/phase;publish(phase,dest)
            if plan.get('capture_full_story_failure_traces',False):
                cfg['full_story_trace']=dict(checkpoint_sha256=sha(checkpoint),history=128)
            p=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model=model_from_config(**p['model_config']).cuda();model.load_state_dict(p['model'],strict=True);del p
            reports[label]=evaluate(torch,model,cfg,dest,episodes=20,seed=plan['seed'],native_split='test',
                check_stop=stop,progress=lambda row:publish(phase,dest,row))
            if reports[label].get('action_selection','sampled')!=protocol['full_story_action_selection']:
                raise ValueError('Executed full-game inference differs from the declared protocol')
            del model;torch.cuda.empty_cache()
        old={x['native_seed']:x for x in reports['initial']['episode_results']}
        new={x['native_seed']:x for x in reports['final']['episode_results']}
        if len(old)!=20 or old.keys()!=new.keys():raise ValueError('Full-game initial RNG cohorts differ')
        if old.keys()!=set(plan['expected_native_seeds']):raise ValueError('Full-game native seeds differ from the declared new test cohort')
        pairs=[]
        for seed,x in old.items():
            y=new[seed]
            if x['action_seed']!=y['action_seed']:raise ValueError('Full-game action seeds differ')
            pairs.append(dict(native_seed=seed,initial_frames=x['native_frames'],final_frames=y['native_frames'],
                initial_stage=x['stage'],final_stage=y['stage'],initial_clear=x['outcome']=='clear',final_clear=y['outcome']=='clear'))
        summary={k:{x:v[x] for x in ['episodes','clears','mean_native_frames','mean_max_stage','mean_misses','mean_bombs_used']} for k,v in reports.items()}
        write(output/'report.json',dict(status='completed',utc=utc_now(),schema='B1-entity-PPO-frozen-native-comparison-v1',
            no_Bomb_report=str(output/'no_Bomb_fresh100/report.json'),full_story=summary,paired_full_games=pairs,
            comparison_scope=('whole-full-teacher-BC-and-shared-policy-PPO-pipeline' if plan.get('pipeline_ancestor') else 'candidate-PPO-only'),
            initial_baseline_semantics=config['initial_baseline_semantics'],
            native_seed=plan['seed'],initial_RNG_matched=True,same_future_hazards_guaranteed=False,
            native_games=140,new_optimizer_updates=0,formal_six_stage_acceptance=False,
            evaluation_protocol=protocol,
            full_story_trace_capture={label:r.get('full_game_trace_capture') for label,r in reports.items()}
                if plan.get('capture_full_story_failure_traces',False) else None))
        phase='completed';publish(phase)
    except BaseException:
        write(output/'failure.json',dict(utc=utc_now(),phase=phase,traceback=traceback.format_exc(),no_automatic_restart=True))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',required=True,type=Path);run(p.parse_args())
