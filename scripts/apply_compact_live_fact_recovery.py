"""Restore archived native deathbomb facts during collection, without resetting games."""
from datetime import datetime,timezone
from pathlib import Path
import os
import sys


def park_collection_for_patch(output,expected_pid):
    """Temporarily park the owned main loop at its normal guard callback."""
    import threading
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid()!=expected_pid:raise ValueError('Expected owned trainer')
    main=None;rollout=None
    for frame in sys._current_frames().values():
        while frame is not None:
            if Path(frame.f_code.co_filename).name=='train_compact_story_ppo.py' and frame.f_code.co_name=='main':main=frame
            if Path(frame.f_code.co_filename).name=='compact_story_ready.py' and frame.f_code.co_name=='rollout':rollout=frame
            frame=frame.f_back
    if main is None or rollout is None:raise RuntimeError('Expected active ready-driven student collection')
    guard=main.f_locals['guard'];namespace=guard.__globals__
    previous_gate=namespace.get('_compact_patch_gate')
    if previous_gate is not None and not previous_gate.is_set():raise RuntimeError('A patch gate already exists')
    def factory(a,begun):
        def guarded():
            _compact_patch_gate.wait()
            if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
            if time.monotonic()-begun>a.wall_hours*3600:raise TimeoutError('Finite full-story training wall budget')
        return guarded
    fresh=factory(None,None)
    if fresh.__code__.co_freevars!=guard.__code__.co_freevars:raise RuntimeError('Guard closure changed')
    namespace['_compact_patch_gate']=threading.Event()
    namespace['_compact_original_guard_code']=guard.__code__
    guard.__code__=fresh.__code__
    write_json(Path(output),dict(status='parking_at_next_guard',PID=os.getpid(),cycle=main.f_locals['cycle'],
        sample_rows_preserved=True,optimizer_changed=False,native_games_reset=False))


def seal_partial_collection_for_patch(output,expected_pid):
    """Save current learned state; drain an aligned prefix without losing rows."""
    import ctypes,math
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid()!=expected_pid:raise ValueError('Expected owned trainer')
    main=None;rollout=None;guarded=False
    for frame in sys._current_frames().values():
        while frame is not None:
            if Path(frame.f_code.co_filename).name=='train_compact_story_ppo.py' and frame.f_code.co_name=='main':main=frame
            if Path(frame.f_code.co_filename).name=='compact_story_ready.py' and frame.f_code.co_name=='rollout':rollout=frame
            if frame.f_code.co_name=='guarded':guarded=True
            frame=frame.f_back
    if main is None or rollout is None or not guarded:raise RuntimeError('Wait until main is parked at its guard')
    local=rollout.f_locals
    old_total=int(local['total']);reserved=int(local['reserved']);length=int(local['length'])
    if local['fixed_quota']:
        import numpy as np
        # Keep every completed and already dispatched action. The one-off
        # prefix may have different lengths; pack_world_records supports this.
        limits=np.array([max(length,int(math.ceil((int(n)+int(i in local['pending']))/length))*length)
                         for i,n in enumerate(local['counts'])],dtype=np.int64)
        padding=(-int(limits.sum())//length)%(256//length)
        for _ in range(padding):limits[int(np.argmin(limits))]+=length
        target=int(limits.sum())
        if target>old_total:raise RuntimeError('Aligned prefix exceeds the original collection budget')
        local['limits'][:]=limits
        local['reserved']=target//length
        local['fixed_quota']=False
    else:target=min(old_total,int(math.ceil(reserved*length/256))*256)
    local['total']=target;local['blocks']=target//length
    sync=ctypes.pythonapi.PyFrame_LocalsToFast;sync.argtypes=[ctypes.py_object,ctypes.c_int];sync.restype=None
    sync(rollout,0)
    checkpoint=main.f_locals['save']('before-curve-repair-cycle%03d'%main.f_locals['cycle'])
    receipt=dict(status='aligned_prefix_will_be_sealed_then_learned',PID=os.getpid(),cycle=main.f_locals['cycle'],
        collected_rows=int(local['counts'].sum()),previous_target=old_total,aligned_target=target,
        current_pending_native_steps=len(local['pending']),parameter_checkpoint=str(checkpoint),
        optimizer_checkpoint_saved=True,discarded_rows=0,native_games_reset=False,brain_states_reset=False,
        note='Native positions and unsealed rows remain live; the normal collector saves the complete aligned tape after draining.')
    write_json(Path(output),receipt)


def release_patch_gate(output,expected_pid):
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid()!=expected_pid:raise ValueError('Expected owned trainer')
    import __main__
    namespace=vars(__main__);gate=namespace.get('_compact_patch_gate')
    if gate is None:raise RuntimeError('No active patch gate')
    original=namespace.get('_compact_original_guard_code')
    if original is not None:
        for frame in sys._current_frames().values():
            while frame is not None:
                if Path(frame.f_code.co_filename).name=='train_compact_story_ppo.py' and frame.f_code.co_name=='main':
                    frame.f_locals['guard'].__code__=original
                frame=frame.f_back
    gate.set()
    write_json(Path(output),dict(status='resumed_preserved_native_worlds',PID=os.getpid(),
        resumed_utc=datetime.now(timezone.utc).isoformat(),native_games_reset=False))


def stage_curve_worker(marker,output,expected_pid):
    """Load the new geometry code; switch only at the trainer's boundary."""
    import importlib
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid()!=expected_pid:raise ValueError('Expected owned native worker')
    for name in ('native_curve_laser_motion','native_curve_motion_paths','native_curve_laser_paths','compact_spiking_sensation','native_curve_laser_reader'):
        importlib.reload(importlib.import_module('fly_touhou.'+name))
    import fly_touhou.native_curve_laser_reader as reader
    original=reader.read_native_curve_lasers;boundary_marker=Path(marker)
    def controlled(*args,**kwargs):
        value=original(*args,**kwargs)
        if boundary_marker.exists():value=dict(value,motion_forecast_enabled=True)
        return value
    reader.read_native_curve_lasers=controlled
    write_json(Path(output),dict(status='curve_forecast_staged',PID=os.getpid(),
        current_enabled=boundary_marker.exists(),marker=str(boundary_marker),native_games_reset=False,
        source='Original curve trajectory records; unsupported native programs remain unknown'))


def stage_curve_trainer(output,expected_pid):
    """Keep this partial tape unchanged; enable new observations next cycle."""
    import ast,importlib
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_study_io import read_shared_json
    if os.getpid()!=expected_pid:raise ValueError('Expected owned trainer')
    main=None
    for frame in sys._current_frames().values():
        while frame is not None:
            if Path(frame.f_code.co_filename).name=='train_compact_story_ppo.py' and frame.f_code.co_name=='main':main=frame
            frame=frame.f_back
    if main is None:raise RuntimeError('Owned trainer main frame required')
    local=main.f_locals;model=local['model'];folder=local['a'].output
    gate=local['guard'].__globals__.get('_compact_patch_gate')
    if gate is None or gate.is_set():raise RuntimeError('Install while the preserved collector is parked')
    root=Path(main.f_code.co_filename).resolve().parents[1]
    import fly_touhou.compact_story_runtime_control as control_module
    import fly_touhou.compact_full_story as story
    replacements=[]
    for relative,owner,names in (
        ('src/fly_touhou/compact_story_model.py',type(model),('parameter_payload',)),
        ('src/fly_touhou/compact_story_runtime_control.py',control_module,('apply_control',)),
        ('src/fly_touhou/compact_full_story.py',story,('story_kwargs','evaluate')),
    ):
        path=root/relative;tree=ast.parse(path.read_text(encoding='utf8'))
        nodes=(next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='CompactStoryPolicy').body
               if isinstance(owner,type) else tree.body)
        for name in names:
            old=getattr(owner,name);node=next(x for x in nodes if isinstance(x,ast.FunctionDef) and x.name==name)
            if hasattr(old,'__wrapped__'):
                old=old.__wrapped__;node.decorator_list=[]
            namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),old.__globals__,namespace)
            fresh=namespace[name]
            if old.__code__.co_freevars!=fresh.__code__.co_freevars:raise ValueError('Unexpected sensor method closure change')
            replacements.append((old,fresh.__code__))
    # New geometry is explicitly tagged. Old teacher windows and this tape
    # keep their stored geometry and unknown-motion masks.
    for name in ('native_curve_laser_motion','native_curve_motion_paths','native_curve_laser_paths','compact_spiking_sensation'):
        importlib.reload(importlib.import_module('fly_touhou.'+name))
    # Active frames keep their old code; later rollouts report the new flag.
    path=root/'src/fly_touhou/compact_story_ready.py';tree=ast.parse(path.read_text(encoding='utf8'))
    node=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='ReadyCollector'
              for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='rollout')
    node.decorator_list=[]
    old=type(local['collector']).rollout.__wrapped__;namespace={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),old.__globals__,namespace)
    replacements.append((old,namespace['rollout'].__code__))
    # Future evaluation creates fresh environment instances with the new flag.
    # Reloading the module updates its class for future worker spawn; existing
    # model/evaluate references are updated by code replacement above.
    importlib.reload(story)
    for old,code in replacements:old.__code__=code
    model.native_curve_motion_forecast=False
    path=folder/'run-control.json';data=read_shared_json(path) if path.exists() else {}
    revision=max(local['control_revision'],data.get('revision',0))+1
    data.update(revision=revision,native_curve_motion_forecast=True,kl_early_stop=False,adaptive_policy_lr=False,target_kl=None)
    write_json(path,data)
    local['launch']['pending_native_curve_motion_forecast']=dict(first_cycle=local['cycle']+1,control_revision=revision)
    write_json(folder/'launch.json',local['launch'])
    write_json(Path(output),dict(status='staged_for_next_collection',PID=os.getpid(),cycle=local['cycle'],
        first_collection_cycle=local['cycle']+1,control_revision=revision,model_parameters_changed=False,
        prior_PPO_tape_changed=False,neural_topology_changed=False,native_games_reset=False))


def stage_damage_bound_worker(marker,output,expected_pid):
    """Opt in at the next learner boundary; inspect one current paused boss."""
    import importlib
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid()!=expected_pid:raise ValueError('Expected owned native worker')
    for name in ('native_enemy_damage','native_enemy_ecl','native_enemy_motion','compact_native_forecast'):
        importlib.reload(importlib.import_module('fly_touhou.'+name))
    import fly_touhou.native_enemy_motion as motion
    original=motion.read_enemy_motion;boundary_marker=Path(marker);probe_pending=True
    def controlled(*args,**kwargs):
        nonlocal probe_pending
        enabled=boundary_marker.exists()
        kwargs['forecast_damage_bound']=enabled
        value=original(*args,**kwargs)
        if probe_pending and not enabled and any(r['hp']>=0 and r['hp_sub'] for r in value['interrupts']):
            probe_pending=False
            # The game remains at its normal owned observation checkpoint.
            # Return the original state; this second read is diagnostic only.
            newer=original(*args,**dict(kwargs,forecast_damage_bound=True))
            previous=motion.predict_enemy_positions(value)
            current=motion.predict_enemy_positions(newer)
            write_json(Path(output).with_name(Path(output).stem+'-probe.json'),dict(
                PID=os.getpid(),source='same owned paused enemy;no native ticks or actions added',
                health=newer['health'],flags=newer['flags'],damage_bound=newer['damage_bound'],
                previous_unknown_frame=previous['unknown_from_frame'],
                previous_reason=previous['unknown_reason'],
                proposed_unknown_frame=current['unknown_from_frame'],
                proposed_reason=current['unknown_reason'],phase=current['phase_coverage'],
                current_training_observation_changed=False))
        return value
    motion.read_enemy_motion=controlled
    write_json(Path(output),dict(status='native_damage_bound_staged',PID=os.getpid(),
        current_enabled=boundary_marker.exists(),marker=str(boundary_marker),native_games_reset=False))


def stage_damage_bound_trainer(output,expected_pid):
    """Install a tested sensor change without truncating the active rollout."""
    import ast,importlib
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_study_io import read_shared_json
    if os.getpid()!=expected_pid:raise ValueError('Expected owned trainer')
    main=None;collecting=False
    for frame in sys._current_frames().values():
        while frame is not None:
            if Path(frame.f_code.co_filename).name=='train_compact_story_ppo.py' and frame.f_code.co_name=='main':main=frame
            if Path(frame.f_code.co_filename).name=='compact_story_ready.py' and frame.f_code.co_name=='rollout':collecting=True
            frame=frame.f_back
    if main is None or not collecting:raise RuntimeError('Stage during preserved student collection')
    local=main.f_locals;model=local['model'];folder=local['a'].output
    root=Path(main.f_code.co_filename).resolve().parents[1]
    import fly_touhou.compact_story_runtime_control as control_module
    import fly_touhou.compact_full_story as story
    replacements=[]
    for relative,owner,class_name,names in (
        ('src/fly_touhou/compact_story_model.py',type(model),'CompactStoryPolicy',('parameter_payload',)),
        ('src/fly_touhou/compact_story_runtime_control.py',control_module,None,('apply_control',)),
        ('src/fly_touhou/compact_full_story.py',story,None,('story_kwargs','evaluate')),
        ('src/fly_touhou/compact_story_ready.py',type(local['collector']),'ReadyCollector',('rollout',)),
    ):
        path=root/relative;tree=ast.parse(path.read_text(encoding='utf8'))
        nodes=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==class_name).body if class_name else tree.body
        for name in names:
            old=getattr(owner,name);node=next(x for x in nodes if isinstance(x,ast.FunctionDef) and x.name==name)
            if hasattr(old,'__wrapped__'):old=old.__wrapped__;node.decorator_list=[]
            namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),old.__globals__,namespace)
            fresh=namespace[name]
            if old.__code__.co_freevars!=fresh.__code__.co_freevars:raise ValueError('Unexpected source closure change')
            replacements.append((old,fresh.__code__))
    importlib.reload(importlib.import_module('fly_touhou.compact_native_forecast'))
    importlib.reload(story)
    for old,code in replacements:old.__code__=code
    model.native_enemy_damage_bound=False
    path=folder/'run-control.json';data=read_shared_json(path)
    revision=max(local['control_revision'],data['revision'])+1
    data.update(revision=revision,native_enemy_damage_bound=True,
                kl_early_stop=False,adaptive_policy_lr=False,target_kl=None)
    write_json(path,data)
    local['launch']['pending_native_enemy_damage_bound']=dict(first_cycle=local['cycle']+1,control_revision=revision)
    write_json(folder/'launch.json',local['launch'])
    write_json(Path(output),dict(status='staged_for_next_collection',PID=os.getpid(),cycle=local['cycle'],
        first_collection_cycle=local['cycle']+1,control_revision=revision,
        active_collection_target_unchanged=True,discarded_rows=0,model_parameters_changed=False,
        prior_PPO_tape_changed=False,neural_topology_changed=False,native_games_reset=False))


def enable_live_contact_policy_log(output,expected_pid):
    """Enrich only human-readable contact logs, including the active rollout."""
    import importlib
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid()!=expected_pid:raise ValueError('Expected owned trainer')
    import fly_touhou.compact_story_contact_evidence as evidence
    import fly_touhou.compact_story_ready as ready
    importlib.reload(evidence)
    original=ready.append_jsonl
    if getattr(original,'_includes_contact_policy_context',False):
        raise RuntimeError('Contact policy logging already enabled')
    def append_with_policy(path,value):
        if Path(path).name=='native-contact-evidence.jsonl' and 'policy_context' not in value:
            local=sys._getframe(1).f_locals
            value=dict(value,policy_context=evidence.recorded_policy_context(
                local['records'][local['i']],local['record'],local['before'],
                risk_scale=getattr(local['model'],'neural_risk_scale',1.)))
        return original(path,value)
    append_with_policy._includes_contact_policy_context=True
    ready.append_jsonl=append_with_policy
    write_json(Path(output),dict(status='active_for_future_contact_records',PID=os.getpid(),
        additional_model_calls=0,additional_native_steps=0,additional_teacher_queries=0,
        inputs_changed=False,losses_changed=False,optimizer_changed=False,games_reset=False,
        history_decisions=32,records_from_current_rollout=True))


def install_wrap_interior_forecast(output,expected_pid):
    """Install the measured wrap-free prefix while the native worlds are idle."""
    import ast,importlib
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid()!=expected_pid:raise ValueError('Expected owned process')
    modules={name:importlib.import_module('fly_touhou.'+name) for name in
        ('native_bullet_release','hazard_reader','compact_story_contact_evidence','native_bullet_turn')}
    targets=[('native_bullet_release','predict_bullet_release',None),
        ('native_bullet_release','_polar_ex8_hint',None),
        ('native_bullet_turn','read_turn_metadata',None),
        ('native_bullet_turn','predict_turn_prefix',None),
        ('hazard_reader','_bullet_metadata','VersionedHazardReader'),
        ('compact_story_contact_evidence','_points',None)]
    installed=[]
    for module_name,name,class_name in targets:
        module=modules[module_name];path=Path(module.__file__)
        body=ast.parse(path.read_text(encoding='utf-8-sig')).body
        if class_name:
            body=next(n for n in body if isinstance(n,ast.ClassDef) and n.name==class_name).body
        node=next(n for n in body if isinstance(n,ast.FunctionDef) and n.name==name)
        old=getattr(getattr(module,class_name) if class_name else module,name)
        namespace={}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),old.__globals__,namespace)
        fresh=namespace[name]
        if old.__code__.co_freevars!=fresh.__code__.co_freevars:raise ValueError('Unexpected forecast closure')
        old.__code__=fresh.__code__;installed.append(module_name+'.'+name)
    write_json(Path(output),dict(status='installed',PID=os.getpid(),functions=installed,
        forecast='EX0x1000 interior prefix only;edge-crossing remains unknown',
        existing_known_forecast_semantics_changed=False,model_parameters_changed=False,
        native_games_reset=False,extra_native_steps=0))


def install_polar_completion_forecast(output,expected_pid):
    """Cover the measured final EX8 phase without changing its nominal path."""
    import ast
    import fly_touhou.native_bullet_release as module
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid()!=expected_pid:raise ValueError('Expected owned process')
    path=Path(module.__file__)
    node=next(n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body
        if isinstance(n,ast.FunctionDef) and n.name=='_polar_ex8_hint')
    old=module._polar_ex8_hint;namespace={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),old.__globals__,namespace)
    old.__code__=namespace[node.name].__code__
    write_json(Path(output),dict(status='installed',PID=os.getpid(),
        forecast='unit-rate mature EX8 phase completion with no pending command,then linear motion',
        nominal_trajectory_changed=False,float32_bound_changed=False,
        existing_known_prefix_changed=False,model_parameters_changed=False,
        native_games_reset=False,extra_native_steps=0,
        completed_rollout_labels_changed=False,pending_commands_remain_unknown=True))


def install_wait_cancel_forecast(output,expected_pid):
    """Install final cancellation collision schedules on parked training workers."""
    import ast,importlib
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid()!=expected_pid:raise ValueError('Expected owned process')
    targets=[('native_bullet_release','predict_bullet_release',None),
        ('native_bullet_release','bullet_release_clearances',None),
        ('compact_partial_risk','partial_point_danger',None),
        ('hazard_reader','_bullet_metadata','VersionedHazardReader'),
        ('compact_story_contact_evidence','_points',None)]
    installed=[]
    for module_name,name,class_name in targets:
        module=importlib.import_module('fly_touhou.'+module_name);path=Path(module.__file__)
        body=ast.parse(path.read_text(encoding='utf-8-sig')).body
        if class_name:body=next(n for n in body if isinstance(n,ast.ClassDef) and n.name==class_name).body
        node=next(n for n in body if isinstance(n,ast.FunctionDef) and n.name==name)
        old=getattr(getattr(module,class_name) if class_name else module,name);namespace={}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),old.__globals__,namespace)
        if old.__code__.co_freevars!=namespace[name].__code__.co_freevars:raise ValueError('Unexpected observer closure')
        old.__code__=namespace[name].__code__;installed.append(module_name+'.'+name)
    write_json(Path(output),dict(status='installed',PID=os.getpid(),functions=installed,
        forecast='Unit-rate WAIT then final EX400;retains cancellation-tick collision,excludes later fading',
        model_parameters_changed=False,native_games_reset=False,extra_native_steps=0,
        completed_rollout_labels_changed=False,nonterminal_commands_remain_unknown=True))


def install_pending_zero_speed_forecast(output,expected_pid):
    """Install the new sensor/forecast at a sealed collection boundary."""
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    install_wrap_interior_forecast(output,expected_pid)
    receipt=read_shared_json(Path(output))
    receipt.pop('existing_known_forecast_semantics_changed',None)
    receipt.update(forecast='zero-speed circle HOLD/WAIT prefix and timed unit-rate EX8 motion',
        previously_covered_nominal_positions_changed=False,added_float32_roundoff_bound=True,
        unknown_commands_remain_unknown=True,teacher_targets_from_completed_rollouts_relabelled=False)
    write_json(Path(output),receipt)


def install_ex16_forecast(output,expected_pid):
    """Add EX16 timer coverage without replacing any live worlds or weights."""
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    install_wrap_interior_forecast(output,expected_pid)
    receipt=read_shared_json(Path(output))
    receipt.pop('existing_known_forecast_semantics_changed',None)
    receipt.update(forecast='native EX16 unit-time deceleration before its next turn',
        non_EX16_motion_changed=False,old_missing_metadata_remains_unknown=True,
        observed_non_unit_EX16_time_remains_unknown=True,
        completed_samples_relabelled=False,model_parameters_changed=False,
        source='TH14 4168ec dispatch,419cb5 velocity,419cf9 timer increment',
        runtime_endpoint_observer='ex16-motion-shadow-worker10.json; inspect its actual coverage separately')
    write_json(Path(output),receipt)


def install_polar_growth_forecast(output,expected_pid):
    """Cover observed state2 EX8 release without changing a parked game."""
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    install_wrap_interior_forecast(output,expected_pid)
    receipt=read_shared_json(Path(output))
    receipt.pop('existing_known_forecast_semantics_changed',None)
    receipt.update(forecast='state2 half-step and animation-release envelope into a unit-time EX8 phase',
        non_state2_EX8_motion_changed=False,unknown_release_time_not_guessed=True,
        native_pre_release_collision_retained=True,pending_commands_not_crossed=True,
        EX16_turn_parameters_recorded=True,
        EX16_turn_forecast='last deterministic turn without pending EX; other turns remain unknown',
        completed_samples_relabelled=False,model_parameters_changed=False,
        source='TH14 4167e4 half movement,41685c collision,41687c promotion,419970 polar phase',
        endpoint_evidence='8 same-ID four-frame original records:2 promoted,6 still growing')
    write_json(Path(output),receipt)


def install_final_turn_forecast(output,expected_pid):
    """Install the measured last-turn branch in parked training workers."""
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    install_wrap_interior_forecast(output,expected_pid)
    receipt=read_shared_json(Path(output))
    receipt.pop('existing_known_forecast_semantics_changed',None)
    receipt.update(forecast='EX16 last deterministic turn then linear motion, with no pending EX',
        supported_turn_modes=[0,4,5],measured_turn_modes=[0],
        aimed_random_repeating_turns_remain_unknown=True,
        pre_turn_positions_and_bounds_unchanged=True,completed_samples_relabelled=False,
        model_parameters_changed=False,native_games_reset=False,
        source='419b2e relative,419bf9 absolute,419c0c speed,419c83..419cab final velocity and flag clear',
        endpoint_evidence='12 original same-ID intervals across final relative turns',
        current_frozen087_workers_untouched=True)
    write_json(Path(output),receipt)


def probe_pending_motion_snapshot(output,expected_pid):
    """Compare coverage on one owned paused snapshot without advancing TH14."""
    import gc
    from fly_touhou.hazard_reader import VersionedHazardReader
    from fly_touhou.native_bullet_release import predict_bullet_release
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid()!=expected_pid:raise ValueError('Expected owned native worker')
    reader=next(o for o in gc.get_objects() if isinstance(o,VersionedHazardReader))
    snapshot=reader.read();rows=snapshot.extra['hazard_metadata']['bullets']
    before={str(h):0 for h in (4,12,24)};after=dict(before);modes={};samples=[]
    for raw in rows:
        old=dict(raw)
        for field in ('native_ex_program_prefix_hex','native_motion_time_rate','native_ex8_time_rate'):
            old.pop(field,None)
        a=predict_bullet_release(old,24);b=predict_bullet_release(raw,24)
        for h in (4,12,24):
            before[str(h)]+=int(a['known_by_frame'][h]);after[str(h)]+=int(b['known_by_frame'][h])
        modes[b['mode']]=modes.get(b['mode'],0)+1
        if a['known_by_frame']!=b['known_by_frame'] and len(samples)<4:
            samples.append(dict(raw=raw,old_known=a['known_by_frame'],new_known=b['known_by_frame'],mode=b['mode']))
    write_json(Path(output),dict(PID=os.getpid(),native_frame=snapshot.frame,points=len(rows),
        old_covered_points=before,new_covered_points=after,new_modes=modes,samples=samples,
        coherent_snapshot=reader.global_frame()==snapshot.frame,extra_native_steps=0))


def audit_live_motion_prefix(output,expected_pid,pair_limit=192,start_stage=3):
    """Check natural training motion at 4/8/12/24 frames, then remove itself."""
    import math
    import fly_touhou.hazard_reader as hazard
    from fly_touhou.native_bullet_release import predict_bullet_release
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid()!=expected_pid:raise ValueError('Expected owned native worker')
    original=hazard.VersionedHazardReader.read;anchors={};calls=0;last_frame=None
    counts={};failures=[];pairs=0;extended_pairs=0;max_excess=0.;last_written=0
    modes={'native_pending_zero_speed_motion_envelope','native_ex8_polar_timed_prefix'}
    def save(status,error=None):
        try:
            write_json(Path(output),dict(status=status,PID=os.getpid(),pairs=pairs,extended_prefix_pairs=extended_pairs,
                start_stage=start_stage,by_mode_and_dt=counts,
                outside_envelope_count=len(failures),failures=failures,max_excess_px=max_excess,
                scope='motion endpoints on student trajectories; not collision or counterfactual proof',
                calls=calls,reader_restored=status!='armed' and status!='recording',error=error,extra_native_steps=0))
        except Exception as error:
            sys.stderr.write('Motion endpoint audit write failed: '+str(error)+'\n')
    def read(self,*args,**kwargs):
        nonlocal calls,last_frame,pairs,extended_pairs,max_excess,last_written
        snapshot=original(self,*args,**kwargs)
        try:
            if snapshot.stage<start_stage or snapshot.frame==last_frame:return snapshot
            last_frame=snapshot.frame
            raw_by_id={r['id']:r for r in snapshot.extra['hazard_metadata']['bullets']}
            eligible=[]
            for raw in raw_by_id.values():
                flags=raw.get('native_ex_active_flags')
                if flags not in (8,0x04000000,0x80000000):continue
                legacy_prefix=(4 if flags==8 else raw.get(
                    'native_ex_hold_remaining' if flags==0x04000000 else 'native_ex_wait_remaining',24))
                if legacy_prefix>=24:continue
                prediction=predict_bullet_release(raw,24)
                if prediction['mode'] in modes and sum(prediction['known_by_frame'])>legacy_prefix+1:
                    eligible.append((raw,prediction,legacy_prefix))
                if len(eligible)>=8:break
            # Wait for the new forecast region to occur naturally, rather
            # than exhausting the audit on stage-opening ordinary bullets.
            if not calls and not eligible:return snapshot
            calls+=1
            for identity,anchor in list(anchors.items()):
                elapsed=snapshot.frame-anchor['frame'];raw=raw_by_id.get(identity)
                old=anchor['raw'];prediction=anchor['prediction']
                if (elapsed in (4,8,12,24) and raw is not None and prediction['known_by_frame'][elapsed]
                        and raw['native_timer']==old['native_timer']+elapsed
                        and raw['native_hitbox_value']==old['native_hitbox_value'] and raw['native_scale']==old['native_scale']):
                    p=raw['native_position'];a,b=prediction['bounds'][elapsed]
                    vx,vy=b[0]-a[0],b[1]-a[1];den=vx*vx+vy*vy
                    along=min(1.,max(0.,((p[0]-a[0])*vx+(p[1]-a[1])*vy)/den)) if den else 0.
                    distance=math.hypot(p[0]-a[0]-along*vx,p[1]-a[1]-along*vy)
                    radius=prediction.get('extra_radii',[0.]*25)[elapsed]
                    excess=max(0.,distance-radius);max_excess=max(max_excess,excess)
                    key=prediction['mode']+':'+str(elapsed);counts[key]=counts.get(key,0)+1;pairs+=1
                    extended_pairs+=int(elapsed>anchor['legacy_prefix'])
                    if excess>1e-3:
                        failures.append(dict(id=identity,frame=snapshot.frame,dt=elapsed,excess_px=excess,
                            old_raw=old,current_raw=raw,predicted_bounds=[a,b],radius=radius))
                if elapsed>=24 or elapsed<0 or raw is None:anchors.pop(identity,None)
            if pairs>=pair_limit or calls>=256 or failures:
                hazard.VersionedHazardReader.read=original
                save('completed' if not failures else 'motion_mismatch')
                return snapshot
            for raw,prediction,legacy_prefix in eligible:
                if len(anchors)>=8:break
                if raw['id'] in anchors:continue
                anchors[raw['id']]=dict(frame=snapshot.frame,raw=raw,prediction=prediction,legacy_prefix=legacy_prefix)
            if pairs-last_written>=32:
                save('recording');last_written=pairs
        except Exception as error:
            hazard.VersionedHazardReader.read=original;save('diagnostic_failed',str(error))
        return snapshot
    hazard.VersionedHazardReader.read=read;save('armed')


def capture_pending_hold_programs(output,expected_pid):
    """One-shot read of two existing EX programs; do not alter sensor output."""
    import fly_touhou.hazard_reader as hazard
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid()!=expected_pid:raise ValueError('Expected owned worker')
    original=hazard.VersionedHazardReader._bullet_metadata;samples={}
    def read(self,bullet):
        result=original(self,bullet);flag=result.get('native_ex_active_flags')
        if flag in (0x04000000,0x80000000) and str(flag) not in samples:
            first=self.global_frame()
            program=self.mem.read(bullet.id+0xc28,18*0x2c)
            last=self.global_frame()
            if first==last:
                samples[str(flag)]=dict(frame=first,raw=dict(result),remaining_program_hex=program.hex(),
                    program_start_index=0,program_records=18)
                finished=len(samples)==2
                if finished:hazard.VersionedHazardReader._bullet_metadata=original
                write_json(Path(output),dict(status='captured' if finished else 'partial_capture',
                    PID=os.getpid(),samples=samples,reader_restored=finished,
                    policy_observations_changed=False,extra_native_steps=0))
        return result
    hazard.VersionedHazardReader._bullet_metadata=read
    write_json(Path(output),dict(status='armed',PID=os.getpid(),samples={},
        policy_observations_changed=False,extra_native_steps=0))


def apply(archive,output,expected_pid):
    from fly_touhou.compact_story_tape_retention import load_replays,merge_native_deathbombs
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid()!=expected_pid:raise ValueError('Expected the owned trainer process')
    main=None;collecting=False;sealing=False
    for frame in sys._current_frames().values():
        while frame is not None:
            name=Path(frame.f_code.co_filename).name
            if name=='compact_story_ready.py' and frame.f_code.co_name=='rollout':collecting=True
            if name=='compact_story_storage.py' and frame.f_code.co_name=='save_rollout':sealing=True
            if name=='train_compact_story_ppo.py' and frame.f_code.co_name=='main':main=frame
            frame=frame.f_back
    if main is None or not (collecting or sealing):raise RuntimeError('Apply during collection or tape sealing, before learning')
    bank=main.f_locals['critical_replay']
    restored=load_replays(archive,bank.native_repair_revisions)
    before=bank.summary();change=merge_native_deathbombs(bank,restored)
    receipt=dict(status='applied',applied_utc=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),
        cycle=main.f_locals['cycle'],PPO_updates=main.f_locals['updates'],source=str(archive),
        before=before,after=bank.summary(),change=change,
        geometry_revision_migration=restored.get('geometry_revision_migration'),
        model_parameters_changed=False,live_brain_states_changed=False,native_games_reset=False)
    write_json(Path(output),receipt)
    write_json(main.f_locals['a'].output/'critical-teacher-replay.json',bank.summary())


def set_future_bomb_temperature(temperature,output,expected_pid):
    return _set_future_sampling_temperature('bomb',temperature,output,expected_pid)


def set_future_movement_temperature(temperature,output,expected_pid):
    return _set_future_sampling_temperature('movement',temperature,output,expected_pid)


def _set_future_sampling_temperature(family,temperature,output,expected_pid):
    """Change only the next rollout; the sealed PPO tape keeps its own temperature."""
    from fly_touhou.compact_story_behavior import sampling_temperature
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid()!=expected_pid:raise ValueError('Expected the owned trainer process')
    if family not in ('bomb','movement'):raise ValueError('Known sampling family required')
    field=family+'_sampling_temperature'
    value=sampling_temperature(temperature);main=None
    for frame in sys._current_frames().values():
        while frame is not None:
            name=Path(frame.f_code.co_filename).name
            if name=='compact_story_ready.py' and frame.f_code.co_name=='rollout':
                raise RuntimeError('Do not change the behavior during collection')
            if name=='train_compact_story_ppo.py' and frame.f_code.co_name=='main':main=frame
            frame=frame.f_back
    if main is None:raise RuntimeError('Owned training main frame required')
    local=main.f_locals;collector=local['collector'];save=local['save']
    cells=dict(zip(save.__code__.co_freevars,save.__closure__))
    # This is the actual closure cell used by main and save(), not a write to
    # the frame's locals copy. Native games and the recurrent state are retained.
    cell=cells[family+'_temperature'];before=getattr(collector,field)
    if cell.cell_contents!=before:raise ValueError('Checkpoint and live behavior settings differ')
    setattr(collector,field,value);cell.cell_contents=value
    change=dict(status='applied_for_next_collection',PID=os.getpid(),
        applied_utc=datetime.now(timezone.utc).isoformat(),cycle=local['cycle'],
        first_collection_cycle=local['cycle']+1,family=family,before=before,after=value,
        sealed_rollout_temperature=local['rollout']['metadata'][field],
        saved_checkpoint_temperature=cell.cell_contents,model_parameters_changed=False,
        frozen_decoder_changed=False,prior_PPO_behavior_changed=False,native_games_reset=False)
    local['launch'][field]=value
    local['launch']['live_'+family+'_exploration_change']=change
    write_json(local['a'].output/'launch.json',local['launch'])
    write_json(Path(output),change)


def stage_protection_expiry_fix(output,expected_pid):
    """Install backward-compatible code, activate only before the next rollout."""
    import ast
    import importlib
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_study_io import read_shared_json
    import fly_touhou.compact_story_risk_composition as composition
    import fly_touhou.compact_story_runtime_control as controls
    if os.getpid()!=expected_pid:raise ValueError('Expected the owned trainer process')
    main=None
    for frame in sys._current_frames().values():
        while frame is not None:
            if Path(frame.f_code.co_filename).name=='train_compact_story_ppo.py' and frame.f_code.co_name=='main':
                main=frame
            frame=frame.f_back
    if main is None:raise RuntimeError('Owned training main frame required')
    local=main.f_locals;model=local['model'];folder=local['a'].output
    if read_shared_json(folder/'progress.json')['phase']!='frozen_full_story_development':
        raise RuntimeError('Stage this repair during the unchanged frozen evaluation')
    if getattr(model,'protection_expiry_inclusive',False):raise ValueError('Boundary repair already active')
    if not (model.phase_valid_control and model.protection_aware_risk and model.soft_risk_composition):
        raise ValueError('Expected finite protection-aware native policy')
    root=Path(main.f_code.co_filename).resolve().parents[1]
    replacements=[]
    for relative,owner,names in (
        ('src/fly_touhou/compact_story_model.py',type(model),('heads','parameter_payload')),
        ('src/fly_touhou/compact_story_runtime_control.py',controls,('apply_control',)),
    ):
        path=root/relative;tree=ast.parse(path.read_text(encoding='utf8'))
        nodes=(next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='CompactStoryPolicy').body
               if isinstance(owner,type) else tree.body)
        for name in names:
            old=getattr(owner,name)
            node=next(x for x in nodes if isinstance(x,ast.FunctionDef) and x.name==name)
            namespace={}
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),old.__globals__,namespace)
            fresh=namespace[name]
            if old.__code__.co_freevars!=fresh.__code__.co_freevars:
                raise ValueError('Unexpected method closure change')
            replacements.append((old,fresh.__code__))
    importlib.reload(composition)
    for old,code in replacements:old.__code__=code
    # New code preserves the old output while this checkpoint's flag is false.
    # The existing loop consumes this command only after evaluation, before
    # collecting new student actions, so no PPO tape mixes two decoders.
    path=folder/'run-control.json'
    control=read_shared_json(path) if path.exists() else {}
    revision=max(local['control_revision'],control.get('revision',0))+1
    control.update(revision=revision,protection_expiry_inclusive=True,
        kl_early_stop=False,adaptive_policy_lr=False,target_kl=None)
    write_json(path,control)
    receipt=dict(status='staged_for_next_collection',PID=os.getpid(),
        staged_utc=datetime.now(timezone.utc).isoformat(),cycle=local['cycle'],
        first_collection_cycle=local['cycle']+1,control_revision=revision,
        current_frozen_mode=getattr(model,'protection_expiry_inclusive',False),
        pending_mode=True,model_parameters_changed=False,native_games_reset=False,
        neural_topology_changed=False,Bomb_probability_preserved=True,
        activation_receipt=str(folder/'runtime-control-applied.jsonl'))
    write_json(Path(output),receipt)


def stage_critical_replay_priority(output,expected_pid):
    """Change future supervised replay selection while no update is running."""
    import ast
    import fly_touhou.compact_story_ppo as learner
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid()!=expected_pid:raise ValueError('Expected the owned trainer process')
    main=None
    for frame in sys._current_frames().values():
        while frame is not None:
            if Path(frame.f_code.co_filename).name=='train_compact_story_ppo.py' and frame.f_code.co_name=='main':
                main=frame
            frame=frame.f_back
    if main is None:raise RuntimeError('Owned training main frame required')
    local=main.f_locals;bank=local['critical_replay'];folder=local['a'].output
    phase=read_shared_json(folder/'progress.json')['phase']
    if phase not in ('frozen_full_story_development','student_full_story_collection'):
        raise RuntimeError('Install during collection or frozen evaluation, before further learning')
    root=Path(main.f_code.co_filename).resolve().parents[1]
    replacements=[];additions=[]
    for relative,owner,names,globals_ in (
        ('src/fly_touhou/compact_story_critical_replay.py',type(bank),
         ('_normal_priority','_copy_block','loss','summary','sampling_probabilities','sample_index',
          'normal_sampling_probabilities','sample_normal_index'),bank.loss.__func__.__globals__),
        ('src/fly_touhou/compact_story_ppo.py',learner,('update',),learner.update.__globals__),
    ):
        path=root/relative;tree=ast.parse(path.read_text(encoding='utf8'))
        nodes=(next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='CriticalTeacherReplay').body
               if isinstance(owner,type) else tree.body)
        for name in names:
            node=next(x for x in nodes if isinstance(x,ast.FunctionDef) and x.name==name)
            namespace={}
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),globals_,namespace)
            fresh=namespace[name];old=getattr(owner,name,None)
            if old is None:additions.append((owner,name,fresh))
            else:
                if old.__code__.co_freevars!=fresh.__code__.co_freevars:
                    raise ValueError('Unexpected replay method closure change')
                replacements.append((old,fresh.__code__))
    for owner,name,fresh in additions:setattr(owner,name,fresh)
    for old,code in replacements:old.__code__=code
    receipt=dict(status='installed_for_next_learning',PID=os.getpid(),
        applied_utc=datetime.now(timezone.utc).isoformat(),cycle=local['cycle'],
        first_learning_cycle=local['cycle']+int(phase=='frozen_full_story_development'),
        selection=bank.summary()['critical_sampling'],normal_selection=bank.summary()['normal_sampling'],
        critical_windows=len(bank.sequences),normal_windows=len(bank.normal_sequences),
        model_parameters_changed=False,neural_topology_changed=False,native_games_reset=False,
        frozen_evaluation_changed=False,PPO_sampling_changed=False,loss_weights_changed=False,
        development_data_added_to_training=False,KL_controls_changed=False)
    write_json(folder/'critical-teacher-replay.json',bank.summary())
    write_json(Path(output),receipt)


def enable_curve_motion_telemetry(output,expected_pid,*,capture_dir=None,capture_limit=16):
    """Read trajectory programs into diagnostics; do not enable a predictor."""
    import importlib
    from fly_touhou.thrl_durable_records import write_json
    import fly_touhou.native_curve_laser_reader as reader
    if os.getpid()!=expected_pid:raise ValueError('Expected the owned trainer process')
    importlib.reload(reader)
    if capture_dir is not None:
        original=reader.read_native_curve_lasers
        destination=Path(capture_dir)
        destination.mkdir(parents=True,exist_ok=True)
        count=0;previous_frame=None
        def captured(*args,**kwargs):
            nonlocal count,previous_frame
            value=original(*args,**kwargs)
            frame=value.get('frame')
            if (count<capture_limit and frame!=previous_frame and
                    any(q.get('motion_records',{}).get('available') for q in value.get('curves',[]))):
                try:
                    write_json(destination/f'snapshot-{count:03d}.json',value)
                    count+=1;previous_frame=frame
                except Exception as error:
                    captured.last_error=repr(error)
            return value
        reader.read_native_curve_lasers=captured
    write_json(Path(output),dict(status='trajectory_record_capture_enabled',PID=os.getpid(),
        applied_utc=datetime.now(timezone.utc).isoformat(),forecast_enabled=False,
        policy_inputs_changed=False,teacher_labels_changed=False,model_parameters_changed=False,
        native_games_reset=False,capture_dir=capture_dir,capture_limit=capture_limit if capture_dir else 0,
        scope='Additional native curve trajectory records in existing snapshots'))
