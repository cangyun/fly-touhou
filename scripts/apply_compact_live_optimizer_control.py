"""Apply the authorized optimizer-only change to an already owned trainer.

Used once through the Python debugger's attach helper. It changes no tensors,
games, actions, Adam moments, or sampling state. Future runs use normal CLI
and run-control.json settings; no debugger server or tracing is installed.
"""
import ast
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import sys


def definition(module,name,kind):
    path=Path(module.__file__)
    nodes=ast.parse(path.read_text(encoding='utf-8'),filename=str(path)).body
    node=next(n for n in nodes if isinstance(n,kind) and n.name==name)
    namespace={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),vars(module),namespace)
    return namespace[name]


def replace_function(old,new,**defaults):
    if old.__closure__ or new.__closure__:raise ValueError('Only module-level functions supported')
    old.__code__=new.__code__
    old.__defaults__=new.__defaults__
    old.__kwdefaults__=dict(new.__kwdefaults__ or {},**defaults)


def apply(plan_path):
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    import fly_touhou.compact_story_ppo as ppo
    import fly_touhou.compact_story_learning_rate as rates
    import fly_touhou.compact_story_runtime_control as runtime
    plan=read_shared_json(Path(plan_path))
    output=Path(plan['output']).resolve()
    receipt=output/'live-optimizer-control.json'
    if os.getpid()!=plan['PID']:raise ValueError('Only the specified owned trainer may be changed')
    revision=plan.get('revision','no-kl-fixed-v1')
    if getattr(ppo,'_fixed_kl_live_revision',None)==revision:return
    frames=[]
    for frame in sys._current_frames().values():
        while frame is not None:
            frames.append(frame);frame=frame.f_back
    mains=[f for f in frames if f.f_code.co_name=='main'
        and Path(f.f_code.co_filename).resolve()==Path(plan['trainer_script']).resolve()]
    if len(mains)!=1:raise RuntimeError('Expected one active trainer main frame')
    local=mains[0].f_locals
    if Path(local['a'].output).resolve()!=output:raise ValueError('Trainer output does not match')
    if any(f.f_code.co_name=='after_rollout' and Path(f.f_code.co_filename).resolve()==Path(rates.__file__).resolve()
           for f in frames):raise RuntimeError('LR transition in progress;retry at the next boundary')
    controller=local['learning_rate']
    old_update=local['update'];old_control=local['apply_control']
    if old_update is not ppo.update or old_control is not runtime.apply_control:
        raise RuntimeError('Trainer callables differ from the expected loaded modules')
    new_update=definition(ppo,'update',ast.FunctionDef)
    new_control=definition(runtime,'apply_control',ast.FunctionDef)
    new_rate_type=definition(rates,'PolicyLearningRate',ast.ClassDef)
    active_updates=[f for f in frames if f.f_code.co_name=='update'
        and Path(f.f_code.co_filename).resolve()==Path(ppo.__file__).resolve()]
    before={g['name']:float(g['lr']) for g in controller.optimizer.param_groups}
    # A currently executing legacy frame still owns its previous code object.
    # Infinity disables that frame's old numeric comparison. New calls use the
    # explicit None branch in the tested source; no infinite value enters loss
    # arithmetic, parameter updates, serialized settings, or learning-rate logs.
    for frame in active_updates:
        cfg=frame.f_locals.get('cfg')
        if cfg is not None:object.__setattr__(cfg,'target_kl',float('inf'))
    replace_function(old_update,new_update,target_kl=None)
    replace_function(old_control,new_control)
    rates.PolicyLearningRate=new_rate_type
    controller.__class__=new_rate_type
    controller.configure_kl(adaptive=False,kl_early_stop=False,target_kl=None)
    args=local['a'];args.adaptive_policy_lr=False;args.kl_early_stop=False;args.target_kl=None
    launch=local['launch']
    change=dict(revision=revision,applied_at_utc=datetime.now(timezone.utc).isoformat(),
        PID=os.getpid(),cycle=local.get('cycle'),PPO_updates=local.get('updates'),
        active_update_frames=len(active_updates),kl_early_stop=False,target_kl=None,
        adaptive_policy_lr=False,fixed_learning_rates=before,
        existing_update_callable_default_target_kl=old_update.__kwdefaults__['target_kl'],
        brain_or_game_reset=False,model_tensors_modified=False,Adam_moments_reset=False,
        loss_clipping_unchanged=True,gradient_clipping_unchanged=True,
        debugger_server_started=False,debugger_tracing_enabled=False,
        source_update='Only update,PolicyLearningRate,apply_control definitions;trainer main loop retained')
    launch.update(kl_early_stop=False,target_kl=None,adaptive_policy_lr=False,
        runtime_optimizer_control=change)
    ppo._fixed_kl_live_revision=change['revision']
    write_json(output/'launch.json',launch)
    write_json(receipt,dict(status='applied',**change))
    write_json(output/('live-optimizer-control-'+revision+'.json'),dict(status='applied',**change))
    snapshots=output/'runtime-code'/revision;snapshots.mkdir(parents=True,exist_ok=True)
    for path in (Path(ppo.__file__),Path(rates.__file__),Path(runtime.__file__),Path(__file__)):
        shutil.copy2(path,snapshots/path.name)

