"""Fair uncapped original TH14 NoBomb/first-contact evaluation for deep and D95.

Default NoTorch plan. GPU/native execution belongs to an explicit root dispatch.
The same native environment,seeds,actions and endpoint definition serve both.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
SCHEMA='continuous-and-D95-NoBomb-first-contact-entry-v1'
DEEP_SCHEMA='continuous-full-connectome-DN-actor-threevalue-normalized-risk128x3-parameters-v1'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=('deep','d95'),required=True)
    for name in ('checkpoint','environment-config','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--connectome',type=Path,default=ROOT/'data/connectome')
    p.add_argument('--episodes',type=int,default=20);p.add_argument('--workers',type=int,default=20)
    p.add_argument('--seed',type=int,default=2027431601)
    p.add_argument('--max-seconds',type=int,required=True)
    p.add_argument('--execute-cuda',action='store_true')
    a=p.parse_args(argv)
    if (not 1<=a.episodes<=20 or not 1<=a.workers<=min(20,a.episodes) or not 0<=a.seed<2**31
            or a.max_seconds<1 or len(a.checkpoint_sha256)!=64
            or any(c not in '0123456789abcdef' for c in a.checkpoint_sha256)):
        p.error('Explicit artifact SHA,finite wall budget,cohort<=20 and common controlled seed required')
    return a


def plan(a):
    cfg=json.loads(a.environment_config.read_text('utf-8'))
    if (cfg.get('input_schema')!='semantic-separated-observed-laser-segments-v5'
            or cfg.get('training_mode')!='full-story-v1' or cfg.get('reward_schema')!='th14-completion-reward-v3'
            or cfg.get('headless') is not True or cfg.get('continues_allowed') is not False
            or cfg.get('policy_action_count')!=19):raise ValueError('Existing same original LunaticV5/source19/private full-story environment required')
    return dict(schema=SCHEMA,status='prepared_not_executed',family=a.family,checkpoint=str(a.checkpoint),
        checkpoint_sha256=a.checkpoint_sha256,output=str(a.output),environment_config=str(a.environment_config),
        environment_config_sha256=sha(a.environment_config),Torch_imported='torch' in sys.modules,
        checkpoint_deserialized=False,episodes=a.episodes,workers=a.workers,ready_neural_width_max=16,
        master_seed=a.seed,native_seed_split='development',seed_protocol='SeedSequence([master_seed,zero_based_game]) then existing draw_native_seed',
        difficulty=3,character='original ReimuA character0/subshot0',frame_skip=4,
        fast_speed=4.5,focused_speed=2.,auto_shot=True,full_original_opening=True,
        Bomb='evaluation mask source19 action18 to-inf;movement0..17 unchanged',
        action_selection='greedy',action_temperature=1.,native_frame_cap=None,first_hit_cap=None,
        endpoint='first native accepted state4 or Miss increase,without waiting for life decrement;natural clear right-censored',
        endpoint_time='acknowledged native decision endpoint,not exact native contact tick',
        natural_clear_right_censored=True,external_wall_stop_is_not_clear_or_score=True,
        minimum_policy_decisions=None,native_continue=False,
        source_parameter_subtype=DEEP_SCHEMA if a.family=='deep' else 'complete D95 EntityRiskPolicy baseGRU+point+laser beta8',
        traces='last8actual observations/actions/costs;deep same-forward DN/VP from actual worker.state,not cursor',
        teacher_takeovers=0,optimizer_updates=0,extra_neural_probe_forwards=0,
        maximum_wall_seconds=a.max_seconds,RAM_or_VRAM_caps=None,CUDA_started=False,native_started=False)


def execute(a,prepared):
    if 'torch' in sys.modules:raise RuntimeError('Fresh executor required;CVD before Torch')
    if a.output.exists():raise FileExistsError('Fresh output required;no auto retry')
    if sha(a.checkpoint)!=a.checkpoint_sha256:raise ValueError('Explicit frozen checkpoint identity differs')
    cfg=json.loads(a.environment_config.read_text('utf-8'))
    import psutil
    current=psutil.Process();native=Path(cfg['game_exe']).name.lower()
    if any((p.info['name'] or '').lower()==native for p in psutil.process_iter(['name'])):
        raise RuntimeError('Existing native worlds;do not overlap or take over another job')
    from fly_touhou.owned_worker_affinity import worker_cpu_pool
    worker_cpus=worker_cpu_pool(current.cpu_affinity())
    current.cpu_affinity(current.cpu_affinity()[-2:]);current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-continuous-'+str(os.getpid()))
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp
    a.output.mkdir(parents=True);configure_native_temp(a.output);begun=time.monotonic()
    launch=dict(prepared,status='running',PID=os.getpid(),create_time=current.create_time(),
        started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),entry_sha256=sha(__file__),
        helper_sha256=sha(ROOT/'src/fly_touhou/continuous_no_bomb_first_contact.py'),
        game_exe=cfg['game_exe'],game_exe_sha256=sha(cfg['game_exe']),profile=cfg['profile'],profile_sha256=sha(cfg['profile']))
    write_json(a.output/'launch.json',launch)
    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('Operator STOP;unfinished games are not scores')
        if time.monotonic()-begun>a.max_seconds:raise RuntimeError('Finite wall budget;unfinished games are not natural-clear censoring')
    try:
        import torch
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(False)
        if a.family=='deep':
            from fly_touhou.thrl_dodge_diagnostics import _shared_binary
            with _shared_binary(a.checkpoint) as stream:saved=torch.load(stream,map_location='cpu',weights_only=True)
            payload=saved.get('model_payload',saved)
            if payload.get('schema')!=DEEP_SCHEMA:raise ValueError('Explicit full deep128x3 parameter subtype required')
            from fly_touhou.continuous_deep_risk_readout import build_deep_frozen_model
            model=build_deep_frozen_model(payload,a.connectome)
            source_meta=dict(parameter_schema=payload['schema'],paid_counters=saved.get('counters'),
                risk_topology=payload['configuration']['risk_topology'])
            del saved,payload
        else:
            from fly_touhou.thrl_parameter_transfer import read_evaluation_source
            from fly_touhou.thrl_pair_policy import model_from_config
            saved=read_evaluation_source(torch,a.checkpoint,a.checkpoint_sha256)
            wrapper=saved['model_config']['pair_risk_wrapper']
            expected=dict(beta=8.,buffer=1.,risk_width=.5,width=128,laser_width=128,
                laser_buffer=4.,laser_risk_width=1.,aggregation='worst-known-margin-v2',
                laser_buffer_weight=.1,frozen_readout=True)
            if any(wrapper.get(k)!=v for k,v in expected.items()):raise ValueError('Actual complete D95 beta8 point+laser wrapper required')
            model=model_from_config(**saved['model_config']).to('cuda:0')
            model.load_state_dict(saved['model'],strict=True);model.eval().requires_grad_(False)
            source_meta=dict(parameter_schema=saved['schema'],pair_risk_wrapper=wrapper,
                base_GRU_and_both_risk_networks_retained=True,risk_composed_exactly_once=True)
            del saved
        from fly_touhou.continuous_no_bomb_first_contact import make_no_bomb_policy,evaluate_no_bomb
        policy=make_no_bomb_policy(torch,model,family=a.family,episodes=a.episodes)
        versions=tuple(p._version for p in model.parameters())
        def frozen_guard():
            guard()
            if tuple(p._version for p in model.parameters())!=versions:raise RuntimeError('Frozen weights changed')
        launch.update(source=source_meta,CUDA_started=True,native_started=False)
        write_json(a.output/'launch.json',launch)
        if a.family=='deep':write_json(a.output/'neural-port-order.json',dict(
            VP_graph_indices=model.core.input_indices.detach().cpu().tolist(),
            DN_graph_indices=model.core.output_indices.detach().cpu().tolist(),
            graph_provenance=model.core.graph_provenance,units='dimensionless continuous ANN activity'))
        report=evaluate_no_bomb(torch,policy,cfg,a.output/'evaluation',checkpoint_sha256=a.checkpoint_sha256,
            episodes=a.episodes,workers=a.workers,seed=a.seed,check_stop=frozen_guard,worker_cpus=worker_cpus,
            progress=lambda row:write_json(a.output/'progress.json',dict(row,entry_schema=SCHEMA)))
        frozen_guard()
        write_json(a.output/'result.json',dict(schema=SCHEMA,status='completed',family=a.family,
            source=source_meta,checkpoint_sha256=a.checkpoint_sha256,evaluation=report,
            optimizer_updates=0,teacher_actions=0,elapsed_seconds=time.monotonic()-begun))
    except BaseException as error:
        write_json(a.output/'failure.json',dict(schema=SCHEMA,error=repr(error),optimizer_updates=0,
            unfinished_games_are_not_scores=True,automatic_retry=False));raise


def main(argv=None):
    a=arguments(argv);prepared=plan(a)
    if not a.execute_cuda:print(json.dumps(prepared,ensure_ascii=False,indent=2));return 0
    execute(a,prepared);return 0


if __name__=='__main__':raise SystemExit(main())
