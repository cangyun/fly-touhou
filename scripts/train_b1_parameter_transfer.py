"""One finite B1 experiment from evaluation weights and a fresh empty Adam.

Distinct from BC and exact recovery. --preflight-only never starts CUDA/games.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.thrl_parameter_transfer import (
    build_parameter_transfer,configuration_sha,file_sha,read_evaluation_source,
    validate_finite_config,validate_target_parameters)

GIB=1024**3
START_MEMORY_BYTES=16*GIB
RUNNING_MEMORY_BYTES=3*GIB
STRICT_FP32_PRECISION=dict(schema='strict-FP32-no-TF32-no-autocast-v1',
    parameter_dtype='float32',observation_dtype='float32',autocast_enabled=False,
    matmul_allow_tf32=False,cudnn_allow_tf32=False)


def precision_state(torch):
    return dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        autocast_enabled=torch.is_autocast_enabled(),default_dtype=str(torch.get_default_dtype()))


def configure_fp32_math(torch,config):
    if config.get('precision')!=STRICT_FP32_PRECISION:
        raise ValueError('Explicit strict FP32 configuration with both TF32 flags and autocast disabled is required')
    # These backend flags do not initialize CUDA, including CPU preflight.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    actual=precision_state(torch)
    if actual['autocast_enabled'] or actual['default_dtype']!='torch.float32':
        raise ValueError('The independent entry requires FP32 default dtype and no enclosing autocast context')
    return actual


def require_start_memory(available):
    if available<START_MEMORY_BYTES:
        raise RuntimeError('At least16GiB available RAM is required before the30-world experiment')


def require_running_memory(available,stop_type):
    if available<RUNNING_MEMORY_BYTES:
        raise stop_type('Available RAM below3GiB;save committed work and close this run without restart')


class ProgressMemoryGate:
    """Raise once so train_trial can publish its StopRequested cleanup report."""
    def __init__(self):self.stop_requested=False
    def check(self,available,stop_type):
        if self.stop_requested:return
        if available<RUNNING_MEMORY_BYTES:
            self.stop_requested=True
            require_running_memory(available,stop_type)


def effective_parameters(config):
    from dataclasses import asdict
    from fly_touhou.thrl_ppo import PPOConfig
    from fly_touhou.th14_completion_protocol import learning_overrides
    parameters=asdict(PPOConfig(**{**config['parameters'],**learning_overrides(config)}))
    if configuration_sha(parameters)!=configuration_sha(config['parameters']):
        raise ValueError('Explicit parameters differ from the effective PPO recipe;do not silently adopt defaults')
    return parameters


def prepare_transfer(torch,config,output):
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_ppo import PPOConfig,make_optimizer
    from fly_touhou.thrl_branching import warm_start
    origin=config['parameter_transfer']
    source=read_evaluation_source(torch,origin['source_checkpoint'],origin['source_sha256'])
    if config['policy_model_config']!=source['model_config']:
        raise ValueError('The new experiment must use the unchanged evaluation model configuration')
    parameters=effective_parameters(config)
    validate_target_parameters(source['config']['parameters'],parameters)
    # CPU preparation only. train_trial will establish its own declared fresh
    # sampling stream and create a separate new GPU model/Adam when run.
    with torch.random.fork_rng(devices=[]):
        model=model_from_config(**config['policy_model_config']).cpu()
        transfer=build_parameter_transfer(torch,model,origin['source_checkpoint'],origin['source_sha256'],
            parameters,seed=config['seed'])
        optimizer=make_optimizer(model,PPOConfig(**parameters))
        target=Path(output)/'evaluation_parameter_transfer.pt'
        temporary=target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.tmp')
        with temporary.open('xb') as stream:
            torch.save(transfer,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,target)
        audit=warm_start(torch,model,optimizer,target,file_sha(target),model_factory=model_from_config)
        if not audit['all_state_tensors_exact'] or optimizer.state:
            raise ValueError('CPU parameter-transfer verification failed')
    write(Path(output)/'parameter_transfer_audit.json',audit)
    return {**config,'warm_start_checkpoint':str(target.resolve()),'warm_start_sha256':file_sha(target)},audit


def run(args):
    import psutil
    import torch
    from train_thrl_branch import verify_sources
    from train_thrl_native import train_trial,StopRequested
    from fly_touhou.thrl_branching import candidate_trial_config
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_job_paths import configure_native_temp
    config=read(args.config)
    capacity=validate_finite_config(config)
    config={**config,'effective_precision':configure_fp32_math(torch,config)}
    if Path(config['source_runtime']).resolve()!=ROOT.resolve():
        raise ValueError('Execute the independently frozen parameter-transfer runtime')
    if file_sha(ROOT/'source_manifest.json')!=config['source_manifest_sha256']:
        raise ValueError('Frozen experiment runtime identity changed')
    if not args.preflight_only:
        require_start_memory(psutil.virtual_memory().available)
        if any((p.info['name'] or '').lower()=='th14.exe' for p in psutil.process_iter(['name'])):
            raise RuntimeError('Existing original-game worlds remain;this entry will not close or replace them')
    torch.set_num_threads(config.get('torch_threads',2))
    args.output.mkdir(parents=True,exist_ok=False);args.output_created=True
    prepared,audit=prepare_transfer(torch,config,args.output)
    verify_sources(prepared)
    from fly_touhou.thrl_native_read_qualification import verify as verify_item_reader
    reading=verify_item_reader(prepared,ROOT)
    if reading is not None:prepared['qualified_native_item_reading']=reading
    write(args.output/'study_config.json',prepared)
    write(args.output/'effective_study_config.json',prepared)
    if args.preflight_only:
        write(args.output/'preflight.json',dict(status='CPU-parameter-transfer-qualified',utc=utc_now(),
            capacity=capacity,source_schema='evaluation-only',source_sha256=config['parameter_transfer']['source_sha256'],
            all_tensors_exact=audit['all_state_tensors_exact'],Adam_state_entries=0,
            source_optimizer_inherited=False,source_RNG_inherited=False,candidate_decisions=0,
            source_temperature=audit['source_action_temperature'],target_temperature=audit['target_action_temperature'],
            precision=config['precision'],effective_precision=precision_state(torch),
            available_RAM_bytes=psutil.virtual_memory().available,CUDA_initialized=torch.cuda.is_initialized(),
            GPU_work=False,native_worlds_started=0,optimizer_updates=0,training_started=False))
        print(json.dumps(dict(status='CPU-parameter-transfer-qualified',output=str(args.output))))
        return
    require_start_memory(psutil.virtual_memory().available)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required;there is no CPU training fallback')
    configure_native_temp(args.output)
    os.environ['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-B1-D95-parameter-transfer-'+str(os.getpid())
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    write(args.output/'launch.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now(),
        optimizer_initialization='fresh empty Adam',source_optimizer_inherited=False,source_RNG_inherited=False,
        candidate_decisions=0,native_worlds_max=30,automatic_restart=False,desktop_input=False,
        precision=prepared['precision'],effective_precision=precision_state(torch)))
    if config.get('goal_panel_registration'):
        from register_b1_goal_phase import register
        panel=config['goal_panel_registration']
        write(args.output/'panel_registration.json',register(args.output,panel['job_id'],'training',
            control_root=Path(panel['control_root'])))
    started=time.monotonic()
    memory_gate=ProgressMemoryGate()
    def progress(row):
        available=psutil.virtual_memory().available
        memory_gate.check(available,StopRequested)
        write(args.output/'study_status.json',dict(status=row.get('status','training'),
            current_run=str(args.output/'trial_000_both'),current_progress=row,updated_utc=utc_now(),
            wall_seconds=time.monotonic()-started,planned_trials=1,device='cuda:0',
            optimizer_initialization='fresh Adam parameter transfer;not BC or numerical resume',
            action_temperature=prepared['parameters']['action_temperature'],available_RAM_bytes=available,
            precision=prepared['precision'],effective_precision=precision_state(torch),
            memory_stop_requested=memory_gate.stop_requested))
    try:
        result=train_trial(torch,candidate_trial_config(prepared,0),args.output,args.output/'trial_000_both',
            effective_parameters(prepared),trial=None,study_started=started,on_progress=progress,
            attention='both',model_factory=model_from_config)
        write(args.output/'study_result.json',dict(status='finished',training=result,formal_acceptance=False,
            source_optimizer_inherited=False,source_RNG_inherited=False,BC_repeated=False,
            PPO_optimizer_fresh=True,numerical_PPO_resume=False,automatic_restart=False))
    except StopRequested as error:
        write(args.output/'study_result.json',dict(status='operator_or_memory_stop',reason=str(error),
            automatic_restart=False,committed_checkpoints_retained=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    try:run(args)
    except BaseException:
        if getattr(args,'output_created',False):
            write(args.output/'failure.json',dict(utc=utc_now(),traceback=traceback.format_exc(),
                no_automatic_restart=True))
        raise
