"""Twenty sampled-action original games for the preserved full-fly reference."""
from pathlib import Path
import argparse
import gzip
import json
import os
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'src'))
import numpy as np
import torch
from fly_touhou.connectome import load_connectome
from fly_touhou.experiment import ExperimentLogger,file_sha256
from fly_touhou.geometry_migration import PROOF_SOURCE_FILES
from fly_touhou.native_diagnostic import load_protocol,write_json
from fly_touhou.noise_movement_session import NoiseMovementLessonSession
from fly_touhou.whole_brain_learning import load_whole_brain_checkpoint
from fly_touhou.whole_brain_ppo import campaign_environment_for_policy,_campaign_runtime_files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--ports',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    protocol,encoder=load_protocol(args.protocol)
    if args.output.exists():raise FileExistsError('Preserve older reference trials')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    parent_hash=file_sha256(args.checkpoint)
    graph=load_connectome(args.data,max_neurons=None)
    policy=load_whole_brain_checkpoint(args.checkpoint,graph,device='cuda')
    model=policy.student
    assert policy.encoder.specification()==encoder.specification()
    model.enable_cuda_capture()
    env=campaign_environment_for_policy(policy,protocol['game_exe'],protocol['profile'],
        difficulty=3,frame_skip=4,max_episode_frames=1024,
        private_appdata=args.output/'game_userdata',dialogue_skip=True,parallel_startup=True,
        responsive_wait=True,hide_window=True,window_title='完整果蝇脑 · 20局采样动作参考测试')
    env.curriculum_first_hit=True
    config={'phase':'preserved-full-fly-pretraining-reference20-not-new-training',
        'mode':'evaluate','difficulty':3,'frame_limit':1024,'episodes_requested':20,
        'game_exe_sha256':protocol['game_exe_sha256'],'checkpoint_sha256':parent_hash,
        'protocol_sha256':file_sha256(args.protocol),'model_config':model.config(),
        'encoder':encoder.specification(),'seed':1_000_014,'reference_training_seed_label':11,
        'action_selection':'categorical sampling18,temperature1;matchesGRUdiagnostic',
        'neural_noise_keys':'(10001+episode_index*9973)&0x7fffffff;freshwarmup each new original game',
        'movement_curriculum':{'action_domain':list(range(18))},
        'reference_contains_historical_training':True,'new_training_decisions':0,
        'not_a_matched_new_training_budget_comparison':True,'desktop_input':False,
        'final_acceptance':False}
    files={'checkpoint':args.checkpoint,'protocol':args.protocol,'runner':Path(__file__),
        'game_executable':Path(protocol['game_exe']),'profile':Path(protocol['profile']),
        'connectivity':args.data/'Connectivity_783.parquet','completeness':args.data/'Completeness_783.csv',
        'upstream_model':args.data/'model.py','ports':args.ports,
        'source_manifest':SOURCE/'snapshot_manifest.json',**_campaign_runtime_files(env)}
    for relative in PROOF_SOURCE_FILES:files[relative]=SOURCE/relative
    for name in ('movement_lesson_session','noise_movement_session'):
        files[name]=SOURCE/'src/fly_touhou'/f'{name}.py'
    logger=ExperimentLogger(args.output,config,files=files)
    logger.record_runtime()
    logger.bind_whole_brain(graph,model,args.data,ports_manifest=args.ports)
    session=NoiseMovementLessonSession(env,policy,logger=logger,run_id=logger.run_id,
        seed=config['seed'],labels=False,deterministic=False,feedback=None)
    started=time.monotonic()
    try:
        while len(session.completed_episodes)<20:
            path=args.output/f'native_eval_{session.native_steps:06d}.jsonl.gz'
            with gzip.open(path,'xt',encoding='utf-8',compresslevel=1) as stream:
                session.collect(128,stream=stream,stop_after_episodes=20)
            logger.event('sealed_native_movement_evaluation_part',{'path':str(path),'sha256':file_sha256(path)})
            write_json(args.output/'progress.json',{'phase':'sampled-reference20',
                'completed_episodes':len(session.completed_episodes),'native_decisions':session.native_steps})
        frames=[e['native_frames_to_first_hit_or_limit'] for e in session.completed_episodes]
        assert file_sha256(args.checkpoint)==parent_hash
        report={'status':'finished','config':config,'episodes':session.completed_episodes,
            'median_frames_to_first_hit_or_limit':float(np.median(frames)),
            'no_hit_rate':sum(not e['hit'] and e['native_frames_to_first_hit_or_limit']==1024 for e in session.completed_episodes)/20,
            'teacher_label_queries':0,'teacher_action_calls':0,'safety_takeovers':0,
            'wall_seconds':time.monotonic()-started,'checkpoint_file_unchanged':True,
            'G3_achieved':False,'final_acceptance':False}
        write_json(args.output/'report.json',report,exclusive=True)
        logger.event('sampled_full_fly_reference_finished',{'report_sha256':file_sha256(args.output/'report.json')})
        print(json.dumps({'episodes':20,'frames':frames,'no_hit_rate':report['no_hit_rate'],
            'median':report['median_frames_to_first_hit_or_limit'],'G3_achieved':False},ensure_ascii=False),flush=True)
    except BaseException as error:
        logger.record_error(error,phase='sampled-full-fly-reference')
        raise
    finally:env.close()


if __name__=='__main__':main()
