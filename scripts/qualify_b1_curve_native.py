"""Finite frozen-policy original-game curve-format probe, no optimizer.

Six or fewer fresh Story openings on train-partition RNG. Stop once a genuine
single native curve node and its same-object growth both pass the encoder.
These bounded prefixes are not scored as completed games or skill tests.
"""
from pathlib import Path
from collections import deque
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write


def run(args):
    import psutil,torch
    from fly_touhou import thrl_vector
    from fly_touhou.thrl_curve_geometry_probe import curve_geometry_probe_factory
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_inputs import forward_observation
    from fly_touhou.thrl_policy_actions import policy_logits
    from fly_touhou.thrl_job_paths import configure_native_temp
    if args.output.exists() or not 1<=args.games<=6 or not 1<=args.workers<=6:
        raise ValueError('Fresh bounded qualification required')
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=args.sha256:
        raise ValueError('Frozen interrupted model identity changed')
    if any((p.info['name'] or '').lower().startswith('th14') for p in psutil.process_iter(['name'])):
        raise RuntimeError('Existing original worlds must close before this geometry probe')
    if not torch.cuda.is_available():raise RuntimeError('Frozen-policy qualification requires CUDA')
    args.output.mkdir(parents=True);configure_native_temp(args.output)
    os.environ['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-curve-format-'+str(os.getpid())
    torch.set_num_threads(2);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    cfg=read(args.config);payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=model_from_config(**payload['model_config']).cuda().eval();model.load_state_dict(payload['model'],strict=True);del payload
    workers=min(args.workers,args.games);kwargs=[]
    for worker in range(workers):
        kwargs.append(dict(executable=cfg['game_exe'],profile_path=cfg['profile'],
            private_appdata=str(args.output/'native'/f'worker{worker}'/'appdata'),difficulty=3,
            frame_skip=4,policy_action_count=19,input_schema=cfg['input_schema'],headless=True,hide_window=True,
            timeout=25,require_complete_reward_mapping=True,training_mode='full-story-v1',
            full_story_split='train',full_story_seed=args.seed,reward_schema=cfg['reward_schema'],
            reward_gamma=cfg.get('reward_gamma',.999),curve_probe_directory=str(args.output/'raw'/f'worker{worker}')))
    jobs=deque([])
    for game in range(args.games):
        e,a=np.random.SeedSequence([args.seed,game]).generate_state(2)
        jobs.append(dict(game=game,evaluation_seed=int(e),rng=np.random.default_rng(int(a))))
    write(args.output/'launch.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now()))
    write(args.output/'plan.json',dict(games_max=args.games,workers=workers,native_partition='train',
        seed=args.seed,checkpoint=str(args.checkpoint),checkpoint_sha256=args.sha256,
        optimizer_updates=0,original_story_openings=True,training=False,not_a_skill_score=True))
    vector=None;slots={};hidden=torch.zeros(1,workers,model.gru_hidden_size,device='cuda');receipts={};decisions=0;completed=0
    begun=time.monotonic();last_write=0.;qualified=False;prior=thrl_vector.worker_factory
    def check():
        if (args.output/'STOP').exists() or time.monotonic()-begun>args.wall_seconds:
            raise RuntimeError('Finite curve format probe stopped or wall budget reached')
    def assign(index):
        slots[index]=jobs.popleft();hidden[:,index]=0
        vector.submit_at([index],'reset',options=[{'evaluation_seed':slots[index]['evaluation_seed']}])
    try:
        thrl_vector.worker_factory=curve_geometry_probe_factory
        vector=thrl_vector.THRLNativeVector(kwargs,args.output/'workers')
        for index in range(workers):assign(index)
        with torch.inference_mode():
            while vector._inflight:
                check();ready=[];observations={}
                for message in vector.receive_ready(max_wait_ms=2,min_batch=1,check_stop=check):
                    index=message['index'];info=message['info'];observations[index]=message['observation']
                    if message['operation']=='step':decisions+=1
                    for path in info.get('curve_probe_receipts',[]):receipts[path]=read(Path(path))
                    kinds={row['kind'] for row in receipts.values() if row['encoder_accepted']}
                    if {'single_node','grown_after_single'}<=kinds:qualified=True
                    if message.get('terminated'):
                        completed+=1
                        if jobs:assign(index)
                    else:ready.append(index)
                if qualified:break
                if ready:
                    batch={key:np.stack([observations[i][key] for i in ready]) for key in observations[ready[0]]}
                    logits,_,following=forward_observation(torch,model,batch,hidden[:,ready]);hidden[:,ready]=following
                    p=policy_logits(logits).softmax(-1).cpu().numpy();actions=[]
                    for j,index in enumerate(ready):
                        action=min(18,int(np.searchsorted(np.cumsum(p[j],dtype=np.float64),slots[index]['rng'].random(),side='right')))
                        actions.append(action)
                    vector.submit_at(ready,'step',actions)
                if time.monotonic()-last_write>5:
                    write(args.output/'progress.json',dict(utc=utc_now(),decisions=decisions,completed_games=completed,
                        captured_receipts=list(receipts),native_workers=workers,optimizer_updates=0));last_write=time.monotonic()
        result=dict(schema='B1-native-single-curve-node-format-qualification-v1',utc=utc_now(),
            status='qualified' if qualified else 'requested-native-case-not-observed',
            actual_native_receipts=list(receipts),acknowledged_decisions=decisions,completed_games=completed,
            optimizer_updates=0,not_a_skill_score=True,early_stop_for_geometry_qualification=qualified)
        write(args.output/'report.json',result);print(json.dumps(result));return result
    except BaseException:
        write(args.output/'failure.json',dict(utc=utc_now(),traceback=traceback.format_exc(),optimizer_updates=0))
        raise
    finally:
        thrl_vector.worker_factory=prior
        if vector is not None:
            vector.close();write(args.output/'cleanup.json',dict(closed=True,close_errors=vector.close_errors))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    for key in ['checkpoint','config','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--sha256',required=True);parser.add_argument('--games',type=int,default=6)
    parser.add_argument('--workers',type=int,default=6);parser.add_argument('--seed',type=int,default=791463)
    parser.add_argument('--wall-seconds',type=float,default=1200.)
    run(parser.parse_args())
