"""One finite V5 CUDA training phase after the pinned V4 comparison closes.

No scheduling, restarts, input injection or additional native pool while waiting.
The existing trainer handles initial development, real PPO, cleanup and final
development. Its full-story19-action protocol and original reward are unchanged.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,traceback
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write


def alive(identity):
    import psutil
    try:return abs(psutil.Process(identity['pid']).create_time()-identity['create_time'])<.01
    except psutil.Error:return False


def dependency(plan):
    status=read(Path(plan['upstream_control'])/'status.json')
    if status.get('status') in ('failed','launcher-error') or (Path(plan['upstream_output'])/'failure.json').exists():
        return dict(phase='failed',reason='pinned V4 comparison failed; no restart requested')
    if status.get('status')!='finished':
        return dict(phase='waiting',upstream_live=alive(plan['upstream_identity']))
    if status.get('exit_code')!=0:return dict(phase='failed',reason='nonzero upstream exit')
    output=Path(plan['upstream_output']);result=read(output/'report.json')
    if (result.get('status')!='completed' or result.get('native_games')!=140
            or result.get('new_optimizer_updates')!=0):
        return dict(phase='failed',reason='declared frozen140-game comparison incomplete')
    clean=read(output/'no_Bomb_fresh100/cleanup.json')
    if clean!=dict(closed=True,close_errors=[]):return dict(phase='failed',reason='no-Bomb native pool not closed')
    def pool_closed(folder,count):
        ownership=read(folder/'workers/ownership.json')
        if ownership.get('status')!='closed' or ownership.get('close_errors')!=[]:return False
        workers=ownership.get('workers',[])
        if isinstance(workers,dict):workers=list(workers.values())
        if len(workers)!=count:return False
        for row in workers:
            if row.get('worker_alive_after_close') is not False:return False
            identities=[row.get('worker_identity'),row.get('native_identity'),*row.get('native_history',[])]
            for identity in identities:
                if identity and (identity.get('create_time') is None or alive(identity)):return False
        return True
    if not pool_closed(output/'no_Bomb_fresh100',30):
        return dict(phase='failed',reason='no-Bomb pinned native/worker ownership is still live or unverified')
    for label in ('initial','final'):
        clean=read(output/('full_story_'+label)/'native_cleanup.json')
        if clean.get('close_errors')!=[] or clean.get('workers')!=20:
            return dict(phase='failed',reason='full-story native pool not cleanly closed')
        if not pool_closed(output/('full_story_'+label),20):
            return dict(phase='failed',reason='full-story owned worker cleanup incomplete')
    if alive(plan['upstream_identity']):return dict(phase='waiting',upstream_live=True)
    return dict(phase='completed',upstream_live=False)


def run(a):
    import psutil
    plan=read(a.plan);control=Path(plan['control']);config_path=Path(plan['config']);cfg=read(config_path)
    if cfg.get('workers')!=30 or cfg.get('input_schema')!='semantic-separated-observed-laser-segments-v5':
        raise ValueError('This finite job is the declared30-world V5 B1 phase only')
    identity=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now())
    write(control/'phase_launch.json',identity)
    try:
        while True:
            if (control/'STOP').exists() or (Path(plan['output'])/'STOP').exists():
                raise RuntimeError('Operator stopped queued V5 phase')
            state=dependency(plan)
            if state['phase']=='failed':raise RuntimeError(state['reason'])
            if state['phase']=='completed':break
            if not state.get('upstream_live') and time.time()-identity['create_time']>30:
                raise RuntimeError('Pinned comparison process disappeared without a successful result')
            write(control/'phase_status.json',dict(status='waiting_for_existing_native_comparison',utc=utc_now(),
                identity=identity,owned_native_instances=0,upstream=state,no_automatic_restart=True))
            time.sleep(10)
        # Model creation and CUDA import begin only after all old owned games
        # are closed. Exact source and parent identities are checked by trainer.
        qualified=read(plan['parent_qualification'])
        parent=cfg['parent_PPO_policy']
        if (qualified.get('status')!='qualified' or qualified.get('scratch_frozen_both_risk_heads_exact') is not True
                or hashlib.sha256(Path(parent['checkpoint']).read_bytes()).hexdigest()!=parent['sha256']
                or parent['sha256']!=qualified['checkpoint_sha256']):
            raise ValueError('Prepared CUDA-qualified parent identity differs')
        write(control/'phase_status.json',dict(status='starting_CUDA_B1_laser_phase',utc=utc_now(),identity=identity,
            owned_native_instances_max=30,old_comparison_complete=True))
        from train_b1_avoidance_ppo import run as train
        from fly_touhou.thrl_pair_policy import model_from_config
        train(SimpleNamespace(config=config_path,output=Path(plan['output'])),model_factory=model_from_config)
        result=read(Path(plan['output'])/'study_result.json')
        status='finished' if result.get('status')=='finished' else 'stopped_without_automatic_restart'
        write(control/'phase_status.json',dict(status=status,utc=utc_now(),identity=identity,
            report=str(Path(plan['output'])/'study_result.json'),owned_native_instances=0))
    except BaseException:
        write(control/'phase_failure.json',dict(status='failed',utc=utc_now(),identity=identity,
            traceback=traceback.format_exc(),no_automatic_restart=True))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',type=Path,required=True);run(p.parse_args())
