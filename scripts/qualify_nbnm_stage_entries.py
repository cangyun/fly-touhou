"""Round-trip real Stage Practice prefixes, without neural optimization."""
from pathlib import Path
import argparse,os,sys,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def run(job):
    spec,bank,entry,folder=job;folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-nbnm-prefix-check-'+str(os.getpid()))
    from fly_touhou.nbnm_run import environment
    from fly_touhou.compact_full_story import factory,story_kwargs
    from fly_touhou.thrl_job_paths import configure_native_temp,cleanup_native_temp
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.nbnm_sensation import current_attack
    cfg=environment(spec,evaluation=True)
    cfg['nbnm'].update(task=entry['task'],entry_mode='native_stage_prefix',start_bank=str(bank))
    tmp=configure_native_temp(folder);env=factory(story_kwargs(cfg,folder,1,entry['split'],teacher=False)[0])
    try:
        _,info=env.reset(options=dict(native_seed=entry['native_seed']))
        result=dict(status='passed',task=entry['task'],seed=entry['native_seed'],
            actual_attack=current_attack(env.snapshot),stage_frame=env.snapshot.stage_frame,
            expected_stage_frame=entry['landing_stage_frame'],
            warmed_decisions=len(info['policy_burn_in']['actions']),receipt=env._nbnm_entry_receipt,
            game_mode=env.snapshot.extra['run_mode'],initial_state=env.snapshot.player.state,
            network_training=False,optimizer_updates=0)
        if result['stage_frame']!=result['expected_stage_frame']:
            raise RuntimeError('Original entry frame differed from captured prefix')
        write_json(folder/'report.json',result);return result
    except BaseException:
        write_json(folder/'failure.json',dict(error=traceback.format_exc(),task=entry['task']));raise
    finally:env.close();cleanup_native_temp(tmp)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','bank','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--stages',type=int,nargs='+',default=list(range(1,7)))
    p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    from fly_touhou.nbnm_run import read
    from fly_touhou.nbnm_task import attack_key
    from fly_touhou.thrl_durable_records import write_json
    spec=read(a.config);bank=read(a.bank);jobs=[]
    for stage in a.stages:
        rows=[r for r in bank['entries'] if r['task']['stage']==stage and r['split']=='train'
            and not attack_key(r['task']['attack'])[1]]
        # A late nonspell tests a substantial prefix and its actual warm-up.
        row=max(rows,key=lambda r:r['landing_stage_frame'])
        jobs.append((spec,a.bank,row,str(a.output/f's{stage}')))
    from concurrent.futures import ProcessPoolExecutor,as_completed
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(run,j) for j in jobs]):
            results.append(future.result());write_json(a.output/'progress.json',dict(completed=len(results),target=len(jobs)))
    write_json(a.output/'report.json',dict(status='passed',rows=results,network_training=False,optimizer_updates=0))


if __name__=='__main__':main()
