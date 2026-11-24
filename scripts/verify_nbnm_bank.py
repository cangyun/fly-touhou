"""Verify actual prefix replay and whole-attack coverage, without a network."""
from pathlib import Path
import argparse,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    from fly_touhou.nbnm_run import read,environment,runtime
    from fly_touhou.compact_full_story import factory,story_kwargs
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    spec=read(a.config);bank=read(spec['nbnm']['start_bank'])
    a.output.mkdir(parents=True,exist_ok=False);temporary=runtime(a.output);rows=[]
    for i,entry in enumerate(bank['entries'][:a.limit or None]):
        cfg=environment(spec,evaluation=True);cfg['nbnm']['task']=entry['task']
        env=factory(story_kwargs(cfg,a.output/f'entry-{i}',1,entry['split'],teacher=True)[0])
        started=time.perf_counter()
        try:
            obs,info=env.reset(options=dict(native_seed=entry['native_seed']))
            row=dict(native_seed=entry['native_seed'],split=entry['split'],status='verified',
                receipt=env._nbnm_entry_receipt,stage=env.snapshot.stage,stage_frame=env.snapshot.stage_frame,
                input_width=len(obs['features']),entry_policy_burn_in=len(info['policy_burn_in']['actions']),
                reset_seconds=time.perf_counter()-started)
        except Exception:
            row=dict(native_seed=entry['native_seed'],status='failed',error=traceback.format_exc())
        finally:env.close()
        rows.append(row);write_json(a.output/'report.json',dict(network_training=False,entries=rows))
        print({k:v for k,v in row.items() if k not in ('receipt',)},flush=True)
    cleanup_native_temp(temporary)
    if any(r['status']=='failed' for r in rows):raise RuntimeError('Native start verification failed; inspect report')


if __name__=='__main__':main()
