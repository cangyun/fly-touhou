"""Matched native starts, decoder and sensory code; initial versus trained SNN."""
from pathlib import Path
import argparse,json,math,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def reference_checkpoint(spec, explicit=None):
    """Keep the original reference when a new trial warm-starts from a pilot."""
    return Path(explicit or spec.get('reference_checkpoint') or spec['initial_checkpoint'])


def comparison(initial,trained,initial_entries,trained_entries):
    from fly_touhou.nbnm_objective import assert_compatible_evaluations
    assert_compatible_evaluations(initial,trained)
    before={r['native_seed']:r for r in initial['episode_results']}
    after={r['native_seed']:r for r in trained['episode_results']}
    pairs=[]
    for seed in initial['opening_seeds']:
        a,b=before[seed],after[seed]
        pairs.append(dict(native_seed=seed,initial_success=a['nbnm_success'],trained_success=b['nbnm_success'],
            initial_frames=a['task_native_frames'],trained_frames=b['task_native_frames'],
            initial_end=a['end_reason'],trained_end=b['end_reason'],
            identical_entry_observations=initial_entries[str(seed)]==trained_entries[str(seed)]))
    wins=sum(not p['initial_success'] and p['trained_success'] for p in pairs)
    losses=sum(p['initial_success'] and not p['trained_success'] for p in pairs)
    n=wins+losses
    probability=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
    split=initial.get('evaluation_split','development')
    return dict(task=initial['task'],entry_mode=initial.get('entry_mode','story_prefix'),
        evaluation_split=split,
        action_selection=initial['action_selection'],pairs=pairs,
        attempts=len(pairs),initial_successes=initial['successes'],trained_successes=trained['successes'],
        improved_pairs=wins,regressed_pairs=losses,exact_paired_success_p=probability,
        all_entry_observations_identical=all(p['identical_entry_observations'] for p in pairs),
        scope=f'Matched {split} starts; no training or teacher takeover. Small samples are screening evidence, not stable NBNM proof.',
        improvement_supported=(wins>losses and probability<.05 and all(p['identical_entry_observations'] for p in pairs)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--trained',type=Path,required=True)
    parser.add_argument('--initial',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--log-file',type=Path)
    parser.add_argument('--task-group',help='One configured task group; keeps its own native entry protocol')
    parser.add_argument('--split',choices=('development','test'),default='development')
    a=parser.parse_args()
    if a.log_file:
        a.log_file.parent.mkdir(parents=True,exist_ok=True)
        sys.stdout=sys.stderr=a.log_file.open('a',encoding='utf-8',buffering=1)
    from fly_touhou.nbnm_run import read,runtime,load_model,evaluate
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    import psutil
    spec=read(a.config)
    initial_checkpoint=reference_checkpoint(spec,a.initial)
    if a.task_group:
        from fly_touhou.nbnm_run import task_settings
        spec['nbnm']=task_settings(spec,spec['tasks'][a.task_group])
    a.output.mkdir(parents=True,exist_ok=True);temporary=runtime(a.output)
    proc=psutil.Process();start=time.monotonic()
    def progress(data):
        write_json(a.output/'progress.json',dict(data,PID=proc.pid,create_time=proc.create_time(),
            elapsed_seconds=time.monotonic()-start,network_training=False))
    reports={};entries={}
    try:
        import torch,gc
        for name,point in [('initial',initial_checkpoint),('trained',a.trained)]:
            progress(dict(phase='loading_'+name,checkpoint=str(point)))
            folder=a.output/name
            if (folder/'report.json').exists():reports[name]=read(folder/'report.json')
            else:
                torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed'])
                model,_=load_model(spec,point)
                reports[name]=evaluate(model,spec,folder,split=a.split,progress=lambda row:progress(dict(row,model=name)))
                del model;gc.collect();torch.cuda.empty_cache()
            entries[name]=read(folder/'entry-observations.json')
        report=comparison(reports['initial'],reports['trained'],entries['initial'],entries['trained'])
        report.update(initial_checkpoint=str(initial_checkpoint),trained_checkpoint=str(a.trained),
            optimizer_updates=0,elapsed_seconds=time.monotonic()-start)
        write_json(a.output/'report.json',report);progress(dict(phase='complete',summary=report));print(json.dumps(report,ensure_ascii=False),flush=True)
    except BaseException:
        write_json(a.output/'failure.json',dict(error=traceback.format_exc()));progress(dict(phase='failed'));raise
    finally:cleanup_native_temp(temporary)


if __name__=='__main__':main()
