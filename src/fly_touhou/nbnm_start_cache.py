"""Build missing independent native prefixes only when that task is evaluated.

No repeated seed is counted as another attempt. A prefix controller is never
part of student PPO or NBNM scoring. Ready prefixes are reused within a study.
"""
from pathlib import Path
import copy


def missing_seeds(bank, task, seeds, split):
    from .nbnm_task import entry_matches
    available={r['native_seed'] for r in bank['entries'] if r['split']==split and entry_matches(r['task'],task)}
    return [int(s) for s in seeds if s not in available]


def prepare_evaluation_bank(spec, settings, task, seeds, split, output, *, progress=lambda row:None, guard=lambda:None):
    """Return a task-local manifest; leave the training bank unchanged."""
    from .nbnm_run import read
    from .nbnm_prefix import SCHEMA
    from .nbnm_task import entry_matches
    from .thrl_durable_records import write_json
    from .thrl_training_mode import seed_split
    if any(seed_split(s)!=split for s in seeds):raise ValueError('Evaluation prefix seed partition mismatch')
    bank=read(settings['start_bank']);rows=list(bank['entries'])
    cache=Path(spec.get('entry_cache_directory',Path(spec['output']).parent/'native-entry-cache'))
    cache.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for seed in missing_seeds(bank,task,seeds,split):
        folder=cache/f"s{task['stage']}-{split}-{seed}-{task['attack']}"
        previous=folder/'report.json'
        if previous.exists():
            cached=read(previous)
            if any(entry_matches(e['task'],task) for e in cached['entries']):
                rows.extend(cached['entries']);continue
        local=copy.deepcopy(spec);local['stage_bank_stop_after_attacks']=[task['attack']]
        jobs.append((local,task['stage'],seed,split,str(folder)))
    if jobs:
        # CLI entry points already add scripts/; retain package-only use too.
        import sys
        scripts=Path(__file__).resolve().parents[2]/'scripts'
        if str(scripts) not in sys.path:sys.path.insert(0,str(scripts))
        from prepare_nbnm_stage_bank import capture
        from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
        progress(dict(phase='preparing_independent_native_entries',task=task,target=len(jobs),completed=0))
        with ProcessPoolExecutor(max_workers=min(int(spec.get('evaluation_workers',4)),len(jobs))) as pool:
            pending={pool.submit(capture,job):job for job in jobs};completed=0
            while pending:
                guard();ready,_=wait(pending,timeout=.5,return_when=FIRST_COMPLETED)
                for future in ready:
                    job=pending.pop(future);result=future.result()
                    if not any(entry_matches(e['task'],task) for e in result['entries']):
                        raise RuntimeError(f'Native prefix did not reach requested task for seed {job[2]}')
                    rows.extend(result['entries']);completed+=1
                    progress(dict(phase='preparing_independent_native_entries',task=task,target=len(jobs),completed=completed))
    manifest=dict(schema=SCHEMA,entries=rows,network_training=False,entry_protocol='native_stage_practice')
    absent=missing_seeds(manifest,task,seeds,split)
    if absent:raise RuntimeError('Independent native starts missing: '+str(absent))
    path=Path(output)/'evaluation-start-bank.json';write_json(path,manifest)
    return str(path)
