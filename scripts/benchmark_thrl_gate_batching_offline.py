"""Real spawn/shared-memory, fake game worlds and frozen CPU M1 architecture."""
from pathlib import Path
import argparse,json,sys,time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import torch
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.thrl_model import MOActorCritic
    from fly_touhou.thrl_course_evaluation import evaluate_cases
    from fly_touhou.thrl_curriculum import configuration
    from tests.thrl_gate_batching_fixture import factory
    torch.set_num_threads(2);torch.manual_seed(917)
    config=read_shared_json(a.source_config)
    model=MOActorCritic(**config['model_config']).cpu()
    config.update(game_exe='NEVER_OPENED.exe',profile='NEVER_OPENED.json',curriculum_manifest='fixture-only')
    cases=[('current','s1_opening',2360,20),('chain','s1_opening',None,20)]
    results={};reference=None
    for workers,window in ((1,0),(8,0),(8,2),(16,2)):
        name=f'workers{workers}-wait{window}';folder=a.output/name
        candidate={**config,'curriculum_eval_batch_wait_ms':window,'curriculum_eval_min_batch_size':4}
        begun=time.monotonic()
        with patch('fly_touhou.thrl_vector.worker_factory',factory):
            groups,timing=evaluate_cases(torch,model,candidate,configuration({'version':2}),cases,folder,
                lambda:None,root_seed=731142,frontier=0,workers=workers)
        timing['wall_seconds']=time.monotonic()-begun
        rows=sorted([json.loads(line) for line in (folder/'episodes.jsonl').read_text('utf-8').splitlines()],
                    key=lambda row:row['evaluation_job_id'])
        normalized=[{k:v for k,v in row.items() if k!='evaluation_worker'} for row in rows]
        if reference is None:reference=normalized
        results[name]=dict(timing=timing,groups=groups,episode_actions_and_metrics_equal=normalized==reference,
            cleanup=read_shared_json(folder/'native_cleanup.json'))
        write(a.output/'progress.json',dict(complete=False,finished=name,results=results))
        print(json.dumps(dict(variant=name,wall_seconds=timing['wall_seconds'],
            mean_batch=timing['mean_policy_batch_size'],matched=normalized==reference)),flush=True)
    report=dict(complete=True,status='passed' if all(r['episode_actions_and_metrics_equal'] and
        not r['cleanup']['close_errors'] for r in results.values()) else 'failed',
        fixture='real multiprocessing spawn and shared memory; delayed fake worlds, no game executable',
        policy='actual M1 architecture with fixed random seed917 weights, frozen CPU inference; not trained checkpoint',
        model_configuration=model.model_config(),parameters=model.parameter_counts()['total'],CPU_threads=2,
        cases=cases,root_seed=731142,results=results,
        source_config=str(a.source_config),PPO_updates=0,native_games_started=0,
        claim_scope='Scheduling/parity and CPU fixture timings only; not measured original-game or CUDA speedup')
    write(a.output/'report.json',report)
    if report['status']!='passed':raise RuntimeError('Offline parity or cleanup failed')
    print(str(a.output/'report.json'))


if __name__=='__main__':main()
