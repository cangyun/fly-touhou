"""Bounded read-only state comparison before policy sampling; no training."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def probe(task):
    from fly_touhou.thrl_native_env import THRLNativeEnv
    from fly_touhou.native_replay import state_fingerprint,semantic_state
    folder=Path(task['directory']);folder.mkdir(parents=True,exist_ok=False)
    env=THRLNativeEnv(task['exe'],task['profile'],private_appdata=folder/'userdata',
        native_seed=task['seed'],input_schema=task['schema'],policy_action_count=19,
        window_title='THRL train 0',headless=True,binary_reader=True)
    report={k:task[k] for k in ('seed','schema','expected')}
    try:
        env.reset();s=env.snapshot
        report['initial_stage_frame']=s.stage_frame
        while s.stage_frame<64:s=env.controller.advance(1,1)
        report.update(actual=state_fingerprint(s),state=semantic_state(s),pid=env.pid,
            seed_audit=s.extra.get('native_seed'),startup=env.controller.startup_recovery_audit())
        report['matched']=report['actual']==report['expected']
        report['read_again_sha256']=[state_fingerprint(env.controller.read()) for _ in range(2)]
        from fly_touhou.native_replay import replay_entry
        try:
            replayed=replay_entry(env.controller,task['entry'])
            report['prefix_landing_matched']=state_fingerprint(replayed)==task['entry']['landing_state_sha256']
        except Exception as error:report['prefix_error']=repr(error)
    finally:env.close()
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return {k:v for k,v in report.items() if k not in ('state','startup')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    base=Path(__file__).resolve().parents[1]
    evidence=Path('./outputs/engineering/headless-20261022')
    tasks=[]
    for seed in (12345,23023,37037):
        entry=json.loads((evidence/f'curriculum-bank-seed{seed}-v2/bank.json').read_text('utf-8'))['entries'][0]
        for schema in ('original273-v1','semantic-separated-v1'):
            tasks.append({'directory':str(a.output/f'{seed}-{schema}'),'seed':seed,'schema':schema,
                'expected':entry['base_state_sha256'],'entry':entry,'profile':str(base/'profiles/th14_1.00b_original.json'),
                'exe':'games/th14/th14.exe'})
    with ProcessPoolExecutor(max_workers=6) as pool:results=list(pool.map(probe,tasks))
    (a.output/'report.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results))


if __name__=='__main__':main()
