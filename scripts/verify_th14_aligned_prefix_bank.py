"""Reverify every saved landing with explicit, narrowly guarded visual RNG alignment."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def verify(task):
    from fly_touhou.thrl_segment_env import THRLSegmentEnv
    from fly_touhou.native_replay import state_fingerprint
    folder=Path(task['directory']);folder.mkdir(parents=True,exist_ok=False)
    env=THRLSegmentEnv(task['exe'],task['profile'],private_appdata=folder/'userdata',
        start_bank=task['entry_path'],input_schema=task['input_schema'],policy_action_count=19,
        headless=True,binary_reader=True)
    try:
        observation,info=env.reset()
        receipt={'id':task['id'],'input_schema':task['input_schema'],'pid':env.pid,
            'landing_sha256':state_fingerprint(env.snapshot),'prefix':info['prefix_reset'],
            'observation_shapes':{key:list(value.shape) for key,value in observation.items()},'status':'passed'}
    finally:env.close()
    (folder/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt


def main():
    from fly_touhou.native_replay import semantic_fingerprint,VISUAL_ALIGNMENT_SCHEMA
    from fly_touhou.thrl_segment_env import load_start_bank
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];evidence=Path('./outputs/engineering/headless-20261022')
    entries=[];tasks=[]
    for seed in (12345,23023,37037):
        reference=json.loads((evidence/f'prefix-base-diagnostic-v1/{seed}-original273-v1/report.json').read_text('utf-8'))['state']
        for original in load_start_bank(evidence/f'curriculum-bank-seed{seed}-v2/bank.json'):
            if semantic_fingerprint(reference)!=original['base_state_sha256']:raise ValueError('Base reference changed')
            entry=deepcopy(original);entry['base_visual_rng_alignment']={'schema':VISUAL_ALIGNMENT_SCHEMA,
                'reference_state':reference,'no_resource_or_gameplay_rng_writes':True}
            index=len(entries);entries.append(entry)
            entry_path=a.output/f'entry{index}.json';entry_path.write_text(json.dumps(entry,indent=2),encoding='utf-8')
            for schema in ('original273-v1','semantic-separated-v1'):
                tasks.append({'id':entry['id'],'entry_path':str(entry_path),'directory':str(a.output/f'proof{index}-{schema}'),
                    'input_schema':schema,'profile':str(root/'profiles/th14_1.00b_original.json'),
                    'exe':'games/th14/th14.exe'})
    with ProcessPoolExecutor(max_workers=9) as pool:receipts=list(pool.map(verify,tasks))
    for entry in entries:
        verified=[row for row in receipts if row['id']==entry['id']]
        if len(verified)!=2 or any(row['landing_sha256']!=entry['landing_state_sha256'] for row in verified):
            raise ValueError('Independent aligned replay mismatch')
        entry['aligned_verification_receipts']=verified
    (a.output/'bank.json').write_text(json.dumps({'schema':'th14-native-start-bank-aligned-v1','entries':entries,
        'resources_injected':False,'formal_test':False},indent=2),encoding='utf-8')
    result={'status':'passed','entries':len(entries),'new_independent_replays':len(receipts),
        'visual_rng_alignments':sum(row['prefix']['visual_rng_alignment'] is not None for row in receipts),
        'strict_full_state_hash_required':True,'gameplay_rng_or_resource_writes':False}
    (a.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
