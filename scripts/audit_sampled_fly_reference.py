"""Independent CPU full-fly readout, random-action and twenty-game audit."""
from pathlib import Path
from collections import OrderedDict
import argparse
import gzip
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from fly_touhou.background import BackgroundTouhouEnv
from fly_touhou.experiment import file_sha256,verify_events
from fly_touhou.native_diagnostic import write_json
from fly_touhou.state import Snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve older audits')
    torch.set_num_threads(2)
    report=json.loads((args.run/'report.json').read_text(encoding='utf-8'))
    manifest=json.loads((args.run/'manifest.json').read_text(encoding='utf-8'))
    events=verify_events(args.run/'events.jsonl')
    assert events[0]['payload']['manifest_sha256']==file_sha256(args.run/'manifest.json')
    config=report['config']
    assert config['seed']==1_000_014 and config['episodes_requested']==20 and config['mode']=='evaluate'
    assert config['action_selection'].startswith('categorical sampling18')
    proof=next(e['payload'] for e in events if e['event']=='whole_brain_verified')
    assert proof['neurons']==138639 and proof['directed_edges']==15091983
    assert proof['neuron_coverage']==proof['connection_record_coverage']==1
    for item in manifest['files'].values():assert file_sha256(item['path'])==item['sha256']
    checkpoint=torch.load(manifest['files']['checkpoint']['path'],map_location='cpu',weights_only=True)
    head=checkpoint['trainable_parameters']['action_readout.weight'].double().numpy()
    reference=checkpoint['model_config']['parameters']['reference_poisson_rate_hz']
    assert head.shape==(36,1299) and reference==150
    sampler=torch.Generator(device='cpu').manual_seed(config['seed'])
    observed=OrderedDict();max_error=0.;actions=0;previous=None
    parts=[e['payload'] for e in events if e['event']=='sealed_native_movement_evaluation_part']
    for part in parts:
        assert file_sha256(part['path'])==part['sha256']
        with gzip.open(part['path'],'rt',encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);identity=row['episode_id']
                before=Snapshot.from_dict(row['snapshot']);after=Snapshot.from_dict(row['next_snapshot'])
                for snapshot in (before,after):
                    snapshot.validate()
                    assert snapshot.extra['desktop_input'] is False and snapshot.difficulty==3
                    assert snapshot.extra['continues']==snapshot.extra['run_mode']==snapshot.extra['replay_playback']==0
                if identity not in observed:
                    if previous is not None:assert previous['terminated'] or previous['truncated']
                    observed[identity]={'frames':0,'hits':0,'steps':0}
                else:
                    assert previous['episode_id']==identity and not previous['hit']
                    assert previous['next_snapshot']==row['snapshot']
                e=row['action_evidence'];rates=np.asarray(e['descending_filtered_rates_hz'],dtype=np.float64)
                declared=np.asarray(e['logits36'],dtype=np.float64)
                expected=(rates/reference)@head.T
                error=float(np.max(np.abs(expected-declared)));max_error=max(max_error,error)
                assert error<5e-5 and e['selected_action']==row['action']<18
                logp=torch.tensor(declared[:18],dtype=torch.float64).log_softmax(-1)
                action=int(torch.multinomial(logp.exp(),1,generator=sampler))
                assert action==row['action']
                assert abs(float(logp[action])-row['selected_conditional_log_probability'])<1e-12
                assert row['teacher_action_calls']==row['teacher_label_queries']==row['safety_takeovers']==0
                assert row['elapsed_frames']==after.frame-before.frame==4
                assert row['hit']==BackgroundTouhouEnv.reward(before,after,4)[1]
                index=list(observed).index(identity)
                assert row['neural_rng_key']==(10001+index*9973)&0x7fffffff
                item=observed[identity];item['frames']+=4;item['hits']+=int(row['hit']);item['steps']+=1
                assert item['hits']<=1 and item['frames']<=1024
                previous=row;actions+=1
    assert len(observed)==len(report['episodes'])==20
    assert previous['terminated'] or previous['truncated']
    for episode,(identity,actual) in zip(report['episodes'],observed.items()):
        assert identity==episode['episode_id'] and actual['frames']==episode['native_frames_to_first_hit_or_limit']
        assert actual['hits']==int(episode['hit']) and actual['steps']==episode['decisions']
        assert episode['bomb_actions']==episode['teacher_action_calls']==episode['safety_takeovers']==0
    result={'schema':'independent-sampled-full-fly-reference20-audit-v1','status':'passed',
        'report_sha256':file_sha256(args.run/'report.json'),'auditor_sha256':file_sha256(Path(__file__)),
        'neurons':138639,'connection_records':15091983,'episodes':20,'native_actions_verified':actions,
        'maximum_CPU_GPU_DN_readout_error':max_error,'sampling_sequence_verified':True,
        'frames':[e['native_frames_to_first_hit_or_limit'] for e in report['episodes']],
        'no_hit_rate':report['no_hit_rate'],'teacher_actions':0,'safety_takeovers':0,'Bomb_requests':0,
        'G3_achieved':False,'final_acceptance':False}
    write_json(args.output,result,exclusive=True)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
