"""Compare CPU native-line batching with the serial native-validated helper."""
import argparse
import json
from pathlib import Path
import runpy
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reference=runpy.run_path(str(ROOT/'tests/test_native_line_batch.py'))
    scene,serial,batch=(reference[k] for k in ('scene','serial_reference','native_line_path_clearances'))
    reports=[]
    for name,count,individual,duration,first in [
            ('teacher_18_independent_paths',18,True,24,0),
            ('sensory_18_moves',18,False,24,0),
            ('sensory_324_continuations',324,False,20,4)]:
        features,geometry,starts,velocity=scene(lines=129,candidates=count)
        def execute(use_batch):
            def call(s,v):
                if use_batch:return batch(features,geometry,s,v,duration=duration,initial_time=first)
                return serial(geometry,s,v,duration=duration,initial_time=first)
            if individual:return [call(starts[i:i+1],velocity[i:i+1]) for i in range(count)]
            return call(starts,velocity)
        execute(False);execute(True)
        timings={False:[],True:[]}
        for repetition in range(5):
            for mode in ([False,True] if repetition%2==0 else [True,False]):
                begin=time.perf_counter();execute(mode);timings[mode].append(time.perf_counter()-begin)
        before,after=(statistics.median(timings[mode]) for mode in (False,True))
        reports.append(dict(case=name,lines=129,candidate_paths=count,old_seconds=before,
            batch_seconds=after,speedup=before/after,old_samples=timings[False],batch_samples=timings[True]))
    report=dict(scope='Deterministic synthetic native-line workloads,CPU component only. Five alternating timings with concurrent machine workloads;not end-to-end native-game throughput.',
        cases=reports,live_training_modified=False,reference='Existing scalar native_line_contact_clearances')
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
