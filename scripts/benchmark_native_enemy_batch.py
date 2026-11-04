"""Measure shared enemy forecasts against independent teacher paths on CPU."""
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
    namespace=runpy.run_path(str(ROOT/'tests/test_native_enemy_batch.py'))
    scene,single,batch=(namespace[k] for k in
        ('scene','teacher_enemy_path_clearance','teacher_enemy_paths_clearances'))
    cases=[]
    for enemies in (1,12):
        for position in ((0.,300.),(181.,33.)):
            encoding,paths,_=scene(position,enemies=enemies)
            def execute(packed):
                if packed:return batch(encoding,paths)
                return [single(encoding,path) for path in paths]
            execute(False);execute(True)
            timings={False:[],True:[]}
            for repetition in range(5):
                for mode in ([False,True] if repetition%2==0 else [True,False]):
                    begin=time.perf_counter();execute(mode);timings[mode].append(time.perf_counter()-begin)
            before,after=(statistics.median(timings[mode]) for mode in (False,True))
            cases.append(dict(enemies=enemies,player=position,candidate_paths=18,
                old_seconds=before,batch_seconds=after,speedup=before/after,
                old_samples=timings[False],batch_samples=timings[True]))
    report=dict(scope='Synthetic native enemy movement/contact schedules. CPU component only;five alternating timings,not measured end-to-end game throughput.',
        cases=cases,live_training_modified=False)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
