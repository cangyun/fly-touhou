"""CPU-only compression of a bounded subset of existing original training data."""
from pathlib import Path
import argparse,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('expert','own','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--rows',type=int,default=256);a=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    from fly_touhou.compact_spiking_data import prepare
    from fly_touhou.thrl_durable_records import write_json
    from datetime import datetime,timezone
    import psutil,traceback
    current=psutil.Process();current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    try:
        result=prepare(a.expert,a.own,a.output,rows=a.rows,on_progress=lambda d:write_json(a.output/'progress.json',d))
        write_json(a.output/'result.json',dict(status='completed',rows=sum(r['rows'] for r in result['records']),CUDA_started=False,
            PID=current.pid,create_time=current.create_time(),date=datetime.now(timezone.utc).isoformat()))
    except BaseException as error:
        if a.output.exists():write_json(a.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
