"""Launch a single visible original game, controlled by live frozen B1 inference."""
from pathlib import Path
import argparse
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))

if __name__=='__main__':
    from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
    from fly_touhou.thrl_durable_records import write_json as write
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',required=True,type=Path);a=p.parse_args()
    try:
        from fly_touhou.thrl_visible_demo import run
        run(a.plan,ROOT)
    except BaseException:
        plan=read(a.plan);output=Path(plan['output'])
        error=dict(status='failed',utc=utc_now(),traceback=traceback.format_exc(),
            teacher_actions=0,optimizer_updates=0,no_automatic_restart=True)
        write(output/'failure.json',error);write(output/'status.json',error)
        raise
