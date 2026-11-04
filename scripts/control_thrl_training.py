"""Detached local control-panel executor. No conversation/model API calls."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json
from fly_touhou.thrl_study_io import read_shared_json,utc_now
from fly_touhou.thrl_control import prepare_execution


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--execute',required=True,type=Path)
    args=parser.parse_args();request=read_shared_json(args.execute);control=Path(request['control'])
    request['python']=sys.executable
    try:
        prepare_execution(request)
        if all((control/f'STOP_{key}').exists() for key in request['keys']):
            atomic_write_json(control/'cancelled_before_launch.json',{'status':'stopped','utc':utc_now()})
            return 0
        command=[sys.executable,str(ROOT/'scripts/run_thrl_comparison.py'),
            '--config',str(control/'comparison_config.json'),'--output',request['output']]
        result=subprocess.run(command,cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        return result.returncode
    except BaseException as error:
        atomic_write_json(control/'action_failure.json',{'error':str(error),
            'traceback':traceback.format_exc(),'stage':'prepare_or_execute','utc':utc_now()})
        raise


if __name__=='__main__':raise SystemExit(main())
