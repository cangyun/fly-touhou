"""Read stage/resources of existing training games without operating them."""
import argparse
import json
from pathlib import Path
import sys

import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.native import NativeReader,ProcessMemory,Profile
from fly_touhou.thrl_study_io import read_shared_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ownership',type=Path,required=True)
    p.add_argument('--environment-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=read_shared_json(a.environment_config)
    profile=Profile.load(cfg['profile']);rows=[]
    for worker in read_shared_json(a.ownership)['workers']:
        identity=worker['native_identity'];process=psutil.Process(identity['pid'])
        if abs(process.create_time()-identity['create_time'])>.01:continue
        memory=ProcessMemory(process.pid,writable=False,can_suspend=False)
        try:
            reader=object.__new__(NativeReader);reader.mem=memory;reader.profile=profile;reader.o=profile.offsets
            frame=reader.global_frame()
            stage=reader.static('environment','stage');stage_frame=reader.stage_frame()
            lives=reader.static('statics','lives',signed=True);bombs=reader.static('statics','bombs')
            rows.append(dict(world=worker['index'],stage=stage,stage_frame=stage_frame,
                             lives=lives,bombs=bombs,same_frame=reader.global_frame()==frame))
        finally:memory.close()
    result=dict(worlds=rows,scope='Read-only snapshots of existing game instances;no game input or pause')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
