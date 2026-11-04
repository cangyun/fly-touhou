"""Read current compact sensory evidence alongside an owned native probe."""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fly_touhou.native import ProcessMemory,Profile,NativeError
from fly_touhou.hazard_reader import GeometryHazardReader
from fly_touhou.compact_story_sensation import CompactStoryEncoder
from fly_touhou.thrl_durable_records import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid',type=int,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=120.)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'));rows=[];files=[];count=0;last=-1;errors=0;start=time.monotonic()
    def flush():
        nonlocal rows
        if rows:
            name=f'sensory-{len(files):04}.pkl.gz'
            with gzip.open(a.output/name,'wb',compresslevel=2) as stream:pickle.dump(rows,stream,protocol=5)
            files.append(dict(file=name,rows=len(rows),first=rows[0]['snapshot'].frame,last=rows[-1]['snapshot'].frame));rows=[]
        write_json(a.output/'index.json',dict(pid=a.pid,records=count,files=files,
            scope='read-only full Snapshot and current compact labels',game_control=False,
            skipped_incoherent_reads=errors,seconds=time.monotonic()-start))
    try:
        with ProcessMemory(a.pid) as mem:
            reader=GeometryHazardReader(mem,Profile.load(cfg['profile']))
            encoder=CompactStoryEncoder();encoder.set_native_episode(1)
            for key in ('anticipate_arming','growth_motion_envelope','protected_exit_forecast','native_motion_forecast',
                        'native_focus_gate','unknown_laser_player_motion','native_curve_laser_geometry'):
                source={'anticipate_arming':'compact_forecast_native_arming',
                        'growth_motion_envelope':'compact_forecast_growth_motion'}.get(key,'compact_'+key)
                setattr(encoder.base,key,bool(cfg.get(source,False)))
            encoder.primary_attack_target=bool(cfg.get('compact_primary_attack_target',False))
            while time.monotonic()-start<a.seconds and not (a.output/'STOP').exists():
                try:
                    if reader.global_frame()==last:time.sleep(.002);continue
                    snap=reader.read()
                    if snap.player is None:continue
                    encoder.encode(snap)
                except (NativeError,ValueError):errors+=1;time.sleep(.002);continue
                if snap.frame==last:continue
                last=snap.frame;rows.append(dict(snapshot=snap,encoding=encoder.last_result));count+=1
                if len(rows)>=64:flush()
    finally:flush()


if __name__=='__main__':main()
