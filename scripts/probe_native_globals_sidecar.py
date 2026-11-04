"""Read ECL globals from an already owned native probe; never control its game."""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import struct
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fly_touhou.native import ProcessMemory
from fly_touhou.thrl_durable_records import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    rows=[];files=[];count=0;last=-1;started=time.monotonic()
    def flush():
        nonlocal rows
        if rows:
            name=f'globals-{len(files):04}.pkl.gz'
            with gzip.open(a.output/name,'wb',compresslevel=2) as stream:pickle.dump(rows,stream,protocol=5)
            files.append(dict(file=name,rows=len(rows),first=rows[0]['frame'],last=rows[-1]['frame']))
            rows=[]
        write_json(a.output/'index.json',dict(pid=a.pid,records=count,files=files,
            scope='read-only external ECL globals supplement',two_reads_consistent=True,
            game_control=False,seconds=time.monotonic()-started))
    try:
        with ProcessMemory(a.pid) as mem:
            def read():
                blocks={mem.base+0xdb510:mem.read(mem.base+0xdb510,0x180),
                    mem.base+0xf5800:mem.read(mem.base+0xf5800,0x100)}
                enemy=struct.unpack_from('<I',blocks[mem.base+0xdb510],0x34)[0]
                if enemy:blocks[enemy]=mem.read(enemy,0x100)
                player=mem.integer(mem.base+898684)
                if player:blocks[player]=mem.read(player,0x700)
                ascii_manager=mem.integer(mem.base+898336)
                frame=mem.integer(ascii_manager+102880)
                game=mem.integer(mem.base+898392)
                stage_frame=mem.integer(game+20)
                stage=mem.integer(mem.base+1005732)
                return dict(frame=frame,stage=stage,stage_frame=stage_frame,blocks=blocks)
            while not (a.output/'STOP').exists():
                first=read()
                if first['frame']==last:time.sleep(.001);continue
                time.sleep(.002)
                second=read()
                if first!=second:continue
                last=second['frame'];rows.append(second);count+=1
                if len(rows)>=128:flush()
    except Exception as exc:
        write_json(a.output/'ended.json',dict(reason=repr(exc),records=count,game_control=False))
    finally:flush()


if __name__=='__main__':main()
