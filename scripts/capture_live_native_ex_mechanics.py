"""Read active acceleration fields from the existing private mechanics probe."""
import json,os,struct,sys,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.native import ProcessMemory


def capture(probe):
    owner=psutil.Process(json.loads((probe/'status.json').read_text())['pid'])
    game=next(p for p in owner.children(recursive=True) if p.name().lower()=='th14.exe')
    offsets=json.loads((probe.parent/'th14_1.00b_native_forecast.json').read_text(encoding='utf-8'))['offsets']
    memory=ProcessMemory(game.pid);samples=[];until=time.monotonic()+120
    try:
        while time.monotonic()<until and len(samples)<32:
            owner.suspend()
            try:
                with (probe/'stage3.jsonl').open('rb') as stream:
                    stream.seek(max(0,os.fstat(stream.fileno()).st_size-2000000));lines=stream.read().splitlines()
                for line in reversed(lines):
                    try:previous=json.loads(line);break
                    except json.JSONDecodeError:pass
                ascii_ptr=memory.integer(memory.base+offsets['misc']['ascii_manager_ptr'])
                frame=memory.integer(ascii_ptr+offsets['misc']['global_timer']);rows=[]
                for b in previous['tracked_bullets']:
                    a=b['id'];active=memory.integer(a+0xc04)
                    if not active&(4|0x200000):continue
                    state=struct.unpack('<H',memory.read(a+0xc0e,2))[0]
                    if state not in (1,2):continue
                    g=struct.unpack('<9f',memory.read(a+0xbc0,36));index=struct.unpack('<i',memory.read(a+0xc20,4))[0]
                    command=memory.read(a+0xc28+index*44,44) if 0<=index<18 else None
                    source=4 if active&4 else 0x200000
                    timer,vector,speed=(0xf8c,0xfb0,0xf9c) if source==4 else (0x1214,0x1238,0x1224)
                    tick,elapsed=struct.unpack('<if',memory.read(a+timer,8))
                    ax,ay,az,duration=struct.unpack('<fffi',memory.read(a+vector,16));wait=struct.unpack('<if',memory.read(a+0x10ac,8))
                    rows.append(dict(id=a,native_state=state,native_flags=memory.integer(a+0x20),native_timer=memory.integer(a+0x13c4),
                        native_position=list(g[:2]),native_velocity=list(g[3:5]),native_speed=g[6],native_angle=g[7],
                        native_ex_active_flags=active,native_ex_index=index,native_ex_current_record_hex=command.hex() if command else None,
                        native_animation_gate=memory.integer(a+0x4d4),native_ex_acceleration_tick=tick,native_ex_acceleration_time=elapsed,
                        native_ex_acceleration_vector=[ax,ay],native_ex_acceleration_duration=duration,
                        native_ex_acceleration_speed_delta=struct.unpack('<f',memory.read(a+speed,4))[0],native_ex_acceleration_source=source,
                        native_ex_wait_remaining=wait[0],native_ex_wait_remaining_float=wait[1]))
                if rows and (not samples or samples[-1]['frame']!=frame):samples.append(dict(frame=frame,bullets=rows))
            finally:owner.resume()
            time.sleep(.12)
        path=probe/'active-ex-acceleration-capture.json'
        path.write_text(json.dumps(samples),encoding='utf-8')
        print(json.dumps(dict(path=str(path),snapshots=len(samples),rows=sum(len(s['bullets']) for s in samples))))
    finally:memory.close()


if __name__=='__main__':capture(Path(sys.argv[1]))
