"""CPU-only replay of closed or actively appended native stage3 captures."""
from __future__ import annotations
import argparse
from collections import Counter
import ctypes
from ctypes import wintypes
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def shared_lines(path):
    if os.name!='nt':
        with path.open(encoding='utf-8') as stream:yield from stream
        return
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=(wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
        ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE)
    kernel.CreateFileW.restype=wintypes.HANDLE
    kernel.ReadFile.argtypes=(wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),ctypes.c_void_p)
    kernel.ReadFile.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=(wintypes.HANDLE,)
    handle=kernel.CreateFileW(str(path.resolve()),0x80000000,7,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    pending=b''
    try:
        while True:
            block=ctypes.create_string_buffer(65536);size=wintypes.DWORD()
            if not kernel.ReadFile(handle,block,len(block),ctypes.byref(size),None):
                raise ctypes.WinError(ctypes.get_last_error())
            if not size.value:break
            pending+=block.raw[:size.value]
            parts=pending.split(b'\n');pending=parts.pop()
            for line in parts:yield line.decode('utf-8-sig')
        # A partial last line belongs to the active writer, so omit it.
    finally:kernel.CloseHandle(handle)


def audit(path):
    from fly_touhou.native_bullet_release import predict_bullet_release
    history={};counts=Counter();modes=Counter();bad=[];examples=[];maximum=0.;rows=0;last=None
    for line in shared_lines(path):
        row=json.loads(line);frame=row['frame'];rows+=1;last=frame
        following={point['id']:point for point in row['tracked_bullets']}
        for horizon in (1,4,12,24):
            previous=history.get((row['episode'],frame-horizon))
            if previous is None:continue
            for raw,prediction in previous:
                current=following.get(raw['id'])
                if current is None or current['age']!=raw['native_timer']+horizon:continue
                if not prediction['known_by_frame'][horizon]:
                    counts[f'unknown_{horizon}']+=1;continue
                a,b=prediction['bounds'][horizon]
                dx,dy=b[0]-a[0],b[1]-a[1];length=dx*dx+dy*dy
                fraction=max(0.,min(1.,((current['x']-a[0])*dx+(current['y']-a[1])*dy)/length)) if length else 0.
                error=math.hypot(current['x']-a[0]-fraction*dx,current['y']-a[1]-fraction*dy)
                error=max(0.,error-prediction.get('extra_radii',[0.]*25)[horizon])
                counts[f'known_{horizon}']+=1;maximum=max(maximum,error)
                if error>.002:
                    counts[f'outside_{horizon}']+=1
                    if len(bad)<12:bad.append(dict(frame=frame-horizon,horizon=horizon,raw=raw,
                        actual=current,bounds=[a,b],error=error))
                if (horizon==4 and raw['native_state']==2 and current['active']
                        and math.hypot(*raw['native_velocity'])<1e-8 and len(examples)<6):
                    examples.append(dict(frame=frame-horizon,raw=raw,after4=current,
                        bounds_after4=[a,b],error=error))
        predictions=[]
        for raw in row['growth']:
            prediction=predict_bullet_release(raw)
            modes[f"state{raw['native_state']}_ex{raw.get('native_ex_active_flags')}_{prediction['mode']}_{prediction['reason']}"]+=1
            predictions.append((raw,prediction))
        history[(row['episode'],frame)]=predictions
        history={key:value for key,value in history.items() if key[0]==row['episode'] and key[1]>=frame-24}
    return dict(schema='native-bullet-release-motion-audit-v1',source=str(path.resolve()),
        rows=rows,last_native_frame=last,checks=dict(counts),modes=dict(modes),max_error_px=maximum,
        outside_tolerance_px=.002,contradictions=bad,zero_velocity_release_examples=examples,
        scope='Native same-ID/age motion envelope verification; not collision truth or fly-policy evaluation')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('capture',type=Path);parser.add_argument('--output',type=Path,required=True)
    options=parser.parse_args()
    import psutil
    process=psutil.Process()
    if os.name=='nt':process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[-2:])
    result=audit(options.capture)
    options.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('rows','last_native_frame','checks','max_error_px','modes')},ensure_ascii=False))
