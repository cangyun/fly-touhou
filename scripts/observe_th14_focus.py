"""Observe existing foreground/focus events; never launch or activate a window.

Stores HWND/PID/event timing only, never titles, text, screenshots or input.
An empty observation interval cannot prove future launches are focus-safe.
"""
import argparse
import ctypes
from ctypes import wintypes as W
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import psutil
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.thrl_study_io import utc_now
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--seconds',type=float,default=45.)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 0<args.seconds<=300:raise ValueError('Observation must be between 0 and 300 seconds')
    if args.output.exists():raise FileExistsError('Preserve previous observations')
    args.output.mkdir(parents=True)
    user=ctypes.WinDLL('user32',use_last_error=True)
    class GUI(ctypes.Structure):
        _fields_=[('cbSize',W.DWORD),('flags',W.DWORD),('hwndActive',W.HWND),
            ('hwndFocus',W.HWND),('hwndCapture',W.HWND),('hwndMenuOwner',W.HWND),
            ('hwndMoveSize',W.HWND),('hwndCaret',W.HWND),('rcCaret',W.RECT)]
    callback_type=ctypes.WINFUNCTYPE(None,W.HANDLE,W.DWORD,W.HWND,W.LONG,W.LONG,W.DWORD,W.DWORD)
    user.SetWinEventHook.argtypes=[W.DWORD,W.DWORD,W.HMODULE,callback_type,W.DWORD,W.DWORD,W.DWORD]
    user.SetWinEventHook.restype=W.HANDLE
    user.UnhookWinEvent.argtypes=[W.HANDLE];user.UnhookWinEvent.restype=W.BOOL
    user.GetForegroundWindow.restype=W.HWND
    user.GetWindowThreadProcessId.argtypes=[W.HWND,ctypes.POINTER(W.DWORD)]
    user.GetWindowThreadProcessId.restype=W.DWORD
    user.GetGUIThreadInfo.argtypes=[W.DWORD,ctypes.POINTER(GUI)];user.GetGUIThreadInfo.restype=W.BOOL
    user.PeekMessageW.argtypes=[ctypes.POINTER(W.MSG),W.HWND,W.UINT,W.UINT,W.UINT]
    user.PeekMessageW.restype=W.BOOL
    user.TranslateMessage.argtypes=[ctypes.POINTER(W.MSG)]
    user.DispatchMessageW.argtypes=[ctypes.POINTER(W.MSG)];user.DispatchMessageW.restype=ctypes.c_ssize_t
    started=time.monotonic();start_utc=utc_now();events=[];transitions=[];native={};last=None

    def owner(hwnd):
        pid=W.DWORD()
        if hwnd:user.GetWindowThreadProcessId(hwnd,ctypes.byref(pid))
        return dict(hwnd=int(hwnd or 0),pid=int(pid.value))

    def sample():
        state=GUI();state.cbSize=ctypes.sizeof(state)
        value=dict(foreground=owner(user.GetForegroundWindow()))
        if user.GetGUIThreadInfo(0,ctypes.byref(state)):
            value.update(focus=owner(state.hwndFocus),active=owner(state.hwndActive),capture=owner(state.hwndCapture))
        return value

    @callback_type
    def event_callback(_hook,event,hwnd,object_id,child_id,thread,event_tick):
        events.append(dict(elapsed=time.monotonic()-started,event=int(event),target=owner(hwnd),
            object_id=int(object_id),child_id=int(child_id),thread=int(thread),event_tick=int(event_tick),
            foreground=owner(user.GetForegroundWindow())))

    def native_processes():
        for p in psutil.process_iter(['pid','name','create_time']):
            if (p.info['name'] or '').lower()=='th14.exe':
                key=str(p.info['pid'])+':'+str(p.info['create_time'])
                native.setdefault(key,{**p.info,'first_seen_seconds':time.monotonic()-started})

    hooks=[]
    try:
        for event in (3,0x8005):
            hook=user.SetWinEventHook(event,event,None,event_callback,0,0,0)
            if not hook:raise ctypes.WinError(ctypes.get_last_error())
            hooks.append(hook)
        native_processes();next_scan=started+.5;message=W.MSG()
        while time.monotonic()-started<args.seconds:
            while user.PeekMessageW(ctypes.byref(message),None,0,0,1):
                user.TranslateMessage(ctypes.byref(message));user.DispatchMessageW(ctypes.byref(message))
            current=sample()
            if current!=last:
                transitions.append(dict(elapsed=time.monotonic()-started,**current));last=current
            if time.monotonic()>=next_scan:
                native_processes();next_scan=time.monotonic()+.5
            time.sleep(.01)
    finally:
        for hook in hooks:user.UnhookWinEvent(hook)
    pids={row['pid'] for row in native.values()}
    related=[e for e in events if e['target']['pid'] in pids or e['foreground']['pid'] in pids]
    captured=[e for e in transitions if any(v.get('pid') in pids for v in e.values() if isinstance(v,dict))]
    write(args.output/'events.json',dict(events=events,focus_transitions=transitions))
    result=dict(status='observed',started_utc=start_utc,ended_utc=utc_now(),elapsed_seconds=time.monotonic()-started,
        existing_native_processes=list(native.values()),foreground_or_focus_events=len(events),
        native_related_events=related,native_related_sample_transitions=captured,
        new_native_processes_seen=sum(row['first_seen_seconds']>.5 for row in native.values()),
        observer_started_games=0,observer_sent_input=False,observer_activated_windows=False,
        observed_window_titles_or_content=False,
        limitation='Passive finite interval; no occurrence does not prove future startup safety; PID reuse is not causal proof')
    write(args.output/'report.json',result)
    print(json.dumps({k:result[k] for k in ('status','elapsed_seconds','new_native_processes_seen','native_related_events')},ensure_ascii=False))


if __name__=='__main__':main()
