"""Losslessly compress only immutable, hash-sealed artifacts of one owned run.

NTFS/WOF compression retains the original pathname and logical bytes, so the
existing experiment hashes/replay readers remain valid. No file is deleted,
moved, quantized or reserialized. The attached process handle prevents PID
reuse from being mistaken for a live learner.
"""
from pathlib import Path
import argparse
import ctypes
from ctypes import wintypes
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.experiment import ExperimentLogger, verify_events, file_sha256, EventIntegrityError


def sealed_artifacts(events,root):
    found={}
    for event in events:
        payload=event['payload']
        if event['event']=='ppo_behavior_rollout_sealed':
            path=Path(payload['path']).resolve()
            allowed=path.parent==root/'rollouts' and path.name.endswith('.states.pt')
            if path.name.endswith('.gz'):
                continue
        elif event['event']=='checkpoint':
            path=Path(payload['path']).resolve()
            allowed=path.parent==root/'checkpoints' and path.suffix=='.pt'
        else:
            continue
        if not allowed:
            raise ValueError(f'Refuse non-immutable or out-of-run compression target: {path}')
        digest=payload['sha256']
        if len(digest)!=64 or any(v not in '0123456789abcdef' for v in digest):
            raise ValueError('Sealed artifact requires a complete SHA256')
        if path in found and found[path]!=digest:
            raise ValueError('Immutable artifact changed identity across events')
        found[path]=digest
    return list(found.items())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--follow-pid',type=int)
    parser.add_argument('--once',action='store_true',help='Compress an already terminal run in one pass')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--interval',type=float,default=10.)
    parser.add_argument('--max-files',type=int)
    args=parser.parse_args()
    if os.name!='nt' or (not args.once and (args.follow_pid is None or args.follow_pid<1)) or args.interval<1:
        parser.error('Requires an explicit live Windows learner PID and interval>=1')
    root=args.run.resolve(strict=True)
    manifest_path=root/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    api.OpenProcess.restype=wintypes.HANDLE
    api.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    api.WaitForSingleObject.restype=wintypes.DWORD
    api.CloseHandle.argtypes=[wintypes.HANDLE]
    api.GetCompressedFileSizeW.argtypes=[wintypes.LPCWSTR,ctypes.POINTER(wintypes.DWORD)]
    api.GetCompressedFileSizeW.restype=wintypes.DWORD
    handle=None if args.once else api.OpenProcess(0x100000,False,args.follow_pid)
    if not args.once and not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    def alive():
        if args.once:
            return False
        result=api.WaitForSingleObject(handle,0)
        if result not in (0,258):
            raise ctypes.WinError(ctypes.get_last_error())
        return result==258
    if not args.once and not alive():
        api.CloseHandle(handle)
        raise ValueError('Follow PID is already terminal; inspect authoritative process state')
    def allocated(path):
        def stream_size(name,optional=False):
            high=wintypes.DWORD()
            ctypes.set_last_error(0)
            low=api.GetCompressedFileSizeW(name,ctypes.byref(high))
            error=ctypes.get_last_error()
            if low==0xffffffff and error:
                if optional and error in (2,3):
                    return 0
                raise ctypes.WinError(error)
            return (int(high.value)<<32)|int(low)
        # Some Windows versions already report WOF payload through the main
        # filename; others expose the backing stream. These are aliases for
        # the same payload, so adding them would double-count occupied bytes.
        return max(stream_size(str(path)),stream_size(str(path)+':WofCompressedData',True))
    compact=Path(os.environ['SystemRoot'])/'System32/compact.exe'
    config={'kind':'lossless-sealed-artifact-NTFS-WOF-compression','algorithm':'LZX',
        'target_run':str(root),'target_run_id':manifest['run_id'],'follow_pid':args.follow_pid,
        'one_pass':args.once,
        'interval':args.interval,'max_files':args.max_files,'desktop_input':False,
        'logical_bytes_or_manifest_paths_changed':False,'artifacts_deleted':False}
    logger=ExperimentLogger(args.output,config,files={'script':Path(__file__),'target_manifest':manifest_path})
    completed=set()
    saved=0
    try:
        while True:
            try:
                events=verify_events(root/'events.jsonl',run_id=manifest['run_id'])
            except EventIntegrityError:
                # A live writer can be observed before its final newline. A
                # terminal or otherwise corrupt chain remains a real error.
                if alive() and not (root/'events.jsonl').read_bytes().endswith(b'\n'):
                    time.sleep(.2)
                    continue
                raise
            for path,digest in sealed_artifacts(events,root):
                if path in completed:
                    continue
                if file_sha256(path)!=digest:
                    raise ValueError(f'Sealed artifact changed before compression: {path}')
                before=allocated(path)
                started=time.monotonic()
                result=subprocess.run([str(compact),'/C','/F','/Q','/EXE:LZX',str(path)],
                    capture_output=True,stdin=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                if result.returncode:
                    raise RuntimeError(f'compact.exe failed for {path}: {result.stderr.decode(errors="replace")}')
                if file_sha256(path)!=digest:
                    raise ValueError(f'Logical bytes changed after compression: {path}')
                after=allocated(path)
                saved+=before-after
                completed.add(path)
                logger.event('sealed_artifact_lossless_compression_verified',{
                    'path':str(path),'sha256_before':digest,'sha256_after':digest,
                    'logical_bytes':path.stat().st_size,'allocated_before':before,'allocated_after':after,
                    'seconds':time.monotonic()-started,'algorithm':'LZX'})
                if len(completed)%10==0 or len(completed)==1:
                    print(json.dumps({'verified_files':len(completed),'saved_bytes':saved}),flush=True)
                if args.max_files is not None and len(completed)>=args.max_files:
                    logger.event('compression_finished',{'reason':'explicit-max-files','verified_files':len(completed),'saved_bytes':saved})
                    return
            if not alive():
                logger.event('compression_finished',{'reason':'one-pass' if args.once else 'followed-process-terminal','verified_files':len(completed),'saved_bytes':saved})
                return
            time.sleep(args.interval)
    except BaseException as error:
        logger.record_error(error,phase='sealed-artifact-compression')
        raise
    finally:
        if handle:
            api.CloseHandle(handle)


if __name__=='__main__':
    main()
