"""Inventory an explicit old build directory;only prepare a deletion manifest."""
from pathlib import Path
import collections
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
import re
import time


def main():
    root=Path('./outputs/legacy-build/experiments').resolve(strict=True)
    out=Path(__file__).resolve().parents[1]/'artifacts/engineering/CodexBuild-cleanup-20261025-v1'
    out.mkdir(parents=True,exist_ok=True)
    plan_path=out/'cleanup_plan.json'
    if plan_path.exists():raise FileExistsError('Never overwrite a reviewed cleanup plan')
    cutoff=time.mktime((2026,10,21,0,0,0,0,0,-1))
    critical=re.compile(r'best|final|initial|latest|parent|baseline|selected|failed|proof|qualif|evidence',re.I)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCompressedFileSizeW.argtypes=[wintypes.LPCWSTR,ctypes.POINTER(wintypes.DWORD)]
    kernel.GetCompressedFileSizeW.restype=wintypes.DWORD
    def allocated(path):
        high=wintypes.DWORD();ctypes.set_last_error(0)
        low=kernel.GetCompressedFileSizeW(str(path),ctypes.byref(high))
        if low==0xffffffff and ctypes.get_last_error():raise ctypes.WinError(ctypes.get_last_error())
        return (high.value<<32)|low
    delete=[];preserve=[];groups=[];all_files=[]
    for folder,dirs,files in os.walk(root,followlinks=False):
        parent=Path(folder)
        if parent.lstat().st_file_attributes&0x400:raise RuntimeError('Unexpected reparse parent')
        for name in dirs:
            if (parent/name).lstat().st_file_attributes&0x400:raise RuntimeError('Unexpected reparse subtree')
        streams=collections.defaultdict(list)
        for name in files:
            path=parent/name;st=path.lstat()
            if st.st_file_attributes&0x400:raise RuntimeError('Unexpected reparse file')
            record=dict(path=str(path),relative=str(path.relative_to(root)),bytes=st.st_size,
                mtime_ns=st.st_mtime_ns,mtime_filetime=st.st_mtime_ns//100+116444736000000000,
                mtime_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(st.st_mtime)))
            all_files.append(record)
            state=parent.name=='rollouts' and re.fullmatch(r'\d{6}\.states\.pt(?:\.gz)?',name)
            checkpoint=parent.name in ('checkpoints','epochs') and re.fullmatch(
                r'(?:epoch[_-]\d+.*|checkpoint[_-]\d+.*|\d{6})\.pt',name)
            if not (state or checkpoint) or st.st_mtime>=cutoff or critical.search(name):
                preserve.append(record);continue
            stream='state-tensors' if state else re.sub(r'(?i)[0-9a-f]{12,64}','HASH',
                re.sub(r'(?i)^(epoch|checkpoint)[_-]\d+',r'\1_N',name))
            record=dict(record,kind='obsolete_neuron_state_snapshot' if state else 'obsolete_intermediate_checkpoint')
            streams[stream].append(record)
        for stream,records in streams.items():
            records.sort(key=lambda x:(x['mtime_ns'],x['path']))
            keep={r['path'] for r in records[:1]+records[-2:]}
            for record in records:
                if record['path'] in keep:preserve.append(record)
                else:delete.append(dict(record,allocated_bytes=allocated(Path(record['path']))))
            groups.append(dict(folder=str(parent),stream=stream,candidates=len(records),
                retained_first_and_last_two=len(keep),delete_count=len(records)-len(keep)))
    plan=dict(schema='CodexBuild-user-authorized-expired-artifact-cleanup-v1',
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),root=str(root),
        authorized_parent='./outputs/legacy-build',
        rules='Only numbered old rollout state tensors and intermediate model checkpoints;keep first and last2 per stream,all critical/root checkpoints,and every source,log,native trajectory and tool',
        delete=delete,preserve=preserve,groups=groups,total_files_before=len(all_files),
        total_bytes_before=sum(x['bytes'] for x in all_files),expected_deleted_files=len(delete),
        expected_deleted_bytes=sum(x['bytes'] for x in delete),
        expected_deleted_allocated_bytes=sum(x['allocated_bytes'] for x in delete),
        retained_bytes=sum(x['bytes'] for x in preserve),current_tests_not_modified=True)
    plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(plan=str(plan_path),plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        delete_files=len(delete),delete_logical_GiB=round(plan['expected_deleted_bytes']/2**30,3),
        expected_freed_allocated_GiB=round(plan['expected_deleted_allocated_bytes']/2**30,3),
        retained_logical_GiB=round(plan['retained_bytes']/2**30,3),retained_files=len(preserve),
        by_kind={key:dict(files=sum(x['kind']==key for x in delete),
            logical_GiB=round(sum(x['bytes'] for x in delete if x['kind']==key)/2**30,3),
            allocated_GiB=round(sum(x['allocated_bytes'] for x in delete if x['kind']==key)/2**30,3))
            for key in sorted({x['kind'] for x in delete})}),ensure_ascii=False))


if __name__=='__main__':main()
