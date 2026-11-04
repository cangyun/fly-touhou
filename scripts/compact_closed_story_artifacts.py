"""Lossless NTFS compression of closed,owned training artifacts;never delete."""
from pathlib import Path
import argparse,ctypes,hashlib,json,os,stat,subprocess,time,zipfile
import psutil
from fly_touhou.thrl_study_io import read_shared_json
from fly_touhou.thrl_durable_records import write_json


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def closed(folder):
    launch=folder/'launch.json'
    if not launch.exists():return False
    d=read_shared_json(launch);pid=d.get('PID',d.get('pid'));created=d.get('create_time')
    if not pid or created is None:return False
    try:p=psutil.Process(pid);return abs(p.create_time()-created)>.01
    except psutil.NoSuchProcess:return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--target-saved-gib',type=float,default=4);p.add_argument('--max-files',type=int,default=40)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    root=a.root.resolve();files=[]
    for f in root.glob('*/*.pt'):
        q=f.resolve();s=q.stat()
        if (not q.is_relative_to(root) or s.st_size<200*2**20 or s.st_nlink!=1
                or s.st_file_attributes&stat.FILE_ATTRIBUTE_COMPRESSED or not closed(q.parent)):continue
        if q.name=='rollout-001.pt':continue  # Existing critical replay source.
        try:
            with zipfile.ZipFile(q) as z:
                large=[i for i in z.infolist() if i.file_size>2**20]
                if large and any(i.compress_type!=zipfile.ZIP_STORED for i in large):continue
        except zipfile.BadZipFile:pass
        files.append((q,s.st_size,s.st_mtime_ns))
    files=sorted(files,key=lambda x:x[1],reverse=True)[:a.max_files]
    if not a.execute:print(json.dumps(dict(candidates=[str(x[0]) for x in files])));return
    if os.name!='nt':raise RuntimeError('NTFS compaction requires Windows')
    proc=psutil.Process();proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'launch.json',dict(PID=proc.pid,create_time=proc.create_time(),root=str(root),
        operation='NTFS compression only;no deletion/move;SHA256 before/after every file',GPU_used=False,native_actions=0))
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    size=kernel.GetCompressedFileSizeW;size.argtypes=(ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_uint32));size.restype=ctypes.c_uint32
    def allocated(path):
        high=ctypes.c_uint32();ctypes.set_last_error(0);low=size(str(path),ctypes.byref(high))
        if low==0xffffffff and ctypes.get_last_error():raise ctypes.WinError(ctypes.get_last_error())
        return (high.value<<32)|low
    results=[];saved=0;begun=time.monotonic()
    for f,logical,mtime in files:
        if not closed(f.parent) or f.stat().st_mtime_ns!=mtime:continue
        before=allocated(f);digest=sha(f)
        result=subprocess.run(['compact.exe','/C','/I','/Q',str(f)],text=True,capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        if sha(f)!=digest:raise IOError('Artifact content changed:'+str(f))
        after=allocated(f);delta=before-after;saved+=delta
        results.append(dict(path=str(f),logical_bytes=logical,before_allocated=before,after_allocated=after,
            saved_bytes=delta,SHA256=digest,content_verified_unchanged=True,returncode=result.returncode))
        write_json(a.output/'progress.json',dict(completed_files=len(results),saved_bytes=saved,
            elapsed_seconds=time.monotonic()-begun,files=results))
        print(json.dumps(dict(completed_files=len(results),saved_GiB=round(saved/2**30,3))),flush=True)
        if saved>=a.target_saved_gib*2**30:break
    write_json(a.output/'result.json',dict(status='completed_lossless_NTFS_compaction',saved_bytes=saved,files=results,
        elapsed_seconds=time.monotonic()-begun,no_files_deleted=True,all_processed_contents_verified_unchanged=True))


if __name__=='__main__':main()
