"""Offline exact-byte item-reader parity and memory-copy microbenchmark.

The RPM fixture copies a pinned-size buffer; it is not native wall-time or a
measurement of Windows ReadProcessMemory. No executable/world/CUDA is opened.
"""
from pathlib import Path
import argparse,ctypes as C,hashlib,importlib.util,json,statistics,struct,sys,time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.native import ProcessMemory
from fly_touhou.resources import ResourceNativeReader
from fly_touhou.atomic_json_io import atomic_write_json as write


def run(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    old_path=a.baseline_runtime/'src/fly_touhou/resources.py'
    spec=importlib.util.spec_from_file_location('fly_touhou._sealed_resource_baseline',old_path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    profile=json.loads((ROOT/'profiles/th14_1.00b_original.json').read_text(encoding='utf-8'))
    offsets=profile['offsets']['items'];length=offsets['zItemManager_array_len'];stride=offsets['zItem_len']
    size=length*stride;storage=C.create_string_buffer(size);count=0;calls=0
    def rpm(handle,address,destination,n,got):
        nonlocal calls
        if address!=0x10000+offsets['zItemManager_array'] or n!=size:raise AssertionError('Unexpected fixture read')
        calls+=1;C.memmove(destination,storage,n);C.cast(got,C.POINTER(C.c_size_t))[0]=n;return True
    def make(cls):
        m=object.__new__(ProcessMemory);m.handle=1;m.k=SimpleNamespace(ReadProcessMemory=rpm)
        m.integer=lambda *_:count
        r=object.__new__(cls);r.o={'items':offsets};r.mem=m;r.pointer=lambda *_:0x10000
        return r
    baseline=make(old.ResourceNativeReader);candidate=make(ResourceNativeReader);rows=[]
    for active in [1,64,512,4096]:
        C.memset(storage,0,size);count=active
        for index in range(active):
            at=index*stride
            struct.pack_into('<II',storage,at+offsets['zItem_state'],[1,3,4][index%3],1 if index%11 else 99)
            struct.pack_into('<ff',storage,at+offsets['zItem_pos'],float(index%128),float(index%384))
            struct.pack_into('<ff',storage,at+offsets['zItem_vel'],.5,-.25)
            struct.pack_into('<i',storage,at+offsets['zItem_timer'],index)
        expected=baseline._resource_items();actual=candidate._resource_items()
        if actual!=expected:raise AssertionError('Frozen reader extraction differs')
        storage_identity=id(candidate._item_read_buffer);times={'baseline':[],'candidate':[]}
        for repeat in range(15):
            for name,r in ([('baseline',baseline),('candidate',candidate)] if repeat%2 else [('candidate',candidate),('baseline',baseline)]):
                start=time.perf_counter();value=r._resource_items();elapsed=time.perf_counter()-start
                if value!=expected:raise AssertionError('Repeated extraction differs')
                times[name].append(elapsed)
        if id(candidate._item_read_buffer)!=storage_identity:raise AssertionError('Candidate storage not reused')
        row=dict(active_slots=active,decoded_known_items=len(actual[0]),results_exact=True,
                 median_seconds={k:statistics.median(v) for k,v in times.items()})
        row['memory_fixture_speed_ratio']=row['median_seconds']['baseline']/row['median_seconds']['candidate'];rows.append(row)
    receipts=[]
    for p in [old_path,ROOT/'src/fly_touhou/resources.py',ROOT/'src/fly_touhou/native.py',Path(__file__)]:
        receipts.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    result=dict(schema='TH14-item-buffer-storage-offline-qualification-v1',status='qualified-offline-only',
        table_slots=length,table_stride=stride,table_bytes=size,results=rows,read_calls=calls,
        native_worlds=0,CUDA_initialized=False,model_actions_reward_unchanged=True,
        original_Windows_read_parity_measured=False,native_throughput_speedup_measured=False,
        limitation='Memmove fixture isolates allocation/copy/decoding overhead;does not measure native game throughput',sources=receipts)
    write(out/'report.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--baseline-runtime',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
