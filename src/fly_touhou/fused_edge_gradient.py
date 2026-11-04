"""Optional first-order CUDA edge gradient, compiled with Torch's NVRTC.

Compilation is CPU-only. Driver/module loading is lazy and happens only when
the explicitly enabled CUDA backward calls edge_gradient. No package install,
matrix approximation, dropped edge or reduced neural precision is involved.
"""
from pathlib import Path
import ctypes as C
import hashlib,json,os
import torch

SOURCE=Path(__file__).parent/'csrc/fused_edge_gradient.cu'
_modules={}


def _checked(code,name):
    if code:raise RuntimeError(name+' failed with CUDA/NVRTC status '+str(code))


def compile_ptx(directory,source_path=None):
    """Use the installed NVRTC; this function does not initialize CUDA."""
    libdir=Path(torch.__file__).parent/'lib'
    libraries=sorted(p for p in libdir.glob('nvrtc64_*.dll') if '.alt.' not in p.name)
    if not libraries:raise RuntimeError('Bundled NVRTC DLL unavailable')
    dll_directory=os.add_dll_directory(str(libdir))
    try:
        nv=C.CDLL(str(libraries[0]))
        def bind(name,args):
            f=getattr(nv,name);f.argtypes=args;f.restype=C.c_int;return f
        version=bind('nvrtcVersion',[C.POINTER(C.c_int),C.POINTER(C.c_int)])
        major=C.c_int();minor=C.c_int();_checked(version(C.byref(major),C.byref(minor)),'nvrtcVersion')
        source_path=SOURCE if source_path is None else Path(source_path)
        source=source_path.read_bytes();options=[b'--gpu-architecture=compute_75',b'--std=c++14',b'--fmad=false']
        key=hashlib.sha256(source+b'|'.join(options)+f'{major.value}.{minor.value}'.encode()).hexdigest()
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);path=directory/(key+'.ptx')
        if path.exists():return path
        create=bind('nvrtcCreateProgram',[C.POINTER(C.c_void_p),C.c_char_p,C.c_char_p,C.c_int,C.c_void_p,C.c_void_p])
        compile_=bind('nvrtcCompileProgram',[C.c_void_p,C.c_int,C.POINTER(C.c_char_p)])
        logsize=bind('nvrtcGetProgramLogSize',[C.c_void_p,C.POINTER(C.c_size_t)])
        log=bind('nvrtcGetProgramLog',[C.c_void_p,C.c_void_p])
        ptxsize=bind('nvrtcGetPTXSize',[C.c_void_p,C.POINTER(C.c_size_t)])
        ptx=bind('nvrtcGetPTX',[C.c_void_p,C.c_void_p])
        destroy=bind('nvrtcDestroyProgram',[C.POINTER(C.c_void_p)])
        program=C.c_void_p();_checked(create(C.byref(program),source,source_path.name.encode(),0,None,None),'nvrtcCreateProgram')
        try:
            result=compile_(program,len(options),(C.c_char_p*len(options))(*options))
            if result:
                size=C.c_size_t();_checked(logsize(program,C.byref(size)),'nvrtcGetProgramLogSize')
                text=C.create_string_buffer(size.value);_checked(log(program,text),'nvrtcGetProgramLog')
                raise RuntimeError('NVRTC compilation failed: '+text.value.decode(errors='replace'))
            size=C.c_size_t();_checked(ptxsize(program,C.byref(size)),'nvrtcGetPTXSize')
            data=C.create_string_buffer(size.value);_checked(ptx(program,data),'nvrtcGetPTX')
            temp=path.with_suffix('.ptx.tmp');temp.write_bytes(data.raw);os.replace(temp,path)
            path.with_suffix('.json').write_text(json.dumps(dict(source_SHA256=hashlib.sha256(source).hexdigest(),
                compiler=str(libraries[0]),NVRTC=f'{major.value}.{minor.value}',architecture='compute_75',
                options=[s.decode() for s in options],CUDA_initialized_by_compilation=False)),encoding='utf8')
            return path
        finally:destroy(C.byref(program))
    finally:dll_directory.close()


class _Kernel:
    def __init__(self,ptx,entrypoint=b'fused_edge_gradient'):
        driver=C.WinDLL('nvcuda.dll')
        def bind(name,args):
            f=getattr(driver,name);f.argtypes=args;f.restype=C.c_int;return f
        self.load=bind('cuModuleLoadData',[C.POINTER(C.c_void_p),C.c_void_p])
        self.function=bind('cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p])
        self.launch=bind('cuLaunchKernel',[C.c_void_p,*([C.c_uint]*7),C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p])
        self.module=C.c_void_p();data=C.create_string_buffer(Path(ptx).read_bytes())
        _checked(self.load(C.byref(self.module),data),'cuModuleLoadData')
        self.handle=C.c_void_p();_checked(self.function(C.byref(self.handle),self.module,entrypoint),'cuModuleGetFunction')
        self.driver=driver
    def __call__(self,a,g,pre,post,out):
        B=a.shape[1];E=len(pre)
        values=[C.c_void_p(t.data_ptr()) for t in (a,g,pre,post,out)]+[C.c_int(B),C.c_longlong(E)]
        args=(C.c_void_p*len(values))(*(C.cast(C.pointer(v),C.c_void_p) for v in values))
        stream=C.c_void_p(torch.cuda.current_stream(a.device).cuda_stream)
        _checked(self.launch(self.handle,(E+7)//8,1,1,256,1,1,0,stream,args,None),'cuLaunchKernel')


def edge_gradient(activity,gradient,pre,post):
    if not activity.is_cuda or activity.dtype!=torch.float32 or gradient.shape!=activity.shape:
        raise ValueError('CUDA FP32 activity/gradient[B,N] required')
    with torch.cuda.device(activity.device):
        device=activity.device.index
        if device not in _modules:
            directory=Path(os.environ.get('FLY_TOUHOU_EDGE_GRAD_CACHE',str(SOURCE.parents[3]/'artifacts/runtime-kernels/fused-edge-gradient')))
            _modules[device]=_Kernel(compile_ptx(directory))
        a=activity.T.contiguous();g=gradient.T.contiguous();out=activity.new_empty(len(pre))
        _modules[device](a,g,pre,post,out)
        return out
