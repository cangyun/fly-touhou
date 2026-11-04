"""Original signed CSR matrix times activity with a fixed FP32 sum order."""
from pathlib import Path
import ctypes as C
import os
import torch
from .fused_edge_gradient import _Kernel,compile_ptx,_checked

SOURCE=Path(__file__).parent/'csrc/deterministic_csr_spmm.cu'
_kernels={}


class _CSRKernel(_Kernel):
    def __init__(self,ptx):super().__init__(ptx,b'deterministic_csr_spmm')
    def __call__(self,crow,columns,weights,activity,out):
        N,B=activity.shape
        values=[C.c_void_p(t.data_ptr()) for t in (crow,columns,weights,activity,out)]+[C.c_int(B),C.c_int(N)]
        args=(C.c_void_p*len(values))(*(C.cast(C.pointer(v),C.c_void_p) for v in values))
        stream=C.c_void_p(torch.cuda.current_stream(activity.device).cuda_stream)
        _checked(self.launch(self.handle,N,(B+31)//32,1,128,1,1,0,stream,args,None),'cuLaunchKernel fixed-order CSR')


def multiply(activity,values,crow,columns):
    if not activity.is_cuda or activity.dtype!=torch.float32 or values.dtype!=torch.float32:
        raise ValueError('CUDA FP32 activity and original signed FP32 edge values required')
    with torch.cuda.device(activity.device):
        device=activity.device.index
        if device not in _kernels:
            cache=Path(os.environ.get('FLY_TOUHOU_SPMM_CACHE',str(SOURCE.parents[3]/'artifacts/runtime-kernels/deterministic-spmm')))
            _kernels[device]=_CSRKernel(compile_ptx(cache,source_path=SOURCE))
        transposed=activity.T.contiguous();out=torch.empty_like(transposed)
        _kernels[device](crow,columns,values,transposed,out)
        return out.T.contiguous()
