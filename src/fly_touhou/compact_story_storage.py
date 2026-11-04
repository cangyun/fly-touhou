"""Lossless compression of ordinary torch ZIP checkpoints, no tensor casting."""
from pathlib import Path
import os
import shutil
import time
import zipfile


def save_compressed_torch_records(value,path):
    """Stream standard torch ZIP records to DEFLATE without an uncompressed tape.

    The installed torch serializer still supplies all pickle/storage metadata.
    This adapter only replaces its archive writer; ordinary torch.load and
    selective observation reads retain the same file format.
    """
    import pickle
    import torch
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    temporary=path.with_suffix(path.suffix+'.incomplete');begun=time.monotonic()
    class Records:
        source_bytes=0
        entries=0
        def __init__(self,archive):self.archive=archive
        def write_record(self,name,data,size):
            if isinstance(data,str):data=data.encode('utf-8')
            if torch.is_storage(data):
                if data.device.type!='cpu':raise ValueError('Torch serializer must transfer storage to CPU')
                data=memoryview(torch.empty(0,dtype=torch.uint8).set_(data,0,(size,),(1,)).numpy())
            else:data=memoryview(data)
            if data.nbytes!=size:raise ValueError('Incorrect torch record byte count')
            with self.archive.open('archive/'+name,'w',force_zip64=True) as writer:
                for offset in range(0,size,4*1024*1024):writer.write(data[offset:offset+4*1024*1024])
            self.source_bytes+=size;self.entries+=1
    with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        writer=Records(archive)
        torch.serialization._save(value,writer,pickle,torch.serialization.DEFAULT_PROTOCOL,False)
        writer.write_record('version',b'3\n',2)
    os.replace(temporary,path)
    return dict(source_bytes=writer.source_bytes,compressed_bytes=path.stat().st_size,
        compression_seconds=time.monotonic()-begun,tensor_dtypes_changed=False,
        archive_records=writer.entries,entry_CRC_recorded=True,
        compression_mode='direct_streamed_torch_records',uncompressed_temporary_bytes=0)


def _restore_byte_plane_spiking_state(shape,planes):
    """Inverse byte permutation; actual returned states remain ordinary FP32."""
    import torch
    from .spiking_engineered_control import SpikingState
    B,N=shape
    return SpikingState(*(p.permute(1,2,0).contiguous().view(torch.float32).reshape(N,B).T.contiguous() for p in planes))


class _BytePlaneSpikingState:
    def __init__(self,state):
        import torch
        self.shape=tuple(state.membrane.shape);B,N=self.shape
        if any(v.device.type!='cpu' or v.dtype!=torch.float32 or tuple(v.shape)!=self.shape for v in state.tensors()):
            raise ValueError('Lossless state packing expects the existing CPU FP32 brain states')
        # Group the same IEEE754 byte across nearby worlds of each neuron.
        # view(uint8) reinterprets bits; it is not a numeric conversion/cast.
        self.planes=tuple(v.T.contiguous().reshape(-1).view(torch.uint8).reshape(N,B,4).permute(2,0,1).contiguous() for v in state.tensors())
    def __reduce__(self):return (_restore_byte_plane_spiking_state,(self.shape,self.planes))


def compress_pt(source,destination):
    source,destination=Path(source),Path(destination)
    if destination.exists():raise FileExistsError(destination)
    temporary=destination.with_suffix(destination.suffix+'.incomplete')
    begun=time.monotonic()
    with zipfile.ZipFile(source,'r') as original,zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,allowZip64=True) as packed:
        names=original.namelist()
        for name in names:
            with original.open(name,'r') as reader,packed.open(name,'w',force_zip64=True) as writer:
                shutil.copyfileobj(reader,writer,length=4*1024*1024)
        checksums={name:(original.getinfo(name).CRC,original.getinfo(name).file_size) for name in names}
    with zipfile.ZipFile(temporary,'r') as packed:
        if packed.namelist()!=names or any((packed.getinfo(n).CRC,packed.getinfo(n).file_size)!=checksums[n] for n in names):
            raise IOError('Lossless torch archive entry content differs')
        if packed.testzip() is not None:raise IOError('Lossless torch archive failed decompression/CRC verification')
    os.replace(temporary,destination)
    return dict(source_bytes=source.stat().st_size,compressed_bytes=destination.stat().st_size,
        entry_bytes_and_CRC_verified=True,tensor_dtypes_changed=False,source_preserved=True,
        compression_seconds=time.monotonic()-begun)


def save_rollout(rollout,path,*,compressed=False):
    import torch
    path=Path(path)
    if not compressed:
        with path.open('xb') as f:torch.save(rollout,f)
        return dict(compressed=False,bytes=path.stat().st_size,tensor_dtypes_changed=False)
    packed=dict(rollout)
    if 'block_states' in rollout:
        from .compact_story_ready_data import IndexedSpikingStates
        packed['block_states']={t:(IndexedSpikingStates(s.worlds,_BytePlaneSpikingState(s.state))
            if isinstance(s,IndexedSpikingStates) else _BytePlaneSpikingState(s))
            for t,s in rollout['block_states'].items()}
    result=save_compressed_torch_records(packed,path)
    return dict(result,compressed=True,source_preserved=False,source_was_new_temporary=False,
        brain_storage='lossless IEEE754 byte planes;torch.load restores original contiguous FP32 SpikingState',
        quantization_or_brain_states_removed=False)
