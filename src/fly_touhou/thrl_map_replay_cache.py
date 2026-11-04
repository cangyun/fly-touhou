"""Bounded-memory world-major map cache; no original arrays are modified."""
from pathlib import Path
import os
import math
import hashlib
import numpy as np


def cache_layout(maps_shape,dtype):
    frames,worlds=maps_shape[:2]
    shape=(frames*worlds,*maps_shape[2:])
    # NumPy1.x on Windows reduces platform integers in32 bits. A full map
    # cache exceeds that range; byte accounting must use Python integers.
    return shape,math.prod(shape)*np.dtype(dtype).itemsize


def open_verified_cache(path,maps_shape,*,dtype,sha256):
    path=Path(path);shape,expected=cache_layout(maps_shape,dtype)
    if path.stat().st_size!=expected:raise ValueError('Existing map cache size differs')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!=sha256:raise ValueError('Existing map cache identity differs')
    return np.memmap(path,dtype=dtype,mode='r+',shape=shape)


def build_map_cache(maps,path,*,dtype=np.float16,chunk_frames=64,on_progress=None,release_source_views=False):
    path=Path(path);frames,worlds=maps.shape[:2]
    if dtype not in (np.float16,np.float32) or type(chunk_frames) is not int or chunk_frames<1:
        raise ValueError('Explicit replay-cache precision and positive bounded chunk required')
    shape,expected=cache_layout(maps.shape,dtype)
    with path.open('xb') as stream:
        for world in range(worlds):
            for first in range(0,frames,chunk_frames):
                block=np.asarray(maps[first:min(frames,first+chunk_frames),world],dtype=dtype)
                block.tofile(stream)
                del block
            stream.flush();os.fsync(stream.fileno())
            if release_source_views:
                if hasattr(maps,'arrays'):
                    pool=int(np.searchsorted(maps.ends,world,side='right'));source=maps.arrays[pool]
                    if isinstance(source,np.memmap):
                        name=source.filename;source._mmap.close();maps.arrays[pool]=np.load(name,mmap_mode='r')
                elif isinstance(maps,np.memmap):
                    name=maps.filename;maps._mmap.close();maps=np.load(name,mmap_mode='r')
            if on_progress:on_progress(dict(worlds_prepared=world+1,worlds=worlds,bytes_written=stream.tell(),target_bytes=expected))
    if path.stat().st_size!=expected:raise ValueError('Incomplete map cache')
    return np.memmap(path,dtype=dtype,mode='r+',shape=shape)


def close_map_sources(maps):
    """Call only after all maps were copied, with no outstanding source views."""
    for a in getattr(maps,'arrays',[maps]):
        handle=getattr(a,'_mmap',None)
        if handle is not None:handle.close()
