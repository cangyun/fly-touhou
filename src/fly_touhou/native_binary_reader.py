"""Owned-checkpoint binary transport and exact, short-lived memory prefetch.

Only the dense bullet list uses CModule. Player/enemy/laser/items/events/ECL
remain existing verified readers. This is intentionally not called a full
native sensor port. Cache has one checkpoint lifetime, never only a frame key.
"""
from bisect import bisect_right
from pathlib import Path
import queue
import struct
import time

from .native import NativeError, ProcessMemory


def decode_blocks(payload):
    if not isinstance(payload, bytes) or len(payload)>2097152:
        raise ValueError('Invalid bounded native binary payload')
    at=0;result={}
    while at<len(payload):
        if at+8>len(payload):raise ValueError('Truncated native block header')
        address,size=struct.unpack_from('<II',payload,at);at+=8
        if not 0x10000<=address<0x100000000 or not 0<size<=4096 or at+size>len(payload):
            raise ValueError('Invalid native block range')
        block=payload[at:at+size];at+=size
        if address in result and result[address]!=block:
            raise ValueError('Conflicting native blocks')
        result[address]=block
    starts=sorted(result)
    if any(a+len(result[a])>b for a,b in zip(starts,starts[1:])):
        raise ValueError('Overlapping native blocks')
    return result


class PrefetchedProcessMemory(ProcessMemory):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.invalidate_prefetch()

    def invalidate_prefetch(self):
        self._prefetch={};self._prefetch_starts=[];self.prefetch_epoch=None
        self.prefetch_hits=0;self.prefetch_misses=0

    def set_prefetch(self,blocks,epoch):
        self._prefetch=dict(blocks);self._prefetch_starts=sorted(blocks);self.prefetch_epoch=epoch

    def read(self,address,size):
        index=bisect_right(self._prefetch_starts,address)-1
        if index>=0:
            start=self._prefetch_starts[index];raw=self._prefetch[start]
            if address+size<=start+len(raw):
                self.prefetch_hits+=1
                return raw[address-start:address-start+size]
        self.prefetch_misses+=1
        return super().read(address,size)


class NativeBinaryReader:
    def __init__(self,controller):
        self.controller=controller;self.sequence=0;self.events=queue.Queue();self.closed=False
        self.script=controller._session.create_script((Path(__file__).parent/'hooks/th14_binary_reader.js').read_text(encoding='utf-8'))
        self.script.on('message',self._message)
        self.script.load()
        self._wait('binary-ready',None)
        self.last_audit=None

    def _message(self,message,data):
        if self.closed:return
        payload=message.get('payload',{}) if message.get('type')=='send' else {
            'type':'binary-error','detail':message.get('stack',str(message))}
        self.events.put((payload,data))

    def _wait(self,kind,request):
        deadline=time.monotonic()+self.controller.timeout
        while True:
            try:payload,data=self.events.get(timeout=max(.001,deadline-time.monotonic()))
            except queue.Empty:raise NativeError('Compiled native reader deadline expired') from None
            if payload.get('type')=='binary-error':raise NativeError(payload['detail'])
            if payload.get('type')==kind and (request is None or payload.get('request_id')==request):
                return payload,data
            if time.monotonic()>deadline:raise NativeError('Compiled native reader protocol mismatch')

    def prepare(self):
        c=self.controller;epoch=(c.pid,c._native_callback_generation,c._checkpoint['sequence'])
        if c._memory.prefetch_epoch==epoch:return
        c._memory.invalidate_prefetch();self.sequence+=1
        self.script.post({'type':'binary-read','payload':{'request_id':self.sequence,'checkpoint':epoch[2]}})
        reply,data=self._wait('binary-reply',self.sequence)
        if reply.get('pid')!=c.pid or reply.get('checkpoint')!=epoch[2] or reply.get('bytes')!=len(data):
            raise NativeError('Compiled native read belongs to another process/checkpoint')
        blocks=decode_blocks(bytes(data))
        if reply['records']!=len(blocks):raise NativeError('Compiled native block count mismatch')
        c._memory.set_prefetch(blocks,epoch)
        self.last_audit={'schema':'th14-bullet-binary-reader-v1','compiled':True,'bytes':len(data),
            'blocks':len(blocks),'epoch':list(epoch),'coverage':'bullet/list fields; other entities use existing reader'}

    def close(self):
        self.closed=True
        try:self.script.unload()
        except Exception:pass
