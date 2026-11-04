"""Read a captured paused original-game memory image without a game process."""
from pathlib import Path
import gzip
import pickle
import struct


class CapturedMemory:
    def __init__(self,frame,folder):
        self.base=0x400000
        self.dynamic={int(address):data for address,data in frame['blocks'].items()}
        self.program={}
        for filename in frame.get('programs',()):
            with gzip.open(Path(folder)/filename,'rb') as stream:program=pickle.load(stream)
            self.program.update(program['pages']);self.program.update(program['blocks'])

    def read(self,address,size):
        if size<0:raise ValueError('Negative memory read')
        result=[];end=address+size
        while address<end:
            found=None
            # Same-frame context/body captures take priority over the static
            # ECL heap pages retained earlier in this same game.
            for store in (self.dynamic,self.program):
                candidates=[(start,data) for start,data in store.items() if start<=address<start+len(data)]
                if candidates:
                    found=min(candidates,key=lambda item:len(item[1]));break
            if found is None:raise OSError(f'Address {address:#x} was not captured; no substitute memory is supplied')
            start,data=found;take=min(end-address,start+len(data)-address)
            result.append(data[address-start:address-start+take]);address+=take
        return b''.join(result)

    def integer(self,address,size=4,signed=False):return int.from_bytes(self.read(address,size),'little',signed=signed)
    def floating(self,address):return struct.unpack('<f',self.read(address,4))[0]
    def pair(self,address):return struct.unpack('<ff',self.read(address,8))


def frames(folder):
    for path in sorted(Path(folder).glob('native-frames-*.pkl.gz')):
        with gzip.open(path,'rb') as stream:
            yield from pickle.load(stream)
