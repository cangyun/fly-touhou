"""Execute original TH14 scalar interpolation bytes on scratch emulated data.

No original-game process is started, stepped or modified. This checks the
scalar implementation, separately from live trajectory/integration evidence.
"""
from pathlib import Path
import argparse
import struct
import sys
import numpy as np
from fly_touhou.native_bullet_scale import scale_path
from fly_touhou.thrl_durable_records import write_json


def verify(exe,output):
    sys.path.insert(0,'./outputs/legacy-build/native-validation-tools')
    import pefile
    from unicorn import Uc,UC_ARCH_X86,UC_MODE_32
    from unicorn.x86_const import UC_X86_REG_ESP,UC_X86_REG_ECX,UC_X86_REG_FPCW
    pe=pefile.PE(str(exe));base=pe.OPTIONAL_HEADER.ImageBase
    emulator=Uc(UC_ARCH_X86,UC_MODE_32)
    emulator.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095)
    emulator.mem_write(base,pe.get_memory_mapped_image())
    scratch=0x10000000;return_address=scratch+0x1000;out=scratch+0x2000
    emulator.mem_map(scratch,0x10000)
    emulator.mem_write(return_address,b'\xf3\x0f\x11\x05'+struct.pack('<I',out)) # movss [out],xmm0
    emulator.reg_write(UC_X86_REG_FPCW,0x27f)
    records=[]
    for mode in range(18):
        for timer in (0,10,30):
            for start,end in ((1.,3.),(3.,.5)):
                interp=dict(dimension=1,start=[start],end=[end],control1=[.2],control2=[.03],
                    current=[start],timer=timer,previous_timer=timer-1,elapsed=float(timer),
                    timer_scale=None,duration=32,mode=mode,flags=0)
                emulator.mem_write(scratch,struct.pack('<5fiifIIii',start,end,.2,.03,start,
                    timer-1,timer,float(timer),0,0,32,mode))
                observed=[start];active=True
                for _ in range(24):
                    if active:
                        emulator.reg_write(UC_X86_REG_ESP,scratch+0xf000)
                        emulator.reg_write(UC_X86_REG_ECX,scratch)
                        emulator.mem_write(scratch+0xf000,struct.pack('<I',return_address))
                        emulator.emu_start(base+0x14140,return_address+8,count=20000)
                        value=struct.unpack('<f',emulator.mem_read(out,4))[0]
                        active=struct.unpack('<i',emulator.mem_read(scratch+0x28,4))[0]>0
                    else:value=observed[-1]
                    observed.append(value)
                predicted=scale_path(dict(native_scale=start,native_scale_interpolator=interp),24)
                error=float(np.max(np.abs(predicted-observed)))
                records.append(dict(mode=mode,timer=timer,start=start,end=end,max_abs_error=error))
    result=dict(function='TH14 1.00b 0x414140',kind='original_machine_code_on_emulated_scratch_state',
        live_game_changed=False,cases=len(records),predicted_ticks=len(records)*24,
        max_abs_error=max(r['max_abs_error'] for r in records),
        passed=all(r['max_abs_error']<1e-5 for r in records),records=records,
        scope='Scalar interpolation only; live combined bullet motion/size is observed separately')
    write_json(Path(output),result)
    print({k:v for k,v in result.items() if k!='records'})
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('exe',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();verify(a.exe,a.output)
