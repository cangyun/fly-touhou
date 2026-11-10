import math
import struct
import pytest
from fly_touhou.native_enemy_motion import GAME_SPEED_RVA,read_enemy_motion,predict_enemy_positions
from fly_touhou.native_enemy_contact import read_enemy_contact

DATA=0x10000000


class Memory:
    base=0x400000
    def __init__(self):self.bytes={}
    def put(self,address,fmt,*values):
        self.bytes.update(enumerate(struct.pack('<'+fmt,*values),address))
    def read(self,address,size):
        if size<=0:raise ValueError('Native reads require positive length')
        return bytes(self.bytes.get(address+i,0) for i in range(size))


def native_state(*,position=(0.,64.,0.)):
    memory=Memory()
    memory.put(memory.base+GAME_SPEED_RVA,'f',1.)
    memory.put(DATA+0x2c8,'I',memory.base+GAME_SPEED_RVA)
    for offset in (0x44,0x88):memory.put(DATA+offset,'3f',*position)
    return memory


def ecl_instruction(opcode,fmt='',*values,time=0,mask=0,rank=0xff,count=None):
    payload=struct.pack('<'+fmt,*values)
    return struct.pack('<ihHHBBI',time,opcode,16+len(payload),mask,rank,
                       len(values) if count is None else count,0)+payload

def program(script,callback=0x42c8a0):
    script+=ecl_instruction(1)
    memory=native_state(position=(0.,64.,0.))
    owner,context,ip=DATA-0x11f0,0x20000008,0x30000000
    memory.put(DATA+0x40ec,'I',owner);memory.put(DATA+0x410c,'I',callback)
    memory.put(owner+0x11d0,'II',context,0)
    memory.put(context,'fI',0.,ip);memory.put(context+0x101c,'I',8)
    memory.put(DATA+0x118,'3f',16.,16.,0.)
    memory.bytes.update(enumerate(script,ip))
    return memory,owner

def test_callback_without_geometry_reads_does_not_invalidate_motion_or_contact():
    script=ecl_instruction(23,'i',100)+ecl_instruction(1)
    memory,owner=program(script)
    state=read_enemy_motion(memory,DATA)
    assert predict_enemy_positions(state,24)['forecast_complete']
    contact=read_enemy_contact(memory,owner,DATA,motion_state=state)
    assert all(row['known'] and row['contactable'] for row in contact['forecast']['ticks'])

def test_callback_values_stay_available_in_current_tick_but_not_future_ecl():
    move=ecl_instruction(400,'ff',-9981.,64.,mask=1)
    wait=ecl_instruction(23,'i',1)
    jump=ecl_instruction(12,'ii',-len(move)-len(wait),0)
    memory,owner=program(move+wait+jump)
    memory.put(owner+0x148c,'f',10.)
    state=read_enemy_motion(memory,DATA);prediction=predict_enemy_positions(state,4)
    assert prediction['positions'][1][:2]==[10.,64.]
    assert prediction['exact_until_frame']==1
    assert state['ecl_forecast']['unknown_from_tick']==2
    contact=read_enemy_contact(memory,owner,DATA,motion_state=state)
    assert contact['forecast']['ticks'][0]['known']
    assert not contact['forecast']['ticks'][1]['known']

def test_other_callbacks_remain_unknown():
    memory,owner=program(ecl_instruction(23,'i',100),callback=0x42c990)
    state=read_enemy_motion(memory,DATA)
    assert predict_enemy_positions(state,4)['exact_until_frame']==0
    assert not read_enemy_contact(memory,owner,DATA,motion_state=state)['forecast']['ticks'][0]['known']

@pytest.mark.parametrize('target,angle',[( (10.,64.),0.),((0.,74.),math.pi/2),((-10.,64.),math.pi),((0.,54.),-math.pi/2)])
def test_ecl87_direction_is_used_by_following_native_movement(target,angle):
    script=ecl_instruction(87,'fffff',0.,-9997.,-9996.,*target,mask=7)
    script+=ecl_instruction(404,'ff',0.,2.,mask=1)+ecl_instruction(23,'i',100)
    memory,owner=program(script,callback=0)
    state=read_enemy_motion(memory,DATA);forecast=predict_enemy_positions(state,4)
    assert forecast['forecast_complete']
    assert state['ecl_forecast']['events'][0]['arguments'][0]==pytest.approx(angle,abs=1e-7)
    assert forecast['positions'][2][0]==pytest.approx(2*math.cos(angle),abs=.011)
    assert forecast['positions'][2][1]==pytest.approx(64+2*math.sin(angle),abs=.011)
