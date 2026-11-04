"""Recover only the owned demonstration's measured Return-to-Game menu.

No global keyboard/mouse events, no native movement/Bomb overrides. The added
presentation guard lives only until this exact owned original process exits.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

SOURCE=r'''
const c=DEMO_CONFIG;
if(Process.id!==c.pid)throw new Error('Owned demo PID mismatch');
const m=Process.getModuleByName(c.module),base=m.base;
const at=r=>base.add(r);
const keep=[];
const audit={menu_ticks:0,keyboard_queries_blocked:0,focus_messages_normalized:0,
  gameplay_action_overrides:0,done:false,reason:null};
function exported(name){return Process.getModuleByName('user32.dll').getExportByName(name);}
for(const name of ['GetAsyncKeyState','GetKeyState']){
 const cb=new NativeCallback(()=>{audit.keyboard_queries_blocked++;return 0;},'int',['int'],'stdcall');
 keep.push(cb);Interceptor.replace(exported(name),cb);
}
Interceptor.attach(at(c.window_proc),{onEnter(args){
 const msg=args[1].toUInt32(),value=args[2].toUInt32();
 if(msg===0x6&&(value&0xffff)===0){args[2]=ptr(1);audit.focus_messages_normalized++;}
 if(msg===0x8){args[1]=ptr(0);audit.focus_messages_normalized++;}
}});
let listener=null;
listener=Interceptor.attach(at(c.input),{onLeave(retval){
 if(audit.done)return;
 const pause=at(c.pause_state).readU32();
 if(pause===2){audit.done=true;audit.reason='playing';listener.detach();return;}
 const menu=at(c.menu_pointer).readPointer();
 if(pause!==0||menu.isNull()){audit.done=true;audit.reason='unexpected_screen';listener.detach();return;}
 const state=menu.add(c.menu_state).readU32(),selection=menu.add(c.menu_selection).readU32();
 if(state===3&&selection!==0){audit.done=true;audit.reason='refused_non_resume_selection';listener.detach();return;}
 if(audit.menu_ticks>=64){audit.done=true;audit.reason='menu_budget_exhausted';listener.detach();return;}
 // A release followed by Z is an ordinary menu selection, and only executes
 // while native pause_state==0. Never inject an action during gameplay.
 retval.replace(audit.menu_ticks<4?0:1);audit.menu_ticks++;
}});
rpc.exports={audit(){return {...audit};}};
'''


def main():
    import frida
    from fly_touhou.thrl_demo_control import DemoControl
    from fly_touhou.thrl_control import alive,process_identity
    from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
    from fly_touhou.thrl_durable_records import write_json as write
    from fly_touhou.native import Profile,ProcessMemory
    control=DemoControl(ROOT/'artifacts/engineering/real-model-demonstration-v1');state=control.status()
    if state['status']!='paused':raise RuntimeError('Pause this owned demonstration before menu qualification')
    output=Path(state['output']);cfg=read(control.catalogue()['environment_config']);profile=Profile.load(cfg['profile'])
    identity=process_identity(state['game']['pid']);memory=ProcessMemory(identity['pid'])
    pause_rva=profile.offsets['statics']['pause_state'];hooks=profile.hooks
    menu=memory.integer(memory.base+hooks['retry_pointer_rva'])
    if (memory.integer(memory.base+pause_rva)!=0 or not menu
            or memory.integer(menu+hooks['pause_menu_state_offset'])!=3
            or memory.integer(menu+hooks['pause_menu_selection_offset'])!=0):
        memory.close();raise RuntimeError('Require the original paused Return-to-Game menu at selection0')
    configuration=dict(pid=identity['pid'],module=Path(cfg['game_exe']).name,input=hooks['input']['rva'],
        window_proc=hooks['window_proc']['rva'],pause_state=pause_rva,menu_pointer=hooks['retry_pointer_rva'],
        menu_state=hooks['pause_menu_state_offset'],menu_selection=hooks['pause_menu_selection_offset'])
    source=SOURCE.replace('DEMO_CONFIG',json.dumps(configuration))
    (output/'owned_presentation_menu_repair.js').write_text(source,encoding='utf-8')
    session=frida.attach(identity['pid']);script=session.create_script(source);script.load()
    try:
        control.pause()  # Release only our Python wait; the game is still natively paused.
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            if not alive(identity):raise RuntimeError('Owned original process exited during menu recovery')
            audit=dict(script.exports_sync.audit())
            if audit['done']:break
            time.sleep(.05)
        else:raise RuntimeError('Bounded native menu recovery timed out')
        if audit['reason']!='playing':raise RuntimeError('Native menu recovery refused: '+str(audit))
        script.eternalize()
        receipt=dict(utc=utc_now(),actual_native_identity=identity,status='resumed',audit=audit,
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),desktop_input=False,
            menu_only=True,neural_gameplay_actions_overridden=0,formal_test=False,
            persistent_guard_lifetime='this owned original child only')
        write(output/'owned_presentation_menu_repair.json',receipt)
        write(ROOT/'artifacts/engineering/B1-autonomous-dodge-20261024/real-demo-owned-menu-repair-v1.json',receipt)
        print(json.dumps(receipt,ensure_ascii=False))
    finally:
        session.detach();memory.close()


if __name__=='__main__':main()
