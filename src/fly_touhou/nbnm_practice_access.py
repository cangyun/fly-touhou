"""Open practice menu entries in the owned TH14 process, then restore code.

Three menu-only conditional branches consult score-record unlock flags.
No score file, ECL script, RNG, player, enemy, resource or collision state is
written. Stage gate follows thprac_th14.cpp's th14_prac_menu_enter_1. The two
spell gates were read from TH14 1.00b's row/difficulty table constructors.
"""
PATCHES=((0x5e847,0x75),(0x635e4,0x75),(0x6398b,0x75))


def unlock(controller):
    if getattr(controller,'_nbnm_practice_access',None) is not None:return
    import json
    from pathlib import Path
    source='const rows='+json.dumps(PATCHES)+';const name='+json.dumps(Path(controller.executable).name)+''';
const base=Process.getModuleByName(name).base;
rpc.exports={apply(){
  for (const [rva,original] of rows) {
    const p=base.add(rva);
    if (p.readU8()!==original) throw new Error('Unexpected TH14 practice menu instruction');
    Memory.patchCode(p,1,writer=>writer.writeU8(0xeb));
  }
  return true;
},restore(){
  for (const [rva,original] of rows) Memory.patchCode(base.add(rva),1,p=>p.writeU8(original));
  return true;
}};
'''
    script=controller._session.create_script(source);script.load();script.exports_sync.apply()
    controller._nbnm_practice_access=script


def restore(controller):
    script=getattr(controller,'_nbnm_practice_access',None)
    if script is not None:
        script.exports_sync.restore();script.unload();controller._nbnm_practice_access=None
