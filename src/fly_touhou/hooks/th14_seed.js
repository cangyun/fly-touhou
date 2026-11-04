// Deterministic NEW runs only. Not a savestate or a Python/Gym seed.
(function installNativeSeed() {
    if (CONFIG.native_seed === null || CONFIG.native_seed === undefined) return;
    if (!Number.isInteger(CONFIG.native_seed) || CONFIG.native_seed<0 || CONFIG.native_seed>65535)
        throw new Error('Native seed must be a u16');
    const module=Process.getModuleByName(CONFIG.module_name),base=module.base;
    if (!base.equals(ptr('0x400000'))) throw new Error('Native seed requires pinned TH14 base');
    const copySite=base.add(0x44a1f);
    const expected='a388964d0066a310b54d0066a308b54d';
    const actual=Array.from(new Uint8Array(copySite.readByteArray(expected.length/2)))
        .map(v=>v.toString(16).padStart(2,'0')).join('');
    if (actual!==expected) throw new Error('Native RNG initialization signature differs');
    // Inline stores retain timestamp/registers/flags and avoid entering JS
    // while the original engine constructs multithreaded callback chains.
    const seed=CONFIG.native_seed,site=base.add(0x44a24);
    const marker=Memory.alloc(4),trampoline=Memory.alloc(Process.pageSize);
    marker.writeU32(0);
    function u32(x) {return [x&255,(x>>>8)&255,(x>>>16)&255,(x>>>24)&255];}
    Memory.patchCode(trampoline,80,code=>{
        const writer=new X86Writer(code,{pc:trampoline});
        for (const address of [0x4db510,0x4db508])
            writer.putBytes([0x66,0xc7,0x05,...u32(address),seed&255,(seed>>>8)&255]);
        for (const address of [0x4db514,0x4db50c])
            writer.putBytes([0xc7,0x05,...u32(address),0,0,0,0]);
        writer.putBytes([0xc7,0x05,...u32(marker.toUInt32()),1,0,0,0]);
        writer.putJmpAddress(base.add(0x44a30));writer.flush();
    });
    Memory.patchCode(site,6,code=>{
        const writer=new X86Writer(code,{pc:site});
        writer.putJmpAddress(trampoline);writer.putNop();writer.flush();
    });
    globalThis.th14SeedBacking={marker,trampoline,seed};
})();
