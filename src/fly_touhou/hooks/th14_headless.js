// Original Japanese TH14 1.00b only. Loaded before the owned process resumes.
// Keep every logical/ANM callback. Resource memory belongs to the locally built
// nokozero-derived COM implementation; no system display device is created.
(function installTH14Execution() {
    if (!CONFIG.execution || !CONFIG.execution.enabled) return;
    if (Process.arch !== 'ia32') throw new Error('TH14 execution requires ia32');
    const module = Process.getModuleByName(CONFIG.module_name);
    if (!module.base.equals(ptr('0x400000'))) throw new Error('Unexpected TH14 base');
    function site(rva, hex) {
        const address = module.base.add(rva);
        const actual = Array.from(new Uint8Array(address.readByteArray(hex.length/2)))
            .map(v => v.toString(16).padStart(2, '0')).join('');
        if (actual !== hex) throw new Error('TH14 execution signature mismatch at '+rva.toString(16));
        return address;
    }
    // Verify ALL instruction sites before any mutation. No original disk write.
    const main = site(0x69a1d, 'f6c140740c');
    const gate = site(0x6a74a, 'f20f108688200000');
    site(0x6a778, 'f20f108e78200000660f2f8e882000000f862f010000');
    site(0x69a4c, 'b9185a4f00e8ca0c0000');
    if (!CONFIG.execution.isolated_desktop || !CONFIG.execution.fake_d3d9_path ||
            !/^[0-9a-f]{64}$/.test(CONFIG.execution.fake_d3d9_sha256 || ''))
        throw new Error('Private desktop and verified fake D3D9 DLL are required');
    const audit = {schema: 'th14-fake-com-uncapped-v1', enabled: true,
        render_backend:'fake-com-v1',requested_device: 'host-memory-fake-com', device_created: false,
        fake_d3d9_path:CONFIG.execution.fake_d3d9_path,fake_d3d9_sha256:CONFIG.execution.fake_d3d9_sha256,
        fake_d3d9_loaded:false,system_create_device_called:false,
        frame_limit_removed: true, draw_callbacks_retained: true,
        original_clock_retained: true, native_logic_retained: true,
        changes_movement_or_collision: false, desktop_input: false,
        window_changes_disabled: true, reset_calls: 0};
    const d3d = Process.getModuleByName('d3d9.dll');
    const create9 = d3d.getExportByName('Direct3DCreate9');
    const fakeModule=Module.load(CONFIG.execution.fake_d3d9_path);
    const fakeCreate9=fakeModule.getExportByName('FlyTouhouDirect3DCreate9');
    const fakeAudit=new NativeFunction(fakeModule.getExportByName('FlyTouhouFakeD3D9Audit'),
        'uint',['uint'],{abi:'stdcall',scheduling:'cooperative'});
    if (fakeAudit(0)!==1) throw new Error('Unexpected fake D3D9 audit ABI');
    Interceptor.replace(create9,fakeCreate9);
    audit.fake_d3d9_loaded=true;
    let createDeviceHook = null;
    const resetHooks = new Map();
    const ownedDevices = new Set();
    const isolatedDesktop=Boolean(CONFIG.execution.isolated_desktop);
    const ownedDeviceWindows=new Map();
    const user32=isolatedDesktop ? Process.getModuleByName('user32.dll') : null;
    const getClientRect=isolatedDesktop ? new NativeFunction(user32.getExportByName('GetClientRect'),
        'int',['pointer','pointer'],{abi:'stdcall',scheduling:'cooperative'}) : null;
    const getWindowThreadProcess=isolatedDesktop ? new NativeFunction(user32.getExportByName('GetWindowThreadProcessId'),
        'uint',['pointer','pointer'],{abi:'stdcall',scheduling:'cooperative'}) : null;
    const clientRect=isolatedDesktop ? Memory.alloc(16) : null;
    const windowPid=isolatedDesktop ? Memory.alloc(4) : null;
    const displayMode=isolatedDesktop ? Memory.alloc(16) : null;
    function explicitPrivateWindow(parameters,focusWindow) {
        if (!isolatedDesktop) return;
        parameters.add(28).writePointer(focusWindow);
        parameters.add(8).writeU32(0); // D3DFMT_UNKNOWN: system resolves current windowed format.
    }
    function keepWindowed(parameters) {
        parameters.add(32).writeU32(1); // Windowed, hidden by input hook
        parameters.add(48).writeU32(0); // no fullscreen refresh rate
        parameters.add(52).writeU32(0x80000000); // IMMEDIATE
    }
    function observeFocus(reason) {
        // This read-only observer is installed by the input hook before resume.
        if (globalThis.th14FocusAudit) globalThis.th14FocusAudit.sample(reason);
    }
    Interceptor.attach(fakeCreate9, {
        onLeave(result) {
            if (result.isNull()) throw new Error('Direct3DCreate9 returned NULL');
            if (createDeviceHook !== null) return;
            const createDevice = result.readPointer().add(16*4).readPointer();
            const adapterCount=isolatedDesktop ? new NativeFunction(result.readPointer().add(4*4).readPointer(),
                'uint',['pointer'],{abi:'stdcall',scheduling:'cooperative'}) : null;
            const adapterMode=isolatedDesktop ? new NativeFunction(result.readPointer().add(8*4).readPointer(),
                'int',['pointer','uint','pointer'],{abi:'stdcall',scheduling:'cooperative'}) : null;
            createDeviceHook = Interceptor.attach(createDevice, {
                onEnter(args) {
                    args[2] = ptr(1); // Fake COM reports a HAL-compatible resource interface.
                    args[4] = ptr((args[4].toUInt32() & ~0xd0) | 0x20 | 0x800);
                    audit.behavior_flags=args[4].toUInt32();
                    const parameters = args[5];
                    this.focusWindow=args[3];
                    if (isolatedDesktop) {
                        audit.presentation_parameters_before=Array.from({length:14},(_,i)=>parameters.add(4*i).readU32());
                        windowPid.writeU32(0);
                        const rectOK=getClientRect(args[3],clientRect)!==0;
                        const windowThread=getWindowThreadProcess(args[3],windowPid);
                        audit.private_device_context={input_thread:Process.getCurrentThreadId(),window_thread:windowThread,
                            window_process:windowPid.readU32(),client_rect_ok:rectOK,
                            client_rect:rectOK ? Array.from({length:4},(_,i)=>clientRect.add(i*4).readS32()) : null,
                            adapter:args[1].toUInt32(),adapter_count:adapterCount(args[0])};
                        const modeResult=adapterMode(args[0],args[1].toUInt32(),displayMode);
                        audit.private_device_context.adapter_display_mode_hresult=modeResult;
                        audit.private_device_context.adapter_display_mode=modeResult>=0 ?
                            Array.from({length:4},(_,i)=>displayMode.add(4*i).readU32()) : null;
                        explicitPrivateWindow(parameters,args[3]);
                    }
                    keepWindowed(parameters);
                    if (isolatedDesktop)
                        audit.presentation_parameters_after=Array.from({length:14},(_,i)=>parameters.add(4*i).readU32());
                    this.output = args[6];
                    audit.focus_window=args[3].toString();
                    audit.presentation_window=parameters.add(28).readPointer().toString();
                    audit.back_buffer=[parameters.readU32(),parameters.add(4).readU32()];
                },
                onLeave(result) {
                    audit.hresult = result.toInt32();
                    audit.fake_devices_created=fakeAudit(1);
                    audit.unsupported_getters=fakeAudit(2);
                    audit.device_created = result.toInt32() >= 0 && !this.output.readPointer().isNull();
                    if (audit.device_created) {
                        const device=this.output.readPointer();
                        ownedDevices.add(device.toString());
                        ownedDeviceWindows.set(device.toString(),this.focusWindow);
                        const reset=device.readPointer().add(16*4).readPointer();
                        if (!resetHooks.has(reset.toString())) {
                            resetHooks.set(reset.toString(),Interceptor.attach(reset,{
                                onEnter(args) {
                                    this.owned=ownedDevices.has(args[0].toString());
                                    if (!this.owned) return;
                                    explicitPrivateWindow(args[1],ownedDeviceWindows.get(args[0].toString()));
                                    keepWindowed(args[1]);audit.reset_calls++;
                                },
                                onLeave(result) {
                                    if (!this.owned) return;
                                    audit.last_reset_hresult=result.toInt32();
                                    observeFocus('d3d-reset');
                                    send({type:'execution-ready',execution:{...audit}});
                                }
                            }));
                        }
                    }
                    observeFocus('d3d-create-device');
                    send({type: audit.device_created ? 'execution-ready' : 'error',
                        detail: 'TH14 host-memory COM device creation', execution: audit});
                }
            });
        }
    });
    // Native SSE trampoline: replace ONLY next_deadline with saved_now-1us.
    // XMM0 was the deadline destination; XMM2, ESI and integer flags survive.
    // This avoids callbacks/JS work on each native frame and leaves QPC intact.
    const epsilon = Memory.alloc(8); epsilon.writeDouble(0.000001);
    const trampoline = Memory.alloc(Process.pageSize);
    Memory.patchCode(trampoline, 64, code => {
        const writer = new X86Writer(code, {pc: trampoline});
        writer.putBytes([0xf2,0x0f,0x10,0x86,0x78,0x20,0x00,0x00]);
        const address = epsilon.toUInt32();
        writer.putBytes([0xf2,0x0f,0x5c,0x05, address & 255, (address>>>8)&255,
                         (address>>>16)&255, (address>>>24)&255]);
        writer.putBytes([0xf2,0x0f,0x11,0x86,0x88,0x20,0x00,0x00]);
        writer.putJmpAddress(gate.add(8)); writer.flush();
    });
    Memory.patchCode(gate, 8, code => {
        const writer = new X86Writer(code, {pc: gate});
        writer.putJmpAddress(trampoline); writer.putNopPadding(3); writer.flush();
    });
    Memory.patchCode(main, 5, code => {
        const writer = new X86Writer(code, {pc: main});
        writer.putJmpAddress(module.base.add(0x69a4c)); writer.flush();
    });
    // Retain allocated backing and interceptor handles for the script lifetime.
    globalThis.th14ExecutionBacking = {epsilon, trampoline, audit, resetHooks, ownedDevices,fakeModule,
        refreshAudit(){audit.fake_devices_created=fakeAudit(1);audit.unsupported_getters=fakeAudit(2);return audit;}};
    send({type: 'execution-installed', execution: audit});
})();
