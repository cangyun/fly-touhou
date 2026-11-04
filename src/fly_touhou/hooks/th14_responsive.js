/* Version-pinned original TH14 control: responsive checkpoint wait v1. */
'use strict';
const gameModule = Process.getModuleByName(CONFIG.module_name);
const base = gameModule.base;
const hook = CONFIG.hooks;
const at = rva => base.add(rva);
if (Process.arch !== 'ia32') throw new Error('Expected 32-bit original game');
function verify(spec, name) {
    const expected = spec.pristine_bytes.toLowerCase();
    const actual = Array.from(new Uint8Array(at(spec.rva).readByteArray(expected.length / 2)))
        .map(x => x.toString(16).padStart(2, '0')).join('');
    if (actual !== expected) throw new Error(name + ' bytes differ: ' + actual);
}
for (const name of ['input', 'window_proc', 'window_init']) verify(hook[name], name);
const retained = [];
let mask = 0, remaining = 0, sequence = 0, inputThread = null, disabled = false;
let controlTape = null, tapeIndex = 0, tapeRemaining = 0;
let previousStage = -1, endingSeen = false, previousContinues = -1;
// Main-thread waiting must continue retrieving messages for Windows' hung-app
// detection. Only queued WM_PAINT is dispatched; no TranslateMessage, input,
// timer or native game update is generated here. PeekMessage also serves sent
// messages (for example bounded WM_NULL health checks and WM_SETTEXT labels).
const peekPaint = new NativeFunction(exported('user32.dll', 'PeekMessageW'),
    'int', ['pointer','pointer','uint','uint','uint'], {abi:'stdcall', scheduling:'cooperative'});
const dispatchPaint = new NativeFunction(exported('user32.dll', 'DispatchMessageW'),
    'int32', ['pointer'], {abi:'stdcall', scheduling:'cooperative'});
const cooperativeSleep = new NativeFunction(exported('kernel32.dll', 'Sleep'),
    'void', ['uint'], {abi:'stdcall', scheduling:'cooperative'});
const messageBuffer = Memory.alloc(48);
// Profiling and a shorter cooperative wait are explicit experiment options.
// The production default remains 10ms with no native timing calls.
const performanceOptions = CONFIG.checkpoint_performance || {};
const waitSleepMs = performanceOptions.wait_sleep_ms === undefined ? 10 : performanceOptions.wait_sleep_ms;
if (!Number.isInteger(waitSleepMs) || waitSleepMs < 1 || waitSleepMs > 10)
    throw new Error('Expected checkpoint wait_sleep_ms in [1,10]');
const collectTiming = performanceOptions.timing === true;
let counter = null, counterBuffer = null, counterFrequency = 1;
if (collectTiming) {
    const frequency = new NativeFunction(exported('kernel32.dll', 'QueryPerformanceFrequency'),
        'int', ['pointer'], {abi:'stdcall', scheduling:'cooperative'});
    counter = new NativeFunction(exported('kernel32.dll', 'QueryPerformanceCounter'),
        'int', ['pointer'], {abi:'stdcall', scheduling:'cooperative'});
    counterBuffer = Memory.alloc(8);
    if (!frequency(counterBuffer)) throw new Error('QueryPerformanceFrequency failed');
    counterFrequency = counterBuffer.readS64().toNumber();
    if (!(counterFrequency > 0)) throw new Error('Invalid performance counter frequency');
}
function counterMs() {
    if (!counter(counterBuffer)) throw new Error('QueryPerformanceCounter failed');
    return counterBuffer.readS64().toNumber() * 1000 / counterFrequency;
}
let priorWaitTiming = null, lastControlReleaseMs = null;
function waitForControl() {
    let arrived = false;
    const startedMs = collectTiming ? counterMs() : 0;
    let sweeps = 0;
    recv('control', message => {
        const command = message.payload;
        if (command.stop) { disabled = true; mask = 0; remaining = 1; }
        else {
            controlTape = null;
            if (command.tape !== undefined) {
                if (!Array.isArray(command.tape) || command.tape.length<1 || command.tape.length>36000)
                    throw new Error('Invalid bounded native input tape');
                let total=0;
                for (const segment of command.tape) {
                    if (!Array.isArray(segment) || segment.length!==2 || !Number.isInteger(segment[0])
                        || segment[0]<0 || segment[0]>0xffffff || !Number.isInteger(segment[1]) || segment[1]<1)
                        throw new Error('Invalid native input tape segment');
                    total+=segment[1];
                }
                if (total>36000) throw new Error('Native tape exceeds36000 input ticks');
                controlTape=command.tape; tapeIndex=0; tapeRemaining=controlTape[0][1];
                mask=controlTape[0][0]; remaining=total;
            } else {
                mask = command.mask >>> 0;
                remaining = Math.max(1, Math.min(36000, command.ticks | 0));
            }
        }
        arrived = true;
    });
    while (!arrived) {
        sweeps++;
        // An unfiltered non-removing poll acknowledges the thread's queue to
        // Windows without dispatching keyboard, timer or gameplay messages.
        peekPaint(messageBuffer, ptr(0), 0, 0, 0);
        // Bound each sweep so a repaint storm cannot starve the control reply.
        for (let n = 0; n < 32 && !arrived; n++) {
            if (!peekPaint(messageBuffer, ptr(0), 0x000f, 0x000f, 1)) break;
            dispatchPaint(messageBuffer);
        }
        if (!arrived) cooperativeSleep(waitSleepMs);
    }
    if (collectTiming) {
        const releasedMs = counterMs();
        priorWaitTiming = {checkpoint_sequence: sequence,
            wait_ms: releasedMs - startedMs, poll_sweeps: sweeps};
        lastControlReleaseMs = releasedMs;
    }
}
function milestones() {
    const p = CONFIG.progress;
    const stage = at(p.stage_rva).readS32();
    const continues = at(p.continues_rva).readU32();
    const thread = at(p.thread_rva).readPointer();
    const flags = thread.isNull() ? 0 : thread.add(p.thread_flags_offset).readU32();
    const ending = (flags & 0x4000) !== 0;
    if (stage !== previousStage || continues !== previousContinues || (ending && !endingSeen)) {
        send({type: 'milestone', sequence: sequence, stage: stage, continues: continues,
              difficulty: at(p.difficulty_rva).readU32(), thread_flags: flags,
              ending: ending});
        previousStage = stage; previousContinues = continues; endingSeen = ending;
    }
}
function nativeGameOver() {
    if (hook.pause_menu_state_offset === undefined) return false;
    const menu = at(hook.retry_pointer_rva).readPointer();
    if (menu.isNull()) return false;
    const state = menu.add(hook.pause_menu_state_offset).readU32();
    return (state === hook.pause_menu_loss_start_state || state === hook.pause_menu_continue_ready_state)
        && menu.add(hook.pause_menu_result_offset).readU32() === 0;
}
function active() {
    at(hook.active_rva).writeU32(1); at(hook.inactive_rva).writeU32(0);
}
function mute() {
    for (const rva of hook.mute_u8) at(rva).writeU8(0);
    for (const rva of hook.mute_i32) at(rva).writeS32(0);
    at(hook.mute_attenuation_rva).writeS32(-10000);
}
function exported(module, name, optional = false) {
    const found = Process.enumerateModules().find(m => m.name.toLowerCase() === module.toLowerCase());
    if (!found) { if (optional) return null; throw new Error('Missing ' + module); }
    const result = found.findExportByName(name);
    if (result === null && !optional) throw new Error('Missing ' + module + '!' + name);
    return result;
}
const replacedExports=new Set();
function replaceExport(module, name, implementation, result, args) {
    const address=exported(module,name);
    if (replacedExports.has(address.toString())) return;
    const callback = new NativeCallback(implementation, result, args, 'stdcall');
    retained.push(callback); Interceptor.replace(address, callback);replacedExports.add(address.toString());
}
// Affect only API calls made inside this owned child; never send desktop input.
const suppressPresentation = !!(CONFIG.execution && CONFIG.execution.enabled);
const privatePresentation=!!(CONFIG.execution && CONFIG.execution.isolated_desktop);
const suppressWindowPresentation=suppressPresentation && !privatePresentation;
const visibilityAudit = {schema:'th14-pre-show-suppression-v1', installed_before_resume:true,
    enabled:suppressPresentation, created_windows:0, blocked_show_calls:0, visible_checkpoints:0,
    on_screen_visible_checkpoints:0,desktop_switch_called:false,offscreen_creation:suppressWindowPresentation,
    visibility_scope:privatePresentation ? 'private-non-input-desktop' : 'user-input-desktop',
    private_window_presentation_allowed:privatePresentation};
const focusAudit = {schema:'th14-no-activation-v1',enabled:suppressPresentation,
    isolated_desktop_name:(CONFIG.execution || {}).isolated_desktop || null,
    focus_scope:(CONFIG.execution || {}).isolated_desktop ? 'private-non-input-desktop' : 'user-input-desktop',
    installed_before_resume:true,blocked_activation_calls:0,blocked_cursor_calls:0,
    blocked_capture_calls:0,activation_messages:0,focus_messages:0,app_activation_messages:0,
    foreground_samples:0,owned_foreground_samples:0,last_foreground_pid:null,
    sampling_only:true,complete_desktop_event_coverage:false,foreground_restore_called:false};
if (focusAudit.isolated_desktop_name) {
    // STARTUPINFO is authoritative before the main thread first initializes
    // user32 (GetThreadDesktop(main_tid) can legitimately return NULL then).
    const startup=Memory.alloc(68); // exact ia32 STARTUPINFOW size
    const getStartup=new NativeFunction(exported('kernel32.dll','GetStartupInfoW'),
        'void',['pointer'],{abi:'stdcall',scheduling:'cooperative'});
    getStartup(startup);
    const desktop=startup.add(8).readPointer();
    const assigned=desktop.isNull() ? '' : desktop.readUtf16String();
    if (assigned.toLowerCase()!==('WinSta0\\'+focusAudit.isolated_desktop_name).toLowerCase())
        throw new Error('Private STARTUPINFO desktop mismatch; refusing main-thread resume');
    focusAudit.startup_desktop_path=assigned;focusAudit.startup_desktop_verified=true;
}
const nativeThreadDesktop=focusAudit.isolated_desktop_name ? new NativeFunction(
    exported('user32.dll','GetThreadDesktop'),'pointer',['uint'],
    {abi:'stdcall',scheduling:'cooperative'}) : null;
const nativeObjectInformation=focusAudit.isolated_desktop_name ? new NativeFunction(
    exported('user32.dll','GetUserObjectInformationW'),'int',['pointer','int','pointer','uint','pointer'],
    {abi:'stdcall',scheduling:'cooperative'}) : null;
const desktopNameBuffer=focusAudit.isolated_desktop_name ? Memory.alloc(1024) : null;
const desktopNameLength=focusAudit.isolated_desktop_name ? Memory.alloc(4) : null;
let nativeDesktopActual=null;
function verifyMainThreadDesktop(tid) {
    if (!focusAudit.isolated_desktop_name) return null;
    if (nativeDesktopActual!==null) return nativeDesktopActual;
    const handle=nativeThreadDesktop(tid);
    const ok=!handle.isNull() && nativeObjectInformation(handle,2,desktopNameBuffer,1024,desktopNameLength)!==0;
    const name=ok ? desktopNameBuffer.readUtf16String() : null;
    nativeDesktopActual={schema:'th14-main-thread-desktop-v1',thread_id:tid,
        expected_thread_id:CONFIG.execution.isolated_desktop_main_thread_id,
        name:name,queried_in_original_input_thread:true,
        verified:ok && tid===CONFIG.execution.isolated_desktop_main_thread_id
            && name.toLowerCase()===focusAudit.isolated_desktop_name.toLowerCase()};
    // The existing input hook still waits for Python after sending the packet.
    // A failed verification is rejected before the next mask can be released.
    return nativeDesktopActual;
}
const foregroundWindow = suppressPresentation ? new NativeFunction(
    exported('user32.dll','GetForegroundWindow'),'pointer',[],
    {abi:'stdcall',scheduling:'cooperative'}) : null;
const windowProcess = suppressPresentation ? new NativeFunction(
    exported('user32.dll','GetWindowThreadProcessId'),'uint',['pointer','pointer'],
    {abi:'stdcall',scheduling:'cooperative'}) : null;
const foregroundPidBuffer = suppressPresentation ? Memory.alloc(4) : null;
function sampleForeground(reason) {
    if (!suppressPresentation) return;
    const hwnd=foregroundWindow();foregroundPidBuffer.writeU32(0);
    if (!hwnd.isNull()) windowProcess(hwnd,foregroundPidBuffer);
    const pid=foregroundPidBuffer.readU32();
    focusAudit.foreground_samples++;
    focusAudit.last_foreground_pid=pid;
    focusAudit.last_sample_reason=reason;
    if (pid===Process.id) focusAudit.owned_foreground_samples++;
}
if (suppressPresentation) globalThis.th14FocusAudit={sample:sampleForeground,audit:focusAudit};
const ownedWindows = [];
const isWindowVisible = suppressPresentation ? new NativeFunction(exported('user32.dll','IsWindowVisible'),
    'int',['pointer'],{abi:'stdcall',scheduling:'cooperative'}) : null;
const getWindowStyle = suppressPresentation ? new NativeFunction(exported('user32.dll','GetWindowLongW'),
    'int',['pointer','int'],{abi:'stdcall',scheduling:'cooperative'}) : null;
const getWindowRect = suppressPresentation ? new NativeFunction(exported('user32.dll','GetWindowRect'),
    'int',['pointer','pointer'],{abi:'stdcall',scheduling:'cooperative'}) : null;
const getMetric = suppressPresentation ? new NativeFunction(exported('user32.dll','GetSystemMetrics'),
    'int',['int'],{abi:'stdcall',scheduling:'cooperative'}) : null;
const screenRect = suppressPresentation ? [getMetric(76),getMetric(77),getMetric(78),getMetric(79)] : null;
const windowRectBuffer = suppressPresentation ? Memory.alloc(16) : null;
function keepOffscreen(args,xIndex,yIndex,flagsIndex) {
    if (!suppressWindowPresentation || (getWindowStyle(args[0],-16)&0x40000000)!==0) return;
    args[xIndex]=ptr(-32000);args[yIndex]=ptr(-32000);
    args[flagsIndex]=ptr((args[flagsIndex].toUInt32() & ~0x42) | 0x80 | 0x10);
}
for (const name of ['CreateWindowExA', 'CreateWindowExW']) Interceptor.attach(exported('user32.dll', name), {
    onEnter(args) {
        if (privatePresentation) return; // Local HWNDs cannot enter the user's input desktop.
        args[3] = ptr(args[3].toUInt32() & ~0x10000000);
        if (suppressWindowPresentation) {
            args[0] = ptr((args[0].toUInt32() & ~0x40000) | 0x08000080);
            if ((args[3].toUInt32()&0x40000000)===0) {args[4]=ptr(-32000);args[5]=ptr(-32000);}
        }
    },
    onLeave(result) {
        if (suppressPresentation && !result.isNull()) {
            visibilityAudit.created_windows++;
            if (ownedWindows.length<64) ownedWindows.push(ptr(result));
            sampleForeground('window-created');
        }
    }
});
if (suppressWindowPresentation) {
    // The first ShowWindow may IGNORE nCmdShow in favour of STARTUPINFO.
    // Do not call the original at all: this also avoids its activation side effect.
    replaceExport('user32.dll','ShowWindow',() => {visibilityAudit.blocked_show_calls++;return 0;},
        'int',['pointer','int']);
    replaceExport('user32.dll','ShowWindowAsync',() => {visibilityAudit.blocked_show_calls++;return 1;},
        'int',['pointer','int']);
    replaceExport('user32.dll','AnimateWindow',() => {visibilityAudit.blocked_show_calls++;return 1;},
        'int',['pointer','uint','uint']);
    // Direct3D can call win32u directly and bypass the user32 wrappers.
    const nativeShow=exported('win32u.dll','NtUserShowWindow',true);
    if (nativeShow!==null && !replacedExports.has(nativeShow.toString())) {
        const callback=new NativeCallback(()=>{visibilityAudit.blocked_show_calls++;return 0;},
            'int',['pointer','int'],'stdcall');retained.push(callback);
        try {Interceptor.replace(nativeShow,callback);}
        catch(error) {if (!String(error).includes('already replaced this function')) throw error;}
    }
    const nativePosition=exported('win32u.dll','NtUserSetWindowPos',true);
    if (nativePosition!==null) Interceptor.attach(nativePosition,{
        onEnter(args){keepOffscreen(args,2,3,6);}
    });
    const nativeCreate=exported('win32u.dll','NtUserCreateWindowEx',true);
    if (nativeCreate!==null) Interceptor.attach(nativeCreate,{
        onEnter(args){
            args[4]=ptr(args[4].toUInt32() & ~0x10000000);
            args[0]=ptr((args[0].toUInt32() & ~0x40000) | 0x08000080);
            if ((args[4].toUInt32()&0x40000000)===0) {args[5]=ptr(-32000);args[6]=ptr(-32000);}
        }
    });
    for (const name of ['NtUserSetWindowLong','NtUserSetWindowLongPtr']) {
        const address=exported('win32u.dll',name,true);
        if (address!==null) Interceptor.attach(address,{onEnter(args){
            if (args[1].toInt32()===-16) args[2]=ptr(args[2].toUInt32() & ~0x10000000);
            if (args[1].toInt32()===-20) args[2]=ptr((args[2].toUInt32() & ~0x40000) | 0x08000080);
        }});
    }
    Interceptor.attach(exported('user32.dll','SetWindowPlacement'), {
        onEnter(args) {
            if (args[1].isNull()) return;
            this.placement=Memory.alloc(44);Memory.copy(this.placement,args[1],44);
            this.placement.add(8).writeU32(0);args[1]=this.placement;
        }
    });
    for (const name of ['SetWindowLongA','SetWindowLongW']) Interceptor.attach(exported('user32.dll',name),{
        onEnter(args) {
            if (args[1].toInt32()===-16) args[2]=ptr(args[2].toUInt32() & ~0x10000000);
            if (args[1].toInt32()===-20) args[2]=ptr((args[2].toUInt32() & ~0x40000) | 0x08000080);
        }
    });
} else if (!privatePresentation) Interceptor.attach(exported('user32.dll', 'ShowWindow'), {
    onEnter(args) { args[1] = ptr(0); }
});
Interceptor.attach(exported('user32.dll', 'SetWindowPos'), {
    onEnter(args) {
        if (privatePresentation) return;
        args[6] = ptr((args[6].toUInt32() & ~0x40) | 0x80 | 0x10);
        keepOffscreen(args,2,3,6);
    }
});
function denyActivation(pointerResult) {
    if (suppressPresentation) focusAudit.blocked_activation_calls++;
    return pointerResult ? ptr(0) : 0;
}
if (!privatePresentation) {
    replaceExport('user32.dll', 'SetForegroundWindow', () => denyActivation(false), 'int', ['pointer']);
    replaceExport('user32.dll', 'BringWindowToTop', () => denyActivation(false), 'int', ['pointer']);
    replaceExport('user32.dll', 'SetActiveWindow', () => denyActivation(true), 'pointer', ['pointer']);
    replaceExport('user32.dll', 'SetFocus', () => denyActivation(true), 'pointer', ['pointer']);
}
if (suppressPresentation) {
    // Process-local suppression: the hidden game has no physical mouse input.
    // Never restore/steal desktop foreground to compensate for an activation.
    replaceExport('user32.dll','SetCursor',()=>{focusAudit.blocked_cursor_calls++;return ptr(0);},
        'pointer',['pointer']);
    replaceExport('user32.dll','ShowCursor',show=>{
        focusAudit.blocked_cursor_calls++;
        // Sign matches the requested state so native show/hide loops terminate.
        return show ? 0 : -1;
    },'int',['int']);
    replaceExport('user32.dll','SetCursorPos',()=>{focusAudit.blocked_cursor_calls++;return 1;},
        'int',['int','int']);
    replaceExport('user32.dll','ClipCursor',()=>{focusAudit.blocked_capture_calls++;return 1;},
        'int',['pointer']);
    replaceExport('user32.dll','SetCapture',()=>{focusAudit.blocked_capture_calls++;return ptr(0);},
        'pointer',['pointer']);
}
replaceExport('user32.dll', 'GetKeyboardState', buffer => {
    buffer.writeByteArray(new Uint8Array(256)); return 1;
}, 'int', ['pointer']);

// Redirect this game's roaming save/config folder. User-owned files stay intact.
const redirects = [];
if (CONFIG.private_appdata) {
    const getenv = exported('kernel32.dll', 'GetEnvironmentVariableA');
    Interceptor.attach(getenv, {
        onEnter(args) {
            this.name = args[0].readAnsiString();
            this.output = args[1]; this.capacity = args[2].toUInt32();
        },
        onLeave(retval) {
            if (this.name.toUpperCase() === 'APPDATA') {
                const needed = CONFIG.private_appdata.length + 1;
                if (this.capacity >= needed && !this.output.isNull()) {
                    this.output.writeAnsiString(CONFIG.private_appdata);
                    retval.replace(needed - 1);
                } else retval.replace(needed);
            }
        }
    });
    redirects.push('GetEnvironmentVariableA:APPDATA');
    for (const wide of [false, true]) {
        const name = 'SHGetFolderPath' + (wide ? 'W' : 'A');
        const address = exported('shell32.dll', name, true);
        if (address !== null) {
            Interceptor.attach(address, {
                onEnter(args) {
                    this.csidl = args[1].toUInt32() & 0xff;
                    this.output = args[4];
                },
                onLeave(retval) {
                    if (this.csidl === 0x1a && retval.toInt32() >= 0 && !this.output.isNull()) {
                        if (wide) this.output.writeUtf16String(CONFIG.private_appdata);
                        else this.output.writeAnsiString(CONFIG.private_appdata);
                    }
                }
            });
            redirects.push(name);
        }
    }
    if (!redirects.length) throw new Error('No appdata redirection export available');
}
Interceptor.attach(at(hook.window_proc.rva), {
    onEnter(args) {
        const message=args[1].toUInt32(),value=args[2].toUInt32();
        if (suppressPresentation) {
            // Count the original OS message before faking activity for gameplay.
            if (message===0x0006 && (value&0xffff)!==0) focusAudit.activation_messages++;
            if (message===0x0007) focusAudit.focus_messages++;
            if (message===0x001c && value!==0) focusAudit.app_activation_messages++;
        }
        if (message === 0x1c) args[2] = ptr(1);
    },
    onLeave() { active(); }
});
Interceptor.attach(at(hook.window_init.rva), {
    onEnter() { at(hook.window_mode_rva).writeU8(hook.window_mode_value); mute(); }
});
Interceptor.attach(at(hook.input.rva), {
    onLeave(retval) {
        const tid = Process.getCurrentThreadId();
        if (inputThread === null) inputThread = tid;
        if (tid !== inputThread || disabled) { retval.replace(0); return; }
        active(); mute(); milestones();
        if (remaining === 0) {
            sequence++;
            const checkpoint = {type: 'checkpoint', sequence: sequence, thread: tid};
            if (globalThis.th14ExecutionBacking && globalThis.th14ExecutionBacking.refreshAudit)
                checkpoint.execution={...globalThis.th14ExecutionBacking.refreshAudit()};
            if (focusAudit.isolated_desktop_name)
                checkpoint.native_desktop_actual={...verifyMainThreadDesktop(tid)};
            if (suppressPresentation) {
                // Dense startup checks, then sparse read-only sampling. Do not
                // add two desktop syscalls to every training decision forever.
                if (sequence<=32 || sequence%256===0) sampleForeground('input-checkpoint');
                const visibleWindows=ownedWindows.filter(hwnd => isWindowVisible(hwnd)!==0);
                const visible=visibleWindows.length;
                if (visible) visibilityAudit.visible_checkpoints++;
                const onScreen=visibleWindows.filter(hwnd=>{
                    if (!getWindowRect(hwnd,windowRectBuffer)) return false;
                    const left=windowRectBuffer.readS32(),top=windowRectBuffer.add(4).readS32();
                    const right=windowRectBuffer.add(8).readS32(),bottom=windowRectBuffer.add(12).readS32();
                    return right>screenRect[0] && bottom>screenRect[1]
                        && left<screenRect[0]+screenRect[2] && top<screenRect[1]+screenRect[3];
                }).length;
                if (onScreen) visibilityAudit.on_screen_visible_checkpoints++;
                checkpoint.native_visibility={...visibilityAudit,currently_visible_windows:visible,
                    currently_on_screen_windows:onScreen,virtual_screen:screenRect};
                checkpoint.native_focus={...focusAudit};
            }
            if (globalThis.th14SeedBacking && globalThis.th14SeedBacking.marker.readU32()===1)
                checkpoint.native_seed = {seed:globalThis.th14SeedBacking.seed,
                    applied:true,counters_explicitly_reset:true,native_timestamp_seeded:false};
            if (collectTiming) checkpoint.checkpoint_timing = {
                schema: 'responsive-checkpoint-wall-time-v1',
                native_interval_ms: lastControlReleaseMs === null ? null : counterMs() - lastControlReleaseMs,
                preceding_wait: priorWaitTiming, wait_sleep_ms: waitSleepMs};
            send(checkpoint);
            waitForControl();
        }
        // A completed Game Over is lifecycle handling, outside gameplay. Do not
        // let an already held SHOT key accidentally confirm a native Continue.
        retval.replace(nativeGameOver() ? 0 : mask); remaining--;
        if (controlTape!==null && --tapeRemaining===0 && ++tapeIndex<controlTape.length) {
            mask=controlTape[tapeIndex][0];tapeRemaining=controlTape[tapeIndex][1];
        }
    }
});
active(); mute();
const ready = {type: 'hook-ready', module: gameModule.path, arch: Process.arch,
    desktop_input: false, save_redirects: redirects, profile: CONFIG.profile, checkpoint_wait: 'paint-pump-v1',
    native_tape: {schema:1,max_input_ticks:36000}};
if (CONFIG.checkpoint_performance) ready.checkpoint_performance = {
    wait_sleep_ms: waitSleepMs, timing: collectTiming, changes_native_frame_duration: false};
if (suppressPresentation) {sampleForeground('before-resume');ready.native_focus={...focusAudit};}
send(ready);

