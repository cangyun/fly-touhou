/* Version-pinned original TH14 control. CONFIG is injected by the host. */
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
let previousStage = -1, endingSeen = false, previousContinues = -1;
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
function replaceExport(module, name, implementation, result, args) {
    const callback = new NativeCallback(implementation, result, args, 'stdcall');
    retained.push(callback); Interceptor.replace(exported(module, name), callback);
}
// Affect only API calls made inside this owned child; never send desktop input.
for (const name of ['CreateWindowExA', 'CreateWindowExW']) Interceptor.attach(exported('user32.dll', name), {
    onEnter(args) { args[3] = ptr(args[3].toUInt32() & ~0x10000000); }
});
Interceptor.attach(exported('user32.dll', 'ShowWindow'), {
    onEnter(args) { args[1] = ptr(0); }
});
Interceptor.attach(exported('user32.dll', 'SetWindowPos'), {
    onEnter(args) { args[6] = ptr((args[6].toUInt32() & ~0x40) | 0x80 | 0x10); }
});
replaceExport('user32.dll', 'SetForegroundWindow', () => 0, 'int', ['pointer']);
replaceExport('user32.dll', 'BringWindowToTop', () => 0, 'int', ['pointer']);
replaceExport('user32.dll', 'SetActiveWindow', () => ptr(0), 'pointer', ['pointer']);
replaceExport('user32.dll', 'SetFocus', () => ptr(0), 'pointer', ['pointer']);
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
    onEnter(args) { if (args[1].toUInt32() === 0x1c) args[2] = ptr(1); },
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
            send({type: 'checkpoint', sequence: sequence, thread: tid});
            recv('control', message => {
                const command = message.payload;
                if (command.stop) { disabled = true; mask = 0; remaining = 1; return; }
                mask = command.mask >>> 0;
                remaining = Math.max(1, Math.min(36000, command.ticks | 0));
            }).wait();
        }
        // A completed Game Over is lifecycle handling, outside gameplay. Do not
        // let an already held SHOT key accidentally confirm a native Continue.
        retval.replace(nativeGameOver() ? 0 : mask); remaining--;
    }
});
active(); mute();
send({type: 'hook-ready', module: gameModule.path, arch: Process.arch,
      desktop_input: false, save_redirects: redirects, profile: CONFIG.profile});
