/* Exact SHA-pinned TH14 web trial 0.01b. All hooks live in our child process. */
'use strict';
const gameModule = Process.getModuleByName('th14.exe');
const base = gameModule.base;
const at = rva => base.add(rva);
if (Process.arch !== 'ia32') {
    throw new Error('Wrong injected architecture: ' + Process.arch);
}
const expectedInput = '558bec83e4f881ec54010000a1c0dc4c';
const actualInput = Array.from(new Uint8Array(at(0x18b0).readByteArray(16)))
    .map(x => x.toString(16).padStart(2, '0')).join('');
if (actualInput !== expectedInput) throw new Error('Input bytes mismatch: ' +
    JSON.stringify({main: gameModule, actual: actualInput}));
const retained = [];
let mask = 0, remaining = 0, sequence = 0, inputThread = null;
let disabled = false;
function active() {
    at(0xf2148).writeU32(1);
    at(0xf214c).writeU32(0);
}
function mute() {
    at(0xd5892).writeU8(0); at(0xd5893).writeU8(0);
    at(0xf98b0).writeS32(0); at(0xf98b4).writeS32(0);
    at(0xf98b8).writeS32(-10000);
}
function exported(module, name) {
    const found = Process.enumerateModules().find(m => m.name.toLowerCase() === module.toLowerCase());
    if (found === undefined) throw new Error('Required module is absent: ' + module);
    const result = found.findExportByName(name);
    if (result === null) throw new Error('Required export is absent: ' + module + '!' + name);
    return result;
}
function replaceExport(module, name, implementation, result, args) {
    const address = exported(module, name);
    if (address === null) return;
    const callback = new NativeCallback(implementation, result, args, 'stdcall');
    retained.push(callback);
    Interceptor.replace(address, callback);
}
for (const name of ['CreateWindowExA', 'CreateWindowExW']) {
    const address = exported('user32.dll', name);
    if (address !== null) Interceptor.attach(address, {
        onEnter(args) { args[3] = ptr(args[3].toUInt32() & ~0x10000000); }
    });
}
const show = exported('user32.dll', 'ShowWindow');
if (show !== null) Interceptor.attach(show, { onEnter(args) { args[1] = ptr(0); } });
const position = exported('user32.dll', 'SetWindowPos');
if (position !== null) Interceptor.attach(position, {
    onEnter(args) { args[6] = ptr((args[6].toUInt32() & ~0x40) | 0x80 | 0x10); }
});
// These replacements suppress the game's own activation requests. They never
// activate any window, and cannot affect calls made by the user's other apps.
replaceExport('user32.dll', 'SetForegroundWindow', () => 0, 'int', ['pointer']);
replaceExport('user32.dll', 'BringWindowToTop', () => 0, 'int', ['pointer']);
replaceExport('user32.dll', 'SetActiveWindow', () => ptr(0), 'pointer', ['pointer']);
replaceExport('user32.dll', 'SetFocus', () => ptr(0), 'pointer', ['pointer']);
// Suppress physical keyboard state in the child, including menu shortcuts.
replaceExport('user32.dll', 'GetKeyboardState', buffer => {
    buffer.writeByteArray(new Uint8Array(256)); return 1;
}, 'int', ['pointer']);
Interceptor.attach(at(0x66570), {
    onEnter(args) { if (args[1].toUInt32() === 0x1c) args[2] = ptr(1); },
    onLeave() { active(); }
});
Interceptor.attach(at(0x675b0), {
    onEnter() { at(0xd588f).writeU8(3); mute(); }
});
Interceptor.attach(at(0x18b0), {
    onLeave(retval) {
        const tid = Process.getCurrentThreadId();
        if (inputThread === null) inputThread = tid;
        if (tid !== inputThread || disabled) { retval.replace(0); return; }
        active(); mute();
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
        retval.replace(mask);
        remaining--;
    }
});
active(); mute();
send({type: 'hook-ready', module: gameModule.path, arch: Process.arch,
      background: true, desktop_input: false});
