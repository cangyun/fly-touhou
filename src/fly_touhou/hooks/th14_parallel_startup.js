/* Exact original TH14 startup-only mutex isolation; separate Frida sidecar. */
'use strict';
const expectedSha = 'f08e885e28d2e0fc8c579e88034dd7de8f2a648d8f08ce781cf08eadc9ffe353';
const originalName = 'Touhou 12 App';
const returnRva = 0x69371;
if (PARALLEL_CONFIG.schema !== 1 || PARALLEL_CONFIG.exe_sha256 !== expectedSha)
    throw new Error('Unexpected original TH14 startup identity');
if (Process.arch !== 'ia32') throw new Error('Expected 32-bit original TH14');
if (!/^[\x20-\x7e]{1,220}$/.test(PARALLEL_CONFIG.isolated_name)
        || PARALLEL_CONFIG.isolated_name === originalName
        || !PARALLEL_CONFIG.isolated_name.startsWith(originalName + '.fly-touhou.' + Process.id + '.'))
    throw new Error('Invalid owned-worker mutex name');
const startupModule = Process.getModuleByName(PARALLEL_CONFIG.module_name);
const base = startupModule.base;
const at = rva => base.add(rva);
const little32 = value => [value & 255, (value >>> 8) & 255,
                           (value >>> 16) & 255, (value >>> 24) & 255];
const hex = bytes => Array.from(bytes).map(x => x.toString(16).padStart(2, '0')).join('');
const checks = [];
function verify(rva, expected) {
    const actual = new Uint8Array(at(rva).readByteArray(expected.length));
    if (hex(actual) !== hex(expected))
        throw new Error('Startup bytes differ at RVA 0x' + rva.toString(16) + ': ' + hex(actual));
    checks.push({rva: rva, bytes: hex(actual)});
}
// Absolute operands are rebuilt from the actual module base, so rebased PE
// loads are verified without incorrectly comparing preferred-base addresses.
verify(0x69362, [0x68, ...little32(at(0xc06d8).toUInt32()), 0x6a, 1, 0x6a, 0]);
verify(0x6936b, [0xff, 0x15, ...little32(at(0xb1100).toUInt32())]);
verify(0x69371, [0xa3, ...little32(at(0xfd84c).toUInt32())]);
verify(0x69376, [0xff, 0x15, ...little32(at(0xb1094).toUInt32())]);
verify(0x69382, [0x3d, 0xb7, 0, 0, 0]);
verify(0x69387, [0x0f, 0x85, 0xc9, 1, 0, 0]);
if (at(0xc06d8).readAnsiString() !== originalName)
    throw new Error('Original startup mutex string differs');
const replacementName = Memory.allocUtf8String(PARALLEL_CONFIG.isolated_name);
const createMutexA = at(0xb1100).readPointer();
if (createMutexA.isNull()) throw new Error('CreateMutexA import is unresolved');
const listener = Interceptor.attach(createMutexA, {
    onEnter(args) {
        const beforeLastError = this.lastError;
        this.isolatedStartupCall = false;
        try {
            if (!this.returnAddress.equals(at(returnRva)) || args[2].isNull()) return;
            if (args[2].readAnsiString() !== originalName) return;
            args[2] = replacementName;
            this.isolatedStartupCall = true;
            send({type: 'parallel-startup-mutex-replaced', pid: Process.id,
                  worker_name: PARALLEL_CONFIG.worker_name,
                  caller_return: this.returnAddress.toString(), return_rva: returnRva,
                  original_name: originalName, isolated_name: PARALLEL_CONFIG.isolated_name,
                  security_attributes: args[0].toString(), initial_owner: args[1].toUInt32()});
        } finally {
            // Attach preserves the real API and its return. Keep caller's
            // incoming Win32 error state intact across inspection/logging too.
            this.lastError = beforeLastError;
        }
    },
    onLeave(retval) {
        if (!this.isolatedStartupCall) return;
        const nativeLastError = this.lastError;
        try {
            send({type: 'parallel-startup-mutex-result', pid: Process.id,
                  handle: retval.toString(), native_last_error: nativeLastError,
                  isolated_name: PARALLEL_CONFIG.isolated_name});
        } finally {
            this.lastError = nativeLastError;
        }
    }
});
send({type: 'parallel-startup-ready', schema: 1, pid: Process.id, arch: Process.arch,
      exe_sha256: expectedSha, module_name: startupModule.name, module_base: base.toString(),
      worker_name: PARALLEL_CONFIG.worker_name, original_name: originalName,
      isolated_name: PARALLEL_CONFIG.isolated_name, return_rva: returnRva,
      create_mutex_address: createMutexA.toString(), checks: checks,
      desktop_input: false, gameplay_or_resource_writes: false});
