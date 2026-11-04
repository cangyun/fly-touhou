/* Read-only semantics observer for a newly owned, signature-pinned child. */
'use strict';
const module = Process.getModuleByName(PATH_CONFIG.module_name);
const base = module.base;
if (Process.arch !== 'ia32') throw new Error('Path observer requires ia32');
if (base.toUInt32() !== PATH_CONFIG.profile.image_base)
    throw new Error('Path observer signatures require the nominal pinned image base');
const specs = PATH_CONFIG.profile.functions;
const checks = [];
// Prepare before the normal control hook attaches: it legitimately changes the
// input function prologue. Observation attaches only after control is ready.
for (const spec of specs) {
    const actual = Array.from(new Uint8Array(base.add(spec.rva).readByteArray(spec.pristine_bytes.length / 2)))
        .map(value => value.toString(16).padStart(2, '0')).join('');
    if (actual !== spec.pristine_bytes.toLowerCase()) throw new Error(spec.name + ' pristine bytes differ: ' + actual);
    checks.push({name: spec.name, rva: spec.rva, bytes: actual});
}
const kernel = Process.getModuleByName('kernel32.dll');
const qpc = new NativeFunction(kernel.getExportByName('QueryPerformanceCounter'),
    'int', ['pointer'], {abi: 'stdcall', scheduling: 'cooperative'});
const frequency = new NativeFunction(kernel.getExportByName('QueryPerformanceFrequency'),
    'int', ['pointer'], {abi: 'stdcall', scheduling: 'cooperative'});
const timeBuffer = Memory.alloc(8);
if (!frequency(timeBuffer)) throw new Error('Observer QPC frequency failed');
const hz = timeBuffer.readS64().toNumber();
function nowMs() {
    if (!qpc(timeBuffer)) throw new Error('Observer QPC read failed');
    return timeBuffer.readS64().toNumber() * 1000 / hz;
}
let enabled = false, trace = [], dropped = 0, eventOrdinal = 0;
const stats = {};
const listeners = [];
const traceLimit = PATH_CONFIG.profile.trace_limit;
for (const spec of specs) stats[spec.name] = {
    enters: 0, leaves: 0, inflight: 0, inclusive_wall_ms: 0, max_inclusive_wall_ms: 0};
function event(name, kind, timestamp) {
    const row = {ordinal: ++eventOrdinal, name: name, kind: kind,
        thread: Process.getCurrentThreadId(), wall_ms: timestamp};
    if (trace.length < traceLimit) trace.push(row);
    else dropped++;
}
function enable() {
    if (enabled) return;
    for (const spec of specs) {
        listeners.push(Interceptor.attach(base.add(spec.rva), {
            onEnter() {
                this.pathObserverStartedMs = nowMs();
                const value = stats[spec.name];
                value.enters++; value.inflight++;
                event(spec.name, 'enter', this.pathObserverStartedMs);
            },
            onLeave() {
                // A callback that started before observation was enabled is
                // left-censored; it cannot supply an observed call duration.
                if (this.pathObserverStartedMs === undefined) return;
                const ended = nowMs(), value = stats[spec.name];
                const elapsed = ended - this.pathObserverStartedMs;
                value.leaves++; value.inflight--;
                value.inclusive_wall_ms += elapsed;
                value.max_inclusive_wall_ms = Math.max(value.max_inclusive_wall_ms, elapsed);
                event(spec.name, 'leave', ended);
            }
        }));
    }
    enabled = true;
}
function snapshot() {
    const rva = PATH_CONFIG.profile.state_rvas;
    return {schema: 'original-th14-frame-path-sample-v1', enabled: enabled,
        pid: Process.id, wall_ms: nowMs(), counters: JSON.parse(JSON.stringify(stats)),
        state: {mode_flags: base.add(rva.mode_flags).readU32(),
            windowed: base.add(rva.windowed).readU32(),
            presentation_interval: base.add(rva.presentation_interval).readU32(),
            frameskip_setting: base.add(rva.frameskip_setting).readU8(),
            pacing_setting: base.add(rva.pacing_setting).readU8()},
        trace: trace.slice(), dropped_trace_events: dropped, event_ordinal: eventOrdinal};
}
function receiveRequest() {
    recv('path-observer-request', message => {
        const request = message.payload;
        try {
            if (request.operation === 'enable') enable();
            else if (request.operation !== 'snapshot') throw new Error('Unknown observer operation');
            const value = snapshot();
            if (request.clear_trace) { trace = []; dropped = 0; }
            send({type: 'path-observer-reply', request_id: request.request_id, value: value});
        } catch (error) {
            send({type: 'path-observer-error', request_id: request.request_id, detail: String(error.stack || error)});
        }
        receiveRequest();
    });
}
receiveRequest();
send({type: 'path-observer-prepared', schema: 'original-th14-frame-path-observer-v1',
    pid: Process.id, arch: Process.arch, module: module.path, module_base: base.toString(),
    checks: checks, enabled: false, changes_input_or_clock_or_rendering: false});
