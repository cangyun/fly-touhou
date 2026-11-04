/* Default-off component. No listener, data write or native clock call on load. */
'use strict';
const cfg = Object.freeze(DEADLINE_CONFIG);
if (Process.arch !== 'ia32' || Process.pointerSize !== 4 || Process.id !== cfg.expected_pid)
    throw new Error('Deadline candidate child/architecture mismatch');
const targetModule = Process.getModuleByName(cfg.module_name);
const base = targetModule.base;
const normalize = value => value.replace(/\\/g, '/').toLowerCase();
if (base.toUInt32() !== 0x400000 || cfg.image_base !== 0x400000 ||
    targetModule.size !== cfg.size_of_image || normalize(targetModule.path) !== normalize(cfg.module_path))
    throw new Error('Deadline candidate exact module/base/size binding differs');
if (cfg.schema !== 'original-th14-B-deadline-candidate-v1' || cfg.site_rva !== 0x6a74a ||
    cfg.site_instruction_size !== 8 || cfg.object_rva !== 0xf5a18 ||
    cfg.now_offset !== 0x2078 || cfg.deadline_offset !== 0x2088 || cfg.install_on_load !== false ||
    typeof cfg.allow_deadline_write !== 'boolean' || !Number.isFinite(cfg.epsilon_seconds) ||
    !(cfg.epsilon_seconds > 0 && cfg.epsilon_seconds < 1/60) ||
    !Number.isInteger(cfg.max_hits) || cfg.max_hits < 1 || cfg.max_hits > 1000000 ||
    !Number.isInteger(cfg.trace_limit) || cfg.trace_limit < 0 || cfg.trace_limit > 128)
    throw new Error('Deadline candidate configuration is invalid');
const site = base.add(cfg.site_rva), expectedObject = base.add(cfg.object_rva);
const nowAddress = expectedObject.add(cfg.now_offset);
const deadlineAddress = expectedObject.add(cfg.deadline_offset);
let listener = null, installedPatch = null, mode = 'disabled', stoppedReason = null;
let trace = [], dropped = 0;
const counts = {hits: 0, observations: 0, write_attempts: 0, writes: 0,
    rejected_object: 0, rejected_signature: 0, rejected_numeric: 0,
    rejected_access: 0, readback_failed: 0, budget_exhausted: 0};
function hex(address, size) {
    return Array.from(new Uint8Array(address.readByteArray(size)))
        .map(value => value.toString(16).padStart(2, '0')).join('');
}
function record(row) {
    if (trace.length < cfg.trace_limit) trace.push(row);
    else dropped = Math.min(1000000, dropped+1);
}
function fail(reason, detail) {
    mode = 'disabled'; stoppedReason = reason;
    if (Object.prototype.hasOwnProperty.call(counts, reason)) counts[reason]++;
    record({kind: 'rejected', reason: reason, detail: String(detail)});
}
function checkSignatures() {
    if (!Array.isArray(cfg.guard_blocks) || cfg.guard_blocks.length !== 5)
        throw new Error('Incomplete signature guard set');
    const names = new Set();
    for (const guard of cfg.guard_blocks) {
        names.add(guard.name);
        const bytes = guard.pristine_bytes;
        if (typeof bytes !== 'string' || !/^(?:[0-9a-f]{2})+$/.test(bytes))
            throw new Error('Invalid complete guard bytes');
        const address = base.add(guard.rva);
        if (listener !== null && guard.name === 'gate_block') {
            const prefix = cfg.site_rva-guard.rva, suffix = prefix+cfg.site_instruction_size;
            if (prefix !== 42 || bytes.length/2 !== 154 ||
                hex(address, prefix) !== bytes.slice(0, prefix*2) ||
                hex(address.add(suffix), bytes.length/2-suffix) !== bytes.slice(suffix*2) ||
                hex(site, cfg.site_instruction_size) !== installedPatch)
                throw new Error('Live guarded gate bytes changed outside this component instrumentation');
        } else if (hex(address, bytes.length/2) !== bytes) {
            throw new Error('Complete pristine guard differs: '+guard.name);
        }
    }
    for (const name of ['gate_block', 'mainloop_B_call', 'native_frame_interval',
                       'native_sleep_threshold', 'native_ms_factor'])
        if (!names.has(name)) throw new Error('Missing guard '+name);
}
function checkReadable(address, writable) {
    const range = Process.findRangeByAddress(address);
    if (range === null || !range.protection.includes('r') ||
        (writable && !range.protection.includes('w')) ||
        address.toUInt32()+8 > range.base.toUInt32()+range.size)
        throw new Error('Native timing field lacks existing readable/writable range');
}
function onGate(context) {
    if (mode === 'disabled') return;
    if (counts.hits >= cfg.max_hits) { fail('budget_exhausted', 'finite hit budget'); return; }
    counts.hits++;
    try { checkSignatures(); }
    catch (error) { fail('rejected_signature', error); return; }
    if (!context.esi.equals(expectedObject)) { fail('rejected_object', context.esi); return; }
    try { checkReadable(nowAddress, false); checkReadable(deadlineAddress, mode === 'candidate'); }
    catch (error) { fail('rejected_access', error); return; }
    let now, previousDeadline;
    try { now = nowAddress.readDouble(); previousDeadline = deadlineAddress.readDouble(); }
    catch (error) { fail('rejected_access', error); return; }
    const proposed = now-cfg.epsilon_seconds, delta = now-proposed;
    if (!Number.isFinite(now) || !Number.isFinite(previousDeadline) || !Number.isFinite(proposed) ||
        !(proposed < now && delta > 0 && delta < 1/60)) {
        fail('rejected_numeric', 'non-finite or unrepresentable strictly smaller deadline'); return;
    }
    if (mode === 'observe') {
        counts.observations++;
        record({kind: 'observed', now: now, deadline: previousDeadline});
    } else {
        // Recheck for intervening state changes; no clock query or register edit.
        try {
            if (nowAddress.readDouble() !== now || deadlineAddress.readDouble() !== previousDeadline)
                throw new Error('Timing fields changed during callback');
            counts.write_attempts++;
            deadlineAddress.writeDouble(proposed); // The sole permitted native DATA write.
            if (deadlineAddress.readDouble() !== proposed) { fail('readback_failed', 'deadline readback'); return; }
        } catch (error) { fail('rejected_access', error); return; }
        counts.writes++;
        record({kind: 'deadline-adjusted', now: now, before: previousDeadline, after: proposed});
    }
    if (counts.hits === cfg.max_hits) fail('budget_exhausted', 'finite hit budget reached');
}
function activate(requestedMode) {
    if (requestedMode === 'candidate' && cfg.allow_deadline_write !== true)
        throw new Error('This default-off source is not allowed to enable deadline writes');
    if (counts.hits >= cfg.max_hits || stoppedReason !== null)
        throw new Error('Stopped component requires teardown, not automatic re-enable');
    checkSignatures();
    if (listener === null) {
        listener = Interceptor.attach(site, {onEnter() { onGate(this.context); }});
        Interceptor.flush();
        installedPatch = hex(site, cfg.site_instruction_size);
        checkSignatures();
    }
    mode = requestedMode;
}
function disable() {
    mode = 'disabled';
    if (listener !== null) { listener.detach(); Interceptor.flush(); listener = null; installedPatch = null; }
    checkSignatures();
}
function snapshot() {
    return {schema: cfg.schema, pid: Process.id, installed: listener !== null,
        mode: mode, enabled: mode === 'candidate', allow_deadline_write: cfg.allow_deadline_write,
        counters: Object.assign({}, counts), stopped_reason: stoppedReason,
        trace: trace.slice(), dropped_trace_events: dropped, max_hits: cfg.max_hits,
        native_trajectory_equivalence_verified: false, ready_for_training: false,
        register_clock_input_render_resource_writes: false,
        allowed_data_write_address: deadlineAddress.toString()};
}
function receive() {
    recv('deadline-candidate-request', message => {
        const request = message.payload || {};
        try {
            if (!Number.isSafeInteger(request.request_id) || request.request_id < 1)
                throw new Error('Invalid request identity');
            if (request.operation === 'enable') activate('candidate');
            else if (request.operation === 'observe') activate('observe');
            else if (request.operation === 'disable') disable();
            else if (request.operation !== 'snapshot') throw new Error('Unknown operation');
            send({type: 'deadline-candidate-reply', request_id: request.request_id, value: snapshot()});
        } catch (error) {
            mode = 'disabled';
            send({type: 'deadline-candidate-error', request_id: request.request_id, detail: String(error.stack || error)});
        }
        receive();
    });
}
checkSignatures();
receive();
send({type: 'deadline-candidate-prepared', value: snapshot(), module: targetModule.path,
    exe_sha256_verified_by_host: cfg.exe_sha256, native_data_writes: 0,
    listener_installed_on_load: false, runtime_equivalence_claimed: false});
