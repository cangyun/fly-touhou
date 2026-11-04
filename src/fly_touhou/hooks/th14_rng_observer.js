/* Opt-in RNG observation only. No seed, register, input or clock writes. */
'use strict';
const rngModule = Process.getModuleByName(RNG_CONFIG.module_name);
const rngBase = rngModule.base, rngProfile = RNG_CONFIG.profile;
if (Process.arch !== 'ia32' || rngBase.toUInt32() !== rngProfile.image_base)
    throw new Error('RNG observer requires pinned nominal-base ia32');
const rngChecks = [];
for (const name of ['init_before', 'init_after']) {
    const site = rngProfile[name];
    const actual = Array.from(new Uint8Array(rngBase.add(site.rva).readByteArray(site.pristine_bytes.length/2)))
        .map(value => value.toString(16).padStart(2, '0')).join('');
    if (actual !== site.pristine_bytes) throw new Error('Pristine RNG site differs: '+name);
    rngChecks.push({name: name, rva: site.rva, pristine_bytes: actual});
}
let rngEvents = [], rngDropped = 0, rngOrdinal = 0;
let rngBeforeHits = 0, rngAfterHits = 0, rngOrphans = 0, rngPairingCapacityDrops = 0;
let rngLabel = 'owned-startup-before-first-native-resume';
const rngPending = new Map();
function rngState() {
    const state = {}, errors = {};
    for (const [name, field] of Object.entries(rngProfile.state_fields)) {
        try {
            const address = rngBase.add(field.rva);
            state[name] = field.kind === 'u16' ? address.readU16() : address.readU32();
        } catch (error) { state[name] = null; errors[name] = String(error); }
    }
    for (const [name, field] of Object.entries(rngProfile.pointer_fields)) {
        try {
            const object = rngBase.add(field.pointer_rva).readPointer();
            state[name] = object.isNull() ? null : object.add(field.offset).readU32();
        } catch (error) { state[name] = null; errors[name] = String(error); }
    }
    return {values: state, read_errors: errors};
}
function rngRecord(kind, context, beforeOrdinal) {
    const row = {ordinal: ++rngOrdinal, kind: kind,
        thread: Process.getCurrentThreadId(), host_phase_label: rngLabel,
        eax_u32_observed: context.eax.toUInt32(), state: rngState()};
    if (beforeOrdinal !== undefined) row.before_ordinal = beforeOrdinal;
    if (rngEvents.length < rngProfile.event_limit) rngEvents.push(row);
    else rngDropped++;
    return row.ordinal;
}
// Both hooks are installed before the newly owned child is first resumed.
// Their callbacks observe an interior instruction; onLeave is intentionally
// not used as though this were a whole-function entry.
const rngListeners = [
    Interceptor.attach(rngBase.add(rngProfile.init_before.rva), {
        onEnter() {
            rngBeforeHits++;
            const thread = Process.getCurrentThreadId();
            const pending = rngPending.get(thread) || {stack: [], overflow: 0};
            const ordinal = rngRecord('before-seed-copies', this.context);
            if (!rngPending.has(thread) && rngPending.size >= rngProfile.event_limit) {
                rngPairingCapacityDrops++;
                return;
            }
            if (pending.stack.length < rngProfile.event_limit) pending.stack.push(ordinal);
            else { pending.overflow++; rngPairingCapacityDrops++; }
            rngPending.set(thread, pending);
        }
    }),
    Interceptor.attach(rngBase.add(rngProfile.init_after.rva), {
        onEnter() {
            rngAfterHits++;
            const thread = Process.getCurrentThreadId();
            const pending = rngPending.get(thread) || {stack: [], overflow: 0};
            let before;
            if (pending.overflow) { pending.overflow--; before = null; }
            else before = pending.stack.pop();
            if (before === undefined) rngOrphans++;
            rngRecord('after-seed-copies', this.context, before === undefined ? null : before);
        }
    })
];
function rngSnapshot() {
    return {schema: 'original-th14-rng-observation-v1', pid: Process.id,
        before_hits: rngBeforeHits, after_hits: rngAfterHits,
        orphan_after_hits: rngOrphans, pairing_capacity_drops: rngPairingCapacityDrops,
        pending_before_hits:
            Array.from(rngPending.values()).reduce((total, value) => total+value.stack.length+value.overflow, 0),
        host_phase_label: rngLabel, current: rngState(),
        events: rngEvents.slice(), dropped_events: rngDropped,
        seed_written: false, counter_reset_semantics: 'unobserved',
        registers_or_return_values_modified: false};
}
function receiveRngRequest() {
    recv('rng-observer-request', message => {
        const request = message.payload;
        try {
            if (request.operation !== 'snapshot' || typeof request.label !== 'string'
                    || request.label.length > 80) throw new Error('Invalid RNG observer request');
            rngLabel = request.label;
            const value = rngSnapshot();
            if (request.clear_events) { rngEvents = []; rngDropped = 0; }
            send({type: 'rng-observer-reply', request_id: request.request_id, value: value});
        } catch (error) {
            send({type: 'rng-observer-error', request_id: request.request_id, detail: String(error.stack || error)});
        }
        receiveRngRequest();
    });
}
receiveRngRequest();
send({type: 'rng-observer-prepared', pid: Process.id, arch: Process.arch,
    module: rngModule.path, image_base: rngBase.toUInt32(), checks: rngChecks,
    installed_before_first_owned_resume: true, seed_written: false,
    observes_initialization_without_counter_reset_claim: true});
