/* Observe complete routines in pinned TH14 1.00b; no resource/input writes.
 * Explicit prefix-only RPC can align the two visual RNG fields at a pause.
 * Never place a callback inside dec/mov/js with live flags across the hook.
 * Actual decrement is checked immediately on return, before another game tick.
 */
'use strict';
const module = Process.getModuleByName(CONFIG.module_name);
const at = rva => module.base.add(rva);
const counters = {misses: 0, bombs_used: 0};
let recent = [];
const listeners = [];
let attackTransitions = [], attackSequence = 0;
function bytes(address, length) {
    return Array.from(new Uint8Array(address.readByteArray(length)))
        .map(x => x.toString(16).padStart(2, '0')).join('');
}
function liveStory() {
    return at(CONFIG.screen).readS32() === 7 && at(CONFIG.continues).readS32() === 0;
}
function event(kind, before, after) {
    counters[kind]++;
    recent.push({kind, before, after});
    if (recent.length > 16) recent.shift();
}
for (const hook of CONFIG.hooks) {
    const address = at(hook.rva);
    if (bytes(address, hook.bytes.length / 2) !== hook.bytes)
        throw new Error('THRL event instruction differs: '+hook.name);
    listeners.push(Interceptor.attach(address, {
        onEnter() {
            this.acceptEvent = liveStory();
            if (this.acceptEvent) this.beforeInventory = at(hook.inventory).readS32();
        },
        onLeave() {
            if (!this.acceptEvent) return;
            const before = this.beforeInventory;
            const after = at(hook.inventory).readS32();
            if (before >= 0 && after === before - 1)
                event(hook.counter, before, after);
        }
    }));
}
if (CONFIG.attack_events) {
    // 423c1e/423d39: original HP/timer interrupt branches resolve the chosen
    // sub and immediately install its IP. Observing the whole lookup routine
    // catches one-tick MBossDead paths which disappear before a 4f snapshot.
    // Other lookups (registration, async helpers, ordinary calls) earn nothing.
    const cfg = CONFIG.attack_events;
    listeners.push(Interceptor.attach(at(0x848b0), {
        onEnter(args) {
            this.attack = null;
            const caller = this.returnAddress.sub(module.base).toUInt32();
            if (!liveStory() || (caller !== 0x23c23 && caller !== 0x23d3e)) return;
            const owner = this.context.esi, data = owner.add(cfg.enemy_data);
            if (!(data.add(cfg.flags).readU32() & cfg.boss_mask) || data.add(cfg.subboss_id).readS32() !== 0) return;
            const target = args[0].readUtf8String();
            if (!/^(M?Boss[AB]?)(Card\d+|\d+|Dead|Escape)$/.test(target)) return;
            const manager = at(cfg.ascii_manager).readPointer();
            this.attack = {sequence: 0, owner: owner.toUInt32(), target,
                stage: at(cfg.stage).readS32(), frame: manager.add(cfg.global_timer).readS32(),
                caller_rva: caller, source: 'native-phase-interrupt'};
        },
        onLeave(result) {
            if (!this.attack || result.isNull()) return;
            this.attack.sequence = ++attackSequence;
            attackTransitions.push(this.attack);
            if (attackTransitions.length > 128) attackTransitions.shift();
        }
    }));
}
rpc.exports = {
    alignprefixvisualrng(expected, target) {
        if (!module.base.equals(ptr('0x400000')) || !liveStory())
            throw new Error('Visual RNG alignment requires pinned live Story at owned pause');
        const layout={visual_seed:[0xdb508,65535],visual_counter:[0xdb50c,4294967295],
            gameplay_seed:[0xdb510,65535],gameplay_counter:[0xdb514,4294967295]};
        for (const key of Object.keys(layout)) {
            if (!Number.isInteger(expected[key]) || expected[key]<0 || expected[key]>layout[key][1])
                throw new Error('Invalid expected prefix RNG');
            const value=key.endsWith('seed')?at(layout[key][0]).readU16():at(layout[key][0]).readU32();
            if (value!==expected[key]) throw new Error('Prefix RNG changed before alignment');
        }
        for (const key of ['visual_seed','visual_counter'])
            if (!Number.isInteger(target[key]) || target[key]<0 || target[key]>layout[key][1])
                throw new Error('Invalid target visual RNG');
        at(0xdb508).writeU16(target.visual_seed);
        at(0xdb50c).writeU32(target.visual_counter);
        if (at(0xdb510).readU16()!==expected.gameplay_seed || at(0xdb514).readU32()!==expected.gameplay_counter)
            throw new Error('Gameplay RNG changed during visual alignment');
        return {schema:'th14-prefix-visual-rng-alignment-v1',before:expected,
            after:{...expected,...target},gameplay_rng_written:false,native_input_ticks:0,
            write_addresses:['0x4db508:u16','0x4db50c:u32']};
    },
    state() {
        return {schema: 'th14-thrl-events-v1', observed: true,
            misses: counters.misses, bombs_used: counters.bombs_used,
            point_items: at(CONFIG.point_items).readU32(),
            recent_events: recent.slice(), exact_inventory_commit_hooks: true,
            event_read_method: 'whole-native-routine-entry-and-return-commit-check-v2',
            point_items_source: 'native_point_item_counter_0x4f5848',
            ...(CONFIG.attack_events ? {attack_event_revision: CONFIG.attack_events.revision,
                attack_event_sequence: attackSequence, attack_transitions: attackTransitions.slice()} : {})};
    },
    reset() {
        counters.misses = counters.bombs_used = 0;
        recent = [];
        attackTransitions = []; attackSequence = 0;
        return true;
    },
    close() {
        for (const listener of listeners) listener.detach();
        return true;
    }
};
