/* Owned original TH08, advanced only by its native input polls. */
'use strict';

if (Process.arch !== 'ia32') throw new Error('TH08 requires x86');
const image = Process.getModuleByName(CONFIG.module);
const poll = image.base.add(CONFIG.input.rva);
const signature = Array.from(new Uint8Array(
    poll.readByteArray(CONFIG.input.pristine_bytes.length / 2)))
    .map(byte => byte.toString(16).padStart(2, '0')).join('');
if (signature !== CONFIG.input.pristine_bytes)
    throw new Error('Installed TH08 poll instructions differ from the profile');

function win32(module, name, returns, args) {
    return new NativeFunction(Process.getModuleByName(module).getExportByName(name),
        returns, args, {abi: 'stdcall', scheduling: 'cooperative'});
}

const startup = Memory.alloc(68);
win32('kernel32.dll', 'GetStartupInfoW', 'void', ['pointer'])(startup);
const desktop = startup.add(8).readPointer().readUtf16String();
if (desktop.toLowerCase() !== ('WinSta0\\' + CONFIG.desktop).toLowerCase())
    throw new Error('TH08 did not inherit its private desktop');
const nativeSeed = Number.isInteger(CONFIG.native_seed) ? {
    requested: CONFIG.native_seed, original: null, applied: 0
} : null;
if (nativeSeed !== null) {
    // Supervisor's original one-time seed store. The wall clock and the RNG's
    // later evolution continue to use the original instructions.
    const seedStore = image.base.add(0x464ff);
    if (seedStore.readU16() !== 0xa366 ||
        !seedStore.add(2).readPointer().equals(image.base.add(0x124d520)))
        throw new Error('Installed TH08 startup seed instruction differs from the profile');
    Interceptor.attach(seedStore, {
        onEnter() {
            if (nativeSeed.applied !== 0) return;
            const before = this.context.eax.toUInt32();
            nativeSeed.original = before & 0xffff;
            this.context.eax = ptr(((before & 0xffff0000) | nativeSeed.requested) >>> 0);
            nativeSeed.applied = 1;
        }
    });
}
let updateFramePacing = null;
if (CONFIG.unpaced) {
    // Keep the original menu timing. Only remove the gameplay branch that
    // yields instead of running the next native update.
    const gate = image.base.add(0x41ecf);
    const play = Object.fromEntries(Object.entries(CONFIG.pacing_state)
        .map(([name, rva]) => [name, image.base.add(rva)]));
    const expected = [0x0f, 0x8a, 0x6f, 0x01, 0x00, 0x00];
    const actual = new Uint8Array(gate.readByteArray(expected.length));
    if (!expected.every((value, i) => value === actual[i]))
        throw new Error('Installed TH08 frame-deadline gate differs from the pacing profile');
    const nops = expected.map(() => 0x90);
    updateFramePacing = () => {
        const flags = play.game_flags.readU32();
        const inPlay = play.mode.readS32() === 2 && play.player_state.readU8() === 0
            && play.stage_timer.readS32() > 0 && (flags & 0x64) === 4;
        if (present.frame_wait_removed !== inPlay) {
            Memory.patchCode(gate, expected.length, code => code.writeByteArray(inPlay ? nops : expected));
            present.frame_wait_removed = inPlay;
        }
    };
}
const threadDesktop = win32('user32.dll', 'GetThreadDesktop', 'pointer', ['uint']);
const objectInformation = win32('user32.dll', 'GetUserObjectInformationW',
    'int', ['pointer', 'int', 'pointer', 'uint', 'pointer']);
const nameBuffer = Memory.alloc(1024);
const nameLength = Memory.alloc(4);
let actual = null;

function trace(event, fields = {}) {
    send({type: 'milestone', event, ...fields});
}

let startupMutex = null;
const mutex = CONFIG.startup_mutex;
if (mutex) {
    const callerReturn = image.base.add(mutex.caller_return_rva);
    const createMutex = image.base.add(mutex.import_rva).readPointer();
    const name = mutex.name + '.fly-touhou.' + Process.id;
    const privateName = Memory.allocUtf8String(name);
    Interceptor.attach(createMutex, {
        onEnter(args) {
            this.isolated = false;
            const originalError = this.lastError;
            try {
                if (!this.returnAddress.equals(callerReturn) || args[2].isNull()
                        || args[2].readAnsiString() !== mutex.name) return;
                // Keep the real API result and original startup logic.
                args[2] = privateName;
                this.isolated = true;
            } finally {
                this.lastError = originalError;
            }
        },
        onLeave(result) {
            if (!this.isolated) return;
            const nativeError = this.lastError;
            startupMutex = {name, created: !result.isNull(), native_last_error: nativeError};
            this.lastError = nativeError;
        }
    });
}

const user32 = Process.getModuleByName('user32.dll');
for (const suffix of ['A', 'W']) {
    Interceptor.attach(user32.getExportByName('MessageBox' + suffix), {
        onEnter(args) {
            const message = args[1].isNull() ? '' :
                suffix === 'W' ? args[1].readUtf16String() : args[1].readAnsiString();
            send({type: 'error', detail: 'TH08 startup dialog: ' + message});
        }
    });
}

// Use the existing memory COM device through the version-pinned DX8 bridge.
const system9 = Process.findModuleByName('d3d9.dll') || Module.load('d3d9.dll');
const memory9 = Module.load(CONFIG.fake_d3d9_path);
const deviceAudit = new NativeFunction(memory9.getExportByName('FlyTouhouFakeD3D9Audit'),
    'uint', ['uint'], {abi: 'stdcall', scheduling: 'cooperative'});
if (deviceAudit(0) !== 1) throw new Error('Memory device audit ABI differs');
Interceptor.replace(system9.getExportByName('Direct3DCreate9'),
    memory9.getExportByName('FlyTouhouDirect3DCreate9'));
const d3d = Process.findModuleByName('d3d8.dll') || Module.load('d3d8.dll');
const bridge = Module.load(CONFIG.d3d8to9);
const create8 = bridge.getExportByName('Direct3DCreate8');
Interceptor.replace(d3d.getExportByName('Direct3DCreate8'), create8);
const deviceEntries = new Set();
const present = {backend: 'd3d8to9-1.16.0-memory-device', windowed: null, created: false,
    frame_wait_removed: false};
Interceptor.attach(create8, {
    onLeave(api) {
        trace('d3d8', {api: api.toString()});
        if (api.isNull()) return;
        const createDevice = api.readPointer().add(15 * Process.pointerSize).readPointer();
        if (deviceEntries.has(createDevice.toString())) return;
        deviceEntries.add(createDevice.toString());
        Interceptor.attach(createDevice, {
            onEnter(args) {
                const params = args[5];
                present.windowed_requested = params.add(28).readU32() !== 0;
                params.add(28).writeU32(1);
                params.add(44).writeU32(0);
                params.add(48).writeU32(0);
                present.windowed = true;
            },
            onLeave(result) {
                present.hresult = result.toInt32();
                present.created = result.toInt32() >= 0;
                trace('create-device', {...present});
            }
        });
    }
});

let sequence = 0;
let remaining = 0;
let mask = 0;
let tape = null;
let tapeIndex = 0;
let segmentRemaining = 0;
let contactSource = null;
const beamCallbacks = new Set([0x247f1, 0x24810, 0x248e1, 0x24900, 0x249d1, 0x249f0]);
const branchCheckpoints = CONFIG.native_branch_checkpoints ? th08BranchCheckpoints() : null;
const contactKinds = {0x4a353: 'point', 0x4a465: 'enemy-body', 0x4a920: 'beam'};
function xy(vector) {
    return [vector.readFloat(), vector.add(4).readFloat()];
}

let currentBeamSite = null;
const scriptedFrames = [];
let testedGeometry = null;
let testedRngDraws = null;
let branchClearances = null;
let branchRisk = null;
const branchState = Object.fromEntries(Object.entries(CONFIG.branch_state || {})
    .map(([name, rva]) => [name, image.base.add(rva)]));
let installedGeometryCapture = false;

function setBranchRiskActive(active) {
    if (active && branchRisk === null) {
        const data = Memory.alloc(16);
        data.add(12).writeU32(CONFIG.main_thread);
        const math = Process.findModuleByName('msvcrt.dll') || Module.load('msvcrt.dll');
        const callbacks = new CModule(TH08_CLEARANCE_C, {
            th08_branch: data, th08_player: image.base.add(0x13d5ef8),
            branch_cos: math.getExportByName('cos'), branch_sin: math.getExportByName('sin')
        });
        for (const [kind, rva] of [[1, 0x4a230], [2, 0x4a360], [3, 0x4a6a0]]) {
            Interceptor.attach(image.base.add(rva),
                {onEnter: callbacks.enter_collision, onLeave: callbacks.leave_collision}, ptr(kind));
        }
        branchRisk = {data, callbacks};
        Interceptor.flush();
    }
    if (branchRisk !== null) {
        branchRisk.data.writeDouble(Infinity);
        branchRisk.data.add(8).writeU32(active ? 1 : 0);
    }
}

function captureGeometry() {
    if (installedGeometryCapture) return;
    installedGeometryCapture = true;
    const player = image.base.add(0x13d5ef8);
    const rng = image.base.add(0x124d520);
    const angleBits = new DataView(new ArrayBuffer(4));
    for (const [kind, rva] of [['bullet', 0x4a230], ['body', 0x4a360], ['beam', 0x4a6a0]]) {
        Interceptor.attach(image.base.add(rva), {
            onEnter(args) {
                if (testedGeometry === null || this.threadId !== CONFIG.main_thread) return;
                this.geometry = {kind, position: xy(args[0]), size: xy(args[1]),
                    site: this.returnAddress.sub(image.base).toInt32(), player: xy(player.add(0x2b4))};
                if (kind === 'beam') {
                    angleBits.setUint32(0, args[3].toUInt32(), true);
                    this.geometry.origin = xy(args[2]);
                    this.geometry.angle = angleBits.getFloat32(0, true);
                    this.geometry.player_half = xy(player.add(0x3d4));
                } else {
                    this.geometry.player_min = xy(player.add(0x38c));
                    this.geometry.player_max = xy(player.add(0x398));
                }
            },
            onLeave(result) {
                if (!this.geometry) return;
                this.geometry.result = result.toInt32();
                testedGeometry.push(this.geometry);
            }
        });
    }
    Interceptor.attach(image.base.add(0x3ecc0), {
        onEnter() {
            if (testedRngDraws !== null && this.threadId === CONFIG.main_thread && this.context.ecx.equals(rng))
                this.seedBefore = rng.readU16();
        },
        onLeave(result) {
            if (this.seedBefore !== undefined)
                testedRngDraws.push([this.seedBefore, result.toUInt32() & 0xffff]);
        }
    });
    Interceptor.flush();
}

if (CONFIG.forecast_scripted_beams) {
    // These three original enemy callbacks invoke the damage test each tick.
    // Their other call has a narrower box and also allows the graze test.
    const damageSites = new Set([0x24810, 0x24900, 0x249f0]);
    const angleBits = new DataView(new ArrayBuffer(4));
    Interceptor.attach(image.base.add(0x4a6a0), {
        onEnter(args) {
            this.previousBeamSite = currentBeamSite;
            const site = this.returnAddress.sub(image.base).toInt32();
            currentBeamSite = site;
            if (!damageSites.has(site)) return;
            let frame = scriptedFrames[scriptedFrames.length - 1];
            if (!frame || frame.poll !== sequence) {
                frame = {poll: sequence, beams: []};
                scriptedFrames.push(frame);
                if (scriptedFrames.length > 3) scriptedFrames.shift();
            }
            const owner = this.context.ebp.sub(0x34).readPointer();
            angleBits.setUint32(0, args[3].toUInt32(), true);
            frame.beams.push({
                site, owner: owner.toUInt32(),
                center: xy(args[0]), size: xy(args[1]), origin: xy(args[2]),
                angle: angleBits.getFloat32(0, true)
            });
        },
        onLeave() {
            currentBeamSite = this.previousBeamSite;
        }
    });
}
Interceptor.attach(image.base.add(0x4ab40), {
    onEnter() {
        const caller = this.returnAddress.sub(image.base).toInt32();
        const kind = contactKinds[caller] || 'unclassified';
        contactSource = {kind, input_poll: sequence, caller_rva: caller};
        try {
            const stack = this.context.ebp;
            const upstream = kind === 'beam' && currentBeamSite !== null
                ? currentBeamSite : stack.add(4).readPointer().sub(image.base).toInt32();
            contactSource.source_rva = upstream;
            if (kind === 'beam') {
                Object.assign(contactSource, {
                    enemy_callback: beamCallbacks.has(upstream),
                    center: xy(stack.add(8).readPointer()),
                    extent: xy(stack.add(12).readPointer()),
                    origin: xy(stack.add(16).readPointer()),
                    angle: stack.add(20).readFloat(), mode: stack.add(24).readS32()
                });
            }
        } catch (error) {
            contactSource.read_error = String(error);
        }
    }
});
Interceptor.attach(poll, {
    onEnter() {
        this.owned = this.threadId === CONFIG.main_thread;
        if (!this.owned) return;
        if (updateFramePacing) updateFramePacing();
        if (branchClearances !== null) {
            const sameStage = branchState.mode.readS32() === 2
                && branchState.stage.readS32() === branchClearances.stage;
            const playerState = branchState.player_state.readU8();
            const contact = sameStage && (playerState === 2 || contactSource !== null);
            const active = sameStage && (branchState.game_flags.readU32() & 0x64) === 4;
            const gap = branchRisk.data.readDouble();
            branchClearances.frames.push({position: xy(branchState.player_position), contact,
                clearance: Number.isFinite(gap) ? gap : null,
                geometry_valid: active && (Number.isFinite(gap) || gap === Infinity)});
            branchRisk.data.writeDouble(Infinity);
            if (!active || playerState !== 0 || contact) remaining = 0;
        }
        if (!actual) {
            const handle = threadDesktop(this.threadId);
            const queried = !handle.isNull() && objectInformation(handle, 2, nameBuffer, 1024, nameLength);
            const name = queried ? nameBuffer.readUtf16String() : '';
            actual = {schema: 'th14-main-thread-desktop-v1', thread_id: this.threadId,
                queried_in_original_input_thread: true, name,
                verified: name.toLowerCase() === CONFIG.desktop.toLowerCase()};
        }
        if (remaining === 0) {
            present.memory_devices = deviceAudit(1);
            present.unsupported_calls = deviceAudit(2);
            const beams = scriptedFrames.filter(frame => sequence - frame.poll < 3);
            send({type: 'checkpoint', sequence, native_desktop_actual: actual, present, native_seed: nativeSeed,
                startup_mutex: startupMutex, contact_source: contactSource,
                tested_geometry: testedGeometry, tested_rng_draws: testedRngDraws,
                branch_clearances: branchClearances === null ? null : branchClearances.frames,
                scripted_beams: beams.length && beams[beams.length - 1].poll === sequence ? beams : []});
            let advance = false;
            while (!advance) recv('control', message => {
                const control = message.payload;
                if (control.branch_checkpoint) {
                    try {
                        if (branchCheckpoints === null) throw new Error('TH08 branch checkpoints are disabled');
                        if (control.branch_checkpoint === 'save') {
                            const root = branchCheckpoints.save({sequence, contactSource, mask,
                                scriptedFrames: JSON.stringify(scriptedFrames)});
                            send({type: 'branch-checkpoint', operation: 'save', sequence, ...root});
                        } else {
                            const root = branchCheckpoints.restore(control.release_root);
                            sequence = root.sequence;
                            contactSource = root.contactSource;
                            mask = root.mask;
                            scriptedFrames.splice(0, scriptedFrames.length, ...JSON.parse(root.scriptedFrames));
                            testedGeometry = null;
                            testedRngDraws = null;
                            branchClearances = null;
                            setBranchRiskActive(false);
                            send({type: 'branch-checkpoint', operation: 'restore', sequence,
                                branch_allocations_released: root.released, root_frees_deferred: root.deferred});
                        }
                    } catch (error) {
                        send({type: 'branch-checkpoint', error: String(error)});
                    }
                    return;
                }
                // Scope Player::Die evidence to this command, even if the
                // original already left DYING before the next checkpoint.
                contactSource = null;
                tape = control.tape || null;
                tapeIndex = 0;
                if (tape) {
                    [mask, segmentRemaining] = tape[0];
                    remaining = tape.reduce((total, segment) => total + segment[1], 0);
                } else {
                    mask = control.mask;
                    remaining = control.ticks;
                }
                testedGeometry = control.capture_geometry ? [] : null;
                testedRngDraws = control.capture_geometry ? [] : null;
                branchClearances = control.branch_clearances
                    ? {stage: branchState.stage.readS32(), frames: []} : null;
                setBranchRiskActive(branchClearances !== null);
                if (testedGeometry !== null) captureGeometry();
                advance = true;
            }).wait();
        }
        this.mask = mask;
        remaining--;
        sequence++;
        if (tape && --segmentRemaining === 0 && remaining > 0)
            [mask, segmentRemaining] = tape[++tapeIndex];
    },
    onLeave(result) {
        if (this.owned) result.replace(this.mask);
    }
});
send({type: 'hook-ready', desktop, native_tape: {schema: 1, max_input_ticks: 36000}});
