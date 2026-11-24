/* Same-session forecast checkpoints for an isolated retail TH08 worker. */

function th08BranchCheckpoints() {
    // The retail executable owns these writable gameplay and runtime data sections.
    const imageRanges = [
        {address: image.base.add(0xc6000), size: 0x1413840},
        {address: image.base.add(0x14da000), size: 0x8e0},
    ];
    const liveAllocations = new Map();
    let checkpoint = null;
    let storage = null;
    let capacity = 0;

    function remember(address, size, heap = 'crt') {
        if (address.isNull()) return;
        const key = address.toUInt32();
        liveAllocations.set(key, {address: ptr(key), size, heap});
    }

    function deferFree(address) {
        if (address.isNull()) return false;
        const key = address.toUInt32();
        const wasLive = liveAllocations.delete(key);
        if (checkpoint === null) return false;
        if (checkpoint.allocations.has(key)) {
            checkpoint.deferred++;
            return true;
        }
        if (!wasLive) {
            checkpoint.unknownRelease = address.toString();
            return true;
        }
        return false;
    }

    Interceptor.attach(image.base.add(0xa423d), {
        onEnter(args) {
            this.bytes = args[0].toUInt32();
        },
        onLeave(result) {
            remember(result, this.bytes);
        }
    });
    const malloc = new NativeFunction(image.base.add(0xa4269), 'pointer', ['uint'], 'mscdecl');
    const crtFree = image.base.add(0xa427b);
    const free = new NativeFunction(crtFree, 'void', ['pointer'], 'mscdecl');
    Interceptor.replace(crtFree, new NativeCallback(address => {
        if (!deferFree(address)) free(address);
    }, 'void', ['pointer'], 'mscdecl'));
    Interceptor.attach(image.base.add(0xa6b99), {
        onEnter(args) {
            this.bytes = args[0].toUInt32() * args[1].toUInt32();
        },
        onLeave(result) {
            remember(result, this.bytes);
        }
    });
    const crtRealloc = image.base.add(0xa6c54);
    const realloc = new NativeFunction(crtRealloc, 'pointer', ['pointer', 'uint'], 'mscdecl');
    Interceptor.replace(crtRealloc, new NativeCallback((address, size) => {
        const key = address.toUInt32();
        const original = checkpoint === null ? undefined : checkpoint.allocations.get(key);
        if (original === undefined) {
            if (checkpoint !== null && !address.isNull() && !liveAllocations.has(key)) {
                checkpoint.unknownRelease = address.toString();
                return ptr(0);
            }
            const result = realloc(address, size);
            if (!result.isNull() || size === 0) liveAllocations.delete(key);
            remember(result, size);
            return result;
        }
        if (size === 0) {
            deferFree(address);
            return ptr(0);
        }
        const result = malloc(size);
        if (!result.isNull()) {
            Memory.copy(result, address, Math.min(size, original.size));
            deferFree(address);
        }
        return result;
    }, 'pointer', ['pointer', 'uint'], 'mscdecl'));
    const kernel = Process.getModuleByName('kernel32.dll');
    function isGameCode(address) {
        return address.compare(image.base) >= 0 && address.compare(image.base.add(image.size)) < 0;
    }
    Interceptor.attach(kernel.getExportByName('GlobalAlloc'), {
        onEnter(args) {
            if (isGameCode(this.returnAddress) && (args[0].toUInt32() & 2) === 0) {
                this.bytes = args[1].toUInt32();
            }
        },
        onLeave(result) {
            if (this.bytes !== undefined) remember(result, this.bytes, 'global');
        }
    });
    const winGlobalFree = kernel.getExportByName('GlobalFree');
    const globalFree = new NativeFunction(winGlobalFree, 'pointer', ['pointer'], 'stdcall');
    Interceptor.replace(winGlobalFree, new NativeCallback(function (address) {
        if (isGameCode(this.returnAddress) && deferFree(address)) return ptr(0);
        return globalFree(address);
    }, 'pointer', ['pointer'], 'stdcall'));

    return {
        save(metadata) {
            if (checkpoint !== null && checkpoint.metadata.sequence !== metadata.sequence) {
                throw new Error('Restore the TH08 forecast checkpoint before saving another');
            }
            const spans = [...imageRanges, ...Array.from(liveAllocations.values()).filter(span => span.size)];
            const bytes = spans.reduce((total, span) => total + span.size, 0);
            if (capacity < bytes) {
                storage = Memory.alloc(bytes);
                capacity = bytes;
            }
            let offset = 0;
            for (const span of spans) {
                try {
                    Memory.copy(storage.add(offset), span.address, span.size);
                } catch (error) {
                    throw new Error('Unable to save TH08 state at ' + span.address +
                        ' for ' + span.size + ' bytes. ' + error);
                }
                offset += span.size;
            }
            checkpoint = {spans, allocations: new Map(liveAllocations), metadata, deferred: 0, unknownRelease: null};
            return {bytes, allocations: liveAllocations.size};
        },
        restore(release) {
            if (checkpoint === null) throw new Error('No TH08 forecast checkpoint is saved');
            if (checkpoint.unknownRelease !== null) {
                throw new Error('TH08 forecast released an untracked allocation at ' + checkpoint.unknownRelease);
            }
            let released = 0;
            for (const [key, span] of liveAllocations) {
                if (checkpoint.allocations.has(key)) continue;
                if (span.heap === 'global') globalFree(span.address);
                else free(span.address);
                released++;
            }
            liveAllocations.clear();
            for (const [key, span] of checkpoint.allocations) liveAllocations.set(key, span);
            let offset = 0;
            for (const span of checkpoint.spans) {
                Memory.copy(span.address, storage.add(offset), span.size);
                offset += span.size;
            }
            const restored = {...checkpoint.metadata, released, deferred: checkpoint.deferred};
            if (release) checkpoint = null;
            else checkpoint.deferred = 0;
            return restored;
        }
    };
}
