"""Read active TH08 pickups from the original ItemManager pool."""

import ctypes
import struct

import numpy as np


MANAGER_RVA = 0x1253648
CAPACITY = 2096
ITEM = np.dtype({
    "names": ("position", "kind", "live", "state"),
    "formats": (("<f4", 2), "i1", "u1", "i1"),
    "offsets": (0x2A4, 0x2D4, 0x2D5, 0x2D7),
    "itemsize": 0x2E4,
})
COUNT_OFFSET = (CAPACITY + 1) * ITEM.itemsize + 4
# Shared observation kind codes for small, large and full power, Bomb and life.
PICKUP_KIND = {0: 1, 2: 3, 4: 8, 3: 6, 5: 4}


class TH08ItemReader:
    def __init__(self):
        self.buffer = ctypes.create_string_buffer(COUNT_OFFSET + 4)

    def read(self, memory):
        address = memory.base + MANAGER_RVA
        if not memory.integer(address + 0x17B088):
            return {"count": 0, "native_item_count": 0, "time_items": 0, "pickups": []}
        memory.read_into(address, self.buffer)
        pool = np.frombuffer(self.buffer, dtype=ITEM, count=CAPACITY)
        items = pool[pool["live"] != 0]
        pickups = [dict(x=float(x), y=float(y), kind=PICKUP_KIND[kind])
                   for (x, y), kind, state in zip(items["position"], items["kind"], items["state"])
                   if kind in PICKUP_KIND and state != 1]
        return {"count": len(items), "time_items": int(np.count_nonzero(items["kind"] == 7)),
                "native_item_count": struct.unpack_from("<I", self.buffer, COUNT_OFFSET)[0],
                "pickups": pickups}
