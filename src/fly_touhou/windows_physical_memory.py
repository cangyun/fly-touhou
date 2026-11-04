"""Fresh available physical RAM bytes, without a cached measurement.

Windows uses GlobalMemoryStatusEx.ullAvailPhys (standby + free + zero lists),
not virtual/commit availability. Only the DLL function binding is cached.
Every public call allocates/initializes a new 64-byte status and calls the API.

Primary definitions:
https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex
https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-globalmemorystatusex
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import os

_WINDOWS = os.name == "nt"


class MEMORYSTATUSEX(ctypes.Structure):
    # Fixed Windows widths, including when ABI tests run on non-Windows hosts.
    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


if ctypes.sizeof(MEMORYSTATUSEX) != 64:
    raise RuntimeError("Unexpected MEMORYSTATUSEX ABI: require64 bytes")


@lru_cache(maxsize=1)
def _global_memory_status_ex():
    """Cache the function address only; never call it or cache a RAM value here."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel32.GlobalMemoryStatusEx
    function.argtypes = (ctypes.POINTER(MEMORYSTATUSEX),)
    function.restype = ctypes.c_int32  # Win32 BOOL
    return function


def available_physical_memory_bytes() -> int:
    """Read available physical RAM now. Successful zero is valid; failure raises."""
    if not _WINDOWS:
        import psutil
        value = psutil.virtual_memory().available
        if type(value) is not int or value < 0:
            raise ValueError("psutil available physical RAM is unknown or invalid")
        return value
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not _global_memory_status_ex()(ctypes.byref(status)):
        error = ctypes.get_last_error()
        raise OSError(error, "GlobalMemoryStatusEx failed (GetLastError=%d)" % error)
    return int(status.ullAvailPhys)
