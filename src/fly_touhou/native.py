"""Version-pinned, bounded Win32 memory access and native-game input.

The game remains the original Windows process. No browser, replacement engine,
or fabricated game states are used by this backend. Memory reads use a 32-bit
target pointer size even when Python itself is 64-bit.
"""
from __future__ import annotations

import ctypes as C
from ctypes import wintypes as W
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

import psutil

from .state import Bullet, Enemy, Laser, Player, Snapshot, decode_action


class NativeError(RuntimeError):
    pass


class FocusLost(NativeError):
    pass


@dataclass
class Profile:
    name: str
    game_id: int
    executable_names: list[str]
    sha256: list[str]
    offsets: dict
    verification: str = "reference-only"
    bounds: tuple[float, float, float, float] = (-192.0, 192.0, 0.0, 448.0)
    slow_speed: float = 2.0
    fast_speed: float = 4.0
    provenance: dict | None = None
    hooks: dict | None = None

    @classmethod
    def load(cls, path: str | Path) -> "Profile":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d["bounds"] = tuple(d.get("bounds", (-192, 192, 0, 448)))
        return cls(**d)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def find_game(names: list[str]) -> list[dict]:
    wanted = {n.lower() for n in names}
    out = []
    for p in psutil.process_iter(["pid", "name", "exe"]):
        try:
            if (p.info["name"] or "").lower() in wanted:
                out.append(dict(p.info))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


class ProcessMemory:
    def __init__(self, pid: int, *, writable: bool = False, can_suspend: bool = False):
        if sys.platform != "win32":
            raise NativeError("The native backend requires Windows")
        self.pid = pid
        self.exe = Path(psutil.Process(pid).exe())
        self.k = C.WinDLL("kernel32", use_last_error=True)
        self.k.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
        self.k.OpenProcess.restype = W.HANDLE
        self.k.ReadProcessMemory.argtypes = [W.HANDLE, C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
        self.k.ReadProcessMemory.restype = W.BOOL
        self.k.WriteProcessMemory.argtypes = [W.HANDLE, C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
        self.k.WriteProcessMemory.restype = W.BOOL
        self.k.CloseHandle.argtypes = [W.HANDLE]
        rights = 0x0400 | 0x0010 | (0x0020 | 0x0008 if writable else 0) | (0x0800 if can_suspend else 0)
        self.handle = self.k.OpenProcess(rights, False, pid)
        if not self.handle:
            raise NativeError(f"Cannot open game process {pid}: {C.WinError(C.get_last_error())}")
        self.base = self._module_base()
        self.writable = writable
        self.can_suspend, self.suspended = can_suspend, False
        if can_suspend:
            self.nt = C.WinDLL("ntdll")
            for name in ("NtSuspendProcess", "NtResumeProcess"):
                fn = getattr(self.nt, name)
                fn.argtypes, fn.restype = [W.HANDLE], C.c_long

    def _module_base(self) -> int:
        class MODULEENTRY32(C.Structure):
            _fields_ = [("dwSize", W.DWORD), ("th32ModuleID", W.DWORD),
                        ("th32ProcessID", W.DWORD), ("GlblcntUsage", W.DWORD),
                        ("ProccntUsage", W.DWORD), ("modBaseAddr", C.POINTER(C.c_ubyte)),
                        ("modBaseSize", W.DWORD), ("hModule", W.HMODULE),
                        ("szModule", W.WCHAR * 256), ("szExePath", W.WCHAR * 260)]
        self.k.CreateToolhelp32Snapshot.argtypes = [W.DWORD, W.DWORD]
        self.k.CreateToolhelp32Snapshot.restype = W.HANDLE
        self.k.Module32FirstW.argtypes = [W.HANDLE, C.POINTER(MODULEENTRY32)]
        self.k.Module32FirstW.restype = W.BOOL
        self.k.Module32NextW.argtypes = [W.HANDLE, C.POINTER(MODULEENTRY32)]
        self.k.Module32NextW.restype = W.BOOL
        snap = self.k.CreateToolhelp32Snapshot(0x08 | 0x10, self.pid)
        if snap == C.c_void_p(-1).value:
            raise NativeError("Cannot enumerate the game module")
        try:
            entry = MODULEENTRY32()
            entry.dwSize = C.sizeof(entry)
            ok = self.k.Module32FirstW(snap, C.byref(entry))
            while ok:
                if entry.szModule.lower() == self.exe.name.lower():
                    return int(C.cast(entry.modBaseAddr, C.c_void_p).value)
                ok = self.k.Module32NextW(snap, C.byref(entry))
        finally:
            self.k.CloseHandle(snap)
        raise NativeError("Game image not found in its module list")

    def read(self, address: int, size: int) -> bytes:
        if not (0x10000 <= address < 0x100000000) or not 0 < size <= 16 * 1024 * 1024:
            raise NativeError(f"Invalid 32-bit read range at {address:#x}, length={size}")
        buf = C.create_string_buffer(size)
        count = C.c_size_t()
        if not self.k.ReadProcessMemory(self.handle, address, buf, size, C.byref(count)) or count.value != size:
            raise NativeError(f"Game memory read failed at {address:#x} ({size} bytes)")
        return buf.raw

    def read_into(self, address: int, buffer) -> None:
        """One bounded native read into caller-owned reusable byte storage.

        The caller must finish decoding before reusing this buffer. No view of
        this mutable storage is included in a returned game snapshot.
        """
        if not isinstance(buffer, C.Array) or C.sizeof(buffer._type_) != 1:
            raise NativeError('Native read buffer must be a ctypes byte array')
        size=C.sizeof(buffer)
        if not (0x10000 <= address < 0x100000000) or not 0 < size <= 16 * 1024 * 1024 or address+size>0x100000000:
            raise NativeError(f'Invalid 32-bit read range at {address:#x}, length={size}')
        count=C.c_size_t()
        if not self.k.ReadProcessMemory(self.handle,address,buffer,size,C.byref(count)) or count.value != size:
            raise NativeError(f'Game memory read failed at {address:#x} ({size} bytes)')

    def integer(self, address: int, *, signed: bool = False, size: int = 4) -> int:
        return int.from_bytes(self.read(address, size), "little", signed=signed)

    def floating(self, address: int) -> float:
        return struct.unpack("<f", self.read(address, 4))[0]

    def pair(self, address: int) -> tuple[float, float]:
        return struct.unpack("<ff", self.read(address, 8))

    def write(self, address: int, data: bytes) -> None:
        if not self.writable:
            raise NativeError("This handle is read-only")
        count = C.c_size_t()
        buf = C.create_string_buffer(data)
        if not self.k.WriteProcessMemory(self.handle, address, buf, len(data), C.byref(count)) or count.value != len(data):
            raise NativeError(f"Game memory write failed at {address:#x}")

    def close(self):
        if self.handle:
            self.resume()
            self.k.CloseHandle(self.handle)
            self.handle = None

    def suspend(self):
        if not self.can_suspend:
            raise NativeError("Process was opened without frame-control rights")
        if not self.suspended:
            if self.nt.NtSuspendProcess(self.handle) < 0:
                raise NativeError("Cannot suspend game between RL actions")
            self.suspended = True

    def resume(self):
        if self.suspended:
            if self.nt.NtResumeProcess(self.handle) < 0:
                raise NativeError("Cannot resume game")
            self.suspended = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class NativeReader:
    def __init__(self, memory: ProcessMemory, profile: Profile, *, inspect_unverified: bool = False):
        self.mem, self.profile = memory, profile
        self.digest = sha256_file(memory.exe)
        self.verified_identity = self.digest in profile.sha256
        if not self.verified_identity and not inspect_unverified:
            raise NativeError(f"Executable SHA256 {self.digest} is not pinned by {profile.name}")
        self.o = profile.offsets

    def static(self, group: str, key: str, signed: bool = False) -> int:
        return self.mem.integer(self.mem.base + self.o[group][key], signed=signed)

    def pointer(self, group: str, key: str) -> int:
        return self.static(group, key)

    def global_frame(self) -> int:
        p = self.pointer("misc", "ascii_manager_ptr")
        return self.mem.integer(p + self.o["misc"]["global_timer"]) if p else 0

    def stage_frame(self) -> int:
        p = self.pointer("game_thread", "game_thread_pointer")
        return self.mem.integer(p + self.o["game_thread"]["zGameThread_stage_timer"]) if p else 0

    def _entries(self, head: int, limit: int = 8192):
        seen = set()
        while head:
            if head in seen or len(seen) >= limit:
                raise NativeError("Entity list is cyclic or exceeds the supported limit")
            seen.add(head)
            entry, head = struct.unpack("<II", self.mem.read(head, 8))
            if entry:
                yield entry

    def _player(self) -> Player | None:
        o, m = self.o["player"], self.mem
        p = self.pointer("player", "player_pointer")
        if not p:
            return None
        x, y = m.pair(p + o["zPlayer_pos"])
        state = m.integer(p + o["zPlayer_state"])
        radius = m.floating(p + o["zPlayer_hit_rad"])
        # A frame-gated constructor can publish the spawn position (0,400)
        # before the state and hitbox are initialized. Position alone does
        # not make that allocation a playable player; let menu stepping
        # continue until the native initialization supplies its geometry.
        if state == 0 and radius == 0:
            return None
        # This field is the elapsed state clock, including while alive.  In
        # TH14, only state 4 is the eight-frame deathbomb window (trial VA
        # 0x44d3e1 enters it; VA 0x44bd54 compares the clock against 8).
        deathbomb_remaining = 0
        if state == 4:
            elapsed = m.integer(p + o["zPlayer_db_timer"], signed=True)
            window = o.get("deathbomb_window_frames", 8)
            deathbomb_remaining = max(0, min(window, window - elapsed))
        profile_hooks = getattr(getattr(self, 'profile', None), 'hooks', None)
        if profile_hooks and profile_hooks.get('player_scale_verified'):
            scale = m.floating(p + self.o['game_specific']['zPlayer_scale'])
            radius *= scale
        return Player(x, y, radius,
                      m.integer(p + o["zPlayer_iframes"], signed=True),
                      m.integer(p + o["zPlayer_focused"]) == 1,
                      state, deathbomb_remaining)

    def _bullets(self) -> tuple[Bullet, ...]:
        o, m = self.o["bullets"], self.mem
        p = self.pointer("bullets", "bullet_manager_pointer")
        if not p:
            return ()
        out = []
        # In TH13-18 the ZUN list head is embedded in BulletManager.
        for b in self._entries(p + o["zBulletManager_list"]):
            start = min(o["zBullet_pos"], o["zBullet_velocity"], o["zBullet_hitbox_radius"], o["zBullet_state"])
            end = max(o["zBullet_pos"] + 8, o["zBullet_velocity"] + 8,
                      o["zBullet_hitbox_radius"] + 4, o["zBullet_state"] + 2)
            raw = m.read(b + start, end - start)
            x, y = struct.unpack_from("<ff", raw, o["zBullet_pos"] - start)
            vx, vy = struct.unpack_from("<ff", raw, o["zBullet_velocity"] - start)
            radius = struct.unpack_from("<f", raw, o["zBullet_hitbox_radius"] - start)[0]
            state = struct.unpack_from("<H", raw, o["zBullet_state"] - start)[0]
            kind = m.integer(b + o["zBullet_type"], size=2)
            flags_raw = m.read(b + o["zBullet_flags"], 8)
            _, iframes = struct.unpack("<II", flags_raw)
            out.append(Bullet(b, x, y, vx, vy, radius, state == 1,
                              radius == 0 and self.profile.game_id == 16,
                              iframes, kind, m.integer(b + o["zBullet_timer"])))
        return tuple(out)

    def _enemies(self) -> tuple[Enemy, ...]:
        o, m = self.o["enemies"], self.mem
        p = self.pointer("enemies", "enemy_manager_pointer")
        if not p:
            return ()
        out = []
        for e in self._entries(m.integer(p + o["zEnemyManager_list"]), limit=2048):
            d = e + o["zEnemy_data"]
            flags = m.integer(d + o["zEnemyData_flags"], size=8)
            boss = bool(flags & o["zEnemyFlags_is_boss"])
            if not boss and not m.integer(d + o["zEnemyData_anm_vm_id"]):
                continue
            x, y = m.pair(d + o["zEnemyData_pos"])
            rx, ry = m.pair(d + o["zEnemyData_hitbox"])
            out.append(Enemy(e, x, y, m.integer(d + o["zEnemyData_hp"], signed=True),
                             rx, ry, boss, bool(flags & o["zEnemyFlags_intangible"])))
        return tuple(out)

    def _lasers(self) -> tuple[Laser, ...]:
        o, m = self.o["laser_base"], self.mem
        p = self.pointer("laser_base", "laser_manager_pointer")
        if not p:
            return ()
        cur = m.integer(p + o["zLaserManager_list"])
        out, seen = [], set()
        # The last entry is an engine dummy tail, as documented by ParaKit.
        while cur:
            nxt = m.integer(cur + 4)
            if not nxt:
                break
            if cur in seen or len(seen) >= 2048:
                raise NativeError("Laser list is cyclic or oversized")
            seen.add(cur)
            state = m.integer(cur + o["zLaserBaseClass_state"])
            kind = m.integer(cur + o["zLaserBaseClass_type"])
            x, y = m.pair(cur + o["zLaserBaseClass_offset"])
            angle = m.floating(cur + o["zLaserBaseClass_angle"])
            length = m.floating(cur + o["zLaserBaseClass_length"])
            width = m.floating(cur + o["zLaserBaseClass_width"])
            nodes = ()
            if kind == 2:
                co = self.o["laser_curve"]
                derived = cur + o["zLaserBaseClass_len"]
                n = m.integer(derived + co["zLaserCurve_max_length"])
                if not 0 <= n <= 8192:
                    raise NativeError("Invalid curve-laser node count")
                arr = m.integer(derived + co["zLaserCurve_array"])
                stride = self.o["laser_curve_node"]["zLaserCurveNode_size"]
                if arr and n:
                    data = m.read(arr, n * stride)
                    pos = self.o["laser_curve_node"]["zLaserCurveNode_pos"]
                    nodes = tuple(struct.unpack_from("<ff", data, i * stride + pos) for i in range(n))
            # Preserve engine state for audit; treating every visible-width laser
            # as hazardous is a conservative planner input, not a collision oracle.
            iframes=m.integer(cur + o["zLaserBaseClass_iframes"])
            active = width > 0 and length > 0 and iframes == 0
            out.append(Laser(cur, kind, x, y, angle, length, width, active, nodes,
                             m.floating(cur + o["zLaserBaseClass_speed"]),
                              m.integer(cur + o["zLaserBaseClass_timer"]),
                              native_state=state,native_iframes=iframes))
            cur = nxt
        return tuple(out)

    def read(self, retries: int = 6) -> Snapshot:
        last_error = None
        for _ in range(retries):
            try:
                before = self.global_frame()
                thread = self.pointer("game_thread", "game_thread_pointer")
                screen = self.static("statics", "game_screen")
                player = self._player() if screen == 7 and thread else None
                gui = self.pointer("gui", "gui_pointer")
                kw = {k: self.static("statics", k, signed=(k == "lives"))
                      for k in ("lives", "bombs", "power", "score", "graze")}
                kw["score"] *= 10
                extra = {"exe_sha256": self.digest, "identity_pinned": self.verified_identity,
                         "input": self.static("statics", "input"),
                         "rng": self.static("statics", "replay_rng"),
                         "continues": self.static("statics", "continues")}
                extra["game_thread_flags"] = self.mem.integer(
                    thread + self.o["game_thread"]["zGameThread_flags"]) if thread else 0
                hooks = self.profile.hooks or {}
                if hooks.get('game_thread_replay_offset') is not None:
                    extra['replay_playback'] = self.mem.integer(thread + hooks['game_thread_replay_offset']) if thread else 0
                if hooks.get('run_flags_rva') is not None:
                    run_flags = self.mem.integer(self.mem.base + hooks['run_flags_rva'])
                    extra['run_mode'] = (run_flags & hooks['run_mode_mask']) >> hooks['run_mode_shift']
                if hooks.get('pause_menu_state_offset') is not None:
                    pause_menu = self.mem.integer(self.mem.base + hooks['retry_pointer_rva'])
                    if pause_menu:
                        extra['pause_menu_state'] = self.mem.integer(pause_menu + hooks['pause_menu_state_offset'])
                        extra['pause_menu_result'] = self.mem.integer(pause_menu + hooks['pause_menu_result_offset'])
                        extra['native_gameover'] = (extra['pause_menu_state'] in (
                            hooks['pause_menu_loss_start_state'], hooks['pause_menu_continue_ready_state'])
                            and extra['pause_menu_result'] == 0)
                if hooks.get("menu_pointer_rva") is not None:
                    menu = self.mem.integer(self.mem.base + hooks["menu_pointer_rva"])
                    if menu:
                        for label, key in (("menu_page", "menu_page_offset"),
                                           ("menu_substate", "menu_substate_offset"),
                                           ("menu_selection", "menu_selection_offset")):
                            extra[label] = self.mem.integer(menu + hooks[key], signed=True)
                if player and hooks.get("player_scale_verified"):
                    pp = self.pointer("player", "player_pointer")
                    extra["player_scale"] = self.mem.floating(pp + self.o["game_specific"]["zPlayer_scale"])
                    extra['base_player_radius'] = self.mem.floating(pp + self.o['player']['zPlayer_hit_rad'])
                env = {k: self.static("environment", k) for k in ("stage", "difficulty", "character", "subshot")}
                snap = Snapshot(frame=before, stage_frame=self.stage_frame(), player=player,
                                bullets=self._bullets() if thread and player else (),
                                enemies=self._enemies() if thread and player else (),
                                lasers=self._lasers() if thread and player else (),
                                screen=screen,
                                pause=self.static("statics", "pause_state"),
                                dialogue=bool(gui and self.mem.integer(gui + self.o["gui"]["zGui_dialogue"])),
                                season_power=self.static("game_specific", "season_power") if "season_power" in self.o.get("game_specific", {}) else 0,
                                version=self.profile.name,
                                extra=extra,
                                **kw, **env)
                if self.global_frame() == before:
                    snap.validate()
                    return snap
            except (NativeError, ValueError) as e:
                last_error = e
            time.sleep(0.0005)
        raise NativeError(f"Cannot read a coherent native frame: {last_error or 'game frame changed during extraction'}")


class NativeInput:
    """Desktop input is disabled: this project must never take over user input."""

    def __init__(self, *args, **kwargs):
        raise NativeError(
            "Desktop keyboard/mouse/focus control is disabled by the project requirement. "
            "Use the native background engine IPC backend."
        )
