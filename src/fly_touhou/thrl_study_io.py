"""Small local training status API; no game, Torch, or UI dependency."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_shared_json(path):
    """Never deny a concurrent status writer its replace/delete sharing rights."""
    path = Path(path)
    if os.name != 'nt':
        return json.loads(path.read_text('utf-8'))
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.ReadFile.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                               ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p)
    kernel.ReadFile.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.CreateFileW(str(path.resolve()), 0x80000000, 1 | 2 | 4,
                                None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    chunks = []
    try:
        while True:
            buffer = ctypes.create_string_buffer(65536)
            count = wintypes.DWORD()
            if not kernel.ReadFile(handle, buffer, len(buffer), ctypes.byref(count), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if not count.value:
                break
            chunks.append(buffer.raw[:count.value])
    finally:
        kernel.CloseHandle(handle)
    return json.loads(b''.join(chunks).decode('utf-8-sig'))


def status_snapshot(output):
    output = Path(output)
    result = {'observed_utc': utc_now(), 'study': read_shared_json(output/'study_status.json')}
    current = result['study'].get('current_run')
    if current:
        try:
            result['progress'] = read_shared_json(Path(current)/'progress.json')
        except FileNotFoundError:
            result['progress'] = None
    return result


class StatusAPI:
    """Loopback read-only JSON API. Operator STOP uses an explicit local file."""
    def __init__(self, output, port=18714, snapshot_reader=None):
        folder = Path(output)
        reader = snapshot_reader or (lambda: status_snapshot(folder))

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path not in ('/api/status', '/health'):
                    self.send_error(404)
                    return
                try:
                    data = ({'status': 'ok', 'observed_utc': utc_now()} if self.path == '/health'
                            else reader())
                    body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (OSError, ValueError) as error:
                    self.send_error(503, type(error).__name__)

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', int(port)), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}/api/status'

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
