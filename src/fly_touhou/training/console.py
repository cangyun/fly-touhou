"""Local browser controls for the existing original-game job runners."""

from __future__ import annotations

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os
import secrets
import sys
from threading import Lock
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from urllib.request import ProxyHandler, build_opener
import webbrowser

from .public_jobs import job_summary, list_jobs, request_stop
from .public_runs import SUPPORT, build_plan, launch

DEFAULT_PORT = 18768
ASSETS = Path(__file__).with_name("console_web")
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/console.css": ("console.css", "text/css; charset=utf-8"),
    "/console.js": ("console.js", "text/javascript; charset=utf-8"),
}
TEXT_OPTIONS = (
    "executable", "checkpoint", "connectome", "spec", "output", "resume", "guide", "course",
    "environment", "profile", "d3d8to9",
)
NUMBER_OPTIONS = ("workers", "decisions", "seed", "segment", "fit_steps")


def plan_from_form(body, jobs):
    defaults = dict(operation="train", game="th14", objective="nbnm", difficulty="lunatic",
                    train_scope="joint", workers=None, decisions=None, seed=None, segment=None, fit_steps=640,
                    record_observations=False, **dict.fromkeys(TEXT_OPTIONS))
    if body.keys() - defaults.keys():
        raise ValueError("Unknown job option")
    options = {**defaults, **{key: value for key, value in body.items() if value is not None}}
    if options["operation"] not in ("train", "evaluate", "fit-attack"):
        raise ValueError("Choose train, evaluate or fit-attack")
    if options["train_scope"] not in ("joint", "encoder-heads"):
        raise ValueError("Choose joint or encoder-heads")
    if not isinstance(options["record_observations"], bool):
        raise ValueError("record_observations must be true or false")
    for key in TEXT_OPTIONS:
        if options[key] is not None:
            if not isinstance(options[key], str):
                raise ValueError(f"Invalid path for {key}")
            options[key] = options[key].strip() or None
    for key in NUMBER_OPTIONS:
        value = options[key]
        if value is not None and (type(value) is not int or value < (0 if key == "seed" else 1)):
            raise ValueError(f"Invalid number for {key}")
    if options["operation"] == "fit-attack" and not options["guide"]:
        raise ValueError("Enter the folder containing the verified attack guide")
    options.update(jobs=jobs, trainable_heads=None)
    return build_plan(SimpleNamespace(**options))


def _tail(path, size=16 * 1024):
    try:
        with path.open("rb") as stream:
            end = stream.seek(0, os.SEEK_END)
            stream.seek(max(0, end - size))
            raw = stream.read(size)
    except FileNotFoundError:
        return ""
    if end > size and b"\n" in raw:
        raw = raw.split(b"\n", 1)[1]
    return raw.decode("utf-8", errors="replace")


def create_console_server(jobs, port=DEFAULT_PORT):
    jobs = Path(jobs).resolve()
    token = secrets.token_urlsafe(24)
    dispatch = Lock()

    def record_for(job_id):
        if not isinstance(job_id, str) or not job_id or job_id in {".", ".."} or Path(job_id).name != job_id:
            raise ValueError("Invalid job")
        record = jobs / job_id / "job.json"
        if record.resolve().parent.parent != jobs or not record.is_file():
            raise FileNotFoundError("Job record not found")
        return record

    class Handler(BaseHTTPRequestHandler):
        def send_content(self, status, mime, payload):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; object-src 'none'; "
                             "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.end_headers()
            self.wfile.write(payload)

        def json(self, status, payload):
            self.send_content(status, "application/json; charset=utf-8",
                              json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"))

        def local_host(self):
            return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}",
                                                f"localhost:{self.server.server_port}")

        def snapshot(self, query):
            rows = list_jobs(jobs, active_only=query.get("history") != ["1"])
            return {"observed_at": datetime.now(timezone.utc).isoformat(),
                    "jobs": [dict(row, id=Path(row["job"]).parent.name) for row in rows]}

        def do_GET(self):
            if not self.local_host():
                self.json(403, {"error": "Use the local console address"})
                return
            request = urlsplit(self.path)
            query = parse_qs(request.query)
            try:
                if request.path in STATIC:
                    name, mime = STATIC[request.path]
                    self.send_content(200, mime, (ASSETS / name).read_bytes())
                elif request.path == "/api/session":
                    self.json(200, {"service": "fly-touhou-console", "token": token, "control_root": str(jobs),
                                    "support": [{"game": game, "objective": objective} for game, objective in SUPPORT]})
                elif request.path == "/api/jobs":
                    self.json(200, self.snapshot(query))
                elif request.path == "/api/log":
                    record = record_for(query.get("id", [""])[0])
                    channel = query.get("channel", ["stdout"])[0]
                    if channel not in ("stdout", "stderr"):
                        raise ValueError("Unknown log channel")
                    self.json(200, {"text": _tail(record.parent / f"{channel}.log")})
                else:
                    self.json(404, {"error": "Route not found"})
            except FileNotFoundError as error:
                self.json(404, {"error": str(error)})
            except (OSError, TypeError, ValueError, KeyError) as error:
                self.json(503, {"error": str(error)})

        def do_POST(self):
            origin = f"http://{self.headers.get('Host')}"
            if (not self.local_host() or self.headers.get("Origin") != origin
                    or not secrets.compare_digest(self.headers.get("X-Console-Token", ""), token)):
                self.json(403, {"error": "Open the local console to manage jobs"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16 * 1024 or self.headers.get_content_type() != "application/json":
                    raise ValueError("Send a JSON job request")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Send an object")
                if self.path in ("/api/plan", "/api/start"):
                    plan = plan_from_form(body, jobs)
                    if self.path == "/api/plan":
                        self.json(200, {"plan": plan.describe()})
                    else:
                        with dispatch:
                            record = launch(plan)
                        self.json(201, {"id": record.parent.name})
                elif self.path in ("/api/stop", "/api/open"):
                    record = record_for(body.get("id"))
                    if self.path == "/api/stop":
                        request_stop(record)
                    else:
                        output = Path(job_summary(record)["output"])
                        os.startfile(output if output.is_dir() else record.parent)
                    self.json(200, {"id": record.parent.name})
                else:
                    self.json(404, {"error": "Route not found"})
            except (OSError, TypeError, ValueError, KeyError) as error:
                self.json(400, {"error": str(error)})

        def log_message(self, *_args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def open_console(jobs, *, port=DEFAULT_PORT, launch_browser=True):
    jobs = Path(jobs).resolve()
    url = f"http://127.0.0.1:{port}/"
    try:
        server = create_console_server(jobs, port)
    except OSError:
        try:
            # Bypass system proxies for the local instance.
            with build_opener(ProxyHandler({})).open(url + "api/session", timeout=2) as response:
                existing = json.load(response)
        except (OSError, ValueError):
            existing = {}
        if existing.get("service") != "fly-touhou-console" or existing.get("control_root") != str(jobs):
            raise
        if sys.stdout is not None:
            print(url, flush=True)
        if launch_browser:
            webbrowser.open(url)
        return 0

    url = f"http://127.0.0.1:{server.server_port}/"
    with server:
        if sys.stdout is not None:
            print(url, flush=True)
        if launch_browser:
            webbrowser.open(url)
        try:
            server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass
    return 0
