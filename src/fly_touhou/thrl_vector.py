"""Spawn-only THRL native workers, shared observations, explicit episode reset.

No Torch/SB3/Frida imports. The trainer must keep its own Torch imports inside
main(), because Windows spawn reimports the trainer module in each worker.
Observation views are read-only but are overwritten by the next command: copy
them into the rollout BEFORE step/reset_at. There is deliberately no auto-reset.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait as _connection_wait
from multiprocessing.shared_memory import SharedMemory
from numbers import Integral
import os
from pathlib import Path
import time
import traceback

import numpy as np
import psutil

from .atomic_json_io import atomic_write_json
from .thrl_inputs import observation_shapes, SEPARATED_INPUT

SHAPES = observation_shapes()


def wait(objects, timeout=None):
    """Windows 3.11 cannot wait on >63 pipe handles in a single native call.

    Scan disjoint batches for every ready worker, then block briefly on rotating
    batches. This preserves EOF/error readiness and the caller's total deadline
    without leaving waiter threads reading pipes after the vector closes.
    """
    objects = list(objects)
    if os.name != 'nt' or len(objects) <= 60:
        return _connection_wait(objects, timeout)
    batches = [objects[i:i + 60] for i in range(0, len(objects), 60)]
    deadline = None if timeout is None else time.monotonic() + max(0., float(timeout))
    cursor = 0
    while True:
        ready = {item for batch in batches for item in _connection_wait(batch, 0)}
        if ready:
            return [item for item in objects if item in ready]
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            return []
        interval = .005 if remaining is None else min(.005, remaining)
        ready = set(_connection_wait(batches[cursor], interval))
        cursor = (cursor + 1) % len(batches)
        if ready:
            for batch in batches:
                ready.update(_connection_wait(batch, 0))
            return [item for item in objects if item in ready]


def _native_factory(kwargs):
    from .thrl_training_mode import is_full_story
    if is_full_story(kwargs):
        if kwargs.get('online_teacher_revision'):
            from .thrl_student_aggregation import OnlineResourceReferenceEnv
            options=dict(kwargs);revision=options.pop('online_teacher_revision')
            return OnlineResourceReferenceEnv(**options,teacher_revision=revision)
        from .thrl_full_story_env import THRLFullStoryEnv
        return THRLFullStoryEnv(**kwargs)
    if kwargs.get('curriculum_manifest'):
        from .thrl_course_env import THRLCourseEnv
        return THRLCourseEnv(**kwargs)
    if kwargs.get('start_bank'):
        from .thrl_segment_env import THRLSegmentEnv
        return THRLSegmentEnv(**kwargs)
    from .thrl_native_env import THRLNativeEnv
    return THRLNativeEnv(**kwargs)


# Captured and explicitly pickled into each spawn. Tests may replace this with
# an importable module-level fake factory; they never construct a native game.
worker_factory = _native_factory


class THRLVectorError(RuntimeError):
    def __init__(self, message, failures):
        super().__init__(message + "\n" + "\n".join(
            row.get("traceback", row.get("error", repr(row))) for row in failures))
        self.failures = failures


def _filetime(handle):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel.GetProcessTimes
    query.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    query.restype = wintypes.BOOL
    values = [wintypes.FILETIME() for _ in range(4)]
    if not query(handle, *(ctypes.byref(value) for value in values)):
        raise ctypes.WinError(ctypes.get_last_error())
    return (values[0].dwHighDateTime << 32) | values[0].dwLowDateTime


def _identity(pid, *, owned_handle=None):
    if pid is None:
        return None
    result = {"pid": int(pid), "create_time": None, "creation_filetime": None}
    try:
        process = psutil.Process(int(pid))
        result.update(create_time=process.create_time(), executable=process.exe())
        if os.name == "nt":
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.CloseHandle.argtypes, kernel.CloseHandle.restype = [wintypes.HANDLE], wintypes.BOOL
            handle = owned_handle or kernel.OpenProcess(0x1000, False, int(pid))
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                result["creation_filetime"] = _filetime(handle)
                handle_created = (result["creation_filetime"] - 116444736000000000) / 10000000.
                if abs(result["create_time"] - handle_created) > .000001:
                    result["pid_currently_reused"] = True
                    result["executable"] = None
                result["create_time"] = handle_created
            finally:
                if owned_handle is None:
                    kernel.CloseHandle(handle)
        result["identity_complete"] = True
    except (psutil.Error, OSError, ValueError) as error:
        result.update(identity_complete=False, error=repr(error))
    return result


def _same_identity(expected, current):
    if not expected or not current or expected.get("pid") != current.get("pid"):
        return False
    if not expected.get("identity_complete") or not current.get("identity_complete"):
        return False
    if expected.get("creation_filetime") is not None:
        return expected["creation_filetime"] == current.get("creation_filetime")
    return expected.get("create_time") is not None and expected["create_time"] == current.get("create_time")


def _finish_owned_worker(process, expected, record):
    """Join the owned process HANDLE; never act on an unverified numeric PID.

    A worker can finish naturally after is_alive() but before identity lookup.
    Rejoin the owned HANDLE before classifying that normal exit as a failure.
    """
    process.join(timeout=.2)
    if not process.is_alive():
        return None
    if _same_identity(expected, _identity(process.pid)):
        process.terminate()
        process.join(timeout=2.)
        record['forced_worker_close'] = True
        return {'error':'Owned worker still alive after bounded termination'} if process.is_alive() else None
    process.join(timeout=.2)
    if not process.is_alive():
        record['natural_exit_during_identity_check'] = True
        return None
    return {'error':'Worker identity unknown/reused; not terminated'}


def _native_identity(env):
    controller = getattr(env, "controller", None)
    memory = getattr(controller, "_memory", None)
    return _identity(getattr(env, "pid", None), owned_handle=getattr(memory, "handle", None))


def _terminate_native(identity):
    """Only an exact recorded identity. Windows verifies and kills ONE handle."""
    if not identity or not identity.get("identity_complete"):
        return {"action": "skipped-unknown-native-identity", "identity": identity}
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes, kernel.CloseHandle.restype = [wintypes.HANDLE], wintypes.BOOL
        kernel.TerminateProcess.argtypes, kernel.TerminateProcess.restype = [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        handle = kernel.OpenProcess(0x1000 | 1 | 0x100000, False, identity["pid"])
        if not handle:
            return {"action": "native-absent-or-unavailable", "identity": identity,
                    "winerror": ctypes.get_last_error()}
        try:
            if identity.get("creation_filetime") != _filetime(handle):
                return {"action": "skipped-reused-native-pid", "identity": identity}
            if not kernel.TerminateProcess(handle, 1):
                raise ctypes.WinError(ctypes.get_last_error())
            waited = int(kernel.WaitForSingleObject(handle, 2000))
            return {"action": "terminated-exact-native-handle", "identity": identity,
                    "wait_result": waited, "exit_confirmed": waited == 0}
        finally:
            kernel.CloseHandle(handle)
    # Portability for CPU tests. Actual native games are supported on Windows.
    process = psutil.Process(identity["pid"])
    if not _same_identity(identity, _identity(process.pid)):
        return {"action": "skipped-reused-native-pid", "identity": identity}
    process.terminate()  # psutil also checks its cached PID/create_time.
    return {"action": "terminated-exact-native-identity", "identity": identity}


class _GuardedDevice:
    def __init__(self, device, identity, actions):
        self.device, self.identity, self.actions = device, identity, actions

    def __getattr__(self, name):
        return getattr(self.device, name)

    def kill(self, pid):
        if self.identity and self.identity.get("pid") == pid:
            self.actions.append(_terminate_native(self.identity))
        else:
            self.actions.append({"action": "skipped-unproven-native-pid", "pid": pid})


def _close_env(env, known_native):
    """Retain env.close ordering while guarding its PID-only device.kill call."""
    actions = []
    controller = getattr(env, "controller", None)
    device = getattr(controller, "_device", None)
    if device is not None:
        # Other close actions still detach exact owned sessions and close the
        # existing kill-on-close Job object; no hook/input semantics are changed.
        controller._device = _GuardedDevice(device, known_native, actions)
    env.close()
    return actions


def _observe_owned_creation(env, callback, native_actions):
    """Observe the existing spawn-to-own boundary, including startup failures.

    No extra attach, input poll, game-state read or hook is introduced. The
    instance wrapper also survives the controller's in-place __init__ Retry.
    """
    original_factory = getattr(env, "_create_controller", None)
    if original_factory is None:
        return  # Pure CPU fake environments have no native controller.

    def observe_controller(controller):
        original_own = controller._own_process

        def own_process():
            identity = _identity(controller.pid)
            controller._device = _GuardedDevice(controller._device, identity, native_actions)
            callback(identity)
            result = original_own()  # Existing Job object / startup isolation.
            if not identity.get("identity_complete"):
                raise RuntimeError("Cannot establish the spawned native PID/creation identity")
            return result

        controller._own_process = own_process
        return controller

    def create_controller():
        return observe_controller(original_factory())

    env._create_controller = create_controller
    # Native course preparation also owns every process it creates, including
    # its independent verification resets. Reuse this exact identity boundary.
    env._course_observe_controller = observe_controller


def _small_info(info):
    result = dict(info)
    result.pop("terminal_observation", None)
    result.pop('_policy_burn_in',None)
    return result


def _worker(index, connection, descriptors, kwargs, factory):
    segments, rows, env = {}, {}, None
    sequence = 0
    native = None
    native_actions = []
    worker_identity = _identity(os.getpid())
    try:
        for key, spec in descriptors.items():
            segment = SharedMemory(name=spec["name"])
            segments[key] = segment
            rows[key] = np.ndarray(tuple(spec["shape"]), dtype=np.float32, buffer=segment.buf)[index]
        env = factory(kwargs)

        def native_created(identity):
            nonlocal native
            native = identity
            connection.send({"kind": "ownership", "sequence": sequence, "index": index,
                "worker_identity": worker_identity, "native_identity": identity})

        _observe_owned_creation(env, native_created, native_actions)
        connection.send({"kind": "ready", "sequence": 0, "index": index,
                         "worker_identity": worker_identity, "native_identity": None})
        while True:
            command = connection.recv()
            sequence, operation = command["sequence"], command["operation"]
            if operation == "close":
                break
            if operation == "reset":
                observation, info = (env.reset(options=command['options']) if 'options' in command else env.reset())
                payload = {"info": _small_info(info)}
                if info.get('_policy_burn_in') is not None:payload['burn_in']=info['_policy_burn_in']
            elif operation == "step":
                observation, reward, terminated, truncated, info = env.step(command["action"])
                if not math.isfinite(float(reward)) or bool(terminated) and bool(truncated):
                    raise ValueError("Invalid native reward or conflicting termination flags")
                payload = {"info": _small_info(info), "reward": float(reward),
                           "terminated": bool(terminated), "truncated": bool(truncated)}
                vector = np.asarray(payload["info"].get("reward_vector"))
                if vector.shape != (3,) or not np.isfinite(vector).all():
                    raise ValueError("Native info must contain a finite reward_vector[3]")
            else:
                raise ValueError(f"Unknown vector operation {operation!r}")
            if set(observation)!=set(rows):
                raise ValueError('Worker observation modalities differ from the declared shared schema')
            for key, destination in rows.items():
                shape=destination.shape
                values = observation[key]
                if (not isinstance(values, np.ndarray) or values.dtype != np.float32
                        or values.shape != shape or not np.isfinite(values).all()):
                    raise ValueError(f"Invalid native shared observation {key}")
                np.copyto(rows[key], values, casting="no")
            native = _native_identity(env)
            connection.send({"kind": operation, "sequence": sequence, "index": index,
                "worker_identity": worker_identity, "native_identity": native, **payload})
    except EOFError:
        pass  # Parent's pipe closed: finally still closes this owned game.
    except BaseException as error:
        try:
            connection.send({"kind": "error", "sequence": sequence, "index": index,
                "worker_identity": worker_identity, "native_identity": native,
                "error": repr(error), "traceback": traceback.format_exc(),
                "last_failure": getattr(env, "last_failure", None)})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        cleanup = {"env_close_called": env is not None, "env_close_error": None}
        try:
            if env is not None:
                # Existing owned memory HANDLE, when available, supplies the
                # original creation even if the numeric PID has been reused.
                current = _native_identity(env)
                if current and current.get("identity_complete"):
                    native = current
                cleanup["native_actions"] = _close_env(env, native)
                cleanup["earlier_guarded_native_actions"] = native_actions
        except BaseException:
            cleanup["env_close_error"] = traceback.format_exc()
        rows.clear()
        for segment in segments.values():
            segment.close()  # Always after env.close, never unlink parent's map.
        try:
            connection.send({"kind": "closed", "sequence": sequence, "index": index,
                "worker_identity": worker_identity, "native_identity": native, "cleanup": cleanup})
        except (BrokenPipeError, EOFError, OSError):
            pass
        connection.close()


class THRLNativeVector:
    """Synchronously advance independent native workers; never auto-reset.

    reset_at(indices) returns infos in the supplied index order, NOT all rows.
    A finished row must be explicitly reset before the next full-vector step.
    After any worker error/timeout only close() is allowed; no implicit retry.
    """
    def __init__(self, env_kwargs: list[dict], output: Path, timeout=90):
        if not env_kwargs or not all(isinstance(row, dict) for row in env_kwargs):
            raise ValueError("env_kwargs must be a nonempty list of dictionaries")
        if isinstance(timeout, bool) or not math.isfinite(float(timeout)) or float(timeout) <= 0:
            raise ValueError("timeout must be positive and finite")
        private = [os.path.normcase(str(Path(row["private_appdata"]).resolve())) for row in env_kwargs]
        if len(set(private)) != len(private):
            raise ValueError("Every native worker requires a distinct private_appdata")
        schemas={row.get('input_schema',SEPARATED_INPUT) for row in env_kwargs}
        if len(schemas)!=1:
            raise ValueError('One vector must share exactly one input schema')
        self.shapes=observation_shapes(schemas.pop())
        self.num_envs, self.timeout = len(env_kwargs), float(timeout)
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.ownership_path = self.output / "ownership.json"
        if self.ownership_path.exists():
            raise FileExistsError("Refusing to overwrite existing native ownership metadata")
        self._closed = self._failed = False
        self._sequence = 0
        self._inflight = {}
        self.burn_in = [None]*self.num_envs
        self._segments, self._arrays = {}, {}
        self._processes, self._connections, self._records = [], [], []
        self._needs_reset = np.ones(self.num_envs, dtype=bool)
        self.failures, self.close_errors = [], []
        self._owner_identity = _identity(os.getpid())
        context = mp.get_context("spawn")
        try:
            descriptors = {}
            for key, shape in self.shapes.items():
                full_shape = (self.num_envs, *shape)
                segment = SharedMemory(create=True, size=int(np.prod(full_shape)) * 4)
                self._segments[key] = segment
                self._arrays[key] = np.ndarray(full_shape, dtype=np.float32, buffer=segment.buf)
                self._arrays[key].fill(0.)
                descriptors[key] = {"name": segment.name, "shape": full_shape, "dtype": "float32"}
            self._descriptors = descriptors
            factory = worker_factory
            for index, kwargs in enumerate(env_kwargs):
                parent, child = context.Pipe()
                process = context.Process(target=_worker,
                    args=(index, child, descriptors, dict(kwargs), factory), name=f"thrl-native-{index}")
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                self._processes.append(process)
                self._connections.append(parent)
                self._records.append({"index": index, "worker_identity": _identity(process.pid),
                    "native_identity": None, "native_history": [], "private_appdata": private[index]})
            self._persist()
            self._gather(list(range(self.num_envs)), 0, "ready")
        except BaseException:
            self._failed = True
            self.close()
            raise

    @property
    def observation(self):
        if self._closed:
            raise RuntimeError("Native vector is closed")
        result = {key: values.view() for key, values in self._arrays.items()}
        for values in result.values():
            values.setflags(write=False)
        return result

    def _persist(self):
        rows = []
        for record in self._records:
            row = dict(record)
            worker, native = row.get("worker_identity") or {}, row.get("native_identity") or {}
            row.update(worker_pid=worker.get("pid"), worker_create_time=worker.get("create_time"),
                       native_pid=native.get("pid"), native_create_time=native.get("create_time"))
            rows.append(row)
        payload = {"schema": "thrl-native-vector-ownership-v1",
            "status": "closed" if self._closed else "failed" if self._failed else "active",
            "owner": self._owner_identity, "workers": rows,
            "shared_memory": getattr(self, "_descriptors", {}),
            "failures": self.failures, "close_errors": self.close_errors}
        # Exceptional protocol responses can contain tiny NumPy reward arrays;
        # retain their evidence without masking the original worker exception.
        atomic_write_json(self.ownership_path, json.loads(json.dumps(payload,
            default=lambda value: value.tolist() if isinstance(value, (np.ndarray, np.generic)) else repr(value))))

    def _note(self, index, message):
        record = self._records[index]
        if not _same_identity(record["worker_identity"], message.get("worker_identity")):
            raise ValueError("Worker PID/creation identity differs from the spawned child")
        native = message.get("native_identity")
        changed = native is not None and native != record.get("native_identity")
        if changed:
            record["native_identity"] = native
            if native not in record["native_history"]:
                record["native_history"].append(native)
        if "cleanup" in message:
            record["cleanup"] = message["cleanup"]
            if message["cleanup"].get("env_close_error"):
                self.close_errors.append({"index": index, **message["cleanup"]})
        if changed or "cleanup" in message:
            self._persist()

    def _gather(self, indices, sequence, kind):
        pending = set(indices)
        result, failures = {}, []
        deadline = time.monotonic() + self.timeout
        while pending:
            remaining = deadline - time.monotonic()
            ready = wait([self._connections[i] for i in pending], max(0., remaining)) if remaining > 0 else []
            if not ready:
                failures.append({"error": f"Native vector {kind} timeout", "pending_workers": sorted(pending),
                                 "sequence": sequence, "timeout": self.timeout})
                break
            for connection in ready:
                index = self._connections.index(connection)
                try:
                    message = connection.recv()
                    self._note(index, message)
                    if message.get("kind") == "ownership" and message.get("sequence") == sequence:
                        continue
                    if message.get("kind") == "error":
                        failures.append(message)
                    elif message.get("sequence") != sequence or message.get("kind") != kind:
                        failures.append({"index": index, "error": "Unexpected worker response", "response": message})
                    else:
                        result[index] = message
                except (EOFError, OSError, ValueError) as error:
                    failures.append({"index": index, "error": repr(error), "traceback": traceback.format_exc()})
                pending.remove(index)
        if failures:
            self._failed = True
            self.failures.extend(failures)
            self._persist()
            raise THRLVectorError(f"Native vector {kind} failed; no retry/reset performed", failures)
        return [result[index] for index in indices]

    def _exchange(self, indices, operation, actions=None, options=None):
        if self._inflight:
            raise RuntimeError('Cannot mix synchronous exchanges with outstanding ready-driven commands')
        if self._closed or self._failed:
            raise RuntimeError("Native vector is closed/failed; only close is allowed")
        self._sequence += 1
        try:
            # Submit every worker's action BEFORE waiting for any response.
            for position, index in enumerate(indices):
                command = {"operation": operation, "sequence": self._sequence}
                if actions is not None:
                    command["action"] = actions[position]
                if options is not None:command['options']=options[position]
                self._connections[index].send(command)
        except (EOFError, OSError) as error:
            self._failed = True
            failure = {"error": repr(error), "traceback": traceback.format_exc(), "operation": operation}
            self.failures.append(failure)
            self._persist()
            raise THRLVectorError("Native command delivery failed", [failure]) from error
        return self._gather(indices, self._sequence, operation)

    def submit_at(self, indices, operation, actions=None, options=None):
        """Start independent commands; each world has at most one in flight."""
        if self._closed or self._failed:raise RuntimeError('Native vector is closed/failed')
        indices=list(indices)
        if operation not in ('step','reset') or not indices or len(set(indices))!=len(indices):
            raise ValueError('Explicit distinct ready-driven step/reset indices required')
        if any(type(i) is not int or not 0<=i<self.num_envs or i in self._inflight for i in indices):
            raise ValueError('Invalid or busy ready-driven worker')
        if operation=='step':
            if actions is None or len(actions)!=len(indices) or any(
                    isinstance(a,(bool,np.bool_)) or not isinstance(a,Integral) or not 0<=a<36 for a in actions):
                raise ValueError('One valid action required for each ready world')
            if self._needs_reset[indices].any():raise RuntimeError('Finished world requires explicit reset')
        if options is not None and (operation!='reset' or len(options)!=len(indices)
                                   or not all(isinstance(row,dict) for row in options)):
            raise ValueError('Reset options must match the explicit world indices')
        self._sequence+=1
        try:
            for position,index in enumerate(indices):
                command={'operation':operation,'sequence':self._sequence}
                if operation=='step':command['action']=int(actions[position])
                if options is not None:command['options']=options[position]
                self._connections[index].send(command)
                self._inflight[index]={'operation':operation,'sequence':self._sequence,
                                       'started':time.monotonic()}
        except (EOFError,OSError) as error:
            self._failed=True;self.failures.append({'error':repr(error),'operation':operation})
            self._persist();raise THRLVectorError('Ready-driven dispatch failed',self.failures) from error

    def receive_ready(self, *, max_wait_ms=0., min_batch=1, check_stop=None,
                      copy_observations=True):
        """Return ready copies, optionally coalescing steps for a bounded window.

        The default retains immediate delivery. Once the first completed
        response arrives, only outstanding *steps* may extend the small batch;
        slow resets are never awaited to fill it. The deadline is not restarted
        when another response arrives. Native timeout and ownership checks stay
        per command. A caller stop callback is polled during long native waits.

        The default owns observation copies. The ready collector may explicitly
        borrow read-only rows instead: each acknowledged row is stable until
        that same world's next command. A borrowed row must not be retained as
        an immutable rollout sample or read again while that world is in flight.
        """
        if type(copy_observations) is not bool:
            raise ValueError('Explicit boolean observation-copy mode required')
        if (isinstance(max_wait_ms,bool) or not isinstance(max_wait_ms,(int,float))
                or not math.isfinite(max_wait_ms) or not 0<=max_wait_ms<=2):
            raise ValueError('Ready coalescing window must be finite in [0,2] milliseconds')
        if type(min_batch) is not int or not 1<=min_batch<=100:
            raise ValueError('Ready minimum batch must be in [1,100]')
        if not self._inflight or self._closed or self._failed:
            raise RuntimeError('No active ready-driven commands')
        result=[];coalesce_deadline=None;coalesce_wait=0.;window_exhausted=False

        def finish():
            self.last_receive_timing=dict(max_wait_ms=float(max_wait_ms),min_batch=min_batch,
                ready_count=len(result),coalesce_wait_seconds=coalesce_wait,
                window_exhausted=window_exhausted,
                copied_observations=copy_observations,
                reset_responses=sum(row['operation']=='reset' for row in result))
            return result

        while True:
            if check_stop is not None:check_stop()
            if result:
                if max_wait_ms==0 or len(result)>=min_batch or not self._inflight:return finish()
                if time.monotonic()>=coalesce_deadline:
                    window_exhausted=True;return finish()
                eligible=[i for i,item in self._inflight.items() if item['operation']=='step']
                if not eligible:return finish()
            else:eligible=list(self._inflight)
            earliest=min(item['started']+self.timeout for item in self._inflight.values())
            deadline=min(earliest,coalesce_deadline) if result else earliest
            remaining=max(0.,deadline-time.monotonic())
            if check_stop is not None:remaining=min(remaining,.05)
            wait_started=time.monotonic()
            connections=wait([self._connections[i] for i in eligible],remaining)
            if result:coalesce_wait+=time.monotonic()-wait_started
            if not connections:
                now=time.monotonic()
                if result and now>=coalesce_deadline:
                    window_exhausted=True;return finish()
                if now<earliest:continue
                self._failed=True
                failure={'error':'Ready-driven native timeout','pending_workers':sorted(self._inflight)}
                self.failures.append(failure);self._persist()
                raise THRLVectorError('Ready-driven native timeout',[failure])
            for connection in connections:
                index=self._connections.index(connection);expected=self._inflight[index]
                try:
                    message=connection.recv();self._note(index,message)
                    if message.get('kind')=='ownership' and message.get('sequence')==expected['sequence']:continue
                    if message.get('kind')=='error':raise THRLVectorError('Native worker failed',[message])
                    if message.get('sequence')!=expected['sequence'] or message.get('kind')!=expected['operation']:
                        raise ValueError('Ready-driven native response sequence/operation differs')
                    self._needs_reset[index]=(bool(message['terminated'] or message['truncated'])
                        if expected['operation']=='step' else False)
                    if expected['operation']=='reset':self.burn_in[index]=message.get('burn_in')
                    message['observation']={key:(row[index].copy() if copy_observations else row[index].view())
                        for key,row in self._arrays.items()}
                    if not copy_observations:
                        for value in message['observation'].values():value.setflags(write=False)
                    message['operation']=expected['operation']
                    result.append(message);del self._inflight[index]
                    if coalesce_deadline is None:
                        coalesce_deadline=time.monotonic()+max_wait_ms/1000.
                except (EOFError,OSError,ValueError,THRLVectorError) as error:
                    self._failed=True
                    failures=error.failures if isinstance(error,THRLVectorError) else [{'index':index,'error':repr(error)}]
                    self.failures.extend(failures);self._persist()
                    raise THRLVectorError('Ready-driven native receive failed',failures) from error

    def reset(self, options=None):
        infos = self.reset_at(list(range(self.num_envs)),options=options)
        return self.observation, infos

    def reset_at(self, indices, options=None):
        indices = list(indices)
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
               or not 0 <= i < self.num_envs for i in indices) or len(set(indices)) != len(indices):
            raise ValueError("reset_at indices must be distinct valid worker integers")
        indices = [int(i) for i in indices]
        if not indices:
            return []
        if options is not None and (len(options)!=len(indices) or not all(isinstance(row,dict) for row in options)):
            raise ValueError('Reset options must match explicit indices')
        messages = self._exchange(indices, "reset",options=options)
        self._needs_reset[indices] = False
        for index,message in zip(indices,messages):self.burn_in[index]=message.get('burn_in')
        return [message["info"] for message in messages]

    def step(self, actions):
        actions = list(actions)
        if len(actions) != self.num_envs or any(isinstance(a, (bool, np.bool_))
                or not isinstance(a, Integral) or not 0 <= a < 36 for a in actions):
            raise ValueError("actions must contain one integer in [0,36) per worker")
        if self._needs_reset.any():
            raise RuntimeError("Explicit reset_at is required for all finished/uninitialized rows")
        messages = self._exchange(list(range(self.num_envs)), "step", [int(a) for a in actions])
        terminated = np.asarray([row["terminated"] for row in messages], dtype=bool)
        truncated = np.asarray([row["truncated"] for row in messages], dtype=bool)
        self._needs_reset = terminated | truncated
        return (self.observation, np.asarray([row["reward"] for row in messages], dtype=np.float32),
                terminated, truncated, [row["info"] for row in messages])

    def close(self):
        if self._closed:
            return
        self._sequence += 1
        # An exited/error worker can still have its final env.close receipt in
        # the pipe. Drain it even when is_alive() has already become False.
        pending = set(range(len(self._processes)))
        for index, (process, connection) in enumerate(zip(self._processes, self._connections)):
            if process.is_alive():
                try:
                    connection.send({"operation": "close", "sequence": self._sequence})
                except (EOFError, OSError):
                    pass
        deadline = time.monotonic() + min(5., self.timeout)
        while pending and time.monotonic() < deadline:
            ready = wait([self._connections[i] for i in pending], max(0., deadline - time.monotonic()))
            if not ready:
                break
            for connection in ready:
                index = self._connections.index(connection)
                try:
                    message = connection.recv()
                    self._note(index, message)
                    if message.get("kind") == "error" and message not in self.failures:
                        self._failed = True
                        self.failures.append(message)
                    if message.get("kind") == "closed":
                        pending.remove(index)
                except EOFError:
                    self._records[index]["pipe_closed_without_cleanup_receipt"] = True
                    pending.remove(index)
                except (OSError, ValueError) as error:
                    self.close_errors.append({"index": index, "error": repr(error)})
                    pending.remove(index)
        for index, process in enumerate(self._processes):
            error = _finish_owned_worker(process, self._records[index]['worker_identity'], self._records[index])
            if error:
                self.close_errors.append({'index':index, **error})
            for native in self._records[index]["native_history"]:
                if _same_identity(native, _identity(native.get("pid"))):
                    try:
                        self._records[index].setdefault("parent_native_cleanup", []).append(_terminate_native(native))
                    except (psutil.Error, OSError) as error:
                        self.close_errors.append({"index": index, "error": repr(error)})
                current = _identity(native.get("pid"))
                self._records[index].setdefault("native_exit_checks", []).append({
                    "expected": native, "current": current,
                    "same_identity_still_live": _same_identity(native, current)})
            self._records[index].update(worker_exitcode=process.exitcode,
                                       worker_alive_after_close=process.is_alive())
        for connection in self._connections:
            connection.close()
        self._arrays.clear()
        for segment in self._segments.values():
            segment.close()
            segment.unlink()
        self._closed = True
        self._persist()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
