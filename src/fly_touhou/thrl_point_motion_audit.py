"""Optional bounded worker-local point telemetry; never a policy input.

Every acknowledged snapshot advances its own history, while JSONL is throttled
and contains at most eight recent changed objects. All failures are contained
inside telemetry. No controller, reward, teacher, IPC payload, or GPU calls.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
import json
import math
from pathlib import Path
import time
import uuid

from .atomic_json_io import atomic_write_json
from .thrl_point_motion_history import PointMotionHistory


class PointMotionAudit:
    def __init__(self, path, *, expected_frame_step=4, interval_frames=60,
                 max_objects=8, max_records=4096, max_bytes=8 * 1024 ** 2,
                 radius_rate_threshold=1e-4, velocity_rate_threshold=1e-4):
        self.path = Path(path)
        self.interval_frames, self.max_objects = interval_frames, max_objects
        self.max_records, self.max_bytes = max_records, max_bytes
        self.radius_rate_threshold, self.velocity_rate_threshold = radius_rate_threshold, velocity_rate_threshold
        self._stream = None
        self._closed = False
        self.disabled_reason = None
        self.last_error = None
        self.snapshots_seen = self.snapshots_measured = self.records_written = self.bytes_written = 0
        self.records_at_open = self.bytes_at_open = self.validation_errors = self.throttled = self.pending_evictions = 0
        self.total_ns = self.max_ns = 0
        self.write_attempts = self.write_ns = self.maximum_write_ns = 0
        self.known_pairs = self.unknown_pairs = 0
        self.duplicate_acknowledged_snapshots = 0
        self.unknown_reasons, self.changes, self.stages = Counter(), Counter(), Counter()
        self._instance = uuid.uuid4().hex
        self._generation = 0
        self._episode = None
        self._world = None
        self._pid = None
        self._pending = OrderedDict()
        self._previous_points = {}
        self._previous_frame = None
        self._last_stage = self._last_protection = None
        self._last_write_frame = None
        self._force_context = True
        self._last_result = None
        try:
            if (type(interval_frames) is not int or interval_frames < 1
                    or type(max_objects) is not int or not 1 <= max_objects <= 8
                    or type(max_records) is not int or not 1 <= max_records <= 4096
                    or type(max_bytes) is not int or not 1 <= max_bytes <= 8 * 1024 ** 2
                    or not 0 < radius_rate_threshold < float("inf")
                    or not 0 < velocity_rate_threshold < float("inf")):
                raise ValueError("Bounded passive point telemetry configuration required")
            self.history = PointMotionHistory(expected_frame_step=expected_frame_step)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                self.bytes_at_open = self.path.stat().st_size
                if self.bytes_at_open > self.max_bytes:
                    self._disable("byte_budget_already_reached")
                    return
                with self.path.open("rb") as stream:
                    self.records_at_open = sum(chunk.count(b"\n") for chunk in iter(lambda: stream.read(65536), b""))
            if self.records_at_open >= self.max_records or self.bytes_at_open >= self.max_bytes:
                self._disable("existing_file_budget_reached")
                return
            self._stream = self.path.open("a", encoding="utf-8", buffering=1)
        except Exception as error:
            self._disable("initialization_or_io_error", error)

    def _disable(self, reason, error=None):
        self.disabled_reason = reason
        if error is not None:
            self.last_error = (type(error).__name__ + ": " + str(error))[:512]
        self.reset_history()
        try:
            if self._stream is not None:
                self._stream.close()
        except Exception:
            pass
        self._stream = None

    def reset_history(self):
        self._break_history()
        self._last_stage = self._last_protection = None
        self._last_write_frame = None
        self._force_context = True

    def _break_history(self):
        if hasattr(self, "history"):
            self.history.reset()
        self._previous_points = {}
        self._previous_frame = None
        self._pending.clear()
        self._last_result = None

    def begin_episode(self, *, native_episode, native_pid):
        if self._closed or self.disabled_reason:
            return
        self.reset_history()
        self._generation += 1
        self._episode, self._pid = native_episode, native_pid
        # New owned-run generation even when native PID/addresses/frame repeat.
        self._world = f"{self._instance}:generation{self._generation}:pid{native_pid}"

    @staticmethod
    def _protection(snapshot):
        p = snapshot.player
        return None if p is None else (p.state, p.invulnerable_frames > 0, p.deathbomb_timer > 0)

    def _remember(self, measurement, previous, frame, kinds):
        point = measurement.current
        event = dict(native_id=point.identity, native_kind=point.kind, measured_frame=frame,
                     native_age=point.native_timer, native_state=point.native_state, native_flags=point.native_flags,
                     shape=point.geometry_kind, radius=point.effective_radius, raw_hitbox_value=point.raw_hitbox_value,
                     scale=point.native_scale, scale_applies=point.scale_applies,
                     position=list(point.position), stored_velocity=list(point.stored_velocity),
                     dr_dt=measurement.dr_dt, dv_dt=measurement.dv_dt,
                     observed_center_velocity=measurement.observed_center_velocity,
                     known=measurement.known, unknown_reason=measurement.unknown_reason,
                     elapsed_native_frames=measurement.elapsed_native_frames, change_kinds=kinds,
                     prior_same_address_sample=None if previous is None else dict(
                         frame=self._previous_frame, radius=previous.effective_radius,
                         native_age=previous.native_timer, stored_velocity=list(previous.stored_velocity)))
        self._pending[point.identity] = event
        self._pending.move_to_end(point.identity)
        if len(self._pending) > self.max_objects:
            # Latest frame first; simultaneous real derivative changes take
            # precedence over new-object baselines, without a subtype rule.
            evict = min(self._pending, key=lambda key: (
                self._pending[key]["measured_frame"],
                int(any(name in self._pending[key]["change_kinds"] for name in ("radius_rate", "stored_velocity_rate"))),
            ))
            del self._pending[evict]
            self.pending_evictions += 1

    def _write(self, result, protection, *, forced):
        if self.records_at_open + self.records_written >= self.max_records:
            self._disable("record_budget_reached")
            return
        objects = sorted(self._pending.values(), key=lambda event: event["measured_frame"], reverse=True)
        record = dict(record_kind="passive_point_motion", world_id=self._world, native_pid=self._pid,
                      native_episode=self._episode, frame=result.frame, stage_frame=result.stage_frame, stage=result.stage,
                      forced_context_record=forced, player_protection_state=protection,
                      point_count=len(result.points), active_state1_count=sum(row.current.native_state == 1 for row in result.points),
                      known_points=sum(row.known for row in result.points), history_barrier_reason=result.history_barrier_reason,
                      objects=objects, max_objects=self.max_objects, future_motion_guaranteed=False,
                      native_collision_truth_claimed=False, policy_input_changed=False, actions_changed=False)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        size = len(line.encode("utf-8"))
        if self.bytes_at_open + self.bytes_written + size > self.max_bytes:
            self._disable("byte_budget_reached")
            return
        started = time.perf_counter_ns()
        self.write_attempts += 1
        try:self._stream.write(line)
        finally:
            elapsed = time.perf_counter_ns() - started
            self.write_ns += elapsed
            self.maximum_write_ns = max(self.maximum_write_ns, elapsed)
        self.records_written += 1
        self.bytes_written += size
        self._last_write_frame = result.frame
        self._pending.clear()

    def record(self, snapshot, *, native_episode):
        self.snapshots_seen += 1
        if self._closed or self.disabled_reason:
            return
        started = time.perf_counter_ns()
        try:
            if self._episode != native_episode or self._world is None:
                raise ValueError("begin_episode with a native world generation is required")
            result = self.history.observe(snapshot, world_id=self._world, native_episode=native_episode)
            if result is self._last_result:
                self.duplicate_acknowledged_snapshots += 1
                return
            protection = self._protection(snapshot)
            forced = self._force_context or result.stage != self._last_stage or protection != self._last_protection
            if self._last_stage is not None and result.stage != self._last_stage:
                self.pending_evictions += len(self._pending)
                self._pending.clear()
                self._previous_points = {}
                self._previous_frame = None
            self.snapshots_measured += 1
            self.stages[str(result.stage)] += 1
            self.known_pairs += sum(row.known for row in result.points)
            self.unknown_pairs += sum(not row.known for row in result.points)
            current = {row.current.identity: row.current for row in result.points}
            for row in result.points:
                point, kinds = row.current, []
                previous = self._previous_points.get(point.identity)
                if row.known and abs(row.dr_dt) >= self.radius_rate_threshold:
                    kinds.append("radius_rate")
                if row.known and math.hypot(*row.dv_dt) >= self.velocity_rate_threshold:
                    kinds.append("stored_velocity_rate")
                if not row.known:
                    self.unknown_reasons[row.unknown_reason] += 1
                if previous is None:
                    kinds.append("new_or_reappeared_address")
                elif point.mode_signature != previous.mode_signature:
                    kinds.append("active_or_native_mode_change")
                elif not row.known and (point.effective_radius != previous.effective_radius
                                        or point.stored_velocity != previous.stored_velocity):
                    kinds.append("raw_difference_with_unknown_continuity")
                if kinds:
                    self.changes.update(kinds)
                    self._remember(row, previous, result.frame, kinds)
            removed = len(set(self._previous_points) - set(current))
            if removed:
                self.changes["disappeared_addresses"] += removed
            ready = self._last_write_frame is None or result.frame - self._last_write_frame >= self.interval_frames
            if forced or (ready and (self._pending or removed)):
                self._write(result, protection, forced=forced)
            elif self._pending or removed:
                self.throttled += 1
            self._previous_points, self._previous_frame = current, result.frame
            self._last_stage, self._last_protection = result.stage, protection
            self._force_context = False
            self._last_result = result
        except ValueError as error:
            self.validation_errors += 1
            self.last_error = (type(error).__name__ + ": " + str(error))[:512]
            self._break_history()
        except OSError as error:
            self._disable("io_error", error)
        except Exception as error:
            self._disable("observer_error", error)
        finally:
            elapsed = time.perf_counter_ns() - started
            self.total_ns += elapsed
            self.max_ns = max(self.max_ns, elapsed)

    def summary(self):
        return dict(record_kind="passive_point_motion_summary", path=str(self.path),
                    snapshots_seen=self.snapshots_seen, snapshots_measured=self.snapshots_measured,
                    records_written=self.records_written, bytes_written=self.bytes_written,
                    existing_records_at_open=self.records_at_open, existing_bytes_at_open=self.bytes_at_open,
                    interval_native_frames=self.interval_frames, max_objects=self.max_objects,
                    max_records=self.max_records, max_bytes=self.max_bytes,
                    radius_rate_threshold=self.radius_rate_threshold, velocity_rate_threshold=self.velocity_rate_threshold,
                    disabled_reason=self.disabled_reason, last_error=self.last_error,
                    validation_errors=self.validation_errors, throttled_snapshots=self.throttled,
                    duplicate_acknowledged_snapshots=self.duplicate_acknowledged_snapshots,
                    pending_object_evictions=self.pending_evictions, known_point_pairs=self.known_pairs,
                    unknown_point_pairs=self.unknown_pairs, unknown_reason_counts=dict(self.unknown_reasons),
                    observed_change_counts=dict(self.changes), measured_snapshots_by_stage=dict(self.stages),
                    total_observer_seconds=self.total_ns / 1e9, maximum_observer_seconds=self.max_ns / 1e9,
                    write_attempts=self.write_attempts, total_write_seconds=self.write_ns / 1e9,
                    mean_write_seconds=self.write_ns / max(1,self.write_attempts) / 1e9,
                    maximum_write_seconds=self.maximum_write_ns / 1e9,
                    coverage_scope="accepted measured snapshots until finite file budget/disable; throttled objects are a bounded subset",
                    whole_objects_in_IPC=False, policy_input_changed=False, rewards_changed=False, actions_changed=False,
                    native_collision_truth_claimed=False, future_motion_guaranteed=False)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._stream is not None:
                self._stream.close()
        except Exception as error:
            self.disabled_reason = self.disabled_reason or "close_io_error"
            self.last_error = (type(error).__name__ + ": " + str(error))[:512]
        try:
            atomic_write_json(self.path.with_suffix(".summary.json"), self.summary())
        except Exception as error:
            self.disabled_reason = self.disabled_reason or "summary_io_error"
            self.last_error = (type(error).__name__ + ": " + str(error))[:512]
