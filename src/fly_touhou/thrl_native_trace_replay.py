"""Verify a recorded no-Bomb action tape against an independent original run.

Comparison uses episode-relative clocks, not startup/menu global frame counts
or process addresses. A matching replay is a prerequisite for meaningful
original-game counterfactual branches; this module claims no unexecuted result.
"""
from pathlib import Path
import hashlib
import numpy as np

from .thrl_study_io import read_shared_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RecordedAvoidanceCase:
    def __init__(self, report_path, row):
        root = Path(report_path).resolve().parent
        if row.get('policy') not in ('trained_sampled', 'trained_greedy'):
            raise ValueError('Only actual captured trained-policy actions are replayable')
        pointer = row['failure_trace']
        trace = Path(pointer['arrays']).resolve()
        receipt_path = Path(pointer['receipt']).resolve()
        trace_root = root / 'failure_traces'
        if not trace.is_relative_to(trace_root) or not receipt_path.is_relative_to(trace_root):
            raise ValueError('Captured input is outside its authoritative report')
        receipt = read_shared_json(receipt_path)
        if (sha256(trace) != pointer['sha256'] or receipt['sha256'] != pointer['sha256']
                or (receipt['policy'], receipt['game']) != (row['policy'], row['game'])):
            raise ValueError('Captured input receipt/provenance differs')
        descriptor = receipt['full_episode_action_tape']
        action_path = Path(descriptor['path']).resolve()
        if not action_path.is_relative_to(trace_root):
            raise ValueError('Action tape is outside its authoritative report')
        actions = action_path.read_bytes()
        if (descriptor['schema'] != 'B1-original-source18-four-tick-action-tape-v1'
                or hashlib.sha256(actions).hexdigest() != descriptor['sha256']
                or descriptor['sha256'] != row['action_sha256']
                or len(actions) != descriptor['decisions']
                or len(actions) != row['policy_decisions']
                or len(actions) != receipt['whole_episode_decisions']
                or not 1 <= len(actions) <= 16384 or max(actions) >= 18):
            raise ValueError('Complete acknowledged source18 action tape required')
        with np.load(trace, allow_pickle=False) as z:
            arrays = {k:z[k].copy() for k in z.files}
        n = len(arrays['action'])
        if (not 1 <= n <= len(actions) or not np.array_equal(
                arrays['action'], np.frombuffer(actions[-n:], dtype=np.uint8))
                or n != receipt['history_rows']):
            raise ValueError('Captured tail does not match the full episode action tape')
        if any(arrays[k].dtype != np.float32 for k in
               ('features', 'objects', 'last_preaction_maps',
                'after_features', 'after_objects', 'after_maps')):
            raise ValueError('Replay comparison needs original FP32 sensor arrays')
        if 'maps' in arrays and (receipt.get('earlier_history_full_map_coverage_known') is not True
                or arrays['maps'].dtype != np.float32 or arrays['maps'].shape != (n,16,92,96)
                or not np.isfinite(arrays['maps']).all()
                or not np.array_equal(arrays['maps'][-1],arrays['last_preaction_maps'])):
            raise ValueError('Declared actual map history must match the entire paired FP32 tail')
        self.row, self.receipt, self.arrays = row, receipt, arrays
        self.actions, self.position = actions, 0
        self.tail_begin, self.tail_count = len(actions)-n, n
        self.checked_tail_rows, self.differences = 0, []
        self.source_trace_sha256, self.action_sha256 = pointer['sha256'], descriptor['sha256']

    def compare_before(self, observation, info):
        if self.position < self.tail_begin:
            return
        offset = self.position - self.tail_begin
        if not 0 <= offset < self.tail_count:
            raise ValueError('Replay requested an action beyond the complete captured tape')
        fields = {k:self.arrays[k][offset] for k in ('features', 'objects')}
        if 'maps' in self.arrays:
            fields['maps'] = self.arrays['maps'][offset]
        elif self.position == len(self.actions)-1:
            fields['maps'] = self.arrays['last_preaction_maps']
        self._compare(observation, fields, 'preaction')
        for source, key in (('episode_frame', 'episode_frames'), ('stage', 'stage'),
                            ('stage_frame', 'stage_frame')):
            expected = int(self.arrays[source][offset])
            if expected >= 0 and info.get(key) != expected:
                self._difference('preaction', key, expected, info.get(key))
        self.checked_tail_rows += 1

    def _difference(self, when, key, expected, actual):
        if len(self.differences) < 32:
            self.differences.append(dict(decision=self.position, when=when,
                field=key, expected=expected, actual=actual))

    def _compare(self, observation, expected_fields, when):
        for key, expected in expected_fields.items():
            actual = np.asarray(observation[key])
            if actual.dtype != expected.dtype or actual.shape != expected.shape:
                self._difference(when, key+'_shape_dtype',
                    [list(expected.shape), str(expected.dtype)],
                    [list(actual.shape), str(actual.dtype)])
            elif not np.array_equal(actual, expected):
                self._difference(when, key, 'bitwise equal original FP32',
                    dict(differing_values=int(np.count_nonzero(actual != expected)),
                         max_absolute_difference=float(np.max(np.abs(actual-expected)))))

    def next_action(self):
        if self.position >= len(self.actions):
            raise ValueError('Replay tape exhausted')
        return self.actions[self.position]

    def acknowledge(self):
        self.position += 1

    def finish(self, observation, info, *, terminal):
        if self.position != len(self.actions):
            self._difference('endpoint', 'acknowledged_decisions', len(self.actions), self.position)
        if self.checked_tail_rows != self.tail_count:
            self._difference('endpoint', 'checked_tail_rows', self.tail_count, self.checked_tail_rows)
        self._compare(observation,
            {k:self.arrays['after_'+k] for k in ('features', 'objects', 'maps')}, 'endpoint')
        actual = info.get('probe_episode', {})
        if not terminal:
            self._difference('endpoint', 'diagnostic_terminal', True, False)
        for key in ('native_seed', 'native_frames', 'stage', 'stage_frame',
                    'first_collision_observed', 'right_censored_at_cap', 'actual_bombs_used'):
            if actual.get(key) != self.row.get(key):
                self._difference('endpoint', key, self.row.get(key), actual.get(key))
        return dict(policy=self.row['policy'], game=self.row['game'],
            evaluation_seed=self.row['evaluation_seed'], native_seed=self.row['native_seed'],
            actions_acknowledged=self.position, expected_decisions=len(self.actions),
            captured_tail_rows=self.tail_count, checked_tail_rows=self.checked_tail_rows,
            original_native_replay_verified=not self.differences,
            differences=self.differences, source_trace_sha256=self.source_trace_sha256,
            action_sha256=self.action_sha256, replay_endpoint=actual,
            native_counterfactuals_executed=False, optimizer_updates=0)
