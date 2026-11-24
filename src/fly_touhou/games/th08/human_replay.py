"""Read original TH08 v6 replay stage inputs for teacher-data selection."""

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


_HEADER_SIZE = 0x68
_STAGE_HEADER_SIZE = 0x40
_MOVEMENT_BITS = 0xF4


@dataclass(frozen=True)
class ReplayStage:
    raw_stage: int
    rng_seed: int
    inputs: np.ndarray


def _decompress(payload, expected_size):
    buffer = offset = pending = 0
    dictionary = bytearray(8192)
    result = bytearray()

    def bits(width):
        nonlocal buffer, offset, pending
        while pending < width:
            if offset == len(payload):
                raise ValueError("Incomplete TH08 replay input stream")
            buffer = (buffer << 8) | payload[offset]
            offset += 1
            pending += 8
        pending -= width
        value = buffer >> pending
        buffer &= (1 << pending) - 1
        return value

    def output(value):
        dictionary[(len(result) + 1) & 8191] = value
        result.append(value)

    while len(result) < expected_size:
        if bits(1):
            output(bits(8))
        else:
            start = bits(13)
            if start == 0:
                break
            for index in range(bits(4) + 3):
                output(dictionary[(start + index) & 8191])
    if len(result) != expected_size:
        raise ValueError("TH08 replay payload differs from its declared size")
    return result


def read_replay(path):
    """Decode the original frame words. Replay playback proves game outcomes."""
    raw = bytearray(Path(path).read_bytes())
    if len(raw) < _HEADER_SIZE or raw[:6] != b"T8RP\x06\x00":
        raise ValueError("Expected an original TH08 version 6 replay")
    data_end, checksum = struct.unpack_from("<II", raw, 0x0C)
    if not _HEADER_SIZE < data_end <= len(raw):
        raise ValueError("Invalid TH08 replay data end")
    key = raw[0x15]
    for offset in range(0x18, data_end):
        raw[offset] = (raw[offset] - key - 7 * (offset - 0x18)) & 255
    packed, size = struct.unpack_from("<II", raw, 0x18)
    if packed != data_end - _HEADER_SIZE or not 0xCC <= size <= 16 * 1024 * 1024:
        raise ValueError("Invalid TH08 compressed or decoded size")
    if ((0x3F000318 + sum(raw[0x15:data_end])) & 0xFFFFFFFF) != checksum:
        raise ValueError("TH08 replay checksum differs")
    replay = raw[:_HEADER_SIZE] + _decompress(raw[_HEADER_SIZE:data_end], size)
    pointers = struct.unpack_from("<18I", replay, 0x20)
    boundaries = sorted(set(pointers) - {0}) + [len(replay)]
    stages = []
    for raw_stage, offset in enumerate(pointers[:9]):
        if not offset:
            continue
        end = boundaries[boundaries.index(offset) + 1]
        begin = offset + _STAGE_HEADER_SIZE
        if offset < 0x134 or not begin < end <= len(replay) or (end - begin) % 2:
            raise ValueError("Invalid TH08 stage input bounds")
        seed = struct.unpack_from("<H", replay, offset + 0x1A)[0]
        inputs = np.frombuffer(replay[begin:end], dtype="<u2").copy()
        stages.append(ReplayStage(raw_stage, seed, inputs))
    if not stages:
        raise ValueError("TH08 replay contains no stage input stream")
    metadata = dict(replay_version=6, team=replay[0x6A], difficulty=replay[0x6B],
                    spell_practice=bool(replay[0x7B]),
                    recorded_clear_state_code=replay[0x11C],
                    executable_version=bytes(replay[0x12C:0x132]).split(b"\0", 1)[0].decode("ascii"))
    return metadata, stages


def exact_four_frame_actions(inputs):
    """Find potential held commands at every input offset, -1 for unsupported."""
    from ...th08_sensation import POLICY_MASKS

    masks = np.asarray(inputs, dtype=np.uint16)
    if masks.ndim != 1:
        raise ValueError("Expected one original replay stage's frame words")
    if len(masks) < 4:
        return np.empty(0, dtype=np.int8)
    groups = np.lib.stride_tricks.sliding_window_view(masks, 4)
    trailing = groups[:, 1:] & _MOVEMENT_BITS
    possible = (trailing == trailing[:, :1]).all(axis=1)
    possible &= ((groups & 1) != 0).all(axis=1) & ((groups & 2) == 0).all(axis=1)
    lookup = np.full(256, -1, dtype=np.int8)
    lookup[np.asarray(POLICY_MASKS) & _MOVEMENT_BITS] = np.arange(len(POLICY_MASKS), dtype=np.int8)
    return np.where(possible, lookup[trailing[:, 0]], -1).astype(np.int8)


def observed_four_frame_action(motion, inputs, positions):
    """Accept a teacher command only when it reproduces all four native positions."""
    from ...th08_sensation import POLICY_MASKS

    candidates = exact_four_frame_actions(inputs)
    if len(candidates) != 1 or candidates[0] < 0:
        return -1
    action = int(candidates[0])
    predicted = motion.paths([POLICY_MASKS[action]], 4)[0, 1:]
    return action if np.allclose(positions, predicted, rtol=0., atol=3e-5) else -1
