import struct
import runpy
from pathlib import Path

import numpy as np
import pytest

from fly_touhou.th08_human_replay import (
    ReplayStage,
    _decompress,
    exact_four_frame_actions,
    observed_four_frame_action,
    read_replay,
)
from fly_touhou.th08_motion import TH08Motion
from fly_touhou.th08_sensation import POLICY_MASKS


def pack_bits(bits):
    padded = bits.ljust(8 * ((len(bits) + 7) // 8), "0")
    return bytes(int(padded[i:i + 8], 2) for i in range(0, len(padded), 8))


def test_read_th08_little_endian_input_words_and_stage_pointer(tmp_path):
    decoded = bytearray(0x134 + 0x40 + 8 + 5)
    decoded[:6] = b"T8RP\x06\x00"
    decoded[0x15] = 0x39
    struct.pack_into("<I", decoded, 0x20 + 6 * 4, 0x134)
    struct.pack_into("<I", decoded, 0x20 + 15 * 4, 0x17C)
    decoded[0x6B] = 3
    decoded[0x12C:0x132] = b"0100d\0"
    struct.pack_into("<H", decoded, 0x134 + 0x1A, 21509)
    struct.pack_into("<4H", decoded, 0x174, 0x0101, 0x0085, 0x0085, 0x0085)
    body = decoded[0x68:]
    compressed = pack_bits("".join(f"1{byte:08b}" for byte in body))
    wrapped = decoded[:0x68] + compressed
    struct.pack_into("<I", wrapped, 0xC, len(wrapped))
    struct.pack_into("<II", wrapped, 0x18, len(compressed), len(body))
    struct.pack_into("<I", wrapped, 0x10, 0x3F000318 + sum(wrapped[0x15:]))
    for i in range(0x18, len(wrapped)):
        wrapped[i] = (wrapped[i] + wrapped[0x15] + 7 * (i - 0x18)) & 255
    path = tmp_path / "final-a.rpy"
    path.write_bytes(wrapped)
    metadata, stages = read_replay(path)
    assert metadata["difficulty"] == 3 and metadata["executable_version"] == "0100d"
    assert len(stages) == 1 and stages[0].raw_stage == 6 and stages[0].rng_seed == 21509
    assert stages[0].inputs.tolist() == [0x0101, 0x0085, 0x0085, 0x0085]


def test_th08_dictionary_copy_and_native_path_validation():
    source = pack_bits(f"1{65:08b}0{1:013b}{3:04b}")
    assert _decompress(source, 7) == b"AAAAAAA"
    with pytest.raises(ValueError, match="declared size"):
        _decompress(source, 6)

    motion = TH08Motion((192., 384.), 1, (8., 32.), (376., 432.), (191., 383.), (193., 385.),
                        (1., 1.), 4., 2.828, 2., 1.414, (1., 1.), 1.)
    index = POLICY_MASKS.index(0x85)
    correct = [1, 0x85, 0x85, 0x85]
    positions = motion.paths([0x85], 4)[0, 1:]
    assert exact_four_frame_actions(correct).tolist() == [index]
    assert observed_four_frame_action(motion, correct, positions) == index
    different = positions.copy()
    different[2, 0] += .25
    assert observed_four_frame_action(motion, correct, different) == -1
    assert exact_four_frame_actions([1, 0x85, 0x81, 0x85]).tolist() == [-1]
    assert exact_four_frame_actions([3, 0x85, 0x85, 0x85]).tolist() == [-1]


def test_th08_teacher_archive_has_only_contiguous_actions(tmp_path):
    from fly_touhou.nbnm_rehearsal import SuccessRehearsal

    script = Path(__file__).resolve().parents[1] / "scripts/capture_th08_teacher_windows.py"
    save_windows = runpy.run_path(str(script))["save_windows"]
    commands = [1] * 32 + [-1] + [2] * 10 + [-1] + [3] * 18 + [4] * 20
    rows = [dict(features=np.zeros(407, np.float32), action=action, native_action_frames=4,
                 observed_contact=False, verified_exit_focus=i == 16,
                 native_poll=4 * i + (8 if action == 4 else 0), stage=2)
            for i, action in enumerate(commands)]
    staged = ReplayStage(1, 32568, np.zeros(4 * len(rows), np.uint16))
    save_windows(tmp_path, rows, staged.raw_stage, staged.rng_seed, replay="original-test.rpy")
    supported = SuccessRehearsal(tmp_path / "manifest.json")
    assert len(supported.rows) == 3
    assert [row["decisions"] for row in supported.rows] == [32, 18, 20]
    assert [row["training_start"] for row in supported.rows] == [16, 2, 4]
    assert all(row["source_kind"] == "native_TH08_teacher_replay_window" for row in supported.rows)
    assert supported.rows[0]["correction_windows"] == [[16, 32]]
    assert supported.rows[0]["native_exit_focus_rows"] == 1
    assert supported.rows[0]["verified_native_exposure_rows"] == 32
    with np.load(supported.rows[-1]["file"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["native_frames"], archive["native_poll"])
        assert np.all(archive["native_action_frames"] == 4)
        assert not archive["observed_contacts"].any()
        assert np.all(archive["actions"] == 4)
        assert np.all(np.diff(archive["native_poll"]) == 4)


def test_native_replay_alignment_extends_an_ambiguous_128_inputs(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/capture_th08_teacher_windows.py"
    locate = runpy.run_path(str(script))["locate_replay_input"]
    recorded = np.ones(500, dtype=np.uint16)
    recorded[160] = 17
    stage = ReplayStage(1, 32568, recorded)
    entry = 1000

    class Replay:
        index = 0

        def state(self):
            return dict(native_replay=True, stage=1, input_poll=entry + self.index,
                        gameplay_input=int(recorded[self.index]))

        def advance(self, action, polls):
            self.index += polls
            return self.state()

    replay = Replay()
    state, target, proof = locate(replay, stage, replay.state(), 180, entry, tmp_path / "STOP")
    assert proof == dict(replay_index=0, count=161, first_native_poll=entry)
    assert state["input_poll"] == entry + 161
    assert target == entry + 180
