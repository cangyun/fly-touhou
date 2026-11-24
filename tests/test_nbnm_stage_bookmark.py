from types import SimpleNamespace

from fly_touhou.nbnm_stage_bookmark import story_bookmark_index,restore_entry_encoder


def test_bookmark_waits_for_the_real_stage_clock_after_loading():
    stages=[5]+[6]*13
    clocks=[22000,4,8,12,16,20,24,28,32,4,8,16,28,32]
    assert story_bookmark_index(stages,clocks)==13
    assert story_bookmark_index([5,5],[32,36]) is None


def test_restored_attack_age_uses_relative_time_after_a_cold_start():
    class Sensor:
        def reset(self):self._attack_start=None
        def set_native_episode(self,episode):self.episode=episode
        def encode(self,snap):
            if self._attack_start is None:self._attack_start=snap.frame
            return snap.frame-self._attack_start
    values=[0.]*407
    values[276]=58./658
    sensor=Sensor()
    assert restore_entry_encoder(sensor,SimpleNamespace(frame=32),values)==58
    assert sensor.encode(SimpleNamespace(frame=36))==62


def test_story_bookmark_ignores_stale_stage_clock_and_publishes_once(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import numpy as np
    from fly_touhou import native_replay, nbnm_stage_bookmark, nbnm_story_bookmark as story

    monkeypatch.setattr(native_replay, "semantic_state", lambda s: {"stage": s.stage})
    monkeypatch.setattr(nbnm_stage_bookmark, "replay_stage_record", lambda memory, stage: {"stage": stage, "continues": 0})
    snapshot = SimpleNamespace(stage=2, stage_frame=9990, frame=11500, screen=7, dialogue=False, lasers=[],
        player=SimpleNamespace(state=1,invulnerable_frames=0),
        extra={"thrl_events":{"misses":0,"bombs_used":0,"point_items":30}})
    env = SimpleNamespace(frozen_story_bookmarks=True, snapshot=snapshot, stage_history=[1,2],
        _saved_bookmark_stages=set(), stage6_prefix_capture=str(tmp_path/"stage6-prefix/world0.json"),
        controller=SimpleNamespace(_input_tape=[(1, 1000)],_memory=None), policy_steps=240,episode_frames=4040,
        encoder=SimpleNamespace(_attack=(2,''),_attack_start=11200,corridor_carry={
            "frame":11500,"stage":2,"position":np.array([4.,300.]),"course":np.array([0,1],dtype=np.int8)}))
    info = {}
    native = dict(ticks=1000, transitions=[{"after":{"stage":2,"stage_frame":9990}}])
    story.capture(env,native,info,False)
    assert not info
    native["transitions"].append({"after":{"stage":2,"stage_frame":4}})
    snapshot.stage_frame=28
    story.capture(env,native,info,False)
    assert not info
    snapshot.stage_frame=32
    story.capture(env,native,info,False)
    capsule=json.loads(Path(info["frozen_story_bookmark"]["prefix"]).read_text(encoding="utf-8"))
    assert capsule["status"] == story.PREFIX_STATUS
    assert capsule["raw_input_segments"] == [[1,1000]]
    assert capsule["encoder_state"]["corridor_carry"]["course"] == [0,1]
    assert capsule["native_events"]["point_items"] == 30
    advance = {}
    story.capture(env,native,advance,False)
    assert not advance
    assert snapshot.frame == 11500


def test_story_bookmark_loads_portable_bundle_and_rebases_carry(tmp_path):
    from fly_touhou.nbnm_story_bookmark import _decode_carry,load,SCHEMA
    from fly_touhou.thrl_durable_records import write_json

    book=tmp_path/"stage2";book.mkdir()
    write_json(book/"bookmark.json",dict(schema=SCHEMA,native_input_prefix="../prefix.json",
        brain="brain.pt",configuration="configuration.json",checkpoint="../policy.pt",seed=60036))
    write_json(tmp_path/"latest.json",{"bookmark":"stage2/bookmark.json"})
    entry=load(tmp_path/"latest.json")
    assert entry["checkpoint"] == str(tmp_path/"policy.pt")
    assert entry["native_input_prefix"] == str(tmp_path/"prefix.json")
    saved=dict(frame_offset=-1,stage=2,position=[2.,304.],course=[0,6],issun_lure=True)
    restored=_decode_carry(saved,12201)
    assert restored["frame"] == 12200 and restored["issun_lure"]
    assert restored["course"].tolist() == [0,6]
