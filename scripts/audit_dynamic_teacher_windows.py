"""CPU-only geometry comparison on recorded native teacher death windows.

No ResourceTeacher instance is created or reset. Recorded actions and modes
stay fixed, and only the 18 base-action hazard geometry is compared. Neither
the player nor the hazards are run forward in the original game.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from fly_touhou.dynamic_planner import DynamicCollisionPlanner, DynamicPredictionConfig
from fly_touhou.planner import CollisionPlanner, PlannerConfig
from fly_touhou.state import Bullet, Laser, Player, Snapshot
from audit_native_policy_trajectories import (FORMAL_EXE_SHA, encoded, file_evidence,
    save_json, sha, stable_bytes, verified_manifest, verify_events, write_csv)


SCHEMA = "native-teacher-dynamic-window-audit-v1"
DEFAULT_TRACE_SHA = "0f849c2172f9344468aa36f0fe8dc8d273b432f1ab220e5dbedc80b991bf0ed3"


def stream_evidence(path, expected=None):
    path=Path(path).resolve(); before=path.stat(); digest=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            digest.update(block)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError("Trajectory changed during hashing")
    result={"path":str(path),"sha256":digest.hexdigest(),"bytes":after.st_size}
    if expected is not None and result["sha256"]!=expected:
        raise ValueError("Trajectory differs from the previously sealed source hash")
    return result


def extract_windows(path, window_frames):
    """One streaming pass, keeping only 60-frame buffers and selected windows."""
    recent=deque(); windows=[]; episode=None; metadata=None; transitions=0
    pending=None; rescued=[]; hit_entries=0
    with gzip.open(path,"rb") as stream:
        for line_number,line in enumerate(stream,1):
            record=json.loads(line)
            if record["record_type"]=="metadata":
                metadata=record
            if record["record_type"]!="transition":
                continue
            transitions+=1
            if record["episode_id"]!=episode:
                if pending is not None:
                    raise ValueError("Unresolved hit entry at episode boundary")
                recent.clear(); episode=record["episode_id"]
            s,n=record["snapshot"],record["next_snapshot"]
            while recent and s["frame"]-recent[0]["record"]["snapshot"]["frame"]>window_frames:
                recent.popleft()
            entry={"source_line":line_number,"line_sha256":sha(line),"raw_line":line,"record":record}
            recent.append(entry)
            p,q=s.get("player"),n.get("player")
            if p and q and p["state"]!=4 and q["state"]==4:
                hit_entries+=1
                if pending is not None:
                    raise ValueError("A second hit entry preceded resolution of the first")
                pending={"episode_id":episode,
                    "death_step":record["step"],"death_stage":s["stage"],
                    "death_stage_frame":s["stage_frame"],"death_frame":s["frame"],
                    "first_state4_observed_stage_frame":n["stage_frame"],
                    "records":list(recent)}
            if s["lives"]>n["lives"]:
                if pending is None:
                    raise ValueError("Life decrement has no observed preceding state4 entry")
                pending.update(window=len(windows)+1,life_decrement_step=record["step"],
                    life_decrement_stage_frame=n["stage_frame"])
                windows.append(pending);pending=None
            elif pending is not None and p and q and p["state"]==4 and q["state"]==1:
                rescued.append({k:v for k,v in pending.items() if k!="records"})
                pending=None
    if metadata is None:
        raise ValueError("Trace has no metadata")
    if pending is not None:
        raise ValueError("Trace ends with an unresolved hit entry")
    return metadata,windows,transitions,{"state4_entry_count":hit_entries,
        "state4_entries_returned_to_alive_without_life_decrement":rescued,
        "selection_rule":"First entry into state4, retrospectively confirmed by later native life decrement; rescued hit windows are excluded"}


def metrics_record(metric):
    d=asdict(metric)
    d["clearance_infinite"]=math.isinf(d["clearance"])
    if d["clearance_infinite"]:
        d["clearance"]=None
    return d


def geometry(planner,snapshot):
    ms=[planner.action_metrics(snapshot,a) for a in range(18)]
    return {"actions":[metrics_record(x) for x in ms],
        "positive_clearance_actions":[x.action for x in ms if x.clearance>0],
        "buffer6_clearance_actions":[x.action for x in ms if x.clearance>=6],
        "best_hazard_clearance_actions":[x.action for x in ms if x.clearance==max(y.clearance for y in ms)]}


def hazard_inventory(snapshot):
    return {"old_active_bullets":sum(b.active and not b.intangible for b in snapshot.bullets),
        "inactive_unknown_bullets":sum(not b.active and not b.intangible for b in snapshot.bullets),
        "inactive_age_ge8_candidates":sum(not b.active and not b.intangible and b.age>=8 for b in snapshot.bullets),
        "inactive_kinds":dict(Counter(b.kind for b in snapshot.bullets if not b.active and not b.intangible)),
        "lasers":len(snapshot.lasers),"old_active_lasers":sum(l.active for l in snapshot.lasers),
        "kind0_no_nodes":sum(l.kind==0 and not l.nodes for l in snapshot.lasers),
        "kind0_nonzero_speed":sum(l.kind==0 and not l.nodes and l.speed!=0 for l in snapshot.lasers),
        "nonkind0_lasers":sum(l.kind!=0 for l in snapshot.lasers)}


def analyze_window(window, planners):
    decisions=[]; long_rows=[]
    for entry in window["records"]:
        record=entry["record"]
        snapshot=Snapshot.from_dict(record["snapshot"])
        snapshot.validate()
        if snapshot.source!="native-process" or snapshot.extra.get("exe_sha256")!=FORMAL_EXE_SHA:
            raise ValueError("Window does not come from the pinned original native executable")
        base=record["action"]%18
        diagnostic=record.get("teacher_diagnostic",{})
        mode=diagnostic.get("mode",diagnostic.get("teacher_mode",diagnostic.get("current_mode")))
        variants={name:geometry(planner,snapshot) for name,planner in planners.items()}
        legacy=variants["legacy_active_stationary"]
        default=variants["dynamic_phase_unknown_union"]
        step={"window":window["window"],"source_line":entry["source_line"],
            "source_line_sha256":entry["line_sha256"],"step":record["step"],
            "frame":snapshot.frame,"stage":snapshot.stage,"stage_frame":snapshot.stage_frame,
            "frames_before_death":window["death_frame"]-snapshot.frame,
            "original_action":record["action"],"original_base_action":base,
            "original_requested_bomb":record["action"]>=18,"recorded_teacher_mode":mode,
            "recorded_teacher_diagnostic":diagnostic,
            "player":asdict(snapshot.player) if snapshot.player else None,
            "hazards":hazard_inventory(snapshot),"variants":variants,
            "default_positive_added_vs_legacy":sorted(set(default["positive_clearance_actions"])-set(legacy["positive_clearance_actions"])),
            "default_positive_removed_vs_legacy":sorted(set(legacy["positive_clearance_actions"])-set(default["positive_clearance_actions"])),
            "default_buffer6_added_vs_legacy":sorted(set(default["buffer6_clearance_actions"])-set(legacy["buffer6_clearance_actions"])),
            "default_buffer6_removed_vs_legacy":sorted(set(legacy["buffer6_clearance_actions"])-set(default["buffer6_clearance_actions"]))}
        decisions.append(step)
        for name,g in variants.items():
            for metric in g["actions"]:
                long_rows.append({"window":window["window"],"step":record["step"],
                    "stage":snapshot.stage,"stage_frame":snapshot.stage_frame,
                    "frames_before_death":step["frames_before_death"],"variant":name,
                    "original_action":record["action"],"original_base_action":base,
                    "is_original_base_action":metric["action"]==base,
                    "recorded_teacher_mode":mode,
                    "old_active_bullets":step["hazards"]["old_active_bullets"],
                    "inactive_unknown_bullets":step["hazards"]["inactive_unknown_bullets"],
                    "lasers":step["hazards"]["lasers"],**metric})
    death=decisions[-1]
    summary={k:v for k,v in window.items() if k!="records"}
    summary.update({"decision_count":len(decisions),"original_death_action":death["original_action"],
        "original_recorded_mode":death["recorded_teacher_mode"],"death_hazards":death["hazards"],
        "death_geometry":{name:{"original_base_metrics":g["actions"][death["original_base_action"]],
            "positive_clearance_actions":g["positive_clearance_actions"],
            "buffer6_clearance_actions":g["buffer6_clearance_actions"]} for name,g in death["variants"].items()},
        "window_geometry":{name:{
            "recorded_base_nonpositive_decisions":sum(not g["actions"][d["original_base_action"]]["clearance_infinite"] and g["actions"][d["original_base_action"]]["clearance"]<=0 for d in decisions for nm,g in d["variants"].items() if nm==name),
            "recorded_base_risk_ge085_decisions":sum(d["variants"][name]["actions"][d["original_base_action"]]["risk"]>=.85 for d in decisions),
            "any_buffer6_direction_decisions":sum(bool(d["variants"][name]["buffer6_clearance_actions"]) for d in decisions)} for name in planners},
        "newly_rejected_legacy_positive_directions":sum(len(d["default_positive_removed_vs_legacy"]) for d in decisions),
        "newly_positive_vs_legacy_directions":sum(len(d["default_positive_added_vs_legacy"]) for d in decisions),
        "future_native_survival_tested":False})
    return summary,decisions,long_rows


def render_report(audit):
    lines=["# 教师死亡窗口的动态几何离线对照", "",
        "只读同一条已封口原作轨迹，在四次实际掉命对应的state4受击入口前各60原生帧上，固定当时Snapshot比较18个基础动作。入口按前一条仍存活的Snapshot定位，并由随后原生残机减少确认；获救后恢复state1的受击另列、未混入四次死亡。没有创建、重置或重新运行ResourceTeacher；原始模式和动作仅作记录。没有启动游戏或GPU。", "",
        "|受击|stage/frame|原动作/原模式|旧选中间隙|默认动态间隙|旧预测正间隙方向|动态预测正间隙方向|", "|---|---|---|---:|---:|---|---|"]
    for w in audit["windows"]:
        old=w["death_geometry"]["legacy_active_stationary"]; new=w["death_geometry"]["dynamic_phase_unknown_union"]
        def value(m):
            return "∞" if m["clearance_infinite"] else f"{m['clearance']:.3f}"
        lines.append(f"|D{w['window']}|{w['death_stage']}/{w['death_stage_frame']}|{w['original_death_action']}/{w['original_recorded_mode']}|{value(old['original_base_metrics'])}|{value(new['original_base_metrics'])}|{old['positive_clearance_actions']}|{new['positive_clearance_actions']}|")
    lines += ["", "间隙为当时的18帧预测区间最小距离减半径；正值只表示该几何模型未预测重叠，不表示满足ResourceTeacher全部接触/资源/模式约束，也不保证未来生存。CSV保存每帧、每模型、每动作的clearance/risk与原动作标记。", "",
        "## 受控比较", "", "- `legacy_active_stationary`：原始CollisionPlanner，只含旧active弹及静止active激光。源码哈希与该次采集归档一致。", "- `dynamic_phase_unknown_union`：当前冻结DynamicCollisionPlanner默认；纳入unknown inactive候选；kind0采用静止/平移/增长阶段未知的共线包络。", "- `dynamic_angle_speed`：相同unknown候选，kind0按当前angle/speed刚性平移。", "- `dynamic_stationary`：相同unknown候选，激光固定。它与旧方法之差主要显示纳入未知inactive候选的影响；动态三个版本之差显示kind0运动假设的影响。", "",
        "## 必须保留的边界", "", "1. 这些旧raw仅有active、age和stored velocity，没有raw Bullet enum/碰撞flags。原生补证显示state2 timer>=8就可碰撞，但仍可能active=False；并且state2运动只有0.5×stored velocity。因此“把所有inactive按完整stored velocity从现在起看成危险”是诊断假设，可能产生误报或错过时序，不能宣称完整保守覆盖。", "2. 默认kind0 union在所声明的恒定方向/宽度和当前速度界内包络多种阶段；原生可加速、转向、切段、缩短。它也不是逐帧原生碰撞oracle。", "3. 本比较不重建原始ResourceTeacher的collect/POC承诺、冷却、目标与历史状态。原动作可能是资源/路径模式选择，而非18动作全局几何最优；不能靠重新reset教师来宣称模式一致。", "4. 原动作含Bomb时只比较action%18的基础移动几何，不模拟Bomb清弹、无敌或资源变化。也未加入额外敌机接触安全约束。", "5. 离线出现替代正间隙方向，只能定位旧预测与新预测的区别；没有执行这些方向，不是实机反事实可生存证据。", "",
        "下一步由新的有界实机测试验证：危险raw状态、实际相邻位置、激光phase与速度一起记录；保留此次原始记录和各预测版本，分别统计误报、漏报和真实受击。", ""]
    return "\n".join(lines)


def self_test():
    # A moving line crosses a stationary player between forecast endpoints.
    config=PlannerConfig(horizon_frames=4)
    s=Snapshot(frame=0,stage_frame=0,player=Player(0,20,radius=1),
        lasers=(Laser(1,0,-20,20,0,5,2,True,speed=8),))
    old=CollisionPlanner(config).action_metrics(s,0).clearance
    rigid=DynamicCollisionPlanner(config,DynamicPredictionConfig(kind0_motion="angle_speed")).action_metrics(s,0).clearance
    union=DynamicCollisionPlanner(config).action_metrics(s,0).clearance
    stationary=DynamicCollisionPlanner(config,DynamicPredictionConfig(kind0_motion="stationary")).action_metrics(s,0).clearance
    assert old>0 and rigid<0
    assert union<=min(rigid,stationary)+1e-8
    unknown=Snapshot(frame=0,stage_frame=0,player=Player(0,20,radius=1),
        bullets=(Bullet(1,0,20,0,0,2,active=False,age=12),))
    assert math.isinf(CollisionPlanner(config).action_metrics(unknown,0).clearance)
    assert DynamicCollisionPlanner(config).action_metrics(unknown,0).clearance<0
    print("self-test: phase envelope and explicit unknown-state inclusion passed (synthetic geometry only)")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace",type=Path,default=Path(r"./outputs/legacy-build\experiments\curriculum-v5-skip-a\lunatic.jsonl.gz"))
    parser.add_argument("--trace-sha256",default=DEFAULT_TRACE_SHA)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--window-frames",type=int,default=60)
    parser.add_argument("--expect-deaths",type=int,default=4)
    parser.add_argument("--native-seal",type=Path,default=PROJECT/"artifacts/full_game/native_hazard_semantics_20261018/seal.json")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    if args.out is None or args.window_frames<0:
        parser.error("--out and a nonnegative window size are required")
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    try:
        source=stream_evidence(args.trace,args.trace_sha256)
        run=Path(str(args.trace)+".experiment")
        manifest,manifest_proof=verified_manifest(run)
        events=verify_events(stable_bytes(run/"events.jsonl"))
        if not any(e["event"]=="native-curriculum-file-sealed" for e in events):
            raise ValueError("No completed curriculum seal event")
        source_files={}
        for name in ("planner","state"):
            item=manifest["files"][name]
            source_files[name]=file_evidence(item["path"],item["sha256"])
        dynamic_file=PROJECT/"src/fly_touhou/dynamic_planner.py"
        source_files["dynamic_planner"]=file_evidence(dynamic_file)
        source_files["audit_helpers"]=file_evidence(PROJECT/"scripts/audit_native_policy_trajectories.py")
        source_files["audit_script"]=file_evidence(__file__)
        for name,item in source_files.items():
            target=out/(name+Path(item["path"]).suffix)
            shutil.copyfile(item["path"],target)
            item["archive"]=file_evidence(target,item["sha256"])
        native_seal=json.loads(stable_bytes(args.native_seal))
        if native_seal["original_exe_sha256"]!=FORMAL_EXE_SHA:
            raise ValueError("Native semantics evidence targets a different executable")
        native_proof={"seal":file_evidence(args.native_seal),
            "files":[file_evidence(x["path"],x["sha256"]) for x in native_seal["files"]]}
        metadata,windows,total,selection=extract_windows(args.trace,args.window_frames)
        if len(windows)!=args.expect_deaths:
            raise ValueError(f"Expected {args.expect_deaths} death entries, found {len(windows)}")
        config=PlannerConfig(**metadata["config"]["teacher_config"]["planner"])
        planners={"legacy_active_stationary":CollisionPlanner(config),
            "dynamic_phase_unknown_union":DynamicCollisionPlanner(config),
            "dynamic_angle_speed":DynamicCollisionPlanner(config,DynamicPredictionConfig(kind0_motion="angle_speed")),
            "dynamic_stationary":DynamicCollisionPlanner(config,DynamicPredictionConfig(kind0_motion="stationary"))}
        summaries=[];decisions=[];long_rows=[];selected=[]
        for w in windows:
            summary,ds,ls=analyze_window(w,planners)
            summaries.append(summary);decisions+=ds;long_rows+=ls
            fn=out/f"window_{w['window']:02d}.jsonl.gz"
            with gzip.open(fn,"wb") as stream:
                stream.write(encoded({"record_type":"audit_window_metadata","source":source,
                    "death_step":w["death_step"],"window_frames":args.window_frames})+b"\n")
                for entry in w["records"]:
                    stream.write(entry["raw_line"])
            selected.append(file_evidence(fn))
            print(f"window {w['window']}: stage {w['death_stage']} frame {w['death_stage_frame']}, {len(ds)} fixed observations",flush=True)
        # Refuse to silently mix changing sources into an audit already in flight.
        for item in source_files.values():
            file_evidence(item["path"],item["sha256"])
        stream_evidence(args.trace,source["sha256"])
        provenance={"source_trace":source,"manifest":manifest_proof,
            "events":file_evidence(run/"events.jsonl"),"event_chain_verified":len(events),
            "native_semantics_evidence":native_proof,"source_files":source_files,
            "selected_native_windows":selected,
            "window_source_lines":[{"window":w["window"],"lines":[{"line":e["source_line"],"sha256":e["line_sha256"]} for e in w["records"]]} for w in windows]}
        audit={"schema":SCHEMA,"created_utc":datetime.now(timezone.utc).isoformat(),
            "execution":{"cpu_only":True,"game_launched":False,"gpu_used":False,
                "resource_teacher_instantiated":False,"resource_teacher_reset":False},
            "source_total_transitions":total,"window_frames":args.window_frames,
            "window_selection":selection,
            "planner_config":asdict(config),
            "prediction_variants":{name:planner.specification() for name,planner in planners.items() if isinstance(planner,DynamicCollisionPlanner)},
            "windows":summaries,
            "interpretation":"Geometry-only fixed-snapshot diagnostic, not teacher-mode replay and not a real-game counterfactual survival test"}
        save_json(out/"diagnostic_summary.json",audit)
        save_json(out/"provenance.json",provenance)
        save_json(out/"decisions.json",decisions)
        write_csv(out/"action_geometry.csv",long_rows)
        (out/"report.md").write_text(render_report(audit),encoding="utf-8")
        save_json(out/"audit_seal.json",{"schema":SCHEMA,
            "files":{x.name:file_evidence(x) for x in out.iterdir() if x.is_file()}})
        print(json.dumps({"out":str(out),"deaths":len(windows),"window_decisions":len(decisions),"action_model_rows":len(long_rows)}))
    except BaseException as exc:
        save_json(out/"failure.json",{"error":repr(exc),"partial_evidence_preserved":True})
        raise


if __name__=="__main__":
    main()
