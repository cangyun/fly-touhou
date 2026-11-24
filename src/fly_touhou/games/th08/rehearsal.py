"""Original TH08 route certificates and supervised rehearsal."""

import json
from pathlib import Path

import numpy as np

from ...nbnm_rehearsal import SCHEMA, SuccessRehearsal
from ...th08_attack_readout import OBSERVATION_COLUMN, OPENING_NONSPELL_CONTEXT, observed_attack_context
from ...th08_game_state import STAGE_NUMBERS
from ...thrl_durable_records import write_json
from ...thrl_study_io import read_shared_json, read_shared_text


STAGE_WEIGHTS = (4, 5, 6, 1, 4, 5, 6, 2, 4, 5, 6, 3)
FIRST_SPELL = {4: 80, 5: 103, 6: 122}


def success_certificate(*, stage_clear=False):
    return dict(contacts=0, Bombs=0, misses=0, continues=0, stage_clear=stage_clear, task_complete=True)


def write_student_route(folder, features, actions, training_start, task, scope, seed, status):
    if status["contact"] or any(status[key] != 0 for key in ("misses", "Bombs", "continues")):
        return
    if len(actions) - training_start < 16:
        return
    folder.mkdir(parents=True)
    filename = "route.npz"
    np.savez_compressed(folder / filename, features=np.asarray(features, dtype=np.float32),
                        actions=np.asarray(actions, dtype=np.int64))
    stage = status["raw_stage"]
    stage_clear = status["outcome"] in ("story-clear", "stage-practice-clear")
    row = dict(file=filename, raw_stage=stage, task=task, scope=scope, native_seed=seed, split="train",
               decisions=len(actions), nbnm_success=True, training_start=training_start,
               source_kind="native_TH08_student_success", native_validation=success_certificate(stage_clear=stage_clear))
    write_json(folder / "manifest.json", dict(schema=SCHEMA, histories=[row], original_game="TH08",
        native_outcome=status["outcome"], native_frames=status["native_frames"], PPO_rows=0))


def _complete_teacher(report_path):
    report = read_shared_json(report_path)
    if report.get("original_completed") and report.get("zero_contact") and report.get("teacher_manifest"):
        manifest_path = Path(report["teacher_manifest"])
        if not manifest_path.is_file():
            return
        manifest = read_shared_json(manifest_path)
        if manifest.get("schema") != SCHEMA or not manifest.get("source_replay"):
            return
        stage, spell = report.get("raw_stage"), report.get("target_spell")
        if type(stage) is not int or not 0 <= stage < len(STAGE_NUMBERS) or type(spell) is not int:
            return
        for row in manifest["histories"]:
            proof = row.get("native_validation", {})
            if proof.get("contacts") != 0 or proof.get("Bombs") != 0 or not row.get("verified_native_exposure_rows"):
                continue
            yield dict(row, file=str((manifest_path.parent / row["file"]).resolve()),
                teacher_certificate=str(report_path), primary_replay=True, replay_spell_id=spell,
                preceding_nonspell=bool(report.get("preceding_nonspell")), raw_stage=stage)
    elif (report.get("teacher_only") and report.get("zero_contact")
          and all(report.get(key) == 0 for key in ("misses", "bombs", "continues"))):
        repaired_failure = report.get("matched_student_failure") is True
        partial = not report.get("complete_attack")
        end = report.get("supervised_native_end")
        if partial and not (repaired_failure and type(end) is int
                and report["source_contact_frame"] <= end <= report["native_end"] - 96
                and end == report["native_start"] + 4 * report["verified_guide_decisions"]):
            return
        file = report_path.with_name("matched-history.npz")
        if not file.is_file():
            return
        first = report["training_start"]
        stage, spell = report.get("raw_stage"), report.get("spell")
        if spell is not None and stage is None:
            stage = report["native_spell_result"]["raw_stage"]
        nonspell = report.get("task") == "nonspell"
        task = dict(mode="attack", stage=STAGE_NUMBERS[stage])
        if nonspell:
            task["attack"] = f"nonspell-before-{report['next_spell']}"
        elif partial:
            task.update(mode="route", end_event="native-frame-arrival", end_native_frame=end, native_spell_id=spell)
        else:
            task["native_spell_id"] = spell
        source_kind = ("native_TH08_teacher_verified_continuation" if partial
                       else "native_TH08_teacher_completed_route")
        row = dict(file=str(file.resolve()), teacher_certificate=str(report_path), task=task, raw_stage=stage,
                   decisions=first + report["verified_guide_decisions"], nbnm_success=True, split="train",
                   training_start=first, source_kind=source_kind,
                   native_validation=success_certificate(),
                   replay_spell_id=report.get("next_spell") if nonspell else spell,
                   preceding_nonspell=nonspell)
        if report.get("original_autonomous_reference") or repaired_failure:
            row.update(focus_reference=True, correction_windows=report["correction_windows"])
        if repaired_failure:
            row["matched_student_failure"] = True
        yield row


def _eligible(row):
    proof = row.get("native_validation", {})
    return (row.get("split") == "train" and row.get("nbnm_success") is True
            and row.get("decisions", 0) - row.get("training_start", 0) >= 16
            and all(proof.get(key) == 0 for key in ("contacts", "Bombs", "misses", "continues"))
            and proof.get("task_complete") is True and Path(row["file"]).is_file())


def build_catalog(path, teacher_roots, student_root, prior_student_roots=()):
    """Publish one immutable training view after all certificates for the segment settle."""
    rows = {}
    for root in teacher_roots:
        for report in sorted(Path(root).rglob("report.json")):
            for row in _complete_teacher(report):
                # This proof covers the whole local attack in the original Replay.
                if row.get("primary_replay"):
                    row["native_validation"] = success_certificate()
                if _eligible(row):
                    rows[row["file"]] = row
    for bank in (*prior_student_roots, student_root):
        for manifest_path in sorted(Path(bank).glob("*/*/manifest.json")):
            manifest = read_shared_json(manifest_path)
            if manifest.get("schema") != SCHEMA or manifest.get("original_game") != "TH08":
                continue
            for record in manifest["histories"]:
                row = dict(record, file=str((manifest_path.parent / record["file"]).resolve()))
                if _eligible(row) and row.get("source_kind") == "native_TH08_student_success":
                    rows[row["file"]] = row
    if not rows:
        raise ValueError("No complete original TH08 routes are available for rehearsal")
    for row in rows.values():
        stage = row["task"].get("stage", STAGE_NUMBERS[row["raw_stage"]])
        row["task_key"] = f"TH08 stage {stage}"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(schema=SCHEMA, histories=list(rows.values()), original_game="TH08", PPO_rows=0))
    return list(rows.values())


def student_spell_ids(rows):
    return {row["task"]["native_spell_id"] for row in rows
            if row["source_kind"] == "native_TH08_student_success"
            and row.get("scope") == "spell" and row["task"].get("native_spell_id") is not None}


def training_coverage(outputs, spell_ids, nonspell_ids):
    """Read terminal student training events across the saved lineage."""
    successes = dict(spell=set(), nonspell=set())
    scopes = {scope: dict(finished=0, success=0) for scope in ("spell", "nonspell", "stage", "full_story")}
    for output in outputs:
        for log in sorted(Path(output).glob("native-*/training-nonspell-successes.jsonl")):
            lines = read_shared_text(log).splitlines()
            for i, line in enumerate(lines):
                try:
                    successes["nonspell"].add(json.loads(line)["end_spell_id"])
                except json.JSONDecodeError:
                    if i == len(lines) - 1:
                        break
                    raise
        for log in sorted(Path(output).glob("native-*/training-episodes.jsonl")):
            lines = read_shared_text(log).splitlines()
            for i, line in enumerate(lines):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    if i == len(lines) - 1:
                        break
                    raise
                scope = row["entry_scope"]
                scopes[scope]["finished"] += 1
                passed = (row["nbnm_success"] and row["contact"] is False
                          and all(row[k] == 0 for k in ("misses", "Bombs", "continues")))
                if passed:
                    scopes[scope]["success"] += 1
                    if scope == "spell":
                        successes[scope].add(row["task"]["native_spell_id"])
                    elif scope == "nonspell":
                        successes[scope].add(int(row["task"]["attack"].rsplit("-", 1)[1]))
    missing_spells = sorted(set(spell_ids) - successes["spell"])
    missing_nonspells = sorted(set(nonspell_ids) - successes["nonspell"])
    return dict(scopes=scopes, spell_passes=sorted(successes["spell"]),
                nonspell_passes=sorted(successes["nonspell"]), missing_spells=missing_spells,
                missing_nonspells=missing_nonspells, complete=not (missing_spells or missing_nonspells))


class TH08RouteRehearsal(SuccessRehearsal):
    def __init__(self, manifest, *, focus_spells=(), first_successes=(), focus_order=()):
        super().__init__(manifest)
        self.focus_order = tuple(focus_order)
        self.first_tasks = {}
        self.focus_tasks = {}
        for spell in first_successes:
            row = next((r for r in self.rows if r["source_kind"] == "native_TH08_student_success"
                        and r["task"].get("native_spell_id") == spell), None)
            if row is not None:
                self.first_tasks[spell] = row["task_key"] = f"TH08 first student success {spell}"
        for spell in focus_spells:
            teachers = [r for r in self.rows if r.get("teacher_certificate")
                        and r.get("replay_spell_id") == spell and not r.get("preceding_nonspell")]
            if teachers:
                self.focus_tasks[spell] = key = f"TH08 focus spell {spell}"
                references = [row for row in teachers if row.get("focus_reference")]
                for row in references or teachers:
                    row["task_key"] = key

    def _select_row(self, index, reserved, windows):
        pool = [self.rows[i] for i in reserved] if reserved else self.rows
        teachers = [row for row in pool if row.get("teacher_certificate")]
        students = [row for row in pool if row["source_kind"] == "native_TH08_student_success"]
        primary = [row for row in teachers if row.get("primary_replay")]
        # Teachers keep the dominant allocation once student successes enter.
        rng = np.random.default_rng(index)
        draw = rng.random()
        repairs = [row for row in teachers if row.get("matched_student_failure")] if reserved else []
        if repairs:
            references = [row for row in teachers if not row.get("matched_student_failure")]
            pool = repairs if draw < .75 or not references else references
            return pool[int(rng.integers(len(pool)))], rng
        if students and draw < .2:
            pool = students
        elif primary and draw < .8:
            pool = primary
        elif teachers:
            pool = teachers
        return pool[int(rng.integers(len(pool)))], rng

    def _sequence_loss(self, model, features, actions, row, first, *args, **kwargs):
        target = row.get("replay_spell_id")
        if target is not None and model.native_attack_readout is not None:
            features = features.copy()
            stage = row["raw_stage"]
            if row.get("preceding_nonspell"):
                if (model.nbnm["task_context"] == OPENING_NONSPELL_CONTEXT and FIRST_SPELL.get(stage) == target):
                    select = (features[:, 273] > .5) & (features[:, 274] <= .5)
                    features[select, OBSERVATION_COLUMN] = observed_attack_context(
                        None, None, True, opening_nonspell_stage=stage)
            elif row.get("primary_replay"):
                active_spell = features[:, 274] > .5
                features[active_spell, OBSERVATION_COLUMN] = observed_attack_context(target, None, True)
        return super()._sequence_loss(model, features, actions, row, first, *args, **kwargs)

    def scheduled_tasks(self, segment, slots):
        present = {row["task_key"] for row in self.rows}
        stages = [f"TH08 stage {stage}" for stage in STAGE_WEIGHTS if f"TH08 stage {stage}" in present]
        specific = list(self.first_tasks.values())
        if self.focus_order:
            remaining = list(self.focus_order)
            for spell in self.first_tasks:
                if spell in remaining:
                    remaining.remove(spell)
            focus = [self.focus_tasks[spell] for spell in remaining if spell in self.focus_tasks]
        else:
            focus = [key for spell, key in self.focus_tasks.items() if spell not in self.first_tasks]
        if focus:
            start = segment % len(focus)
            specific.extend((focus[start:] + focus[:start])[:max(0, slots // 2 - len(specific))])
        specific = specific[:slots]
        if stages:
            specific.extend(stages[(segment * slots + i) % len(stages)] for i in range(slots - len(specific)))
        return tuple(specific)
