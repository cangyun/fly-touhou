import pytest

from fly_touhou.th08_game_state import story_outcome


def test_observed_spell_cue_leaves_the_existing_connectome_input_unchanged():
    from types import SimpleNamespace

    import torch

    from fly_touhou.compact_story_model import CompactStoryPolicy
    from fly_touhou.th08_attack_readout import OBSERVATION_COLUMN, TASK_CONTEXT, observed_attack_context
    from fly_touhou.th08_sensation import TH08ObservationEncoder

    encoder = TH08ObservationEncoder(task_context=TASK_CONTEXT)
    game = dict(stage=4, spell_active=False, native_spell_id=None, native_spell_result_id=88)
    cue = encoder._attack_context(100, game, [1])[-1]
    assert cue == observed_attack_context(None, 88, True) != 0.
    current = encoder._attack_context(104, dict(game, spell_active=True, native_spell_id=92), [1])[-1]
    assert current == observed_attack_context(92, 88, True) != cue

    original = torch.zeros(1, 407)
    observed = original.clone()
    observed[:, OBSERVATION_COLUMN] = float(cue)
    policy = SimpleNamespace(nbnm=dict(task_context=TASK_CONTEXT, corridor_risk={"actor_weight": 1.}))
    assert torch.equal(CompactStoryPolicy.candidate_features(policy, original),
                       CompactStoryPolicy.candidate_features(policy, observed))


def test_selected_original_spell_is_known_during_practice_intro():
    from fly_touhou.th08_attack_readout import PRACTICE_INTRO_CONTEXT, TASK_CONTEXT, observed_attack_context
    from fly_touhou.th08_sensation import TH08ObservationEncoder

    paired = dict(stage=6, spell_active=False, native_spell_id=None, native_spell_result_id=0,
                  spell_practice=True, practice_spell=126)
    encoder = TH08ObservationEncoder(task_context=PRACTICE_INTRO_CONTEXT)
    assert encoder._attack_context(100, paired, [])[-1] == observed_attack_context(126, None, False)
    assert encoder._attack_context(104, dict(paired, spell_active=True, native_spell_id=130), [1])[-1] == (
        observed_attack_context(130, None, True))
    assert encoder._attack_context(108, dict(paired, spell_practice=False), [])[-1] == 0.
    legacy = TH08ObservationEncoder(task_context=TASK_CONTEXT)
    assert legacy._attack_context(100, paired, [])[-1] == 0.


@pytest.mark.parametrize(("stage", "previous", "spell"), [(4, 53, 80), (5, 99, 103), (6, 118, 122)])
def test_first_nonspell_uses_a_trainable_observed_context(stage, previous, spell):
    from types import SimpleNamespace

    import torch

    from fly_touhou.compact_story_model import CompactStoryPolicy
    from fly_touhou.th08_attack_readout import (
        CONTEXT_COUNT, OBSERVATION_COLUMN, OPENING_NONSPELL_CONTEXT, OPENING_NONSPELL_ROWS,
        PRACTICE_INTRO_CONTEXT, observed_attack_context,
    )
    from fly_touhou.th08_sensation import TH08ObservationEncoder

    game = dict(stage=stage, raw_stage=stage, spell_active=False, native_spell_id=None, native_spell_result_id=0)
    encoder = TH08ObservationEncoder(task_context=OPENING_NONSPELL_CONTEXT)
    cue = OPENING_NONSPELL_ROWS[stage][0] / CONTEXT_COUNT
    assert encoder._attack_context(100, game, [1])[-1] == cue
    assert encoder._attack_context(104, dict(game, native_spell_result_id=previous), [1])[-1] == cue
    assert encoder._attack_context(108, game, [])[-1] == 0.
    assert encoder._attack_context(112, dict(game, stage=1, raw_stage=0), [1])[-1] == 0.
    assert encoder._attack_context(116, dict(game, native_spell_result_id=spell), [1])[-1] == (
        observed_attack_context(None, spell, True))
    assert encoder._attack_context(120, dict(game, spell_active=True, native_spell_id=spell), [1])[-1] == (
        observed_attack_context(spell, None, True))
    legacy = TH08ObservationEncoder(task_context=PRACTICE_INTRO_CONTEXT)
    assert legacy._attack_context(120, game, [1])[-1] == 0.

    original = torch.zeros(1, 407)
    observed = original.clone()
    observed[:, OBSERVATION_COLUMN] = cue
    policy = SimpleNamespace(nbnm=dict(task_context=OPENING_NONSPELL_CONTEXT, corridor_risk={"actor_weight": 1.}))
    assert torch.equal(CompactStoryPolicy.candidate_features(policy, original),
                       CompactStoryPolicy.candidate_features(policy, observed))


def test_paired_spell_practice_can_start_on_first_playable_poll():
    from types import SimpleNamespace

    from fly_touhou.th08_controller import TH08InputController

    class OriginalMenu:
        def __init__(self):
            self.entered = False
            self.poll = 0

        def select(self, screen, *_):
            self.entered = screen == 17

        def advance(self, mask, ticks):
            if self.entered and mask == 0:
                self.poll += ticks
            return dict(mode=2, player_state=0, stage_timer=int(self.poll >= 5),
                        stage=6, game_flags=1 << 14, input_poll=self.poll)

    menu = OriginalMenu()
    original = SimpleNamespace(_move_title_cursor=menu.select, advance=menu.advance)
    result = TH08InputController.enter_spell_practice(original, 6, 7, first_playable=True)
    assert result["input_poll"] == 5

    menu = OriginalMenu()
    original = SimpleNamespace(_move_title_cursor=menu.select, advance=menu.advance)
    result = TH08InputController.enter_spell_practice(original, 6, 7)
    assert result["input_poll"] == 10


def test_new_observed_spell_readout_preserves_untrained_context_scores():
    import torch

    from fly_touhou.th08_attack_readout import CONTEXT_COUNT, TH08AttackReadout, observed_attack_context

    route = TH08AttackReadout(torch.nn.Linear(4, 8))
    activity = torch.randn(2, 18, 4, generator=torch.Generator().manual_seed(3))
    cue = torch.tensor([observed_attack_context(134, None, True), observed_attack_context(99, None, True)])
    assert torch.count_nonzero(route(activity, cue)) == 0
    with torch.no_grad():
        route.context_weight.weight[int(cue[0] * CONTEXT_COUNT)].fill_(1.)
    result = route(activity, cue)
    assert result[0].std() > 0.
    assert torch.count_nonzero(result[1]) == 0


def test_original_branch_can_verify_a_short_spell_completion():
    from types import SimpleNamespace

    from fly_touhou.th08_controller import TH08InputController

    active = dict(raw_stage=4, native_spell_id=92)
    finish = dict(raw_stage=4, native_spell_id=None, native_spell_result_id=92, native_spell_captured=True)
    game = iter((active, finish, finish))
    state = dict(native_contact=False, misses=0, bombs_used=0, deathbombs_used=0)
    traces = iter(([dict(geometry_valid=False, contact=False)] * 4,
                   [dict(geometry_valid=False, contact=True)] * 4))
    restored = []
    original = SimpleNamespace(save_branch_checkpoint=lambda: {},
                               trace_branch_clearances=lambda *_, **__: next(traces),
                               read=lambda: state, read_game_state=lambda *_: next(game),
                               restore_branch_checkpoint=lambda *, release: restored.append(release))
    result, _ = TH08InputController.trace_branch_routes(
        original, [[(0, 12)], [(4, 12)]], measure_spell_completion=True)
    assert result["spell_completions"] == [dict(spell_id=92, frames=4, captured=True), None]
    assert restored == [False, True]


def test_original_spell_damage_uses_each_verified_route_length(monkeypatch):
    from types import SimpleNamespace

    from fly_touhou import th08_game_state
    from fly_touhou.th08_controller import TH08InputController

    health = iter((220, 170, 55))
    monkeypatch.setattr(th08_game_state, "read_spell_attack",
                        lambda _: dict(health=next(health), spell_id=134, target=1))
    clear = dict(geometry_valid=True, contact=False)
    frames = iter(([clear] * 48, [clear] * 128, [clear] * 160))
    released = []

    def restore(*, release):
        released.append(release)

    original = SimpleNamespace(_memory=object(), save_branch_checkpoint=lambda: {},
                               trace_branch_clearances=lambda *_, **__: next(frames),
                               restore_branch_checkpoint=restore)
    result, _ = TH08InputController.trace_branch_routes(
        original, [[(0, 48)], [(0, 192)], [(0, 160)]], measure_spell_damage=True, max_ticks=192)
    assert result["spell_damage"] == [50, None, 165]
    assert released == [False, False, True]


def test_portable_story_keeps_learned_readouts_and_discards_th14_encounters():
    from fly_touhou.nbnm_corridor import REVISION, validate
    from fly_touhou.th08_training import portable_spec

    learned = dict(revision=REVISION, loss_weight=.3, witness_weight=8., actor_weight=4., keep_course=True)
    native = dict(learned, issun_lure=True, issun_lure_tail_turns=True, corner_reserve_stages=[6],
                  refinement_phases="boss_spells", refinement_first_boss_spell=2, reactive_retry_stages=[6],
                  reactive_retry_boss_cards=[2], refine_reactive_turns=True, refine_early_turns=True,
                  exit_supervision=True, exit_supervision_48_fallback=True,
                  anticipate_line_births=True, native_emission_recheck_limit=4)
    validate(native)
    parent = dict(frozen_corridor_exit_scoring=True, nbnm=dict(objective={},
        native_emitter_forecast_frames=96, native_rule_geometry={"revision": "TH14"},
        corridor_risk=native, forecast_native_branches=False))
    adapted = portable_spec(parent)
    settings = adapted["nbnm"]
    assert validate(settings["corridor_risk"]) == learned
    assert "native_emitter_forecast_frames" not in settings and "native_rule_geometry" not in settings
    assert settings["forecast_native_branches"] is True
    assert portable_spec(parent, native_forecasts=False)["nbnm"]["forecast_native_branches"] is False
    assert not adapted["frozen_corridor_exit_scoring"] and parent["frozen_corridor_exit_scoring"]
    assert parent["nbnm"]["corridor_risk"]["issun_lure"]
    assert portable_spec(adapted) == adapted


def test_th08_completed_legacy_run_can_continue_with_native_forecasts():
    from fly_touhou.th08_training import portable_spec
    from scripts.train_th08_nbnm import SCHEMA, validate_training_parent

    source = dict(nbnm=dict(objective={}, corridor_risk={}))
    old = dict(spec="old.json", decisions=256)
    stored = dict(schema=SCHEMA, spec=source, plan=old, decisions=256, pending=None)
    continued = dict(old, spec="new.json", decisions=512, forecast_native_branches=True)
    assert validate_training_parent(stored, continued, portable_spec(source), extending=True) == [
        "forecast_native_branches",
    ]


def test_attack_readout_specialist_preserves_other_contexts_and_earlier_moves():
    import torch
    from fly_touhou.th08_attack_readout import CONTEXT_COUNT, TH08AttackReadout

    projection = torch.nn.Linear(5, 7)
    readout = TH08AttackReadout(projection)
    with torch.no_grad():
        readout.context_weight.weight.uniform_(-1., 1.)
    activity = torch.randn(4, 18, 5)
    context = torch.tensor([100, 100, 101, 135]) / CONTEXT_COUNT
    age = torch.tensor([.39, .4, .9, .9])
    original = readout(activity, context, age)
    specialist = readout.add_specialist(100, .4)
    with torch.no_grad():
        specialist.context_weight.add_(2.)
    specialized = readout(activity, context, age)
    assert torch.equal(original[[0, 2, 3]], specialized[[0, 2, 3]])
    assert not torch.equal(original[1], specialized[1])

    restored = TH08AttackReadout(projection)
    restored.add_specialist(100, 0.)
    restored.load_state_dict(readout.state_dict(), strict=True)
    assert torch.equal(specialized, restored(activity, context, age))


def test_th08_specialist_course_preserves_parent_and_records_its_temperature():
    from fly_touhou.th08_training import portable_spec
    from scripts.train_th08_nbnm import SCHEMA, validate_training_parent

    spec = portable_spec(dict(nbnm=dict(objective={}, corridor_risk={})))
    old = dict(spec="shared.json", decisions=256, forecast_native_branches=True, course_lanes=[[{}]])
    stored = dict(schema=SCHEMA, spec=spec, plan=old, decisions=256, pending=None)
    continued = dict(old, decisions=512, movement_sampling_temperature=.85, rehearsal_focus_order=[134, 134, 99])
    assert validate_training_parent(stored, continued, spec, extending=True, retarget_course=True) == []
    assert "movement_sampling_temperature" not in old
    with pytest.raises(ValueError, match="Retain the collection"):
        validate_training_parent(stored, continued, spec, extending=True)


def test_th08_specialist_rehearses_continuations_and_keeps_stage_retention():
    from fly_touhou.th08_rehearsal import TH08RouteRehearsal

    teacher = object.__new__(TH08RouteRehearsal)
    teacher.focus_order = (134, 134, 134, 134, 99, 99, 99)
    teacher.focus_tasks = {134: "TH08 focus spell 134", 99: "TH08 focus spell 99"}
    teacher.first_tasks = {134: "TH08 first student success 134"}
    teacher.rows = [{"task_key": f"TH08 stage {stage}"} for stage in range(1, 7)]
    tasks = teacher.scheduled_tasks(3, 16)
    assert tasks.count("TH08 focus spell 134") == 3
    assert tasks.count("TH08 focus spell 99") == 3
    assert tasks.count("TH08 first student success 134") == 1
    assert len(tasks) == 16
    assert len(set(tasks[7:])) >= 4


@pytest.mark.parametrize("wide_exit", [False, True])
def test_original_branch_keeps_verified_narrow_exit_only_without_a_wider_route(monkeypatch, wide_exit):
    from types import SimpleNamespace

    import numpy as np
    from fly_touhou import th08_sensation

    count = len(th08_sensation.POLICY_MASKS)
    proposals = np.full((count, 2, 24), -1, dtype=np.int8)
    motion = SimpleNamespace(paths=lambda masks, ticks: np.zeros((len(masks), ticks + 1, 2)))
    velocities = np.zeros((count, 2))

    def query(controller, tapes):
        traces = []
        for action in range(len(tapes)):
            clear = action == 11 or (wide_exit and action == 2)
            row = dict(position=[0., 0.], clearance=.55 if action == 11 else 1.25,
                       geometry_valid=True, contact=not clear)
            traces.append([row] * (96 if clear else 1))
        return traces, {}

    monkeypatch.setattr(th08_sensation, "query_live_native_routes", query)
    result = th08_sensation._native_route_geometry(
        None, proposals, motion, velocities, keep_narrow_exits=True)
    corridor, courses, stats = result[3], result[4], result[-2]
    action = 2 if wide_exit else 11
    assert list(np.flatnonzero(corridor[:, 3] > 0.)) == [action]
    np.testing.assert_array_equal(courses[action, 1], [action] * 24)
    assert stats["narrow_exits_48_96"] == [int(not wide_exit)] * 2


def test_original_practice_prefix_retains_its_spell_menu_selection():
    from fly_touhou.th08_prefix import match_pose, replay_practice_slot

    report = dict(native_rng_controlled=True, entry_mode="original-spell-practice",
                  practice_stage_slot=6, practice_spell_slot=15)
    assert replay_practice_slot(report) == 6
    with pytest.raises(ValueError, match="Spell Practice"):
        replay_practice_slot(dict(report, practice_spell_slot=None))
    recorded = dict(decision=9, raw_stage=6, stage_frame=30, native_frames=36,
                    position=[192., 385.], native_spell_id=None)
    observed = dict(recorded, outcome=None, native_spell_id=134)
    match_pose(recorded, observed, 0, check_spell=False)
    with pytest.raises(RuntimeError, match="native_spell_id"):
        match_pose(recorded, observed, 0)
    with pytest.raises(RuntimeError, match="native_frames"):
        match_pose(recorded, dict(observed, native_frames=40), 0, check_spell=False)


def test_original_story_requires_all_six_stages_and_the_native_normal_ending():
    state = dict(mode=2, native_replay=False, native_contact=False, misses=0, bombs_used=0, deathbombs_used=0)
    game = dict(stage=6, practice=False, demo=False, spell_practice=False, continues=0, retry_menu=0,
                raw_phase=3, next_mode=9, normal_ending=True)
    assert story_outcome(state, game, [1, 2, 3, 4, 5, 6]) == "story-clear"
    assert story_outcome(state, game, [6]) == "incomplete-ending"
    assert story_outcome(state, dict(game, stage=5, normal_ending=False), [1, 2, 3, 4, 5]) == "incomplete-ending"
    with pytest.raises(ValueError, match="original Story"):
        story_outcome(dict(state, native_replay=True), game, [1, 2, 3, 4, 5, 6])


def test_spell_practice_waits_for_the_real_original_first_frame():
    from fly_touhou.th08_controller import TH08InputController

    class Original:
        def __init__(self):
            ready = dict(mode=2, player_state=0, stage=4, game_flags=1 << 14, stage_timer=30)
            self.checkpoints = iter([dict(ready, stage_timer=0), ready])

        def _move_title_cursor(self, *args):
            pass

        def advance(self, mask, ticks):
            return next(self.checkpoints) if ticks == 8 else None

    state = TH08InputController.enter_spell_practice(Original(), 4, 22)
    assert state["stage_timer"] == 30


@pytest.mark.parametrize("fresh_frames,carried_frames", [(7, 96), (60, 92)])
def test_original_branch_offers_a_revalidated_course_suffix(monkeypatch, fresh_frames, carried_frames):
    from types import SimpleNamespace

    import numpy as np

    from fly_touhou import th08_sensation

    moves = len(th08_sensation.POLICY_MASKS)
    proposals = np.full((moves, 2, 24), -1, dtype=np.int8)
    motion = SimpleNamespace(paths=lambda masks, ticks: np.zeros((len(masks), ticks + 1, 2)))
    velocities = np.zeros((moves, 2))
    suffix = np.asarray([4] * 5 + [8] * 6 + [16] * 6 + [13] * 6, dtype=np.int8)
    calls = []

    def query(controller, tapes, **measurements):
        calls.append(tapes)
        clear = dict(position=[0., 0.], clearance=2.5, geometry_valid=True, contact=False)
        hit = dict(clear, clearance=-1., contact=True)
        paths = [[clear] * fresh_frames + [hit] for _ in range(moves)]
        return paths + [[clear] * carried_frames], {"spell_damage": [None] * moves + [7.]}

    monkeypatch.setattr(th08_sensation, "query_live_native_routes", query)
    result = th08_sensation._native_route_geometry(None, proposals, motion, velocities, carried_course=suffix)
    corridor, courses, stats = result[3], result[4], result[-2]
    assert len(calls[0]) == moves + 1
    assert stats["revalidated_course_action"] == 4
    assert stats["revalidated_course_frames"] == stats["verified_course_frames"][4] == carried_frames
    assert len(stats["spell_damage"]) == moves and stats["spell_damage"][4] == 7.
    assert (corridor[4, 3] > 0) == (carried_frames == 96)
    assert stats["first_contact_by_action"][4] is None
    np.testing.assert_array_equal(courses[4, 0, :12], suffix[:12])

    encoder = th08_sensation.TH08ObservationEncoder()
    encoder.last_result = {"native_forecast": stats, "native_candidate_routes": result[-1]}
    encoder.retain_verified_course(4, {"stage": 4, "input_poll": 100})
    np.testing.assert_array_equal(encoder._carried_course[2], [*suffix[1:], suffix[-1]][:carried_frames // 4 - 1])


@pytest.mark.parametrize("already_safe", [False, True])
def test_original_route_forecast_checks_straight_commands_when_all_proposals_hit(monkeypatch, already_safe):
    from types import SimpleNamespace

    import numpy as np

    from fly_touhou import th08_sensation

    moves = len(th08_sensation.POLICY_MASKS)
    proposals = np.full((moves, 2, 24), -1, dtype=np.int8)
    for action in range(moves):
        proposals[action, 1] = [action, *([(action + 1) % moves] * 23)]
    motion = SimpleNamespace(paths=lambda masks, ticks: np.zeros((len(masks), ticks + 1, 2)))
    calls = []
    clear = dict(position=[0., 0.], clearance=24., geometry_valid=True, contact=False)
    hit = dict(clear, clearance=-3., contact=True)

    def query(controller, tapes, **options):
        calls.append(tapes)
        paths = [[clear] * 93 + [hit] for _ in tapes]
        if len(calls) == 2 or already_safe:
            paths[2] = [clear] * 96
        return paths, dict(input_poll=369, candidates=len(tapes), simulated_frames=sum(map(len, paths)), elapsed_ms=1.)

    monkeypatch.setattr(th08_sensation, "query_live_native_routes", query)
    result = th08_sensation._native_route_geometry(
        None, proposals, motion, np.zeros((moves, 2)), try_straight_holds=True)
    stats, routes = result[-2:]
    assert len(calls) == (1 if already_safe else 2)
    assert stats["first_contact_by_action"][2] is None
    assert stats.get("held_route_actions") == (None if already_safe else [2])
    if not already_safe:
        assert calls[1][2] == [(th08_sensation.POLICY_MASKS[2], 96)]
        assert stats["verified_course_frames"][2] == 96
        np.testing.assert_array_equal(routes[2], [2] * 24)


@pytest.mark.parametrize("late_contact", [False, True])
def test_native_boundary_turn_requires_a_contact_free_extension(monkeypatch, late_contact):
    from types import SimpleNamespace

    import numpy as np

    from fly_touhou import th08_sensation

    moves = len(th08_sensation.POLICY_MASKS)
    right, up, down = 2, 3, 4
    paths = np.broadcast_to([170., 280.], (moves, 97, 2)).copy()
    paths[right, :, 0] = np.minimum(376., 166. + 4. * np.arange(97))
    motion = SimpleNamespace(minimum=[8., 16.], maximum=[376., 432.],
                             paths=lambda masks, ticks: paths[:, :ticks + 1])
    proposals = np.full((moves, 2, 24), -1, dtype=np.int8)
    clear = dict(position=[170., 280.], clearance=24., geometry_valid=True, contact=False)
    hit = dict(clear, clearance=-3., contact=True)
    calls = []

    def query(controller, tapes, **options):
        calls.append((tapes, options))
        if len(calls) == 1:
            traces = [[clear] * 95 + [hit] for _ in tapes]
        else:
            extension = [clear] * 115 + [hit]
            traces = [extension if late_contact else [clear] * 160, extension]
        return traces, dict(input_poll=369, candidates=len(tapes), simulated_frames=sum(map(len, traces)), elapsed_ms=1.)

    monkeypatch.setattr(th08_sensation, "query_live_native_routes", query)
    *_, stats, routes = th08_sensation._native_route_geometry(
        None, proposals, motion, np.zeros((moves, 2)), try_corner_turns=True)
    mask = th08_sensation.POLICY_MASKS
    assert calls[1] == ([[(mask[right], 52), (mask[up], 108)], [(mask[right], 52), (mask[down], 108)]],
                        {"max_ticks": 160})
    if late_contact:
        assert stats["corner_route_actions"] == {}
        assert stats["first_contact_by_action"][right] == 96
    else:
        assert stats["corner_route_actions"][right]["checked_frames"] == 160
        assert stats["first_contact_by_action"][right] is None
        np.testing.assert_array_equal(routes[right], [right] * 13 + [up] * 11)


def test_final_spark_survival_label_requires_an_original_exit_and_contact(monkeypatch):
    import numpy as np
    from fly_touhou import th08_sensation
    from fly_touhou.th08_lasers import TH08FinalSparkCharge
    from fly_touhou.th08_motion import TH08Motion
    from fly_touhou.th08_training import native_survival_guidance

    motion = TH08Motion((376., 16.), 0, (8., 16.), (376., 432.), (375.175, 15.175), (376.825, 16.825),
                        (.825, .825), 4., 2.828427, 2., 1.414214, (1., 1.), 1.)
    charge = TH08FinalSparkCharge(np.array([224., 104.]), -.525, 116)
    masks = th08_sensation.POLICY_MASKS
    clear = dict(position=list(motion.position), clearance=3., geometry_valid=True, contact=False)
    hit = dict(clear, clearance=-1., contact=True)

    def query(controller, tapes, *, max_ticks):
        assert max_ticks == 148 and len(tapes) == 4
        trajectories = []
        for tape in tapes:
            mask = tape[0][0]
            case = [clear] * (148 if mask == masks[1] else 110)
            trajectories.append(case + [hit] if mask == masks[0] else case)
        return trajectories, dict(input_poll=400, candidates=4, simulated_frames=500, elapsed_ms=1.)

    monkeypatch.setattr(th08_sensation, "query_live_native_routes", query)
    routes = np.repeat(np.arange(18, dtype=np.int8)[:, None], 24, axis=1)
    traces = [[] for _ in masks]
    lengths = [0] * len(masks)
    stats = dict(input_poll=400, candidates=0, simulated_frames=0, elapsed_ms=0.)
    th08_sensation._probe_final_spark_exits(None, charge, routes, traces, lengths, stats, motion)
    assert lengths[1] == 96
    outcomes = stats["final_spark_exits"]["outcomes"]
    assert outcomes[0]["first_contact"] == 111
    assert outcomes[1]["verified_exit"]
    assert all(not row["verified_exit"] for action, row in outcomes.items() if action != 1)
    label = native_survival_guidance(dict(native_forecast=stats))
    assert label[1] == 1. and sum(label) == 1.


def test_native_contact_ends_nbnm_before_the_miss_or_ending_counter():
    state = dict(mode=2, native_replay=False, native_contact=True, misses=0, bombs_used=0, deathbombs_used=0)
    game = dict(stage=6, practice=False, demo=False, spell_practice=False, continues=0, retry_menu=0,
                raw_phase=3, next_mode=9, normal_ending=True)
    assert story_outcome(state, game, [1, 2, 3, 4, 5, 6]) == "nbnm-failure"


def test_original_die_callback_marks_contact_after_dying_state_ends():
    from types import SimpleNamespace
    from fly_touhou.th08_controller import TH08InputController

    controller = object.__new__(TH08InputController)
    controller._closed = False
    controller.profile = SimpleNamespace(offsets={"mode": (0, 4), "player_state": (1, 1), "game_flags": (2, 4)})
    controller._memory = SimpleNamespace(base=0, integer=lambda address, **options: (2, 3, 4)[address])
    callback = {"input_poll": 1058, "kind": "point"}
    controller._checkpoint = dict(sequence=1060, present={"created": True, "memory_devices": 1}, contact_source=callback)
    assert controller.read()["native_contact"] is True
    controller._checkpoint["contact_source"] = None
    assert controller.read()["native_contact"] is False


def test_original_practice_requires_the_native_result_transition():
    from fly_touhou.th08_game_state import practice_outcome

    state = dict(mode=2, native_replay=False, native_contact=False, misses=0, bombs_used=0, deathbombs_used=0)
    game = dict(stage=4, practice=True, demo=False, spell_practice=False, continues=0, retry_menu=0,
                raw_phase=3, next_mode=6, normal_ending=False)
    assert practice_outcome(state, game) == "stage-practice-clear"
    assert practice_outcome(state, dict(game, raw_phase=0, next_mode=0)) is None
    assert practice_outcome(dict(state, native_contact=True), game) == "nbnm-failure"
    with pytest.raises(ValueError, match="original Story"):
        story_outcome(state, game, [1, 2, 3, 4])


def test_original_spell_practice_counts_the_end_of_an_observed_spell_only():
    from fly_touhou.th08_game_state import spell_practice_outcome

    state = dict(mode=2, native_replay=False, native_contact=False, misses=0, bombs_used=0, deathbombs_used=0)
    game = dict(stage=4, practice=True, demo=False, spell_practice=True, continues=0, retry_menu=0,
                spell_active=False)
    assert spell_practice_outcome(state, game, False) is None
    assert spell_practice_outcome(state, dict(game, spell_active=True), True) is None
    assert spell_practice_outcome(state, game, True) == "spell-practice-survived"
    assert spell_practice_outcome(dict(state, native_contact=True), game, True) == "nbnm-failure"
    with pytest.raises(ValueError, match="original Story"):
        story_outcome(state, game, [1, 2, 3, 4])


@pytest.mark.parametrize("captured", [True, False])
def test_original_spell_ledger_records_capture_bit_once(captured):
    from fly_touhou.th08_story import TH08StoryWorld

    world = TH08StoryWorld(None)
    world._observe_spell(dict(raw_stage=4, native_spell_id=84), 100)
    world._observe_spell(dict(raw_stage=4, native_spell_id=84), 104)
    result = dict(raw_stage=4, native_spell_result_id=84, native_spell_captured=captured)
    world._observe_spell(result, 108)
    world._observe_spell(result, 108)
    assert world.spell_results == [
        dict(raw_stage=4, spell_id=84, start_native_frame=100, end_native_frame=108, captured=captured),
    ]


def test_original_spell_ledger_cannot_count_an_unobserved_spell_result():
    from fly_touhou.th08_story import TH08StoryWorld

    world = TH08StoryWorld(None)
    world._observe_spell(dict(raw_stage=4, native_spell_id=84), 100)
    world._observe_spell(dict(raw_stage=4, native_spell_id=88, native_spell_captured=True), 400)
    assert world.spell_results[0]["captured"] is None


def test_capture_training_distinguishes_a_capture_timeout_and_a_skipped_spell():
    from fly_touhou.th08_game_state import capture_goal_outcome

    pending = dict(raw_stage=0, outcome=None)
    result = dict(raw_stage=0, spell_id=5, captured=True)
    assert capture_goal_outcome(pending, [result], ((0, 5),), local_attack=True) == "spell-capture-clear"
    assert capture_goal_outcome(pending, [dict(result, captured=False)], ((0, 5),)) == "spell-capture-failure"
    timeout = dict(pending, outcome="spell-practice-survived")
    assert capture_goal_outcome(timeout, [dict(result, captured=False)], ((0, 5),),
                                local_attack=True) == "spell-capture-failure"
    assert capture_goal_outcome(timeout, [result], ((0, 5),), local_attack=True) == "spell-capture-clear"
    assert capture_goal_outcome(dict(pending, raw_stage=1), [result], ((0, 5), (0, 12))) == "spell-capture-failure"
    assert capture_goal_outcome(dict(pending, outcome="nbnm-failure"), [result], ((0, 5),),
                                local_attack=True) == "nbnm-failure"


def test_damage_guidance_never_prefers_uncovered_or_contacting_routes():
    import numpy as np

    from fly_touhou.th08_training import native_capture_guidance

    coverage = np.ones((5, 18), dtype=bool)
    coverage[-1, 3] = False
    margins = np.zeros((5, 18), dtype=np.float32)
    margins[-1, :4] = (2., -3., 3., 10.)
    damage = [10., 100., 40., 120.] + [None] * 14
    probs = native_capture_guidance(dict(native_forecast=dict(spell_damage=damage),
                                        native_coverage=coverage, margins=margins))
    assert probs[2] > probs[0] > 0.
    assert probs[1] == probs[3] == probs[18] == 0.
    assert sum(probs) == pytest.approx(1.)
    powers = [0., 8., 1., 128.] + [None] * 14
    probs = native_capture_guidance(dict(native_forecast=dict(power_gain=powers),
                                        native_coverage=coverage, margins=margins))
    assert probs[2] > probs[0] > 0.
    assert probs[1] == probs[3] == 0.


def test_nbnm_guidance_prefers_verified_exits_before_short_term_safety():
    import numpy as np

    from fly_touhou.th08_training import native_survival_guidance

    coverage = np.zeros((5, 18), dtype=bool)
    coverage[:4, :4] = True
    margins = np.full((5, 18), 3., dtype=np.float32)
    margins[3, 2] = .3
    contacts = [8, None, 32, None] + [None] * 14
    scene = dict(native_forecast=dict(first_contact_by_action=contacts),
                 margins=margins, native_coverage=coverage)

    probability = native_survival_guidance(scene)
    assert probability[1] == probability[3] == .5
    assert sum(probability) == 1.
    assert probability[2] == probability[4] == probability[18] == 0.
    assert native_survival_guidance(dict(scene, native_coverage=np.zeros_like(coverage))) is None


def test_difficult_spell_guidance_accepts_a_narrow_native_exit_and_prefers_a_long_one():
    import numpy as np

    from fly_touhou.th08_training import native_survival_guidance

    coverage = np.ones((5, 18), dtype=bool)
    margins = np.full((5, 18), -.1, dtype=np.float32)
    margins[0, 1] = .3
    forecast = dict(first_contact_by_action=[1, 6] + [1] * 16, verified_course_frames=[0, 4] + [0] * 16)
    scene = dict(native_forecast=forecast, margins=margins, native_coverage=coverage)
    assert native_survival_guidance(scene) is None
    assert native_survival_guidance(scene, prioritize_exit=True)[1] == 1.

    forecast["first_contact_by_action"][2] = None
    forecast["verified_course_frames"][2] = 96
    margins[:, 2] = 3.
    assert native_survival_guidance(scene, prioritize_exit=True)[2] == 1.


@pytest.mark.parametrize(("failures", "timeouts"), [(1, [15, 20]), (3, [15, 20, 25])])
def test_th08_retries_only_the_owned_startup(tmp_path, monkeypatch, failures, timeouts):
    from types import SimpleNamespace

    from fly_touhou.native import NativeError
    from fly_touhou.nbnm_corridor import SCHEMA
    from fly_touhou.th08_training import TH08TrainingEnv

    env = TH08TrainingEnv(executable=None, profile=None, d3d8to9=None, private_appdata=tmp_path,
                          input_schema=SCHEMA, settings={"objective": {}, "native_start_retries": 2})
    opened, closed = [], []

    def create(*, timeout):
        opened.append(timeout)

        def start():
            if len(opened) <= failures:
                error = NativeError("First original checkpoint timed out")
                error.native_event = {"kind": "native-checkpoint-timeout", "expected": "checkpoint"}
                raise error

        return SimpleNamespace(start=start, close=lambda: closed.append(timeout), timeout=timeout)

    monkeypatch.setattr(env, "_create_controller", create)
    monkeypatch.setattr("time.sleep", lambda _: None)
    ready, errors = env._start_controller()
    assert opened == timeouts and closed == timeouts[:failures]
    assert len(errors) == failures
    assert ready is (failures < 3)
    assert env.controller.timeout == 15 if ready else env.controller is None


def test_nbnm_spell_practice_reports_safe_timeout_as_success(tmp_path):
    from types import SimpleNamespace

    from fly_touhou.nbnm_corridor import SCHEMA
    from fly_touhou.th08_training import TH08TrainingEnv

    env = TH08TrainingEnv(executable=None, profile=None, d3d8to9=None, private_appdata=tmp_path,
                          input_schema=SCHEMA, settings={"objective": {}}, practice_slot=4,
                          capture_entry={"raw_stage": 4, "spell_slot": 15, "spell_id": 92, "seed": 1})
    status = dict(stage=4, native_frames=120, power=128, outcome="spell-practice-survived",
                  contact=False, misses=0, Bombs=0)
    env.world = SimpleNamespace(game={"stage": 4, "power": 128},
                                state={"input_poll": 116}, raw_scene=None, spell_results=[],
                                step=lambda action: (None, status))
    _, _, terminal, _, result = env.step(0)
    assert terminal and result["nbnm_success"]
    assert result["nbnm_progress_events"] == []
    assert result["episode_summary"]["task"]["native_spell_id"] == 92


def test_full_spell_clear_requires_the_original_route_and_each_verified_capture():
    from scripts.evaluate_th08_nbnm import verified_full_spell_clear

    spells = [dict(raw_stage=0, spell_id=5, captured=True), dict(raw_stage=7, spell_id=166, captured=True)]
    reference = dict(stages=[dict(raw_stage=0), dict(raw_stage=7)],
                     native_spellcards_captured=2, spell_results=spells)
    complete = dict(full_game_NBNM=True, contact=False, route_history=[0, 7],
                    native_spellcards_captured=2, spell_results=spells)
    assert verified_full_spell_clear(complete, reference)
    assert not verified_full_spell_clear(dict(complete, route_history=[0, 6]), reference)
    assert not verified_full_spell_clear(dict(complete, spell_results=spells[:-1]), reference)
    missed_card = [spells[0], dict(spells[1], captured=False)]
    assert not verified_full_spell_clear(dict(complete, spell_results=missed_card), reference)


def test_first_contact_evidence_stops_at_an_unmeasured_native_interval(tmp_path):
    import json

    from scripts.evaluate_th08_nbnm import write_contact_evidence

    def decision(before, after, contact=False):
        return dict(native_frames=before, raw_stage=2, action=3,
                    geometry=dict(clearance=[[2.] * 18], forecast_covered=[False]),
                    after=dict(native_frames=after, raw_stage=2, contact=contact))

    rows = [decision(4, 12), decision(12, 16), decision(16, 20, True)]
    path = tmp_path / "first-contact-evidence.json"
    write_contact_evidence(path, rows)
    measured = json.loads(path.read_text(encoding="utf-8"))
    assert measured["verified_contact_command"] and measured["contact_command"] == 3
    assert measured["elapsed_native_frames"] == 8
    assert measured["contact_command_clearance4"] == 2.
    assert not measured["contact_forecast4_covered"]
    assert [row["native_frames"] for row in measured["decisions"]] == [12, 16]

    write_contact_evidence(path, [decision(12, 20, True)])
    unassigned = json.loads(path.read_text(encoding="utf-8"))
    assert not unassigned["verified_contact_command"]


def test_stage_confirmation_releases_and_presses_shot_only_between_playable_stages():
    from fly_touhou.th08_story import TH08StoryWorld

    class NativeMenu:
        def __init__(self):
            self.commands = []

        def read_game_state(self, state):
            step = len(self.commands)
            return dict(state["game"], raw_phase=2 if step == 0 else 3, active=False,
                        next_mode=9 if step == 2 else 0, normal_ending=step == 2)

        def advance(self, mask, ticks):
            self.commands.append((mask, ticks))
            return dict(world.state, input_poll=world.state["input_poll"] + ticks)

    menu = NativeMenu()
    world = TH08StoryWorld(menu)
    world.initial_poll = 0
    world.stage_history = [1, 2, 3, 4, 5, 6]
    world.state = dict(mode=2, player_state=0, native_replay=False, native_contact=False, misses=0, bombs_used=0,
        deathbombs_used=0, input_poll=40, player_position=(192., 384.), stage_timer=80, difficulty=3,
        shot_type=0, stage=7, game=dict(stage=6, raw_stage=7, practice=False, demo=False,
            spell_practice=False, continues=0, retry_menu=0, spell_active=False))
    # Two original transition states and a terminal read without another command.
    obs, outcome = world._prepare()
    assert menu.commands == [(0, 4), (1, 4)]
    assert obs is None and outcome["full_game_NBNM"]


def test_blocking_stage_five_dialogue_selects_final_a_then_returns_to_play():
    from fly_touhou.th08_story import TH08StoryWorld

    class Original:
        def __init__(self):
            self.commands = []

        def read_game_state(self, state):
            return dict(stage=5, raw_stage=5, active=True, raw_phase=0, pause_state=0, next_mode=0,
                        practice=False, spell_practice=False, spell_active=False, normal_ending=False,
                        demo=False, continues=0, retry_menu=0, stage_rng_seed=27109)

        def read_dialogue_wait(self):
            if len(self.commands) >= 4:
                return None
            return dict(index=0, opcode=21, selected_option=int(len(self.commands) < 2))

        def advance(self, mask, ticks):
            self.commands.append((mask, ticks))
            return dict(world.state, input_poll=world.state["input_poll"] + ticks)

    original = Original()
    world = TH08StoryWorld(original, native_dialogue_edges=True)
    world.initial_poll = 40
    world.stage_history = [1, 2, 3, 4]
    world.state = dict(mode=2, player_state=0, native_replay=False, native_contact=False, misses=0,
                       bombs_used=0, deathbombs_used=0, input_poll=40, player_position=(192., 384.),
                       stage_timer=7482, difficulty=3, shot_type=0, stage=5)
    _, result = world._prepare(observe=False)
    assert original.commands == [(0, 4), (16, 4), (0, 4), (1, 4)]
    assert result["stage_history"] == [1, 2, 3, 4, 5] and result["native_dialogue_frames"] == 16
    assert result["outcome"] is None and not result["contact"]



def test_fast_original_replay_still_checks_contact_and_native_stage():
    from fly_touhou.th08_story import POLICY_MASKS, TH08StoryWorld

    class Original:
        def read_game_state(self, state):
            return dict(stage=1, raw_stage=0, active=True, raw_phase=0, pause_state=0, next_mode=0,
                        practice=False, spell_practice=False, spell_active=False, normal_ending=False,
                        demo=False, continues=0, retry_menu=0, stage_rng_seed=27109)

        def advance(self, mask, ticks):
            assert (mask, ticks) == (POLICY_MASKS[2], 4)
            return dict(world.state, input_poll=44, native_contact=True)

    world = TH08StoryWorld(Original())
    world.initial_poll = 40
    world.state = dict(mode=2, player_state=0, native_replay=False, native_contact=False, misses=0,
                       bombs_used=0, deathbombs_used=0, input_poll=40, player_position=(192., 384.),
                       stage_timer=241, difficulty=3, shot_type=0, stage=0)
    observation, start = world._prepare(observe=False)
    assert observation is None and start["outcome"] is None and start["stage_rng_seeds"] == [27109]
    observation, finish = world.step(2, observe=False)
    assert observation is None and finish["outcome"] == "nbnm-failure"
    assert finish["native_frames"] == 4 and finish["stage_history"] == [1]


def test_prefix_source_replays_the_recorded_story_then_the_diagnostic_commands(tmp_path):
    import json

    from fly_touhou.th08_prefix import read_prefix_source

    def row(index, frame, position, action, spell=None):
        return dict(decision=index, native_frames=frame, position=position, action=action,
                    raw_stage=4, stage_frame=6924, native_spell_id=spell)

    def write_trace(directory, rows):
        (directory / "decisions.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    source = tmp_path / "original" / "report.json"
    source_dir = source.parent / "episode-001"
    source_dir.mkdir(parents=True)
    original = dict(entry_mode="original-story", native_rng_controlled=True,
                    episodes=[dict(episode=1, stage_rng_seeds=[1, 2, 3, 4])])
    source.write_text(json.dumps(original), encoding="utf-8")
    write_trace(source_dir, [row(0, 0, [20, 30], 5), row(1, 4, [22, 30], 2), row(2, 8, [24, 30], 2)])
    branch = tmp_path / "diagnostic" / "report.json"
    branch.parent.mkdir()
    start = dict(row(0, 3, [22, 30], 4), stage_rng_seeds=[1, 2, 3, 4])
    report = dict(rollout_scope="remaining-story", source_report=str(source), source_decision=1,
                  start=start, end=dict(start, native_frames=11))
    branch.write_text(json.dumps(report), encoding="utf-8")
    suffix = [row(0, 3, [22, 30], 4), row(1, 7, [22, 38], 8, 92)]
    for r in suffix:
        del r["raw_stage"]  # Older single-stage local reports omit this field.
    write_trace(branch.parent, suffix)

    got, _, directory, tape = read_prefix_source(branch, through_decision=2)
    assert got == original and directory == source_dir
    assert [(r["decision"], r["action"], r["native_frames"]) for r in tape] == [(0, 5, 0), (1, 4, 4), (2, 8, 8)]
    assert tape[-1]["native_spell_id"] == 92
    proof = tmp_path / "native_route.json"
    proof_start = dict(tape[-1], native_frames=7, stage_rng_seeds=[1, 2, 3, 4])
    proof_end = dict(proof_start, native_frames=79, native_spell_id=None, action=6, contact=False, outcome=None)
    proof.write_text(json.dumps(dict(autonomous_policy=False, source_report=str(branch),
                                     matched_prefix_decisions=2, branch_entry={"info": proof_start},
                                     after=[proof_end])), encoding="utf-8")
    _, _, _, resumed = read_prefix_source(proof)
    assert resumed[-2]["action"] == 6 and resumed[-2]["native_spell_id"] == 92
    assert resumed[-1]["action"] is None and resumed[-1]["native_frames"] == 80
    assert resumed[-1]["native_spell_id"] is None

    suffix[0]["position"] = [0, 0]
    write_trace(branch.parent, suffix)
    with pytest.raises(ValueError, match="recorded Story join"):
        read_prefix_source(branch)


def test_corridor_keeps_a_safe_held_command_after_failed_greedy_turns():
    import numpy as np

    from fly_touhou.nbnm_corridor import HORIZONS
    from fly_touhou.th08_motion import TH08Motion
    from fly_touhou.th08_sensation import _rolling_corridors

    class DelayedHazard:
        has_hazards = True

        def path_margins(self, paths, first=0):
            ticks = first + np.arange(1, paths.shape[1])
            x = paths[:, 1:, 0]
            return np.where((ticks >= 25) & (x > 18.), -2., 3.)

    scene = DelayedHazard()
    motion = TH08Motion(position=(8., 180.), previous_input=0, minimum=(8., 16.),
                        maximum=(376., 432.), hitbox_minimum=(7., 179.),
                        hitbox_maximum=(9., 181.), laser_half_extents=(1., 1.),
                        fast_speed=4., fast_diagonal=2.8, focused_speed=2.,
                        focused_diagonal=1.4, speed_multiplier=(1., 1.), time_scale=1.)
    options = (0, 0x80)
    velocities = motion.velocities(options)
    straight = motion.paths(options, HORIZONS[-1])
    margins = np.minimum.accumulate(scene.path_margins(straight), axis=1)[:, np.asarray(HORIZONS) - 1].T
    ends = straight[:, 4]
    turn_paths = motion.command_paths(np.repeat(ends, 2, axis=0), np.repeat(velocities, 2, axis=0),
                                      np.tile(velocities, (2, 1)), 12)
    turn_gaps = scene.path_margins(turn_paths, first=4)

    corridor, courses = _rolling_corridors(scene, motion, velocities, ends, margins, turn_paths, turn_gaps)

    np.testing.assert_array_equal(corridor[0, 2:4], [.75, .75])
    np.testing.assert_array_equal(corridor[0, 4:6], [0., 0.])
    np.testing.assert_allclose(corridor[0, :2], [3. / 11., 3. / 11.])
    np.testing.assert_array_equal(corridor[1, 2:4], [-.5, -.5])
    np.testing.assert_array_equal(courses[0, 0, :12], 0)
    np.testing.assert_array_equal(courses[0, 0, 12:], -1)
    np.testing.assert_array_equal(courses[0, 1], 0)
    np.testing.assert_array_equal(courses[1], -1)


def test_active_relative_turn_reproduces_native_projectile_displacement():
    import numpy as np

    from fly_touhou.th08_projectiles import BULLET, active_relative_turns

    native_slots = np.zeros(3, dtype=BULLET)
    native_slots["ex_flags"] = [0, 0x40, 0x41]
    native_slots["position"][1] = [368.78418, 48.467583]
    native_slots["angle"][1] = -.6519774
    native_slots["speed"][1] = 3.4625
    native_slots["turn_speed"][1] = 2.
    native_slots["turn_angle"][1] = -.8 * np.pi
    native_slots["turn_timer"][1] = 59
    native_slots["turn_interval"][1] = 60
    native_slots["turn_limit"][1] = 1
    bullets = {name: native_slots[name] for name in BULLET.names}

    columns, forecast = active_relative_turns(bullets, np.ones(3, bool), 4)
    assert columns.tolist() == [1]
    np.testing.assert_allclose(forecast[-1, 0], [362.8317, 48.57451], rtol=0., atol=.0001)
