"""Time and compare TH14 forecast reuse on trusted local snapshot fixtures."""

import argparse
from contextlib import ExitStack
import json
import pickle
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np

from benchmark_waiting_aim import assert_equal


def run(repo, scenes, output):
    sys.path.insert(0, str(repo / "src"))
    from fly_touhou import native_curve_motion_paths as curve, native_ex_children as children
    from fly_touhou import native_curve_laser_motion
    from fly_touhou import native_ex_conditional, native_line_laser_paths as line, native_wall_geometry
    from fly_touhou.native_bullet_program import decode_program, predict_program
    from fly_touhou.native_projection_cache import share_native_projections
    from fly_touhou.nbnm_corridor import plan

    cases = dict(children=(children, "_coasting_suffix"),
                 curve=(curve, "projection_cache"), line=(line, "projection_cache"))
    comparisons = dict(children={"s3-14964"}, curve={"s4-16716"}, line={"s6-19788"})
    results = []
    repeated_children = 0
    repeated_velocities = 0

    def changed(disabled):
        stack = ExitStack()
        for key in disabled:
            module, method = cases[key]
            stack.enter_context(patch.object(module, method, new=lambda *args: None))
            if key == "curve":
                stack.enter_context(patch.object(native_curve_laser_motion, "projection_cache", new=lambda *args: None))
        return stack

    for path in sorted(scenes.glob("s*-*.pkl")):
        source = path.read_bytes()

        def measure(disabled):
            encoding, args, kwargs = pickle.loads(source)
            native_ex_conditional._CACHE.clear()
            curve._NODE_CACHE.clear()
            native_wall_geometry._cached_positions.cache_clear()
            with changed(disabled):
                start = perf_counter()
                result = plan(encoding, *args, **kwargs)
                return result, 1e3 * (perf_counter() - start)

        cases_for_scene = ("all", *(key for key, subset in comparisons.items() if path.stem in subset))
        for variant in cases_for_scene:
            disabled = tuple(cases) if variant == "all" else (variant,)
            baseline, _ = measure(disabled)
            tested, _ = measure(())
            assert_equal(baseline, tested, path.stem)
            pairs = []
            for first in (False, True, False, True, False):
                _, t = measure(() if first else disabled)
                _, u = measure(disabled if first else ())
                pairs.append((u, t) if first else (t, u))
            a, b = np.median(pairs, axis=0)
            results.append(dict(scene=path.stem, variant=variant, baseline_ms=a, optimized_ms=b,
                                repeat_pairs_ms=pairs, equal_numeric_outputs=True))
            print(path.stem, variant, round(a, 2), round(b, 2), flush=True)

        if path.stem == "s3-14964":
            encoding, _, _ = pickle.loads(source)
            raw_parents = []
            for row in encoding["source_native_forecasts"]["bullets"]:
                raw = row.get("program_raw")
                if raw and any(cmd[2] == 8192 for cmd in decode_program(bytes.fromhex(raw["native_ex_program_hex"]))):
                    raw_parents.append(raw)
            with share_native_projections():
                for horizon in (24, 96):
                    for raw in raw_parents:
                        arrays = horizon != 24
                        with changed(("children",)):
                            expected = predict_program(raw, horizon, _clearance_arrays=arrays)
                        actual = predict_program(raw, horizon, _clearance_arrays=arrays)
                        assert_equal(expected, actual, f"{path.stem}:{horizon}")
                        repeated_children += len(actual.get("virtual_children", ())) if actual is not None else 0
                cache = children.projection_cache("coasting_child")
                repeated_velocities = len(cache)
    output.write_text(json.dumps(dict(scenes=results, coasting_child_forecasts_compared=repeated_children,
                                      distinct_coasting_velocities=repeated_velocities), indent=2),
                      encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--scenes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    run(options.repo, options.scenes, options.output)
