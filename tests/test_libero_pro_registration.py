"""Unit tests for LIBERO-PRO benchmark registration and task map.

These tests verify that the LIBERO-PRO sub-suites are correctly registered in
the LIBERO benchmark system.  They are skipped if libero is not installed.
"""

from __future__ import annotations

import pytest

try:
    from libero.libero import benchmark as libero_benchmark
    from libero.libero.benchmark import libero_pro_suites, task_maps
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map

    LIBERO_AVAILABLE = True
except Exception:
    LIBERO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not LIBERO_AVAILABLE,
    reason="libero is not installed or failed to import",
)

# All 16 expected LIBERO-PRO sub-suite names.
EXPECTED_PRO_SUITES = [
    "libero_spatial_lan",
    "libero_spatial_object",
    "libero_spatial_swap",
    "libero_spatial_task",
    "libero_object_lan",
    "libero_object_object",
    "libero_object_swap",
    "libero_object_task",
    "libero_goal_lan",
    "libero_goal_object",
    "libero_goal_swap",
    "libero_goal_task",
    "libero_10_lan",
    "libero_10_object",
    "libero_10_swap",
    "libero_10_task",
]


class TestLiberoProTaskMap:
    """Verify the task map contains all LIBERO-PRO sub-suites."""

    def test_all_pro_suites_in_task_map(self):
        for suite in EXPECTED_PRO_SUITES:
            assert suite in libero_task_map, f"{suite!r} missing from libero_task_map"

    def test_each_pro_suite_has_10_tasks(self):
        for suite in EXPECTED_PRO_SUITES:
            tasks = libero_task_map[suite]
            assert len(tasks) == 10, (
                f"Expected 10 tasks for {suite!r}, got {len(tasks)}"
            )

    def test_pro_suite_tasks_match_base(self):
        """Each PRO sub-suite should have the same task names as its base suite."""
        base_suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
        perturbations = ["lan", "object", "swap", "task"]
        for base in base_suites:
            for pert in perturbations:
                pro_suite = f"{base}_{pert}"
                assert sorted(libero_task_map[pro_suite]) == sorted(
                    libero_task_map[base]
                ), f"Tasks in {pro_suite!r} do not match base {base!r}"


class TestLiberoProBenchmarkRegistration:
    """Verify that LIBERO-PRO benchmark classes are registered."""

    def test_all_pro_suites_registered(self):
        bench_dict = libero_benchmark.get_benchmark_dict()
        for suite in EXPECTED_PRO_SUITES:
            assert suite in bench_dict, (
                f"{suite!r} not registered in BENCHMARK_MAPPING"
            )

    def test_pro_benchmark_instantiation(self):
        bench_dict = libero_benchmark.get_benchmark_dict()
        for suite in EXPECTED_PRO_SUITES:
            bm = bench_dict[suite]()
            assert bm.get_num_tasks() == 10, (
                f"Benchmark {suite!r} has {bm.get_num_tasks()} tasks, expected 10"
            )

    def test_pro_benchmark_task_names(self):
        bench_dict = libero_benchmark.get_benchmark_dict()
        bm = bench_dict["libero_spatial_task"]()
        names = bm.get_task_names()
        assert len(names) == 10
        assert all(isinstance(n, str) for n in names)

    def test_pro_benchmark_bddl_paths(self):
        bench_dict = libero_benchmark.get_benchmark_dict()
        bm = bench_dict["libero_goal_object"]()
        for i in range(bm.get_num_tasks()):
            path = bm.get_task_bddl_file_path(i)
            assert "libero_goal_object" in path, (
                f"BDDL path should reference the pro sub-suite folder: {path}"
            )
            assert path.endswith(".bddl")

    def test_pro_suites_list_matches_expected(self):
        assert sorted(libero_pro_suites) == sorted(EXPECTED_PRO_SUITES)

    def test_task_maps_populated(self):
        for suite in EXPECTED_PRO_SUITES:
            assert suite in task_maps, f"{suite!r} missing from task_maps"
            assert len(task_maps[suite]) == 10
