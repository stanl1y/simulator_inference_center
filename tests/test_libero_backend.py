"""Integration tests for the LiberoBackend (including LIBERO-PRO suites).

These tests are skipped if libero is not installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

# Skip the entire module if libero is not importable.
try:
    from simulator_inference_center.backends.libero import (
        LiberoBackend,
        _LIBERO_PRO_SUITES,
    )
    from simulator_inference_center.config import LiberoBackendConfig

    LIBERO_AVAILABLE = True
except Exception:
    LIBERO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not LIBERO_AVAILABLE,
    reason="libero is not installed or failed to import",
)


@pytest.fixture(scope="module")
def backend():
    """Create a LiberoBackend with default config. Shared across tests in this module."""
    config = LiberoBackendConfig()
    be = LiberoBackend(config)
    yield be
    be.close()


class TestLiberoBackend:
    """Integration tests for the real LiberoBackend."""

    def test_list_tasks_returns_nonempty(self, backend):
        tasks = backend.list_tasks()
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        assert all(isinstance(t, str) for t in tasks)

    def test_load_task(self, backend):
        tasks = backend.list_tasks()
        task_info = backend.load_task(tasks[0])
        assert "task_name" in task_info
        assert task_info["task_name"] == tasks[0]
        assert "action_space" in task_info
        assert "max_episode_steps" in task_info

    def test_load_task_unknown_raises(self, backend):
        with pytest.raises(ValueError, match="Unknown task"):
            backend.load_task("this_task_does_not_exist_12345")

    def test_reset_returns_observation(self, backend):
        tasks = backend.list_tasks()
        backend.load_task(tasks[0])
        obs = backend.reset()
        assert isinstance(obs, dict)
        # Should have at least one ndarray descriptor
        has_ndarray = any(
            isinstance(v, dict) and v.get("__type__") == "ndarray"
            for v in obs.values()
        )
        assert has_ndarray, f"Expected ndarray descriptors in observation, got keys: {list(obs.keys())}"

    def test_step_returns_expected_keys(self, backend):
        tasks = backend.list_tasks()
        backend.load_task(tasks[0])
        backend.reset()

        # Build a zero action with the right dimension
        info = backend.get_info()
        action_dim = info["action_space"]["shape"][0] if info.get("action_space") else 7
        action_arr = np.zeros(action_dim, dtype=np.float64)
        result = backend.step({"action": action_arr})

        assert "observation" in result
        assert "reward" in result
        assert "terminated" in result
        assert "truncated" in result
        assert isinstance(result["reward"], float)
        assert isinstance(result["terminated"], bool)
        assert isinstance(result["truncated"], bool)

    def test_get_info(self, backend):
        info = backend.get_info()
        assert info["backend_name"] == "libero"
        assert "backend_version" in info
        assert "current_task" in info

    def test_close_and_reopen(self, backend):
        tasks = backend.list_tasks()
        backend.load_task(tasks[0])
        backend.close()
        # After close, current task should be None
        info = backend.get_info()
        assert info["current_task"] is None

    def test_multiple_steps(self, backend):
        """Run a short episode with random actions."""
        tasks = backend.list_tasks()
        backend.load_task(tasks[0])
        backend.reset()

        info = backend.get_info()
        action_dim = info["action_space"]["shape"][0] if info.get("action_space") else 7

        for i in range(5):
            action_arr = np.random.uniform(-1, 1, size=action_dim).astype(np.float64)
            result = backend.step({"action": action_arr})
            if result["terminated"] or result["truncated"]:
                break


class TestLiberoProBackend:
    """Integration tests for LIBERO-PRO perturbation suites."""

    def test_pro_tasks_present_in_list(self, backend):
        """All 16 LIBERO-PRO sub-suites should contribute tasks to list_tasks()."""
        tasks = backend.list_tasks()
        for suite in _LIBERO_PRO_SUITES:
            prefix = f"{suite}/"
            matching = [t for t in tasks if t.startswith(prefix)]
            assert len(matching) > 0, (
                f"No tasks found for LIBERO-PRO suite {suite!r}. "
                f"Expected tasks prefixed with {prefix!r}."
            )

    def test_pro_task_count(self, backend):
        """Each LIBERO-PRO sub-suite should have exactly 10 tasks."""
        tasks = backend.list_tasks()
        for suite in _LIBERO_PRO_SUITES:
            prefix = f"{suite}/"
            matching = [t for t in tasks if t.startswith(prefix)]
            assert len(matching) == 10, (
                f"Expected 10 tasks for {suite!r}, got {len(matching)}"
            )

    def test_pro_total_task_count(self, backend):
        """Should have 160 LIBERO-PRO tasks total (16 suites x 10 tasks)."""
        tasks = backend.list_tasks()
        pro_tasks = [t for t in tasks if "/" in t]
        assert len(pro_tasks) == 160, (
            f"Expected 160 LIBERO-PRO tasks, got {len(pro_tasks)}"
        )

    def test_load_pro_task(self, backend):
        """Should be able to load a LIBERO-PRO task by its prefixed name."""
        tasks = backend.list_tasks()
        pro_tasks = [t for t in tasks if t.startswith("libero_spatial_task/")]
        assert len(pro_tasks) > 0, "No libero_spatial_task tasks found"
        task_info = backend.load_task(pro_tasks[0])
        assert "task_name" in task_info
        assert task_info["task_name"] == pro_tasks[0]
        assert "action_space" in task_info

    def test_reset_pro_task(self, backend):
        """Should be able to reset a LIBERO-PRO task and get observations."""
        tasks = backend.list_tasks()
        pro_tasks = [t for t in tasks if t.startswith("libero_goal_swap/")]
        assert len(pro_tasks) > 0
        backend.load_task(pro_tasks[0])
        obs = backend.reset()
        assert isinstance(obs, dict)
        has_ndarray = any(
            isinstance(v, dict) and v.get("__type__") == "ndarray"
            for v in obs.values()
        )
        assert has_ndarray

    def test_step_pro_task(self, backend):
        """Should be able to step through a LIBERO-PRO task."""
        tasks = backend.list_tasks()
        pro_tasks = [t for t in tasks if t.startswith("libero_object_lan/")]
        assert len(pro_tasks) > 0
        backend.load_task(pro_tasks[0])
        backend.reset()
        info = backend.get_info()
        action_dim = info["action_space"]["shape"][0] if info.get("action_space") else 7
        action_arr = np.zeros(action_dim, dtype=np.float64)
        result = backend.step({"action": action_arr})
        assert "observation" in result
        assert "reward" in result
        assert isinstance(result["terminated"], bool)

    def test_pro_and_base_tasks_coexist(self, backend):
        """Base suite tasks should still be present alongside LIBERO-PRO tasks."""
        tasks = backend.list_tasks()
        # Check a known base task exists without prefix
        base_tasks = [t for t in tasks if "/" not in t]
        assert len(base_tasks) > 0, "No base (non-PRO) tasks found"
        # Specifically check a libero_spatial base task
        assert any(
            "pick_up_the_black_bowl" in t and "/" not in t for t in tasks
        ), "Expected base libero_spatial tasks without prefix"
