"""Integration tests for the LiberoBackend.

These tests are skipped if libero is not installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

# Skip the entire module if libero is not importable.
try:
    from simulator_inference_center.backends.libero import LiberoBackend
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
