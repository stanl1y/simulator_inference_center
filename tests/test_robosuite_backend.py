"""Integration tests for the RobosuiteBackend.

These tests are skipped if robosuite is not installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

# Skip the entire module if robosuite is not importable.
try:
    from simulator_inference_center.backends.robosuite import RobosuiteBackend
    from simulator_inference_center.config import RobosuiteBackendConfig

    ROBOSUITE_AVAILABLE = True
except Exception:
    ROBOSUITE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not ROBOSUITE_AVAILABLE,
    reason="robosuite is not installed or failed to import",
)


@pytest.fixture(scope="module")
def backend():
    """Create a RobosuiteBackend with default config. Shared across tests in this module."""
    config = RobosuiteBackendConfig()
    be = RobosuiteBackend(config)
    yield be
    be.close()


class TestRobosuiteBackend:
    """Integration tests for the real RobosuiteBackend."""

    def test_list_tasks_returns_nonempty(self, backend):
        tasks = backend.list_tasks()
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        assert all(isinstance(t, str) for t in tasks)

    def test_list_tasks_contains_known_envs(self, backend):
        tasks = backend.list_tasks()
        for t in ["Lift", "Stack", "Door", "NutAssembly", "PickPlace"]:
            assert t in tasks, f"Expected {t!r} in task list"

    def test_load_task(self, backend):
        task_info = backend.load_task("Lift")
        assert "task_name" in task_info
        assert task_info["task_name"] == "Lift"
        assert "action_space" in task_info
        assert "shape" in task_info["action_space"]
        assert "dtype" in task_info["action_space"]
        assert "low" in task_info["action_space"]
        assert "high" in task_info["action_space"]

    def test_load_task_unknown_raises(self, backend):
        with pytest.raises(ValueError, match="Unknown task"):
            backend.load_task("this_task_does_not_exist_12345")

    def test_reset_returns_observation(self, backend):
        backend.load_task("Lift")
        obs = backend.reset()
        assert isinstance(obs, dict)
        assert len(obs) > 0
        # Should have at least one ndarray descriptor
        has_ndarray = any(
            isinstance(v, dict) and v.get("__type__") == "ndarray"
            for v in obs.values()
        )
        assert has_ndarray, f"Expected ndarray descriptors in observation, got keys: {list(obs.keys())}"

    def test_reset_without_load_raises(self, backend):
        """Reset on a fresh backend (before any load) should raise."""
        config = RobosuiteBackendConfig()
        fresh = RobosuiteBackend(config)
        try:
            with pytest.raises(RuntimeError):
                fresh.reset()
        finally:
            fresh.close()

    def test_step_returns_expected_keys(self, backend):
        backend.load_task("Lift")
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

    def test_step_observation_has_ndarrays(self, backend):
        """Observations returned by step should contain ndarray descriptors."""
        backend.load_task("Lift")
        backend.reset()

        info = backend.get_info()
        action_dim = info["action_space"]["shape"][0] if info.get("action_space") else 7
        action_arr = np.zeros(action_dim, dtype=np.float64)
        result = backend.step({"action": action_arr})

        obs = result["observation"]
        assert isinstance(obs, dict)
        has_ndarray = any(
            isinstance(v, dict) and v.get("__type__") == "ndarray"
            for v in obs.values()
        )
        assert has_ndarray, f"Expected ndarray descriptors in step observation, got keys: {list(obs.keys())}"

    def test_step_with_random_actions(self, backend):
        """Run a short episode with random actions."""
        backend.load_task("Lift")
        backend.reset()

        info = backend.get_info()
        action_dim = info["action_space"]["shape"][0] if info.get("action_space") else 7

        for i in range(5):
            action_arr = np.random.uniform(-1, 1, size=action_dim).astype(np.float64)
            result = backend.step({"action": action_arr})
            assert "observation" in result
            if result["terminated"] or result["truncated"]:
                break

    def test_step_without_reset_raises(self, backend):
        """Step before reset should raise RuntimeError."""
        config = RobosuiteBackendConfig()
        fresh = RobosuiteBackend(config)
        try:
            fresh.load_task("Lift")
            with pytest.raises(RuntimeError):
                action_arr = np.zeros(7, dtype=np.float64)
                fresh.step({"action": action_arr})
        finally:
            fresh.close()

    def test_step_with_ndarray_descriptor(self, backend):
        """Test that step() accepts ndarray descriptors in action."""
        from simulator_inference_center.protocol import encode_ndarray

        backend.load_task("Lift")
        backend.reset()

        info = backend.get_info()
        action_dim = info["action_space"]["shape"][0] if info.get("action_space") else 7
        action_arr = np.zeros(action_dim, dtype=np.float64)
        result = backend.step({"action": encode_ndarray(action_arr)})
        assert "observation" in result

    def test_get_info(self, backend):
        backend.load_task("Lift")
        info = backend.get_info()
        assert info["backend_name"] == "robosuite"
        assert "backend_version" in info
        assert "current_task" in info
        assert info["current_task"] == "Lift"
        assert "action_space" in info

    def test_get_info_action_space_shape(self, backend):
        """Action space info should have the expected structure."""
        backend.load_task("Lift")
        info = backend.get_info()
        action_space = info["action_space"]
        assert action_space is not None
        assert "shape" in action_space
        assert isinstance(action_space["shape"], list)
        assert len(action_space["shape"]) == 1
        assert action_space["shape"][0] > 0

    def test_close_and_reopen(self, backend):
        backend.load_task("Lift")
        backend.close()
        # After close, current task should be None
        info = backend.get_info()
        assert info["current_task"] is None

    def test_load_different_task(self, backend):
        """Test that backend can load a different task after the first."""
        backend.load_task("Lift")
        backend.reset()

        backend.load_task("Stack")
        obs = backend.reset()
        assert isinstance(obs, dict)
        assert len(obs) > 0

        info = backend.get_info()
        assert info["current_task"] == "Stack"
