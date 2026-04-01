"""Integration tests for the LiberoPlusBackend.

These tests are skipped if libero is not installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

# Skip the entire module if libero is not importable.
try:
    from simulator_inference_center.backends.libero_plus import LiberoPlusBackend
    from simulator_inference_center.config import LiberoPlusBackendConfig

    LIBERO_AVAILABLE = True
except Exception:
    LIBERO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not LIBERO_AVAILABLE,
    reason="libero is not installed or failed to import",
)


@pytest.fixture(scope="module")
def backend():
    """Create a LiberoPlusBackend with default config."""
    config = LiberoPlusBackendConfig()
    be = LiberoPlusBackend(config)
    yield be
    be.close()


@pytest.fixture(scope="module")
def loaded_backend(backend):
    """Return a backend with a task loaded and reset."""
    tasks = backend.list_tasks()
    backend.load_task(tasks[0])
    backend.reset()
    return backend


class TestLiberoPlusRegistration:
    """Test that LiberoPlusBackend is registered correctly."""

    def test_registered_as_libero_plus(self):
        from simulator_inference_center.backends import get_backend_class

        cls = get_backend_class("libero_plus")
        assert cls is LiberoPlusBackend

    def test_list_backends_includes_libero_plus(self):
        from simulator_inference_center.backends import list_backends

        backends = list_backends()
        assert "libero_plus" in backends


class TestLiberoPlusObservations:
    """Test that observations include camera extrinsics."""

    def test_reset_has_camera_extrinsics(self, backend):
        tasks = backend.list_tasks()
        backend.load_task(tasks[0])
        obs = backend.reset()
        assert "camera_extrinsics" in obs
        extrinsics = obs["camera_extrinsics"]
        assert isinstance(extrinsics, dict)
        # At minimum, agentview should be present.
        assert "agentview" in extrinsics
        cam = extrinsics["agentview"]
        assert "position" in cam
        assert "quaternion" in cam
        # Verify they are ndarray descriptors.
        assert cam["position"]["__type__"] == "ndarray"
        assert cam["position"]["shape"] == [3]
        assert cam["quaternion"]["__type__"] == "ndarray"
        assert cam["quaternion"]["shape"] == [4]

    def test_step_has_camera_extrinsics(self, backend):
        tasks = backend.list_tasks()
        backend.load_task(tasks[0])
        backend.reset()
        info = backend.get_info()
        action_dim = info["action_space"]["shape"][0] if info.get("action_space") else 7
        action_arr = np.zeros(action_dim, dtype=np.float64)
        result = backend.step({"action": action_arr})
        obs = result["observation"]
        assert "camera_extrinsics" in obs
        assert "agentview" in obs["camera_extrinsics"]

    def test_get_info_reports_libero_plus(self, backend):
        info = backend.get_info()
        assert info["backend_name"] == "libero_plus"
        assert info["expose_camera_extrinsics"] is True


class TestLiberoPlusSetCamera:
    """Test set_camera and get_observation."""

    def test_set_camera_returns_observation(self, loaded_backend):
        new_pos = np.array([0.5, 0.0, 1.5], dtype=np.float64)
        new_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        obs = loaded_backend.set_camera("agentview", new_pos, new_quat)
        assert isinstance(obs, dict)
        assert "camera_extrinsics" in obs
        # Verify the extrinsics reflect the new pose.
        cam = obs["camera_extrinsics"]["agentview"]
        pos_arr = np.frombuffer(
            cam["position"]["data"], dtype=cam["position"]["dtype"]
        ).reshape(cam["position"]["shape"])
        np.testing.assert_allclose(pos_arr, new_pos, atol=1e-6)
        quat_arr = np.frombuffer(
            cam["quaternion"]["data"], dtype=cam["quaternion"]["dtype"]
        ).reshape(cam["quaternion"]["shape"])
        np.testing.assert_allclose(quat_arr, new_quat, atol=1e-6)

    def test_set_camera_invalid_name_raises(self, loaded_backend):
        with pytest.raises(ValueError, match="not found"):
            loaded_backend.set_camera(
                "nonexistent_camera",
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            )

    def test_set_camera_accepts_lists(self, loaded_backend):
        obs = loaded_backend.set_camera(
            "agentview",
            [0.6, 0.1, 1.2],
            [1.0, 0.0, 0.0, 0.0],
        )
        assert isinstance(obs, dict)
        assert "agentview_image" in obs or "camera_extrinsics" in obs

    def test_get_observation_returns_current_frame(self, loaded_backend):
        obs = loaded_backend.get_observation()
        assert isinstance(obs, dict)
        assert "camera_extrinsics" in obs
        # Should have image data.
        has_ndarray = any(
            isinstance(v, dict) and v.get("__type__") == "ndarray"
            for k, v in obs.items()
            if k != "camera_extrinsics"
        )
        assert has_ndarray

    def test_get_observation_without_task_raises(self):
        config = LiberoPlusBackendConfig()
        be = LiberoPlusBackend(config)
        with pytest.raises(RuntimeError, match="No task loaded"):
            be.get_observation()
        be.close()

    def test_set_camera_without_task_raises(self):
        config = LiberoPlusBackendConfig()
        be = LiberoPlusBackend(config)
        with pytest.raises(RuntimeError, match="No task loaded"):
            be.set_camera("agentview", [0, 0, 0], [1, 0, 0, 0])
        be.close()


class TestLiberoPlusExtrinsicsDisabled:
    """Test behavior when expose_camera_extrinsics is disabled."""

    def test_no_extrinsics_when_disabled(self):
        config = LiberoPlusBackendConfig(expose_camera_extrinsics=False)
        be = LiberoPlusBackend(config)
        tasks = be.list_tasks()
        be.load_task(tasks[0])
        obs = be.reset()
        assert "camera_extrinsics" not in obs
        be.close()
