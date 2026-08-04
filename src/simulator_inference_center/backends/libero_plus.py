"""LIBERO-Plus backend with camera extrinsics and dynamic camera control."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from simulator_inference_center.backends import register_backend
from simulator_inference_center.backends.libero import LiberoBackend, _encode_observation
from simulator_inference_center.config import LiberoPlusBackendConfig
from simulator_inference_center.protocol import decode_ndarray, encode_ndarray

if TYPE_CHECKING:
    from simulator_inference_center.task_store import TaskStore

logger = logging.getLogger(__name__)

# Camera names to include extrinsics for by default. Cameras not in the
# loaded MuJoCo model are silently skipped, so listing the full LIBERO
# tabletop roster here costs nothing for tasks that only render a subset.
_DEFAULT_CAMERAS = (
    "agentview", "frontview", "sideview", "birdview", "robot0_eye_in_hand",
)


def _get_camera_extrinsics(env, camera_names=_DEFAULT_CAMERAS) -> dict[str, Any]:
    """Extract camera position and quaternion from the MuJoCo model.

    Returns a dict mapping camera name to ``{"position": ..., "quaternion": ...}``
    where each value is an encoded ndarray descriptor.  Cameras that do not
    exist in the model are silently skipped.
    """
    extrinsics: dict[str, Any] = {}
    for cam_name in camera_names:
        try:
            cam_id = env.sim.model.camera_name2id(cam_name)
        except Exception:
            # Camera not present in this model -- skip.
            continue
        pos = np.array(env.sim.model.cam_pos[cam_id], dtype=np.float64).copy()
        quat = np.array(env.sim.model.cam_quat[cam_id], dtype=np.float64).copy()
        fovy = float(env.sim.model.cam_fovy[cam_id])  # vertical FOV in degrees
        extrinsics[cam_name] = {
            "position": encode_ndarray(np.ascontiguousarray(pos)),
            "quaternion": encode_ndarray(np.ascontiguousarray(quat)),
            "fovy_deg": fovy,
        }
    return extrinsics


_BASE_BODY_CANDIDATES = (
    "robot0_base", "robot0_link0", "base", "panda_link0", "panda_base",
)

# Link body names to expose poses for. Covers CtRNet-X's keypoint set
# (base = link0, plus link6 and link7 for the 3+3 wrist/flange keypoints).
_ROBOT_LINK_BODIES = (
    "robot0_link0", "robot0_link1", "robot0_link2", "robot0_link3",
    "robot0_link4", "robot0_link5", "robot0_link6", "robot0_link7",
    "robot0_right_hand",
)


def _get_robot_base_pose(env) -> dict[str, Any] | None:
    """Extract robot base body pose in world frame from MuJoCo. Tries several
    common body names (robosuite prefixes with ``robot0_``).
    """
    for name in _BASE_BODY_CANDIDATES:
        try:
            pos = np.array(env.sim.data.get_body_xpos(name), dtype=np.float64).copy()
            quat = np.array(env.sim.data.get_body_xquat(name), dtype=np.float64).copy()
            return {
                "body_name": name,
                "position": encode_ndarray(np.ascontiguousarray(pos)),
                "quaternion": encode_ndarray(np.ascontiguousarray(quat)),
            }
        except Exception:
            continue
    return None


def _get_robot_link_poses(env) -> dict[str, Any]:
    """Return a dict mapping link body name -> {position, quaternion} in world
    frame, for all robot arm links present in the model. Bypasses URDF-based
    FK by going to the authoritative MuJoCo state directly.
    """
    poses: dict[str, Any] = {}
    for name in _ROBOT_LINK_BODIES:
        try:
            pos = np.array(env.sim.data.get_body_xpos(name), dtype=np.float64).copy()
            quat = np.array(env.sim.data.get_body_xquat(name), dtype=np.float64).copy()
            poses[name] = {
                "position": encode_ndarray(np.ascontiguousarray(pos)),
                "quaternion": encode_ndarray(np.ascontiguousarray(quat)),
            }
        except Exception:
            continue
    return poses


class LiberoPlusBackend(LiberoBackend):
    """LIBERO backend with camera extrinsics and dynamic camera control.

    Extends :class:`LiberoBackend` with:
    - Camera extrinsics (position + quaternion) in every observation
    - ``set_camera()`` to move a camera at runtime
    - ``get_observation()`` to re-render without stepping
    """

    def __init__(
        self,
        config: LiberoPlusBackendConfig | None = None,
        task_store: TaskStore | None = None,
    ) -> None:
        if config is None:
            config = LiberoPlusBackendConfig()
        self._expose_extrinsics = config.expose_camera_extrinsics
        super().__init__(config=config, task_store=task_store)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, Any]:
        obs = super().reset()
        if self._expose_extrinsics and self._env is not None:
            obs["camera_extrinsics"] = _get_camera_extrinsics(self._env)
            base_pose = _get_robot_base_pose(self._env)
            if base_pose is not None:
                obs["robot_base_pose"] = base_pose
            link_poses = _get_robot_link_poses(self._env)
            if link_poses:
                obs["robot_link_poses"] = link_poses
        return obs

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        result = super().step(action)
        if self._expose_extrinsics and self._env is not None:
            result["observation"]["camera_extrinsics"] = _get_camera_extrinsics(
                self._env
            )
            base_pose = _get_robot_base_pose(self._env)
            if base_pose is not None:
                result["observation"]["robot_base_pose"] = base_pose
            link_poses = _get_robot_link_poses(self._env)
            if link_poses:
                result["observation"]["robot_link_poses"] = link_poses
        return result

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["backend_name"] = "libero_plus"
        info["expose_camera_extrinsics"] = self._expose_extrinsics
        return info

    # ------------------------------------------------------------------
    # New methods: camera control
    # ------------------------------------------------------------------

    def set_camera(
        self,
        camera_name: str,
        position: Any,
        quaternion: Any,
        fovy_deg: float | None = None,
    ) -> dict[str, Any]:
        """Set camera pose (and optionally the field of view) and return a
        freshly rendered observation.

        Parameters
        ----------
        camera_name:
            MuJoCo camera name (e.g. ``"agentview"``).
        position:
            Camera position as a 3-element array, list, or ndarray descriptor.
        quaternion:
            Camera orientation as a 4-element array, list, or ndarray descriptor.
        fovy_deg:
            Optional vertical field of view in degrees. ``None`` (the default)
            leaves the model's ``cam_fovy`` untouched, so every existing caller
            behaves exactly as before. When given, it changes the camera
            INTRINSICS (focal length), not just the pose -- the value is
            readable back from the returned observation's
            ``camera_extrinsics[camera_name]["fovy_deg"]``.

        Returns
        -------
        dict
            Encoded observation dict (same format as ``reset()``).
        """
        if self._env is None:
            raise RuntimeError("No task loaded. Call load_task() first.")

        # Decode ndarray descriptors if they were sent over the wire.
        if isinstance(position, dict) and position.get("__type__") == "ndarray":
            position = decode_ndarray(position)
        else:
            position = np.asarray(position, dtype=np.float64)

        if isinstance(quaternion, dict) and quaternion.get("__type__") == "ndarray":
            quaternion = decode_ndarray(quaternion)
        else:
            quaternion = np.asarray(quaternion, dtype=np.float64)

        try:
            cam_id = self._env.sim.model.camera_name2id(camera_name)
        except Exception:
            raise ValueError(
                f"Camera {camera_name!r} not found in the MuJoCo model."
            )

        self._env.sim.model.cam_pos[cam_id] = position.ravel()[:3]
        self._env.sim.model.cam_quat[cam_id] = quaternion.ravel()[:4]
        if fovy_deg is not None:
            fovy = float(fovy_deg)
            if not 1.0 < fovy < 179.0:
                raise ValueError(
                    f"fovy_deg must be in (1, 179) degrees; got {fovy}"
                )
            self._env.sim.model.cam_fovy[cam_id] = fovy
        self._env.sim.forward()

        return self.get_observation()

    def set_lighting(
        self,
        light_name: str | None = None,
        position: Any = None,
        direction: Any = None,
        diffuse: Any = None,
        ambient: Any = None,
        specular: Any = None,
        active: Any = None,
    ) -> dict[str, Any]:
        """Set real MuJoCo light properties and return a freshly rendered obs.

        This is the physical light path (equivalent to robosuite's
        LightingModder): it edits the MuJoCo model's ``light_*`` arrays so the
        renderer re-shades AND re-casts shadows. Purely additive -- any field
        left ``None`` is untouched.

        Parameters
        ----------
        light_name:
            MuJoCo light name (e.g. ``"light1"``). If ``None`` the change is
            applied to ALL lights in the scene.
        position / direction:
            3-vectors (world frame) for the light origin / aim direction.
        diffuse / ambient / specular:
            3-vectors (RGB in [0, 1]) for the respective light components.
        active:
            0/1 (or bool) to disable/enable the light.

        Returns
        -------
        dict
            Encoded observation dict (same format as ``reset()``).
        """
        if self._env is None:
            raise RuntimeError("No task loaded. Call load_task() first.")

        model = self._env.sim.model

        def _vec(v):
            if isinstance(v, dict) and v.get("__type__") == "ndarray":
                return decode_ndarray(v)
            return np.asarray(v, dtype=np.float64)

        if light_name is None:
            light_ids = list(range(int(model.nlight)))
        else:
            try:
                light_ids = [model.light_name2id(light_name)]
            except Exception:
                raise ValueError(
                    f"Light {light_name!r} not found in the MuJoCo model "
                    f"(have {int(model.nlight)} lights)."
                )

        for lid in light_ids:
            if position is not None:
                model.light_pos[lid] = _vec(position).ravel()[:3]
            if direction is not None:
                d = _vec(direction).ravel()[:3]
                n = np.linalg.norm(d)
                model.light_dir[lid] = d / n if n > 1e-9 else d
            if diffuse is not None:
                model.light_diffuse[lid] = _vec(diffuse).ravel()[:3]
            if ambient is not None:
                model.light_ambient[lid] = _vec(ambient).ravel()[:3]
            if specular is not None:
                model.light_specular[lid] = _vec(specular).ravel()[:3]
            if active is not None:
                model.light_active[lid] = int(np.asarray(active).ravel()[0])

        self._env.sim.forward()
        return self.get_observation()

    def get_observation(self) -> dict[str, Any]:
        """Re-render the current scene and return an observation dict.

        This is useful after ``set_camera()`` or at any time you want a
        fresh frame without advancing the simulation.

        Returns
        -------
        dict
            Encoded observation dict (same format as ``reset()``).
        """
        if self._env is None:
            raise RuntimeError("No task loaded. Call load_task() first.")

        # Re-render: OffScreenRenderEnv wraps the robosuite env.
        # Force observable update then grab observations from the inner env.
        self._env._update_observables(force=True)
        raw_obs = self._env.env._get_observations()
        obs = _encode_observation(raw_obs)

        if self._expose_extrinsics:
            obs["camera_extrinsics"] = _get_camera_extrinsics(self._env)
            base_pose = _get_robot_base_pose(self._env)
            if base_pose is not None:
                obs["robot_base_pose"] = base_pose
            link_poses = _get_robot_link_poses(self._env)
            if link_poses:
                obs["robot_link_poses"] = link_poses

        return obs


register_backend("libero_plus", LiberoPlusBackend)
