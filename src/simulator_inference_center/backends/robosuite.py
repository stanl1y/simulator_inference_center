"""Robosuite simulator backend."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

# Ensure MuJoCo can render off-screen without a display.  If neither
# MUJOCO_GL nor DISPLAY is set, default to osmesa so the server works
# headless out of the box.
if not os.environ.get("MUJOCO_GL") and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "osmesa"

import robosuite
import robosuite.controllers as suite_controllers

from simulator_inference_center.backend import SimulatorBackend
from simulator_inference_center.backends import register_backend
from simulator_inference_center.config import RobosuiteBackendConfig
from simulator_inference_center.protocol import encode_ndarray

logger = logging.getLogger(__name__)

# Tasks that require two robot arms.
_TWO_ARM_TASKS = {
    "TwoArmLift",
    "TwoArmPegInHole",
    "TwoArmHandover",
    "TwoArmTransport",
}


def _encode_observation(raw_obs: dict) -> dict[str, Any]:
    """Convert a raw robosuite observation dict into the wire-protocol format.

    All numpy array values are encoded as ndarray descriptors.  Robosuite
    already uses canonical key names (e.g. ``agentview_image``,
    ``robot0_joint_pos_cos``), so no remapping is performed.
    """
    encoded: dict[str, Any] = {}
    for key, value in raw_obs.items():
        if isinstance(value, np.ndarray):
            encoded[key] = encode_ndarray(np.ascontiguousarray(value))
    return encoded


class RobosuiteBackend(SimulatorBackend):
    """Backend wrapping the robosuite robotic manipulation simulator.

    Exposes all environments from ``robosuite.ALL_ENVIRONMENTS`` via the
    standard ``SimulatorBackend`` interface.
    """

    def __init__(self, config: RobosuiteBackendConfig | None = None) -> None:
        self._config = config or RobosuiteBackendConfig()
        self._env = None
        self._current_task: str | None = None
        self._needs_reset: bool = False
        self._episode_steps: int = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _close_env(self) -> None:
        """Close the current environment if one is open."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                logger.warning("Error closing robosuite env", exc_info=True)
            self._env = None

    def _robots_for_task(self, task_name: str) -> str | list[str]:
        """Return the robot(s) config appropriate for *task_name*."""
        if task_name in _TWO_ARM_TASKS:
            return [self._config.robot, self._config.robot]
        return self._config.robot

    # ------------------------------------------------------------------
    # SimulatorBackend interface
    # ------------------------------------------------------------------

    def list_tasks(self) -> list[str]:
        return list(robosuite.ALL_ENVIRONMENTS)

    def load_task(self, task_name: str) -> dict[str, Any]:
        if task_name not in robosuite.ALL_ENVIRONMENTS:
            raise ValueError(
                f"Unknown task {task_name!r}. "
                f"Use list_tasks() to see available tasks."
            )

        # Close any existing env before loading a new one.
        self._close_env()

        camera_list = [
            c.strip() for c in self._config.camera_names.split(",") if c.strip()
        ]

        # Build controller config if specified.
        controller_config = None
        if self._config.controller is not None:
            controller_config = suite_controllers.load_controller_config(
                default_controller=self._config.controller,
            )

        self._env = robosuite.make(
            env_name=task_name,
            robots=self._robots_for_task(task_name),
            has_renderer=False,
            has_offscreen_renderer=self._config.use_camera_obs,
            use_camera_obs=self._config.use_camera_obs,
            camera_names=camera_list,
            camera_heights=self._config.render_height,
            camera_widths=self._config.render_width,
            horizon=self._config.horizon,
            reward_shaping=self._config.reward_shaping,
            controller_configs=controller_config,
        )

        self._current_task = task_name
        self._needs_reset = True
        self._episode_steps = 0

        # Build action space description.
        action_low, action_high = self._env.action_spec
        action_dim = self._env.action_dim

        task_info: dict[str, Any] = {
            "task_name": task_name,
            "action_space": {
                "shape": [action_dim],
                "dtype": "float64",
                "low": action_low.tolist(),
                "high": action_high.tolist(),
            },
            "max_episode_steps": self._config.horizon,
        }

        logger.info("Loaded robosuite task %r", task_name)
        return task_info

    def reset(self) -> dict[str, Any]:
        if self._env is None:
            raise RuntimeError("No task loaded. Call load_task() first.")

        raw_obs = self._env.reset()

        self._needs_reset = False
        self._episode_steps = 0

        return _encode_observation(raw_obs)

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        if self._env is None:
            raise RuntimeError("No task loaded. Call load_task() first.")
        if self._needs_reset:
            raise RuntimeError("Environment needs reset. Call reset() first.")

        # Parse action: accept either ndarray descriptor or plain list.
        from simulator_inference_center.protocol import decode_ndarray

        raw_action = action.get("action")
        if raw_action is None:
            raise ValueError("Action must contain an 'action' key (flat array).")

        if isinstance(raw_action, dict) and raw_action.get("__type__") == "ndarray":
            action_arr = decode_ndarray(raw_action)
        else:
            action_arr = np.asarray(raw_action, dtype=np.float64).ravel()

        # robosuite returns a 4-tuple: (obs, reward, done, info).
        raw_obs, reward, done, info = self._env.step(action_arr)
        self._episode_steps += 1

        # Determine terminated vs truncated.
        # Check for task success via the environment's _check_success method.
        success = False
        if hasattr(self._env, "_check_success"):
            try:
                success = bool(self._env._check_success())
            except Exception:
                pass

        terminated = bool(done) and success
        truncated = bool(done) and not terminated

        if terminated or truncated:
            self._needs_reset = True

        return {
            "observation": _encode_observation(raw_obs),
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "info": dict(info) if info else {},
        }

    def get_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "backend_name": "robosuite",
            "backend_version": "0.1.0",
            "current_task": self._current_task,
            "num_tasks": len(robosuite.ALL_ENVIRONMENTS),
            "action_space": None,
            "observation_space": None,
        }

        if self._env is not None:
            action_low, action_high = self._env.action_spec
            info["action_space"] = {
                "shape": [self._env.action_dim],
                "dtype": "float64",
                "low": action_low.tolist(),
                "high": action_high.tolist(),
            }
            # List the observation keys currently available.
            obs_keys = list(self._env.observation_names) if hasattr(
                self._env, "observation_names"
            ) else []
            image_keys = [k for k in obs_keys if "image" in k]
            state_keys = [k for k in obs_keys if "image" not in k]
            info["observation_space"] = {
                "images": image_keys,
                "robot_state": state_keys,
                "image_shape": [
                    self._config.render_height,
                    self._config.render_width,
                    3,
                ],
            }

        return info

    def close(self) -> None:
        self._close_env()
        self._current_task = None
        self._needs_reset = False
        self._episode_steps = 0
        logger.info("RobosuiteBackend closed.")


register_backend("robosuite", RobosuiteBackend)
