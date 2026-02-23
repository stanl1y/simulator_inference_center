"""Libero simulator backend."""

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

from simulator_inference_center.backend import SimulatorBackend
from simulator_inference_center.backends import register_backend
from simulator_inference_center.config import LiberoBackendConfig
from simulator_inference_center.protocol import encode_ndarray

logger = logging.getLogger(__name__)

# Mapping from canonical observation key names (sent over the wire) to the raw
# Libero observation dict keys.  The canonical names match the message schema
# in docs/message_schema.md.
_OBS_KEY_MAP = {
    # Images
    "agentview_image": "agentview_image",
    "eye_in_hand_image": "robot0_eye_in_hand_image",
    # Robot proprioception
    "joint_positions": "robot0_joint_pos",
    "joint_velocities": "robot0_joint_vel",
    "ee_pos": "robot0_eef_pos",
    "ee_quat": "robot0_eef_quat",
    "gripper_position": "robot0_gripper_qpos",
    "gripper_velocity": "robot0_gripper_qvel",
}

# Suites to load, in order.  libero_100 is excluded because its task map is
# broken in the current Libero install.
_SUITES_TO_LOAD = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
]


def _encode_observation(raw_obs: dict) -> dict[str, Any]:
    """Convert a raw Libero observation dict into the wire-protocol format.

    Keys are remapped to canonical names and all values are encoded as ndarray
    descriptors.
    """
    encoded: dict[str, Any] = {}
    for canonical, libero_key in _OBS_KEY_MAP.items():
        if libero_key in raw_obs:
            encoded[canonical] = encode_ndarray(
                np.ascontiguousarray(raw_obs[libero_key])
            )
    return encoded


class _TaskEntry:
    """Bookkeeping for a single task across suites."""

    __slots__ = ("suite_name", "benchmark", "index", "task_obj")

    def __init__(self, suite_name, benchmark, index, task_obj):
        self.suite_name = suite_name
        self.benchmark = benchmark
        self.index = index
        self.task_obj = task_obj


class LiberoBackend(SimulatorBackend):
    """Backend wrapping the LIBERO robotic manipulation benchmark.

    Loads tasks from all available Libero suites so that ``list_tasks()``
    returns the full catalogue.
    """

    def __init__(self, config: LiberoBackendConfig | None = None) -> None:
        self._config = config or LiberoBackendConfig()
        self._tasks: dict[str, _TaskEntry] = {}
        self._task_names: list[str] = []
        self._env = None
        self._current_task: str | None = None
        self._needs_reset: bool = False
        self._episode_steps: int = 0
        self._init_benchmarks()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_benchmarks(self) -> None:
        """Load all working Libero suites and build a unified task index."""
        from libero.libero import benchmark as libero_benchmark

        bench_dict = libero_benchmark.get_benchmark_dict()

        for suite_name in _SUITES_TO_LOAD:
            if suite_name not in bench_dict:
                logger.warning("Suite %r not found, skipping", suite_name)
                continue
            try:
                bm = bench_dict[suite_name]()
            except Exception:
                logger.warning(
                    "Failed to load suite %r, skipping", suite_name, exc_info=True
                )
                continue

            for idx in range(bm.get_num_tasks()):
                task_obj = bm.get_task(idx)
                name = task_obj.name
                if name in self._tasks:
                    # Duplicate across suites -- keep the first occurrence.
                    continue
                self._tasks[name] = _TaskEntry(suite_name, bm, idx, task_obj)
                self._task_names.append(name)

        logger.info("Loaded %d Libero tasks across suites", len(self._task_names))

    def _close_env(self) -> None:
        """Close the current environment if one is open."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                logger.warning("Error closing Libero env", exc_info=True)
            self._env = None

    # ------------------------------------------------------------------
    # SimulatorBackend interface
    # ------------------------------------------------------------------

    def list_tasks(self) -> list[str]:
        return list(self._task_names)

    def load_task(self, task_name: str) -> dict[str, Any]:
        if task_name not in self._tasks:
            raise ValueError(
                f"Unknown task {task_name!r}. "
                f"Use list_tasks() to see available tasks."
            )

        # Close any existing env before loading a new one.
        self._close_env()

        entry = self._tasks[task_name]
        bddl_path = entry.benchmark.get_task_bddl_file_path(entry.index)

        from libero.libero.envs import OffScreenRenderEnv

        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=self._config.render_height,
            camera_widths=self._config.render_width,
        )
        self._env.seed(0)

        self._current_task = task_name
        self._needs_reset = True
        self._episode_steps = 0

        # Build action space description.
        action_low, action_high = self._env.env.action_spec
        action_dim = self._env.env.action_dim

        task_info = {
            "task_name": task_name,
            "description": entry.task_obj.language,
            "action_space": {
                "shape": [action_dim],
                "dtype": "float64",
                "low": action_low.tolist(),
                "high": action_high.tolist(),
            },
            "max_episode_steps": self._config.max_episode_steps,
        }

        logger.info("Loaded task %r from suite %r", task_name, entry.suite_name)
        return task_info

    def reset(self) -> dict[str, Any]:
        if self._env is None:
            raise RuntimeError("No task loaded. Call load_task() first.")

        raw_obs = self._env.reset()

        # Apply an initial state from the benchmark.
        entry = self._tasks[self._current_task]
        init_states = entry.benchmark.get_task_init_states(entry.index)
        raw_obs = self._env.set_init_state(init_states[0])

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
            # Try to reconstruct from joint_positions + gripper fields.
            jp = action.get("joint_positions")
            gr = action.get("gripper")
            if jp is None:
                raise ValueError(
                    "Action must contain 'action' (flat array) or "
                    "'joint_positions' + 'gripper'."
                )
            if isinstance(jp, dict) and jp.get("__type__") == "ndarray":
                jp = decode_ndarray(jp)
            else:
                jp = np.asarray(jp, dtype=np.float64)

            if gr is not None:
                if isinstance(gr, dict) and gr.get("__type__") == "ndarray":
                    gr = decode_ndarray(gr)
                else:
                    gr = np.asarray(gr, dtype=np.float64)
                action_arr = np.concatenate([jp.ravel(), gr.ravel()])
            else:
                action_arr = np.asarray(jp, dtype=np.float64).ravel()
        else:
            if isinstance(raw_action, dict) and raw_action.get("__type__") == "ndarray":
                action_arr = decode_ndarray(raw_action)
            else:
                action_arr = np.asarray(raw_action, dtype=np.float64).ravel()

        raw_obs, reward, done, info = self._env.step(action_arr)
        self._episode_steps += 1

        terminated = bool(done)
        truncated = self._episode_steps >= self._config.max_episode_steps

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
            "backend_name": "libero",
            "backend_version": "0.1.0",
            "current_task": self._current_task,
            "num_tasks": len(self._task_names),
            "action_space": None,
            "observation_space": None,
        }

        if self._env is not None:
            action_low, action_high = self._env.env.action_spec
            info["action_space"] = {
                "shape": [self._env.env.action_dim],
                "dtype": "float64",
                "low": action_low.tolist(),
                "high": action_high.tolist(),
            }
            info["observation_space"] = {
                "images": ["agentview_image", "eye_in_hand_image"],
                "robot_state": [
                    "joint_positions",
                    "joint_velocities",
                    "ee_pos",
                    "ee_quat",
                    "gripper_position",
                    "gripper_velocity",
                ],
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
        logger.info("LiberoBackend closed.")


register_backend("libero", LiberoBackend)
