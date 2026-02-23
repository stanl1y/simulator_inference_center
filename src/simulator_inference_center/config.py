"""Pydantic settings / config schema."""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    model_config = {"env_prefix": "SIM_"}

    bind_address: str = Field(
        default="tcp://*:5555",
        description="ZMQ ROUTER bind address",
    )
    backend: str = Field(
        default="libero",
        description="Which SimulatorBackend to use",
    )
    session_timeout_s: float = Field(
        default=300.0,
        description="Seconds of inactivity before a session is reaped",
    )
    log_level: str = Field(default="INFO")


class LiberoBackendConfig(BaseSettings):
    model_config = {"env_prefix": "SIM_LIBERO_"}

    render_width: int = Field(default=256)
    render_height: int = Field(default=256)
    max_episode_steps: int = Field(default=300)


class RobosuiteBackendConfig(BaseSettings):
    model_config = {"env_prefix": "SIM_ROBOSUITE_"}

    robot: str = Field(
        default="Panda",
        description="Robot name (Panda, Sawyer, Jaco, IIWA, Kinova3, UR5e)",
    )
    controller: Optional[str] = Field(
        default=None,
        description="Controller type (OSC_POSE, JOINT_VELOCITY, etc). None = task default",
    )
    render_width: int = Field(default=256, description="Camera image width")
    render_height: int = Field(default=256, description="Camera image height")
    camera_names: str = Field(
        default="agentview,robot0_eye_in_hand",
        description="Comma-separated camera names",
    )
    use_camera_obs: bool = Field(
        default=True, description="Include camera image observations"
    )
    horizon: int = Field(default=1000, description="Max steps per episode")
    reward_shaping: bool = Field(
        default=False, description="Use dense reward shaping"
    )
