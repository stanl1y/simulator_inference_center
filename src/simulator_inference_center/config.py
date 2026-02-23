"""Pydantic settings / config schema."""

from __future__ import annotations

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
