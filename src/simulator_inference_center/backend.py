"""Abstract SimulatorBackend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SimulatorBackend(ABC):
    """Abstract interface that every simulator must implement.

    All methods are synchronous (simulators are typically not async-safe).
    """

    @abstractmethod
    def list_tasks(self) -> list[str]:
        """Return the names of all available tasks."""
        ...

    @abstractmethod
    def load_task(self, task_name: str) -> dict[str, Any]:
        """Load/prepare the given task by name.

        Returns task metadata dict (e.g. description, action_space info).
        Raises ValueError if task_name is unknown.
        """
        ...

    @abstractmethod
    def reset(self) -> dict[str, Any]:
        """Reset the loaded task and return the initial observation.

        Must be called after load_task(). Raises RuntimeError if no task loaded.
        """
        ...

    @abstractmethod
    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute one environment step.

        `action` is a dict whose schema is backend-specific.
        Returns a dict with at least:
          - "observation": dict
          - "reward": float
          - "terminated": bool
          - "truncated": bool
          - "info": dict
        """
        ...

    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        """Return backend metadata: name, version, action/observation space, etc."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release all resources (GPU memory, simulator handles, etc.)."""
        ...
