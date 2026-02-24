"""Thread-safe JSON persistence for custom task configurations."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LiberoTaskConfig:
    """Configuration for a custom LIBERO task."""

    task_name: str
    language: str
    workspace: str
    fixtures: dict[str, int] = field(default_factory=dict)
    objects: dict[str, int] = field(default_factory=dict)
    regions: list[dict[str, Any]] = field(default_factory=list)
    objects_of_interest: list[str] = field(default_factory=list)
    goal_states: list[list[Any]] = field(default_factory=list)
    bddl_file_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LiberoTaskConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RobosuiteTaskConfig:
    """Configuration for a custom robosuite task."""

    task_name: str
    description: str = ""
    robot: str = "Panda"
    controller: Optional[str] = None
    base_env: str = "Lift"
    horizon: int = 1000
    reward_type: str = "sparse"
    camera_names: list[str] = field(default_factory=lambda: ["agentview", "robot0_eye_in_hand"])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobosuiteTaskConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# TaskStore
# ---------------------------------------------------------------------------

def _sanitize_name(name: str) -> str:
    """Convert a task name to a safe filename (no extension)."""
    # Replace non-alphanumeric chars (except underscore/hyphen) with underscore
    sanitized = re.sub(r"[^\w\-]", "_", name)
    # Collapse runs of underscores
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "unnamed"


class TaskStore:
    """Thread-safe JSON persistence for custom task configurations.

    Stores configs as individual JSON files under:
        {base_dir}/libero/{sanitized_name}.json
        {base_dir}/robosuite/{sanitized_name}.json
    """

    def __init__(self, base_dir: str) -> None:
        self._base_dir = Path(os.path.expanduser(base_dir))
        self._libero_dir = self._base_dir / "libero"
        self._robosuite_dir = self._base_dir / "robosuite"
        self._lock = threading.Lock()

        # Create directories
        self._libero_dir.mkdir(parents=True, exist_ok=True)
        self._robosuite_dir.mkdir(parents=True, exist_ok=True)
        logger.info("TaskStore initialized at %s", self._base_dir)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    # ------------------------------------------------------------------
    # LIBERO tasks
    # ------------------------------------------------------------------

    def save_libero_task(self, config: LiberoTaskConfig) -> Path:
        """Persist a LIBERO task config. Returns the JSON file path."""
        filename = _sanitize_name(config.task_name) + ".json"
        filepath = self._libero_dir / filename
        with self._lock:
            filepath.write_text(json.dumps(config.to_dict(), indent=2))
        logger.info("Saved LIBERO task %r to %s", config.task_name, filepath)
        return filepath

    def list_libero_tasks(self) -> list[LiberoTaskConfig]:
        """Return all persisted LIBERO task configs."""
        configs: list[LiberoTaskConfig] = []
        with self._lock:
            for filepath in sorted(self._libero_dir.glob("*.json")):
                try:
                    data = json.loads(filepath.read_text())
                    configs.append(LiberoTaskConfig.from_dict(data))
                except Exception:
                    logger.warning("Failed to load %s", filepath, exc_info=True)
        return configs

    def get_libero_task(self, task_name: str) -> LiberoTaskConfig | None:
        """Load a single LIBERO task config by name."""
        filename = _sanitize_name(task_name) + ".json"
        filepath = self._libero_dir / filename
        with self._lock:
            if not filepath.exists():
                return None
            try:
                data = json.loads(filepath.read_text())
                return LiberoTaskConfig.from_dict(data)
            except Exception:
                logger.warning("Failed to load %s", filepath, exc_info=True)
                return None

    # ------------------------------------------------------------------
    # Robosuite tasks
    # ------------------------------------------------------------------

    def save_robosuite_task(self, config: RobosuiteTaskConfig) -> Path:
        """Persist a robosuite task config. Returns the JSON file path."""
        filename = _sanitize_name(config.task_name) + ".json"
        filepath = self._robosuite_dir / filename
        with self._lock:
            filepath.write_text(json.dumps(config.to_dict(), indent=2))
        logger.info("Saved robosuite task %r to %s", config.task_name, filepath)
        return filepath

    def list_robosuite_tasks(self) -> list[RobosuiteTaskConfig]:
        """Return all persisted robosuite task configs."""
        configs: list[RobosuiteTaskConfig] = []
        with self._lock:
            for filepath in sorted(self._robosuite_dir.glob("*.json")):
                try:
                    data = json.loads(filepath.read_text())
                    configs.append(RobosuiteTaskConfig.from_dict(data))
                except Exception:
                    logger.warning("Failed to load %s", filepath, exc_info=True)
        return configs

    def get_robosuite_task(self, task_name: str) -> RobosuiteTaskConfig | None:
        """Load a single robosuite task config by name."""
        filename = _sanitize_name(task_name) + ".json"
        filepath = self._robosuite_dir / filename
        with self._lock:
            if not filepath.exists():
                return None
            try:
                data = json.loads(filepath.read_text())
                return RobosuiteTaskConfig.from_dict(data)
            except Exception:
                logger.warning("Failed to load %s", filepath, exc_info=True)
                return None

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_task(self, backend: str, task_name: str) -> bool:
        """Delete a persisted task config. Returns True if the file existed."""
        if backend == "libero":
            directory = self._libero_dir
        elif backend == "robosuite":
            directory = self._robosuite_dir
        else:
            return False

        filename = _sanitize_name(task_name) + ".json"
        filepath = directory / filename
        with self._lock:
            if filepath.exists():
                filepath.unlink()
                logger.info("Deleted %s task %r", backend, task_name)
                return True
        return False
