"""Per-client Session object."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from simulator_inference_center.backend import SimulatorBackend


@dataclass
class Session:
    identity: bytes
    backend: Optional[SimulatorBackend] = None
    simulator_name: Optional[str] = None
    task_loaded: bool = False
    needs_reset: bool = False
    steps: int = 0
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Update the last_active timestamp."""
        self.last_active = time.time()
