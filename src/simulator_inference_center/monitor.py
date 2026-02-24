"""Thread-safe server state monitor for the Gradio dashboard."""

from __future__ import annotations

import collections
import threading
import time
from typing import Any

import numpy as np


class ServerMonitor:
    """Thread-safe observable store capturing server state for the dashboard."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._total_requests = 0
        self._backend_name: str = ""
        self._bind_address: str = ""
        self._available_backends: list[str] = []
        # session_id (hex str) -> session snapshot dict
        self._sessions: dict[str, dict[str, Any]] = {}
        # session_id (hex str) -> {camera_name: np.ndarray}
        self._latest_images: dict[str, dict[str, np.ndarray]] = {}
        # ring buffer of recent log lines
        self._log_buffer: collections.deque[str] = collections.deque(maxlen=100)
        # Callbacks invoked when a custom task is created via the dashboard.
        self._task_created_callbacks: list[Any] = []

    def set_server_info(
        self,
        *,
        bind_address: str,
        available_backends: list[str] | None = None,
    ) -> None:
        with self._lock:
            self._bind_address = bind_address
            if available_backends is not None:
                self._available_backends = list(available_backends)

    def on_session_created(self, identity: bytes, session: Any) -> None:
        sid = identity.hex()
        with self._lock:
            self._sessions[sid] = self._snapshot_session(sid, session)

    def on_session_removed(self, identity: bytes) -> None:
        sid = identity.hex()
        with self._lock:
            self._sessions.pop(sid, None)
            self._latest_images.pop(sid, None)

    def on_request(
        self,
        identity: bytes,
        method: str,
        request: dict[str, Any],
        response: dict[str, Any],
        session: Any | None,
    ) -> None:
        sid = identity.hex()
        with self._lock:
            self._total_requests += 1
            if session is not None:
                self._sessions[sid] = self._snapshot_session(sid, session)
            # Extract images from observation on reset/step
            if method in ("reset", "step") and response.get("status") == "ok":
                obs = response.get("observation", {})
                images = self._extract_images(obs)
                if images:
                    self._latest_images[sid] = images

    def add_log(self, line: str) -> None:
        with self._lock:
            self._log_buffer.append(line)

    def register_task_created_callback(self, fn: Any) -> None:
        """Register a callback invoked when a custom task is created.

        The callback signature is ``fn(backend: str, task_name: str, **kwargs)``.
        """
        with self._lock:
            self._task_created_callbacks.append(fn)

    def notify_task_created(self, backend: str, task_name: str, **kwargs: Any) -> None:
        """Invoke all registered task-created callbacks."""
        with self._lock:
            callbacks = list(self._task_created_callbacks)
        for cb in callbacks:
            try:
                cb(backend, task_name, **kwargs)
            except Exception:
                import logging
                logging.getLogger(__name__).warning(
                    "Task-created callback failed", exc_info=True
                )

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "backend_name": self._backend_name,
                "bind_address": self._bind_address,
                "available_backends": list(self._available_backends),
                "uptime_s": time.time() - self._start_time,
                "total_requests": self._total_requests,
                "sessions": dict(self._sessions),
                "latest_images": {
                    sid: {k: v.copy() for k, v in imgs.items()}
                    for sid, imgs in self._latest_images.items()
                },
                "logs": list(self._log_buffer),
            }

    @staticmethod
    def _snapshot_session(sid: str, session: Any) -> dict[str, Any]:
        if session.backend is None:
            return {
                "session_id": sid[:12],
                "simulator": session.simulator_name or "(none)",
                "task": "(none)",
                "steps": 0,
                "state": "no_simulator",
                "idle_s": round(time.time() - session.last_active, 1),
            }

        state = "idle"
        if session.task_loaded and session.needs_reset:
            state = "needs_reset"
        elif session.task_loaded and not session.needs_reset:
            state = "running"

        current_task = None
        try:
            info = session.backend.get_info()
            current_task = info.get("current_task")
        except Exception:
            pass

        return {
            "session_id": sid[:12],
            "simulator": session.simulator_name or "(unknown)",
            "task": current_task or "(none)",
            "steps": session.steps,
            "state": state,
            "idle_s": round(time.time() - session.last_active, 1),
        }

    @staticmethod
    def _extract_images(obs: dict[str, Any]) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for key in ("agentview_image", "eye_in_hand_image"):
            val = obs.get(key)
            if val is None:
                continue
            if isinstance(val, dict) and val.get("__type__") == "ndarray":
                arr = np.frombuffer(val["data"], dtype=val["dtype"]).reshape(val["shape"])
                images[key] = arr.copy()
            elif isinstance(val, np.ndarray):
                images[key] = val.copy()
        return images
