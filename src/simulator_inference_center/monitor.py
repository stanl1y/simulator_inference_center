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
        # session_id (hex str) -> session snapshot dict
        self._sessions: dict[str, dict[str, Any]] = {}
        # session_id (hex str) -> {camera_name: np.ndarray}
        self._latest_images: dict[str, dict[str, np.ndarray]] = {}
        # ring buffer of recent log lines
        self._log_buffer: collections.deque[str] = collections.deque(maxlen=100)

    def set_server_info(self, *, backend: str, bind_address: str) -> None:
        with self._lock:
            self._backend_name = backend
            self._bind_address = bind_address

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

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "backend_name": self._backend_name,
                "bind_address": self._bind_address,
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
