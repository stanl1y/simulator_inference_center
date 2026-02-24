"""Tests for ServerMonitor and Gradio dashboard."""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock

import msgpack
import numpy as np
import pytest
import zmq

from simulator_inference_center.backend import SimulatorBackend
from simulator_inference_center.backends import register_backend
from simulator_inference_center.config import ServerConfig
from simulator_inference_center.monitor import ServerMonitor
from simulator_inference_center.protocol import encode_ndarray, pack, unpack
from simulator_inference_center.server import InferenceServer
from simulator_inference_center.session import Session


# ---------------------------------------------------------------------------
# Mock backend (same as test_server.py, registered under "mock_dashboard")
# ---------------------------------------------------------------------------

class _MockBackend(SimulatorBackend):
    """A deterministic mock backend for testing monitor integration."""

    def __init__(self) -> None:
        self._task_loaded: str | None = None
        self._needs_reset = False
        self._steps = 0
        self.closed = False

    def list_tasks(self) -> list:
        return ["task_a", "task_b"]

    def load_task(self, task_name: str) -> dict:
        if task_name not in ("task_a", "task_b"):
            raise ValueError("Unknown task: %s" % task_name)
        self._task_loaded = task_name
        self._needs_reset = True
        return {
            "task_name": task_name,
            "description": "Mock %s" % task_name,
            "action_space": {"shape": [7], "dtype": "float64"},
            "max_episode_steps": 100,
        }

    def reset(self) -> dict:
        if self._task_loaded is None:
            raise RuntimeError("No task loaded")
        self._needs_reset = False
        self._steps = 0
        return {"state": encode_ndarray(np.zeros(3, dtype=np.float64))}

    def step(self, action: dict) -> dict:
        if self._task_loaded is None:
            raise RuntimeError("No task loaded")
        if self._needs_reset:
            raise RuntimeError("Needs reset")
        self._steps += 1
        return {
            "observation": {
                "state": encode_ndarray(np.ones(3, dtype=np.float64) * self._steps),
            },
            "reward": 1.0,
            "terminated": False,
            "truncated": self._steps >= 100,
            "info": {"step": self._steps},
        }

    def get_info(self) -> dict:
        return {
            "backend_name": "mock_dashboard",
            "backend_version": "0.0.1",
            "current_task": self._task_loaded,
            "action_space": None,
            "observation_space": None,
        }

    def close(self) -> None:
        self.closed = True
        self._task_loaded = None


register_backend("mock_dashboard", _MockBackend)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_session(
    identity: bytes = b"\x01\x02\x03\x04",
    task_loaded: bool = False,
    needs_reset: bool = False,
    steps: int = 0,
    current_task: str | None = None,
) -> Session:
    """Create a Session with a mock backend for monitor tests."""
    backend = MagicMock()
    backend.get_info.return_value = {
        "backend_name": "mock",
        "backend_version": "0.0.1",
        "current_task": current_task,
    }
    backend.close.return_value = None
    session = Session(identity=identity, backend=backend)
    session.task_loaded = task_loaded
    session.needs_reset = needs_reset
    session.steps = steps
    return session


# ---------------------------------------------------------------------------
# Monitor unit tests
# ---------------------------------------------------------------------------

class TestMonitorSnapshotDefaults:
    """test_monitor_snapshot_defaults"""

    def test_defaults(self):
        monitor = ServerMonitor()
        snap = monitor.get_snapshot()
        assert snap["backend_name"] == ""
        assert snap["bind_address"] == ""
        assert snap["total_requests"] == 0
        assert snap["sessions"] == {}
        assert snap["latest_images"] == {}
        assert snap["logs"] == []
        assert snap["uptime_s"] >= 0


class TestMonitorSetServerInfo:
    """test_monitor_set_server_info"""

    def test_set_server_info(self):
        monitor = ServerMonitor()
        monitor.set_server_info(backend="libero", bind_address="tcp://*:5555")
        snap = monitor.get_snapshot()
        assert snap["backend_name"] == "libero"
        assert snap["bind_address"] == "tcp://*:5555"


class TestMonitorSessionLifecycle:
    """test_monitor_session_lifecycle"""

    def test_create_and_remove(self):
        monitor = ServerMonitor()
        identity = b"\xaa\xbb\xcc\xdd"
        session = _make_mock_session(
            identity=identity,
            task_loaded=True,
            needs_reset=False,
            steps=5,
            current_task="task_a",
        )

        monitor.on_session_created(identity, session)
        snap = monitor.get_snapshot()
        sid = identity.hex()
        assert sid in snap["sessions"]
        info = snap["sessions"][sid]
        assert info["session_id"] == sid[:12]
        assert info["task"] == "task_a"
        assert info["steps"] == 5
        assert info["state"] == "running"

        monitor.on_session_removed(identity)
        snap = monitor.get_snapshot()
        assert sid not in snap["sessions"]

    def test_state_idle(self):
        monitor = ServerMonitor()
        identity = b"\x01"
        session = _make_mock_session(identity=identity, task_loaded=False)
        monitor.on_session_created(identity, session)
        snap = monitor.get_snapshot()
        info = snap["sessions"][identity.hex()]
        assert info["state"] == "idle"

    def test_state_needs_reset(self):
        monitor = ServerMonitor()
        identity = b"\x02"
        session = _make_mock_session(
            identity=identity, task_loaded=True, needs_reset=True,
        )
        monitor.on_session_created(identity, session)
        snap = monitor.get_snapshot()
        info = snap["sessions"][identity.hex()]
        assert info["state"] == "needs_reset"


class TestMonitorRequestTracking:
    """test_monitor_request_tracking"""

    def test_increments(self):
        monitor = ServerMonitor()
        identity = b"\x10"
        for _ in range(5):
            monitor.on_request(
                identity,
                method="list_tasks",
                request={"method": "list_tasks"},
                response={"status": "ok", "tasks": []},
                session=None,
            )
        snap = monitor.get_snapshot()
        assert snap["total_requests"] == 5


class TestMonitorImageExtraction:
    """test_monitor_image_extraction"""

    def test_ndarray_descriptor(self):
        monitor = ServerMonitor()
        identity = b"\x20"
        h, w = 256, 256
        image_data = bytes(h * w * 3)
        obs = {
            "agentview_image": {
                "__type__": "ndarray",
                "shape": [h, w, 3],
                "dtype": "uint8",
                "data": image_data,
            }
        }
        response = {"status": "ok", "observation": obs}
        session = _make_mock_session(identity=identity, task_loaded=True)
        monitor.on_request(identity, "step", {"method": "step"}, response, session)

        snap = monitor.get_snapshot()
        sid = identity.hex()
        assert sid in snap["latest_images"]
        assert "agentview_image" in snap["latest_images"][sid]
        arr = snap["latest_images"][sid]["agentview_image"]
        assert arr.shape == (h, w, 3)
        assert arr.dtype == np.uint8

    def test_raw_ndarray(self):
        monitor = ServerMonitor()
        identity = b"\x21"
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        obs = {"agentview_image": img}
        response = {"status": "ok", "observation": obs}
        monitor.on_request(identity, "reset", {"method": "reset"}, response, None)

        snap = monitor.get_snapshot()
        sid = identity.hex()
        assert sid in snap["latest_images"]
        np.testing.assert_array_equal(snap["latest_images"][sid]["agentview_image"], img)

    def test_no_images_on_error_response(self):
        monitor = ServerMonitor()
        identity = b"\x22"
        response = {"status": "error", "error_type": "backend_error", "message": "fail"}
        monitor.on_request(identity, "step", {"method": "step"}, response, None)

        snap = monitor.get_snapshot()
        assert identity.hex() not in snap["latest_images"]


class TestMonitorThreadSafety:
    """test_monitor_thread_safety"""

    def test_concurrent_requests(self):
        monitor = ServerMonitor()
        n_threads = 10
        n_per_thread = 100
        barrier = threading.Barrier(n_threads)

        def worker(tid: int) -> None:
            identity = bytes([tid])
            barrier.wait()
            for _ in range(n_per_thread):
                monitor.on_request(
                    identity,
                    method="list_tasks",
                    request={"method": "list_tasks"},
                    response={"status": "ok", "tasks": []},
                    session=None,
                )

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        snap = monitor.get_snapshot()
        assert snap["total_requests"] == n_threads * n_per_thread


class TestMonitorLogs:
    """Test add_log and log retrieval."""

    def test_add_log(self):
        monitor = ServerMonitor()
        monitor.add_log("line 1")
        monitor.add_log("line 2")
        snap = monitor.get_snapshot()
        assert snap["logs"] == ["line 1", "line 2"]

    def test_log_ring_buffer(self):
        monitor = ServerMonitor()
        for i in range(150):
            monitor.add_log(f"line {i}")
        snap = monitor.get_snapshot()
        # maxlen is 100
        assert len(snap["logs"]) == 100
        assert snap["logs"][0] == "line 50"
        assert snap["logs"][-1] == "line 149"


# ---------------------------------------------------------------------------
# Dashboard tests
# ---------------------------------------------------------------------------

class TestCreateDashboard:
    """test_create_dashboard"""

    def test_returns_blocks(self):
        gradio = pytest.importorskip("gradio")
        from simulator_inference_center.dashboard import create_dashboard

        monitor = ServerMonitor()
        app = create_dashboard(monitor)
        assert isinstance(app, gradio.Blocks)

    def test_format_uptime(self):
        from simulator_inference_center.dashboard import _format_uptime

        assert _format_uptime(0) == "00:00:00"
        assert _format_uptime(3661) == "01:01:01"
        assert _format_uptime(86399) == "23:59:59"


# ---------------------------------------------------------------------------
# Integration: server + monitor over ZMQ
# ---------------------------------------------------------------------------

def _run_server_no_signals(server: InferenceServer) -> None:
    """Run the server poll loop without signal handlers (for threads)."""
    server._context = zmq.Context()
    server._socket = server._context.socket(zmq.ROUTER)
    server._socket.bind(server.config.bind_address)
    server._running = True

    if server._monitor is not None:
        server._monitor.set_server_info(
            backend=server.config.backend,
            bind_address=server.config.bind_address,
        )

    poller = zmq.Poller()
    poller.register(server._socket, zmq.POLLIN)

    try:
        while server._running:
            events = dict(poller.poll(timeout=500))
            if server._socket in events:
                frames = server._socket.recv_multipart()
                if len(frames) < 3:
                    continue
                identity, _, body = frames[0], frames[1], frames[2]
                try:
                    request = unpack(body)
                except Exception:
                    response = server._make_error("invalid_params", "Failed to decode")
                    server._socket.send_multipart([identity, b"", pack(response)])
                    continue
                response = server.handle_request(identity, request)
                server._socket.send_multipart([identity, b"", pack(response)])
    finally:
        if server._socket is not None:
            server._socket.close()
        if server._context is not None:
            server._context.term()


class TestServerWithMonitor:
    """test_server_with_monitor -- integration test over ZMQ."""

    def test_monitor_reflects_requests(self):
        port = 18766
        bind_addr = "tcp://127.0.0.1:%d" % port

        monitor = ServerMonitor()
        config = ServerConfig(
            bind_address="tcp://*:%d" % port,
            backend="mock_dashboard",
            session_timeout_s=60.0,
            log_level="WARNING",
        )
        server = InferenceServer(config, monitor=monitor)
        thread = threading.Thread(
            target=_run_server_no_signals, args=(server,), daemon=True,
        )
        thread.start()
        time.sleep(0.3)

        ctx = zmq.Context()
        sock = ctx.socket(zmq.DEALER)
        sock.connect(bind_addr)
        time.sleep(0.1)

        try:
            # list_tasks
            sock.send_multipart([b"", msgpack.packb({"method": "list_tasks"}, use_bin_type=True)])
            assert sock.poll(3000, zmq.POLLIN)
            resp = msgpack.unpackb(sock.recv_multipart()[-1], raw=False)
            assert resp["status"] == "ok"

            # load_task
            sock.send_multipart([b"", msgpack.packb({"method": "load_task", "task_name": "task_a"}, use_bin_type=True)])
            assert sock.poll(3000, zmq.POLLIN)
            resp = msgpack.unpackb(sock.recv_multipart()[-1], raw=False)
            assert resp["status"] == "ok"

            # reset
            sock.send_multipart([b"", msgpack.packb({"method": "reset"}, use_bin_type=True)])
            assert sock.poll(3000, zmq.POLLIN)
            resp = msgpack.unpackb(sock.recv_multipart()[-1], raw=False)
            assert resp["status"] == "ok"

            # step
            sock.send_multipart([b"", msgpack.packb({"method": "step", "action": {"action": [0.0] * 7}}, use_bin_type=True)])
            assert sock.poll(3000, zmq.POLLIN)
            resp = msgpack.unpackb(sock.recv_multipart()[-1], raw=False)
            assert resp["status"] == "ok"

            # Check monitor state
            snap = monitor.get_snapshot()
            assert snap["backend_name"] == "mock_dashboard"
            assert snap["bind_address"] == "tcp://*:%d" % port
            assert snap["total_requests"] == 4  # list_tasks, load_task, reset, step
            assert len(snap["sessions"]) == 1

            # Check session info
            session_info = list(snap["sessions"].values())[0]
            assert session_info["task"] == "task_a"
            assert session_info["steps"] == 1
            assert session_info["state"] == "running"

            # disconnect
            sock.send_multipart([b"", msgpack.packb({"method": "disconnect"}, use_bin_type=True)])
            assert sock.poll(3000, zmq.POLLIN)
            resp = msgpack.unpackb(sock.recv_multipart()[-1], raw=False)
            assert resp["status"] == "ok"

            # Session should be removed
            snap = monitor.get_snapshot()
            assert snap["total_requests"] == 5
            assert len(snap["sessions"]) == 0

        finally:
            server._running = False
            sock.close()
            ctx.term()
            thread.join(timeout=5.0)
