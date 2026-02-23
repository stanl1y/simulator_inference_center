"""Unit tests for the InferenceServer message dispatch using a mock backend."""

from __future__ import annotations

import threading
import time
from typing import Any

import msgpack
import numpy as np
import pytest
import zmq

from simulator_inference_center.backend import SimulatorBackend
from simulator_inference_center.backends import register_backend
from simulator_inference_center.config import ServerConfig
from simulator_inference_center.protocol import encode_ndarray, pack, unpack
from simulator_inference_center.server import InferenceServer


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

class MockBackend(SimulatorBackend):
    """A deterministic mock backend for testing server dispatch."""

    def __init__(self) -> None:
        self._task_loaded: str = None
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
        return {
            "state": encode_ndarray(np.zeros(3, dtype=np.float64)),
        }

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
            "backend_name": "mock",
            "backend_version": "0.0.1",
            "current_task": self._task_loaded,
            "action_space": None,
            "observation_space": None,
        }

    def close(self) -> None:
        self.closed = True
        self._task_loaded = None


# Register the mock backend so the server can find it.
register_backend("mock", MockBackend)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def server():
    """Create an InferenceServer with the mock backend (no ZMQ, just dispatch)."""
    config = ServerConfig(
        bind_address="tcp://*:0",
        backend="mock",
        session_timeout_s=60.0,
        log_level="DEBUG",
    )
    return InferenceServer(config)


# A fake client identity for direct handle_request calls.
FAKE_ID = b"\x00\x01\x02\x03"


# ---------------------------------------------------------------------------
# Tests -- direct handle_request dispatch (no ZMQ needed)
# ---------------------------------------------------------------------------

class TestServerDispatch:
    """Test server message dispatch with the mock backend."""

    def test_list_tasks(self, server):
        resp = server.handle_request(FAKE_ID, {"method": "list_tasks"})
        assert resp["status"] == "ok"
        assert resp["tasks"] == ["task_a", "task_b"]

    def test_load_task_success(self, server):
        resp = server.handle_request(FAKE_ID, {"method": "load_task", "task_name": "task_a"})
        assert resp["status"] == "ok"
        assert resp["task_info"]["task_name"] == "task_a"
        assert resp["task_info"]["description"] == "Mock task_a"

    def test_load_task_not_found(self, server):
        resp = server.handle_request(FAKE_ID, {"method": "load_task", "task_name": "no_such_task"})
        assert resp["status"] == "error"
        assert resp["error_type"] == "task_not_found"

    def test_load_task_missing_param(self, server):
        resp = server.handle_request(FAKE_ID, {"method": "load_task"})
        assert resp["status"] == "error"
        assert resp["error_type"] == "invalid_params"

    def test_reset_without_load(self, server):
        resp = server.handle_request(FAKE_ID, {"method": "reset"})
        assert resp["status"] == "error"
        assert resp["error_type"] == "no_task_loaded"

    def test_step_without_load(self, server):
        resp = server.handle_request(FAKE_ID, {"method": "step", "action": {}})
        assert resp["status"] == "error"
        assert resp["error_type"] == "no_task_loaded"

    def test_step_without_reset(self, server):
        server.handle_request(FAKE_ID, {"method": "load_task", "task_name": "task_a"})
        resp = server.handle_request(FAKE_ID, {"method": "step", "action": {}})
        assert resp["status"] == "error"
        assert resp["error_type"] == "not_reset"

    def test_full_lifecycle(self, server):
        # list_tasks
        resp = server.handle_request(FAKE_ID, {"method": "list_tasks"})
        assert resp["status"] == "ok"

        # load_task
        resp = server.handle_request(FAKE_ID, {"method": "load_task", "task_name": "task_a"})
        assert resp["status"] == "ok"

        # reset
        resp = server.handle_request(FAKE_ID, {"method": "reset"})
        assert resp["status"] == "ok"
        assert "state" in resp["observation"]

        # step
        action = {"action": [0.0] * 7}
        resp = server.handle_request(FAKE_ID, {"method": "step", "action": action})
        assert resp["status"] == "ok"
        assert "observation" in resp
        assert "reward" in resp
        assert resp["reward"] == 1.0
        assert resp["terminated"] is False

        # disconnect
        resp = server.handle_request(FAKE_ID, {"method": "disconnect"})
        assert resp["status"] == "ok"

    def test_step_missing_action(self, server):
        server.handle_request(FAKE_ID, {"method": "load_task", "task_name": "task_a"})
        server.handle_request(FAKE_ID, {"method": "reset"})
        resp = server.handle_request(FAKE_ID, {"method": "step"})
        assert resp["status"] == "error"
        assert resp["error_type"] == "invalid_params"

    def test_unknown_method(self, server):
        resp = server.handle_request(FAKE_ID, {"method": "bogus_method"})
        assert resp["status"] == "error"
        assert resp["error_type"] == "unknown_method"

    def test_missing_method_field(self, server):
        resp = server.handle_request(FAKE_ID, {"not_method": "list_tasks"})
        assert resp["status"] == "error"
        assert resp["error_type"] == "invalid_params"

    def test_get_info(self, server):
        resp = server.handle_request(FAKE_ID, {"method": "get_info"})
        assert resp["status"] == "ok"
        assert resp["backend_name"] == "mock"

    def test_observation_contains_ndarray_descriptor(self, server):
        server.handle_request(FAKE_ID, {"method": "load_task", "task_name": "task_a"})
        resp = server.handle_request(FAKE_ID, {"method": "reset"})
        assert resp["status"] == "ok"
        state = resp["observation"]["state"]
        assert state["__type__"] == "ndarray"
        arr = np.frombuffer(state["data"], dtype=state["dtype"]).reshape(state["shape"])
        np.testing.assert_array_equal(arr, np.zeros(3, dtype=np.float64))

    def test_step_increments_observation(self, server):
        server.handle_request(FAKE_ID, {"method": "load_task", "task_name": "task_b"})
        server.handle_request(FAKE_ID, {"method": "reset"})

        resp = server.handle_request(FAKE_ID, {"method": "step", "action": {"action": [0.0] * 7}})
        assert resp["status"] == "ok"
        state = resp["observation"]["state"]
        arr = np.frombuffer(state["data"], dtype=state["dtype"]).reshape(state["shape"])
        np.testing.assert_array_equal(arr, np.ones(3, dtype=np.float64))

        resp = server.handle_request(FAKE_ID, {"method": "step", "action": {"action": [0.0] * 7}})
        state = resp["observation"]["state"]
        arr = np.frombuffer(state["data"], dtype=state["dtype"]).reshape(state["shape"])
        np.testing.assert_array_equal(arr, np.ones(3, dtype=np.float64) * 2)

    def test_disconnect_cleans_session(self, server):
        server.handle_request(FAKE_ID, {"method": "load_task", "task_name": "task_a"})
        server.handle_request(FAKE_ID, {"method": "disconnect"})
        # After disconnect, the session should be gone.
        # A new request creates a new session -- reset without load should fail.
        resp = server.handle_request(FAKE_ID, {"method": "reset"})
        assert resp["status"] == "error"
        assert resp["error_type"] == "no_task_loaded"

    def test_multiple_clients_isolated(self, server):
        """Different client identities get separate sessions."""
        client_a = b"\xaa"
        client_b = b"\xbb"

        server.handle_request(client_a, {"method": "load_task", "task_name": "task_a"})
        server.handle_request(client_a, {"method": "reset"})

        # Client B has no task loaded.
        resp = server.handle_request(client_b, {"method": "reset"})
        assert resp["status"] == "error"
        assert resp["error_type"] == "no_task_loaded"

        # Client A can still step.
        resp = server.handle_request(client_a, {"method": "step", "action": {"action": [0.0] * 7}})
        assert resp["status"] == "ok"


# ---------------------------------------------------------------------------
# Integration test: actual ZMQ DEALER <-> ROUTER
# ---------------------------------------------------------------------------

def _run_server_no_signals(server):
    """Run the server poll loop without signal handlers (for threads)."""
    server._context = zmq.Context()
    server._socket = server._context.socket(zmq.ROUTER)
    server._socket.bind(server.config.bind_address)
    server._running = True

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


class TestZMQIntegration:
    """Test actual ZMQ DEALER-ROUTER message exchange."""

    def test_zmq_round_trip(self):
        """Full round-trip over ZMQ sockets."""
        port = 18765
        bind_addr = "tcp://127.0.0.1:%d" % port

        config = ServerConfig(
            bind_address="tcp://*:%d" % port,
            backend="mock",
            session_timeout_s=60.0,
            log_level="WARNING",
        )
        server = InferenceServer(config)
        thread = threading.Thread(target=_run_server_no_signals, args=(server,), daemon=True)
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
            assert "task_a" in resp["tasks"]

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
            assert "state" in resp["observation"]

            # step
            sock.send_multipart([b"", msgpack.packb({"method": "step", "action": {"action": [0.0] * 7}}, use_bin_type=True)])
            assert sock.poll(3000, zmq.POLLIN)
            resp = msgpack.unpackb(sock.recv_multipart()[-1], raw=False)
            assert resp["status"] == "ok"
            assert resp["reward"] == 1.0

            # disconnect
            sock.send_multipart([b"", msgpack.packb({"method": "disconnect"}, use_bin_type=True)])
            assert sock.poll(3000, zmq.POLLIN)
            resp = msgpack.unpackb(sock.recv_multipart()[-1], raw=False)
            assert resp["status"] == "ok"
        finally:
            server._running = False
            sock.close()
            ctx.term()
            thread.join(timeout=5.0)
