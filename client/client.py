"""Reusable ZMQ client for the Simulator Inference Center server."""

from __future__ import annotations


from typing import Any

import msgpack
import numpy as np
import zmq


def _decode_ndarray(d: dict) -> np.ndarray:
    """Decode an ndarray descriptor dict back to a numpy array."""
    return np.frombuffer(d["data"], dtype=d["dtype"]).reshape(d["shape"]).copy()


def _decode_observation(obs: dict) -> dict[str, Any]:
    """Walk an observation dict, decoding any ndarray descriptors in place."""
    decoded: dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, dict) and value.get("__type__") == "ndarray":
            decoded[key] = _decode_ndarray(value)
        else:
            decoded[key] = value
    return decoded


def _encode_ndarray(arr: np.ndarray) -> dict:
    """Encode a numpy array into a msgpack-safe ndarray descriptor."""
    return {
        "__type__": "ndarray",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": arr.tobytes(),
    }


def _encode_action(action: dict[str, Any]) -> dict[str, Any]:
    """Encode numpy arrays in an action dict as ndarray descriptors."""
    encoded: dict[str, Any] = {}
    for key, value in action.items():
        if isinstance(value, np.ndarray):
            encoded[key] = _encode_ndarray(value)
        elif isinstance(value, list):
            encoded[key] = value
        else:
            encoded[key] = value
    return encoded


class SimulatorClient:
    """ZMQ DEALER client that speaks the simulator inference protocol."""

    def __init__(self, server_address: str = "tcp://localhost:5555") -> None:
        self._address = server_address
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None

    def connect(self) -> None:
        """Connect to the server."""
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.connect(self._address)

    def _send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a request and wait for the response."""
        if self._socket is None:
            raise RuntimeError("Not connected. Call connect() first.")
        body = msgpack.packb(request, use_bin_type=True)
        self._socket.send_multipart([b"", body])
        frames = self._socket.recv_multipart()
        return msgpack.unpackb(frames[-1], raw=False)

    def _check_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Check response status and raise on error."""
        if response.get("status") == "error":
            error_type = response.get("error_type", "unknown")
            message = response.get("message", "Unknown error")
            raise RuntimeError(f"Server error [{error_type}]: {message}")
        return response

    def list_tasks(self) -> list[str]:
        """Return the list of available task names."""
        resp = self._send_request({"method": "list_tasks"})
        self._check_response(resp)
        return resp["tasks"]

    def load_task(self, task_name: str) -> dict[str, Any]:
        """Load a task by name. Returns task info dict."""
        resp = self._send_request({
            "method": "load_task",
            "task_name": task_name,
        })
        self._check_response(resp)
        return resp["task_info"]

    def reset(self) -> dict[str, Any]:
        """Reset the loaded task. Returns decoded initial observation."""
        resp = self._send_request({"method": "reset"})
        self._check_response(resp)
        return _decode_observation(resp["observation"])

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute one step with the given action.

        Returns a dict with keys: observation, reward, terminated, truncated, info.
        The observation values are decoded back to numpy arrays.
        """
        resp = self._send_request({
            "method": "step",
            "action": _encode_action(action),
        })
        self._check_response(resp)
        return {
            "observation": _decode_observation(resp["observation"]),
            "reward": resp["reward"],
            "terminated": resp["terminated"],
            "truncated": resp["truncated"],
            "info": resp.get("info", {}),
        }

    def get_info(self) -> dict[str, Any]:
        """Get server/backend info."""
        resp = self._send_request({"method": "get_info"})
        self._check_response(resp)
        return resp

    def disconnect(self) -> None:
        """Send a disconnect message to the server (graceful session end)."""
        if self._socket is not None:
            try:
                resp = self._send_request({"method": "disconnect"})
                self._check_response(resp)
            except Exception:
                pass

    def close(self) -> None:
        """Disconnect from server and release ZMQ resources."""
        self.disconnect()
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

    def __enter__(self) -> "SimulatorClient":
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
