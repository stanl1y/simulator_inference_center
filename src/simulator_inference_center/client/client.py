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

    def list_simulators(self) -> list[str]:
        """Return the list of available simulator backends."""
        resp = self._send_request({"method": "list_simulators"})
        self._check_response(resp)
        return resp["simulators"]

    def select_simulator(self, simulator: str) -> dict[str, Any]:
        """Select which simulator backend to use for this session."""
        resp = self._send_request({
            "method": "select_simulator",
            "simulator": simulator,
        })
        self._check_response(resp)
        return resp

    def list_tasks(self, suite: str | None = None) -> list[str]:
        """Return the list of available task names.

        If *suite* is given, only return tasks belonging to that suite
        (e.g. ``"libero_spatial"``).  Requires backend support.
        """
        body: dict[str, Any] = {"method": "list_tasks"}
        if suite is not None:
            body["suite"] = suite
        resp = self._send_request(body)
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

    def set_camera(
        self,
        camera_name: str = "agentview",
        position: Any = None,
        quaternion: Any = None,
        fovy_deg: float | None = None,
    ) -> dict[str, Any]:
        """Set camera pose (and optionally field of view) and return new
        observation.

        Parameters
        ----------
        camera_name:
            MuJoCo camera name (e.g. ``"agentview"``).
        position:
            Camera position as a 3-element list or numpy array.
        quaternion:
            Camera orientation as a 4-element list or numpy array.
        fovy_deg:
            Optional vertical field of view in degrees. ``None`` (default)
            leaves the camera's intrinsics untouched. Read the APPLIED value
            back from the returned observation's
            ``camera_extrinsics[camera_name]["fovy_deg"]``.

        Returns
        -------
        dict
            Decoded observation dict with updated camera view.
        """
        body: dict[str, Any] = {
            "method": "set_camera",
            "camera_name": camera_name,
        }
        if position is not None:
            if isinstance(position, np.ndarray):
                body["position"] = _encode_ndarray(position)
            else:
                body["position"] = list(position)
        if quaternion is not None:
            if isinstance(quaternion, np.ndarray):
                body["quaternion"] = _encode_ndarray(quaternion)
            else:
                body["quaternion"] = list(quaternion)
        if fovy_deg is not None:
            body["fovy_deg"] = float(fovy_deg)
        resp = self._send_request(body)
        self._check_response(resp)
        return _decode_observation(resp["observation"])

    def set_lighting(
        self,
        light_name: str | None = None,
        position: Any = None,
        direction: Any = None,
        diffuse: Any = None,
        ambient: Any = None,
        specular: Any = None,
        active: Any = None,
    ) -> dict[str, Any]:
        """Set real MuJoCo light properties and return new observation.

        Edits the scene's physical ``light_*`` arrays so the render re-shades
        and re-casts shadows (the LightingModder path). Any field left ``None``
        is untouched. ``light_name=None`` applies to all lights.

        Parameters
        ----------
        light_name:
            MuJoCo light name (e.g. ``"light1"``), or ``None`` for all lights.
        position / direction:
            3-vectors (world frame).
        diffuse / ambient / specular:
            3-vectors (RGB in [0, 1]).
        active:
            0/1 to disable/enable the light.

        Returns
        -------
        dict
            Decoded observation dict with the relit scene.
        """
        body: dict[str, Any] = {"method": "set_lighting"}
        if light_name is not None:
            body["light_name"] = light_name
        for key, val in (
            ("position", position), ("direction", direction),
            ("diffuse", diffuse), ("ambient", ambient),
            ("specular", specular),
        ):
            if val is not None:
                body[key] = (_encode_ndarray(val) if isinstance(val, np.ndarray)
                             else list(val))
        if active is not None:
            body["active"] = int(active)
        resp = self._send_request(body)
        self._check_response(resp)
        return _decode_observation(resp["observation"])

    def get_observation(self) -> dict[str, Any]:
        """Get current observation without stepping the simulation.

        Returns
        -------
        dict
            Decoded observation dict.
        """
        resp = self._send_request({"method": "get_observation"})
        self._check_response(resp)
        return _decode_observation(resp["observation"])

    def get_depth(
        self,
        camera_name: str = "agentview",
        with_rgb: bool = False,
    ) -> dict[str, Any]:
        """Ground-truth METRIC depth (metres) for the current scene.

        The returned ``depth`` array is (H, W) float32 in METRES and is
        oriented EXACTLY like the matching ``<camera_name>_image`` in
        ``get_observation()`` (LIBERO default: row 0 = bottom of the image).
        Pass ``with_rgb=True`` to also get the RGB from the SAME render call,
        which lets a caller assert byte-equality with the observation image and
        so prove the depth is aligned rather than assume it.

        Returns
        -------
        dict
            ``{"depth": (H, W) float32 metres, "camera_name", "near", "far",
               "height", "width", "units", ["rgb": (H, W, 3) uint8]}``
        """
        resp = self._send_request({
            "method": "get_depth",
            "camera_name": camera_name,
            "with_rgb": bool(with_rgb),
        })
        self._check_response(resp)
        out: dict[str, Any] = {}
        for key, value in resp.items():
            if key == "status":
                continue
            if isinstance(value, dict) and value.get("__type__") == "ndarray":
                out[key] = _decode_ndarray(value)
            else:
                out[key] = value
        return out

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
