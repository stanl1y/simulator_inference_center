"""Message encode/decode helpers (msgpack)."""

import msgpack
import numpy as np


def pack(obj: dict) -> bytes:
    """Serialize a dict to msgpack bytes."""
    return msgpack.packb(obj, use_bin_type=True)


def unpack(data: bytes) -> dict:
    """Deserialize msgpack bytes to a dict."""
    return msgpack.unpackb(data, raw=False)


def encode_ndarray(arr: np.ndarray) -> dict:
    """Encode a numpy array into a msgpack-safe descriptor."""
    return {
        "__type__": "ndarray",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": arr.tobytes(),
    }


def decode_ndarray(d: dict) -> np.ndarray:
    """Decode a serialized ndarray descriptor back to numpy."""
    assert d.get("__type__") == "ndarray"
    return np.frombuffer(d["data"], dtype=d["dtype"]).reshape(d["shape"])
