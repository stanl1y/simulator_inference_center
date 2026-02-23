# Simulator Inference Center -- Message Schema

All messages are serialized with **msgpack** (`use_bin_type=True` on pack,
`raw=False` on unpack). Every request is a msgpack map (dict) with at least a
`"method"` key. Every response is a msgpack map with at least a `"status"` key.

## Common Envelope

### Request

```python
{
    "method": str,     # one of: "list_tasks", "load_task", "reset", "step",
                       #         "get_info", "disconnect"
    ...                # method-specific fields
}
```

### Successful Response

```python
{
    "status": "ok",
    ...                # method-specific fields
}
```

### Error Response

```python
{
    "status": "error",
    "error_type": str,    # e.g. "invalid_params", "backend_error"
    "message": str        # human-readable description
}
```

---

## Method: `list_tasks`

List all task names available in the current backend.

### Request

```python
{
    "method": "list_tasks"
}
```

### Response

```python
{
    "status": "ok",
    "tasks": ["task_name_1", "task_name_2", ...]   # list[str]
}
```

---

## Method: `load_task`

Load a task by name. This prepares the simulator environment for the specified
task. Can be called multiple times to switch tasks.

### Request

```python
{
    "method": "load_task",
    "task_name": str      # must be one of the names from list_tasks
}
```

### Response

```python
{
    "status": "ok",
    "task_info": {
        "task_name": str,
        "description": str,              # human-readable task description
        "action_space": {                 # describes expected action format
            "joint_positions": {
                "shape": [7],            # list[int]
                "dtype": "float64",
                "low": [-1.0, ...],
                "high": [1.0, ...]
            },
            "gripper": {
                "shape": [1],
                "dtype": "float64",
                "low": [0.0],
                "high": [1.0]
            }
        },
        "max_episode_steps": int
    }
}
```

---

## Method: `reset`

Reset the loaded task. Returns the initial observation. Must be called after
`load_task` and before the first `step`.

### Request

```python
{
    "method": "reset"
}
```

### Response

```python
{
    "status": "ok",
    "observation": <Observation>    # see Observation Encoding below
}
```

---

## Method: `step`

Execute one simulation step with the given action.

### Request

```python
{
    "method": "step",
    "action": <Action>              # see Action Encoding below
}
```

### Response

```python
{
    "status": "ok",
    "observation": <Observation>,   # see Observation Encoding below
    "reward": float,
    "terminated": bool,             # task succeeded or failed terminally
    "truncated": bool,              # episode exceeded max steps
    "info": {                       # backend-specific extra info
        ...
    }
}
```

---

## Method: `get_info`

Retrieve metadata about the backend and current session.

### Request

```python
{
    "method": "get_info"
}
```

### Response

```python
{
    "status": "ok",
    "backend_name": str,            # e.g. "libero"
    "backend_version": str,
    "current_task": str | None,     # None if no task loaded
    "action_space": {...} | None,   # None if no task loaded
    "observation_space": {...} | None
}
```

---

## Method: `disconnect`

Gracefully end the session. The server releases backend resources.

### Request

```python
{
    "method": "disconnect"
}
```

### Response

```python
{
    "status": "ok"
}
```

---

## Data Encoding

### Observation Encoding

An observation is a msgpack **map** (dict). It can contain scalar values,
arrays, and images. Every array or image is encoded as an **Image/Array
descriptor** to preserve shape and dtype through msgpack.

```python
# Full observation example
{
    "agentview_image": {
        "__type__": "ndarray",
        "shape": [256, 256, 3],    # list[int] -- H, W, C
        "dtype": "uint8",          # numpy dtype string
        "data": <bytes>            # raw bytes, row-major (C-order)
    },
    "eye_in_hand_image": {
        "__type__": "ndarray",
        "shape": [256, 256, 3],
        "dtype": "uint8",
        "data": <bytes>
    },
    "joint_positions": {
        "__type__": "ndarray",
        "shape": [7],
        "dtype": "float64",
        "data": <bytes>            # 7 * 8 = 56 bytes
    },
    "gripper_position": {
        "__type__": "ndarray",
        "shape": [1],
        "dtype": "float64",
        "data": <bytes>
    },
    "ee_pos": {
        "__type__": "ndarray",
        "shape": [3],
        "dtype": "float64",
        "data": <bytes>
    },
    "ee_quat": {
        "__type__": "ndarray",
        "shape": [4],
        "dtype": "float64",
        "data": <bytes>
    }
}
```

**Decoding on the client side (Python):**

```python
import numpy as np

def decode_ndarray(d: dict) -> np.ndarray:
    """Decode a serialized ndarray descriptor back to numpy."""
    assert d["__type__"] == "ndarray"
    return np.frombuffer(d["data"], dtype=d["dtype"]).reshape(d["shape"])
```

**Encoding on the server side (Python):**

```python
import numpy as np

def encode_ndarray(arr: np.ndarray) -> dict:
    """Encode a numpy array into a msgpack-safe descriptor."""
    return {
        "__type__": "ndarray",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": arr.tobytes(),
    }
```

The `__type__` sentinel key allows the receiver to distinguish ndarray
descriptors from plain dicts without ambiguity.

### Action Encoding

Actions are plain msgpack maps. Numeric arrays in actions use the same
ndarray descriptor format as observations.

```python
# Example action for Libero (7-DOF arm + gripper)
{
    "joint_positions": {
        "__type__": "ndarray",
        "shape": [7],
        "dtype": "float64",
        "data": <bytes>           # 56 bytes
    },
    "gripper": {
        "__type__": "ndarray",
        "shape": [1],
        "dtype": "float64",
        "data": <bytes>           # 8 bytes
    }
}
```

Backends may also accept flat Python lists for convenience in simple clients:

```python
# Simplified action (backend should accept this too)
{
    "joint_positions": [0.1, -0.2, 0.3, 0.0, 0.1, -0.1, 0.0],
    "gripper": [0.5]
}
```

The server/backend should normalize both forms internally.

---

## Summary Table

| Method       | Request Fields           | Key Response Fields                                      |
|--------------|--------------------------|----------------------------------------------------------|
| `list_tasks` | (none)                   | `tasks: list[str]`                                       |
| `load_task`  | `task_name: str`         | `task_info: dict`                                        |
| `reset`      | (none)                   | `observation: dict`                                      |
| `step`       | `action: dict`           | `observation: dict`, `reward`, `terminated`, `truncated`, `info` |
| `get_info`   | (none)                   | `backend_name`, `backend_version`, `current_task`, ...   |
| `disconnect` | (none)                   | (status only)                                            |
