# Simulator Inference Center -- Architecture

## 1. Project Structure

```
simulator_inference_center/
├── CLAUDE.md
├── docs/
│   ├── architecture.md          # This file
│   └── message_schema.md        # Wire-protocol message schemas
├── src/
│   └── simulator_inference_center/
│       ├── __init__.py
│       ├── server.py            # ZMQ server (ROUTER socket, session mgmt)
│       ├── session.py           # Per-client Session object
│       ├── backend.py           # Abstract SimulatorBackend interface
│       ├── config.py            # Pydantic settings / config schema
│       ├── protocol.py          # Message encode/decode helpers (msgpack)
│       └── backends/
│           ├── __init__.py
│           └── libero.py        # Libero simulator backend
├── tests/
│   ├── __init__.py
│   ├── test_server.py
│   ├── test_protocol.py
│   └── test_client.py          # Integration test client
├── scripts/
│   └── run_server.py           # Entry-point script
├── pyproject.toml
└── requirements.txt
```

## 2. Abstract SimulatorBackend Interface

All simulator backends implement this ABC. The server never touches simulator internals directly; it only calls these methods.

```python
from abc import ABC, abstractmethod
from typing import Any

class SimulatorBackend(ABC):
    """
    Abstract interface that every simulator must implement.
    All methods are synchronous (simulators are typically not async-safe).
    """

    @abstractmethod
    def list_tasks(self) -> list[str]:
        """Return the names of all available tasks."""
        ...

    @abstractmethod
    def load_task(self, task_name: str) -> dict[str, Any]:
        """
        Load/prepare the given task by name.
        Returns task metadata dict (e.g. description, action_space info).
        Raises ValueError if task_name is unknown.
        """
        ...

    @abstractmethod
    def reset(self) -> dict[str, Any]:
        """
        Reset the loaded task and return the initial observation.
        Must be called after load_task(). Raises RuntimeError if no task loaded.
        """
        ...

    @abstractmethod
    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Execute one environment step.
        `action` is a dict whose schema is backend-specific
        (e.g. {"joint_positions": [...], "gripper": 0.5}).

        Returns a dict with at least:
          - "observation": dict  (may contain images, state vectors, etc.)
          - "reward": float
          - "terminated": bool
          - "truncated": bool
          - "info": dict
        """
        ...

    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        """
        Return backend metadata: name, version, action space description,
        observation space description, currently loaded task, etc.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release all resources (GPU memory, simulator handles, etc.)."""
        ...
```

## 3. ZMQ Server Design

### Socket type: ROUTER

The server uses a single **ZMQ ROUTER** socket.

- ROUTER gives us the client identity frame automatically, which we use to
  multiplex sessions over a single port.
- Each connecting client is assigned an identity by ZMQ (or can set one
  explicitly via `zmq.IDENTITY`).
- The server is single-threaded with a poll loop; simulator `step()` calls
  are blocking, so one request is processed at a time. This is acceptable
  because simulation steps are the bottleneck and backends are not thread-safe.

### Bind address

Configured via `server.bind_address`, default `tcp://*:5555`.

### Message framing (over ROUTER)

Every message on the wire consists of ZMQ multipart frames:

```
Client -> Server:
  Frame 0: [identity]      # inserted by ROUTER automatically on recv
  Frame 1: [empty]         # delimiter
  Frame 2: [msgpack body]  # the request payload

Server -> Client:
  Frame 0: [identity]      # ROUTER uses this to route to the right client
  Frame 1: [empty]         # delimiter
  Frame 2: [msgpack body]  # the response payload
```

The client side uses a **DEALER** socket (which handles the identity/delimiter
frames transparently), so client code just sends/receives the msgpack body.

### Poll loop (pseudo-code)

```python
poller = zmq.Poller()
poller.register(router_socket, zmq.POLLIN)

while running:
    events = dict(poller.poll(timeout=1000))
    if router_socket in events:
        identity, _, body = router_socket.recv_multipart()
        request = msgpack.unpackb(body, raw=False)
        response = handle_request(identity, request)
        router_socket.send_multipart([identity, b"", msgpack.packb(response, use_bin_type=True)])
```

## 4. Session Lifecycle

```
Client                              Server
  │                                    │
  │──── DEALER connect ───────────────>│  (ZMQ assigns identity)
  │                                    │
  │──── list_tasks ───────────────────>│
  │<─── {tasks: [...]} ───────────────│
  │                                    │
  │──── load_task(task_name) ─────────>│  Server creates/reuses Session
  │<─── {task_info: {...}} ───────────│
  │                                    │
  │──── reset ────────────────────────>│
  │<─── {observation: {...}} ─────────│
  │                                    │
  │──── step(action) ─────────────────>│  (repeat N times)
  │<─── {observation, reward, ...} ───│
  │                                    │
  │──── disconnect ───────────────────>│  Server calls backend.close()
  │<─── {status: "ok"} ──────────────│
  │                                    │
  │──── (ZMQ socket close) ──────────>│
```

### Session object

The server maintains a `dict[bytes, Session]` keyed by ZMQ client identity.

```python
@dataclass
class Session:
    identity: bytes
    backend: SimulatorBackend
    task_loaded: bool = False
    steps: int = 0
    created_at: float = field(default_factory=time.time)
```

- A new `Session` is created on the first request from an unknown identity.
- `load_task` instantiates / reconfigures the backend for that task.
- `disconnect` (explicit) or identity timeout triggers `backend.close()` and
  session removal.

### Session timeout

If no message is received from a client for `server.session_timeout_s` seconds
(default 300), the server reaps the session and frees resources.

## 5. Error Handling Protocol

Every response is a msgpack dict. On success it contains the expected payload
plus `"status": "ok"`. On error:

```python
{
    "status": "error",
    "error_type": "<category>",   # see below
    "message": "<human-readable>"
}
```

### Error types

| `error_type`       | When                                                    |
|--------------------|---------------------------------------------------------|
| `unknown_method`   | Request `method` field not recognized                   |
| `invalid_params`   | Required fields missing or wrong types in request       |
| `task_not_found`   | `load_task` called with a name not in `list_tasks()`    |
| `no_task_loaded`   | `reset` or `step` called before `load_task`             |
| `not_reset`        | `step` called before `reset` (after load or terminal)   |
| `backend_error`    | Unhandled exception inside the backend                  |
| `internal_error`   | Unexpected server-side error                            |

The server catches all backend exceptions, wraps them in `backend_error`, and
logs the full traceback server-side. The client never receives a raw Python
traceback.

## 6. Config Schema

Configuration is loaded via Pydantic `BaseSettings` (supports env vars and
a YAML/JSON config file).

```python
from pydantic import Field
from pydantic_settings import BaseSettings

class ServerConfig(BaseSettings):
    model_config = {"env_prefix": "SIM_"}

    bind_address: str = Field(
        default="tcp://*:5555",
        description="ZMQ ROUTER bind address",
    )
    backend: str = Field(
        default="libero",
        description="Which SimulatorBackend to use",
    )
    session_timeout_s: float = Field(
        default=300.0,
        description="Seconds of inactivity before a session is reaped",
    )
    log_level: str = Field(default="INFO")


class LiberoBackendConfig(BaseSettings):
    model_config = {"env_prefix": "SIM_LIBERO_"}

    task_suite: str = Field(
        default="libero_goal",
        description="Libero task suite name (libero_goal, libero_spatial, ...)",
    )
    render_width: int = Field(default=256)
    render_height: int = Field(default=256)
    max_episode_steps: int = Field(default=300)
```

Environment variable examples:

```bash
SIM_BIND_ADDRESS=tcp://*:6000
SIM_BACKEND=libero
SIM_SESSION_TIMEOUT_S=600
SIM_LIBERO_TASK_SUITE=libero_spatial
SIM_LIBERO_RENDER_WIDTH=128
```
