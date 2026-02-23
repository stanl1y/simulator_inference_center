# Simulator Inference Center

ZMQ-based inference server that exposes robot simulators (e.g. LIBERO, robosuite) over a network protocol for remote policy evaluation.

## Quick Start

```bash
conda activate sim_inference_center
# Start server (Libero backend, default port 5555)
python scripts/run_server.py --simulator libero --port 5555
# Run tests (mock backend, no GPU needed)
pytest tests/test_server.py -v
# Run Libero integration tests (requires GPU + libero install)
pytest tests/test_libero_backend.py -v
# Start server (robosuite backend)
python scripts/run_server.py --simulator robosuite --port 5555
# Run robosuite integration tests (requires robosuite install)
pytest tests/test_robosuite_backend.py -v
```

## Conda Environment

- **Name:** `sim_inference_center` (cloned from `libero` env)
- **Python:** 3.8.13 -- all code uses `from __future__ import annotations` for modern type hint syntax
- **Libero source:** editable install from `/home/jaroslaw/Code/LIBERO_Workspace/LIBERO`
- **Robosuite:** v1.4.0 installed in env
- **Project install:** editable via `pip install -e .`

## Project Structure

```
src/simulator_inference_center/
    __init__.py             # Package init, __version__
    server.py               # InferenceServer: ZMQ ROUTER socket, poll loop, message dispatch
    session.py              # Session dataclass (per-client state)
    backend.py              # SimulatorBackend ABC (list_tasks, load_task, reset, step, get_info, close)
    config.py               # ServerConfig + LiberoBackendConfig + RobosuiteBackendConfig (pydantic-settings, SIM_ env prefix)
    protocol.py             # msgpack pack/unpack, encode_ndarray/decode_ndarray
    backends/
        __init__.py         # Backend registry: register_backend(), get_backend_class()
        libero.py           # LiberoBackend implementation
        robosuite.py        # RobosuiteBackend implementation
client/
    client.py               # SimulatorClient: ZMQ DEALER client with context manager
    example.py              # Full lifecycle example script
scripts/
    run_server.py           # CLI entrypoint (--simulator, --port, --task, --log-level)
tests/
    test_server.py          # Server dispatch tests using MockBackend (no simulator needed)
    test_libero_backend.py  # Libero integration tests (skipped if libero not available)
    test_robosuite_backend.py # Robosuite integration tests (skipped if robosuite not available)
docs/
    architecture.md         # Full system design (socket types, session lifecycle, error handling)
    message_schema.md       # Wire protocol: all request/response schemas, ndarray encoding
```

## Architecture

- **Transport:** ZMQ ROUTER (server) / DEALER (client) over TCP
- **Serialization:** msgpack with ndarray descriptors (`{"__type__": "ndarray", "shape": [...], "dtype": "...", "data": <bytes>}`)
- **Message framing:** `[identity, empty delimiter, msgpack body]`
- **Single-threaded:** one poll loop, blocking simulator calls (backends are not thread-safe)
- **Session management:** `dict[bytes, Session]` keyed by ZMQ identity, with configurable timeout reaping

## Protocol Methods

| Method       | Key Request Fields  | Key Response Fields                                    |
|--------------|---------------------|--------------------------------------------------------|
| list_tasks   | (none)              | tasks: list[str]                                       |
| load_task    | task_name: str      | task_info: dict                                        |
| reset        | (none)              | observation: dict                                      |
| step         | action: dict        | observation, reward, terminated, truncated, info       |
| get_info     | (none)              | backend_name, backend_version, current_task, ...       |
| disconnect   | (none)              | (status only)                                          |

## Error Types

`unknown_method`, `invalid_params`, `task_not_found`, `no_task_loaded`, `not_reset`, `backend_error`, `internal_error`

All errors return `{"status": "error", "error_type": "...", "message": "..."}`. Backend exceptions are caught and wrapped; no raw tracebacks reach clients.

## Configuration

Via env vars (prefix `SIM_`) or ServerConfig constructor:

| Env Var                | Default          | Description                     |
|------------------------|------------------|---------------------------------|
| SIM_BIND_ADDRESS       | tcp://*:5555     | ZMQ ROUTER bind address         |
| SIM_BACKEND            | libero           | Backend name from registry      |
| SIM_SESSION_TIMEOUT_S  | 300              | Idle session reap timeout (sec) |
| SIM_LOG_LEVEL          | INFO             | Logging level                   |
| SIM_LIBERO_TASK_SUITE  | libero_90        | Libero suite name               |
| SIM_LIBERO_RENDER_WIDTH| 256              | Render width                    |
| SIM_LIBERO_RENDER_HEIGHT| 256             | Render height                   |
| SIM_LIBERO_MAX_EPISODE_STEPS | 300       | Max steps per episode           |
| SIM_ROBOSUITE_ROBOT    | Panda            | Robot name (Panda, Sawyer, etc) |
| SIM_ROBOSUITE_CONTROLLER | (none)         | Controller type (OSC_POSE, etc) |
| SIM_ROBOSUITE_RENDER_WIDTH | 256          | Camera image width              |
| SIM_ROBOSUITE_RENDER_HEIGHT | 256         | Camera image height             |
| SIM_ROBOSUITE_CAMERA_NAMES | agentview,robot0_eye_in_hand | Comma-sep camera names |
| SIM_ROBOSUITE_USE_CAMERA_OBS | true       | Include camera observations     |
| SIM_ROBOSUITE_HORIZON  | 1000             | Max steps per episode           |
| SIM_ROBOSUITE_REWARD_SHAPING | false      | Dense reward shaping            |

## Adding a New Backend

1. Create `src/simulator_inference_center/backends/my_backend.py`
2. Subclass `SimulatorBackend` from `backend.py`, implement all 6 methods
3. Call `register_backend("my_backend", MyBackendClass)` at module level
4. Add import in `backends/__init__.py` `_discover_backends()`
5. Run with `--simulator my_backend`

## Testing

- `test_server.py` uses a `MockBackend` registered as `"mock"` -- tests all dispatch paths, error cases, session cleanup. No GPU or simulator needed.
- `test_libero_backend.py` uses the real `LiberoBackend` -- auto-skipped if libero is not installed. Requires GPU.
- `test_robosuite_backend.py` uses the real `RobosuiteBackend` -- auto-skipped if robosuite is not installed.
- Server tests spin up a real ZMQ server in a background thread on port 15555.
