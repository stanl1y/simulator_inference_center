# Simulator Inference Center

ZMQ-based inference server that exposes robot simulators (e.g. LIBERO, robosuite) over a network protocol for remote policy evaluation.

## Maintenance Rules

- Always modify README.md and CLAUDE.md after every major change or feature addition.

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
# Start server with Gradio dashboard
python scripts/run_server.py --simulator libero --port 5555 --dashboard --dashboard-port 7860
# Run dashboard tests
pytest tests/test_dashboard.py -v
```

## Conda Environment

- **Name:** `sim_inference_center` (cloned from `libero` env)
- **Python:** 3.8.13 -- all code uses `from __future__ import annotations` for modern type hint syntax
- **Libero source:** editable install from `/home/jaroslaw/Code/LIBERO_Workspace/LIBERO`
- **Robosuite:** v1.4.0 installed in env
- **Gradio:** 4.44.1 installed in env (for dashboard)
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
    monitor.py              # ServerMonitor: thread-safe state store for dashboard
    dashboard.py            # Gradio Blocks UI: server status, sessions, simulation images
    backends/
        __init__.py         # Backend registry: register_backend(), get_backend_class()
        libero.py           # LiberoBackend implementation
        robosuite.py        # RobosuiteBackend implementation
client/
    client.py               # SimulatorClient: ZMQ DEALER client with context manager
    example.py              # Full lifecycle example script
scripts/
    run_server.py           # CLI entrypoint (--simulator, --port, --task, --log-level, --dashboard, --dashboard-port)
tests/
    test_server.py          # Server dispatch tests using MockBackend (no simulator needed)
    test_libero_backend.py  # Libero integration tests (skipped if libero not available)
    test_robosuite_backend.py # Robosuite integration tests (skipped if robosuite not available)
    test_dashboard.py       # Monitor + dashboard + integration tests
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
- **Gradio dashboard:** optional web UI for real-time server monitoring, runs in a background thread via `ServerMonitor`

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

## Gradio Dashboard

The server includes an optional Gradio web dashboard for real-time monitoring. Enabled via `--dashboard` flag.

**Architecture:** `ServerMonitor` (thread-safe shared state) sits between the server poll loop and the Gradio UI thread. The server writes state on every request; Gradio reads it every 2 seconds.

**Dashboard panels:**
- Server status (backend, bind address, uptime, total requests, active sessions)
- Active sessions table (session ID, task, steps, state, idle time)
- Simulation view (rendered `agentview_image` for up to 4 concurrent sessions)
- Server logs (last 50 lines)

**Key files:**
- `monitor.py` — `ServerMonitor` class with thread-safe lock, session/image tracking, `get_snapshot()`
- `dashboard.py` — `create_dashboard(monitor)` builds Gradio Blocks app, `launch_dashboard()` runs in daemon thread
- `server.py` — accepts optional `monitor` param, hooks into request handling and session lifecycle

**CLI flags:**

| Flag               | Default | Description                     |
|--------------------|---------|---------------------------------|
| `--dashboard`      | False   | Launch Gradio visualization dashboard |
| `--dashboard-port` | 7860    | Port for the Gradio dashboard   |

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
- `test_dashboard.py` tests `ServerMonitor` (thread safety, image extraction, session lifecycle) and Gradio dashboard creation. Includes an integration test with a real ZMQ server + monitor on port 18766.
- Server tests spin up a real ZMQ server in a background thread on port 15555.
