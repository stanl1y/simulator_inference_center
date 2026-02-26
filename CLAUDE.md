# Simulator Inference Center

ZMQ-based inference server that exposes robot simulators (e.g. LIBERO, robosuite) over a network protocol for remote policy evaluation.

## Maintenance Rules

- Always modify README.md and CLAUDE.md after every major change or feature addition.

## Quick Start

```bash
conda activate sim_inference_center
pip install -e .                     # Core install
pip install -e ".[dashboard]"        # With Gradio dashboard
pip install -e ".[all]"              # Everything (dashboard + dev)
# Start server (clients select simulator dynamically via select_simulator)
sim-server --port 5555
# Run tests (mock backend, no GPU needed)
pytest tests/test_server.py -v
# Run Libero integration tests (requires GPU + libero install)
pytest tests/test_libero_backend.py -v
# Run robosuite integration tests (requires robosuite install)
pytest tests/test_robosuite_backend.py -v
# Start server with Gradio dashboard
sim-server --port 5555 --dashboard --dashboard-port 7860
# Run dashboard tests
pytest tests/test_dashboard.py -v
# Run task builder tests
pytest tests/test_task_builder.py -v
# Start server with dashboard + custom task store directory
sim-server --port 5555 --dashboard --task-store-dir ./my_tasks
```

## Conda Environment

- **Name:** `sim_inference_center` (cloned from `libero` env)
- **Python:** 3.8.13 -- all code uses `from __future__ import annotations` for modern type hint syntax
- **Libero source:** editable install from `/home/jaroslaw/Code/LIBERO_Workspace/LIBERO`
- **Robosuite:** v1.4.0 installed in env
- **Gradio:** 4.44.1 installed in env (for dashboard)
- **Project install:** editable via `pip install -e .`
- **Design principle:** simulators (LIBERO, robosuite) are installed in their own isolated envs, NOT in the inference center env. The inference center calls out to them.

## Project Structure

```
src/simulator_inference_center/
    __init__.py             # Package init, __version__
    cli.py                  # CLI entrypoint (sim-server command)
    server.py               # InferenceServer: ZMQ ROUTER socket, poll loop, message dispatch
    session.py              # Session dataclass (per-client state, optional backend + simulator_name)
    backend.py              # SimulatorBackend ABC (list_tasks, load_task, reset, step, get_info, close)
    config.py               # ServerConfig + LiberoBackendConfig + RobosuiteBackendConfig (pydantic-settings, SIM_ env prefix)
    protocol.py             # msgpack pack/unpack, encode_ndarray/decode_ndarray
    monitor.py              # ServerMonitor: thread-safe state store for dashboard + task-created callbacks
    dashboard.py            # Gradio Blocks UI: Monitor tab + Task Builder tab (when task_store provided)
    task_store.py           # Thread-safe JSON persistence for custom task configs (LiberoTaskConfig, RobosuiteTaskConfig)
    task_generator.py       # LIBERO BDDL generation, robosuite config validation, UI dropdown constants
    task_builder_ui.py      # Gradio UI components for Task Builder tab (LIBERO builder, robosuite builder, saved tasks)
    client/
        __init__.py         # SimulatorClient re-export
        client.py           # SimulatorClient: ZMQ DEALER client with context manager
    backends/
        __init__.py         # Backend registry: register_backend(), get_backend_class()
        libero.py           # LiberoBackend implementation (supports custom tasks via TaskStore)
        robosuite.py        # RobosuiteBackend implementation (supports custom tasks via TaskStore, "custom:" prefix)
client/
    example.py              # Full lifecycle example script
    example_robosuite.py    # Robosuite-specific example
scripts/
    run_server.py           # Legacy CLI wrapper (delegates to cli.py)
tests/
    test_server.py          # Server dispatch tests using MockBackend (no simulator needed)
    test_libero_backend.py  # Libero integration tests (skipped if libero not available)
    test_robosuite_backend.py # Robosuite integration tests (skipped if robosuite not available)
    test_dashboard.py       # Monitor + dashboard + integration tests
    test_task_builder.py    # TaskStore CRUD, task generator validation, dashboard integration, monitor callbacks
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
- **Backend selection:** clients must call `select_simulator` to choose a backend before any simulator methods; each session gets its own backend instance
- **Gradio dashboard:** optional web UI for real-time server monitoring, runs in a background thread via `ServerMonitor`
- **Task Builder:** optional dashboard tab for creating custom LIBERO/robosuite tasks without writing code; persisted as JSON via `TaskStore`

## Protocol Methods

| Method           | Key Request Fields  | Key Response Fields                                    |
|------------------|---------------------|--------------------------------------------------------|
| list_simulators  | (none)              | simulators: list[str]                                  |
| select_simulator | simulator: str      | simulator: str                                         |
| list_tasks       | (none)              | tasks: list[str]                                       |
| load_task        | task_name: str      | task_info: dict                                        |
| reset            | (none)              | observation: dict                                      |
| step             | action: dict        | observation, reward, terminated, truncated, info       |
| get_info         | (none)              | backend_name, backend_version, current_task, ...       |
| disconnect       | (none)              | (status only)                                          |

## Error Types

`unknown_method`, `invalid_params`, `task_not_found`, `no_task_loaded`, `not_reset`, `backend_error`, `internal_error`, `no_simulator_selected`, `unknown_simulator`, `simulator_already_selected`

All errors return `{"status": "error", "error_type": "...", "message": "..."}`. Backend exceptions are caught and wrapped; no raw tracebacks reach clients.

## Configuration

Via env vars (prefix `SIM_`) or ServerConfig constructor:

| Env Var                | Default          | Description                     |
|------------------------|------------------|---------------------------------|
| SIM_BIND_ADDRESS       | tcp://*:5555     | ZMQ ROUTER bind address         |
| SIM_SESSION_TIMEOUT_S  | 300              | Idle session reap timeout (sec) |
| SIM_LOG_LEVEL          | INFO             | Logging level                   |
| SIM_TASK_STORE_DIR     | ~/.simulator_inference_center/custom_tasks | Custom task config directory |
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

**Dashboard tabs:**
- **Monitor** — Server status, active sessions table, simulation view (up to 4 sessions), server logs
- **Task Builder** — Create custom tasks for LIBERO and robosuite without writing code (only shown when `task_store` is provided)

**Task Builder sub-tabs:**
- **LIBERO Task Builder** — Configure workspace, fixtures, objects, goal states; generates BDDL files via LIBERO's task generation pipeline
- **Robosuite Task Builder** — Configure base environment, robot, controller, horizon, cameras; saves parameterized config
- **Saved Tasks** — View and delete persisted custom tasks for both backends

**Key files:**
- `monitor.py` — `ServerMonitor` class with thread-safe lock, session/image tracking, `get_snapshot()`, task-created callbacks
- `dashboard.py` — `create_dashboard(monitor, task_store)` builds Gradio Blocks app with Monitor + Task Builder tabs
- `task_builder_ui.py` — `create_task_builder_tab(task_store, monitor)` builds the Task Builder sub-tabs
- `task_store.py` — `TaskStore` class for thread-safe JSON persistence of `LiberoTaskConfig` and `RobosuiteTaskConfig`
- `task_generator.py` — `LiberoTaskGenerator` (BDDL generation), `RobosuiteTaskGenerator` (validation), UI dropdown constants
- `server.py` — accepts optional `monitor` and `task_store` params, forwards `task_store` to backend constructors

**CLI flags:**

| Flag               | Default | Description                     |
|--------------------|---------|---------------------------------|
| `--dashboard`      | False   | Launch Gradio visualization dashboard |
| `--dashboard-port` | 7860    | Port for the Gradio dashboard   |
| `--task-store-dir` | ~/.simulator_inference_center/custom_tasks | Custom task config directory |

## Task Builder

The Task Builder allows users to create custom tasks for both LIBERO and robosuite simulators through the Gradio dashboard, without writing code.

**Custom task flow:**
1. User fills in task configuration in the Task Builder tab
2. For LIBERO: validates config, generates BDDL file via LIBERO's programmatic pipeline, saves config JSON
3. For robosuite: validates config, saves parameterized config JSON (uses `robosuite.make()` at runtime)
4. Custom tasks appear in `list_tasks()` and can be loaded via `load_task()` by connected clients
5. LIBERO custom tasks are named directly; robosuite custom tasks use a `custom:` prefix to avoid collisions

**Persistence:** Custom task configs are stored as JSON files under `{task_store_dir}/libero/` and `{task_store_dir}/robosuite/`. The `TaskStore` class handles thread-safe read/write.

**Live registration:** When a task is created via the dashboard, `ServerMonitor.notify_task_created()` fires registered callbacks so backends can pick up new tasks without restart.

## Adding a New Backend

1. Create `src/simulator_inference_center/backends/my_backend.py`
2. Subclass `SimulatorBackend` from `backend.py`, implement all 6 methods
3. Optionally accept `task_store: TaskStore | None = None` in `__init__` for custom task support
4. Call `register_backend("my_backend", MyBackendClass)` at module level
5. Add import in `backends/__init__.py` `_discover_backends()`
6. Clients can then select it via `select_simulator("my_backend")`

## Testing

- `test_server.py` uses a `MockBackend` registered as `"mock"` -- tests all dispatch paths, error cases, session cleanup, and dynamic backend selection. No GPU or simulator needed.
- `test_libero_backend.py` uses the real `LiberoBackend` -- auto-skipped if libero is not installed. Requires GPU.
- `test_robosuite_backend.py` uses the real `RobosuiteBackend` -- auto-skipped if robosuite is not installed.
- `test_dashboard.py` tests `ServerMonitor` (thread safety, image extraction, session lifecycle) and Gradio dashboard creation. Includes an integration test with a real ZMQ server + monitor on port 18766.
- `test_task_builder.py` tests `TaskStore` CRUD (save, list, get, delete, overwrite, thread safety), `LiberoTaskGenerator` / `RobosuiteTaskGenerator` validation, `ServerMonitor` task-created callbacks, dashboard creation with/without `task_store`, and server `task_store` forwarding.
- Server tests spin up a real ZMQ server in a background thread on port 18765.
