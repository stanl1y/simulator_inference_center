# Simulator Inference Center

ZMQ-based inference server that exposes robot simulators (e.g. [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [robosuite](https://robosuite.ai/)) over a network protocol for remote policy evaluation.

## Overview

Simulator Inference Center decouples policy code from simulator code by running the simulator behind a lightweight ZMQ server. A remote client connects, lists available tasks, loads one, and runs step loops — all over TCP with msgpack serialization. This lets you evaluate policies from any machine without needing the simulator installed locally.

**Key features:**

- **ROUTER/DEALER transport** over TCP with msgpack + ndarray binary encoding
- **Session isolation** — each client gets its own simulator backend instance
- **Pluggable backends** — ships with LIBERO and robosuite; add new simulators by subclassing `SimulatorBackend`
- **Dynamic backend selection** — clients choose their simulator at connect time via `select_simulator`
- **Configurable** via environment variables (`SIM_` prefix) or constructor args
- **Automatic session reaping** for idle clients
- **Gradio dashboard** — optional web UI for real-time server status, session monitoring, and rendered simulation images
- **Task Builder** — create custom LIBERO and robosuite tasks through the dashboard without writing code

## Installation

### 1. Set up the LIBERO conda environment

```bash
conda create -n libero python=3.8.13
conda activate libero
export PYTHONNOUSERSITE=1

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO

# Ensure core build tools are up to date
python -m pip install --upgrade setuptools wheel packaging

pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

### 2. Set up the robosuite conda environment

```bash
conda create -n robosuite python=3.10
conda activate robosuite
pip install mujoco
pip install robosuite
```

### 3. Install Simulator Inference Center

The server runs in its own conda environment, separate from the simulator environments. When a client requests a backend, the server activates the corresponding simulator env automatically.

```bash
conda create -n sim_inference_center python=3.8.13
conda activate sim_inference_center
cd /path/to/simulator_inference_center

# Core install (server + client)
pip install -e .

# With Gradio dashboard
pip install -e ".[dashboard]"

# With all optional dependencies (dashboard + dev)
pip install -e ".[all]"
```

**Client-only install** (on a remote machine that only needs to connect to the server):

```bash
pip install -e .
```

Then use:

```python
from simulator_inference_center.client import SimulatorClient
```

## Quick Start

**Start the server:**

```bash
conda activate sim_inference_center

# Start server (clients select simulator dynamically via select_simulator)
sim-server --port 5555

# With Gradio dashboard (includes Task Builder)
sim-server --port 5555 --dashboard --dashboard-port 7860

# With custom task store directory
sim-server --port 5555 --dashboard --task-store-dir ./my_tasks
```

**Connect a client:**

```python
from simulator_inference_center.client import SimulatorClient

with SimulatorClient("tcp://localhost:5555") as client:
    # Discover available backends
    simulators = client.list_simulators()   # e.g. ["libero", "robosuite"]

    # Select a backend for this session
    client.select_simulator("libero")

    # Now use the protocol as usual
    tasks = client.list_tasks()
    client.load_task(tasks[0])
    obs = client.reset()

    for _ in range(100):
        result = client.step({"action": my_policy(obs)})
        obs = result["observation"]
        if result["terminated"] or result["truncated"]:
            break
```

**Run the example scripts:**

```bash
# Generic example (works with any backend)
python client/example.py --address tcp://localhost:5555 --steps 50

# Robosuite-specific example
python client/example_robosuite.py --task Lift --steps 50 --episodes 2
```

## Protocol

All messages use msgpack over ZMQ with the frame format `[identity, empty delimiter, msgpack body]`.

| Method           | Key Request Fields  | Key Response Fields                                |
|------------------|---------------------|----------------------------------------------------|
| list_simulators  | ---                 | simulators: list[str]                              |
| select_simulator | simulator: str      | simulator: str                                     |
| list_tasks       | ---                 | tasks: list[str]                                   |
| load_task        | task_name: str      | task_info: dict                                    |
| reset            | ---                 | observation: dict                                  |
| step             | action: dict        | observation, reward, terminated, truncated, info   |
| get_info         | ---                 | backend_name, backend_version, current_task, ...   |
| disconnect       | ---                 | (status only)                                      |

Numpy arrays are encoded as `{"__type__": "ndarray", "shape": [...], "dtype": "...", "data": <bytes>}`.

Errors return `{"status": "error", "error_type": "...", "message": "..."}` with types: `unknown_method`, `invalid_params`, `task_not_found`, `no_task_loaded`, `not_reset`, `backend_error`, `internal_error`, `no_simulator_selected`, `unknown_simulator`, `simulator_already_selected`.

## Dynamic Backend Selection

Each client must call `list_simulators` to discover registered backends, then `select_simulator` to choose one for its session. Calling any simulator method (e.g. `list_tasks`) before selecting a simulator returns a `no_simulator_selected` error.

This allows a single server instance to expose multiple simulators (e.g. LIBERO and robosuite) and different clients can use different backends concurrently.

**Client flow:**

```
list_simulators  ->  ["libero", "robosuite"]
select_simulator("libero")  ->  ok
list_tasks  ->  [...]
load_task("...")  ->  ok
reset  ->  observation
step(action)  ->  observation, reward, ...
disconnect
```

**Error types for backend selection:**

| Error Type                  | Meaning                                                              |
|-----------------------------|----------------------------------------------------------------------|
| `no_simulator_selected`     | Client called a simulator method without first calling `select_simulator` |
| `unknown_simulator`         | The simulator name passed to `select_simulator` is not registered     |
| `simulator_already_selected`| Client tried to call `select_simulator` twice; disconnect first to switch |

## Configuration

All settings can be set via environment variables (prefix `SIM_`) or passed to `ServerConfig`:

| Env Var                      | Default        | Description                     |
|------------------------------|----------------|---------------------------------|
| SIM_BIND_ADDRESS             | tcp://*:5555   | ZMQ ROUTER bind address         |
| SIM_SESSION_TIMEOUT_S        | 300            | Idle session reap timeout (sec) |
| SIM_LOG_LEVEL                | INFO           | Logging level                   |
| SIM_TASK_STORE_DIR           | ~/.simulator_inference_center/custom_tasks | Custom task config directory |
| SIM_LIBERO_TASK_SUITE        | libero_90      | Libero suite name               |
| SIM_LIBERO_RENDER_WIDTH      | 256            | Render width                    |
| SIM_LIBERO_RENDER_HEIGHT     | 256            | Render height                   |
| SIM_LIBERO_MAX_EPISODE_STEPS | 300            | Max steps per episode           |
| SIM_ROBOSUITE_ROBOT          | Panda          | Robot name (Panda, Sawyer, etc) |
| SIM_ROBOSUITE_CONTROLLER     | (none)         | Controller type (OSC_POSE, etc) |
| SIM_ROBOSUITE_RENDER_WIDTH   | 256            | Camera image width              |
| SIM_ROBOSUITE_RENDER_HEIGHT  | 256            | Camera image height             |
| SIM_ROBOSUITE_CAMERA_NAMES   | agentview,robot0_eye_in_hand | Comma-sep camera names |
| SIM_ROBOSUITE_USE_CAMERA_OBS | true           | Include camera observations     |
| SIM_ROBOSUITE_HORIZON        | 1000           | Max steps per episode           |
| SIM_ROBOSUITE_REWARD_SHAPING | false          | Dense reward shaping            |

## Gradio Dashboard

The server includes an optional Gradio web dashboard for real-time monitoring of server state and simulation images.

```bash
sim-server --dashboard
# Dashboard at http://localhost:7860
```

| CLI Flag           | Default | Description                     |
|--------------------|---------|---------------------------------|
| `--dashboard`      | off     | Launch Gradio visualization dashboard |
| `--dashboard-port` | 7860    | Port for the Gradio dashboard   |
| `--task-store-dir` | ~/.simulator_inference_center/custom_tasks | Custom task config directory |

**Dashboard tabs:**

### Monitor Tab
- **Server Status** -- backend name, bind address, uptime, total requests, active session count
- **Active Sessions** -- table with session ID, loaded task, step count, state, idle time
- **Simulation View** -- live rendered `agentview_image` for up to 4 concurrent sessions
- **Server Logs** -- recent log lines

### Task Builder Tab

Create custom tasks for both LIBERO and robosuite simulators without writing code:

**LIBERO Task Builder:**
- Configure task name, language description, workspace
- Select fixtures and objects from dropdown menus
- Define goal states using LIBERO predicates (On, In, Open, Close, TurnOn, TurnOff, Up)
- Generates BDDL files via LIBERO's programmatic task generation pipeline
- Tasks appear in `list_tasks()` and can be loaded by connected clients

**Robosuite Task Builder:**
- Configure task name, base environment, robot, controller
- Set horizon, reward type (sparse/dense), and camera selection
- Saves parameterized config that wraps `robosuite.make()` at runtime
- Custom tasks appear in `list_tasks()` with a `custom:` prefix

**Saved Tasks:**
- View all persisted LIBERO and robosuite custom tasks
- Delete individual tasks

The dashboard runs in a background thread and auto-refreshes every 2 seconds. It does not affect server performance.

## Project Structure

```
src/simulator_inference_center/
    __init__.py             # Package init, __version__
    cli.py                  # CLI entrypoint (sim-server command)
    server.py               # InferenceServer: ZMQ ROUTER socket, poll loop, dispatch
    session.py              # Session dataclass (per-client state, optional backend + simulator_name)
    backend.py              # SimulatorBackend ABC
    config.py               # ServerConfig + LiberoBackendConfig + RobosuiteBackendConfig
    protocol.py             # msgpack pack/unpack, ndarray encoding
    monitor.py              # ServerMonitor: thread-safe state store + task-created callbacks
    dashboard.py            # Gradio Blocks UI: Monitor tab + Task Builder tab
    task_store.py           # Thread-safe JSON persistence for custom task configs
    task_generator.py       # LIBERO BDDL generation, robosuite validation, UI constants
    task_builder_ui.py      # Gradio UI components for the Task Builder tab
    client/
        __init__.py         # SimulatorClient re-export
        client.py           # SimulatorClient: ZMQ DEALER client with context manager
    backends/
        __init__.py         # Backend registry
        libero.py           # LiberoBackend (supports custom tasks)
        robosuite.py        # RobosuiteBackend (supports custom tasks)
client/
    example.py              # Full lifecycle example script
    example_robosuite.py    # Robosuite-specific example
scripts/
    run_server.py           # Legacy CLI wrapper (delegates to cli.py)
tests/
    test_server.py          # Server dispatch + ZMQ integration tests (mock backend)
    test_libero_backend.py  # Libero integration tests (requires GPU)
    test_robosuite_backend.py # Robosuite integration tests
    test_dashboard.py       # Monitor + dashboard + integration tests
    test_task_builder.py    # TaskStore, task generator, and Task Builder tests
```

## Adding a New Backend

1. Create `src/simulator_inference_center/backends/my_backend.py`
2. Subclass `SimulatorBackend` and implement all 6 methods (`list_tasks`, `load_task`, `reset`, `step`, `get_info`, `close`)
3. Optionally accept `task_store: TaskStore | None = None` in `__init__` for custom task support
4. Call `register_backend("my_backend", MyBackendClass)` at module level
5. Add the import in `backends/__init__.py` via `_discover_backends()`
6. Clients can then select it via `select_simulator("my_backend")`

## Testing

```bash
# Unit + integration tests (mock backend, no GPU needed)
pytest tests/test_server.py -v

# Libero integration tests (requires GPU + libero install, auto-skipped otherwise)
pytest tests/test_libero_backend.py -v

# Robosuite integration tests (requires robosuite install, auto-skipped otherwise)
pytest tests/test_robosuite_backend.py -v

# Dashboard + monitor tests
pytest tests/test_dashboard.py -v

# Task builder tests (TaskStore, generators, dashboard integration)
pytest tests/test_task_builder.py -v
```

## License

TBD
