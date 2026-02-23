# Simulator Inference Center

ZMQ-based inference server that exposes robot simulators (e.g. [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)) over a network protocol for remote policy evaluation.

## Overview

Simulator Inference Center decouples policy code from simulator code by running the simulator behind a lightweight ZMQ server. A remote client connects, lists available tasks, loads one, and runs step loops — all over TCP with msgpack serialization. This lets you evaluate policies from any machine without needing the simulator installed locally.

**Key features:**

- **ROUTER/DEALER transport** over TCP with msgpack + ndarray binary encoding
- **Session isolation** — each client gets its own simulator backend instance
- **Pluggable backends** — ships with LIBERO; add new simulators by subclassing `SimulatorBackend`
- **Configurable** via environment variables (`SIM_` prefix) or constructor args
- **Automatic session reaping** for idle clients

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

### 2. Install Simulator Inference Center

```bash
cd /path/to/simulator_inference_center
pip install -e .
```

## Quick Start

**Start the server:**

```bash
# Libero backend on default port
python scripts/run_server.py --simulator libero --port 5555

# With environment variables
SIM_BACKEND=libero SIM_BIND_ADDRESS=tcp://*:5555 python scripts/run_server.py
```

**Connect a client (Python):**

```python
from client.client import SimulatorClient

with SimulatorClient("tcp://localhost:5555") as client:
    tasks = client.list_tasks()
    client.load_task(tasks[0])
    obs = client.reset()

    for _ in range(100):
        result = client.step({"action": my_policy(obs)})
        obs = result["observation"]
        if result["terminated"] or result["truncated"]:
            break
```

**Run the example script:**

```bash
python -m client.example --address tcp://localhost:5555 --steps 50
```

## Protocol

All messages use msgpack over ZMQ with the frame format `[identity, empty delimiter, msgpack body]`.

| Method       | Key Request Fields  | Key Response Fields                                |
|--------------|---------------------|----------------------------------------------------|
| list_tasks   | —                   | tasks: list[str]                                   |
| load_task    | task_name: str      | task_info: dict                                    |
| reset        | —                   | observation: dict                                  |
| step         | action: dict        | observation, reward, terminated, truncated, info   |
| get_info     | —                   | backend_name, backend_version, current_task, ...   |
| disconnect   | —                   | (status only)                                      |

Numpy arrays are encoded as `{"__type__": "ndarray", "shape": [...], "dtype": "...", "data": <bytes>}`.

Errors return `{"status": "error", "error_type": "...", "message": "..."}` with types: `unknown_method`, `invalid_params`, `task_not_found`, `no_task_loaded`, `not_reset`, `backend_error`, `internal_error`.

## Configuration

All settings can be set via environment variables (prefix `SIM_`) or passed to `ServerConfig`:

| Env Var                      | Default        | Description                     |
|------------------------------|----------------|---------------------------------|
| SIM_BIND_ADDRESS             | tcp://*:5555   | ZMQ ROUTER bind address         |
| SIM_BACKEND                  | libero         | Backend name from registry      |
| SIM_SESSION_TIMEOUT_S        | 300            | Idle session reap timeout (sec) |
| SIM_LOG_LEVEL                | INFO           | Logging level                   |
| SIM_LIBERO_TASK_SUITE        | libero_90      | Libero suite name               |
| SIM_LIBERO_RENDER_WIDTH      | 256            | Render width                    |
| SIM_LIBERO_RENDER_HEIGHT     | 256            | Render height                   |
| SIM_LIBERO_MAX_EPISODE_STEPS | 300            | Max steps per episode           |

## Project Structure

```
src/simulator_inference_center/
    server.py               # InferenceServer: ZMQ ROUTER socket, poll loop, dispatch
    session.py              # Session dataclass (per-client state)
    backend.py              # SimulatorBackend ABC
    config.py               # ServerConfig + LiberoBackendConfig (pydantic-settings)
    protocol.py             # msgpack pack/unpack, ndarray encoding
    backends/
        __init__.py         # Backend registry
        libero.py           # LiberoBackend implementation
client/
    client.py               # SimulatorClient: ZMQ DEALER client with context manager
    example.py              # Full lifecycle example script
scripts/
    run_server.py           # CLI entrypoint
tests/
    test_server.py          # Server dispatch + ZMQ integration tests (mock backend)
    test_libero_backend.py  # Libero integration tests (requires GPU)
```

## Adding a New Backend

1. Create `src/simulator_inference_center/backends/my_backend.py`
2. Subclass `SimulatorBackend` and implement all 6 methods (`list_tasks`, `load_task`, `reset`, `step`, `get_info`, `close`)
3. Call `register_backend("my_backend", MyBackendClass)` at module level
4. Add the import in `backends/__init__.py` via `_discover_backends()`
5. Run with `--simulator my_backend`

## Testing

```bash
# Unit + integration tests (mock backend, no GPU needed)
pytest tests/test_server.py -v

# Libero integration tests (requires GPU + libero install, auto-skipped otherwise)
pytest tests/test_libero_backend.py -v
```

## License

TBD
