"""ZMQ ROUTER server with message dispatch loop."""

from __future__ import annotations

import logging
import signal
import time
from typing import TYPE_CHECKING, Any

import zmq

from simulator_inference_center.backend import SimulatorBackend
from simulator_inference_center.backends import get_backend_class, list_backends
from simulator_inference_center.config import ServerConfig
from simulator_inference_center.protocol import pack, unpack
from simulator_inference_center.session import Session

if TYPE_CHECKING:
    from simulator_inference_center.monitor import ServerMonitor
    from simulator_inference_center.task_store import TaskStore

logger = logging.getLogger(__name__)


class InferenceServer:
    """ZMQ ROUTER-based inference server for simulator backends.

    Clients must call ``select_simulator`` to choose a backend before using
    any simulator methods (list_tasks, load_task, reset, step, get_info).
    """

    def __init__(
        self,
        config: ServerConfig,
        monitor: ServerMonitor | None = None,
        task_store: TaskStore | None = None,
    ) -> None:
        self.config = config
        self._running = False
        self._sessions: dict[bytes, Session] = {}
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._monitor = monitor
        self._task_store = task_store

    def _create_backend(self, backend_name: str) -> SimulatorBackend:
        """Instantiate a fresh backend by name."""
        cls = get_backend_class(backend_name)
        if self._task_store is not None:
            try:
                return cls(task_store=self._task_store)
            except TypeError:
                # Backend doesn't accept task_store kwarg — fall back.
                pass
        return cls()

    def _make_error(
        self, error_type: str, message: str
    ) -> dict[str, Any]:
        return {
            "status": "error",
            "error_type": error_type,
            "message": message,
        }

    def _get_or_create_session(self, identity: bytes) -> Session:
        if identity not in self._sessions:
            logger.info("New session for client %s", identity.hex())
            self._sessions[identity] = Session(identity=identity)
            if self._monitor is not None:
                try:
                    self._monitor.on_session_created(identity, self._sessions[identity])
                except Exception:
                    logger.debug("Monitor update failed", exc_info=True)
        session = self._sessions[identity]
        session.touch()
        return session

    def _require_backend(self, session: Session, method: str) -> dict[str, Any] | None:
        """Return an error response if the session has no backend, else None."""
        if session.backend is None:
            return self._make_error(
                "no_simulator_selected",
                f"Call select_simulator before {method}",
            )
        return None

    def _remove_session(self, identity: bytes) -> None:
        session = self._sessions.pop(identity, None)
        if session is not None:
            logger.info("Removing session for client %s", identity.hex())
            if session.backend is not None:
                try:
                    session.backend.close()
                except Exception:
                    logger.exception("Error closing backend for %s", identity.hex())
        if self._monitor is not None:
            try:
                self._monitor.on_session_removed(identity)
            except Exception:
                logger.debug("Monitor update failed", exc_info=True)

    def _reap_stale_sessions(self) -> None:
        now = time.time()
        stale = [
            ident
            for ident, sess in self._sessions.items()
            if (now - sess.last_active) > self.config.session_timeout_s
        ]
        for ident in stale:
            logger.warning(
                "Reaping stale session %s (idle %.0fs)",
                ident.hex(),
                now - self._sessions[ident].last_active,
            )
            self._remove_session(ident)

    def handle_request(
        self, identity: bytes, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Dispatch a request to the appropriate handler."""
        method = request.get("method")
        if method is None:
            return self._make_error(
                "invalid_params", "Missing 'method' field in request"
            )

        handler = {
            "list_simulators": self._handle_list_simulators,
            "select_simulator": self._handle_select_simulator,
            "list_tasks": self._handle_list_tasks,
            "load_task": self._handle_load_task,
            "reset": self._handle_reset,
            "step": self._handle_step,
            "get_info": self._handle_get_info,
            "disconnect": self._handle_disconnect,
        }.get(method)

        if handler is None:
            return self._make_error(
                "unknown_method", f"Unknown method: {method!r}"
            )

        try:
            response = handler(identity, request)
        except Exception:
            logger.exception(
                "Unhandled error processing %s for %s", method, identity.hex()
            )
            response = self._make_error(
                "internal_error", "An internal server error occurred"
            )

        # Notify monitor
        if self._monitor is not None:
            session = self._sessions.get(identity)
            try:
                self._monitor.on_request(identity, method, request, response, session)
            except Exception:
                logger.debug("Monitor update failed", exc_info=True)

        return response

    # ------------------------------------------------------------------
    # Handlers: list_simulators / select_simulator
    # ------------------------------------------------------------------

    def _handle_list_simulators(
        self, identity: bytes, request: dict[str, Any]
    ) -> dict[str, Any]:
        return {"status": "ok", "simulators": list_backends()}

    def _handle_select_simulator(
        self, identity: bytes, request: dict[str, Any]
    ) -> dict[str, Any]:
        simulator = request.get("simulator")
        if simulator is None:
            return self._make_error(
                "invalid_params", "Missing 'simulator' in select_simulator request"
            )

        session = self._get_or_create_session(identity)

        if session.backend is not None:
            return self._make_error(
                "simulator_already_selected",
                f"Simulator already selected: {session.simulator_name!r}. "
                "Disconnect first to choose a different simulator.",
            )

        try:
            backend = self._create_backend(simulator)
        except KeyError:
            available = ", ".join(list_backends()) or "(none)"
            return self._make_error(
                "unknown_simulator",
                f"Unknown simulator {simulator!r}. Available: {available}",
            )
        except Exception as exc:
            logger.exception("Failed to create backend %r", simulator)
            return self._make_error(
                "backend_error", f"Failed to create simulator: {exc}"
            )

        session.backend = backend
        session.simulator_name = simulator
        logger.info(
            "Client %s selected simulator %r", identity.hex(), simulator
        )

        # Update monitor snapshot
        if self._monitor is not None:
            try:
                self._monitor.on_session_created(identity, session)
            except Exception:
                logger.debug("Monitor update failed", exc_info=True)

        return {"status": "ok", "simulator": simulator}

    # ------------------------------------------------------------------
    # Simulator handlers (require select_simulator first)
    # ------------------------------------------------------------------

    def _handle_list_tasks(
        self, identity: bytes, request: dict[str, Any]
    ) -> dict[str, Any]:
        session = self._get_or_create_session(identity)
        err = self._require_backend(session, "list_tasks")
        if err is not None:
            return err
        try:
            tasks = session.backend.list_tasks()
        except Exception as exc:
            logger.exception("backend.list_tasks() failed")
            return self._make_error("backend_error", str(exc))
        return {"status": "ok", "tasks": tasks}

    def _handle_load_task(
        self, identity: bytes, request: dict[str, Any]
    ) -> dict[str, Any]:
        task_name = request.get("task_name")
        if task_name is None:
            return self._make_error(
                "invalid_params", "Missing 'task_name' in load_task request"
            )

        session = self._get_or_create_session(identity)
        err = self._require_backend(session, "load_task")
        if err is not None:
            return err
        try:
            task_info = session.backend.load_task(task_name)
        except ValueError as exc:
            return self._make_error("task_not_found", str(exc))
        except Exception as exc:
            logger.exception("backend.load_task() failed")
            return self._make_error("backend_error", str(exc))

        session.task_loaded = True
        session.needs_reset = True
        session.steps = 0
        return {"status": "ok", "task_info": task_info}

    def _handle_reset(
        self, identity: bytes, request: dict[str, Any]
    ) -> dict[str, Any]:
        session = self._get_or_create_session(identity)
        err = self._require_backend(session, "reset")
        if err is not None:
            return err
        if not session.task_loaded:
            return self._make_error(
                "no_task_loaded", "Call load_task before reset"
            )

        try:
            observation = session.backend.reset()
        except Exception as exc:
            logger.exception("backend.reset() failed")
            return self._make_error("backend_error", str(exc))

        session.needs_reset = False
        session.steps = 0
        return {"status": "ok", "observation": observation}

    def _handle_step(
        self, identity: bytes, request: dict[str, Any]
    ) -> dict[str, Any]:
        session = self._get_or_create_session(identity)
        err = self._require_backend(session, "step")
        if err is not None:
            return err
        if not session.task_loaded:
            return self._make_error(
                "no_task_loaded", "Call load_task before step"
            )
        if session.needs_reset:
            return self._make_error(
                "not_reset", "Call reset before step"
            )

        action = request.get("action")
        if action is None:
            return self._make_error(
                "invalid_params", "Missing 'action' in step request"
            )

        try:
            result = session.backend.step(action)
        except Exception as exc:
            logger.exception("backend.step() failed")
            return self._make_error("backend_error", str(exc))

        session.steps += 1

        # If episode ended, require reset before next step
        if result.get("terminated") or result.get("truncated"):
            session.needs_reset = True

        return {
            "status": "ok",
            "observation": result["observation"],
            "reward": result["reward"],
            "terminated": result["terminated"],
            "truncated": result["truncated"],
            "info": result.get("info", {}),
        }

    def _handle_get_info(
        self, identity: bytes, request: dict[str, Any]
    ) -> dict[str, Any]:
        session = self._get_or_create_session(identity)
        err = self._require_backend(session, "get_info")
        if err is not None:
            return err
        try:
            info = session.backend.get_info()
        except Exception as exc:
            logger.exception("backend.get_info() failed")
            return self._make_error("backend_error", str(exc))
        return {"status": "ok", **info}

    def _handle_disconnect(
        self, identity: bytes, request: dict[str, Any]
    ) -> dict[str, Any]:
        self._remove_session(identity)
        return {"status": "ok"}

    def run(self) -> None:
        """Start the server poll loop. Blocks until SIGINT or shutdown."""
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.ROUTER)
        self._socket.bind(self.config.bind_address)
        self._running = True

        if self._monitor is not None:
            self._monitor.set_server_info(
                bind_address=self.config.bind_address,
                available_backends=list_backends(),
            )

        # Handle SIGINT for graceful shutdown
        original_sigint = signal.getsignal(signal.SIGINT)

        def _signal_handler(signum, frame):
            logger.info("Received signal %d, shutting down...", signum)
            self._running = False

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)

        logger.info(
            "Server started on %s (available backends: %s)",
            self.config.bind_address,
            ", ".join(list_backends()) or "(none)",
        )

        try:
            while self._running:
                events = dict(poller.poll(timeout=1000))
                if self._socket in events:
                    frames = self._socket.recv_multipart()
                    if len(frames) < 3:
                        logger.warning("Malformed message: %d frames", len(frames))
                        continue

                    identity, _, body = frames[0], frames[1], frames[2]

                    try:
                        request = unpack(body)
                    except Exception:
                        logger.warning(
                            "Failed to decode msgpack from %s", identity.hex()
                        )
                        response = self._make_error(
                            "invalid_params", "Failed to decode msgpack body"
                        )
                        self._socket.send_multipart(
                            [identity, b"", pack(response)]
                        )
                        continue

                    logger.debug(
                        "Request from %s: method=%s",
                        identity.hex(),
                        request.get("method"),
                    )

                    response = self.handle_request(identity, request)
                    self._socket.send_multipart(
                        [identity, b"", pack(response)]
                    )

                # Periodically reap stale sessions
                self._reap_stale_sessions()
        finally:
            self._shutdown()
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)

    def _shutdown(self) -> None:
        """Clean up all sessions and ZMQ resources."""
        logger.info("Shutting down server...")
        for ident in list(self._sessions):
            self._remove_session(ident)

        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        logger.info("Server shutdown complete.")
