#!/usr/bin/env python3
"""CLI entrypoint for the Simulator Inference Center server."""

import argparse
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulator Inference Center -- ZMQ inference server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind on (overrides SIM_BIND_ADDRESS env var)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        default=False,
        help="Launch Gradio visualization dashboard",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=7860,
        help="Port for the Gradio dashboard (default: 7860)",
    )
    parser.add_argument(
        "--task-store-dir",
        type=str,
        default="~/.simulator_inference_center/custom_tasks",
        help="Directory for custom task configs (default: ~/.simulator_inference_center/custom_tasks)",
    )
    args = parser.parse_args()

    # Import after argparse so --help is fast
    from simulator_inference_center.backends import list_backends
    from simulator_inference_center.config import ServerConfig
    from simulator_inference_center.server import InferenceServer

    # Build config from env vars, then apply CLI overrides
    config = ServerConfig()

    if args.port is not None:
        config.bind_address = f"tcp://*:{args.port}"
    if args.log_level is not None:
        config.log_level = args.log_level
    if args.task_store_dir is not None:
        config.task_store_dir = args.task_store_dir

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger = logging.getLogger("simulator_inference_center")

    available = list_backends()
    logger.info(
        "Available backends: %s. Clients select via select_simulator.",
        ", ".join(available) or "(none)",
    )

    # Set up optional dashboard monitor
    monitor = None
    if args.dashboard:
        from simulator_inference_center.monitor import ServerMonitor

        monitor = ServerMonitor()

    # Create task store for custom task persistence
    from simulator_inference_center.task_store import TaskStore

    task_store = TaskStore(config.task_store_dir)

    server = InferenceServer(config, monitor=monitor, task_store=task_store)

    # Launch Gradio dashboard in background thread
    if args.dashboard:
        from simulator_inference_center.dashboard import launch_dashboard

        launch_dashboard(monitor, port=args.dashboard_port, task_store=task_store)
        logger.info("Dashboard available at http://localhost:%d", args.dashboard_port)

    server.run()


if __name__ == "__main__":
    main()
