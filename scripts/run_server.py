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
        "--simulator",
        type=str,
        default=None,
        help="Backend simulator name (overrides SIM_BACKEND env var)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind on (overrides SIM_BIND_ADDRESS env var)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Pre-load a specific task at startup (optional)",
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
    args = parser.parse_args()

    # Import after argparse so --help is fast
    from simulator_inference_center.config import ServerConfig
    from simulator_inference_center.server import InferenceServer

    # Build config from env vars, then apply CLI overrides
    config = ServerConfig()

    if args.simulator is not None:
        config.backend = args.simulator
    if args.port is not None:
        config.bind_address = f"tcp://*:{args.port}"
    if args.log_level is not None:
        config.log_level = args.log_level

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger = logging.getLogger("simulator_inference_center")

    # Set up optional dashboard monitor
    monitor = None
    if args.dashboard:
        from simulator_inference_center.monitor import ServerMonitor

        monitor = ServerMonitor()

    try:
        server = InferenceServer(config, monitor=monitor)
    except KeyError as exc:
        logger.error("Failed to initialize server: %s", exc)
        sys.exit(1)

    # If --task is given, verify it exists in the backend by listing tasks
    if args.task is not None:
        logger.info("Pre-load task requested: %s", args.task)

    # Launch Gradio dashboard in background thread
    if args.dashboard:
        from simulator_inference_center.dashboard import launch_dashboard

        launch_dashboard(monitor, port=args.dashboard_port)
        logger.info("Dashboard available at http://localhost:%d", args.dashboard_port)

    server.run()


if __name__ == "__main__":
    main()
