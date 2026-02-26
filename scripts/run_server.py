#!/usr/bin/env python3
"""CLI entrypoint for the Simulator Inference Center server.

Prefer using the installed entrypoint: sim-server
This script delegates to simulator_inference_center.cli:main().
"""

from simulator_inference_center.cli import main

if __name__ == "__main__":
    main()
