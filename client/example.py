#!/usr/bin/env python3
"""Full lifecycle example: connect -> list_tasks -> load_task -> reset -> step loop -> disconnect.

Usage:
    python -m client.example [--address tcp://localhost:5555]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from client.client import SimulatorClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulator client example")
    parser.add_argument(
        "--address",
        type=str,
        default="tcp://localhost:5555",
        help="Server address (default: tcp://localhost:5555)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of steps to run (default: 10)",
    )
    args = parser.parse_args()

    print(f"Connecting to server at {args.address} ...")
    with SimulatorClient(args.address) as client:
        # List available tasks
        tasks = client.list_tasks()
        print(f"Available tasks ({len(tasks)}):")
        for i, name in enumerate(tasks):
            print(f"  [{i}] {name}")

        if not tasks:
            print("No tasks available. Exiting.")
            sys.exit(0)

        # Load the first task
        task_name = tasks[0]
        print(f"\nLoading task: {task_name}")
        task_info = client.load_task(task_name)
        print(f"  Description: {task_info.get('description', 'N/A')}")
        print(f"  Action space: {task_info.get('action_space', {})}")
        print(f"  Max episode steps: {task_info.get('max_episode_steps', 'N/A')}")

        # Reset the environment
        print("\nResetting environment ...")
        obs = client.reset()
        _print_observation_summary(obs, label="Initial observation")

        # Step loop with random actions
        action_space = task_info.get("action_space", {})
        action_dim = action_space.get("shape", [7])[0] if action_space else 7

        print(f"\nRunning {args.steps} steps with random actions (dim={action_dim}) ...")
        total_reward = 0.0
        for step_i in range(args.steps):
            # Generate a random action as a flat array
            action_arr = np.random.uniform(-1.0, 1.0, size=action_dim).astype(np.float64)
            result = client.step({"action": action_arr})

            obs = result["observation"]
            reward = result["reward"]
            terminated = result["terminated"]
            truncated = result["truncated"]
            total_reward += reward

            print(
                f"  Step {step_i + 1:3d}: reward={reward:.4f}  "
                f"terminated={terminated}  truncated={truncated}"
            )

            if terminated or truncated:
                print(f"  Episode ended at step {step_i + 1}.")
                break

        _print_observation_summary(obs, label="Final observation")
        print(f"\nTotal reward: {total_reward:.4f}")

        # Disconnect
        print("\nDisconnecting ...")

    print("Done.")


def _print_observation_summary(obs: dict, label: str = "Observation") -> None:
    """Print a summary of observation keys, shapes, and dtypes."""
    print(f"\n  {label}:")
    for key, value in sorted(obs.items()):
        if isinstance(value, np.ndarray):
            print(f"    {key}: shape={value.shape} dtype={value.dtype}")
        else:
            print(f"    {key}: {type(value).__name__} = {value}")


if __name__ == "__main__":
    main()
