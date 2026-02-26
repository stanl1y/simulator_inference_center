#!/usr/bin/env python3
"""Robosuite example: connect to a robosuite backend, run a short episode on the Lift task.

Start the server first:
    sim-server --port 5555

Then run this example:
    python client/example_robosuite.py [--address tcp://localhost:5555] [--task Lift] [--steps 50]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from simulator_inference_center.client import SimulatorClient


def _print_observation_summary(obs: dict, label: str = "Observation") -> None:
    """Print a summary of observation keys, shapes, and dtypes."""
    print(f"\n  {label}:")
    for key, value in sorted(obs.items()):
        if isinstance(value, np.ndarray):
            print(f"    {key}: shape={value.shape} dtype={value.dtype}")
        else:
            print(f"    {key}: {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Robosuite client example")
    parser.add_argument(
        "--address",
        type=str,
        default="tcp://localhost:5555",
        help="Server address (default: tcp://localhost:5555)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Lift",
        help="Robosuite task name (default: Lift)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of steps to run (default: 50)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="Number of episodes to run (default: 2)",
    )
    args = parser.parse_args()

    print(f"Connecting to server at {args.address} ...")
    with SimulatorClient(args.address) as client:
        # -- Server info --
        info = client.get_info()
        print(f"Backend: {info['backend_name']} v{info.get('backend_version', '?')}")

        # -- List available tasks --
        tasks = client.list_tasks()
        print(f"Available tasks ({len(tasks)}): {', '.join(tasks)}")

        if args.task not in tasks:
            print(f"Error: task {args.task!r} not found. Pick one from the list above.")
            sys.exit(1)

        # -- Load task --
        print(f"\nLoading task: {args.task}")
        task_info = client.load_task(args.task)
        action_space = task_info["action_space"]
        action_dim = action_space["shape"][0]
        print(f"  Action dim: {action_dim}")
        print(f"  Action range: [{action_space['low'][0]:.1f}, {action_space['high'][0]:.1f}]")
        print(f"  Max episode steps: {task_info.get('max_episode_steps', 'N/A')}")

        # -- Run episodes --
        for ep in range(args.episodes):
            print(f"\n{'='*60}")
            print(f"Episode {ep + 1}/{args.episodes}")
            print(f"{'='*60}")

            obs = client.reset()
            _print_observation_summary(obs, label="Initial observation")

            total_reward = 0.0
            for step_i in range(args.steps):
                # Random action in [-1, 1]
                action = np.random.uniform(-1.0, 1.0, size=action_dim).astype(
                    np.float64
                )
                result = client.step({"action": action})

                obs = result["observation"]
                reward = result["reward"]
                terminated = result["terminated"]
                truncated = result["truncated"]
                total_reward += reward

                # Print every 10th step and the last step
                if (step_i + 1) % 10 == 0 or terminated or truncated:
                    print(
                        f"  Step {step_i + 1:4d}: reward={reward:.4f}  "
                        f"cumulative={total_reward:.4f}  "
                        f"terminated={terminated}  truncated={truncated}"
                    )

                if terminated or truncated:
                    reason = "success" if terminated else "horizon reached"
                    print(f"  Episode ended at step {step_i + 1} ({reason}).")
                    break

            _print_observation_summary(obs, label="Final observation")
            print(f"  Total reward: {total_reward:.4f}")

        print("\nDone.")


if __name__ == "__main__":
    main()
