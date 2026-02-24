"""Gradio dashboard for real-time server monitoring and task building."""

from __future__ import annotations

import threading
from typing import Any, TYPE_CHECKING

import gradio as gr
import numpy as np

if TYPE_CHECKING:
    from simulator_inference_center.monitor import ServerMonitor
    from simulator_inference_center.task_store import TaskStore


def _format_uptime(seconds: float) -> str:
    """Format seconds into human-readable HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def create_dashboard(
    monitor: ServerMonitor,
    task_store: TaskStore | None = None,
) -> gr.Blocks:
    """Build the Gradio Blocks dashboard.

    When *task_store* is provided, a "Task Builder" tab is added alongside the
    existing "Monitor" tab.
    """

    with gr.Blocks(title="Simulator Inference Center", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Simulator Inference Center — Dashboard")

        with gr.Tabs():
            # ============================================================
            # Monitor tab (existing functionality)
            # ============================================================
            with gr.Tab("Monitor"):
                # --- Server Status ---
                with gr.Row():
                    backend_box = gr.Textbox(label="Backend", interactive=False)
                    address_box = gr.Textbox(label="Bind Address", interactive=False)
                    uptime_box = gr.Textbox(label="Uptime", interactive=False)
                    requests_box = gr.Number(label="Total Requests", interactive=False, precision=0)
                    sessions_box = gr.Number(label="Active Sessions", interactive=False, precision=0)

                # --- Sessions Table ---
                gr.Markdown("## Active Sessions")
                sessions_table = gr.Dataframe(
                    headers=["Session ID", "Task", "Steps", "State", "Idle (s)"],
                    datatype=["str", "str", "number", "str", "number"],
                    interactive=False,
                )

                # --- Simulation Images ---
                gr.Markdown("## Simulation View")
                no_sim_text = gr.Markdown("*No active simulation images*", visible=True)
                with gr.Row():
                    # Pre-create up to 4 image slots for sessions
                    image_slots = []
                    for i in range(4):
                        with gr.Column(visible=False) as col:
                            label = gr.Markdown(f"### Session {i + 1}")
                            img = gr.Image(label="Agent View", type="numpy", interactive=False)
                            image_slots.append((col, label, img))

                # --- Server Logs ---
                gr.Markdown("## Server Logs")
                log_box = gr.Textbox(
                    label="Recent Logs",
                    lines=10,
                    max_lines=15,
                    interactive=False,
                )

            # ============================================================
            # Task Builder tab (only when task_store is provided)
            # ============================================================
            if task_store is not None:
                with gr.Tab("Task Builder"):
                    from simulator_inference_center.task_builder_ui import (
                        create_task_builder_tab,
                    )
                    create_task_builder_tab(task_store, monitor)

        # --- Refresh logic (bound to Monitor tab outputs) ---
        timer = gr.Timer(2)

        def refresh():
            snap = monitor.get_snapshot()

            # Server status
            backend = snap["backend_name"] or "(not started)"
            address = snap["bind_address"] or "(not bound)"
            uptime = _format_uptime(snap["uptime_s"])
            total_req = snap["total_requests"]
            num_sessions = len(snap["sessions"])

            # Sessions table
            rows = []
            for sid, info in snap["sessions"].items():
                rows.append([
                    info["session_id"],
                    info["task"],
                    info["steps"],
                    info["state"],
                    round(info["idle_s"], 1),
                ])

            # Images -- find sessions with images
            sessions_with_images = []
            for sid, imgs in snap["latest_images"].items():
                if "agentview_image" in imgs:
                    session_info = snap["sessions"].get(sid, {})
                    label_text = f"### {session_info.get('task', sid[:12])}"
                    sessions_with_images.append((label_text, imgs["agentview_image"]))

            show_no_sim = len(sessions_with_images) == 0

            # Build updates for image slots
            slot_updates = []
            for i in range(4):
                if i < len(sessions_with_images):
                    label_text, img_arr = sessions_with_images[i]
                    slot_updates.extend([
                        gr.update(visible=True),      # column
                        gr.update(value=label_text),   # label markdown
                        gr.update(value=img_arr),      # image
                    ])
                else:
                    slot_updates.extend([
                        gr.update(visible=False),
                        gr.update(value=""),
                        gr.update(value=None),
                    ])

            # Logs
            logs_text = "\n".join(snap["logs"][-50:]) if snap["logs"] else "(no logs)"

            return [
                backend, address, uptime, total_req, num_sessions,
                rows,
                gr.update(visible=show_no_sim),
                *slot_updates,
                logs_text,
            ]

        all_outputs = [
            backend_box, address_box, uptime_box, requests_box, sessions_box,
            sessions_table,
            no_sim_text,
        ]
        for col, label, img in image_slots:
            all_outputs.extend([col, label, img])
        all_outputs.append(log_box)

        timer.tick(fn=refresh, outputs=all_outputs)

    return app


def launch_dashboard(
    monitor: ServerMonitor,
    port: int = 7860,
    share: bool = False,
    task_store: TaskStore | None = None,
) -> threading.Thread:
    """Launch the Gradio dashboard in a background daemon thread."""
    app = create_dashboard(monitor, task_store=task_store)

    def _run() -> None:
        app.launch(
            server_port=port,
            share=share,
            prevent_thread_lock=True,
            quiet=True,
        )

    thread = threading.Thread(target=_run, daemon=True, name="gradio-dashboard")
    thread.start()
    return thread
