"""Gradio UI components for the Task Builder tab."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import gradio as gr

from simulator_inference_center.task_generator import (
    LIBERO_FIXTURES,
    LIBERO_OBJECTS,
    LIBERO_PREDICATES,
    LIBERO_WORKSPACES,
    ROBOSUITE_BASE_ENVS,
    ROBOSUITE_CAMERAS,
    ROBOSUITE_CONTROLLERS,
    ROBOSUITE_ROBOTS,
    LiberoTaskGenerator,
    RobosuiteTaskGenerator,
)
from simulator_inference_center.task_store import (
    LiberoTaskConfig,
    RobosuiteTaskConfig,
)

if TYPE_CHECKING:
    from simulator_inference_center.monitor import ServerMonitor
    from simulator_inference_center.task_store import TaskStore

logger = logging.getLogger(__name__)


def create_task_builder_tab(
    task_store: TaskStore,
    monitor: ServerMonitor | None = None,
) -> None:
    """Build the Task Builder tab contents inside an existing gr.Blocks context.

    Must be called within a ``gr.Blocks`` or ``gr.Tab`` context manager.
    """
    with gr.Tabs():
        with gr.Tab("LIBERO Task Builder"):
            _build_libero_tab(task_store, monitor)
        with gr.Tab("Robosuite Task Builder"):
            _build_robosuite_tab(task_store, monitor)
        with gr.Tab("Saved Tasks"):
            _build_saved_tasks_tab(task_store)


# ---------------------------------------------------------------------------
# LIBERO Task Builder
# ---------------------------------------------------------------------------

def _build_libero_tab(
    task_store: TaskStore,
    monitor: ServerMonitor | None,
) -> None:
    gr.Markdown("### Create a custom LIBERO task")

    with gr.Row():
        with gr.Column(scale=2):
            libero_name = gr.Textbox(label="Task Name", placeholder="my_custom_task")
            libero_language = gr.Textbox(
                label="Language Description",
                placeholder="Pick up the red bowl and place it on the plate",
            )
            libero_workspace = gr.Dropdown(
                label="Workspace",
                choices=LIBERO_WORKSPACES,
                value=LIBERO_WORKSPACES[0],
            )

        with gr.Column(scale=2):
            libero_fixtures = gr.Dropdown(
                label="Fixtures",
                choices=LIBERO_FIXTURES,
                multiselect=True,
                value=[],
            )
            libero_objects = gr.Dropdown(
                label="Objects",
                choices=LIBERO_OBJECTS,
                multiselect=True,
                value=[],
            )
            libero_objects_of_interest = gr.Textbox(
                label="Objects of Interest (comma-separated)",
                placeholder="akita_black_bowl, plate",
            )

    gr.Markdown(
        "**Goal States** — JSON array of `[predicate, arg1, ...]`. "
        f"Predicates: {', '.join(LIBERO_PREDICATES.keys())}"
    )
    libero_goal_states = gr.Code(
        label="Goal States (JSON)",
        language="json",
        value='[["On", "akita_black_bowl", "plate"]]',
        lines=4,
    )

    gr.Markdown(
        "**Regions** (optional) — JSON array of `{\"name\": ..., \"count\": ...}`"
    )
    libero_regions = gr.Code(
        label="Regions (JSON, optional)",
        language="json",
        value="[]",
        lines=3,
    )

    with gr.Row():
        libero_validate_btn = gr.Button("Validate", variant="secondary")
        libero_generate_btn = gr.Button("Generate & Save", variant="primary")

    libero_status = gr.Textbox(label="Status", interactive=False, lines=3)
    libero_bddl_preview = gr.Code(
        label="Generated BDDL Preview",
        language=None,
        lines=10,
        interactive=False,
        visible=False,
    )

    def _parse_libero_config(
        name, language, workspace, fixtures, objects, objects_of_interest,
        goal_states_json, regions_json,
    ) -> LiberoTaskConfig | str:
        """Parse UI inputs into a LiberoTaskConfig, or return error string."""
        try:
            goal_states = json.loads(goal_states_json) if goal_states_json.strip() else []
        except json.JSONDecodeError as exc:
            return f"Invalid goal states JSON: {exc}"

        try:
            regions = json.loads(regions_json) if regions_json.strip() else []
        except json.JSONDecodeError as exc:
            return f"Invalid regions JSON: {exc}"

        # Build fixture/object count dicts (default count=1 each)
        fixture_dict = {f: 1 for f in (fixtures or [])}
        object_dict = {o: 1 for o in (objects or [])}

        ooi = [
            s.strip() for s in (objects_of_interest or "").split(",") if s.strip()
        ]

        return LiberoTaskConfig(
            task_name=name or "",
            language=language or "",
            workspace=workspace or "",
            fixtures=fixture_dict,
            objects=object_dict,
            regions=regions,
            objects_of_interest=ooi,
            goal_states=goal_states,
        )

    def _validate_libero(
        name, language, workspace, fixtures, objects, objects_of_interest,
        goal_states_json, regions_json,
    ):
        config = _parse_libero_config(
            name, language, workspace, fixtures, objects, objects_of_interest,
            goal_states_json, regions_json,
        )
        if isinstance(config, str):
            return config, gr.update(visible=False)

        errors = LiberoTaskGenerator.validate(config)
        if errors:
            return "Validation errors:\n" + "\n".join(f"  - {e}" for e in errors), gr.update(visible=False)
        return "Validation passed!", gr.update(visible=False)

    def _generate_libero(
        name, language, workspace, fixtures, objects, objects_of_interest,
        goal_states_json, regions_json,
    ):
        config = _parse_libero_config(
            name, language, workspace, fixtures, objects, objects_of_interest,
            goal_states_json, regions_json,
        )
        if isinstance(config, str):
            return config, gr.update(visible=False)

        errors = LiberoTaskGenerator.validate(config)
        if errors:
            return "Validation errors:\n" + "\n".join(f"  - {e}" for e in errors), gr.update(visible=False)

        # Generate BDDL
        import os
        output_dir = os.path.join(str(task_store.base_dir), "libero", "bddl")
        try:
            bddl_path = LiberoTaskGenerator.generate(config, output_dir)
        except RuntimeError as exc:
            return f"Generation failed: {exc}", gr.update(visible=False)
        except Exception as exc:
            logger.exception("BDDL generation failed")
            return f"Generation error: {exc}", gr.update(visible=False)

        # Save config with bddl path
        config.bddl_file_path = bddl_path
        task_store.save_libero_task(config)

        # Notify monitor
        if monitor is not None:
            monitor.notify_task_created(
                "libero", config.task_name,
                bddl_path=bddl_path, language=config.language,
            )

        # Read BDDL for preview
        try:
            with open(bddl_path, "r") as f:
                bddl_content = f.read()
        except Exception:
            bddl_content = "(could not read generated file)"

        status = f"Task '{config.task_name}' saved successfully!\nBDDL: {bddl_path}"
        return status, gr.update(value=bddl_content, visible=True)

    libero_validate_btn.click(
        fn=_validate_libero,
        inputs=[
            libero_name, libero_language, libero_workspace,
            libero_fixtures, libero_objects, libero_objects_of_interest,
            libero_goal_states, libero_regions,
        ],
        outputs=[libero_status, libero_bddl_preview],
    )

    libero_generate_btn.click(
        fn=_generate_libero,
        inputs=[
            libero_name, libero_language, libero_workspace,
            libero_fixtures, libero_objects, libero_objects_of_interest,
            libero_goal_states, libero_regions,
        ],
        outputs=[libero_status, libero_bddl_preview],
    )


# ---------------------------------------------------------------------------
# Robosuite Task Builder
# ---------------------------------------------------------------------------

def _build_robosuite_tab(
    task_store: TaskStore,
    monitor: ServerMonitor | None,
) -> None:
    gr.Markdown("### Create a custom robosuite task configuration")

    with gr.Row():
        with gr.Column():
            rs_name = gr.Textbox(label="Task Name", placeholder="my_custom_lift")
            rs_description = gr.Textbox(
                label="Description (optional)",
                placeholder="Custom Lift task with Sawyer robot",
            )
            rs_base_env = gr.Dropdown(
                label="Base Environment",
                choices=ROBOSUITE_BASE_ENVS,
                value="Lift",
            )

        with gr.Column():
            rs_robot = gr.Dropdown(
                label="Robot",
                choices=ROBOSUITE_ROBOTS,
                value="Panda",
            )
            rs_controller = gr.Dropdown(
                label="Controller (optional)",
                choices=["(default)"] + ROBOSUITE_CONTROLLERS,
                value="(default)",
            )
            rs_horizon = gr.Slider(
                label="Horizon (max steps)",
                minimum=50,
                maximum=5000,
                step=50,
                value=1000,
            )

    with gr.Row():
        rs_reward_type = gr.Radio(
            label="Reward Type",
            choices=["sparse", "dense"],
            value="sparse",
        )
        rs_cameras = gr.CheckboxGroup(
            label="Camera Names",
            choices=ROBOSUITE_CAMERAS,
            value=["agentview", "robot0_eye_in_hand"],
        )

    rs_save_btn = gr.Button("Save Configuration", variant="primary")
    rs_status = gr.Textbox(label="Status", interactive=False, lines=2)
    rs_preview = gr.Code(
        label="Saved Config Preview",
        language="json",
        lines=10,
        interactive=False,
        visible=False,
    )

    def _save_robosuite(
        name, description, base_env, robot, controller, horizon,
        reward_type, cameras,
    ):
        ctrl = None if controller == "(default)" else controller
        config = RobosuiteTaskConfig(
            task_name=name or "",
            description=description or "",
            robot=robot,
            controller=ctrl,
            base_env=base_env,
            horizon=int(horizon),
            reward_type=reward_type,
            camera_names=cameras or [],
        )

        errors = RobosuiteTaskGenerator.validate(config)
        if errors:
            return (
                "Validation errors:\n" + "\n".join(f"  - {e}" for e in errors),
                gr.update(visible=False),
            )

        task_store.save_robosuite_task(config)

        # Notify monitor
        if monitor is not None:
            monitor.notify_task_created("robosuite", config.task_name)

        preview = json.dumps(config.to_dict(), indent=2)
        return (
            f"Task '{config.task_name}' saved successfully!",
            gr.update(value=preview, visible=True),
        )

    rs_save_btn.click(
        fn=_save_robosuite,
        inputs=[
            rs_name, rs_description, rs_base_env, rs_robot,
            rs_controller, rs_horizon, rs_reward_type, rs_cameras,
        ],
        outputs=[rs_status, rs_preview],
    )


# ---------------------------------------------------------------------------
# Saved Tasks
# ---------------------------------------------------------------------------

def _build_saved_tasks_tab(task_store: TaskStore) -> None:
    gr.Markdown("### Saved Custom Tasks")

    refresh_btn = gr.Button("Refresh", variant="secondary")

    gr.Markdown("#### LIBERO Tasks")
    libero_table = gr.Dataframe(
        headers=["Name", "Language", "Workspace", "BDDL Path"],
        datatype=["str", "str", "str", "str"],
        interactive=False,
    )

    gr.Markdown("#### Robosuite Tasks")
    robosuite_table = gr.Dataframe(
        headers=["Name", "Description", "Base Env", "Robot", "Horizon"],
        datatype=["str", "str", "str", "str", "number"],
        interactive=False,
    )

    gr.Markdown("#### Delete a Task")
    with gr.Row():
        del_backend = gr.Dropdown(
            label="Backend",
            choices=["libero", "robosuite"],
            value="libero",
        )
        del_name = gr.Textbox(label="Task Name", placeholder="task to delete")
        del_btn = gr.Button("Delete", variant="stop")

    del_status = gr.Textbox(label="Status", interactive=False)

    def _refresh():
        libero_rows = []
        for cfg in task_store.list_libero_tasks():
            libero_rows.append([
                cfg.task_name,
                cfg.language,
                cfg.workspace,
                cfg.bddl_file_path or "(not generated)",
            ])

        robosuite_rows = []
        for cfg in task_store.list_robosuite_tasks():
            robosuite_rows.append([
                cfg.task_name,
                cfg.description,
                cfg.base_env,
                cfg.robot,
                cfg.horizon,
            ])

        return libero_rows, robosuite_rows

    def _delete(backend, name):
        if not name or not name.strip():
            return "Please enter a task name."
        deleted = task_store.delete_task(backend, name.strip())
        if deleted:
            return f"Deleted {backend} task '{name.strip()}'."
        return f"Task '{name.strip()}' not found in {backend}."

    refresh_btn.click(fn=_refresh, outputs=[libero_table, robosuite_table])
    del_btn.click(fn=_delete, inputs=[del_backend, del_name], outputs=[del_status])
