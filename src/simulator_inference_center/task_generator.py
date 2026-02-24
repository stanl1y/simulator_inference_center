"""LIBERO BDDL generation and robosuite config validation.

Also exports UI dropdown constants used by both the generator logic and the
Gradio Task Builder tab.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from simulator_inference_center.task_store import LiberoTaskConfig, RobosuiteTaskConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LIBERO constants (for UI dropdowns & validation)
# ---------------------------------------------------------------------------

LIBERO_WORKSPACES: list[str] = [
    "kitchen_table",
    "living_room_table",
    "study_table",
    "coffee_table",
    "main_table",
    "floor",
]

LIBERO_FIXTURES: list[str] = [
    "wooden_cabinet",
    "white_cabinet",
    "flat_stove",
    "microwave",
    "wooden_two_layer_shelf",
]

LIBERO_OBJECTS: list[str] = [
    "akita_black_bowl",
    "plate",
    "moka_pot",
    "wine_bottle",
    "alphabet_soup",
    "cream_cheese",
    "ketchup",
    "tomato_sauce",
    "butter",
    "orange_juice",
    "chocolate_pudding",
    "bbq_sauce",
    "salad_dressing",
    "white_bowl",
    "porcelain_mug",
    "chefmate_8_frypan",
]

# predicate_name -> arity (1 = unary, 2 = binary)
LIBERO_PREDICATES: dict[str, int] = {
    "On": 2,
    "In": 2,
    "Open": 1,
    "Close": 1,
    "TurnOn": 1,
    "TurnOff": 1,
    "Up": 1,
}

# ---------------------------------------------------------------------------
# Robosuite constants (for UI dropdowns & validation)
# ---------------------------------------------------------------------------

ROBOSUITE_ROBOTS: list[str] = [
    "Panda",
    "Sawyer",
    "Jaco",
    "IIWA",
    "Kinova3",
    "UR5e",
]

ROBOSUITE_CONTROLLERS: list[str] = [
    "OSC_POSE",
    "OSC_POSITION",
    "JOINT_VELOCITY",
    "JOINT_TORQUE",
    "JOINT_POSITION",
    "IK_POSE",
]

ROBOSUITE_BASE_ENVS: list[str] = [
    "Lift",
    "Stack",
    "NutAssembly",
    "NutAssemblySingle",
    "NutAssemblySquare",
    "NutAssemblyRound",
    "PickPlace",
    "PickPlaceSingle",
    "PickPlaceMilk",
    "PickPlaceBread",
    "PickPlaceCereal",
    "PickPlaceCan",
    "Door",
    "Wipe",
    "ToolHang",
    "TwoArmLift",
    "TwoArmPegInHole",
    "TwoArmHandover",
    "TwoArmTransport",
]

ROBOSUITE_CAMERAS: list[str] = [
    "agentview",
    "robot0_eye_in_hand",
    "frontview",
    "birdview",
    "sideview",
    "robot0_robotview",
]


# ---------------------------------------------------------------------------
# LIBERO Task Generator
# ---------------------------------------------------------------------------

# Module-level lock to serialize LIBERO global registry access.
_libero_registry_lock = threading.Lock()


class LiberoTaskGenerator:
    """Generate BDDL files for custom LIBERO tasks.

    Uses LIBERO's programmatic task generation pipeline:
    ``InitialSceneTemplates`` -> ``register_task_info`` -> ``generate_bddl_from_task_info``.
    """

    @staticmethod
    def validate(config: LiberoTaskConfig) -> list[str]:
        """Return a list of validation error messages (empty = valid)."""
        errors: list[str] = []

        if not config.task_name or not config.task_name.strip():
            errors.append("Task name is required.")

        if not config.language or not config.language.strip():
            errors.append("Language description is required.")

        if config.workspace not in LIBERO_WORKSPACES:
            errors.append(
                f"Unknown workspace {config.workspace!r}. "
                f"Choose from: {', '.join(LIBERO_WORKSPACES)}"
            )

        for fixture in config.fixtures:
            if fixture not in LIBERO_FIXTURES:
                errors.append(f"Unknown fixture: {fixture!r}")

        # Validate against the hardcoded list AND LIBERO's actual registry
        # when available.  The registry is the authoritative source.
        _real_objects = None
        try:
            from libero.libero.envs.objects import OBJECTS_DICT
            _real_objects = OBJECTS_DICT
        except ImportError:
            pass

        for obj in config.objects:
            if _real_objects is not None:
                if obj not in _real_objects:
                    errors.append(f"Unknown object: {obj!r} (not in LIBERO OBJECTS_DICT)")
            elif obj not in LIBERO_OBJECTS:
                errors.append(f"Unknown object: {obj!r}")

        if not config.goal_states:
            errors.append("At least one goal state is required.")

        for i, goal in enumerate(config.goal_states):
            if not isinstance(goal, (list, tuple)) or len(goal) < 2:
                errors.append(
                    f"Goal state {i}: must be a list [predicate, arg1, ...]. Got: {goal!r}"
                )
                continue
            predicate = goal[0]
            if predicate not in LIBERO_PREDICATES:
                errors.append(
                    f"Goal state {i}: unknown predicate {predicate!r}. "
                    f"Choose from: {', '.join(LIBERO_PREDICATES)}"
                )
            else:
                expected_arity = LIBERO_PREDICATES[predicate]
                actual_args = len(goal) - 1
                if actual_args != expected_arity:
                    errors.append(
                        f"Goal state {i}: predicate {predicate!r} expects "
                        f"{expected_arity} arg(s), got {actual_args}."
                    )

        return errors

    @staticmethod
    def generate(config: LiberoTaskConfig, output_dir: str) -> str:
        """Generate a BDDL file for the given config.

        Returns the path to the generated BDDL file.
        Raises RuntimeError if LIBERO is not installed or generation fails.
        """
        import os

        import numpy as np

        with _libero_registry_lock:
            try:
                from libero.libero.utils.bddl_generation_utils import (
                    get_xy_region_kwargs_list_from_regions_info,
                )
                from libero.libero.utils.mu_utils import (
                    InitialSceneTemplates,
                    register_mu,
                )
                from libero.libero.utils.task_generation_utils import (
                    TASK_INFO,
                    generate_bddl_from_task_info,
                    register_task_info,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "LIBERO is not installed. Cannot generate BDDL files."
                ) from exc

            # Build a dynamic scene template subclass.
            # InitialSceneTemplates.__init__() calls define_regions() and
            # accesses init_states, so we need real method implementations.
            class_name = f"Custom{_sanitize_class_name(config.task_name)}"

            fixture_info = config.fixtures.copy() if config.fixtures else {}
            object_info = config.objects.copy() if config.objects else {}
            workspace = config.workspace

            # The workspace table itself must be in fixture_num_info for
            # LIBERO to set up the scene correctly.
            if workspace not in fixture_info and workspace != "floor":
                fixture_info[workspace] = 1

            # Parse user-supplied regions into the format get_region_dict expects.
            user_regions = config.regions or []

            # Build instance names: LIBERO appends _1, _2, etc. to categories.
            # With count=1 (default), "butter" becomes "butter_1".
            all_instance_names: list[str] = []
            for cat, count in fixture_info.items():
                for i in range(1, count + 1):
                    all_instance_names.append(f"{cat}_{i}")
            for cat, count in object_info.items():
                for i in range(1, count + 1):
                    all_instance_names.append(f"{cat}_{i}")

            # Resolve objects_of_interest: accept both category names and
            # instance names.  Convert bare category names to _1 instances.
            raw_ooi = config.objects_of_interest or []
            resolved_ooi: list[str] = []
            for name in raw_ooi:
                if name in all_instance_names:
                    resolved_ooi.append(name)
                elif f"{name}_1" in all_instance_names:
                    resolved_ooi.append(f"{name}_1")
                else:
                    resolved_ooi.append(name)  # let LIBERO validate

            # Resolve goal state arguments the same way.
            resolved_goals: list[list[str]] = []
            for goal in config.goal_states:
                resolved = [goal[0]]  # predicate
                for arg in goal[1:]:
                    if arg in all_instance_names:
                        resolved.append(arg)
                    elif f"{arg}_1" in all_instance_names:
                        resolved.append(f"{arg}_1")
                    else:
                        resolved.append(arg)
                resolved_goals.append(resolved)

            # Build init_states: place each fixture and object in default
            # init regions on the workspace surface.
            init_state_list: list[tuple[str, ...]] = []
            for cat, count in fixture_info.items():
                if cat == workspace:
                    continue
                for i in range(1, count + 1):
                    init_state_list.append(
                        ("On", f"{cat}_{i}", f"{workspace}_{cat}_init_region")
                    )
            for cat, count in object_info.items():
                for i in range(1, count + 1):
                    init_state_list.append(
                        ("On", f"{cat}_{i}", f"{workspace}_{cat}_init_region")
                    )

            # Capture variables for closures in the dynamic class methods.
            _fixture_info = fixture_info
            _object_info = object_info
            _workspace = workspace
            _user_regions = user_regions
            _init_states = init_state_list

            def _dynamic_init(self):
                super(type(self), self).__init__(
                    workspace_name=_workspace,
                    fixture_num_info=_fixture_info,
                    object_num_info=_object_info,
                )

            def _define_regions(self):
                # Place each fixture and movable object in a default region.
                offset = 0.0
                for cat in list(_fixture_info.keys()) + list(_object_info.keys()):
                    if cat == _workspace:
                        continue
                    region_name = f"{cat}_init_region"
                    self.regions.update(
                        self.get_region_dict(
                            region_centroid_xy=[offset, offset],
                            region_name=region_name,
                            target_name=self.workspace_name,
                            region_half_len=0.025,
                        )
                    )
                    offset += 0.05

                # Add any user-supplied custom regions.
                for region_spec in _user_regions:
                    rname = region_spec.get("name", "")
                    if not rname:
                        continue
                    xy = region_spec.get("centroid_xy", [0.0, 0.0])
                    target = region_spec.get("target", self.workspace_name)
                    half_len = region_spec.get("half_len", 0.025)
                    self.regions.update(
                        self.get_region_dict(
                            region_centroid_xy=xy,
                            region_name=rname,
                            target_name=target,
                            region_half_len=half_len,
                        )
                    )

                self.xy_region_kwargs_list = (
                    get_xy_region_kwargs_list_from_regions_info(self.regions)
                )

            def _init_states_prop(self):
                return list(_init_states)

            scene_cls = type(class_name, (InitialSceneTemplates,), {
                "__init__": _dynamic_init,
                "define_regions": _define_regions,
                "init_states": property(_init_states_prop),
            })

            # Register in MU_DICT.  register_mu converts the CamelCase class
            # name to snake_case for the registry key.
            register_mu(scene_type="custom")(scene_cls)

            # Determine the registry key (snake_case of class name).
            import re as _re
            scene_key = "_".join(
                _re.sub(r"([A-Z])", r" \1", class_name).split()
            ).lower()

            # Register the task in LIBERO's global TASK_INFO.
            # Goal states must be tuples (not lists) for LIBERO's BDDL
            # string formatter to work correctly.
            register_task_info(
                language=config.language,
                scene_name=scene_key,
                objects_of_interest=resolved_ooi,
                goal_states=[tuple(g) for g in resolved_goals],
            )

            # Generate BDDL files from the global registry.
            os.makedirs(output_dir, exist_ok=True)
            bddl_file_names, failures = generate_bddl_from_task_info(
                folder=output_dir,
            )

            if failures:
                raise RuntimeError(
                    f"BDDL generation failed for: {failures}"
                )

            # Find the file that was generated for our scene.
            bddl_path = None
            for fname in bddl_file_names:
                if scene_key in fname.lower() or config.task_name.lower().replace(" ", "_") in fname.lower():
                    bddl_path = fname
                    break
            if bddl_path is None and bddl_file_names:
                # Fall back to the last generated file.
                bddl_path = bddl_file_names[-1]

            if bddl_path is None:
                raise RuntimeError("BDDL generation produced no output files.")

            # Clean up the global TASK_INFO to avoid polluting future calls.
            TASK_INFO.pop(scene_key, None)

            logger.info("Generated BDDL for %r at %s", config.task_name, bddl_path)
            return str(bddl_path)


def _sanitize_class_name(name: str) -> str:
    """Convert a task name to a valid Python class name suffix."""
    import re
    sanitized = re.sub(r"[^a-zA-Z0-9]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized or "unnamed"


# ---------------------------------------------------------------------------
# Robosuite Task Generator
# ---------------------------------------------------------------------------

class RobosuiteTaskGenerator:
    """Validate robosuite task configurations.

    No code generation needed — robosuite tasks use ``robosuite.make()``
    with stored parameters at runtime.
    """

    @staticmethod
    def validate(config: RobosuiteTaskConfig) -> list[str]:
        """Return a list of validation error messages (empty = valid)."""
        errors: list[str] = []

        if not config.task_name or not config.task_name.strip():
            errors.append("Task name is required.")

        if config.robot not in ROBOSUITE_ROBOTS:
            errors.append(
                f"Unknown robot {config.robot!r}. "
                f"Choose from: {', '.join(ROBOSUITE_ROBOTS)}"
            )

        if config.controller is not None and config.controller not in ROBOSUITE_CONTROLLERS:
            errors.append(
                f"Unknown controller {config.controller!r}. "
                f"Choose from: {', '.join(ROBOSUITE_CONTROLLERS)}"
            )

        if config.base_env not in ROBOSUITE_BASE_ENVS:
            errors.append(
                f"Unknown base environment {config.base_env!r}. "
                f"Choose from: {', '.join(ROBOSUITE_BASE_ENVS)}"
            )

        if config.horizon < 1:
            errors.append("Horizon must be >= 1.")

        if config.reward_type not in ("sparse", "dense"):
            errors.append(
                f"Unknown reward type {config.reward_type!r}. "
                f"Choose 'sparse' or 'dense'."
            )

        for cam in config.camera_names:
            if cam not in ROBOSUITE_CAMERAS:
                errors.append(f"Unknown camera: {cam!r}")

        return errors
