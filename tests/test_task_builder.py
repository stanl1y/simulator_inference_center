"""Tests for TaskStore, task generators, and dashboard task builder integration."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from simulator_inference_center.task_store import (
    LiberoTaskConfig,
    RobosuiteTaskConfig,
    TaskStore,
    _sanitize_name,
)
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_store(tmp_path):
    """Create a TaskStore using a temporary directory."""
    return TaskStore(str(tmp_path / "custom_tasks"))


@pytest.fixture()
def sample_libero_config():
    return LiberoTaskConfig(
        task_name="test_pick_bowl",
        language="Pick up the black bowl and place it on the plate",
        workspace="kitchen_table",
        fixtures={"wooden_cabinet": 1},
        objects={"akita_black_bowl": 1, "plate": 1},
        regions=[],
        objects_of_interest=["akita_black_bowl", "plate"],
        goal_states=[["On", "akita_black_bowl", "plate"]],
    )


@pytest.fixture()
def sample_robosuite_config():
    return RobosuiteTaskConfig(
        task_name="test_custom_lift",
        description="Custom Lift with Sawyer",
        robot="Sawyer",
        controller="OSC_POSE",
        base_env="Lift",
        horizon=500,
        reward_type="dense",
        camera_names=["agentview", "frontview"],
    )


# ---------------------------------------------------------------------------
# _sanitize_name
# ---------------------------------------------------------------------------

class TestSanitizeName:
    def test_basic(self):
        assert _sanitize_name("hello_world") == "hello_world"

    def test_spaces_and_special(self):
        assert _sanitize_name("my task (v2)!") == "my_task_v2"

    def test_empty(self):
        assert _sanitize_name("") == "unnamed"

    def test_only_special(self):
        assert _sanitize_name("!!!") == "unnamed"


# ---------------------------------------------------------------------------
# LiberoTaskConfig / RobosuiteTaskConfig
# ---------------------------------------------------------------------------

class TestTaskConfigSerialisation:
    def test_libero_round_trip(self, sample_libero_config):
        d = sample_libero_config.to_dict()
        restored = LiberoTaskConfig.from_dict(d)
        assert restored.task_name == sample_libero_config.task_name
        assert restored.language == sample_libero_config.language
        assert restored.workspace == sample_libero_config.workspace
        assert restored.goal_states == sample_libero_config.goal_states

    def test_robosuite_round_trip(self, sample_robosuite_config):
        d = sample_robosuite_config.to_dict()
        restored = RobosuiteTaskConfig.from_dict(d)
        assert restored.task_name == sample_robosuite_config.task_name
        assert restored.robot == sample_robosuite_config.robot
        assert restored.horizon == sample_robosuite_config.horizon

    def test_from_dict_ignores_extra_keys(self):
        data = {"task_name": "foo", "language": "bar", "workspace": "kitchen_table", "extra_key": 42}
        config = LiberoTaskConfig.from_dict(data)
        assert config.task_name == "foo"


# ---------------------------------------------------------------------------
# TaskStore CRUD
# ---------------------------------------------------------------------------

class TestTaskStoreCRUD:
    def test_save_and_list_libero(self, tmp_store, sample_libero_config):
        tmp_store.save_libero_task(sample_libero_config)
        tasks = tmp_store.list_libero_tasks()
        assert len(tasks) == 1
        assert tasks[0].task_name == "test_pick_bowl"

    def test_save_and_list_robosuite(self, tmp_store, sample_robosuite_config):
        tmp_store.save_robosuite_task(sample_robosuite_config)
        tasks = tmp_store.list_robosuite_tasks()
        assert len(tasks) == 1
        assert tasks[0].task_name == "test_custom_lift"

    def test_get_libero_task(self, tmp_store, sample_libero_config):
        tmp_store.save_libero_task(sample_libero_config)
        config = tmp_store.get_libero_task("test_pick_bowl")
        assert config is not None
        assert config.language == sample_libero_config.language

    def test_get_libero_task_not_found(self, tmp_store):
        assert tmp_store.get_libero_task("nonexistent") is None

    def test_get_robosuite_task(self, tmp_store, sample_robosuite_config):
        tmp_store.save_robosuite_task(sample_robosuite_config)
        config = tmp_store.get_robosuite_task("test_custom_lift")
        assert config is not None
        assert config.robot == "Sawyer"

    def test_get_robosuite_task_not_found(self, tmp_store):
        assert tmp_store.get_robosuite_task("nonexistent") is None

    def test_delete_libero(self, tmp_store, sample_libero_config):
        tmp_store.save_libero_task(sample_libero_config)
        assert tmp_store.delete_task("libero", "test_pick_bowl") is True
        assert tmp_store.list_libero_tasks() == []

    def test_delete_robosuite(self, tmp_store, sample_robosuite_config):
        tmp_store.save_robosuite_task(sample_robosuite_config)
        assert tmp_store.delete_task("robosuite", "test_custom_lift") is True
        assert tmp_store.list_robosuite_tasks() == []

    def test_delete_nonexistent(self, tmp_store):
        assert tmp_store.delete_task("libero", "nonexistent") is False

    def test_delete_invalid_backend(self, tmp_store):
        assert tmp_store.delete_task("invalid", "something") is False

    def test_overwrite_existing(self, tmp_store, sample_libero_config):
        tmp_store.save_libero_task(sample_libero_config)
        sample_libero_config.language = "Updated language"
        tmp_store.save_libero_task(sample_libero_config)
        tasks = tmp_store.list_libero_tasks()
        assert len(tasks) == 1
        assert tasks[0].language == "Updated language"

    def test_multiple_tasks(self, tmp_store):
        for i in range(5):
            config = LiberoTaskConfig(
                task_name=f"task_{i}",
                language=f"Task {i}",
                workspace="kitchen_table",
                goal_states=[["On", "akita_black_bowl", "plate"]],
            )
            tmp_store.save_libero_task(config)
        tasks = tmp_store.list_libero_tasks()
        assert len(tasks) == 5

    def test_json_file_content(self, tmp_store, sample_libero_config):
        filepath = tmp_store.save_libero_task(sample_libero_config)
        data = json.loads(filepath.read_text())
        assert data["task_name"] == "test_pick_bowl"
        assert data["workspace"] == "kitchen_table"


# ---------------------------------------------------------------------------
# TaskStore thread safety
# ---------------------------------------------------------------------------

class TestTaskStoreThreadSafety:
    def test_concurrent_writes(self, tmp_store):
        n_threads = 10
        n_per_thread = 20
        barrier = threading.Barrier(n_threads)

        def worker(tid):
            barrier.wait()
            for i in range(n_per_thread):
                config = LiberoTaskConfig(
                    task_name=f"thread_{tid}_task_{i}",
                    language=f"Task {i} from thread {tid}",
                    workspace="kitchen_table",
                    goal_states=[["On", "akita_black_bowl", "plate"]],
                )
                tmp_store.save_libero_task(config)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        tasks = tmp_store.list_libero_tasks()
        assert len(tasks) == n_threads * n_per_thread


# ---------------------------------------------------------------------------
# LiberoTaskGenerator validation
# ---------------------------------------------------------------------------

class TestLiberoValidation:
    def test_valid_config(self, sample_libero_config):
        errors = LiberoTaskGenerator.validate(sample_libero_config)
        assert errors == []

    def test_missing_name(self):
        config = LiberoTaskConfig(
            task_name="",
            language="Pick up bowl",
            workspace="kitchen_table",
            goal_states=[["On", "akita_black_bowl", "plate"]],
        )
        errors = LiberoTaskGenerator.validate(config)
        assert any("name" in e.lower() for e in errors)

    def test_missing_language(self):
        config = LiberoTaskConfig(
            task_name="test",
            language="",
            workspace="kitchen_table",
            goal_states=[["On", "akita_black_bowl", "plate"]],
        )
        errors = LiberoTaskGenerator.validate(config)
        assert any("language" in e.lower() for e in errors)

    def test_invalid_workspace(self):
        config = LiberoTaskConfig(
            task_name="test",
            language="test",
            workspace="nonexistent_table",
            goal_states=[["On", "akita_black_bowl", "plate"]],
        )
        errors = LiberoTaskGenerator.validate(config)
        assert any("workspace" in e.lower() for e in errors)

    def test_unknown_fixture(self):
        config = LiberoTaskConfig(
            task_name="test",
            language="test",
            workspace="kitchen_table",
            fixtures={"not_a_fixture": 1},
            goal_states=[["On", "akita_black_bowl", "plate"]],
        )
        errors = LiberoTaskGenerator.validate(config)
        assert any("fixture" in e.lower() for e in errors)

    def test_unknown_object(self):
        config = LiberoTaskConfig(
            task_name="test",
            language="test",
            workspace="kitchen_table",
            objects={"not_an_object": 1},
            goal_states=[["On", "akita_black_bowl", "plate"]],
        )
        errors = LiberoTaskGenerator.validate(config)
        assert any("object" in e.lower() for e in errors)

    def test_no_goal_states(self):
        config = LiberoTaskConfig(
            task_name="test",
            language="test",
            workspace="kitchen_table",
            goal_states=[],
        )
        errors = LiberoTaskGenerator.validate(config)
        assert any("goal" in e.lower() for e in errors)

    def test_invalid_predicate(self):
        config = LiberoTaskConfig(
            task_name="test",
            language="test",
            workspace="kitchen_table",
            goal_states=[["BadPredicate", "obj1"]],
        )
        errors = LiberoTaskGenerator.validate(config)
        assert any("predicate" in e.lower() for e in errors)

    def test_wrong_arity(self):
        config = LiberoTaskConfig(
            task_name="test",
            language="test",
            workspace="kitchen_table",
            goal_states=[["On", "only_one_arg"]],  # On is binary, needs 2 args
        )
        errors = LiberoTaskGenerator.validate(config)
        assert any("arg" in e.lower() for e in errors)

    def test_malformed_goal_state(self):
        config = LiberoTaskConfig(
            task_name="test",
            language="test",
            workspace="kitchen_table",
            goal_states=[["single"]],  # too short
        )
        errors = LiberoTaskGenerator.validate(config)
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# RobosuiteTaskGenerator validation
# ---------------------------------------------------------------------------

class TestRobosuiteValidation:
    def test_valid_config(self, sample_robosuite_config):
        errors = RobosuiteTaskGenerator.validate(sample_robosuite_config)
        assert errors == []

    def test_missing_name(self):
        config = RobosuiteTaskConfig(task_name="", base_env="Lift")
        errors = RobosuiteTaskGenerator.validate(config)
        assert any("name" in e.lower() for e in errors)

    def test_invalid_robot(self):
        config = RobosuiteTaskConfig(
            task_name="test", robot="NotARobot", base_env="Lift"
        )
        errors = RobosuiteTaskGenerator.validate(config)
        assert any("robot" in e.lower() for e in errors)

    def test_invalid_controller(self):
        config = RobosuiteTaskConfig(
            task_name="test", controller="BAD_CTRL", base_env="Lift"
        )
        errors = RobosuiteTaskGenerator.validate(config)
        assert any("controller" in e.lower() for e in errors)

    def test_none_controller_is_valid(self):
        config = RobosuiteTaskConfig(
            task_name="test", controller=None, base_env="Lift"
        )
        errors = RobosuiteTaskGenerator.validate(config)
        assert not any("controller" in e.lower() for e in errors)

    def test_invalid_base_env(self):
        config = RobosuiteTaskConfig(
            task_name="test", base_env="NotAnEnv"
        )
        errors = RobosuiteTaskGenerator.validate(config)
        assert any("environment" in e.lower() for e in errors)

    def test_invalid_horizon(self):
        config = RobosuiteTaskConfig(
            task_name="test", base_env="Lift", horizon=0
        )
        errors = RobosuiteTaskGenerator.validate(config)
        assert any("horizon" in e.lower() for e in errors)

    def test_invalid_reward_type(self):
        config = RobosuiteTaskConfig(
            task_name="test", base_env="Lift", reward_type="unknown"
        )
        errors = RobosuiteTaskGenerator.validate(config)
        assert any("reward" in e.lower() for e in errors)

    def test_invalid_camera(self):
        config = RobosuiteTaskConfig(
            task_name="test", base_env="Lift", camera_names=["bad_camera"]
        )
        errors = RobosuiteTaskGenerator.validate(config)
        assert any("camera" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------

class TestConstants:
    def test_libero_workspaces_nonempty(self):
        assert len(LIBERO_WORKSPACES) > 0

    def test_libero_fixtures_nonempty(self):
        assert len(LIBERO_FIXTURES) > 0

    def test_libero_objects_nonempty(self):
        assert len(LIBERO_OBJECTS) > 0

    def test_libero_predicates_nonempty(self):
        assert len(LIBERO_PREDICATES) > 0

    def test_robosuite_robots_nonempty(self):
        assert len(ROBOSUITE_ROBOTS) > 0

    def test_robosuite_controllers_nonempty(self):
        assert len(ROBOSUITE_CONTROLLERS) > 0

    def test_robosuite_base_envs_nonempty(self):
        assert len(ROBOSUITE_BASE_ENVS) > 0

    def test_robosuite_cameras_nonempty(self):
        assert len(ROBOSUITE_CAMERAS) > 0


# ---------------------------------------------------------------------------
# Monitor task_created callback
# ---------------------------------------------------------------------------

class TestMonitorTaskCreatedCallback:
    def test_callback_invoked(self):
        from simulator_inference_center.monitor import ServerMonitor

        monitor = ServerMonitor()
        events = []
        monitor.register_task_created_callback(
            lambda backend, name, **kw: events.append((backend, name))
        )
        monitor.notify_task_created("libero", "my_task")
        assert events == [("libero", "my_task")]

    def test_callback_exception_does_not_propagate(self):
        from simulator_inference_center.monitor import ServerMonitor

        monitor = ServerMonitor()
        monitor.register_task_created_callback(
            lambda backend, name, **kw: 1 / 0  # ZeroDivisionError
        )
        # Should not raise
        monitor.notify_task_created("robosuite", "task")

    def test_multiple_callbacks(self):
        from simulator_inference_center.monitor import ServerMonitor

        monitor = ServerMonitor()
        results = []
        monitor.register_task_created_callback(
            lambda b, n, **kw: results.append(("cb1", b, n))
        )
        monitor.register_task_created_callback(
            lambda b, n, **kw: results.append(("cb2", b, n))
        )
        monitor.notify_task_created("libero", "task_x")
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Dashboard creation with task_store
# ---------------------------------------------------------------------------

class TestDashboardWithTaskStore:
    def test_create_dashboard_without_task_store(self):
        gradio = pytest.importorskip("gradio")
        from simulator_inference_center.dashboard import create_dashboard
        from simulator_inference_center.monitor import ServerMonitor

        monitor = ServerMonitor()
        app = create_dashboard(monitor)
        assert isinstance(app, gradio.Blocks)

    def test_create_dashboard_with_task_store(self, tmp_store):
        gradio = pytest.importorskip("gradio")
        from simulator_inference_center.dashboard import create_dashboard
        from simulator_inference_center.monitor import ServerMonitor

        monitor = ServerMonitor()
        app = create_dashboard(monitor, task_store=tmp_store)
        assert isinstance(app, gradio.Blocks)


# ---------------------------------------------------------------------------
# Server with task_store
# ---------------------------------------------------------------------------

class TestServerWithTaskStore:
    def test_server_accepts_task_store(self, tmp_store):
        from simulator_inference_center.config import ServerConfig
        from simulator_inference_center.server import InferenceServer

        # Register mock backend for test
        from simulator_inference_center.backends import register_backend
        from simulator_inference_center.backend import SimulatorBackend

        class _MockStoreBackend(SimulatorBackend):
            def __init__(self, task_store=None):
                self.task_store = task_store
            def list_tasks(self): return []
            def load_task(self, name): raise ValueError(name)
            def reset(self): return {}
            def step(self, a): return {}
            def get_info(self): return {"backend_name": "mock_store"}
            def close(self): pass

        register_backend("mock_store", _MockStoreBackend)

        config = ServerConfig(
            bind_address="tcp://*:0",
        )
        server = InferenceServer(config, task_store=tmp_store)
        # Verify backend receives task_store
        backend = server._create_backend("mock_store")
        assert backend.task_store is tmp_store
