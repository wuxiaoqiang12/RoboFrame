"""Scheduled reset barriers for the pick-banana simulation task."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from sim_models.tasks.pick_banana import PickBananaTask, _parse_app_args

_PRE_SCHEDULER_PARAMETERS = {
    "arm_command_topic",
    "eval_output_dir",
    "gripper_command_topic",
    "reset_world_service",
    "restart_session_service",
}


def _declared_parameter_names(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    return {
        call.args[0].value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "declare_parameter"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    }


def test_pick_banana_ros_parameter_surface_matches_pre_scheduler_baseline():
    source = Path(__file__).resolve().parents[1] / "sim_models" / "tasks" / "pick_banana.py"

    assert _declared_parameter_names(source) == _PRE_SCHEDULER_PARAMETERS


def test_legacy_app_args_do_not_add_a_scheduler_parameter():
    scheduled, ros_args = _parse_app_args(["--ros-args", "-p", "use_sim_time:=true"])

    assert not scheduled
    assert ros_args == ["--ros-args", "-p", "use_sim_time:=true"]


def test_scheduled_app_arg_is_removed_before_ros_parsing():
    scheduled, ros_args = _parse_app_args(["--scheduled-dispatch", "--ros-args", "-p", "use_sim_time:=true"])

    assert scheduled
    assert ros_args == ["--ros-args", "-p", "use_sim_time:=true"]


def _task() -> PickBananaTask:
    task = object.__new__(PickBananaTask)
    task._scheduled_dispatch = True
    task._disp_stop = object()
    task._disp_start = object()
    task._disp_reset = object()
    task._node = SimpleNamespace(get_logger=lambda: SimpleNamespace(warning=lambda _message: None))
    return task


def test_scheduled_stop_failure_aborts_before_world_reset():
    task = _task()
    calls: list[str] = []
    task._call_service_sync = lambda _client, _request, label, **_kwargs: calls.append(label) or False
    task._publish_rest_pose = lambda: (_ for _ in ()).throw(
        AssertionError("rest pose and world reset must not run after failed Close")
    )

    result = PickBananaTask._clean_world_reset(task, randomize=False, resume=True, settle_s=0)

    assert result == (False, "scheduled dispatcher stop/safe-stop/Close failed; world reset aborted")
    assert calls == ["dispatcher/stop"]


def test_scheduled_open_failure_is_reported_after_successful_world_reset():
    task = _task()
    calls: list[str] = []

    def call_service(_client, _request, label, **_kwargs):
        calls.append(label)
        return label == "dispatcher/stop"

    task._call_service_sync = call_service
    task._publish_rest_pose = lambda: None
    task.reset = lambda: (True, "reset")

    result = PickBananaTask._clean_world_reset(task, randomize=False, resume=True, settle_s=0)

    assert result == (False, "world reset succeeded but scheduled session Open failed")
    assert calls == ["dispatcher/stop", "dispatcher/start"]
