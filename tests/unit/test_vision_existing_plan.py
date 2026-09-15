"""Vision input and write-only execution regressions, without hardware."""

from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from rh56_controller.grasp_executor import GraspExecutor
from rh56_controller.grasp_viz_core import GraspVizCore
from rh56_controller.vision_grasp_target import load_vision_target


@pytest.fixture
def measurement(tmp_path):
    path = tmp_path / "target_validation.yaml"
    path.write_text(yaml.safe_dump({
        "schema": "rh56_ur5_camera_colored_target_validation/v1",
        "quality_pass": True,
        "estimated_target_top_center_base_mm": [-226.0, -611.0, 80.0],
        "physical_reference": {"target_height_mm": 40.0},
    }))
    return path


def test_camera_top_is_not_used_as_grasp_center_or_tcp(measurement):
    target = load_vision_target(measurement, object_width_mm=40, yaw_deg=90)
    np.testing.assert_allclose(target.center_base_m, [-.226, -.611, .060])
    assert target.planner_width_m == .040  # No implicit 20 mm offset.
    explicit = load_vision_target(measurement, object_width_mm=40, yaw_deg=90,
                                  width_offset_mm=20, depth_below_top_mm=10)
    assert explicit.planner_width_m == .060
    assert explicit.center_base_m[2] == .070


@pytest.mark.parametrize("kwargs", [
    {"object_width_mm": float("nan")}, {"yaw_deg": float("inf")},
    {"depth_below_top_mm": 40}, {"depth_below_top_mm": -1},
    {"width_offset_mm": -40},
])
def test_bad_geometry_is_rejected(measurement, kwargs):
    with pytest.raises(ValueError):
        load_vision_target(measurement, **({"object_width_mm": 40, "yaw_deg": 90} | kwargs))


def test_failed_measurement_is_rejected(measurement):
    report = yaml.safe_load(measurement.read_text())
    report["quality_pass"] = False
    measurement.write_text(yaml.safe_dump(report))
    with pytest.raises(ValueError, match="quality"):
        load_vision_target(measurement, object_width_mm=40, yaw_deg=90)


def test_existing_plan_keeps_object_center_and_changes_wrist(measurement, tmp_path, monkeypatch):
    # Use the actual core, but disable optional workers and all hardware.
    target = load_vision_target(measurement, object_width_mm=40, yaw_deg=90,
                                width_offset_mm=20)
    monkeypatch.setattr("rh56_controller.grasp_geometry._CACHE_PATH", str(tmp_path / "fk.npz"))
    core = GraspVizCore(vision_target=target, mink_viz=False, hand_write_only=True)
    try:
        result = core.closure.solve(core._mode, core._width_target_m)
        closures = core._compute_plan_closures(10, result, .09)
        matrices = [core._build_world_T_hand(r) for r in closures]
        for r, T in zip(closures, matrices):
            np.testing.assert_allclose(
                T[:3, :3] @ r.grasp_center("antipodal") + T[:3, 3],
                target.center_base_m, atol=1e-10,
            )
        assert np.linalg.norm(matrices[0][:3, 3] - matrices[-1][:3, 3]) > .01
        assert not np.allclose(matrices[0][:3, :3], matrices[-1][:3, :3])
        assert core._arm is None and core._hand is None
        assert core._grasp_strategy == "Plan"
    finally:
        core.cleanup()


def test_invalid_vision_options_fail_before_hardware(measurement, monkeypatch):
    target = load_vision_target(measurement, object_width_mm=40, yaw_deg=90)
    monkeypatch.setattr(GraspVizCore, "_init_hand", lambda *a: pytest.fail("hardware init"))
    for options in ({"h12_mode": True}, {"send_real": True}, {"ros_send_hand_cmd": True}):
        with pytest.raises(ValueError):
            GraspVizCore(vision_target=target, port="not-a-device", **options)


@pytest.mark.parametrize("write_only", [False, True])
def test_plan_sequence_and_sleeps_unchanged_without_hand_feedback(write_only, monkeypatch):
    events, sleeps, reads = [], [], []
    monkeypatch.setattr("rh56_controller.grasp_executor.time.sleep", sleeps.append)

    class Hand:
        def angle_set(self, cmd):
            events.append(("hand", cmd.copy()))
            return None  # Missing ACK does not gate the legacy Plan.

        def angle_read(self):
            assert not write_only
            reads.append("angle")
            return None

        def force_act(self):
            assert not write_only
            reads.append("force")
            return None

    class Arm:
        T = np.eye(4)

        def move_to_hand_pose(self, T, blocking):
            assert blocking
            self.T = T
            events.append(("arm", float(T[0, 3])))
            return []

        def get_hand_pose_4x4(self):
            return self.T

        def get_joint_angles(self):
            return [0] * 6

    names = ["pinky", "ring", "middle", "index", "thumb_proximal", "thumb_yaw"]
    fk = SimpleNamespace(ctrl_min=dict.fromkeys(names, 0), ctrl_max=dict.fromkeys(names, 1))
    executor = GraspExecutor(Arm(), Hand(), fk, status_cb=lambda _: None,
                             hand_write_only=write_only)
    monkeypatch.setattr(executor, "_start", lambda fn, *a: fn(*a))
    waypoints = []
    for i in range(3):
        T = np.eye(4)
        T[0, 3] = i * .01
        waypoints.append((T, SimpleNamespace(width=.09 - i*.01,
                                            ctrl_values=dict.fromkeys(names, .1*i))))
    logger = SimpleNamespace(log_waypoint=lambda **k: None,
                             log_done=lambda **k: None, close=lambda: None)
    executor.execute_plan_waypoints(waypoints, logger=logger)
    assert [e[0] for e in events] == ["hand", "arm", "arm", "hand", "arm", "hand"]
    assert sleeps == [.3]  # No new per-step wait or sim timing.
    assert len(reads) == (0 if write_only else 6)
    assert events[-1][1] == [800] * 6


def test_write_only_force_mode_fails_before_starting():
    executor = GraspExecutor(None, None, None, hand_write_only=True)
    with pytest.raises(ValueError, match="Force"):
        executor.execute_plan_waypoints([], force_N=1)
    assert executor._thread is None
