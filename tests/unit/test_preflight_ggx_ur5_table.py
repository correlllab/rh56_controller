from pathlib import Path

import mujoco
import numpy as np
import pytest

from tools.preflight_ggx_ur5_table import (
    CandidateResult,
    DEFAULT_START_Q_DEG,
    TrajectorySample,
    annotate_review_frame,
    build_model,
    check_trajectory,
    desk_clearance,
    initial_qpos,
    parse_args,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
XML = REPO_ROOT / "h1_mujoco/inspire/ur5_inspire.xml"


def _model(table_height_m: float) -> mujoco.MjModel:
    return build_model(
        XML,
        table_height_m=table_height_m,
        object_center=np.array([0.0, -0.5, table_height_m + 0.02]),
        object_size_m=np.array([0.04, 0.04, 0.04]),
        object_shape="box",
    )


def test_home_pose_ignores_permanent_shoulder_fixture_contact() -> None:
    model = _model(0.070)
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos(model, np.radians(DEFAULT_START_Q_DEG))
    mujoco.mj_forward(model, data)

    clearance, nearest = desk_clearance(model, data)

    assert clearance > 0.030
    assert "shoulder_link" not in nearest


def test_table_height_directly_changes_robot_clearance() -> None:
    low_model = _model(0.070)
    high_model = _model(0.090)
    low_data = mujoco.MjData(low_model)
    high_data = mujoco.MjData(high_model)
    low_data.qpos[:] = initial_qpos(low_model, np.radians(DEFAULT_START_Q_DEG))
    high_data.qpos[:] = initial_qpos(high_model, np.radians(DEFAULT_START_Q_DEG))
    mujoco.mj_forward(low_model, low_data)
    mujoco.mj_forward(high_model, high_data)

    low_clearance, _ = desk_clearance(low_model, low_data)
    high_clearance, _ = desk_clearance(high_model, high_data)

    assert np.isclose(low_clearance - high_clearance, 0.020, atol=1e-6)


def test_required_clearance_is_an_explicit_gate() -> None:
    model = _model(0.070)
    qpos = initial_qpos(model, np.radians(DEFAULT_START_Q_DEG))
    sample = TrajectorySample("start", 0.0, qpos, 0.0)

    minimum, _, phase = check_trajectory(
        model, [sample], required_clearance_m=0.040
    )

    assert minimum < 0.040
    assert phase == "start"
    assert sample.desk_collision


def test_model_offscreen_size_can_be_enlarged_for_default_video() -> None:
    model = _model(0.070)

    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 720)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), 540)

    assert model.vis.global_.offwidth >= 720
    assert model.vis.global_.offheight >= 540


def test_yaml_defaults_are_overridden_by_explicit_cli(tmp_path: Path) -> None:
    config = tmp_path / "lab.yaml"
    config.write_text(
        "\n".join(
            [
                "graspgenx_yaml: artifacts/candidates.yml",
                "table_height_m: 0.070",
                "object_size_mm: [10, 20, 30]",
                "video: true",
            ]
        )
    )

    args = parse_args(
        [
            "--config",
            str(config),
            "--table-height-m",
            "0.083",
            "--no-video",
        ]
    )

    assert args.graspgenx_yaml == REPO_ROOT / "artifacts/candidates.yml"
    assert args.table_height_m == pytest.approx(0.083)
    assert args.object_size_mm == [10, 20, 30]
    assert not args.video


def test_unknown_yaml_field_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "lab.yaml"
    config.write_text("graspgenx_yaml: candidates.yml\nunsafe_guess: true\n")

    with pytest.raises(ValueError, match="unsafe_guess"):
        parse_args(["--config", str(config)])


def test_invalid_yaml_boolean_is_not_treated_as_truthy(tmp_path: Path) -> None:
    config = tmp_path / "lab.yaml"
    config.write_text("graspgenx_yaml: candidates.yml\nvideo: 'false'\n")

    with pytest.raises(ValueError, match="video"):
        parse_args(["--config", str(config)])


def test_review_overlay_contains_visible_status_panel() -> None:
    pytest.importorskip("PIL")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    sample = TrajectorySample(
        phase="approach",
        phase_alpha=0.5,
        qpos=np.zeros(6),
        closure_alpha=0.0,
        min_desk_clearance_m=0.004,
        nearest_geom="index_distal",
        desk_collision=False,
    )
    result = CandidateResult(
        rank=3,
        name="candidate_3",
        confidence=0.75,
        passed=True,
        reason="pass",
        ik_position_error_mm=1.0,
        ik_orientation_error_deg=1.0,
        min_desk_clearance_mm=4.0,
        nearest_geom="index_distal",
        first_collision_phase="",
        samples_checked=1,
    )

    annotated = annotate_review_frame(frame, sample, result, 0.004)

    assert annotated.shape == frame.shape
    assert np.any(annotated != frame)
