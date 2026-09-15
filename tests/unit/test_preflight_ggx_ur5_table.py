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
    candidate_for_object_origin,
    check_trajectory,
    desk_clearance,
    initial_qpos,
    parse_args,
    evaluate_candidate,
    certified_separation,
    certified_object_separation,
    object_clearance,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
XML = REPO_ROOT / "h1_mujoco/inspire/ur5_inspire.xml"


def _object_collision_model(body="index_intermediate"):
    return mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <geom name="floor" type="plane" size="1 1 .01"/>
      <body name="{body}" pos="0 0 .3">
        <joint type="slide" axis="1 0 0"/>
        <geom name="robot_collision" type="sphere" size=".05"/>
        <geom name="visual_only" type="sphere" size=".4" contype="0" conaffinity="0"/>
      </body>
      <body name="preflight_object" pos="0 0 .3">
        <geom name="preflight_object_visual" type="sphere" size=".05"/>
      </body>
    </worldbody></mujoco>''')


def _certificate_model(x=.3, mesh=False, angle=0.):
    vertices = ' '.join(str(v) for a in (-.05, .05) for b in (-.05, .05)
                        for c in (-.05, .05) for v in (a, b, c))
    asset = f'<asset><mesh name="shape" vertex="{vertices}"/></asset>' if mesh else ''
    geom = 'type="mesh" mesh="shape"' if mesh else 'type="box" size=".05 .05 .05"'
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>{asset}<worldbody>
      <body name="robot" pos="{x} 0 .3" axisangle="0 0 1 {angle}">
        <joint type="slide" axis="1 0 0"/>
        <geom name="robot_collision" {geom}/>
      </body>
      <body name="preflight_object" pos="0 0 .3">
        <geom name="preflight_object_visual" type="box" size=".05 .05 .05"/>
      </body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, model.geom('preflight_object_visual').id, model.geom('robot_collision').id


@pytest.mark.parametrize('mesh', [False, True])
@pytest.mark.parametrize('angle', [0., 45.])
def test_certificate_uses_compiled_world_vertices(mesh, angle):
    model, data, a, b = _certificate_model(mesh=mesh, angle=angle)
    radius_x = .05 if angle == 0 else .05 * 2**.5
    assert certified_separation(model, data, a, b) == pytest.approx(.3 - .05 - radius_x - 1e-7, abs=1e-8)


@pytest.mark.parametrize('x', [0., .1])
@pytest.mark.parametrize('mesh', [False, True])
def test_overlap_and_exact_touch_cannot_get_separation_certificate(x, mesh):
    model, data, a, b = _certificate_model(x=x, mesh=mesh)
    assert certified_separation(model, data, a, b) == 0.
    assert certified_object_separation(model, data, a, b) == 0.


def test_false_zero_replaced_only_by_positive_certificate(monkeypatch):
    model, data, _, _ = _certificate_model(mesh=True)
    monkeypatch.setattr(mujoco, 'mj_geomDistance', lambda *args: 0.)
    corrections = []
    gap, _ = object_clearance(model, data, corrections)
    assert gap == pytest.approx(.2 - 1e-7)
    assert len(corrections) == 1


def test_actual_contact_cannot_be_overridden(monkeypatch):
    from tools import preflight_ggx_ur5_table as preflight
    model, data, _, _ = _certificate_model(x=0.)
    assert len(data.contact) > 0
    monkeypatch.setattr(mujoco, 'mj_geomDistance', lambda *args: 0.)
    monkeypatch.setattr(preflight, 'certified_separation', lambda *args: pytest.fail('Contact must win'))
    corrections = []
    assert object_clearance(model, data, corrections)[0] == 0.
    assert not corrections


def test_negative_distance_is_not_weakened(monkeypatch):
    model, data, _, _ = _certificate_model()
    monkeypatch.setattr(mujoco, 'mj_geomDistance', lambda *args: -.01)
    corrections = []
    assert object_clearance(model, data, corrections)[0] == -.01
    assert not corrections


def test_unsupported_geometry_zero_stays_rejected(monkeypatch):
    model = _object_collision_model()
    data = mujoco.MjData(model)
    data.qpos[:] = [.3]
    mujoco.mj_forward(model, data)
    monkeypatch.setattr(mujoco, 'mj_geomDistance', lambda *args: 0.)
    corrections = []
    assert object_clearance(model, data, corrections)[0] == 0.
    assert not corrections


def test_recorded_good_path_false_zero_regression_without_artifacts():
    # 2026-09-10 original upright sugar rank2/sample13. Literal geometry/joints
    # keep this regression independent of ignored experiment artifacts and IK.
    model = build_model(
        XML, table_height_m=.03995441967699594,
        object_center=np.array([-.2319775848388672, -.6306512451171875, .12795441967699594]),
        object_size_m=np.array([.09, .04, .176]), object_shape='box',
        object_yaw_deg=-1.4176516532898,
    )
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos(model, np.array([
        .4221961013551433, -1.7792961166459826, 2.341459228230319,
        3.201000252801453, -.5395330156968979, -.509699069304641,
    ]))
    mujoco.mj_forward(model, data)
    obj = model.geom('preflight_object_visual').id
    assert not any(obj in (c.geom1, c.geom2) and c.dist <= 0 for c in data.contact)
    assert object_clearance(model, data)[0] > .01


@pytest.fixture
def rank87_pair():
    # Frozen seed 20260909/run_00, rank 87, local sample 16. Original mesh and
    # world body poses: no IK, ignored artifacts, hardware or proxy resizing.
    mesh = REPO_ROOT / 'h1_mujoco/inspire/assets/collision/right_index_intermediate.stl'
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <asset><mesh name="finger" file="{mesh}"/></asset><worldbody>
        <body name="index_intermediate"
              pos="-.16388905021898723 -.6669310956623787 .16367861626693975"
              quat="-.18270583919922426 .9643704159317308 -.188634556695441 -.032020012780801785">
          <freejoint/><geom name="finger" type="mesh" mesh="finger"/>
        </body>
        <body name="preflight_object"
              pos="-.21906597900390626 -.6459269409179688 .13793038326475576"
              quat=".9997731796206614 0 0 -.02129763628182179">
          <geom name="preflight_object_visual" type="box"
                size=".046930649961476535 .025131112345664314 .09851265904926755"/>
        </body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, model.geom('preflight_object_visual').id, model.geom('finger').id


def test_rank87_false_zero_needs_finger_face_certificate(rank87_pair, monkeypatch):
    from tools import ggx_motion as motion
    model, data, obj, finger = rank87_pair
    assert not len(data.contact)
    assert certified_separation(model, data, obj, finger) == 0.
    expected = certified_separation(model, data, finger, obj, True)
    assert expected == pytest.approx(.0024880871731116114, abs=1e-9)
    # Force the recorded zero independently of MuJoCo version / query rounding.
    monkeypatch.setattr(mujoco, 'mj_geomDistance', lambda *args: 0.)
    corrections = []
    assert object_clearance(model, data, corrections)[0] == pytest.approx(expected)
    assert corrections == ['index_intermediate/finger']
    assert motion.object_clearance(model, data)[0] == pytest.approx(expected)
    # If another unresolved pair enters the execution fallback, it must use
    # the same certificate rather than discarding the mesh-face correction.
    monkeypatch.setattr(motion.sim, 'object_clearance', lambda *args: (0., 'forced fallback'))
    assert motion.object_clearance(model, data)[0] == pytest.approx(expected)


@pytest.mark.parametrize('fault', ['negative', 'contact'])
def test_rank87_certificate_never_overrides_contact_or_negative(rank87_pair, monkeypatch, fault):
    from tools import ggx_motion as motion
    model, data, obj, finger = rank87_pair
    if fault == 'contact':
        # An explicit contact record must win even if a vertex certificate
        # could prove separation. Do not depend on the faulty distance query
        # to generate that record for this guard test.
        contact = mujoco.MjContact()
        contact.geom1, contact.geom2, contact.dist = obj, finger, -.001
        mujoco.mj_addContact(model, data, contact)
        assert any(c.dist <= 0 for c in data.contact)
    distance = -.001 if fault == 'negative' else 0.
    monkeypatch.setattr(mujoco, 'mj_geomDistance', lambda *args: distance)
    monkeypatch.setattr(motion.sim, 'certified_separation',
                        lambda *args: pytest.fail('Signed overlap or contact must win'))
    assert object_clearance(model, data)[0] == distance
    assert motion.object_clearance(model, data)[0] == distance


@pytest.mark.parametrize("phase", ["start", "start_to_pregrasp", "approach"])
@pytest.mark.parametrize("body", ["index_intermediate", "gripper_attachment", "upper_arm_link"])
def test_object_penetration_checked_for_all_robot_parts_before_closure(phase, body):
    model = _object_collision_model(body)
    sample = TrajectorySample(phase, 0., np.array([0.]), 0.)
    check_trajectory(model, [sample], required_clearance_m=.002)
    assert sample.object_collision and not sample.desk_collision
    assert sample.object_clearance_m == pytest.approx(-.1)
    assert body in sample.nearest_object_geom


@pytest.mark.parametrize("phase", ["close", "lift"])
def test_object_gate_skips_closure_and_lift(phase):
    model = _object_collision_model()
    sample = TrajectorySample(phase, 1., np.array([0.]), .5)
    check_trajectory(model, [sample], required_clearance_m=.002)
    assert sample.object_clearance_m is None
    assert not sample.object_collision


def test_clear_approach_ignores_visual_only_mesh_and_rejects_touch():
    model = _object_collision_model()
    clear = TrajectorySample("approach", 0., np.array([.2]), 0.)
    touch = TrajectorySample("approach", 1., np.array([.1]), 0.)
    check_trajectory(model, [clear, touch], required_clearance_m=.002)
    assert clear.object_clearance_m == pytest.approx(.1)
    assert not clear.object_collision
    assert touch.object_collision


@pytest.mark.parametrize("collision_phase,expected", [
    ("approach", "object_collision_before_closure"), ("close", "pass"),
])
def test_candidate_gate_rejects_approach_but_not_closure(monkeypatch, collision_phase, expected):
    from tools import preflight_ggx_ur5_table as preflight
    from rh56_controller.graspgenx_baseline import GraspGenXCandidate
    model = _object_collision_model()
    samples = [TrajectorySample("start", 0., np.array([.2]), 0.),
               TrajectorySample(collision_phase, 1., np.array([0.]), 0.)]
    monkeypatch.setattr(preflight, "plan_candidate", lambda *a, **kw: (samples, 0., 0., ""))
    candidate = GraspGenXCandidate("test", .9, np.zeros(3), np.eye(3))
    result, _ = evaluate_candidate(model, samples[0].qpos, candidate, 0, parse_args([]), np.zeros(3))
    assert result.reason == expected
    assert result.passed == (expected == "pass")


def test_base_candidates_are_not_translated_by_object_center_twice() -> None:
    from rh56_controller.graspgenx_baseline import (
        GraspGenXCandidate, graspgenx_to_mujoco_base_pose,
    )
    candidate = GraspGenXCandidate(
        name="base_pose", confidence=0.9,
        position=np.array([-0.25, -0.64, 0.34]),
        rotation=np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]),
    )
    center = np.array([-0.23, -0.63, 0.127])
    rebased = candidate_for_object_origin(candidate, "ur-base", center)
    actual = graspgenx_to_mujoco_base_pose(rebased, object_center=center)
    expected = graspgenx_to_mujoco_base_pose(candidate)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, atol=1e-12)
    assert candidate_for_object_origin(candidate, "object", center) is candidate
    with pytest.raises(ValueError, match="Unknown candidate frame"):
        candidate_for_object_origin(candidate, "camera", center)


def test_proxy_yaw_preserves_center_and_local_dimensions() -> None:
    center = np.array([-0.23, -0.63, 0.127])
    model = build_model(
        XML, table_height_m=0.039, object_center=center,
        object_size_m=np.array([0.09, 0.04, 0.176]),
        object_shape="box", object_yaw_deg=90.0,
    )
    body = model.body("preflight_object")
    np.testing.assert_allclose(body.pos, center)
    np.testing.assert_allclose(body.quat, [2**-0.5, 0., 0., 2**-0.5])
    np.testing.assert_allclose(model.geom("preflight_object_visual").size,
                               [0.045, 0.02, 0.088])


def test_frame_and_yaw_defaults_preserve_legacy_inputs() -> None:
    args = parse_args([])
    assert args.candidate_frame == "object"
    assert args.object_yaw_deg == 0.0
    args = parse_args(["--candidate-frame", "ur-base", "--object-yaw-deg", "-1.4"])
    assert args.candidate_frame == "ur-base"
    assert args.object_yaw_deg == -1.4


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
