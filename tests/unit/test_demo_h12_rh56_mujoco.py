from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from rh56_controller.grasp_geometry import ClosureGeometry, ClosureResult
from tools.demos.demo_h12_rh56_mujoco import parse_args


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_coupled_closure_uses_synchronized_wrist_by_default():
    assert parse_args([]).wrist_policy == "sync"


def test_fixed_wrist_policy_remains_available_for_ablation():
    assert parse_args(["--wrist-policy", "fixed"]).wrist_policy == "fixed"


def test_reachable_right_arm_smoke_pose_is_the_default():
    args = parse_args([])

    assert args.grasp_x == 0.25
    assert args.grasp_y == -0.20
    assert args.plane_rz_deg == -120.0
    assert args.grasp_center_policy == "antipodal"


def test_antipodal_center_matches_yellow_plane_marker():
    tips = {
        "thumb": np.array([0.06, 0.00, 0.10]),
        "index": np.array([-0.02, 0.03, 0.10]),
        "middle": np.array([-0.02, 0.00, 0.10]),
        "ring": np.array([-0.02, -0.03, 0.10]),
    }
    result = ClosureResult(
        mode="4-finger plane",
        midpoint=np.vstack(list(tips.values())).mean(axis=0),
        width=0.08,
        finger_span=0.06,
        cylinder_radius=0.0,
        tip_positions=tips,
        ctrl_values={},
    )

    expected = 0.5 * (
        tips["thumb"]
        + np.vstack([tips["index"], tips["middle"], tips["ring"]]).mean(axis=0)
    )
    assert np.allclose(result.grasp_center("antipodal"), expected)

    world_tips = result.world_tips(0.15, center_policy="antipodal")
    world_marker = 0.5 * (
        world_tips["thumb"]
        + np.vstack([
            world_tips["index"],
            world_tips["middle"],
            world_tips["ring"],
        ]).mean(axis=0)
    )
    assert np.allclose(world_marker, np.array([0.0, 0.0, 0.15]))


def test_open_boundary_uses_live_actuator_minima():
    closure = object.__new__(ClosureGeometry)
    closure.fk = type("FakeFK", (), {
        "ctrl_min": {"thumb_proximal": 0.1, "middle": 0.0},
    })()
    closure._joint_closure_range = lambda *_args, **_kwargs: (0.8, 0.02, 0.11)

    s, thumb_pitch, finger_ctrl = closure._solve_joint_closure("middle", 0.12)

    assert s == 0.0
    assert thumb_pitch == 0.1
    assert finger_ctrl == 0.0


def test_planner_and_h12_use_the_same_fingertip_sites():
    planner_root = ElementTree.parse(
        REPO_ROOT / "h1_mujoco/inspire/inspire_grasp_scene.xml"
    ).getroot()
    h12_hand_root = ElementTree.parse(
        REPO_ROOT / "h1_mujoco/inspire/inspire_right_ur5.xml"
    ).getroot()

    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        site_name = f"right_{finger}_tip"
        planner_site = planner_root.find(f".//site[@name='{site_name}']")
        h12_site = h12_hand_root.find(f".//site[@name='{site_name}']")
        assert planner_site is not None
        assert h12_site is not None
        assert np.allclose(
            np.fromstring(planner_site.attrib["pos"], sep=" "),
            np.fromstring(h12_site.attrib["pos"], sep=" "),
        )
