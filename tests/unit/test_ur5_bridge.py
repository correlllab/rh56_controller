"""Pure transform tests for the UR5-to-RH56 mounting geometry."""

import numpy as np

from rh56_controller.ur5_bridge import (
    UR5Bridge,
    _TCP_TO_HAND_POS,
    _TCP_TO_HAND_QUAT,
    _T_HAND_IN_TCP,
)


def test_zero_tcp_uses_flange_to_hand_distance_not_wrist3_distance():
    """RTDE zero TCP is already at the flange, so UR5e d6 is not repeated."""

    np.testing.assert_allclose(_TCP_TO_HAND_POS, [0.0, 0.0, 0.0564], atol=1e-12)
    assert not np.isclose(_TCP_TO_HAND_POS[2], 0.156)


def test_tcp_to_hand_rotation_is_normalized_rz_90():
    np.testing.assert_allclose(np.linalg.norm(_TCP_TO_HAND_QUAT), 1.0, atol=1e-7)
    np.testing.assert_allclose(
        _T_HAND_IN_TCP[:3, :3],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-7,
    )


def test_hand_tcp_transform_round_trip():
    world_from_hand = np.array(
        [
            [0.0, 1.0, 0.0, -0.2390036252],
            [0.4595974287, 0.0, -0.8881273577, -0.4846433727],
            [-0.8881273577, 0.0, -0.4595974287, 0.1888288549],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    world_from_tcp = UR5Bridge.hand_T_to_tcp_T(world_from_hand)
    reconstructed = UR5Bridge.tcp_T_to_hand_T(world_from_tcp)

    np.testing.assert_allclose(reconstructed, world_from_hand, atol=1e-10)


def test_corrected_grasp_tcp_is_100_mm_beyond_old_tcp_along_tool_axis():
    """Regression for the 2026-09-04 real-camera pregrasp diagnosis."""

    world_from_hand = np.array(
        [
            [0.0, 1.0, 0.0, -0.23900362521113576],
            [0.459597428697, 0.0, -0.888127357722, -0.4846433726786529],
            [-0.888127357722, 0.0, -0.459597428697, 0.18882885494608985],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    corrected_tcp = UR5Bridge.hand_T_to_tcp_T(world_from_hand)
    old_tcp_position = np.array(
        [-0.23900362521113576, -0.34609550487395495, 0.2605260538228378]
    )

    np.testing.assert_allclose(
        np.linalg.norm(corrected_tcp[:3, 3] - old_tcp_position),
        0.0996,
        atol=2e-7,
    )
