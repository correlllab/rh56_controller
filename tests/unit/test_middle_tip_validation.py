"""Offline pointer-frame regression checks, without camera/robot dependencies."""
import numpy as np
import pytest

pytest.importorskip('mujoco')
pytest.importorskip('mink')
from tools import prepare_middle_tip_validation as pointer


def test_level_pointer_uses_existing_tip_and_mount():
    model = pointer.support.cb3_spec().compile()
    qpos = pointer.sim.initial_qpos(model, np.radians([0, -90, 90, -90, 0, 0]))
    pointer.sim.set_hand_qpos(model, qpos, pointer.sim.GGX_OPEN_CTRL)
    target = np.array([-.24, -.68, .16])
    hand, tcp, eeff, local = pointer.level_hand_target(model, qpos, target, [0, -1])
    np.testing.assert_allclose(hand[:3, 0], [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(hand[:3, 2], [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(hand[:3, :3] @ local + hand[:3, 3], target, atol=1e-12)
    np.testing.assert_allclose(pointer.UR5Bridge.tcp_T_to_hand_T(tcp), hand, atol=1e-12)
    np.testing.assert_allclose(eeff[:3, 3], hand[:3, 3]+hand[:3, :3]@pointer.sim.EEFF_LOCAL_M, atol=1e-12)


def test_height_change_does_not_change_xy_or_hand_model():
    model = pointer.support.cb3_spec().compile()
    qpos = pointer.sim.initial_qpos(model, np.radians([0, -90, 90, -90, 0, 0]))
    pointer.sim.set_hand_qpos(model, qpos, pointer.sim.GGX_OPEN_CTRL)
    first = pointer.level_hand_target(model, qpos, [-.24, -.68, .06], [0, -1])
    second = pointer.level_hand_target(model, qpos, [-.24, -.68, .16], [0, -1])
    np.testing.assert_allclose(second[1][:3, 3]-first[1][:3, 3], [0, 0, .1], atol=1e-12)
    np.testing.assert_allclose(first[3], second[3], atol=1e-12)
