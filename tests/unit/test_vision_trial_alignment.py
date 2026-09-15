"""Offline checks for explicit local frame adaptation; no hardware access."""
import hashlib
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools.run_vision_plan_trial import command_hand_pose, inspection_hand_pose, load_local_alignment, plan_prefix, resume_plan_window, validate_resume_checkpoint, validate_rigid_transform
from rh56_controller.ur5_bridge import UR5Bridge


@pytest.mark.parametrize('value,execute,expected', [('-1', False, '[0, 1000]'),
    ('1001', False, '[0, 1000]'), ('1000', True, '--confirm-high-force')])
def test_force_limit_range_and_explicit_confirmation_before_hardware(monkeypatch, capsys, value, execute, expected):
    from tools.run_vision_plan_trial import main
    argv = ['run_vision_plan_trial.py', '--vision-target', 'unused.yaml', '--out', 'unused',
            '--hand-force-limit-raw', value]
    if execute:
        argv += ['--execute', '--confirm-plan', '--start-snapshot', 'unused.json']
    monkeypatch.setattr('sys.argv', argv)
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert expected in capsys.readouterr().err


def test_plan_prefix_retains_original_steps_without_mutation():
    closures = [object() for _ in range(9)]
    before = closures.copy()
    selected = plan_prefix(closures, 1)
    assert selected == before[:2]
    assert all(a is b for a, b in zip(selected, before))
    assert closures == before
    assert plan_prefix(closures, None) is closures
    assert plan_prefix(closures, 8) == closures


@pytest.mark.parametrize('bad', [0, -1, 9, 1.5, True])
def test_invalid_plan_prefix_rejected(bad):
    with pytest.raises(ValueError, match='stop-after-step'):
        plan_prefix([object() for _ in range(9)], bad)


def test_resume_window_uses_completed_shape_as_anchor_not_original_open():
    closures = [object() for _ in range(9)]
    assert resume_plan_window(closures, 1, 2) == closures[1:3]
    assert resume_plan_window(closures, 6, 8) == closures[6:9]


def test_resumed_existing_executor_reasserts_hold_then_arm_then_hand(monkeypatch):
    from types import SimpleNamespace
    from rh56_controller.grasp_executor import GraspExecutor
    events = []
    arm = SimpleNamespace(move_to_hand_pose=lambda T, **kwargs: events.append(('arm', T)) or [])
    hand = SimpleNamespace(angle_set=lambda cmd: events.append(('hand', cmd.copy())))
    executor = GraspExecutor(arm, hand, None, status_cb=lambda _: None, hand_write_only=True)
    executor._ctrl_to_real = lambda ctrl: ctrl.copy()
    executor._report_eef_error = lambda T, **kwargs: (T, 0., 0.)
    monkeypatch.setattr('rh56_controller.grasp_executor.time.sleep', lambda seconds: events.append(('sleep', seconds)))
    commands = [[1000]*5+[0], [1000,1000,960,1000,961,0], [1000,1000,922,1000,922,0]]
    waypoints = [(i, SimpleNamespace(ctrl_values=cmd, width=.12-i*.005)) for i,cmd in enumerate(commands)]
    executor._run_plan_waypoints(resume_plan_window(waypoints, 1, 2), 0., True, None, None)
    assert events == [('hand', commands[1]), ('sleep', .3), ('arm', 1), ('arm', 2), ('hand', commands[2])]


@pytest.mark.parametrize('resume,stop', [(0, 2), (8, 8), (1, 1), (2, 1), (1, None), (True, 2), (1, 9)])
def test_invalid_resume_window_rejected(resume, stop):
    with pytest.raises(ValueError):
        resume_plan_window([object() for _ in range(9)], resume, stop)


def test_resume_checkpoint_requires_same_command_and_only_upward_z():
    from copy import deepcopy
    previous = dict(schema='rh56_existing_plan_trial/v1', status='commands_completed',
        executed_last_plan_step=1, full_closure_steps=8,
        input=dict(source='same.yaml', object_width_mm=40, width_offset_mm=20,
                   yaw_deg=90, center_policy='antipodal', center_base_m=[-.2, -.6, .06]),
        hand_commands=[[1000, 1000, 960, 1000, 961, 0]],
        waypoints=[[-.2, -.5, .27, 0., -2.9, .9]], local_alignment=dict(sha256='frozen'))
    current = deepcopy(previous)
    current['waypoints'][0][2] += .010
    current['input']['center_base_m'][2] += .010
    assert validate_resume_checkpoint(previous, current, 1) == pytest.approx(.010)
    for key, value in [('status', 'failed'), ('executed_last_plan_step', 2),
                       ('hand_commands', [[1000]*6]), ('local_alignment', dict(sha256='different'))]:
        bad = deepcopy(previous); bad[key] = value
        with pytest.raises(ValueError):
            validate_resume_checkpoint(bad, current, 1)
    for idx, delta in [(0, .001), (1, .001), (2, -.02), (2, .02), (3, .01)]:
        bad = deepcopy(current); bad['waypoints'][0][idx] += delta
        with pytest.raises(ValueError):
            validate_resume_checkpoint(previous, bad, 1)


def test_alignment_frame_adapter_preserves_desired_physical_pose():
    nominal = UR5Bridge.tcp_T_to_hand_T(np.eye(4))
    correction = np.eye(4)
    correction[:3, :3] = Rotation.from_euler('xyz', [2, -4, 3], degrees=True).as_matrix()
    correction[:3, 3] = [.002, -.005, .001]
    mount = correction @ nominal
    desired = np.eye(4)
    desired[:3, :3] = Rotation.from_euler('xyz', [30, 20, 90], degrees=True).as_matrix()
    desired[:3, 3] = [-.2, -.5, .2]
    before = desired.copy()
    command = command_hand_pose(desired, mount)
    tcp = UR5Bridge.hand_T_to_tcp_T(command)
    np.testing.assert_allclose(tcp @ mount, desired, atol=1e-6)
    np.testing.assert_array_equal(desired, before)
    np.testing.assert_array_equal(command_hand_pose(desired, None), desired)


def test_hover_is_base_z_only_and_does_not_mutate_original():
    desired = np.eye(4)
    desired[:3, :3] = Rotation.from_euler('xyz', [50, 20, 90], degrees=True).as_matrix()
    desired[:3, 3] = [-.2, -.5, .2]
    before = desired.copy()
    raised = inspection_hand_pose(desired, 50)
    np.testing.assert_array_equal(desired, before)
    np.testing.assert_array_equal(raised[:3, :3], before[:3, :3])
    np.testing.assert_allclose(raised[:3, 3]-before[:3, 3], [0, 0, .05])


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1, 101])
def test_invalid_hover_rejected(bad):
    with pytest.raises(ValueError, match='hover'):
        inspection_hand_pose(np.eye(4), bad)


@pytest.mark.parametrize('bad', [np.eye(3), np.full((4, 4), np.nan), np.diag([-1., 1., 1., 1.]), np.diag([2., 1., 1., 1.])])
def test_nonrigid_alignment_rejected(bad):
    with pytest.raises(ValueError):
        validate_rigid_transform(bad)


def test_frozen_alignment_requires_hash_shape_and_validation(tmp_path):
    nominal = UR5Bridge.tcp_T_to_hand_T(np.eye(4))
    candidate = {'schema': 'rh56_two_finger_local_alignment/v1',
                 'correction_left_multiply_in_tcp': np.eye(4).tolist(),
                 'candidate_tcp_from_hand_current_shape_only': nominal.tolist()}
    path = tmp_path/'candidate.json'; path.write_text(json.dumps(candidate))
    command = [1000, 762, 727, 778, 727, 0]
    validation = {'schema': 'rh56_frozen_two_finger_holdout/v1',
                  'candidate_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'candidate_unchanged': True, 'refit_performed': False,
                  'both_fingers_under_5mm_proxy_gate': True,
                  'hand_command_assumed': command, 'fixed_correction': np.eye(4).tolist()}
    check = tmp_path/'validation.json'; check.write_text(json.dumps(validation))
    mount, metadata = load_local_alignment(path, check, command)
    np.testing.assert_allclose(mount, nominal)
    with pytest.raises(ValueError, match='final hand command'):
        load_local_alignment(path, check, [1000]*6)
    validation['candidate_sha256'] = 'wrong'; check.write_text(json.dumps(validation))
    with pytest.raises(ValueError, match='frozen holdout'):
        load_local_alignment(path, check, command)
