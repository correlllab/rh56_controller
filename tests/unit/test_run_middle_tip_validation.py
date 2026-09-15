"""Pointer execution authorization and no-motion paths with fake devices."""
import json
from types import SimpleNamespace
import numpy as np
import pytest

pytest.importorskip('mujoco')
pytest.importorskip('mink')
from tools import run_middle_tip_validation as run


@pytest.fixture
def setup(tmp_path, monkeypatch):
    path = tmp_path/'plan.json'
    path.write_text('{}')
    events = []
    plan = dict(config='unused', scene={}, source_sha256={},
        measurement_target={'height_mm': 120, 'desired_tip_base_m': [0, 0, .12]},
        hand_ctrl=dict(run.sim.GGX_OPEN_CTRL), hand_raw=[1000]*5+[0],
        goal_q=[0]*6, desired_tcp=[0, 0, .2, 0, 0, 0], reference_tip_in_hand_m=[0, 0, .2])
    monkeypatch.setattr(run, 'load_plan', lambda p: plan)
    monkeypatch.setattr(run.real, 'load_config', lambda p: {})
    monkeypatch.setattr(run.support, 'build_model', lambda *a, **k: object())
    monkeypatch.setattr(run.real, 'check_tcp_offset', lambda *a: None)
    monkeypatch.setattr(run.review, 'show', lambda *a: events.append('video'))
    monkeypatch.setattr(run.motion, 'preflight_goals', lambda *a: [run.pointer_goal(plan)])
    def render(model, commands, plan, config, out):
        video = out/'review.mp4'
        video.write_bytes(b'fake video')
        return video, 29.
    monkeypatch.setattr(run, 'render_checked', render)
    class Fake:
        def __init__(self, *args):
            events.append('construct')
            self.control = object()
        def open_receiver(self): events.append('receive')
        def snapshot(self): return dict(q=[0]*6, tcp=[0, 0, .2, 0, 0, 0])
        def open_control(self): events.append('control')
        def open_hand(self, command): events.append('hand')
        def move_phase(self, command): events.append('move')
        def close(self): events.append('close')
    monkeypatch.setattr(run.real, 'LiveBackend', Fake)
    return SimpleNamespace(path=path, events=events, out=tmp_path/'out')


def test_default_never_constructs_devices(setup):
    assert run.main(['--plan', str(setup.path), '--out', str(setup.out)]) == 0
    assert setup.events == []


def test_hardware_requires_controller_permission(setup):
    with pytest.raises(SystemExit):
        run.main(['--plan', str(setup.path), '--execute'])
    assert setup.events == []


def test_controller_only_cannot_open_or_move(setup):
    assert run.main(['--plan', str(setup.path), '--out', str(setup.out), '--controller-preflight', '--confirm-controller']) == 0
    assert setup.events == ['construct', 'receive', 'control', 'close']


@pytest.mark.parametrize('answers,expected', [([False], []), ([True, False], ['hand']), ([True, True], ['hand', 'move'])])
def test_motion_requires_both_confirmations(setup, monkeypatch, answers, expected):
    responses = iter(answers)
    monkeypatch.setattr(run.real, 'confirm', lambda prompt: next(responses))
    assert run.main(['--plan', str(setup.path), '--out', str(setup.out), '--execute', '--confirm-controller']) == 0
    assert [event for event in setup.events if event in ('hand', 'move')] == expected
    report = json.loads((setup.out/'summary.json').read_text())
    assert report['physical_point_error_mm'] is None
