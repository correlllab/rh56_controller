"""Single write-only hand diagnostic gating and command order; no hardware."""
import json
from types import SimpleNamespace

import pytest

from tools.hardware.plan_airgap import main, send_once, shape


@pytest.mark.parametrize('args', [[], ['--confirm-empty-clear'], ['--start-snapshot','missing']])
def test_execution_requires_clearance_and_snapshot(tmp_path,args):
    out = tmp_path/'out'
    with pytest.raises(SystemExit):
        main(['--execute','--out',str(out),*args])
    assert not out.exists()


def test_one_command_low_settings_no_feedback_or_retry():
    calls=[]
    hand=SimpleNamespace(force_set=lambda x:calls.append(('force',x)),
                         speed_set=lambda x:calls.append(('speed',x)),
                         angle_set=lambda x:calls.append(('angle',x)))
    angles=[1000,1000,900,1000,900,0]
    send_once(hand,angles,lambda:calls.append(('guard',)),lambda t:calls.append(('wait',t)))
    assert calls == [('guard',),('force',[100]*6),('wait',.125),('guard',),
                     ('speed',[100]*6),('wait',.125),('guard',),('angle',angles)]


@pytest.mark.parametrize('failed_guard', [1,2,3])
def test_guard_failure_prevents_subsequent_writes(failed_guard):
    writes=[]; guards=[]
    hand=SimpleNamespace(force_set=lambda x:writes.append('force'),
                         speed_set=lambda x:writes.append('speed'),
                         angle_set=lambda x:writes.append('angle'))
    def guard():
        guards.append(1)
        if len(guards)==failed_guard:
            raise RuntimeError('arm moved')
    with pytest.raises(RuntimeError,match='arm moved'):
        send_once(hand,[1000]*6,guard,lambda t:None)
    assert writes == ['force','speed'][:failed_guard-1]


def test_offline_preview_never_opens_devices(tmp_path,monkeypatch):
    pytest.importorskip('mujoco')
    from tools import calibrate_ur5_external_camera_real as camera
    from rh56_controller.grasp_viz_core import GraspVizCore
    monkeypatch.setattr(camera,'_open_rtde_receiver',lambda *a:pytest.fail('RTDE opened'))
    monkeypatch.setattr(GraspVizCore,'_init_hand',lambda *a,**k:pytest.fail('hand opened'))
    out=tmp_path/'out'
    assert main(['--out',str(out)])==0
    report=json.loads((out/'summary.json').read_text())
    assert report['status']=='offline_preview'
    assert not report['arm_motion_commands_sent'] and not report['hand_commands_attempted']
    assert report['measured_gap_mm'] is None
    assert report['shape']['angles']==[1000,1000,900,1000,900,0]


@pytest.mark.parametrize('width', [90,110,float('nan')])
def test_width_outside_diagnostic_scope_rejected(width):
    pytest.importorskip('mujoco')
    with pytest.raises(ValueError):
        shape(width)
