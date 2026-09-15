"""Offline-only checks for the small, explicitly authorized grip adjustment."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from tools import tighten_vision_pick_grip as grip


def test_default_is_preview():
    args = grip.parse_args(['--config','c','--checkpoint','s','--out','o'])
    assert not args.execute


@pytest.mark.parametrize('missing', ['--confirm-adjust', '--confirm-high-force',
                                    '--acknowledge-open-loop', '--acknowledge-shape-extrapolation'])
def test_execution_needs_every_confirmation(missing):
    options = {'--confirm-adjust':[grip.CONFIRMATION], '--confirm-high-force':[],
               '--acknowledge-open-loop':[], '--acknowledge-shape-extrapolation':[]}
    args = ['--config','c','--checkpoint','s','--out','o','--execute']
    for key, values in options.items():
        if key != missing:
            args += [key, *values]
    with pytest.raises(SystemExit):
        grip.parse_args(args)


@pytest.fixture
def checkpoint(tmp_path):
    c = dict(closure_margin_mm=2, hand_force_raw=1000, speed_profile='planner')
    hashes = {}
    for key in ('calibration', 'desk_plane', 'local_alignment', 'alignment_validation'):
        p = tmp_path/key
        p.write_text(key)
        c[key] = str(p)
        hashes[key] = hashlib.sha256(p.read_bytes()).hexdigest()
    previous = dict(schema='rh56_vision_pick_cycle/v1', status='failed', stage='lift',
        cleanup_errors=[], config={**c,'closure_margin_mm':1,'hand_force_raw':100},
        events=[dict(stage='grasp',event='complete'),dict(stage='settle',event='complete')],
        source_sha256=hashes)
    return c, previous


@pytest.mark.parametrize('problem', [None,'stage','completed_lift','incomplete','source','speed','margin','cleanup'])
def test_checkpoint_rejects_other_operations_and_changed_context(checkpoint, problem):
    c, previous = checkpoint
    if problem == 'stage':
        previous['stage'] = 'grasp'
    elif problem == 'completed_lift':
        previous['events'].append(dict(stage='lift',event='complete'))
    elif problem == 'incomplete':
        previous['events'] = []
    elif problem == 'source':
        previous['source_sha256'] = {}
    elif problem == 'speed':
        c['speed_profile'] = 'slow'
    elif problem == 'margin':
        c['closure_margin_mm'] = 3
    elif problem == 'cleanup':
        previous['cleanup_errors'] = ['stop failed']
    if problem:
        with pytest.raises(ValueError):
            grip.validate_checkpoint(c, previous)
    else:
        grip.validate_checkpoint(c, previous)


@pytest.mark.parametrize('problem', [None,'position','rotation','joints'])
def test_current_arm_must_match_completed_anchor(problem):
    tcp = np.array([-.2,-.46,.22,0,0,0])
    state = dict(tcp=tcp.copy(),q=np.zeros(6))
    if problem == 'position':
        state['tcp'][2] += .002
    elif problem == 'rotation':
        state['tcp'][3] += np.deg2rad(1)
    elif problem == 'joints':
        state['q'][0] += .02
    if problem:
        with pytest.raises(RuntimeError, match='anchor'):
            grip.check_anchor(state, tcp, np.zeros(6))
    else:
        grip.check_anchor(state, tcp, np.zeros(6))


@pytest.mark.parametrize('problem', [None,'opening','distance','rotation','large_command','yaw'])
def test_only_small_closing_plan_delta_allowed(problem):
    a = np.array([-.2,-.46,.22,0,0,0])
    b = a+np.array([0,.0016,-.0023,0,.01,0])
    old = [1000,708,678,722,678,0]
    new = [1000,703,673,717,673,0]
    if problem == 'opening':
        new[1] = 709
    elif problem == 'distance':
        b[2] -= .005
    elif problem == 'rotation':
        b[4] = .1
    elif problem == 'large_command':
        new[1] = 690
    elif problem == 'yaw':
        old[5] = 1
    if problem:
        with pytest.raises(ValueError, match='bounded'):
            grip.validate_delta(a,b,old,new)
    else:
        delta, angle = grip.validate_delta(a,b,old,new)
        assert np.linalg.norm(delta) < 4 and angle < 1


@pytest.mark.parametrize('fail_at', [None,'force','wrist','angle'])
def test_adjustment_never_opens_lifts_retries_or_advances_after_failure(fail_at):
    calls = []
    def action(name, value):
        calls.append((name,value))
        if name == fail_at:
            raise OSError(name)
    live = SimpleNamespace(report={},
        hand=SimpleNamespace(force_set=lambda v:action('force',v),angle_set=lambda v:action('angle',v)),
        move_tcp=lambda v:action('wrist',v),wait_stationary=lambda v:action('wait',v))
    target, command = [1]*6, [1000,703,673,717,673,0]
    expected = [('force',[1000]*6),('wait',.5),('wrist',target),('angle',command),('wait',.5)]
    if fail_at:
        with pytest.raises(OSError):
            grip.apply_adjustment(live,target,command)
        assert calls == expected[:next(i for i,v in enumerate(expected) if v[0] == fail_at)+1]
    else:
        grip.apply_adjustment(live,target,command)
        assert calls == expected
        assert live.report['angle_write_completed']
