"""Known sugar-box input adapter; original movement implementation stays shared."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from tools import run_sugar_box_trial as trial
from tools import run_vision_pick_cycle as cycle


@pytest.fixture
def c():
    return dict(target_bounds_mm=[[-300,-750,60],[-100,-500,120]],object_width_mm=90,
                width_offset_mm=10,closure_margin_mm=2,grasp_depth_mm=10,yaw_deg=90)


def evidence():
    return dict(schema='rh56_tabletop_foreground_preview/v2',executable_target=False,
                geometry_candidate_count=1,candidates=[dict(geometry_candidate=True,rejection_reasons=[],
                top_patch=dict(long_mm=176,short_mm=90,height_median_mm=43,
                               core_spatial_height_mad_mm=1.5,center_base_mm=[-224,-638,83],
                               short_axis_yaw_deg=88,yaw_ambiguous=False))])


def test_known_width_not_noise_driven_internal_input(c):
    r=evidence();r['candidates'][0]['top_patch']['short_mm']=91.5
    target,axis,_=trial.select_target(r,c,'source')
    assert target.planner_width_m==pytest.approx(.098)
    np.testing.assert_allclose(target.center_base_m,[-.224,-.638,.073])
    assert axis==88


@pytest.mark.parametrize('problem',['two','rejected','short','height','center','yaw','noisy','nan','schema'])
def test_bad_object_not_promoted(c,problem):
    r=evidence();p=r['candidates'][0]['top_patch']
    if problem=='two': r['candidates']*=2;r['geometry_candidate_count']=2
    if problem=='rejected': r['candidates'][0]['rejection_reasons']=['touches_roi_or_exclusion_boundary']
    if problem=='short': p['short_mm']=40
    if problem=='height': p['height_median_mm']=80
    if problem=='center': p['center_base_mm']=[-500,-638,83]
    if problem=='yaw': p['yaw_ambiguous']=True
    if problem=='noisy': p['core_spatial_height_mad_mm']=4
    if problem=='nan': p['long_mm']=float('nan')
    if problem=='schema': r['schema']='unknown'
    with pytest.raises((ValueError,RuntimeError)):
        trial.select_target(r,c,'source')


@pytest.mark.parametrize('change',['center','dimensions','yaw'])
def test_temporal_changes_block_grasp(c,change):
    a=trial.select_target(evidence(),c,'a');b=trial.select_target(evidence(),c,'b')
    if change=='center': b[2]['center_base_mm'][0]+=4
    if change=='dimensions': b[2]['long_mm']+=4
    if change=='yaw': b=(b[0],b[1]+3,b[2])
    with pytest.raises(ValueError,match='changed'):
        trial.check_consistency([a,b])


def test_no_new_motion_controller():
    for name in ('open_hand','preflight_pick','grasp','settle','lift','close'):
        assert getattr(trial.SugarCycle,name) is getattr(cycle.LiveCycle,name)


@pytest.mark.parametrize('extra',[['--execute','--capture','old'],['--preflight-only','--capture','old'],[]])
def test_cli_rejects_bad_modes(extra):
    with pytest.raises(SystemExit):
        trial.parse_args(['--out','unused',*extra])


def test_cli_default_is_offline():
    a=trial.parse_args(['--out','unused','--capture','saved'])
    assert not a.execute and not a.preflight_only


def test_camera_only_cli_never_enables_execution():
    a=trial.parse_args(['--out','unused','--camera-check-only'])
    assert a.camera_check_only and not a.execute and not a.preflight_only
    for extra in (['--execute'],['--preflight-only'],['--capture','old']):
        with pytest.raises(SystemExit):
            trial.parse_args(['--out','unused','--camera-check-only',*extra])


def test_reset_only_requires_execute():
    with pytest.raises(SystemExit):trial.parse_args(['--out','unused','--reset-only'])
    a=trial.parse_args(['--out','unused','--execute','--reset-only'])
    assert a.reset_only and a.execute


@pytest.mark.parametrize('failure',[None,'prepare','open_hand','move_scan'])
def test_reset_never_detects_or_grasps(monkeypatch,failure):
    calls=[]
    def step(name):
        def action(_):
            calls.append(name)
            if name==failure:raise RuntimeError('failed')
        return action
    live=SimpleNamespace(**{name:step(name) for name in ('prepare','open_hand','move_scan')},
                         bridge=SimpleNamespace(_iface=SimpleNamespace(recv=None)))
    monkeypatch.setattr(cycle,'stationary_snapshot',lambda r:{'test':'stationary'})
    report={'events':[]}
    if failure:
        with pytest.raises(RuntimeError):trial.run_reset_only(live,report,lambda:None)
        assert calls==['prepare','open_hand','move_scan'][:['prepare','open_hand','move_scan'].index(failure)+1]
    else:
        trial.run_reset_only(live,report,lambda:None)
        assert calls==['prepare','open_hand','move_scan']
        assert report['status']=='reset_completed_no_grasp'


@pytest.mark.parametrize('fault',[None,'rejected','capture_error','changed'])
def test_fixed_camera_burst_no_robot_no_retries(tmp_path,monkeypatch,c,fault):
    from tools import calibrate_ur5_external_camera_real as camera
    from tools.capture_tabletop_background import TABLETOP_WARMUP_FRAMES
    events=[]
    def start(a):
        assert a.warmup_frames==TABLETOP_WARMUP_FRAMES==60 and a.settle_frames==5
        assert a.laser_power is None
        return SimpleNamespace(stop=lambda:events.append('stop')),None,None
    def read(*a):
        events.append('read')
        if fault=='capture_error':raise RuntimeError('camera failed')
        return None,None,1000+len(events),len(events)
    monkeypatch.setattr(camera,'_start_camera',start)
    monkeypatch.setattr(camera,'_read_aligned_frame',read)
    monkeypatch.setattr(camera,'_camera_metadata_to_dict',lambda m:{})
    monkeypatch.setattr(camera,'_save_rgb',lambda *a:None)
    monkeypatch.setattr(camera,'_save_depth',lambda *a:None)
    def no_robot(*a):raise AssertionError('Camera-only mode contacted robot')
    monkeypatch.setattr(cycle,'stationary_snapshot',no_robot)
    c.update(camera_serial='test',frames=3,max_center_spread_mm=3)
    live=trial.SugarCycle(c,SimpleNamespace(out=tmp_path,camera_check_only=True,execute=False),{})
    def observe(capture,out):
        assert events[:4]==['read','read','read','stop']
        assert len(list(tmp_path.glob('capture_*/camera_check.yaml')))==3
        events.append('observe')
        if fault=='rejected':raise ValueError('rejected frame')
        r=evidence()
        if fault=='changed':r['candidates'][0]['top_patch']['center_base_mm'][0]+=5*events.count('observe')
        return trial.select_target(r,c,out/'summary.json')
    live.saved_observation=observe
    if fault:
        with pytest.raises((ValueError,RuntimeError)):live.detect(None)
    else:
        live.detect(None)
        assert live.report['observation_consistency']['center_spread_mm']==0
    assert events.count('read')==(1 if fault=='capture_error' else 3)
    assert events.count('stop')==1
    assert live.bridge is None and live.hand is None


def test_live_detection_still_checks_robot_stationarity(tmp_path,monkeypatch,c):
    from tools import calibrate_ur5_external_camera_real as camera
    stopped=[]
    monkeypatch.setattr(camera,'_start_camera',lambda a:(SimpleNamespace(stop=lambda:stopped.append(True)),None,None))
    def moving(*a):raise RuntimeError('Robot moving')
    monkeypatch.setattr(cycle,'stationary_snapshot',moving)
    c.update(camera_serial='test',frames=3)
    live=trial.SugarCycle(c,SimpleNamespace(out=tmp_path,camera_check_only=False,execute=True),{})
    live.bridge=SimpleNamespace(_iface=SimpleNamespace(recv=object()))
    with pytest.raises(RuntimeError,match='Robot moving'):live.detect(None)
    assert stopped==[True]


@pytest.fixture
def joint_live(tmp_path):
    c=cycle.load_config(cycle.ROOT/'configs/ur5_vision_pick_cycle.example.yaml')
    live=trial.SugarCycle(c,SimpleNamespace(out=tmp_path,execute=True),
                         {'confirmation_method':'interactive_enter','stage':'move_scan'})
    live.scan_q=[.1,0,0,0,0,0]
    live.scan=[-.2,-.39,.35,0,0,0]
    live.reset_tcp=None
    recv=SimpleNamespace(getRobotMode=lambda:7,getSafetyMode=lambda:1,
        isProtectiveStopped=lambda:False,getActualQ=lambda:live.scan_q,
        getActualTCPPose=lambda:live.scan,getActualQd=lambda:[0]*6,
        getActualTCPSpeed=lambda:[0]*6)
    ctrl=SimpleNamespace(getTCPOffset=lambda:[0]*6,
        isJointsWithinSafetyLimits=lambda q:True,isPoseWithinSafetyLimits=lambda p:True,
        getForwardKinematics=lambda q,tcp:[-.2,-.4+q[0]*.1,.35,0,0,0])
    live.bridge=SimpleNamespace(_iface=SimpleNamespace(ctrl=ctrl,recv=recv))
    return live


def test_joint_scan_uses_controller_fk_no_inverse_or_motion(joint_live):
    r=joint_live.check_joint_scan([0]*6,[-.2,-.4,.35,0,0,0])
    assert r['status']=='passed' and not r['collision_free_verified']
    assert len(r['samples'])==11
    assert r['speed_rad_s']==.1
    np.testing.assert_allclose(r['samples'][-1]['q'],joint_live.scan_q)


@pytest.mark.parametrize('fault',['joint_safety','pose_safety','bounds','endpoint','tcp_offset','state','nan','distance'])
def test_joint_scan_blocks_bad_paths(joint_live,fault):
    ctrl=joint_live.bridge._iface.ctrl
    if fault=='joint_safety': ctrl.isJointsWithinSafetyLimits=lambda q:False
    if fault=='pose_safety': ctrl.isPoseWithinSafetyLimits=lambda p:False
    if fault=='bounds': ctrl.getForwardKinematics=lambda *a:[0,0,-1,0,0,0]
    if fault=='endpoint': joint_live.scan[2]+=.01
    if fault=='tcp_offset': ctrl.getTCPOffset=lambda:[0,0,.01,0,0,0]
    if fault=='state': joint_live.bridge._iface.recv.getSafetyMode=lambda:3
    if fault=='nan': joint_live.scan_q[0]=float('nan')
    if fault=='distance': joint_live.scan[0]+=1
    with pytest.raises((ValueError,RuntimeError)):
        joint_live.check_joint_scan([0]*6,[-.2,-.4,.35,0,0,0])


@pytest.mark.parametrize('confirmed',[False,True])
def test_joint_motion_explicit_and_one_async_move(joint_live,confirmed):
    calls=[]
    ctrl=joint_live.bridge._iface.ctrl
    ctrl.moveJ=lambda *a:calls.append(a) or True
    joint_live.wait_stationary=lambda s:None
    if not confirmed:
        joint_live.report.pop('confirmation_method')
        with pytest.raises(RuntimeError,match='confirmation'):joint_live.move_scan(None)
        assert not calls
    else:
        joint_live.move_scan(None)
        assert calls==[(joint_live.scan_q,.1,.2,True)]
        assert joint_live.report['joint_scan_completed']


def test_rejected_joint_move_stops_joints_never_retries(joint_live):
    calls=[]
    ctrl=joint_live.bridge._iface.ctrl
    ctrl.moveJ=lambda *a:calls.append('moveJ') or False
    ctrl.stopJ=lambda a:calls.append('stopJ')
    with pytest.raises(RuntimeError,match='rejected'):joint_live.move_scan(None)
    assert calls==['moveJ','stopJ'] and joint_live.cancelled.is_set()


def test_pick_preflight_still_delegates_unchanged(joint_live,monkeypatch):
    calls=[]
    monkeypatch.setattr(cycle.LiveCycle,'preflight',lambda self,*a,**kw:calls.append((a,kw)))
    joint_live.preflight([[1]*6],[50])
    assert calls==[(([[1]*6],[50]),{'start_snapshot':None})]


def test_adapter_matches_old_plan_for_identical_input(tmp_path,monkeypatch):
    pytest.importorskip('mujoco')
    from tools import run_vision_plan_trial as helpers
    from rh56_controller.vision_grasp_target import load_vision_target
    monkeypatch.setattr('rh56_controller.grasp_geometry._CACHE_PATH',str(tmp_path/'fk.npz'))
    monkeypatch.setattr(helpers,'load_local_alignment',lambda *a:(np.eye(4),{'test':'same mapping'}))
    source=tmp_path/'target.yaml'
    source.write_text(yaml.safe_dump(dict(schema='rh56_ur5_camera_colored_target_validation/v1',
        quality_pass=True,estimated_target_top_center_base_mm=[-224,-638,83],physical_reference=dict(target_height_mm=40))))
    validation=tmp_path/'alignment.yaml';validation.write_text(yaml.safe_dump(dict(hand_command_assumed=[1000]*6)))
    c=cycle.load_config(cycle.ROOT/'configs/ur5_vision_pick_cycle.example.yaml')
    c.update(alignment_validation=str(validation),local_alignment=str(tmp_path/'fake'))
    # Identity test mount is intentionally not realistic; widen test-only TCP envelope.
    c['tcp_bounds_mm']=[[-2000]*3,[2000]*3]
    target=load_vision_target(source,object_width_mm=c['object_width_mm'],yaw_deg=c['yaw_deg'],
        width_offset_mm=c['width_offset_mm']-c['closure_margin_mm'],depth_below_top_mm=c['grasp_depth_mm'])
    old=cycle.Plan(c,source)
    try:
        new=trial.TabletopPlan(c,target)
        try:
            assert old.commands==new.commands
            np.testing.assert_allclose(old.tcp,new.tcp,rtol=0,atol=1e-12)
            np.testing.assert_allclose(old.lift,new.lift,rtol=0,atol=1e-12)
        finally:new.close()
    finally:old.close()
