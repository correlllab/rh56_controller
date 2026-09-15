"""Original write-only reflex sequencing and offline-only matched preview."""
from types import SimpleNamespace
import numpy as np
import pytest


def test_round_object_axis_is_explicit_and_preserves_measured_frame():
    from copy import deepcopy
    from tools.prepare_reflex_comparison import planning_frame
    frame=dict(center_base_mm=[-215.,-712.,40.],short_axis_yaw_deg=-45.7,yaw_ambiguous=True)
    before=deepcopy(frame)
    rotated=planning_frame(frame,90.)
    assert frame==before
    assert rotated==dict(frame,short_axis_yaw_deg=90.)
    assert planning_frame(frame,None)==frame
    with pytest.raises(ValueError,match='ambiguous'):
        planning_frame(dict(frame,yaw_ambiguous=False),90.)


def test_explicit_round_approximation_preserves_measured_axis_metadata():
    from tools.prepare_reflex_comparison import planning_frame,parse_args
    from tools.run_reflex_ten_starts import parse_args as runner_args
    frame=dict(center_base_mm=[-224.,-699.,40.],short_axis_yaw_deg=-70.15,yaw_ambiguous=False)
    rotated=planning_frame(frame,90.,round_object_approximation=True)
    assert rotated==dict(frame,short_axis_yaw_deg=90.)
    assert frame['short_axis_yaw_deg']==-70.15 and not rotated['yaw_ambiguous']
    with pytest.raises(ValueError,match='explicit yaw'):
        planning_frame(frame,None,round_object_approximation=True)
    for parser,extra in [(parse_args,[]),(runner_args,['--object-label','raspberry'])]:
        with pytest.raises(SystemExit):
            parser(['--crop-report','raspberry.json','--round-object-approximation',*extra])
        args=parser(['--crop-report','raspberry.json','--round-object-approximation',
                     '--round-object-yaw-deg','90',*extra])
        assert args.round_object_approximation


@pytest.mark.parametrize('value',['nan','inf','181','-181'])
def test_invalid_round_object_yaw_cli(value):
    from tools.prepare_reflex_comparison import parse_args
    from tools.run_reflex_ten_starts import parse_args as runner_args
    with pytest.raises(SystemExit):
        parse_args(['--crop-report','apple.json','--round-object-yaw-deg',value])
    with pytest.raises(SystemExit):
        runner_args(['--crop-report','apple.json','--object-label','apple','--round-object-yaw-deg',value])


@pytest.mark.parametrize('fingers',[2,4])
def test_round_yaw_rotates_target_not_closure_or_center(monkeypatch,tmp_path,fingers):
    from tools.prepare_reflex_comparison import planning_frame,build_reflex_target
    monkeypatch.setenv('RH56_CACHE_DIR',str(tmp_path/'cache'))
    frame=dict(center_base_mm=[-215.,-712.,40.],short_axis_yaw_deg=-45.7,yaw_ambiguous=True)
    bounds,depth=([26.,35.,27.],10.) if fingers==2 else ([73.,75.,74.],35.)
    old=build_reflex_target(frame,bounds,depth,'object.npy',fingers=fingers,grasp_direction='legacy-top')
    new=build_reflex_target(planning_frame(frame,90.),bounds,depth,'object.npy',fingers=fingers,grasp_direction='legacy-top')
    for key in ['grasp_center_base_mm','measured_width_mm','measured_height_mm','thumb_raw','final_raw','thumb_ctrl','final_ctrl']:
        assert new[key]==old[key]
    tips={k:np.array(v) for k,v in new['active_tip_positions_base_m'].items()}
    axis=tips['thumb']-np.mean([v for k,v in tips.items() if k!='thumb'],axis=0)
    assert np.degrees(np.arctan2(axis[1],axis[0]))==pytest.approx(90.)
    assert new['final_hand_T']!=old['final_hand_T']
from rh56_controller.grasp_executor import GraspExecutor
from tools.prepare_reflex_comparison import parse_args


def test_default_restored_to_original_four_finger_reflex():
    args=parse_args(['--crop-report','crop.json'])
    assert (args.fingers,args.grasp_direction,args.grasp_depth_mm)==(4,'legacy-top',10.)
    assert args.side_roll=='box-aligned'


def test_explicit_native_minimum_preserves_small_object_and_center(monkeypatch,tmp_path):
    from tools.prepare_reflex_comparison import build_reflex_target
    from rh56_controller.grasp_geometry import InspireHandFK,ClosureGeometry
    monkeypatch.setenv('RH56_CACHE_DIR',str(tmp_path/'cache'))
    frame=dict(center_base_mm=[-238.,-680.,40.],short_axis_yaw_deg=90.,yaw_ambiguous=False)
    target=build_reflex_target(frame,[12.,24/np.sqrt(3),11.],6.,'nut.json',
        fingers=2,grasp_direction='legacy-top',use_native_two_finger_minimum=True)
    native=ClosureGeometry(InspireHandFK()).line(.012)
    assert target['grasp_center_base_mm']==pytest.approx([-238.,-680.,45.])
    assert target['measured_width_mm']==12.
    assert target['achieved_width_mm']==pytest.approx(native.width*1000)
    assert target['width_residual_mm']>3.
    assert target['width_policy']=='operator_requested_native_two_finger_minimum'
    for key,value in target['final_ctrl'].items():
        assert value==pytest.approx(native.ctrl_values[key])
    assert target['final_raw'][:3]==[1000]*3
    # Do not secretly level the clipped native minimum posture to force a pass.
    tips=target['active_tip_positions_base_m']
    assert abs(tips['thumb'][2]-tips['index'][2])>.010


def test_native_minimum_is_not_a_general_width_bypass(monkeypatch,tmp_path):
    from tools.prepare_reflex_comparison import build_reflex_target
    monkeypatch.setenv('RH56_CACHE_DIR',str(tmp_path/'cache'))
    frame=dict(center_base_mm=[-238.,-680.,40.],short_axis_yaw_deg=90.,yaw_ambiguous=False)
    with pytest.raises(ValueError,match='outside'):
        build_reflex_target(frame,[12.,14.,11.],6.,'nut.json',fingers=2,grasp_direction='legacy-top')
    with pytest.raises(ValueError,match='no wider'):
        build_reflex_target(frame,[26.,30.,25.],10.,'berry.json',fingers=2,
            grasp_direction='legacy-top',use_native_two_finger_minimum=True)
    with pytest.raises(ValueError,match='limited'):
        build_reflex_target(frame,[12.,14.,11.],6.,'nut.json',fingers=3,
            grasp_direction='legacy-top',use_native_two_finger_minimum=True)


def test_cube_three_finger_mode_reuses_native_solver(monkeypatch,tmp_path):
    from tools.prepare_reflex_comparison import build_reflex_target
    from tools import run_reflex_ten_starts as runner
    monkeypatch.setenv('RH56_CACHE_DIR',str(tmp_path/'cache'))
    args=runner.parse_args(['--crop-report','cube.json','--object-label','cube',
                           '--fingers','3','--grasp-depth-mm','15'])
    target=build_reflex_target(dict(center_base_mm=[-220.,-680.,40.],
        short_axis_yaw_deg=87.,yaw_ambiguous=True),[40.,40.,40.],args.grasp_depth_mm,'cube.npy',
        fingers=args.fingers,grasp_direction='legacy-top')
    runner.validate_target(target)
    assert target['mode']=='GraspMode.PLANE_3F'
    assert set(target['active_tip_positions_base_m'])=={'thumb','index','middle'}
    assert target['final_raw'][:2]==[1000,1000]
    assert target['final_ctrl']['ring']==target['final_ctrl']['pinky']==0.
    assert target['thumb_raw'][:4]==[1000]*4
    assert target['grasp_center_base_mm']==pytest.approx([-220,-680,65])
    assert abs(target['achieved_width_mm']-40)<1.
    assert target['side_pregrasp_tcp_T'] is None
    target['final_raw'][1]=500
    with pytest.raises(ValueError,match='Inactive'):
        runner.validate_target(target)


def test_raspberry_two_finger_mode_reuses_native_line_solver(monkeypatch,tmp_path):
    from copy import deepcopy
    from tools.prepare_reflex_comparison import build_reflex_target
    from tools import run_reflex_ten_starts as runner
    monkeypatch.setenv('RH56_CACHE_DIR',str(tmp_path/'cache'))
    args=runner.parse_args(['--crop-report','raspberry.json','--object-label','raspberry',
                           '--fingers','2','--grasp-depth-mm','10'])
    assert not args.execute and not args.capture_live
    target=build_reflex_target(dict(center_base_mm=[-220.,-680.,34.],
        short_axis_yaw_deg=90.,yaw_ambiguous=True),[25.,25.,25.],args.grasp_depth_mm,
        'raspberry.npy',fingers=args.fingers,grasp_direction='legacy-top')
    runner.validate_target(target)
    assert target['mode']=='GraspMode.LINE_2F'
    assert set(target['active_tip_positions_base_m'])=={'thumb','index'}
    assert target['final_raw'][:3]==[1000]*3
    assert all(target['final_ctrl'][k]==0. for k in ('pinky','ring','middle'))
    assert target['thumb_raw'][:4]==[1000]*4
    assert target['grasp_center_base_mm']==pytest.approx([-220.,-680.,49.])
    assert abs(target['achieved_width_mm']-25)<1.
    assert not target['executable'] and not target['hardware_contacted']
    for index in range(3):
        bad=deepcopy(target)
        bad['final_raw'][index]=500
        with pytest.raises(ValueError,match='Inactive'):
            runner.validate_target(bad)
    bad=deepcopy(target)
    bad['active_tip_positions_base_m']['middle']=[0.,0.,0.]
    with pytest.raises(ValueError,match='active fingertips'):
        runner.validate_target(bad)
    with pytest.raises(SystemExit):
        parse_args(['--crop-report','raspberry.json','--fingers','2','--grasp-direction','side'])


def test_deeper_reflex_translates_only_z_not_orientation_or_closure(monkeypatch,tmp_path):
    from tools.prepare_reflex_comparison import build_reflex_target
    monkeypatch.setenv('RH56_CACHE_DIR',str(tmp_path/'cache'))
    frame=dict(center_base_mm=[-220.,-680.,40.],short_axis_yaw_deg=87.,yaw_ambiguous=False)
    old,new=[build_reflex_target(frame,[40.,90.,180.],d,'saved.npy',
        fingers=4,grasp_direction='legacy-top') for d in (10.,35.)]
    assert np.allclose(np.array(new['grasp_center_base_mm'])-old['grasp_center_base_mm'],[0.,0.,-25.])
    for key in ('final_hand_T','final_tcp_T'):
        before,after=np.asarray(old[key]),np.asarray(new[key])
        assert np.allclose(before[:3,:3],after[:3,:3])
        assert np.allclose(after[:3,3]-before[:3,3],[0.,0.,-.025])
    for key in ('thumb_raw','final_raw','phase_order'):
        assert old[key]==new[key]
    with pytest.raises(ValueError,match='inside measured object height'):
        build_reflex_target(frame,[40.,90.,30.],35.,'saved.npy',fingers=4,grasp_direction='legacy-top')


def test_original_cli_mode_has_no_side_rotation_or_staging(monkeypatch,tmp_path):
    from tools.prepare_reflex_comparison import build_reflex_target
    monkeypatch.setenv('RH56_CACHE_DIR',str(tmp_path/'cache'))
    args=parse_args(['--crop-report','crop.json'])
    result=build_reflex_target(dict(center_base_mm=[-220.,-680.,40.],
        short_axis_yaw_deg=87.,yaw_ambiguous=False),[40.,90.,180.],args.grasp_depth_mm,'saved.npy',
        fingers=args.fingers,grasp_direction=args.grasp_direction,side_roll=args.side_roll)
    assert result['mode']=='GraspMode.PLANE_4F'
    assert result['final_raw'][0]==1000
    assert np.allclose(result['grasp_center_base_mm'],[-220.,-680.,210.])
    assert result['side_pregrasp_tcp_T'] is None and result['side_approach_mm'] is None
    assert result['height_fraction'] is None and result['orientation_only_roll_correction_deg'] is None
    assert result['phase_order']==['thumb_position','blocking_moveL_to_final_pose','remaining_fingers_close']
    assert np.allclose(np.array(result['final_hand_T'])[:3,:3],result['reference_hand_rotation'])
    assert not result['executable']


@pytest.mark.parametrize('option,value',[('--height-fraction','0'),('--height-fraction','1'),
    ('--height-fraction','nan'),('--side-approach-mm','0'),('--side-approach-mm','inf')])
def test_invalid_side_parameters(option,value):
    with pytest.raises(SystemExit):
        parse_args(['--crop-report','crop.json',option,value])


@pytest.mark.parametrize('yaw',[0.,45.,87.])
def test_original_five_finger_side_pose_and_horizontal_approach(yaw,monkeypatch,tmp_path):
    from tools.prepare_reflex_comparison import build_reflex_target
    monkeypatch.setenv('RH56_CACHE_DIR',str(tmp_path/'cache'))
    result=build_reflex_target(dict(center_base_mm=[-220.,-680.,40.],
        short_axis_yaw_deg=yaw,yaw_ambiguous=False),[40.,90.,180.],10.,'saved.npy',side_roll='wrist-above')
    assert result['mode']=='GraspMode.PLANE_5F'
    assert set(result['active_tip_positions_base_m'])=={'thumb','index','middle','ring','pinky'}
    tips={k:np.array(v) for k,v in result['active_tip_positions_base_m'].items()}
    other=np.mean([v for k,v in tips.items() if k!='thumb'],axis=0)
    assert np.allclose((tips['thumb']+other)/2,[-.220,-.680,.130],atol=1e-9)
    # Contact plane vertical; pinch direction still along the observed short axis.
    assert abs(result['contact_plane_normal_base'][2])<1e-9
    opposition=tips['thumb']-other
    axis=np.array([np.cos(np.radians(yaw)),np.sin(np.radians(yaw)),0.])
    assert abs(np.dot(opposition/np.linalg.norm(opposition),axis))>.99
    final=np.array(result['final_tcp_T']); pre=np.array(result['side_pregrasp_tcp_T'])
    assert np.allclose(final[:3,:3],pre[:3,:3])
    delta=final[:3,3]-pre[:3,3]
    assert abs(delta[2])<1e-12
    assert np.linalg.norm(delta)==pytest.approx(.080)
    assert np.allclose(delta/.080,result['local_approach_direction_base'])
    normal=np.array(result['contact_plane_normal_base'])
    assert abs(np.dot(delta/.080,normal))==pytest.approx(1.)
    assert np.array(result['final_hand_T'])[2,3]>.130
    assert np.dot(np.array(result['final_hand_T'])[:2,3]-[-.220,-.680],[.220,.680])>0
    assert result['final_raw'][0]<1000  # Pinky now participates.
    assert not result['executable'] and not result['hardware_contacted']


@pytest.mark.parametrize('yaw',[0.,45.,87.])
def test_thumb_and_index_level_at_top_changes_only_orientation(yaw,monkeypatch,tmp_path):
    from tools.prepare_reflex_comparison import build_reflex_target
    monkeypatch.setenv('RH56_CACHE_DIR',str(tmp_path/'cache'))
    frame=dict(center_base_mm=[-220.,-680.,40.],short_axis_yaw_deg=yaw,yaw_ambiguous=False)
    old=build_reflex_target(frame,[44.46,90.,180.],10.,'saved.npy',side_roll='wrist-above')
    new=build_reflex_target(frame,[44.46,90.,180.],10.,'saved.npy',side_roll='thumb-index-top')
    heights={k:v[2] for k,v in new['active_tip_positions_base_m'].items()}
    assert heights['thumb']==pytest.approx(heights['index'],abs=1e-9)
    assert heights['index']>heights['middle']>heights['ring']>heights['pinky']
    for key in ('grasp_center_base_mm','final_raw','thumb_raw','final_ctrl','phase_order','side_approach_mm'):
        assert new[key]==old[key]
    assert np.allclose(new['contact_plane_normal_base'],old['contact_plane_normal_base'])
    assert not new['executable']


@pytest.mark.parametrize('yaw',[0.,45.,87.,180.])
def test_photo_side_preserves_original_wrist_heading_and_top_pair(yaw,monkeypatch,tmp_path):
    from tools.prepare_reflex_comparison import build_reflex_target
    monkeypatch.setenv('RH56_CACHE_DIR',str(tmp_path/'cache'))
    frame=dict(center_base_mm=[-220.,-680.,40.],short_axis_yaw_deg=yaw,yaw_ambiguous=False)
    new=build_reflex_target(frame,[44.46,90.,180.],10.,'saved.npy',side_roll='photo-side')
    old=build_reflex_target(frame,[44.46,90.,180.],10.,'saved.npy',side_roll='thumb-index-top')
    direction=np.array(new['hand_forward_axis_base'])
    reference=np.array(new['reference_hand_rotation'])
    actual=np.array(new['final_hand_T'])[:3,:3]
    assert np.allclose(direction,reference[:,2],atol=1e-9)
    relative=reference.T@actual
    assert np.allclose(relative[:,2],[0.,0.,1.],atol=1e-9)
    assert np.allclose(relative[2,:],[0.,0.,1.],atol=1e-9)
    h={k:v[2] for k,v in new['active_tip_positions_base_m'].items()}
    assert h['thumb']==pytest.approx(h['index'],abs=1e-9)
    assert h['index']>h['middle']>h['ring']>h['pinky']
    for key in ('final_raw','thumb_raw','grasp_center_base_mm','phase_order','side_approach_mm'):
        assert new[key]==old[key]
    horizontal=direction.copy();horizontal[2]=0.;horizontal/=np.linalg.norm(horizontal)
    assert np.allclose(new['local_approach_direction_base'],horizontal)
    # Existing mount maps TCP +Z to hand +Z: preserve the original axis.
    assert np.allclose(np.array(new['final_tcp_T'])[:3,2],direction,atol=1e-6)


@pytest.mark.parametrize('yaw',[0.,45.,87.,180.])
def test_box_alignment_uses_faces_not_exact_fingertip_height(yaw,monkeypatch,tmp_path):
    from tools.prepare_reflex_comparison import build_reflex_target
    monkeypatch.setenv('RH56_CACHE_DIR',str(tmp_path/'cache'))
    frame=dict(center_base_mm=[-220.,-680.,40.],short_axis_yaw_deg=yaw,yaw_ambiguous=False)
    new=build_reflex_target(frame,[44.46,90.,180.],10.,'saved.npy')
    old=build_reflex_target(frame,[44.46,90.,180.],10.,'saved.npy',side_roll='photo-side')
    r=np.array(new['final_hand_T'])[:3,:3]
    short=np.array([np.cos(np.radians(yaw)),np.sin(np.radians(yaw)),0.])
    assert abs(np.dot(r[:,0],short))==pytest.approx(1.)
    assert np.allclose(r[:,1],[0.,0.,1.])
    assert abs(r[2,2])<1e-9
    assert np.dot(r[:,2],short)==pytest.approx(0.,abs=1e-9)
    assert np.linalg.det(r)==pytest.approx(1.)
    for key in ('final_raw','thumb_raw','grasp_center_base_mm','phase_order','side_approach_mm'):
        assert new[key]==old[key]
    assert new['fingertip_order_contract'] is None
    assert new['box_alignment_contract'] is not None


def test_preview_has_no_hardware_execution_option():
    with pytest.raises(SystemExit):
        parse_args(['--ggx-run','saved','--execute'])


@pytest.mark.parametrize('depth',['0','-1','nan','inf'])
def test_preview_rejects_invalid_grasp_depth(depth):
    with pytest.raises(SystemExit):
        parse_args(['--ggx-run','saved','--grasp-depth-mm',depth])


def test_original_reflex_is_thumb_arm_fingers_without_hand_readback(monkeypatch):
    events=[]
    monkeypatch.setattr('rh56_controller.grasp_executor.time.sleep',lambda t:events.append(('wait',t)))
    class Hand:
        def angle_set(self, command): events.append(('hand',list(command)))
        def angle_read(self): pytest.fail('Write-only reflex must not read hand angles')
        def force_act(self): pytest.fail('Write-only reflex must not read hand forces')
    class Arm:
        pose=np.eye(4)
        def move_to_hand_pose(self,pose,blocking):
            assert blocking
            self.pose=pose
            events.append(('arm',))
            return []
        def get_hand_pose_4x4(self): return self.pose
        def get_joint_angles(self): return [0]*6
    names=['pinky','ring','middle','index','thumb_proximal','thumb_yaw']
    fk=SimpleNamespace(ctrl_min=dict.fromkeys(names,0.),ctrl_max=dict.fromkeys(names,1.))
    executor=GraspExecutor(Arm(),Hand(),fk,hand_write_only=True,status_cb=lambda s:None)
    monkeypatch.setattr(executor,'_start',lambda fn,*a:fn(*a))
    logger=SimpleNamespace(log_waypoint=lambda **k:None,log_done=lambda **k:None,close=lambda:None)
    result=SimpleNamespace(ctrl_values=dict(zip(names,[0.,.2,.3,.4,.5,1.])))
    executor.execute_thumb_reflex(result,np.eye(4),force_N=0.,logger=logger)
    assert events==[('hand',[1000,1000,1000,1000,500,0]),('wait',.2),('arm',),
                    ('hand',[1000,800,700,600,500,0]),('wait',.25)]


def test_reflex_force_feedback_cannot_be_enabled_for_write_only_hand():
    executor=GraspExecutor(None,None,None,hand_write_only=True)
    with pytest.raises(ValueError,match='Force'):
        executor.execute_thumb_reflex(None,np.eye(4),force_N=1.)
    assert executor._thread is None
