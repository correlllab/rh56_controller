"""Hardware-free tests for frozen Reflex trials, phase commands, and stop rules."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from tools import run_reflex_ten_starts as runner


@pytest.mark.parametrize('margin,gap,blocked',[(None,.0018,True),(1.5,.0018,False),
    (1.5,.0014,True),(1.5,0.,True),(1.5,-.001,True),
    (0.,.0005,False),(0.,0.,True),(0.,-.001,True)])
def test_execution_review_uses_same_configured_desk_margin(tmp_path,monkeypatch,margin,gap,blocked):
    import json
    sample=SimpleNamespace(phase='approach',phase_alpha=.5,qpos=np.zeros(1),min_desk_clearance_m=gap)
    monkeypatch.setattr(runner,'checked_samples',lambda *a:[sample])
    seen=[]
    monkeypatch.setattr(runner.sim,'check_trajectory',lambda *a,**kw:seen.append(kw['required_clearance_m']))
    monkeypatch.setattr(runner.sim.mujoco,'MjData',lambda *a:SimpleNamespace(qpos=np.zeros(1)))
    monkeypatch.setattr(runner.sim.mujoco,'mj_forward',lambda *a:None)
    monkeypatch.setattr(runner.motion,'reflex_target_check',lambda *a:dict(
        minimum_target_gap_mm=5.,nearest_target_geom='thumb',forbidden_target_overlap=False))
    monkeypatch.setattr(runner.preview,'render_reflex',lambda *a:None)
    config={} if margin is None else {'reflex_desk_clearance_mm':margin}
    target={'grasp_center_base_mm':[0.,0.,100.]}
    if blocked:
        with pytest.raises(runner.motion.PathCollision,match='geometry failed'):
            runner.review_checked(tmp_path,None,target,[],SimpleNamespace(),config)
    else:
        runner.review_checked(tmp_path,None,target,[],SimpleNamespace(),config)
    assert seen==[.002 if margin is None else margin/1000]
    saved=json.loads((tmp_path/'execution_precheck.json').read_text())
    assert saved['passed']==(not blocked)
    assert saved['required_desk_clearance_mm']==(2. if margin is None else margin)


def test_cli_offline_default_and_live_requires_fresh_xy():
    args=runner.parse_args(['--crop-report','crop.json','--object-label','sugar'])
    assert not args.execute and not args.capture_live and args.point=='all'
    assert args.grasp_depth_mm==35.
    assert args.fingers==4
    assert args.start_snapshot==runner.ROOT/'configs/reflex_scan_wrist_flip.json'
    with pytest.raises(SystemExit):
        runner.parse_args(['--crop-report','crop.json','--object-label','sugar','--execute'])


def test_selected_subset_preserves_point_ids_and_omits_completed_p2():
    base=['--template','raspberry.json','--object-label','raspberry']
    ids=['P1','P3','P4','P5','P6','P7','P8','P9','P10']
    args=runner.parse_args(base+['--points',*ids])
    assert runner.selected_point_ids(args)==ids
    assert 'P2' not in runner.selected_point_ids(args)
    assert not args.execute and not args.capture_live
    for invalid in [['P3','P1'],['P1','P1']]:
        with pytest.raises(SystemExit):
            runner.parse_args(base+['--points',*invalid])
    with pytest.raises(SystemExit):
        runner.parse_args(base+['--point','P2','--points','P3'])
    with pytest.raises(SystemExit):
        runner.parse_args(base+['--points','P3','P4','--trial-captures','one'])
    replay=runner.parse_args(base+['--points','P3','P4','--trial-captures','one','two'])
    assert len(replay.trial_captures)==2


def test_subset_resume_requires_explicit_remaining_selection(tmp_path,monkeypatch):
    args=runner.parse_args(['--resume-from',str(tmp_path/'summary.json'),'--object-label','raspberry'])
    monkeypatch.setattr(runner.review,'read_json',lambda p:dict(selected_points=['P1','P3']))
    with pytest.raises(ValueError,match='Subset run'):
        runner.resume_state(args)


@pytest.mark.parametrize('value',['nan','inf','0','11'])
def test_hold_duration_is_bounded(value):
    with pytest.raises(SystemExit):
        runner.parse_args(['--crop-report','crop.json','--object-label','sugar','--hold-seconds',value])


@pytest.mark.parametrize('value',['nan','inf','0','-1'])
def test_grasp_depth_is_finite_positive(value):
    with pytest.raises(SystemExit):
        runner.parse_args(['--crop-report','crop.json','--object-label','sugar','--grasp-depth-mm',value])


def test_reflex_speed_profile_only_doubles_arm_speed():
    args=runner.parse_args(['--crop-report','crop.json','--object-label','sugar'])
    assert args.config==runner.ROOT/'configs/reflex_real_lab.yaml'
    old=runner.real.load_config(runner.ROOT/'configs/ggx_real_lab.yaml')
    new=runner.real.load_config(args.config)
    for key in ('joint_speed_rad_s','cartesian_speed_m_s'):
        assert new[key]==2*old[key]
    assert {k:v for k,v in old.items() if k not in ('joint_speed_rad_s','cartesian_speed_m_s')}=={
        k:v for k,v in new.items() if k not in ('joint_speed_rad_s','cartesian_speed_m_s')}
    assert 'reset_motion' not in new  # Reset inherits this same profile.


def test_frozen_template_cannot_silently_reuse_old_depth(tmp_path,monkeypatch):
    args=runner.parse_args(['--template',str(tmp_path/'template.json'),'--object-label','sugar'])
    monkeypatch.setattr(runner.review,'read_json',lambda p:dict(
        schema='rh56_reflex_ten_template/v1',object_label='sugar',target=dict(grasp_depth_mm=10.)))
    monkeypatch.setattr(runner,'validate_target',lambda t:None)
    with pytest.raises(ValueError,match='grasp depth differs'):
        runner.freeze(args,tmp_path)


def test_frozen_round_yaw_requires_same_explicit_option(tmp_path,monkeypatch):
    args=runner.parse_args(['--template',str(tmp_path/'template.json'),'--object-label','apple'])
    monkeypatch.setattr(runner.review,'read_json',lambda p:dict(
        schema='rh56_reflex_ten_template/v1',object_label='apple',
        target=dict(grasp_depth_mm=35.,fingers=4,round_object_yaw_deg=90.)))
    monkeypatch.setattr(runner,'validate_target',lambda t:None)
    with pytest.raises(ValueError,match='round-object yaw differs'):
        runner.freeze(args,tmp_path)


def test_frozen_round_approximation_requires_same_explicit_option(tmp_path,monkeypatch):
    args=runner.parse_args(['--template',str(tmp_path/'template.json'),'--object-label','raspberry',
                           '--fingers','2','--grasp-depth-mm','10','--round-object-yaw-deg','90'])
    monkeypatch.setattr(runner.review,'read_json',lambda p:dict(
        schema='rh56_reflex_ten_template/v1',object_label='raspberry',
        target=dict(grasp_depth_mm=10.,fingers=2,round_object_yaw_deg=90.,round_object_approximation=True)))
    monkeypatch.setattr(runner,'validate_target',lambda t:None)
    with pytest.raises(ValueError,match='approximation differs'):
        runner.freeze(args,tmp_path)


def record(phase,desk=0,overlap=0,forbidden=None):
    return dict(status='nominal_sequence_solved_contact_unverified',phase_metrics=[
        dict(phase=phase,desk_reject_samples=desk,target_overlap_samples=overlap,
             target_contact_policy=runner.motion.REFLEX_CONTACT_POLICY,
             forbidden_target_overlap_samples=overlap if forbidden is None else forbidden)])


def test_reflex_finger_contact_allowed_but_desk_preparation_and_palm_blocked():
    runner.nominal_gate(record('reflex_approach',overlap=3,forbidden=0))
    with pytest.raises(runner.motion.PathCollision) as caught:
        runner.nominal_gate(record('reflex_approach',overlap=3))
    assert caught.value.obstacle=='object'
    runner.nominal_gate(record('remaining_fingers_close',overlap=16,forbidden=0))
    with pytest.raises(runner.motion.PathCollision):
        runner.nominal_gate(record('position_to_trial_start',overlap=1))
    with pytest.raises(runner.motion.PathCollision):
        runner.nominal_gate(record('remaining_fingers_close',desk=1,overlap=16))


def test_reflex_executes_thumb_then_arm_then_fingers_no_ggx_closure():
    target=dict(thumb_ctrl=dict(runner.sim.GGX_OPEN_CTRL,thumb_proximal=.4),
        final_ctrl=dict(runner.sim.GGX_OPEN_CTRL,index=.6),
        thumb_raw=[1000]*4+[500,0],final_raw=[1000,600,600,600,500,0])
    commands=[dict(phase=p,revision=runner.motion.REVISION,hand_ctrl=c) for p,c in zip(
        ['position_to_trial_start','approach','lift'],
        [target['thumb_ctrl'],target['thumb_ctrl'],target['final_ctrl']])]
    events=[]
    backend=SimpleNamespace(open_hand=lambda cmd:events.append(('open',cmd)),
        move_phase=lambda cmd:events.append(('move',cmd['phase'])),
        hand_angles=lambda cmd:events.append(('hand',cmd)),dwell=lambda s:events.append(('wait',s)),
        set_stage=lambda s:None)
    runner.execute_sequence(backend,target,commands,3.,[1000]*6)
    assert events==[('open',[1000]*6),
        ('hand',target['thumb_raw']),('wait',.2),('move','position_to_trial_start'),('move','approach'),
        ('hand',target['final_raw']),('wait',.25),('move','lift'),('wait',3.)]
    commands[1]['hand_ctrl']=dict(runner.sim.GGX_OPEN_CTRL)
    events.clear()
    with pytest.raises(ValueError): runner.execute_sequence(backend,target,commands,3.,[1000]*6)
    assert events==[]


def test_checked_hand_preparation_is_at_scan_not_low_start(monkeypatch):
    target=dict(thumb_ctrl=dict(runner.sim.GGX_OPEN_CTRL,thumb_proximal=.4),
                final_ctrl=dict(runner.sim.GGX_OPEN_CTRL,index=.6))
    monkeypatch.setattr(runner.sim,'initial_qpos',lambda model,q:np.r_[q,np.zeros(6)])
    keys=list(runner.sim.GGX_OPEN_CTRL)
    monkeypatch.setattr(runner.sim,'set_hand_qpos',lambda model,q,ctrl:q.__setitem__(slice(6,None),[ctrl[k] for k in keys]))
    commands=[dict(phase=phase,hand_ctrl=ctrl,start_q=[i]*6,q=[i+1]*6,
                   samples=[dict(q=[i]*6,fraction=0.),dict(q=[i+1]*6,fraction=1.)])
              for i,(phase,ctrl) in enumerate(zip(['position_to_trial_start','approach','lift'],
                  [target['thumb_ctrl'],target['thumb_ctrl'],target['final_ctrl']]))]
    samples=runner.checked_samples(None,target,commands)
    assert [s.phase for s in samples[:16]]==['thumb_position']*16
    assert all(np.array_equal(s.qpos[:6],np.zeros(6)) for s in samples[:16])
    assert samples[16].phase=='position_to_trial_start'
    assert samples[16].qpos[6:]==pytest.approx([target['thumb_ctrl'][k] for k in keys])
    assert sum(s.phase=='thumb_position' for s in samples)==16
    close=[s for s in samples if s.phase=='remaining_fingers_close']
    assert len(close)==16 and all(np.array_equal(s.qpos[:6],np.full(6,2)) for s in close)
    commands[0]['hand_ctrl']=dict(runner.sim.GGX_OPEN_CTRL)
    with pytest.raises(ValueError,match='wrong hand shape'):
        runner.checked_samples(None,target,commands)


def test_relocation_changes_xy_only_without_refitting():
    t=np.eye(4);t[:3,3]=[-.2,-.6,.2]
    data=dict(target=dict(final_hand_T=t.tolist(),final_tcp_T=t.tolist(),
        grasp_center_base_mm=[-200.,-600.,200.],active_tip_positions_base_m={'thumb':[-.2,-.6,.2]},
        final_raw=[1]*6,thumb_raw=[2]*6),scene=dict(preflight=dict(object_center_m=[-.2,-.6,.1],
        object_size_mm=[90.,40.,176.],object_yaw_deg=4.)),perception={},crop={},pins={})
    old=deepcopy(data)
    source,target,delta=runner.relocate(data,dict(status='localized',object_xy_mm=[-197.,-604.]))
    assert data==old
    assert np.allclose(delta,[.003,-.004])
    assert target['grasp_center_base_mm']==pytest.approx([-197.,-604.,200.])
    assert np.allclose(np.array(target['final_hand_T'])[:3,:3],np.eye(3))
    assert source['preflight']['object_yaw_deg']==4.
    assert source['preflight']['object_size_mm']==[90.,40.,176.]
    assert target['final_raw']==old['target']['final_raw']


@pytest.fixture
def loop_setup(tmp_path,monkeypatch):
    args=runner.parse_args(['--crop-report','crop.json','--object-label','sugar'])
    config=dict(hand_force_raw=1000,hand_speed_raw=1000)
    target=dict(final_hand_T=np.eye(4).tolist(),final_tcp_T=np.eye(4).tolist(),
        grasp_center_base_mm=[0.,0.,100.],active_tip_positions_base_m={})
    data=dict(target=target,scene=dict(preflight=dict(object_center_m=[0.,0.,.1])),
        perception=dict(capture='saved'),crop={},pins={},lift_mm=80.)
    monkeypatch.setattr(runner,'freeze',lambda *a:(data,tmp_path/'template.json'))
    monkeypatch.setattr(runner.real,'load_config',lambda *a:config)
    monkeypatch.setattr(runner.fixed,'build',lambda *a,**kw:dict(schema='rh56_ggx_fixed_starts/v1',points=[dict(id=f'P{i}') for i in range(1,11)],source_sha256={}))
    monkeypatch.setattr(runner.ten,'code_pins',lambda *a:{})
    monkeypatch.setattr(runner.review,'pin',lambda *a:{})
    events=[]
    def locate(*a,**kw): events.append('locate');return dict(status='localized',object_xy_mm=[0.,0.],input_sha256={})
    def reset(*a): events.append('reset');return 0,args.start_snapshot
    def capture(*a): events.append('capture');return 'new'
    return SimpleNamespace(args=args,out=tmp_path,events=events,locate=locate,reset=reset,capture=capture)


def test_all_ten_skips_reuse_xy_and_never_reset_offline(loop_setup):
    x=loop_setup;report={}
    def trial(*a): raise runner.motion.PathCollision('blocked',phase='approach',obstacle='object')
    code=runner.run(x.args,x.out,report,lambda:None,reset_fn=x.reset,capture_fn=x.capture,
        locate_fn=x.locate,trial_fn=trial)
    assert code==2 and report['status']=='all_points_skipped'
    assert [t['point_id'] for t in report['trials']]==[f'P{i}' for i in range(1,11)]
    assert x.events==['locate']
    assert all(not t['motion_commands_attempted'] for t in report['trials'])


def test_successful_live_rounds_reset_and_relocate_each_time(loop_setup):
    x=loop_setup;x.args.execute=x.args.capture_live=x.args.allow_high_force=True
    report={}
    def trial(*args):
        trial=args[-2];trial.update(status='commands_completed_grasp_unverified',motion_commands_attempted=True)
        x.events.append('execute')
    def score(t): t['operator_outcome']='success';x.events.append('score');return True
    assert runner.run(x.args,x.out,report,lambda:None,reset_fn=x.reset,capture_fn=x.capture,
        locate_fn=x.locate,trial_fn=trial,score_fn=score)==0
    assert x.events==['reset','capture','locate','execute','score']*10
    assert all(t['operator_outcome']=='success' for t in report['trials'])


def test_controller_fault_stops_instead_of_skipping(loop_setup):
    x=loop_setup;report={}
    def trial(*a): raise RuntimeError('Controller joint safety limits failed')
    with pytest.raises(RuntimeError,match='Controller'):
        runner.run(x.args,x.out,report,lambda:None,locate_fn=x.locate,trial_fn=trial)
    assert len(report['trials'])==1 and report['trials'][0]['status']=='failed'


def test_collision_exception_after_motion_is_not_skippable(loop_setup):
    x=loop_setup;report={}
    def trial(*a):
        a[-2]['motion_commands_attempted']=True
        raise runner.motion.PathCollision('late failure',phase='approach',obstacle='object')
    with pytest.raises(runner.motion.PathCollision):
        runner.run(x.args,x.out,report,lambda:None,locate_fn=x.locate,trial_fn=trial)
    assert len(report['trials'])==1 and report['trials'][0]['status']=='failed_after_device_commands'


@pytest.mark.parametrize('stop_at',['nominal','controller_permission','motion_permission'])
def test_live_trial_never_writes_before_final_confirmation(tmp_path,monkeypatch,stop_at):
    import json
    events=[]
    snapshot=dict(q=[0.]*6,tcp=[0.,0.,.35,0.,0.,0.])
    scan=tmp_path/'scan.json';scan.write_text(json.dumps(snapshot))
    backend=SimpleNamespace(open_receiver=lambda:events.append('receive'),snapshot=lambda:snapshot,
        open_control=lambda:events.append('control'),control=object(),close=lambda:events.append('close'))
    monkeypatch.setattr(runner.support,'build_model',lambda *a:object())
    monkeypatch.setattr(runner,'check_shapes',lambda *a:None)
    monkeypatch.setattr(runner.sim,'parse_args',lambda *a:SimpleNamespace())
    nominal=record('reflex_approach',overlap=int(stop_at=='nominal'))
    nominal['nominal_goals']=[]
    monkeypatch.setattr(runner.preview,'simulate_point',lambda *a:nominal)
    answers=iter([False] if stop_at=='controller_permission' else [True,False])
    monkeypatch.setattr(runner.real,'confirm',lambda *a:next(answers))
    monkeypatch.setattr(runner.real,'check_tcp_offset',lambda *a:None)
    monkeypatch.setattr(runner.motion,'preflight_goals',lambda *a:[])
    monkeypatch.setattr(runner,'review_checked',lambda *a:tmp_path/'review.mp4')
    monkeypatch.setattr(runner.review,'pin',lambda *a:{})
    monkeypatch.setattr(runner.review,'show',lambda *a:None)
    monkeypatch.setattr(runner,'execute_sequence',lambda *a:pytest.fail('No motion approval'))
    args=SimpleNamespace(execute=True,video=True,open=False,hold_seconds=3.)
    config=dict(hand_speed_raw=1000,hand_force_raw=1000,joint_speed_rad_s=.2,cartesian_speed_m_s=.02)
    data=dict(simulation_config='sim.yaml',lift_mm=80.)
    trial=dict(pins={},motion_confirmation_received=False)
    expected=runner.motion.PathCollision if stop_at=='nominal' else KeyboardInterrupt
    with pytest.raises(expected):
        runner.run_trial(args,config,data,dict(fingers=4), {},dict(id='P2'),scan,tmp_path,trial,lambda:None,
                         backend_factory=lambda *a:backend)
    assert events==(['receive','control','close'] if stop_at=='motion_permission' else ['receive','close'])
    assert trial['motion_confirmation_received'] is False


def test_resume_keeps_results_and_restarts_first_unexecuted_point(loop_setup,monkeypatch):
    x=loop_setup
    parent=x.out/'previous.json'
    rows=[dict(point_id='P1',status='commands_completed_grasp_unverified',operator_outcome='fail'),
          dict(point_id='P2',status='commands_completed_grasp_unverified',operator_outcome='success'),
          dict(point_id='P3',status='failed',motion_commands_attempted=False)]
    configs=[x.args.config,x.args.layout_config,x.args.perception_config]
    runner.files.write_json(parent,dict(schema='rh56_reflex_ten_experiment/v1',status='failed',
        hold_seconds=3.,grasp_depth_mm=35.,template=str(x.out/'template.json'),trials=rows,
        pins={str(p):'mock' for p in configs}))
    monkeypatch.setattr(runner.review,'pin',lambda paths:{str(p):'mock' for p in paths})
    monkeypatch.setattr(runner.review,'verify',lambda pins:None)
    x.args.resume_from=parent
    seen=[]
    def trial(*a): seen.append(a[-2]['point_id']);a[-2]['status']='offline_path_passed_not_executed'
    report={}
    assert runner.run(x.args,x.out,report,lambda:None,locate_fn=x.locate,trial_fn=trial)==0
    assert seen==[f'P{i}' for i in range(3,11)]
    assert [r['operator_outcome'] for r in report['prior_trials']]==['fail','success']
    assert report['resume']['next_point']=='P3'
    rows[-1]['motion_commands_attempted']=True
    old=runner.review.read_json(parent);old['trials']=rows;runner.files.write_json(parent,old)
    with pytest.raises(ValueError,match='Partially executed'):
        runner.resume_state(x.args)
