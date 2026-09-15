"""Offline deterministic protocol; no camera, controller or lab artifacts."""
import numpy as np
import pytest
pytest.importorskip('mujoco')
pytest.importorskip('mink')
from tools import ggx_planning as p, ggx_real_support as support
from tools import preflight_ggx_ur5_table as sim, ggx_motion as motion


@pytest.fixture
def model():
    return support.build_model(dict(preflight=dict(object_center_m=[-.2,-.6,.14],
        object_size_mm=[90,40,180],object_yaw_deg=0,table_height_m=.05)))


def test_workspace_is_model_derived_not_cube_or_object_dependent():
    b=np.asarray(p.tcp_bounds_from_urdf())
    assert b.shape==(2,3) and b[0,2]==0
    assert b[0,0] < -450 and b[1,0] > 50
    assert np.isfinite(b).all() and (b[0]<b[1]).all()
    assert p.policy()['standoffs_mm']==[80,60,40,20]


def test_derived_config_roundtrips_and_cannot_be_replaced(tmp_path):
    import yaml
    from tools import run_ggx_real as real
    config=real.load_config(real.ROOT/'configs/ggx_real_lab.yaml')
    path=tmp_path/'saved.yaml';path.write_text(yaml.safe_dump(config))
    assert real.load_config(path)==config
    config['tcp_bounds_mm'][0][2]=-100
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match='derived envelope'):real.load_config(path)


def test_equivalent_wrap_preserves_fk_and_limits(model):
    q=sim.initial_qpos(model,np.radians([0,-90,90,350,90,0]))
    near=sim.initial_qpos(model,np.radians([0,-90,90,-10,90,0]))
    adjusted=p.nearest_equivalent(model,q,near)
    assert np.allclose(adjusted,near)
    assert max(support.pose_error(support.fk_tcp(model,q[p.arm_addresses(model)]),
                                 support.fk_tcp(model,adjusted[p.arm_addresses(model)])))<1e-8


def test_reach_bound_accepts_fk_of_valid_configurations(model):
    import mujoco
    args=sim.parse_args([]);data=mujoco.MjData(model)
    for degrees in p.SEEDS_DEG:
        data.qpos[:]=sim.initial_qpos(model,np.radians(degrees));mujoco.mj_forward(model,data)
        T=np.eye(4);T[:3,3]=data.site('eeff').xpos;T[:3,:3]=data.site('eeff').xmat.reshape(3,3)
        assert not p.reach_bound(model,T,args)['unreachable']


def test_multiseed_retries_same_target_and_tolerance(model,monkeypatch):
    target=np.eye(4);args=sim.parse_args([]);seen=[]
    monkeypatch.setattr(p,'reach_bound',lambda *a:{'unreachable':False})
    def solve(m,seed,T,**kw):
        seen.append((seed.copy(),T.copy(),kw))
        return seed.copy(),.001,.001,len(seen)==2
    monkeypatch.setattr(sim,'solve_pose_ik',solve)
    attempts=[]
    assert next(p.solutions(model,target,args,attempts)) is not None
    assert len(seen)==2 and not np.array_equal(seen[0][0],seen[1][0])
    assert all(np.array_equal(T,target) and kw['position_tol_m']==.003 for _,T,kw in seen)


def test_geometry_reference_cannot_be_motion():
    with pytest.raises(ValueError,match='actual travel'):
        motion.phase_goals(dict(geometry_reference_only=True))


def test_template_creation_does_not_check_scan_travel(tmp_path,monkeypatch,model):
    from tools import ggx_template as t
    source=dict(selection_scope='grasp_geometry',rank=0,pins={},desk_envelope={},
        preflight=dict(desk_envelope={},object_center_m=[0,0,0],object_yaw_deg=0,object_size_mm=[40]*3))
    monkeypatch.setattr(support,'read_approval',lambda *a:source)
    monkeypatch.setattr(support,'prepare',lambda *a,**kw:pytest.fail('No fictitious scan travel'))
    calls=[]
    def reference(s,o):
        calls.append(True);(o/'prepared.json').write_text('{}')
        return dict(rows=[dict(phase='approach',tcp=[0,0,.3,0,0,0],q_nominal=[0]*6),
                          dict(phase='close',closure_alpha=.5)]),model
    monkeypatch.setattr(p,'reference',reference)
    monkeypatch.setattr(t,'open_center_tcp',lambda *a:np.zeros(3))
    snapshot=tmp_path/'scan.json';snapshot.write_text('{}')
    (tmp_path/'approval.json').write_text('{}')
    t.create(tmp_path/'approval.json',snapshot,dict(desk_edge_y_m=None,desk_envelope={}),tmp_path)
    assert calls==[True]


def test_relocated_geometry_defers_ik_to_actual_point(monkeypatch):
    from tools import ggx_template as t
    monkeypatch.setattr(support,'build_model',lambda s:None)
    monkeypatch.setattr(t,'solve_tcp',lambda *a:pytest.fail('No shared-reference IK before trial'))
    source=dict(selection_scope='grasp_geometry',preflight={'arguments':{}})
    reference=dict(rows=[dict(phase='start_to_pregrasp',tcp=[0,0,.3,0,0,0],q_nominal=[0]*6),
                         dict(phase='approach',tcp=[0,0,.2,0,0,0],q_nominal=[0]*6)])
    delta=np.eye(4);delta[0,3]=.01
    _,rows=t.canonical({},source,reference,delta)
    assert rows[0]['tcp'][0]==.01 and rows[1]['tcp'][0]==.01


def test_all_geometry_reasons_are_retained(model):
    q=sim.initial_qpos(model,np.zeros(6))
    sample=sim.TrajectorySample('approach',1.,q,0.)
    failures=p.geometry_failures(model,[sample],dict(tcp_bounds_mm=[[0,0,999],[1,1,1000]]),target_gate=True)
    assert any(f['kind']=='workspace' for f in failures)
    assert all('phase' in f for f in failures)


@pytest.mark.parametrize('state',[{}, {'motion_commands_attempted':True}, {'hand_commands':[{}]},
                                {'motion_confirmation_received':True},{'cleanup_errors':['failed']}])
def test_typed_planning_skip_is_premotion_only(state):
    from tools import run_ggx_experiment as e
    assert e.skip_collision(p.PlanningUnavailable('ik_search_exhausted',[]),state)==(not state)
    assert not e.skip_collision(RuntimeError('IK failed'),state)


def test_shorter_predeclared_retreat_survives_80mm_failure(model,monkeypatch):
    from types import SimpleNamespace
    q=sim.initial_qpos(model,np.zeros(6));calls=[]
    monkeypatch.setattr(p,'solutions',lambda *a,**kw:iter([q]))
    monkeypatch.setattr(sim,'graspgenx_to_mujoco_base_pose',lambda *a,**kw:(np.array([0.,0.,.4]),np.eye(3)))
    monkeypatch.setattr(p,'geometry_failures',lambda *a,**kw:[])
    def solve(*a,**kw):
        calls.append(True)
        return q.copy(),0.,0.,len(calls)<5
    monkeypatch.setattr(sim,'solve_pose_ik',solve)
    def evaluate(m,start,candidate,rank,args,center):
        assert args.standoff_mm==40
        samples=[sim.TrajectorySample(phase,1.,q.copy(),.5 if phase in ('close','lift') else 0.)
                 for phase in ('start','start_to_pregrasp','approach','close','lift')]
        return sim.CandidateResult(rank,'fixture',1.,True,'pass',0.,0.,100.,'','',len(samples)),samples
    monkeypatch.setattr(sim,'evaluate_candidate',evaluate)
    detail,_=p.select_geometry(model,SimpleNamespace(approach_direction=np.array([0,0,1])),0,
                               sim.parse_args([]),np.zeros(3),{})
    assert detail['state']=='geometry_eligible' and detail['standoff_mm']==40
    assert detail['actual_travel_checked'] is False
    assert detail['local_execution_check']['passed'] is True


def test_target_overlap_cannot_enter_normal_geometry_review(model,monkeypatch):
    from types import SimpleNamespace
    q=sim.initial_qpos(model,np.zeros(6))
    monkeypatch.setattr(p,'solutions',lambda *a,**kw:iter([q]))
    monkeypatch.setattr(sim,'graspgenx_to_mujoco_base_pose',lambda *a,**kw:(np.array([0.,0.,.4]),np.eye(3)))
    monkeypatch.setattr(sim,'solve_pose_ik',lambda *a,**kw:(q.copy(),0.,0.,True))
    monkeypatch.setattr(p,'geometry_failures',lambda *a,target_gate,**kw:
        [{'kind':'target_proxy_uncertain','phase':'approach','clearance_mm':-20.}] if target_gate else [])
    def evaluate(*a):
        samples=[sim.TrajectorySample(phase,1.,q.copy(),.5 if phase in ('close','lift') else 0.)
                 for phase in ('start','start_to_pregrasp','approach','close','lift')]
        return sim.CandidateResult(0,'fixture',1.,True,'pass',0.,0.,100.,'','',len(samples)),samples
    monkeypatch.setattr(sim,'evaluate_candidate',evaluate)
    detail,samples=p.select_geometry(model,SimpleNamespace(approach_direction=np.array([0,0,1])),0,
                                     sim.parse_args([]),np.zeros(3),{})
    assert detail['state']=='local_approach_unresolved'
    assert detail['target_proxy_uncertain'] and not detail['local_execution_check']['passed']
    assert samples and detail['standoff_mm']==80  # No search/tuning to force a pass.


def test_reference_keeps_exact_ggx_endpoint_not_approximate_ik(model):
    from rh56_controller.graspgenx_baseline import GraspGenXCandidate
    c=GraspGenXCandidate('fixture',1.,np.array([-.2,-.6,.3]),np.eye(3))
    q=sim.initial_qpos(model,np.zeros(6))
    samples=[sim.TrajectorySample(phase,1.,q.copy(),.5 if phase in ('close','lift') else 0.)
             for phase in ('start','start_to_pregrasp','approach','close','lift')]
    rows=p.exact_reference_rows(model,samples,c,np.zeros(3),sim.parse_args([]),40.)
    pos,rot=sim.graspgenx_to_mujoco_base_pose(c,object_center=np.zeros(3))
    hand=np.eye(4);hand[:3,:3]=rot;hand[:3,3]=pos
    expected=support.UR5Bridge.hand_T_to_tcp_T(hand)
    end=[r for r in rows if r['phase']=='approach'][-1]
    assert np.allclose(support.pose_matrix(end['tcp']),expected)
    assert all(r['tcp']==end['tcp'] for r in rows if r['phase']=='close')


def test_geometry_record_cannot_forge_path_pass(tmp_path):
    import json
    path=tmp_path/'summary.json'
    path.write_text(json.dumps(dict(schema=p.SCHEMA,geometry_review_ranks=[0],automatic_pass_ranks=[0],
        geometry={'state':'geometry_eligible'},actual_travel_checked=False,policy=p.policy())))
    with pytest.raises(ValueError,match='cannot inherit path approval'):
        support.read_candidate_source(path,0,path,support.pin([path]),'offline')
