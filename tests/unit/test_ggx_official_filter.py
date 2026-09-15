"""Official-filter adapter: pure data tests without Torch/GPU/hardware."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
cv2=pytest.importorskip('cv2')
from tools import run_ggx_official_filter as official
from tools import ggx_cb3_candidates as cb3
from tools import run_ggx_review as ui


def test_explicit_protocol_defaults_and_no_execute_cli():
    a=official.parse_args(['--inference','saved.json','--out','new'])
    assert a.threshold_mm==20 and not hasattr(a,'execute')
    for flags in (['--threshold-mm','0'],['--threshold-mm','nan'],['--execute']):
        with pytest.raises(SystemExit):official.parse_args(['--inference','x','--out','y',*flags])


def calibrated_fixture():
    depth=np.full((60,80),800,np.uint16)
    mask=np.zeros_like(depth,np.uint8)
    mask[10:25,10:25]=255
    mask[30:45,45:60]=255  # Another obstacle, NOT the selected target.
    cal=dict(selected_intrinsics=dict(camera_matrix=[[500,0,40],[0,500,30],[0,0,1]],
             distortion_coefficients=[0]*5),transforms=dict(base_from_camera=dict(matrix=np.eye(4).tolist())))
    return depth,mask,cal


def test_exclude_only_selected_target_keep_other_obstacles_and_desk():
    depth,mask,cal=calibrated_fixture()
    points,target,stats=official.scene_points(depth,mask,1,cal,.001,[0,0,80,60])
    assert len(points)==depth.size-225
    assert target.sum()==225 and not target[35,50]
    assert stats['non_target_valid_depth_fraction']==1
    assert np.allclose(points[:,2],.8)


@pytest.mark.parametrize('component',[0,-1,3])
def test_invalid_target_cannot_remove_entire_scene(component):
    depth,mask,cal=calibrated_fixture()
    with pytest.raises(ValueError,match='target mask'):
        official.scene_points(depth,mask,component,cal,.001,[0,0,80,60])


def test_missing_scene_is_not_reported_as_official_all_pass():
    depth,mask,cal=calibrated_fixture()
    depth[:]=0;depth[10:25,10:25]=800
    with pytest.raises(ValueError,match='non-target scene'):
        official.scene_points(depth,mask,1,cal,.001,[0,0,80,60])


def test_partial_depth_is_recorded_not_filled():
    depth,mask,cal=calibrated_fixture();depth[45:,:]=0
    points,_,stats=official.scene_points(depth,mask,1,cal,.001,[0,0,80,60])
    assert len(points)==45*80-225
    assert 0<stats['non_target_valid_depth_fraction']<1


def result(reason='pass',desk='',passed=True):
    return SimpleNamespace(reason=reason,first_collision_phase=desk,passed=passed)


def sample(gap=.01,contact=False):
    return SimpleNamespace(min_desk_clearance_m=gap,object_collision=contact)


def test_target_proxy_overlap_is_uncertain_not_grasp_failure_or_auto_pass():
    state,reason=cb3.classify(result('object_collision_before_closure',passed=False),[sample(contact=True)],True)
    assert state=='uncertain_target_proxy' and reason=='target_proxy_contact_not_physical_failure'


@pytest.mark.parametrize('gap,reason',[(-.001,'desk_model_penetration'),(.001,'desk_clearance_below_2mm')])
def test_desk_contact_and_positive_insufficient_clearance_are_separate(gap,reason):
    assert cb3.classify(result('desk_clearance','approach',False),[sample(gap)],True)==('rejected_desk',reason)


def test_ik_workspace_and_proxy_failures_cannot_be_conflated():
    assert cb3.classify(result('ik_failed_pregrasp',passed=False),[sample()],True)[0]=='rejected_ik'
    assert cb3.classify(result(),[sample()],False)[0]=='rejected_workspace'
    assert cb3.classify(result(),[sample()],True)[0]=='passed_current_checks'


def test_official_manifest_binds_rank_order_and_candidate_bytes(tmp_path):
    candidates=tmp_path/'candidates.yml';candidates.write_text('original poses')
    report=dict(schema=official.SCHEMA,status='completed',executable=False,hardware_contacted=False,
        candidate_file=str(candidates),candidate_sha256=ui.digest(candidates),source_sha256={},
        candidates=[dict(rank=0,official_scene_clear=True),dict(rank=1,official_scene_clear=False)])
    path=tmp_path/'summary.json'
    path.write_text(json.dumps(report));assert len(cb3.load_filter(path,candidates)['candidates'])==2
    report['candidates'].reverse();path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='rank order'):cb3.load_filter(path,candidates)
    candidates.write_text('different poses')
    with pytest.raises(ValueError,match='mismatched'):cb3.load_filter(path,candidates)


def test_review_dispatches_modern_protocol_without_legacy_scene(monkeypatch,tmp_path):
    events=[]
    def modern(*a,**kw):events.append('cb3');yield 2,None,{}
    monkeypatch.setattr(cb3,'simulate',modern)
    args=SimpleNamespace(feasibility_mode='official-cb3')
    assert list(ui.simulate_candidates(args,tmp_path,{},tmp_path/'config',{}))==[(2,None,{})]
    assert events==['cb3']


def test_fixed_cb3_desk_height_does_not_follow_proxy_dimensions():
    pytest.importorskip('mujoco');pytest.importorskip('mink')
    from tools import ggx_real_support as support,run_ggx_real as real
    config=real.load_config(real.ROOT/'configs/ggx_real_lab.yaml')
    source=dict(preflight=dict(object_center_m=[-.23,-.65,.14],object_size_mm=[90,40,200],
        object_yaw_deg=0,table_height_m=.04),desk_top_m=.0435,desk_envelope=config['desk_envelope'])
    a=support.build_model(source)
    source['preflight']['table_height_m']=.08
    b=support.build_model(source)
    assert np.allclose(a.geom('floor').pos,b.geom('floor').pos)
    assert a.geom('floor').pos[2]==pytest.approx(.0435-1.)


@pytest.mark.parametrize('state',['uncertain_target_proxy','rejected_ik','not_evaluated'])
def test_uncertain_cb3_cannot_enter_real_handoff_even_with_a_pass_rank(tmp_path,state):
    pytest.importorskip('mujoco');pytest.importorskip('mink')
    from tools import ggx_real_support as support
    path=tmp_path/'preflight.json'
    path.write_text(json.dumps(dict(schema=cb3.SCHEMA,automatic_pass_ranks=[0],
        feasibility=dict(local_state=state))))
    with pytest.raises(ValueError,match='independent review'):
        support.read_candidate_source(tmp_path/'candidate.json',0,path,ui.pin([path]),'unreviewed_candidate')
