"""Frozen-audit contracts; no lab data, GPU, camera or robot needed."""
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from tools import run_ggx_frozen_audit as audit
from tools import run_ggx_cube_inference as inference


def test_pre_topk_archive_preserves_raw_ids_and_selected_output(tmp_path,monkeypatch):
    module=ModuleType('graspgenx.dataset.eval_utils')
    module.save_to_isaac_grasp_format=lambda poses,scores,path:None
    monkeypatch.setitem(sys.modules,'graspgenx.dataset.eval_utils',module)
    raw=np.repeat(np.eye(4)[None],4,axis=0);raw[:,0,3]=np.arange(4)
    scores=np.array([.2,.9,.9,.4]);mean=np.array([1.,2.,3.])
    expected=raw[[1,2]].copy();expected[:,:3,3]+=mean
    result=inference.archive_candidates(tmp_path,[(raw,scores)],mean,expected,scores[[1,2]],2)
    data=np.load(tmp_path/'all_candidates.npz')
    assert result['raw_count']==4 and result['topk_verified_identical']
    np.testing.assert_array_equal(data['poses_sampler_frame'],raw)
    assert data['rank_to_generation_id'].tolist()==[1,2,3,0]
    assert data['selected_generation_ids'].tolist()==[1,2]
    with pytest.raises(ValueError,match='differs'):
        inference.archive_candidates(tmp_path,[(raw,scores)],mean,expected+.1,scores[[1,2]],2)


def test_bundle_detects_mutated_snapshot(tmp_path):
    snapshot=tmp_path/'input.json';snapshot.write_text('{}')
    audit.write(tmp_path/'manifest.json',{'input.json':audit.digest(snapshot)})
    snapshot.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='Frozen file changed'):audit.bundle_inputs(tmp_path)


def test_bundle_detects_checker_drift(tmp_path):
    code=tmp_path/'checker.py';code.write_text('before')
    audit.write(tmp_path/'inputs.json',dict(current_source_sha256={str(code):audit.digest(code)}))
    audit.write(tmp_path/'manifest.json',{'inputs.json':audit.digest(tmp_path/'inputs.json')})
    code.write_text('after')
    with pytest.raises(ValueError,match='Checker/source changed'):audit.bundle_inputs(tmp_path)


def test_reference_nearness_is_not_identity():
    poses=np.repeat(np.eye(4)[None],2,axis=0);poses[1,0,3]=.004
    distance,angle=audit.compare_reference(poses,np.eye(4))
    assert distance.tolist()==[0,4] and angle.tolist()==[0,0]


def test_audit_has_no_execute_option():
    with pytest.raises(SystemExit):audit.parse_args(['freeze','--out','unused','--execute'])


def test_local_audit_does_not_inherit_ui_target_omission(monkeypatch):
    pytest.importorskip('mujoco');pytest.importorskip('mink')
    from tools import ggx_planning as plan, preflight_ggx_ur5_table as sim
    monkeypatch.setattr(sim,'candidate_for_object_origin',lambda c,*a:c)
    monkeypatch.setattr(plan,'select_geometry',lambda *a:({'state':'geometry_eligible'},['same_sample']))
    def check(model,samples,config,*,target_gate):
        assert samples==['same_sample'] and target_gate is True
        return [{'kind':'target_proxy_uncertain','phase':'approach'}]
    monkeypatch.setattr(plan,'geometry_failures',check)
    _,_,result=audit.local_check(None,None,0,None,None,{})
    assert result['initial_review_eligible'] and not result['unified_local_pass']


def test_merge_requires_complete_nonduplicate_rank_coverage(tmp_path):
    from tools.summarize_ggx_frozen_audit import merge_shards
    paths=[]
    for index in range(2):
        path=tmp_path/f'{index}.json';paths.append(path)
        audit.write(path,dict(status='completed',pose_convention='official_export_yaml_readback',rows=[{'rank':index}]))
    assert merge_shards(paths,2)==[{'rank':0},{'rank':1}]
    with pytest.raises(ValueError,match='Missing or duplicated'):merge_shards(paths[:1],2)
    with pytest.raises(ValueError,match='Missing or duplicated'):merge_shards([paths[0],paths[0]],2)


def test_exported_readback_keeps_generation_identity(tmp_path,monkeypatch):
    from rh56_controller import graspgenx_baseline as baseline
    np.savez(tmp_path/'all_candidates.npz',confidence=np.array([.2,.9]),rank_to_generation_id=[1,0])
    candidates=[SimpleNamespace(confidence=.9,rotation=np.eye(3),position=np.array([1.,2.,3.])),
                SimpleNamespace(confidence=.2,rotation=np.eye(3),position=np.zeros(3))]
    monkeypatch.setattr(baseline,'load_isaac_grasp_yaml',lambda p:candidates)
    _,poses=audit.exported_candidates(tmp_path)
    np.testing.assert_array_equal(poses[1,:3,3],[1.,2.,3.])
    np.testing.assert_array_equal(poses[0,:3,3],[0.,0.,0.])
