from types import SimpleNamespace
import numpy as np
from tools import audit_reflex_ik_branches as audit


def test_equivalent_wrist_angles_keep_pose_and_limits(monkeypatch):
    monkeypatch.setattr(audit.planning,'arm_addresses',lambda model:list(range(6)))
    model=SimpleNamespace(joint=lambda name:SimpleNamespace(id=audit.sim.ARM_JOINTS.index(name)),
        jnt_range=np.tile([-2*np.pi,2*np.pi],(6,1)),jnt_limited=np.ones(6,bool))
    q=np.array([.8,-1.2,1.8,5.8,.9,-3.2]);old=q.copy()
    variants=list(audit.wrist_equivalents(model,[q]))
    assert len(variants)==4
    assert np.array_equal(q,old)
    for branch,v in variants:
        assert branch==0
        assert np.all(np.abs(v)<=2*np.pi)
        assert np.allclose(np.sin(v),np.sin(q)) and np.allclose(np.cos(v),np.cos(q))
        assert np.array_equal(v[[0,1,2,4]],q[[0,1,2,4]])


def test_reverse_hint_uses_same_cartesian_segment(monkeypatch):
    monkeypatch.setattr(audit.planning,'arm_addresses',lambda model:list(range(6)))
    monkeypatch.setattr(audit.planning,'nearest_equivalent',lambda model,q,ref:q)
    calls=[]
    def solve(model,q,pose,args): calls.append(pose);return np.array(pose)
    monkeypatch.setattr(audit.files,'solve_tcp',solve)
    end=np.eye(4);end[2,3]=.3
    original=np.array([0.,0.,.3,0.,0.,0.]);point=dict(tcp=[0.,0.,.31,0.,0.,0.])
    result=audit.reverse_hint(None,original,point,dict(final_tcp_T=end.tolist()),None)
    assert np.allclose(result,point['tcp'])
    assert np.allclose(original,[0,0,.3,0,0,0])
    assert all(.3<c[2]<=.31 for c in calls)
