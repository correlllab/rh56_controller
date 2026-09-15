"""Offline lab mapping: object anchor, historical offsets, and shared relocation."""
import numpy as np
import pytest
from tools import ggx_fixed_starts as fixed, ggx_template as template, run_ggx_real as real


def test_object_anchor_mapping_and_translation_preserve_ten_points():
    anchor=np.array([-192.4,-707.2,185.3])
    layout=fixed.build(real.ROOT/'configs/reflex_ten_starts_object_relative_lab.yaml',
        real.load_config(real.ROOT/'configs/reflex_real_lab.yaml'),
        real.ROOT/'configs/tabletop_vision_lab.yaml',grasp_center_mm=anchor)
    points=layout['points']
    assert [p['id'] for p in points]==[f'P{i}' for i in range(1,11)]
    assert np.allclose(points[0]['center_base_mm'],anchor+[0,250,0])
    assert np.allclose(points[1]['center_base_mm'],anchor+[0,0,250])
    old=np.array([p['offset_mm'] for p in template.paper_points()])
    new=np.array([p['center_base_mm'] for p in points])
    assert np.allclose(np.linalg.norm(old[:,None]-old[None,:],axis=-1),
                       np.linalg.norm(new[:,None]-new[None,:],axis=-1))
    for p in points:
        q=fixed.translate_point(p,[.003,-.004])
        assert np.allclose(np.array(q['center_base_mm'])-p['center_base_mm'],[3,-4,0])
        assert np.allclose(np.array(q['tcp'])-p['tcp'],[.003,-.004,0,0,0,0])
        assert q['offset_mm']==p['offset_mm']
        assert q['ik_status']=='unverified_seed_only'
    assert np.allclose(points[0]['center_base_mm'],anchor+[0,250,0])


def test_object_layout_requires_anchor():
    with pytest.raises(ValueError,match='finite 3-vector'):
        fixed.build(real.ROOT/'configs/reflex_ten_starts_object_relative_lab.yaml',{},
                    real.ROOT/'configs/tabletop_vision_lab.yaml')
