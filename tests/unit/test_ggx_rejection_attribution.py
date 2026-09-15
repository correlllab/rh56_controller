import numpy as np

from tools.audit_ggx_rejection_attribution import representatives, point_in_box_witness, common_interior_witness


def test_representatives_keep_original_rank_and_category_priority():
    rows = [{'rank': r, 'first_loss': 'local_approach_unresolved'} for r in [9, 4, 0]]
    details = {r: {'geometry': {'local_failures': [dict(kind='target_proxy_uncertain',
               phase='start' if r == 9 else 'approach', geom='thumb_distal/geom')]}} for r in [9, 4, 0]}
    assert representatives(rows, details) == {'thumb': [0, 4], 'pregrasp_overlap': [9]}


def test_vertex_witness_distinguishes_inside_touch_outside_in_rotated_box():
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    position = np.array([2., 3., 4.]); size = np.array([.1, .2, .3])
    for x, expected in [(.05, 50.), (.1, 0.), (.15, -50.)]:
        vertices = np.array([[x, 0., 0.]]) @ rotation.T + position
        witness = point_in_box_witness(vertices, position, rotation, size)
        assert np.isclose(witness['max_vertex_inside_margin_mm'], expected)
        assert witness['inside_vertex_count'] == int(expected > 0)


def test_common_interior_finds_crossing_boxes_without_inside_vertices():
    corners = np.array([(i,j,k) for i in [-1,1] for j in [-1,1] for k in [-1,1]])
    a = corners*[2., .1, .1]; b = corners*[.1, 2., .1]
    assert point_in_box_witness(a, np.zeros(3), np.eye(3), np.array([.1,2.,.1]))['inside_vertex_count'] == 0
    assert point_in_box_witness(b, np.zeros(3), np.eye(3), np.array([2.,.1,.1]))['inside_vertex_count'] == 0
    proof = common_interior_witness(a,b)
    assert proof['positive_interior_witness']
    assert np.isclose(proof['verified_interior_radius_mm'], 99.9999)
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_euler('xyz', [.3,.7,.2]).as_matrix(); offset = np.array([2.,-1.,3.])
    moved = common_interior_witness(a@rotation.T+offset,b@rotation.T+offset)
    assert np.isclose(moved['verified_interior_radius_mm'], proof['verified_interior_radius_mm'])


def test_no_positive_witness_for_separation_or_touch():
    box = np.array([(i,j,k) for i in [-1,1] for j in [-1,1] for k in [-1,1]], dtype=float)
    for shift in [2., 3.]:
        assert not common_interior_witness(box,box+[shift,0.,0.])['positive_interior_witness']
