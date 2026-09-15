"""Historical fixed-layout and explicit object-relative P1..P10; no hardware."""
from pathlib import Path
import numpy as np
import yaml
import mujoco

from tools import ggx_template as t, ggx_reset as reset, ggx_real_support as support
from tools.preview_tabletop_target import load_perception_config
from tools.run_ggx_review import pin


def build(path, config, perception_path, *, grasp_center_mm=None):
    settings = yaml.safe_load(Path(path).read_text())
    if isinstance(settings,dict) and settings.get('schema')=='rh56_object_relative_starts_config/v1':
        return object_relative(path,settings,config,perception_path,grasp_center_mm)
    keys = {'schema', 'frame', 'p1_tcp_xy_mm', 'thumb_clearance_mm', 'tcp_rotvec', 'seed_q'}
    if (not isinstance(settings, dict) or set(settings) != keys
            or settings['schema'] != 'rh56_ggx_fixed_starts_config/v1' or settings['frame'] != 'ur_base'):
        raise ValueError('Expected fixed-start setup config v1')
    xy, rv, seed_q = (t.vector(settings[k], n) for k, n in
                     [('p1_tcp_xy_mm', 2), ('tcp_rotvec', 3), ('seed_q', 6)])
    margin = settings['thumb_clearance_mm']
    if isinstance(margin, bool) or not np.isfinite(margin) or not 10 <= margin <= 100:
        raise ValueError('Invalid fixed-start thumb clearance')
    perception, _ = load_perception_config(perception_path)
    scene = reset.desk_source(config, perception)
    model = support.build_model(scene, include_object=False)
    seed = support.sim.initial_qpos(model, seed_q)
    data = mujoco.MjData(model)
    data.qpos[:] = seed
    mujoco.mj_forward(model, data)
    tcp = support.pose_matrix(support.fk_tcp(model, seed_q))
    thumb_local = (np.linalg.inv(tcp) @ np.r_[data.site_xpos[model.site('right_thumb_tip').id], 1])[:3]
    center_local = t.open_center_tcp(model, dict(q_nominal=seed_q, tcp=support.matrix_pose(tcp)))
    orientation = support.pose_matrix([0, 0, 0, *rv])[:3, :3]
    # Conservative desk top bound over the whole fixed envelope, not object height.
    p1z = scene['preflight']['table_height_m'] + margin/1000 - (orientation @ thumb_local)[2]
    p1tcp = np.r_[xy/1000, p1z]
    origin_tcp = p1tcp - np.array(t.paper_points()[0]['offset_mm'])/1000
    points, addresses = [], [support.sim._joint_qpos_address(model, n) for n in support.sim.ARM_JOINTS]
    args = support.sim.parse_args([])
    for p in t.paper_points():
        target = [*(origin_tcp + np.array(p['offset_mm'])/1000), *rv]
        qpos = t.solve_tcp(model, seed, target, args)
        q = qpos[addresses]
        actual = support.fk_tcp(model, q)
        t.motion.assert_inside(actual, config)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        clearance, nearest = support.sim.desk_clearance(model, data)
        if clearance < .002:
            raise ValueError(f"Fixed layout {p['id']} hits setup: {nearest}; change setup config, not a candidate")
        thumb_z = data.site_xpos[model.site('right_thumb_tip').id, 2]
        points.append(dict(p, tcp=target, q_nominal=q.tolist(), fixed_orientation=True,
            center_base_mm=((np.array(target[:3]) + orientation @ center_local)*1000).tolist(),
            setup_desk_clearance_mm=clearance*1000,
            thumb_clearance_mm=(thumb_z-scene['preflight']['table_height_m'])*1000,
            start_error_mm=support.pose_error(actual, target)[0]))
    return dict(schema='rh56_ggx_fixed_starts/v1', executable=False,
        frame='ur_base', settings=settings, points=points, setup=scene,
        source_sha256=pin([path, perception_path, perception['calibration'], perception['desk_plane']]),
        origin_grasp_center_mm=((origin_tcp + orientation @ center_local)*1000).tolist(),
        assumptions=['Fixed positions, orientation and GGX open hand for ALL objects/candidates.',
            'Historical pairwise vectors preserved; no per-object translation or optimization.',
            'Thumb target is relative to conservative desk-top bound, not measured physical clearance.',
            'All arm/hand geometry checked at starts against fixed desk; unknown fixtures/self-collision not certified.',
            'Actual object and travel-path collision are evaluated separately per trial; setup endpoints alone do not authorize travel.'])


def object_relative(path,settings,config,perception_path,grasp_center_mm):
    """Paper centers anchored to the object grasp point, with one fixed lab map.

    Legacy paper y- becomes base y+ (toward the arm); lateral x becomes -x.
    This is a rigid 180-degree lab-frame rotation, not candidate optimization.
    The common start hand orientation is retained; per-trial IK is mandatory.
    """
    if (set(settings)!={'schema','frame','paper_to_base_yaw_deg','tcp_rotvec','seed_q'}
            or settings['frame']!='object_grasp_center_base_axes'
            or settings['paper_to_base_yaw_deg']!=180.):
        raise ValueError('Expected confirmed object-relative y+ lab mapping')
    anchor=t.vector(grasp_center_mm,3)
    rv=t.vector(settings['tcp_rotvec'],3);seed=t.vector(settings['seed_q'],6)
    perception,_=load_perception_config(perception_path)
    scene=reset.desk_source(config,perception)
    model=support.build_model(scene,include_object=False)
    center_local=t.open_center_tcp(model,dict(q_nominal=seed.tolist(),tcp=support.fk_tcp(model,seed)))
    orientation=support.pose_matrix([0,0,0,*rv])[:3,:3]
    rotation=np.diag([-1.,-1.,1.])
    points=[]
    for p in t.paper_points():
        offset=rotation@np.asarray(p['offset_mm'])
        center=anchor+offset
        tcp=np.r_[center/1000-orientation@center_local,rv]
        points.append(dict(p,paper_offset_mm=p['offset_mm'],offset_mm=offset.tolist(),
            center_base_mm=center.tolist(),tcp=tcp.tolist(),q_nominal=seed.tolist(),
            fixed_orientation=True,ik_status='unverified_seed_only'))
    return dict(schema='rh56_object_relative_starts/v1',executable=False,
        frame=settings['frame'],settings=settings,points=points,setup=scene,
        origin_grasp_center_mm=anchor.tolist(),
        source_sha256=pin([path,perception_path,perception['calibration'],perception['desk_plane']]),
        assumptions=['Object grasp reference is the common anchor, NOT fixed desk XYZ or object AABB center.',
            'P1=anchor+[0,250,0] mm; P2=anchor+[0,0,250] mm; historical distances/order unchanged.',
            'Fixed paper-to-base 180 degree rotation, no per-object/candidate yaw optimization.',
            'Common scan/open hand orientation retained; not a claim of historical strategy-specific start orientation.',
            'q_nominal entries are seeds only; every relocated point requires fresh IK and complete path checks.'])


def translate_point(point,delta_xy_m):
    """Move target and starts by the SAME measured XY; never alter frozen yaw/Z."""
    from copy import deepcopy
    delta=t.vector(delta_xy_m,2)
    result=deepcopy(point)
    for key,scale in [('tcp',1.),('center_base_mm',1000.)]:
        result[key][:2]=(np.asarray(result[key][:2])+delta*scale).tolist()
    result['xy_translation_m']=delta.tolist()
    return result
