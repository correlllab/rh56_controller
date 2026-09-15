"""Offline GGX -> nominal CB3 preparation. No hardware imports or commands."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree
import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from tools import preflight_ggx_ur5_table as sim
from tools.run_ggx_review import digest, pin, read_json, verify
from rh56_controller.ur5_bridge import UR5Bridge

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / 'magpie_force_control/src/ur_description/ur5.urdf'
HAND_ORDER = ('pinky', 'ring', 'middle', 'index', 'thumb_proximal', 'thumb_yaw')
OFFICIAL_SELECTION_SCHEMA = 'rh56_ggx_official_candidate_selection/v1'
OFFICIAL_SELECTION_SCOPE = 'official_ggx_candidate'


def matrix_pose(T):
    return np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()].tolist()


def pose_matrix(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError('Expected finite six-component TCP pose')
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    return T


def pose_error(first, second):
    a, b = pose_matrix(first), pose_matrix(second)
    return (float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000),
            float(np.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())))


def read_approval(path):
    approval = read_json(path)
    if (approval.get('status') == 'manual_review_accepted_not_executable'
            or any(r.get('decision') == 'approve' and r.get('scope') == 'manual_object_collision'
                   for r in approval.get('reviews', []))):
        raise ValueError('Manual object-collision review is not a real-motion approval; collision remains unresolved')
    if (approval.get('schema') != 'rh56_ggx_interactive_review/v1'
            or approval.get('status') not in ('review_approved_execution_unavailable', 'review_approved',
                                              'official_candidate_selected')
            or approval.get('executable') is not False):
        raise ValueError('Expected saved visual-only GGX approval')
    approved = [r for r in approval['reviews'] if r['decision'] == 'approve']
    if len(approved) != 1 or approved[0]['rank'] != approval['selected_rank']:
        raise ValueError('Approval must identify exactly one candidate')
    row = approved[0]
    if row.get('candidate_report') is not None and row.get('scope') != OFFICIAL_SELECTION_SCOPE:
        raise ValueError('Explicit candidate report is only valid for official candidate preview')
    preflight_path = (Path(row['candidate_report']) if row.get('candidate_report') is not None
                      else Path(row['video']).parent / 'summary.json')
    if row.get('local_preview_report') is not None:
        verify(row['reviewed_sha256'])
        local_path = Path(row['local_preview_report']).resolve()
        if (str(local_path) not in row['reviewed_sha256']
                or str(Path(row['video']).resolve()) not in row['reviewed_sha256']):
            raise ValueError('Local preview/report not bound to the reviewed candidate')
        local = read_json(local_path)
        if (local.get('rank') != row['rank'] or local.get('executable') is not False
                or local.get('usable_for_motion') is not False
                or Path(local.get('video') or '').resolve() != Path(row['video']).resolve()):
            raise ValueError('Local diagnostic preview cannot become motion permission')
    official = read_json(preflight_path).get('schema') == OFFICIAL_SELECTION_SCHEMA
    if (official != (approval['status'] == 'official_candidate_selected')
            or official != (row.get('scope') == OFFICIAL_SELECTION_SCOPE)):
        raise ValueError('Official candidate selection scope/status does not match its report')
    return read_candidate_source(path, row['rank'], preflight_path,
                                 row['reviewed_sha256'], 'visual_approval')


def read_candidate(path):
    """Unreviewed automatic pass: permits preparation, NEVER motion by itself."""
    proposal = read_json(path)
    if (proposal.get('schema') != 'rh56_ggx_execution_candidate/v1'
            or proposal.get('human_approved') is not False
            or proposal.get('executable') is not False
            or type(proposal.get('rank')) is not int or proposal['rank'] < 0):
        raise ValueError('Expected unapproved automatic-pass execution candidate')
    return read_candidate_source(path, proposal['rank'], Path(proposal['preflight']),
                                 proposal['source_sha256'], 'unreviewed_candidate')


def read_candidate_source(path, rank, preflight_path, source_pins, source_kind):
    verify(source_pins)
    if str(preflight_path.resolve()) not in source_pins:
        raise ValueError('Approval does not bind its simulation report')
    report = read_json(preflight_path)
    geometry_only = report.get('schema') == 'rh56_ggx_cb3_grasp_geometry/v1'
    official_only = report.get('schema') == OFFICIAL_SELECTION_SCHEMA
    if official_only:
        if (report.get('selection_scope') != OFFICIAL_SELECTION_SCOPE
                or report.get('official_review_ranks') != [rank]
                or report.get('passed') is not False or report.get('executable') is not False
                or report.get('automatic_pass_ranks') != []
                or report.get('local_execution_checked') is not False
                or report.get('actual_travel_checked') is not False):
            raise ValueError('Invalid official-only candidate; no local/path approval may be inherited')
        # Use the SAME strict endpoint/top-100 reader as the review UI. A
        # direct preparation handoff must not accept a weaker record than
        # reopening the saved candidate would. Re-derived pins may verify the
        # old approval, never silently replace or extend what it actually bound.
        from tools.ggx_official_candidates import reviewed_candidates
        reopened = list(reviewed_candidates(preflight_path.parent, require_video=False))
        if len(reopened) != 1 or reopened[0][0] != rank:
            raise ValueError('Official handoff rank differs from selected candidate')
        for dependency, expected in reopened[0][2].items():
            if source_pins.get(dependency) != expected:
                raise ValueError('Official handoff provenance differs from saved selection')
    if geometry_only:
        from tools.ggx_planning import policy
        if (rank not in report.get('geometry_review_ranks', [])
                or report.get('geometry', {}).get('state') != 'geometry_eligible'
                or report.get('geometry', {}).get('local_execution_check', {}).get('passed') is not True
                or report.get('actual_travel_checked') is not False
                or report.get('automatic_pass_ranks') != [] or report.get('policy') != policy()):
            raise ValueError('Invalid geometry-only candidate; cannot inherit path approval')
    if (report.get('schema')=='rh56_ggx_cb3_scene_preflight/v1'
            and (report.get('feasibility',{}).get('local_state')!='passed_current_checks'
                 or rank in report.get('uncertain_ranks',[]))):
        raise ValueError('Uncertain/rejected CB3 candidate requires independent review, not automatic execution')
    if ((not official_only and report.get('object_collision_gate', {}).get('revision') != sim.OBJECT_GATE_REVISION)
            or (not official_only and not geometry_only and rank not in report['automatic_pass_ranks'])
            or report.get('candidate_frame') != 'ur-base'):
        raise ValueError('Approval lacks current UR-base approach collision precheck')
    candidate_file = Path(report['candidate_yaml'])
    if digest(candidate_file) != report['candidate_yaml_sha256']:
        raise ValueError('Candidate YAML changed')
    inference_path = candidate_file.parent / 'summary.json'
    inference = read_json(inference_path)
    if (inference.get('schema') != 'rh56_ggx_point_cloud_inference/v1'
            or inference.get('input_frame') != 'ur_base'
            or inference.get('candidate_sha256') != digest(candidate_file)):
        raise ValueError('Missing matching measured-cloud inference provenance')
    cloud = Path(inference['source_point_cloud'])
    if digest(cloud) != inference['source_point_cloud_sha256']:
        raise ValueError('Inference point cloud changed')
    crop_path = cloud.parent / 'summary.json'
    crop = read_json(crop_path)
    perception_path = Path(crop['source_preview'])
    from tools.prepare_ggx_tabletop_cloud import FOREGROUND_SCHEMA, validate_foreground_report
    if (crop.get('schema') not in ('rh56_ggx_measured_crop/v1', FOREGROUND_SCHEMA)
            or Path(crop['point_cloud']).resolve() != cloud.resolve()
            or digest(perception_path) != crop['source_preview_sha256']):
        raise ValueError('Crop/perception provenance mismatch')
    foreground_path = None
    if crop['schema'] == FOREGROUND_SCHEMA:
        validate_foreground_report(crop)
        foreground_path = perception_path.parent / 'foreground.png'
        if digest(foreground_path) != crop.get('source_foreground_sha256'):
            raise ValueError('Reviewed foreground mask changed')
    perception = read_json(perception_path)
    if digest(perception['calibration']) != crop['calibration_sha256']:
        raise ValueError('Calibration changed')
    capture = Path(perception['capture'])
    if digest(capture / 'camera_check_depth_u16.png') != crop['source_depth_sha256']:
        raise ValueError('Reference depth changed')
    pins = dict(source_pins)
    pins.update(pin([path, inference_path, crop_path, perception_path, cloud,
                     perception['config'], perception['calibration'], perception['desk_plane'],
                     capture / 'camera_check_rgb.png', capture / 'camera_check_depth_u16.png',
                     capture / 'camera_check.yaml']))
    if foreground_path is not None:
        pins.update(pin([foreground_path]))
        # Both simulation and CB3 must use the same enclosing proxy as the crop,
        # not silently fall back to the smaller operator dimensions.
        from tools.preview_tabletop_target import load_perception_config
        from tools.run_ggx_review import crop_sim_config
        perception_config, _ = load_perception_config(Path(perception['config']))
        expected = crop_sim_config(crop, crop['dimensions_prior_mm'], perception_config,
                                   {'q': [0.] * 6}, candidate_file)
        for key in ('object_size_mm', 'object_yaw_deg', 'table_height_m'):
            if not np.allclose(report[key], expected[key], rtol=0, atol=1e-9):
                raise ValueError(f'Approved foreground collision proxy changed: {key}')
        expected_center = [expected['object_x_m'], expected['object_y_m'],
                           expected['table_height_m'] + expected['object_size_mm'][2] / 2000]
        if not np.allclose(report['object_center_m'], expected_center, rtol=0, atol=1e-9):
            raise ValueError('Approved foreground collision proxy center changed')
    if perception.get('background'):
        background = Path(perception['background'])
        pins.update(pin([background / name for name in (
            'camera_check.yaml', 'camera_check_depth_u16.png', 'depth_statistics.npz')]))
    source=dict(rank=rank, preflight=report, crop=crop, perception=perception,
                pins=pins, approval=str(Path(path).resolve()), source_kind=source_kind)
    if official_only or geometry_only or report.get('schema') == 'rh56_ggx_cb3_scene_preflight/v1':
        from tools.ggx_cb3_candidates import load_filter
        from tools.run_ggx_real import load_config
        from tools.ggx_reset import desk_source
        from tools.preview_tabletop_target import load_perception_config
        verify(report['source_sha256'])
        filter_path=Path(report['official_filter_report']).resolve()
        if str(filter_path) not in source_pins:
            raise ValueError('CB3 approval does not bind official scene filter')
        official=load_filter(filter_path,candidate_file)
        if (not official['candidates'][rank]['official_scene_clear']
                or (not official_only and not geometry_only
                    and report.get('feasibility',{}).get('local_state')!='passed_current_checks')
                or (not official_only and report.get('robot_model')!='CB3 UR5')):
            raise ValueError('Official reject/uncertain candidate is not a motion approval')
        config=load_config(Path(report['real_config']))
        pc,_=load_perception_config(Path(perception['config']))
        desk=desk_source(config,pc)
        fixed_top = desk['preflight']['table_height_m']
        if official_only:
            for key in ('real_config', 'perception_config'):
                if str(Path(report[key]).resolve()) not in source_pins:
                    raise ValueError(f'Official selection does not bind {key}')
            if Path(report['perception_config']).resolve() != Path(perception['config']).resolve():
                raise ValueError('Official selection perception configuration changed')
        if (report['desk_envelope']!=config['desk_envelope']
                or (not official_only
                    and not np.isclose(report['desk_top_m'],fixed_top,rtol=0,atol=1e-12))):
            raise ValueError('Approved CB3 fixed desk changed')
        pins.update(report['source_sha256'])
        source.update(desk_top_m=fixed_top,desk_envelope=report['desk_envelope'],
            feasibility_protocol='official_scene_plus_cb3/v1',checker_config=str(Path(report['real_config']).resolve()))
        if geometry_only:
            source.update(selection_scope='grasp_geometry',planning_policy=report['policy'])
        elif official_only:
            source.update(selection_scope=OFFICIAL_SELECTION_SCOPE,
                feasibility_protocol='official_scene_candidate_then_local/v1',
                official_scene_clear=True, local_execution_checked=False,
                physical_hand_verified=False)
    return source


def _quat(rotation):
    return Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]


def cb3_spec():
    """Compose repository UR5 URDF with unchanged RH56 subtree/nominal mount.

    Import the URDF instead of retuning UR5e DH parameters or copying sim joints
    onto a different robot. Real controller calibration is still checked later.
    """
    tree = ElementTree.parse(URDF).getroot()
    def origin(name):
        node = tree.find(f"joint[@name='{name}']/origin")
        if node is None:
            raise ValueError(f'Missing URDF origin: {name}')
        T = np.eye(4)
        T[:3, 3] = np.fromstring(node.attrib.get('xyz', '0 0 0'), sep=' ')
        T[:3, :3] = Rotation.from_euler('xyz', np.fromstring(node.attrib.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        return T
    arm = mujoco.MjSpec.from_file(str(URDF))
    arm.meshdir = str(URDF.parent / 'meshes/ur5/collision')
    root = mujoco.MjSpec()
    root.modelname = 'GGX nominal CB3 UR5 + unchanged RH56'
    base = np.linalg.inv(origin('base_link-base_fixed_joint'))
    frame = root.worldbody.add_frame(pos=base[:3, 3], quat=_quat(base[:3, :3]))
    root.attach(arm, frame=frame, prefix='cb3_')
    for joint in root.joints:
        joint.name = joint.name.removeprefix('cb3_')
    for body in root.bodies:
        body.name = body.name.removeprefix('cb3_')
    # The bolted base is a permanent fixture; do not test it against its desk.
    for body in root.bodies:
        if body.name in ('world', 'base_link', 'base_link_inertia', 'base'):
            for geom in body.geoms:
                geom.contype = geom.conaffinity = 0
    wrist_to_tcp = origin('wrist_3-flange') @ origin('flange-tool0')
    wrist = root.body('wrist_3_link')
    wrist.add_site(name='tcp', pos=wrist_to_tcp[:3, 3], quat=_quat(wrist_to_tcp[:3, :3]))
    mount = wrist_to_tcp @ UR5Bridge.tcp_T_to_hand_T(np.eye(4))
    hand_frame = wrist.add_frame(pos=mount[:3, 3], quat=_quat(mount[:3, :3]))
    hand_source = mujoco.MjSpec.from_file(str(sim.DEFAULT_XML))
    hand_body = hand_source.body('gripper_attachment')
    hand_body.pos, hand_body.quat = [0, 0, 0], [1, 0, 0, 0]
    hand_frame.attach_body(hand_body, prefix='', suffix='')
    root.worldbody.add_light(pos=[0, -1, 2], dir=[0, 0, -1])
    return root


def desk_envelope_geometry(envelope):
    """Yaw-independent enclosing disk from explicit operator bounds, not a fit.

    Half diagonal bounds every rectangle corner; triangle inequality adds
    center displacement and reference uncertainty. The fixed reference must
    not follow objects that are moved in later captures.
    """
    required = {'center_base_xy_m', 'sides_upper_mm', 'center_offset_upper_mm',
                'reference_margin_mm', 'confirmed'}
    if not isinstance(envelope, dict) or set(envelope) != required:
        raise ValueError('Invalid desk envelope fields')
    center = np.asarray(envelope['center_base_xy_m'], dtype=float)
    sides = np.asarray(envelope['sides_upper_mm'], dtype=float)
    offset, margin = envelope['center_offset_upper_mm'], envelope['reference_margin_mm']
    if (center.shape != (2,) or sides.shape != (2,)
            or not np.all(np.isfinite(np.r_[center, sides]))
            or np.any(np.abs(center) > 2) or np.any(sides <= 0) or np.any(sides > 2000)
            or type(envelope['confirmed']) is not bool
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not np.isfinite(v) for v in (offset, margin))
            or not 0 <= offset <= 500 or not 10 <= margin <= 100):
        raise ValueError('Invalid desk envelope bounds')
    return center, float(np.linalg.norm(sides) / 2000 + (offset + margin) / 1000)


def build_model(source, *, include_object=True):
    """Build the fixed desk and CB3; reset may explicitly omit unknown objects."""
    old = source['preflight']
    desk_top = source.get('desk_top_m', old['table_height_m'])
    if not np.isfinite(desk_top):
        raise ValueError('Invalid fixed desk top')
    spec = cb3_spec()
    edge = source.get('desk_edge_y_m')
    envelope = source.get('desk_envelope')
    plane=source.get('desk_surface_plane')
    if plane is not None and (envelope is None or edge is not None):
        raise ValueError('Calibrated finite plane requires the fixed desk envelope')
    if envelope is not None:
        if edge is not None:
            raise ValueError('Choose desk edge OR envelope, not both')
        center, radius = desk_envelope_geometry(envelope)
        if include_object:
            object_radius = np.linalg.norm(np.asarray(old['object_size_mm'][:2])) / 2000
            if np.linalg.norm(np.asarray(old['object_center_m'])[:2] - center) + object_radius >= radius:
                raise ValueError('Desk envelope does not conservatively support object')
        # Circumscribed (not inscribed!) convex prism contains the whole disk.
        # 64 sides, 1 micrometre outward allowance for mesh compilation roundoff.
        n = 64
        vertex_radius = (radius + 1e-6) / np.cos(np.pi / n)
        angles = np.arange(n) * (2 * np.pi / n)
        vertices = np.array([[vertex_radius*np.cos(a), vertex_radius*np.sin(a), z]
                             for z in (-1., 1.) for a in angles])
        if plane is not None:
            if not isinstance(plane,dict) or set(plane)!={'normal_base','point_base_m'}:
                raise ValueError('Invalid calibrated desk plane fields')
            normal=np.asarray(plane['normal_base'],dtype=float)
            point=np.asarray(plane['point_base_m'],dtype=float)
            if (normal.shape!=(3,) or point.shape!=(3,) or not np.all(np.isfinite(np.r_[normal,point]))
                    or normal[2]<=1e-6 or not np.isclose(np.linalg.norm(normal),1.,atol=1e-8,rtol=0)):
                raise ValueError('Invalid calibrated desk plane geometry')
            center_z=point[2]-np.dot(normal[:2],center-point[:2])/normal[2]
            if not np.isclose(center_z,desk_top,atol=1e-9,rtol=0):
                raise ValueError('Desk reference height disagrees with calibrated plane')
            # Vertical shear preserves EXACTLY the existing conservative XY
            # footprint and 2 m thickness; only the top/bottom planes tilt.
            vertices[:,2]-=vertices[:,:2]@normal[:2]/normal[2]
        spec.add_mesh(name='desk_envelope_mesh', uservert=vertices.ravel().tolist())
        spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_MESH,
                               meshname='desk_envelope_mesh',
                               pos=[*center, desk_top - 1], rgba=[.25, .3, .35, 1])
    elif edge is None:
        spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                               pos=[0, 0, desk_top], size=[2, 2, .01],
                               rgba=[.25, .3, .35, 1])
    else:
        # Explicit measured boundary only: desk occupies y <= edge, extending
        # conservatively 4 m back/sideways and 2 m down. Never infer from a pass.
        if not np.isfinite(edge):
            raise ValueError('Invalid measured desk edge')
        if include_object and (old['object_center_m'][1]
                + np.linalg.norm(np.asarray(old['object_size_mm'][:2])) / 2000 >= edge):
            raise ValueError('Measured desk edge does not conservatively support object')
        spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_BOX,
                               pos=[0, edge - 2, desk_top - 1],
                               size=[2, 2, 1], rgba=[.25, .3, .35, 1])
    if include_object:
        body = spec.worldbody.add_body(name='preflight_object', pos=old['object_center_m'],
            quat=_quat(Rotation.from_euler('z', old['object_yaw_deg'], degrees=True).as_matrix()))
        body.add_geom(name='preflight_object_visual', type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=(np.asarray(old['object_size_mm']) / 2000).tolist(),
                      rgba=[.9, .15, .05, 1], contype=1, conaffinity=1)
    return spec.compile()


def fk_tcp(model, q):
    data = mujoco.MjData(model)
    data.qpos[:] = sim.initial_qpos(model, np.asarray(q))
    mujoco.mj_forward(model, data)
    site = model.site('tcp').id
    T = np.eye(4)
    T[:3, 3] = data.site_xpos[site]
    T[:3, :3] = data.site_xmat[site].reshape(3, 3)
    return matrix_pose(T)


def raw_hand_command(model, alpha):
    # Reuse the existing real executor conversion, sourcing ranges from this
    # exact reviewed model rather than rebuilding a separate FK-table model.
    from rh56_controller.grasp_executor import GraspExecutor
    ranges = {name: model.actuator(name).ctrlrange for name in HAND_ORDER}
    fk = SimpleNamespace(ctrl_min={k: float(v[0]) for k, v in ranges.items()},
                         ctrl_max={k: float(v[1]) for k, v in ranges.items()})
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Closure fraction must be finite and in [0, 1]')
    ctrl = sim._lerp_ctrl(alpha)
    for name in HAND_ORDER:
        lo, hi = ranges[name]
        if not lo - 1e-12 <= ctrl[name] <= hi + 1e-12:
            raise ValueError(f'GGX hand command outside model range: {name}')
        ctrl[name] = float(np.clip(ctrl[name], lo, hi))
    return GraspExecutor(None, None, fk, hand_write_only=True)._ctrl_to_real(ctrl)


def prepare(source, snapshot, out, video=False):
    """Replan selected GGX pose on nominal CB3, preserving closure on lift."""
    verify(source['pins'])
    if source.get('feasibility_protocol') in ('official_scene_plus_cb3/v1',
                                             'official_scene_candidate_then_local/v1'):
        from tools.run_ggx_real import load_config
        expected=load_config(Path(source['checker_config']))
        if source.get('desk_envelope')!=expected['desk_envelope'] or source.get('desk_edge_y_m') is not None:
            raise ValueError('CB3 desk cannot change between selection and execution')
    model = build_model(source)
    if source.get('selection_scope') == OFFICIAL_SELECTION_SCOPE:
        return prepare_official_candidate(source, snapshot, out, model, video=video)
    if source.get('selection_scope') == 'grasp_geometry':
        from tools import ggx_planning as planning
        from tools.run_ggx_real import load_config
        from tools.ggx_template import write_json
        # A direct real-run still needs its own actual current->grasp path.
        # Only template.create uses planning.reference without this travel.
        reference_dir = out/'geometry_reference';reference_dir.mkdir()
        ref,_ = planning.reference(source,reference_dir)
        rows = [r for r in ref['rows'] if r['phase'] not in ('start','start_to_pregrasp')]
        pre = [r for r in ref['rows'] if r['phase']=='start_to_pregrasp'][-1]
        return planning.execution(model,source,ref,[pre,*rows],snapshot,
                                  load_config(Path(source['checker_config'])),out,video=video)
    args = sim.parse_args([])
    old = source['preflight']
    for key, value in old['arguments'].items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.video = video
    args.out = out
    q = np.asarray(snapshot['q'], dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError('Invalid CB3 start joints')
    center = np.asarray(old['object_center_m'])
    candidate = sim.load_isaac_grasp_yaml(old['candidate_yaml'])[source['rank']]
    candidate = sim.candidate_for_object_origin(candidate, 'ur-base', center)
    result, samples = sim.evaluate_candidate(model, sim.initial_qpos(model, q), candidate,
                                             source['rank'], args, center)
    if not result.passed:
        (out / 'cb3_rejection.json').write_text(json.dumps(asdict(result), indent=2) + '\n')
        raise ValueError(f'Selected candidate fails nominal CB3 preflight: {result.reason}')
    close = [s for s in samples if s.phase == 'close']
    if not close:
        raise ValueError('Missing GGX closure samples')
    alpha = close[-1].closure_alpha
    # Do not reproduce the old preview's jump to full closure on lift.
    for sample in samples:
        if sample.phase == 'lift':
            sim.set_hand_qpos(model, sample.qpos, sim._lerp_ctrl(alpha))
            sample.closure_alpha = alpha
    minimum, _, desk_phase = sim.check_trajectory(model, samples, required_clearance_m=args.desk_clearance_mm / 1000)
    if desk_phase:
        raise ValueError(f'Fixed-closure lift hits desk: {desk_phase}')
    result.min_desk_clearance_mm = minimum * 1000
    addresses = [sim._joint_qpos_address(model, name) for name in sim.ARM_JOINTS]
    data = mujoco.MjData(model)
    rows = []
    for index, sample in enumerate(samples):
        data.qpos[:] = sample.qpos
        mujoco.mj_forward(model, data)
        hand = model.body('gripper_attachment').id
        T = np.eye(4)
        T[:3, 3] = data.xpos[hand]
        T[:3, :3] = data.xmat[hand].reshape(3, 3)
        tcp = matrix_pose(UR5Bridge.hand_T_to_tcp_T(T))
        rows.append(dict(index=index, phase=sample.phase, q_nominal=sample.qpos[addresses].tolist(),
                         tcp=tcp, closure_alpha=sample.closure_alpha,
                         hand_raw=raw_hand_command(model, sample.closure_alpha)))
    report = dict(schema='rh56_ggx_cb3_prepared/v1', executable=False, rank=source['rank'],
                  source=source, snapshot=snapshot, rows=rows, closure_alpha=alpha,
                  hand_open_raw=raw_hand_command(model, 0), hand_close_raw=raw_hand_command(model, alpha),
                  nominal_start_tcp=fk_tcp(model, q), minimum_desk_clearance_mm=minimum * 1000,
                  assumptions=['Nominal repository CB3 URDF; controller calibration must be checked before motion.',
                               'Nominal existing flange-to-hand mapping; no cube offsets or local fixed-shape correction.',
                               'New CB3 path requires review; no UR5e joint replay.',
                               'Write-only timed GGX closure to simulated stopping shape; hold same shape on lift.',
                               'No dynamic pickup/closure quality/self-collision certification.'])
    if video:
        sim.write_video(out / 'cb3_review.mp4', model, samples, result, args, center)
    (out / 'prepared.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report, model


def prepare_official_candidate(source, snapshot, out, model, *, video=False):
    """Deferred local assessment; never upgrade or mutate the official selection.

    This deliberately reuses the grasp-geometry path and its exact-reference
    policy. An official endpoint filter result alone supplies no local samples,
    robot path, or evidence for the physical hand's shape.
    """
    from tools import ggx_planning as planning
    from tools.ggx_template import sim_args, write_json
    from tools.run_ggx_real import load_config

    old = source['preflight']
    args = sim_args(source)
    args.desk_clearance_mm = 2.  # Unchanged geometry/continuous-execution policy.
    config = load_config(Path(source['checker_config']))
    center = np.asarray(old['object_center_m'])
    candidate = sim.load_isaac_grasp_yaml(old['candidate_yaml'])[source['rank']]
    candidate = sim.candidate_for_object_origin(candidate, 'ur-base', center)
    assessment = dict(schema='rh56_ggx_deferred_local_assessment/v1',
        status='running', selection_scope=OFFICIAL_SELECTION_SCOPE, rank=source['rank'],
        official_scene_clear=True, official_selection_unchanged=True,
        physical_hand_verified=False, executable=False, hardware_contacted=False,
        source_sha256=source['pins'], policy=planning.policy(),
        local_execution_checked=False, actual_travel_checked=False,
        hand_contract=dict(status='physical_mapping_unverified',
            official_open_descriptor=old.get('official_hand'),
            local_open_controls=sim._lerp_ctrl(0.),
            local_open_raw_command=raw_hand_command(model, 0.),
            raw_command_order=list(HAND_ORDER),
            semantics='Predicted commands/model controls, NOT measured joint angles or hand feedback.'),
        assumptions=['Official selection is retained even if local preparation is unavailable.',
            'Unchanged local model, proxy, thresholds, hand mapping and approach policy.',
            'Model path checks do not validate physical hand shape or authorize motion.'])
    assessment_path = out / 'local_assessment.json'
    write_json(assessment_path, assessment)
    detail, samples = planning.select_geometry(model, candidate, source['rank'], args, center, config)
    ready = (detail.get('state') == 'geometry_eligible'
             and detail.get('local_execution_check', {}).get('passed') is True)
    assessment.update(status='local_geometry_ready' if ready else 'local_geometry_unavailable',
        local_execution_checked=True, geometry=detail)
    write_json(assessment_path, assessment)
    if not ready:
        raise planning.PlanningUnavailable('selected_official_candidate_local_geometry_unavailable',
                                          [detail])

    rows = planning.exact_reference_rows(model, samples, candidate, center, args, detail['standoff_mm'])
    close = [row for row in rows if row['phase'] == 'close'][-1]
    reference = dict(schema='rh56_ggx_cb3_prepared/v1', executable=False,
        geometry_reference_only=True, rank=source['rank'], source=source,
        rows=rows, closure_alpha=close['closure_alpha'],
        hand_open_raw=raw_hand_command(model, 0), hand_close_raw=close['hand_raw'],
        nominal_start_tcp=rows[0]['tcp'], snapshot={'q': rows[0]['q_nominal']},
        minimum_desk_clearance_mm=min(s.min_desk_clearance_m for s in samples)*1000,
        local_assessment=str(assessment_path.resolve()), physical_hand_verified=False)
    reference_dir = out / 'geometry_reference'
    reference_dir.mkdir()
    write_json(reference_dir / 'prepared.json', reference)
    pre = [row for row in rows if row['phase'] == 'start_to_pregrasp'][-1]
    local_rows = [row for row in rows if row['phase'] not in ('start', 'start_to_pregrasp')]
    try:
        prepared, model = planning.execution(model, source, reference, [pre, *local_rows],
            snapshot, config, out, video=video)
    except planning.PlanningUnavailable as exc:
        assessment.update(actual_travel_checked=True, actual_travel_status='unavailable',
                          actual_travel_reason=exc.reason, actual_travel_attempts=exc.attempts)
        write_json(assessment_path, assessment)
        raise
    assessment.update(actual_travel_checked=True, actual_travel_status='nominal_path_ready')
    write_json(assessment_path, assessment)
    return prepared, model
