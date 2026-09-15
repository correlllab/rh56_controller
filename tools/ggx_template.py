"""Object-relative GGX template and translation-only historical paper starts.

No hardware construction. Reuse the original CB3 model, TCP mapping, hand
closure and IK; a template is geometry/data, NEVER a saved motion approval.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from tools import ggx_real_support as support, ggx_motion as motion
from tools import preflight_ggx_ur5_table as sim
from tools.run_ggx_review import pin, verify, read_json

SCHEMA = 'rh56_ggx_grasp_template/v1'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def vector(value, size):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f'Expected finite {size}-vector')
    return result


def object_matrix(center, yaw):
    if not np.isfinite(yaw):
        raise ValueError('Object yaw must be finite')
    T = np.eye(4)
    T[:3, 3] = vector(center, 3)
    T[:3, :3] = Rotation.from_euler('z', yaw, degrees=True).as_matrix()
    return T


def paper_points():
    # Reuse the historical sampler, including its original labels/order.
    from tools.run_strategy_pregrasp_rate import paper_approach_points_mm, approach_basis
    points = paper_approach_points_mm(SimpleNamespace(
        paper_p1_distance_mm=250., paper_p2_height_mm=250.,
        paper_level1_height_mm=100., paper_level2_height_mm=250.,
        paper_d_list_mm=[-150., -50., 50., 150.]))
    approach, lateral = approach_basis('y-')
    return [dict(p, id=f'P{i+1}', offset_mm=(approach*p['approach_distance_mm']
                  + lateral*p['d_mm'] + [0, 0, p['h_mm']]).tolist())
            for i, p in enumerate(points)]


def open_center_tcp(model, row):
    """Existing antipodal center policy, evaluated on this mounted hand model."""
    from rh56_controller.grasp_geometry import TIP_SITES
    from tools.demo_strategy_pregrasp_collision import grasp_center_base
    data = mujoco.MjData(model)
    data.qpos[:] = sim.initial_qpos(model, np.asarray(row['q_nominal']))
    mujoco.mj_forward(model, data)
    tips = {name: data.site_xpos[model.site(site).id].copy() for name, site in TIP_SITES.items()}
    center = grasp_center_base(SimpleNamespace(tip_positions=tips), 'antipodal')
    actual_tcp=support.fk_tcp(model,row['q_nominal'])
    return (np.linalg.inv(support.pose_matrix(actual_tcp)) @ np.r_[center, 1])[:3]


def create(approval, snapshot_path, config, out):
    automatic = read_json(approval).get('schema') == 'rh56_ggx_execution_candidate/v1'
    source = support.read_candidate(approval) if automatic else support.read_approval(approval)
    if automatic and source.get('selection_scope') != 'grasp_geometry':
        raise ValueError('Automatic templates require a strict local geometry selection')
    if source.get('selection_scope') == 'official_ggx_candidate':
        raise ValueError('Official-only selection cannot become a trial template before physical OPEN mapping '
                         'is verified; use run_ggx_real.py without --execute for offline local assessment.')
    source = dict(source, desk_edge_y_m=config['desk_edge_y_m'], desk_envelope=config.get('desk_envelope'))
    reference_dir = out / 'reference'
    reference_dir.mkdir()
    if source.get('selection_scope') == 'grasp_geometry':
        from tools import ggx_planning as planning
        if config['desk_envelope'] != source['preflight']['desk_envelope']:
            raise ValueError('Geometry selection and trial fixed desk differ')
        prepared, model = planning.reference(source, reference_dir)
    else:
        prepared, model = support.prepare(source, read_json(snapshot_path), reference_dir)
    old = source['preflight']
    object_T = object_matrix(old['object_center_m'], old['object_yaw_deg'])
    end = [r for r in prepared['rows'] if r['phase'] == 'approach'][-1]
    payload = dict(schema=SCHEMA, executable=False, motion_authorized=False,
        approval=str(Path(approval).resolve()), rank=source['rank'],
        selection_input_kind='unreviewed_candidate' if automatic else 'visual_approval',
        human_approved=not automatic,
        reference_prepared=str((reference_dir / 'prepared.json').resolve()),
        source_sha256={**source['pins'], **pin([snapshot_path, reference_dir / 'prepared.json',
            support.URDF, support.ROOT / 'rh56_controller/ur5_bridge.py',
            support.ROOT / 'rh56_controller/grasp_executor.py',
            *[p for p in (support.ROOT / 'h1_mujoco/inspire').rglob('*')
              if p.is_file() and p.suffix.lower() in ('.xml', '.stl', '.obj')],
            *list((support.URDF.parent / 'meshes/ur5/collision').glob('*.stl'))])},
        object_frame='UR-base; geometric proxy center; X=long axis; Z=desk-up; metres',
        reference_object_T=object_T.tolist(), object_size_mm=old['object_size_mm'],
        grasp_tcp_in_object=(np.linalg.inv(object_T) @ support.pose_matrix(end['tcp'])).tolist(),
        open_grasp_center_in_tcp_m=open_center_tcp(model, end).tolist(),
        closure_alphas=[r['closure_alpha'] for r in prepared['rows'] if r['phase'] == 'close'],
        assumptions=['Automatic geometry selection or original visual approval is not proof of physical pickup.',
            'Same rigid object, same upright resting orientation; yaw/XY relocation only.',
            'Freeze proxy dimensions, grasp orientation relative to object and all hand commands.',
            'No GGX re-inference or re-ranking. New controller path still needs this-trial Enter.',
            'Open-loop RH56 closure; no measured contact or pickup verification.'])
    write_json(out / 'template.json', payload)
    return payload


def load(path):
    template = read_json(path)
    if (template.get('schema') != SCHEMA or template.get('executable') is not False
            or template.get('motion_authorized') is not False):
        raise ValueError('Expected non-executable grasp template v1')
    verify(template['source_sha256'])
    for key in ('approval', 'reference_prepared'):
        if str(Path(template[key]).resolve()) not in template['source_sha256']:
            raise ValueError(f'Template does not bind {key}')
    kind = template.get('selection_input_kind', 'visual_approval')
    if kind not in ('visual_approval', 'unreviewed_candidate'):
        raise ValueError('Unknown template selection input kind')
    source = (support.read_candidate(template['approval']) if kind == 'unreviewed_candidate'
              else support.read_approval(template['approval']))
    if kind == 'unreviewed_candidate' and (template.get('human_approved') is not False
                                         or source.get('selection_scope') != 'grasp_geometry'):
        raise ValueError('Automatic template must retain strict geometry scope and no human approval')
    if source.get('selection_scope') == 'official_ggx_candidate':
        raise ValueError('Official-only selection cannot inherit a local template or motion approval')
    prepared = read_json(template['reference_prepared'])
    if template['rank'] != source['rank'] or prepared['rank'] != source['rank']:
        raise ValueError('Template/reference rank mismatch')
    old = source['preflight']
    expected = object_matrix(old['object_center_m'], old['object_yaw_deg'])
    if not np.allclose(template['reference_object_T'], expected, rtol=0, atol=1e-12):
        raise ValueError('Template object frame changed')
    if not np.array_equal(template['object_size_mm'], old['object_size_mm']):
        raise ValueError('Template dimensions changed')
    approach = [r for r in prepared['rows'] if r['phase'] == 'approach'][-1]
    relative = np.linalg.inv(expected) @ support.pose_matrix(approach['tcp'])
    if not np.allclose(template['grasp_tcp_in_object'], relative, rtol=0, atol=1e-12):
        raise ValueError('Template grasp changed')
    if template['closure_alphas'] != [r['closure_alpha'] for r in prepared['rows'] if r['phase'] == 'close']:
        raise ValueError('Template closure changed')
    expected_center = open_center_tcp(support.build_model(prepared['source']), approach)
    if not np.allclose(vector(template['open_grasp_center_in_tcp_m'], 3), expected_center, rtol=0, atol=1e-12):
        raise ValueError('Template open-hand grasp center changed')
    source['pins'].update(template['source_sha256'])
    source['pins'].update(pin([path]))
    return template, source, prepared


def relocate(template, source, reference, *, xy_mm=None, yaw_deg=None, lock_z=False):
    """Same object; lock Z for XY-only repeats, or retain legacy desk-plane Z."""
    from tools.analyze_ur5_camera_colored_target import plane_z_at_xy_mm
    import yaml
    source = copy.deepcopy(source)
    old = source['preflight']
    center = np.array(old['object_center_m'])
    if xy_mm is not None:
        xy = vector(xy_mm, 2)
        normal = yaml.safe_load(Path(source['perception']['desk_plane']).read_text())['plane']['normal_base']
        delta = 0. if lock_z else (plane_z_at_xy_mm(normal, 0., *xy)
                 - plane_z_at_xy_mm(normal, 0., *(center[:2]*1000))) / 1000
        center = np.r_[xy/1000, center[2] + delta]
    yaw = old['object_yaw_deg'] if yaw_deg is None else float(yaw_deg)
    T = object_matrix(center, yaw)
    delta_T = T @ np.linalg.inv(np.array(template['reference_object_T']))
    old.update(object_center_m=center.tolist(), object_yaw_deg=yaw,
               table_height_m=float(center[2] - old['object_size_mm'][2]/2000))
    old['arguments'].update(object_x_m=float(center[0]), object_y_m=float(center[1]),
        object_yaw_deg=yaw, table_height_m=old['table_height_m'])
    source.update(source_kind='grasp_template', template_relocation=dict(
        object_T=T.tolist(), delta_T=delta_T.tolist(),
        old_approval_is_motion_authorization=False, inherits_collision_pass=False,
        operator_must_confirm_same_object=True))
    # This is a derived scene description, not the old simulation certificate.
    old.update(schema='rh56_ggx_template_scene/v1', automatic_pass_ranks=[])
    return source, delta_T


def solve_tcp(model, seed, tcp, args):
    hand_T = support.UR5Bridge.tcp_T_to_hand_T(support.pose_matrix(tcp))
    target = sim.target_eeff_transform(hand_T[:3, 3], hand_T[:3, :3])
    qpos, pe, re, passed = sim.solve_pose_ik(model, seed, target,
        max_iters=args.ik_max_iters, dt=args.ik_dt, position_tol_m=min(args.ik_position_tol_mm/1000, .00005),
        orientation_tol_rad=np.radians(min(args.ik_orientation_tol_deg, .05)), solver=args.solver)
    if not passed:
        raise ValueError(f'Nominal IK failed: {pe*1000:.3f} mm, {np.degrees(re):.3f} deg')
    return qpos


def sim_args(source):
    args = sim.parse_args([])
    for key, value in source['preflight']['arguments'].items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def canonical(template, source, reference, delta_T):
    """Keep one common pregrasp/approach/close/lift for all ten starts."""
    model = support.build_model(source)
    args = sim_args(source)
    pre = [r for r in reference['rows'] if r['phase'] == 'start_to_pregrasp'][-1]
    rows = [pre] + [r for r in reference['rows'] if r['phase'] in ('approach', 'close', 'lift')]
    if source.get('selection_scope') == 'grasp_geometry':
        # TCP targets/closure stay fixed relative to the object. Solve branches
        # per actual P_i later, never reject all starts using the reference seed.
        return model, [dict(row,tcp=support.matrix_pose(delta_T @ support.pose_matrix(row['tcp']))) for row in rows]
    identity = np.allclose(delta_T, np.eye(4), rtol=0, atol=1e-12)
    addresses = [sim._joint_qpos_address(model, n) for n in sim.ARM_JOINTS]
    seed = sim.initial_qpos(model, np.array(pre['q_nominal']))
    result, cache = [], {}
    for row in rows:
        target = support.matrix_pose(delta_T @ support.pose_matrix(row['tcp']))
        key = tuple(row['tcp'])
        if identity:
            q = row['q_nominal']
        else:
            if key not in cache:
                seed = solve_tcp(model, seed, target, args)
                cache[key] = seed[addresses].tolist()
            q = cache[key]
        result.append(dict(row, tcp=support.fk_tcp(model, q), q_nominal=list(q)))
    return model, result


def layout(template, source, canonical_rows, origin_mm):
    # The anchor is a GRASP CENTER, not a wrist/TCP. Every point has the same
    # orientation and open hand, so its center-to-TCP vector is constant.
    approach = [r for r in canonical_rows if r['phase'] == 'approach'][-1]
    T = support.pose_matrix(approach['tcp'])
    offset = T[:3, :3] @ vector(template['open_grasp_center_in_tcp_m'], 3)
    origin = vector(origin_mm, 3)/1000
    points = []
    for point in paper_points():
        center = origin + np.array(point['offset_mm'])/1000
        points.append(dict(point, center_base_mm=(center*1000).tolist(),
                           tcp=[*(center-offset), *approach['tcp'][3:]]))
    return points


def default_origin(template, canonical_rows):
    # Legacy candidate-dependent placement, NOT used by the fixed experiment.
    # P1 TCP = (-140,-455,345) mm; P2 = (-140,-205,595) mm.
    # Translation-only offline placement audit: origin_search_02 (2026-09-11).
    end = [r for r in canonical_rows if r['phase'] == 'approach'][-1]
    offset = support.pose_matrix(end['tcp'])[:3, :3] @ vector(template['open_grasp_center_in_tcp_m'], 3)
    return (np.array([-.140, -.205, .345]) + offset)*1000


def prepare_trial(template, source, reference, model, canonical_rows, point, snapshot, out, *, video=False):
    verify(source['pins'])
    if source.get('selection_scope') == 'grasp_geometry':
        from tools import ggx_planning as planning
        return planning.execution(model,source,reference,canonical_rows,snapshot,
                                  source['trial_config'],out,point=point,video=video)
    args = sim_args(source)
    addresses = [sim._joint_qpos_address(model, n) for n in sim.ARM_JOINTS]
    initial = sim.initial_qpos(model, vector(snapshot['q'], 6))
    # Fixed-layout IK branch hint, or legacy common pregrasp for explicit layouts.
    seed = sim.initial_qpos(model, np.array(point.get('q_nominal', canonical_rows[0]['q_nominal'])))
    qpoint = solve_tcp(model, seed, point['tcp'], args)
    point_tcp = support.fk_tcp(model, qpoint[addresses])
    samples = [sim.TrajectorySample('start', 0., initial.copy(), 0.)]
    sim._append_joint_segment(model, samples, initial, qpoint, phase='position_to_trial_start',
                             max_step_rad=np.radians(args.path_step_deg), closure_alpha=0.)
    preq = sim.initial_qpos(model, np.array(canonical_rows[0]['q_nominal']))
    sim._append_joint_segment(model, samples, qpoint, preq, phase='start_to_pregrasp',
                             max_step_rad=np.radians(args.path_step_deg), closure_alpha=0.)
    for row in canonical_rows[1:]:
        qpos = sim.initial_qpos(model, np.array(row['q_nominal']))
        sim.set_hand_qpos(model, qpos, sim._lerp_ctrl(row['closure_alpha']))
        samples.append(sim.TrajectorySample(row['phase'], 0., qpos, row['closure_alpha']))
    minimum, nearest, collision = sim.check_trajectory(model, samples, required_clearance_m=.002)
    data = mujoco.MjData(model)
    object_min = float('inf')
    object_phase = bounds_phase = ''
    rows = []
    for index, sample in enumerate(samples):
        q = sample.qpos[addresses].tolist()
        tcp = support.fk_tcp(model, q)
        data.qpos[:] = sample.qpos
        mujoco.mj_forward(model, data)
        if sample.phase in motion.OBJECT_PHASES:
            gap, _ = motion.object_clearance(model, data)
            sample.object_clearance_m, sample.object_collision = gap, gap <= 1e-9
            object_min = min(object_min, gap)
            if sample.object_collision and not object_phase:
                object_phase = sample.phase
        try:
            motion.assert_inside(tcp, source['trial_config'])
        except RuntimeError:
            bounds_phase = bounds_phase or sample.phase
        rows.append(dict(index=index, phase=sample.phase, tcp=tcp, q_nominal=q,
                         closure_alpha=sample.closure_alpha, hand_raw=support.raw_hand_command(model, sample.closure_alpha)))
    diagnostics = dict(schema='rh56_ggx_ten_start_precheck/v1', point_id=point['id'],
        passed=not bool(collision or object_phase or bounds_phase),
        first_desk_collision_phase=collision, first_object_collision_phase=object_phase,
        first_bounds_failure_phase=bounds_phase, min_desk_clearance_mm=minimum*1000,
        min_object_clearance_mm=object_min*1000,
        commanded_start_tcp=point['tcp'], nominal_start_tcp=point_tcp,
        start_error_mm=support.pose_error(point_tcp, point['tcp'])[0],
        hardware_contacted=False)
    write_json(out / 'precheck.json', diagnostics)
    prepared = dict(reference, source=source, snapshot=snapshot, rows=rows,
        trial_start=dict(point, nominal_tcp=point_tcp), nominal_start_tcp=support.fk_tcp(model, snapshot['q']))
    write_json(out / 'prepared.json', prepared)
    if video:
        candidate = sim.load_isaac_grasp_yaml(source['preflight']['candidate_yaml'])[source['rank']]
        result = sim.CandidateResult(source['rank'], candidate.name, candidate.confidence, diagnostics['passed'],
            'template_trial', None, None, minimum*1000, nearest, collision, len(samples))
        sim.write_video(out / 'cb3_review.mp4', model, samples, result, args,
                        np.asarray(source['preflight']['object_center_m']))
    if bounds_phase:
        raise ValueError(f"{point['id']} outside workspace: {bounds_phase}")
    if collision or object_phase:
        raise motion.PathCollision(
            f"{point['id']} blocked: desk={collision or '-'}, object={object_phase or '-'}",
            phase=collision or object_phase, obstacle='desk' if collision else 'object')
    if not diagnostics['passed']:
        raise ValueError(f"{point['id']} blocked: desk={collision or '-'}, object={object_phase or '-'}, bounds={bounds_phase or '-'}")
    return prepared, model
