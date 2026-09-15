"""Continuous GGX phase primitives and their sampled checks. No device creation.

Samples certify/monitor a primitive's geometry; they are NOT stop waypoints.
Uses the repository's zero-TCP mapping, model, and Slerp/IK sampling convention.
"""
from __future__ import annotations

import math
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from tools import ggx_real_support as support
from tools import preflight_ggx_ur5_table as sim

REVISION = 'ggx_continuous_phases/v1'
KINDS = {'start_to_pregrasp': 'moveJ', 'approach': 'moveL', 'lift': 'moveL'}
TRIAL_KINDS = {'position_to_trial_start': 'moveJ'}
OBJECT_PHASES = sim.OBJECT_GATE_PHASES | frozenset(TRIAL_KINDS)
RESET_KINDS = {'reset_retreat': 'moveL', 'reset_return': 'moveJ'}
PREFLIGHT_MODES = frozenset({'dense', 'keypoints'})
REFLEX_CONTACT_POLICY = 'reflex_fingers_target/v1'
REFLEX_CONTACT_PHASES = frozenset({'reflex_approach', 'approach', 'remaining_fingers_close', 'lift'})
REFLEX_FINGER_BODIES = frozenset({
    'thumb_proximal_base', 'thumb_proximal', 'thumb_intermediate', 'thumb_distal',
    'index_proximal', 'index_intermediate', 'middle_proximal', 'middle_intermediate',
    'ring_proximal', 'ring_intermediate', 'pinky_proximal', 'pinky_intermediate'})


def reflex_desk_clearance_m(config):
    """Per-profile Reflex/reset clearance; existing profiles retain 2 mm.

    This is a nominal geometry margin, not a measured physical safety bound.
    Zero removes positive padding, but contact/penetration must still fail.
    """
    value = config.get('reflex_desk_clearance_mm', 2.)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError('reflex_desk_clearance_mm must be finite and nonnegative')
    return float(value) / 1000


def desk_clearance_blocked(clearance_m, required_m):
    """Zero nominal padding is not permission for contact or unknown distance."""
    return (not math.isfinite(clearance_m) or clearance_m <= 1e-9
            or clearance_m < required_m)


class PathCollision(RuntimeError):
    """Pre-motion geometry rejection, never a live controller/telemetry fault."""

    def __init__(self, message, *, phase, obstacle):
        super().__init__(message)
        self.phase, self.obstacle = phase, obstacle


def finite6(value):
    from tools.run_vision_pick_cycle import finite
    return finite(value, (6,))


def pose_at(start, end, t):
    a, b = finite6(start), finite6(end)
    orientation = Slerp([0, 1], Rotation.from_rotvec([a[3:], b[3:]]))(t).as_rotvec()
    return np.r_[a[:3] + t * (b[:3] - a[:3]), orientation].tolist()


def phase_goals(prepared, mode='grasp-lift'):
    if prepared.get('geometry_reference_only'):
        raise ValueError('Geometry reference has no checked actual travel path')
    if mode not in ('approach-only', 'grasp-lift'):
        raise ValueError('Unknown execution phase selection')
    rows = prepared['rows']
    phases = list(dict.fromkeys(r['phase'] for r in rows))
    trial = prepared.get('trial_start') is not None
    expected = ['start'] + (list(TRIAL_KINDS) if trial else []) + list(KINDS)[:2] + ['close', 'lift']
    if phases != expected:
        raise ValueError('Expected complete ordered GGX phases')
    order = [phases.index(r['phase']) for r in rows]
    if order != sorted(order) or [r['index'] for r in rows] != list(range(len(rows))):
        raise ValueError('GGX phases/indices must be contiguous')
    groups = {phase: [r for r in rows if r['phase'] == phase] for phase in phases}
    approach = groups['approach'][-1]
    for row in groups['close']:
        if (np.max(np.abs(finite6(row['q_nominal']) - finite6(approach['q_nominal']))) > 1e-8
                or max(support.pose_error(row['tcp'], approach['tcp'])) > 1e-6):
            raise ValueError('Cannot replace a wrist-moving closure with stationary closure')
    result = []
    for phase, kind in ((TRIAL_KINDS | KINDS) if trial else KINDS).items():
        if phase == 'lift' and mode == 'approach-only':
            continue
        group, end = groups[phase], groups[phase][-1]
        alpha = 0. if phase != 'lift' else groups['close'][-1]['closure_alpha']
        if any(abs(r['closure_alpha'] - alpha) > 1e-12 for r in group):
            raise ValueError('Hand shape changes inside an arm-only phase')
        result.append(dict(phase=phase, kind=kind, goal_tcp=finite6(end['tcp']).tolist(),
                           q_nominal=finite6(end['q_nominal']).tolist(), closure_alpha=alpha,
                           replaced_sample_count=len(group)))
    return result


def assert_inside(tcp, config):
    p = finite6(tcp)
    bounds = np.asarray(config['tcp_bounds_mm']) / 1000
    if np.any(p[:3] < bounds[0]) or np.any(p[:3] > bounds[1]):
        raise RuntimeError('Continuous path TCP outside configured bounds')


def _box_edge_separation(model, data, first, second):
    """Additional separating axes for two boxes; positive is a gap lower bound.

    The original certificate tests face/frame axes. Skew boxes can instead be
    separated only along an edge cross-product. No clearance allowance is added.
    """
    if any(model.geom_type[g] != sim.mujoco.mjtGeom.mjGEOM_BOX for g in (first, second)):
        return 0.
    a, b = (data.geom_xmat[g].reshape(3, 3).T for g in (first, second))
    axes = np.cross(a[:, None, :], b[None, :, :]).reshape(-1, 3)
    norms = np.linalg.norm(axes, axis=1)
    axes = axes[norms > 1e-12] / norms[norms > 1e-12, None]
    if not len(axes):
        return 0.
    va, vb = (sim._world_convex_vertices(model, data, g) @ axes.T for g in (first, second))
    gap = float(np.max(np.maximum(va.min(0) - vb.max(0), vb.min(0) - va.max(0))))
    return max(0., gap - sim.SEPARATION_ROUNDOFF_M) if math.isfinite(gap) else 0.


def object_clearance(model, data, *, ignored_bodies=frozenset()):
    """Preserve original signed/contact checks; resolve only certified false zeros.

    Kept in the execution layer so the saved, source-pinned GGX simulation and
    its original approval are not rewritten. Every robot pair still contributes.
    """
    original = sim.object_clearance(model, data)
    if original[0] != 0. and not ignored_bodies:
        return original
    obj = model.geom('preflight_object_visual').id
    minimum, nearest = math.inf, ''
    for geom in range(model.ngeom):
        if (int(model.geom_bodyid[geom]) in (0, int(model.geom_bodyid[obj]))
                or not (model.geom_contype[geom] or model.geom_conaffinity[geom])):
            continue
        if model.body(int(model.geom_bodyid[geom])).name in ignored_bodies:
            continue
        distance = float(sim.mujoco.mj_geomDistance(model, data, obj, geom, .5, np.empty(6)))
        if not math.isfinite(distance):
            raise ValueError('Non-finite robot-object distance')
        if distance == 0. and not any(c.dist <= 1e-9 and
                {int(c.geom1), int(c.geom2)} == {obj, geom} for c in data.contact):
            distance = max(sim.certified_object_separation(model, data, obj, geom),
                           _box_edge_separation(model, data, obj, geom))
        if distance < minimum:
            minimum, nearest = distance, sim._geom_label(model, geom)
    if not math.isfinite(minimum):
        raise ValueError('No active robot collision geometry for object check')
    return minimum, nearest


def reflex_target_check(model, data, phase):
    """Record all target contact; allow only finger contact in Reflex task phases.

    Recompute all nonfinger pairs: a deeper thumb overlap cannot hide a palm or
    arm collision. No mesh, padding, signed-distance threshold or desk change.
    Preparation (including thumb positioning at Pi) remains strict.
    """
    gap, nearest = object_clearance(model, data)
    allowed = phase in REFLEX_CONTACT_PHASES
    blocked_gap, blocked_nearest = (object_clearance(model, data, ignored_bodies=REFLEX_FINGER_BODIES)
                                   if allowed else (gap, nearest))
    return dict(policy=REFLEX_CONTACT_POLICY, finger_contact_allowed=allowed,
                minimum_target_gap_mm=gap*1000, nearest_target_geom=nearest,
                target_overlap=gap <= 1e-9, forbidden_target_overlap=blocked_gap <= 1e-9,
                minimum_forbidden_target_gap_mm=blocked_gap*1000,
                nearest_forbidden_target_geom=blocked_nearest)


def preflight(backend, prepared, model, config, report, mode='grasp-lift'):
    return preflight_goals(backend, prepared['snapshot'], phase_goals(prepared, mode),
                           model, config, report)


def preflight_goals(backend, snapshot, goals, model, config, report):
    """Check the WHOLE intended moveJ/moveL primitive, not old waypoint chords.

    MoveJ samples its single synchronized joint segment at <=0.01 rad.
    MoveL uses <=2 mm / 1 degree Cartesian interpolation and adaptive IK
    subdivision to <=0.01 rad; no blending or hidden straight-line shortcuts.
    Keypoints mode reduces remote safety/FK checks, not IK or geometry samples.
    """
    required_clearance = reflex_desk_clearance_m(config)
    report['required_desk_clearance_mm'] = required_clearance * 1000
    mode = config.get('controller_preflight_mode', 'dense')
    if not isinstance(mode, str) or mode not in PREFLIGHT_MODES:
        raise ValueError('Unknown controller_preflight_mode')
    timing = dict(schema='rh56_ggx_controller_preflight_timing/v1', status='running',
                  mode=mode, queries={}, fk_cache_hits=0, validated_samples=0,
                  controller_checked_samples=0, nominal_fk_samples=0)
    report['controller_preflight_timing'] = timing
    report['controller_check_scope'] = dict(mode=mode,
        object_zero_fallback='mesh_face_and_box_edge_certificate/v2; no contact, no negative-distance override',
        remote_safety_fk='every sample' if mode == 'dense' else 'first/middle/last scheduled sample per phase',
        cartesian_ik='controller IK at every Cartesian sample, including adaptive subdivisions',
        local_geometry='every sample; closure also rechecked before execution',
        limitations=['Keypoints mode does not pre-query controller safety/FK at every interior sample.',
                     'Intermediate model/target IK residual uses the sum of the existing model and IK tolerances.',
                     'Nominal interior TCP is an estimate, not a measured/calibrated FK result.',
                     'UR protective stops do not certify clearance from the desk, object or hand.'])
    started = time.monotonic()
    print(f'GGX preflight: {mode}; controller IK + safety/FK + local geometry…', flush=True)
    try:
        result = _preflight_goals(backend, snapshot, goals, model, config, report, timing)
        timing['status'] = 'completed'
        return result
    except BaseException:
        timing['status'] = 'failed'
        raise
    finally:
        timing['elapsed_s'] = time.monotonic() - started
        timing['controller_query_s'] = sum(row['elapsed_s'] for row in timing['queries'].values())
        timing['controller_query_count'] = sum(row['count'] for row in timing['queries'].values())
        # Includes receiver snapshots and Python overhead, not just geometry CPU time.
        timing['other_s'] = max(0., timing['elapsed_s'] - timing['controller_query_s'])
        print(f"GGX preflight: {timing['status']} {timing['elapsed_s']:.2f}s; "
              f"controller queries {timing['controller_query_count']} / {timing['controller_query_s']:.2f}s; "
              f"other {timing['other_s']:.2f}s; reused FK {timing['fk_cache_hits']}", flush=True)


def _preflight_goals(backend, snapshot, goals, model, config, report, timing):
    from tools.run_vision_pick_cycle import check_stationary_origin
    ctrl = backend.control
    dense = timing['mode'] == 'dense'

    def query(method, *args):
        row = timing['queries'].setdefault(method, dict(count=0, errors=0, elapsed_s=0.))
        row['count'] += 1
        started = time.monotonic()
        try:
            return getattr(ctrl, method)(*args)
        except BaseException:
            row['errors'] += 1
            raise
        finally:
            row['elapsed_s'] += time.monotonic() - started

    # Explicit q + zero TCP defines the same calibrated FK within this single
    # stationary preflight. Reuse exact inputs only (no rounding), never IK or
    # safety decisions. Discard on return/failure; no cache crosses a motion/run.
    fk_cache = {}

    def fk(q):
        key = tuple(finite6(q))
        if key in fk_cache:
            timing['fk_cache_hits'] += 1
        else:
            fk_cache[key] = tuple(finite6(query('getForwardKinematics', list(key), [0.] * 6)))
        return list(fk_cache[key])

    initial = backend.snapshot()
    check_stationary_origin(snapshot, initial)
    q = finite6(initial['q'])
    tcp = finite6(initial['tcp']).tolist()
    data = sim.mujoco.MjData(model)
    commands = []
    report['controller_preflight'] = commands
    report['motion_revision'] = REVISION

    def validate(q, phase, alpha, *, remote=True):
        report['preflight_current_sample'] = dict(phase=phase, q=q.tolist(), closure_alpha=alpha)
        nominal = support.fk_tcp(model, q)
        if remote:
            if not query('isJointsWithinSafetyLimits', q.tolist()):
                raise RuntimeError('Controller joint safety limits failed')
            actual = fk(q)
            if not query('isPoseWithinSafetyLimits', actual):
                raise RuntimeError('Controller sampled pose safety failed')
            pe, re = support.pose_error(nominal, actual)
            if pe > config['model_position_tolerance_mm'] or re > config['model_rotation_tolerance_deg']:
                raise RuntimeError('Nominal CB3 model disagrees with calibrated controller')
            timing['controller_checked_samples'] += 1
        else:
            actual = finite6(nominal).tolist()
            timing['nominal_fk_samples'] += 1
        assert_inside(actual, config)
        data.qpos[:] = sim.initial_qpos(model, q)
        # Non-GGX callers may freeze an explicit hand shape per arm phase.
        # Absent that field, preserve the original GGX interpolation unchanged.
        sim.set_hand_qpos(model, data.qpos, active_hand_ctrl if active_hand_ctrl is not None else sim._lerp_ctrl(alpha))
        sim.mujoco.mj_forward(model, data)
        clearance, _ = sim.desk_clearance(model, data)
        if desk_clearance_blocked(clearance, reflex_desk_clearance_m(config)):
            raise PathCollision('Continuous controller path hits desk', phase=phase, obstacle='desk')
        object_gap = None
        contact = None
        if phase in OBJECT_PHASES or active_contact_policy is not None:
            if active_contact_policy is not None:
                contact = reflex_target_check(model, data, phase)
                object_gap = contact['minimum_forbidden_target_gap_mm']/1000
                nearest_object = contact['nearest_forbidden_target_geom']
            else:
                object_gap, nearest_object = object_clearance(model, data)
            if object_gap <= 1e-9:
                report['preflight_current_sample'].update(object_clearance_mm=object_gap * 1000,
                                                          nearest_object_geom=nearest_object)
                raise PathCollision('Continuous controller approach hits object', phase=phase, obstacle='object')
        timing['validated_samples'] += 1
        return dict(q=q.tolist(), tcp=actual, controller_checked=remote,
                    tcp_source='controller_fk' if remote else 'nominal_fk_estimate',
                    desk_clearance_mm=clearance * 1000,
                    target_contact=contact,
                    object_clearance_mm=None if object_gap is None else object_gap * 1000)

    def ik(target, seed, *, remote=True):
        assert_inside(target, config)
        if remote and not query('isPoseWithinSafetyLimits', target):
            raise RuntimeError('Controller rejected TCP safety limits')
        try:
            solution = finite6(query('getInverseKinematics', target, seed.tolist(), 1e-5, 1e-5))
        except (ValueError, TypeError) as exc:
            raise RuntimeError('Controller IK missing/invalid') from exc
        if remote:
            pe, re = support.pose_error(fk(solution), target)
            if pe > 1 or re > .5:
                raise RuntimeError('Controller IK/FK failed to reproduce target')
        else:
            # Do not invent a new local IK or change the controller's IK seed.
            # This is a nominal residual check, NOT a calibrated FK certificate.
            pe, re = support.pose_error(support.fk_tcp(model, solution), target)
            if (pe > config['model_position_tolerance_mm'] + 1.
                    or re > config['model_rotation_tolerance_deg'] + .5):
                raise RuntimeError('Controller IK/nominal FK residual exceeds combined tolerances')
        return solution

    for goal in goals:
        check_stationary_origin(initial, backend.snapshot())
        phase, kind, alpha = goal['phase'], goal['kind'], goal['closure_alpha']
        active_hand_ctrl = goal.get('hand_ctrl')
        active_contact_policy = goal.get('target_contact_policy')
        if active_contact_policy is not None and (active_contact_policy != REFLEX_CONTACT_POLICY
                or phase not in ('approach', 'lift') or active_hand_ctrl is None):
            raise ValueError('Reflex contact policy requires an explicit approach/lift hand shape')
        if active_hand_ctrl is not None:
            if (set(active_hand_ctrl) != set(sim.GGX_OPEN_CTRL)
                    or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
                               for v in active_hand_ctrl.values())):
                raise ValueError('Explicit phase hand shape must contain all finite actuator controls')
        if kind != (KINDS | RESET_KINDS | TRIAL_KINDS).get(phase):
            raise ValueError('Unknown continuous phase/kind')
        print(f'GGX preflight: {phase} ({kind})', flush=True)
        start_q, start_tcp = q.copy(), list(tcp)
        target = goal['goal_tcp']
        samples = []
        if kind == 'moveJ':
            nominal_goal = finite6(goal['q_nominal'])
            end_q = ik(target, nominal_goal)
            if np.max(np.abs(end_q - nominal_goal)) > .15:
                raise RuntimeError('Controller endpoint IK changed nominal branch')
            count = max(2, math.ceil(np.max(np.abs(end_q - q)) / .01) + 1)
            for index, t in enumerate(np.linspace(0, 1, count)):
                row = validate(start_q + t * (end_q - start_q), phase, alpha,
                               remote=dense or index in {0, count // 2, count - 1})
                samples.append(dict(fraction=float(t), **row))
        else:
            distance, angle = support.pose_error(start_tcp, target)
            count = max(2, math.ceil(distance / 2) + 1, math.ceil(angle) + 1)
            samples.append(dict(fraction=0., **validate(q, phase, alpha)))

            def advance(t0, qa, t1, depth=0, *, remote=False):
                qb = ik(pose_at(start_tcp, target, t1), qa, remote=dense or remote)
                change = np.max(np.abs(qb - qa))
                if change > .15:
                    raise RuntimeError('Controller IK missing/discontinuous')
                if change > .01 + 1e-10:
                    if depth >= 12 or len(samples) > 4096:
                        raise RuntimeError('Continuous moveL IK subdivision exhausted')
                    tm = (t0 + t1) / 2
                    qm = advance(t0, qa, tm, depth + 1)
                    return advance(tm, qm, t1, depth + 1, remote=remote)
                row = validate(qb, phase, alpha, remote=dense or remote)
                samples.append(dict(fraction=float(t1), **row))
                return qb

            last_t = 0.
            for index, t in enumerate(np.linspace(0, 1, count)[1:], start=1):
                q = advance(last_t, q, float(t), remote=index in {count // 2, count - 1})
                last_t = float(t)
        q, tcp = finite6(samples[-1]['q']), samples[-1]['tcp']
        command = dict(revision=REVISION, controller_preflight_mode=timing['mode'],
                       **goal, start_q=start_q.tolist(), start_tcp=start_tcp,
                       q=q.tolist(), tcp=list(tcp), samples=samples,
                       speed=config['joint_speed_rad_s' if kind == 'moveJ' else 'cartesian_speed_m_s'],
                       accel=config['joint_accel_rad_s2' if kind == 'moveJ' else 'cartesian_accel_m_s2'])
        commands.append(command)
    check_stationary_origin(initial, backend.snapshot())
    report.pop('preflight_current_sample', None)
    return commands


def joint_path_error(q, samples):
    """Distance to the checked joint polyline (rad), for live branch/path monitoring."""
    path = np.asarray([r['q'] for r in samples], dtype=float)
    a, delta = path[:-1], np.diff(path, axis=0)
    t = np.clip(np.sum((q-a)*delta, axis=1) / np.maximum(np.sum(delta*delta, axis=1), 1e-16), 0, 1)
    return float(np.min(np.max(np.abs(q - (a + t[:,None]*delta)), axis=1)))


def cartesian_path_error(tcp, command):
    a, b, p = finite6(command['start_tcp']), finite6(command['goal_tcp']), finite6(tcp)
    delta = b[:3] - a[:3]
    if np.dot(delta, delta) > 1e-16:
        t = float(np.clip(np.dot(p[:3]-a[:3], delta) / np.dot(delta, delta), 0, 1))
    else:
        _, total = support.pose_error(a, b)
        _, angle = support.pose_error(a, p)
        t = float(np.clip(angle / max(total, 1e-12), 0, 1))
    return support.pose_error(p, pose_at(a, b, t))


def write_review(out, prepared, model, commands, *, video=True):
    """Recheck the actual primitives including closure; optionally visualize.

    Disabling rendering must never skip this geometry gate, particularly the
    closure samples at the controller-resolved wrist pose.
    """
    samples = []
    for command in commands:
        if command['phase'] == 'lift':
            for row in prepared['rows']:
                if row['phase'] == 'close':
                    qpos = sim.initial_qpos(model, np.asarray(commands[-2]['q']))
                    sim.set_hand_qpos(model, qpos, sim._lerp_ctrl(row['closure_alpha']))
                    samples.append(sim.TrajectorySample('close', row['closure_alpha'], qpos, row['closure_alpha']))
        for row in command['samples']:
            qpos = sim.initial_qpos(model, np.asarray(row['q']))
            sim.set_hand_qpos(model, qpos, sim._lerp_ctrl(command['closure_alpha']))
            samples.append(sim.TrajectorySample(command['phase'], row['fraction'], qpos, command['closure_alpha']))
    minimum, nearest, collision = sim.check_trajectory(model, samples, required_clearance_m=.002)
    data = sim.mujoco.MjData(model)
    for sample in samples:
        if sample.phase in OBJECT_PHASES and (sample.object_clearance_m == 0.
                                             or sample.phase in TRIAL_KINDS):
            data.qpos[:] = sample.qpos
            sim.mujoco.mj_forward(model, data)
            sample.object_clearance_m, sample.nearest_object_geom = object_clearance(model, data)
            sample.object_collision = sample.object_clearance_m <= 1e-9
    object_collision = any(s.object_collision for s in samples)
    report = dict(schema='rh56_ggx_execution_precheck/v1', motion_revision=REVISION,
                  object_zero_fallback='box_edge_axis_certificate/v1',
                  controller_preflight_modes=sorted({c.get('controller_preflight_mode', 'dense') for c in commands}),
                  passed=not bool(collision or object_collision), sample_count=len(samples),
                  phase_samples={phase: sum(s.phase == phase for s in samples)
                                 for phase in dict.fromkeys(s.phase for s in samples)},
                  minimum_desk_clearance_mm=minimum * 1000, nearest_desk_geom=nearest,
                  first_desk_collision_phase=collision, object_collision=object_collision,
                  video_requested=video,
                  assumptions=['Same sampled geometry gate with or without video.',
                               'Closure desk check uses controller-resolved wrist joints.',
                               'No dynamic pickup, closure quality or self-collision certification.'])
    (out / 'execution_precheck.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if not report['passed']:
        phase = collision or next(s.phase for s in samples if s.object_collision)
        raise PathCollision('Controller execution preview failed collision recheck',
                            phase=phase, obstacle='desk' if collision else 'object')
    if not video:
        return None
    candidate = sim.load_isaac_grasp_yaml(prepared['source']['preflight']['candidate_yaml'])[prepared['rank']]
    result = sim.CandidateResult(prepared['rank'], candidate.name, candidate.confidence, True,
        'continuous_phase_precheck', None, None, minimum*1000, nearest, '', len(samples))
    args = sim.parse_args([])
    for key, value in prepared['source']['preflight']['arguments'].items():
        if hasattr(args, key): setattr(args, key, value)
    path = out / 'execution_review.mp4'
    sim.write_video(path, model, samples, result, args,
                    np.asarray(prepared['source']['preflight']['object_center_m']))
    return path
