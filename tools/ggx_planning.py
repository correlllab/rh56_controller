"""Deterministic GGX geometry/route separation; no hardware access.

Reuses the existing Mink solver, hand/TCP transform, closure and geometry APIs.
Fixed seed and standoff policies are protocol choices, never tuned to a batch.
An eligible geometry is NOT a checked travel path or physical grasp success.
"""
from dataclasses import asdict
from pathlib import Path
from xml.etree import ElementTree
import math
import numpy as np
import mujoco

from tools import ggx_real_support as support, preflight_ggx_ur5_table as sim

SCHEMA = 'rh56_ggx_cb3_grasp_geometry/v1'
REVISION = 'geometry_then_actual_route/v2_shared_local_gate'
STANDOFFS_MM = (80., 60., 40., 20.)
# Independent of object, GGX confidence and scan position. These are numerical
# initializations, NEVER commanded poses. Hints precede this fixed sequence.
SEEDS_DEG = tuple((pan, -90., elbow, -90. if elbow > 0 else 90., wrist, 0.)
                 for pan in (90., 0., -90., 180.)
                 for elbow, wrist in ((90., -90.), (90., 90.), (-90., -90.), (-90., 90.)))


class PlanningUnavailable(RuntimeError):
    """Only offline, pre-motion planning exhaustion; never a controller fault."""
    def __init__(self, reason, attempts):
        super().__init__(reason)
        self.reason, self.attempts = reason, attempts


def policy():
    return dict(revision=REVISION, standoffs_mm=list(STANDOFFS_MM),
                seed_q_deg=[list(q) for q in SEEDS_DEG],
                seed_rule='hints then fixed seeds; equivalent angles nearest hint within joint limits',
                selection_scope='grasp/local approach only; scan travel is not checked',
                actual_route='current -> fixed trial start -> pregrasp -> approach -> close -> lift',
                actual_ik_tolerance_mm_deg=[.05,.05],
                local_gate='same execution geometry predicates; unresolved pre-closure target overlap is not review-eligible',
                minimum_standoff_assumption='20 mm is a protocol minimum, not measured collision clearance')


def tcp_bounds_from_urdf():
    """Loose finite TCP envelope from the URDF chain, NOT a reachability test.

    The z=0 commissioning floor remains; actual full-body/table checks and IK
    still apply. No candidate, object dimensions or observed pass rate enters.
    """
    tree = ElementTree.parse(support.URDF).getroot()
    edges = {j.find('child').attrib['link']: j for j in tree.findall('joint')}
    node, radius = 'tool0', 0.
    while node in edges:
        edge = edges[node]
        origin = edge.find('origin')
        radius += np.linalg.norm(np.fromstring(origin.attrib.get('xyz', '0 0 0'), sep=' ')) if origin is not None else 0.
        node = edge.find('parent').attrib['link']
    radius_mm = float(math.ceil(radius * 1000))
    return [[-radius_mm, -radius_mm, 0.], [radius_mm, radius_mm, radius_mm]]


def arm_addresses(model):
    return [sim._joint_qpos_address(model, n) for n in sim.ARM_JOINTS]


def nearest_equivalent(model, qpos, reference):
    qpos = qpos.copy()
    for name in sim.ARM_JOINTS:
        j, a = model.joint(name).id, sim._joint_qpos_address(model, name)
        lo, hi = model.jnt_range[j] if model.jnt_limited[j] else (-np.inf, np.inf)
        options = [qpos[a] + k*2*np.pi for k in range(-2, 3)
                   if lo-1e-10 <= qpos[a]+k*2*np.pi <= hi+1e-10]
        if options:
            qpos[a] = min(options, key=lambda value: abs(value-reference[a]))
    return qpos


def reach_bound(model, target, args):
    """Conservative wrist-2 triangle bound, including both IK tolerances."""
    data = mujoco.MjData(model)
    data.qpos[:] = sim.initial_qpos(model, np.zeros(6))
    mujoco.mj_forward(model, data)
    ee, wrist, shoulder = model.site('eeff').id, model.body('wrist_2_link').id, model.body('shoulder_link').id
    offset = data.site_xmat[ee].reshape(3, 3).T @ (data.xpos[wrist]-data.site_xpos[ee])
    # wrist_2 origin is fixed relative to the tool through the axial last-link
    # displacement; use the exact same model as the numerical solver.
    required = np.linalg.norm(target[:3, 3]+target[:3, :3]@offset-data.xpos[shoulder])
    radius, node = 0., wrist
    while node != shoulder:
        if node == 0:
            raise ValueError('Unexpected CB3 kinematic chain')
        radius += np.linalg.norm(model.body_pos[node])
        node = model.body_parentid[node]
    margin = args.ik_position_tol_mm/1000 + 2*np.linalg.norm(offset)*np.sin(np.radians(args.ik_orientation_tol_deg)/2)
    return dict(required_mm=float(required*1000), bound_mm=float(radius*1000),
                tolerance_mm=float(margin*1000), unreachable=bool(required > radius+margin))


def solutions(model, target, args, attempts, hints=()):
    bound = reach_bound(model, target, args)
    if bound['unreachable']:
        attempts.append(dict(reason='outside_urdf_reach_bound', **bound))
        return
    seeds = [q.copy() for q in hints]
    seeds += [sim.initial_qpos(model, np.radians(q)) for q in SEEDS_DEG]
    reference = seeds[0]
    seen_seeds, seen_solutions = [], []
    for index, seed in enumerate(seeds):
        if any(np.allclose(seed, other, atol=1e-10, rtol=0) for other in seen_seeds):
            continue
        seen_seeds.append(seed)
        q, pe, re, ok = sim.solve_pose_ik(model, seed, target, max_iters=args.ik_max_iters,
            dt=args.ik_dt, position_tol_m=max(args.ik_position_tol_mm/1000,.003),
            orientation_tol_rad=np.radians(max(args.ik_orientation_tol_deg,3.)), solver=args.solver)
        if ok and (args.ik_position_tol_mm<3. or args.ik_orientation_tol_deg<3.):
            # Recenter the existing posture regularizer at the coarse solution;
            # do not accept its far-seed residual as the executed endpoint.
            q,pe,re,ok=sim.solve_pose_ik(model,q,target,max_iters=args.ik_max_iters,dt=args.ik_dt,
                position_tol_m=args.ik_position_tol_mm/1000,
                orientation_tol_rad=np.radians(args.ik_orientation_tol_deg),solver=args.solver)
        attempts.append(dict(seed=index, solved=bool(ok), position_error_mm=pe*1000,
                             rotation_error_deg=float(np.degrees(re))))
        if not ok:
            continue
        q = nearest_equivalent(model, q, reference)
        if any(np.max(np.abs(q-other)) < .01 for other in seen_solutions):
            continue
        seen_solutions.append(q)
        yield q


def geometry_failures(model, samples, config, *, target_gate):
    """Record ALL failures, not only the first priority label."""
    from tools import ggx_motion as motion
    sim.check_trajectory(model, samples, required_clearance_m=.002)
    failures, data = [], mujoco.MjData(model)
    bounds = np.asarray(config['tcp_bounds_mm'])/1000
    for sample in samples:
        if sample.min_desk_clearance_m < .002:
            failures.append(dict(kind='desk', phase=sample.phase, clearance_mm=sample.min_desk_clearance_m*1000))
        tcp = support.fk_tcp(model, sample.qpos[arm_addresses(model)])
        if np.any(np.asarray(tcp[:3]) < bounds[0]) or np.any(np.asarray(tcp[:3]) > bounds[1]):
            failures.append(dict(kind='workspace', phase=sample.phase, tcp_mm=(np.asarray(tcp[:3])*1000).tolist()))
        if target_gate and sample.phase in motion.OBJECT_PHASES:
            data.qpos[:] = sample.qpos
            mujoco.mj_forward(model, data)
            gap, nearest = motion.object_clearance(model, data)
            sample.object_clearance_m, sample.object_collision = gap, gap <= 1e-9
            if sample.object_collision:
                failures.append(dict(kind='target_proxy_uncertain', phase=sample.phase,
                                     clearance_mm=gap*1000, geom=nearest))
    return failures


def rows_from_samples(model, samples):
    rows = []
    for index, sample in enumerate(samples):
        q = sample.qpos[arm_addresses(model)]
        rows.append(dict(index=index, phase=sample.phase, q_nominal=q.tolist(),
            tcp=support.fk_tcp(model, q), closure_alpha=sample.closure_alpha,
            hand_raw=support.raw_hand_command(model, sample.closure_alpha)))
    return rows


def exact_reference_rows(model,samples,candidate,center,args,standoff_mm):
    """Original GGX endpoint/axis, not a seed-dependent approximate IK target."""
    rows=rows_from_samples(model,samples)
    p,R=sim.graspgenx_to_mujoco_base_pose(candidate,object_center=center)
    pre=sim.pregrasp_base_position(p,candidate.approach_direction,standoff_mm/1000)
    def tcp(position):
        T=np.eye(4);T[:3,:3]=R;T[:3,3]=position
        return support.matrix_pose(support.UR5Bridge.hand_T_to_tcp_T(T))
    result=[dict(rows[0],phase='start',tcp=tcp(pre)),
            dict(rows[0],phase='start_to_pregrasp',tcp=tcp(pre))]
    for phase,distance,start,end in [('approach',standoff_mm,pre,p),('lift',args.lift_mm,p,p+[0,0,args.lift_mm/1000])]:
        group=[r for r in rows if r['phase']==phase]
        count=max(1,int(np.ceil(distance/args.cartesian_step_mm)))
        for i in range(1,count+1):
            hint=group[min(len(group)-1,int((i-1)*len(group)/count))]
            result.append(dict(hint,tcp=tcp(start+(end-start)*i/count)))
        if phase=='approach':
            result.extend(dict(r,tcp=tcp(p)) for r in rows if r['phase']=='close')
    return [dict(r,index=i,q_nominal_role='branch_hint_only') for i,r in enumerate(result)]


def select_geometry(model, candidate, rank, args, center, config):
    """Solve grasp first; retreat locally. Never evaluate a scan-to-grasp path."""
    import copy
    p, R = sim.graspgenx_to_mujoco_base_pose(candidate, object_center=center)
    attempts, checks = [], []
    for qgrasp in solutions(model, sim.target_eeff_transform(p, R), args, attempts):
        endpoint = sim.TrajectorySample('approach', 1., qgrasp, 0.)
        bad = geometry_failures(model, [endpoint], config, target_gate=False)
        if bad:
            checks.append(dict(stage='grasp_endpoint', failures=bad)); continue
        # Backtrack from the grasp using the same approach axis, checking every
        # 10 mm. Each shorter distance is predeclared, not a batch-specific edit.
        q, retreat = qgrasp.copy(), {}
        for mm in np.arange(10., max(STANDOFFS_MM)+.1, 10.):
            position = sim.pregrasp_base_position(p, candidate.approach_direction, mm/1000)
            solved, pe, re, ok = sim.solve_pose_ik(model, q, sim.target_eeff_transform(position, R),
                max_iters=args.ik_max_iters, dt=args.ik_dt,
                position_tol_m=args.ik_position_tol_mm/1000,
                orientation_tol_rad=np.radians(args.ik_orientation_tol_deg), solver=args.solver)
            if not ok:
                checks.append(dict(stage='retreat', standoff_mm=float(mm), reason='ik_search_exhausted')); break
            q = nearest_equivalent(model, solved, q)
            if mm in STANDOFFS_MM:
                retreat[float(mm)] = q.copy()
        for mm in STANDOFFS_MM:
            if mm not in retreat:
                continue
            local = copy.copy(args); local.standoff_mm = mm
            result, samples = sim.evaluate_candidate(model, retreat[mm], candidate, rank, local, center)
            close = [s for s in samples if s.phase == 'close']
            if close:
                for s in samples:
                    if s.phase == 'lift':
                        s.closure_alpha = close[-1].closure_alpha
                        sim.set_hand_qpos(model, s.qpos, sim._lerp_ctrl(s.closure_alpha))
            failures = geometry_failures(model, samples, config, target_gate=False)
            if result.reason.startswith('ik_failed') or failures:
                checks.append(dict(stage='local_approach', standoff_mm=mm, reason=result.reason, failures=failures)); continue
            # Same predicate/model/threshold/phase policy as actual execution.
            # Do not turn an uncertain local approach into a normal review
            # candidate and only reject it after the operator selects it.
            # Keep this first-found local path: no new collision-driven search,
            # proxy shrinking, tolerance change or alternate grasp selection.
            proxy = geometry_failures(model, samples, config, target_gate=True)
            return dict(state='local_approach_unresolved' if proxy else 'geometry_eligible',
                local_execution_check=dict(revision='shared_geometry_predicates/v1',passed=not bool(proxy)),
                standoff_mm=mm, ik_attempts=attempts,
                checks=checks, target_proxy_uncertain=any(f['kind']=='target_proxy_uncertain' for f in proxy),
                local_failures=proxy, actual_travel_checked=False, policy=policy(),
                result=asdict(result)), samples
    state = 'grasp_ik_search_exhausted' if not any(a.get('solved') for a in attempts) else 'no_local_approach_solution'
    if attempts and attempts[0].get('unreachable'):
        state = 'grasp_outside_urdf_reach'
    return dict(state=state, ik_attempts=attempts, checks=checks, actual_travel_checked=False, policy=policy()), []


def reference(source, out):
    """Materialize reviewed local geometry, without inventing scan travel."""
    from tools.ggx_template import write_json
    support.verify(source['pins'])
    saved = source['preflight']['local_reference']
    model = support.build_model(source)
    rows = saved['rows']
    close = [r for r in rows if r['phase']=='close'][-1]
    prepared = dict(schema='rh56_ggx_cb3_prepared/v1', executable=False, geometry_reference_only=True,
        rank=source['rank'], source=source, rows=rows, closure_alpha=close['closure_alpha'],
        hand_open_raw=support.raw_hand_command(model, 0), hand_close_raw=close['hand_raw'],
        nominal_start_tcp=rows[0]['tcp'], snapshot={'q': rows[0]['q_nominal']},
        minimum_desk_clearance_mm=saved['minimum_desk_clearance_mm'])
    write_json(out/'prepared.json', prepared)
    return prepared, model


def execution(model, source, reference, canonical_rows, snapshot, config, out, point=None, video=False):
    """Check the actual route, trying branches without moving any hardware."""
    from tools import ggx_template as t
    from types import SimpleNamespace
    support.verify(source['pins'])
    args = t.sim_args(source)
    args.ik_position_tol_mm=min(args.ik_position_tol_mm,.05)
    args.ik_orientation_tol_deg=min(args.ik_orientation_tol_deg,.05)
    initial = sim.initial_qpos(model, t.vector(snapshot['q'], 6))
    current = initial
    prefix = [sim.TrajectorySample('start', 0., initial.copy(), 0.)]
    if point is not None:
        hint = sim.initial_qpos(model, np.asarray(point.get('q_nominal', snapshot['q'])))
        target = support.UR5Bridge.tcp_T_to_hand_T(support.pose_matrix(point['tcp']))
        strict = SimpleNamespace(**vars(args)); strict.ik_position_tol_mm=.05; strict.ik_orientation_tol_deg=.05
        search=[]
        current = next(solutions(model, sim.target_eeff_transform(target[:3,3],target[:3,:3]), strict, search, [hint,initial]), None)
        if current is None:
            raise PlanningUnavailable('trial_start_ik_search_exhausted', search)
        sim._append_joint_segment(model,prefix,initial,current,phase='position_to_trial_start',
                                  max_step_rad=np.radians(args.path_step_deg),closure_alpha=0.)
    prefix_failures = geometry_failures(model,prefix,config,target_gate=True)
    if prefix_failures:
        raise PlanningUnavailable('trial_positioning_unavailable', prefix_failures)
    hand = support.UR5Bridge.tcp_T_to_hand_T(support.pose_matrix(canonical_rows[0]['tcp']))
    attempts, routes = [], []
    hints = [current, sim.initial_qpos(model,np.array(canonical_rows[0]['q_nominal']))]
    for preq in solutions(model,sim.target_eeff_transform(hand[:3,3],hand[:3,:3]),args,attempts,hints):
        samples = list(prefix)
        sim._append_joint_segment(model,samples,current,preq,phase='start_to_pregrasp',
                                  max_step_rad=np.radians(args.path_step_deg),closure_alpha=0.)
        q, last_tcp = preq, None
        failed = False
        for row in canonical_rows[1:]:
            if row['tcp'] != last_tcp:
                hand = support.UR5Bridge.tcp_T_to_hand_T(support.pose_matrix(row['tcp']))
                solved,pe,re,ok=sim.solve_pose_ik(model,q,sim.target_eeff_transform(hand[:3,3],hand[:3,:3]),
                    max_iters=args.ik_max_iters,dt=args.ik_dt,position_tol_m=args.ik_position_tol_mm/1000,
                    orientation_tol_rad=np.radians(args.ik_orientation_tol_deg),solver=args.solver)
                if not ok:
                    routes.append(dict(kind='ik_search_exhausted',phase=row['phase'])); failed=True; break
                q=nearest_equivalent(model,solved,q);last_tcp=row['tcp']
            q=q.copy();sim.set_hand_qpos(model,q,sim._lerp_ctrl(row['closure_alpha']))
            samples.append(sim.TrajectorySample(row['phase'],1.,q,row['closure_alpha']))
        if failed:
            continue
        failures=geometry_failures(model,samples,config,target_gate=True)
        if failures:
            routes.append(dict(kind='route_rejected',failures=failures));continue
        rows=rows_from_samples(model,samples)
        prepared=dict(reference,source=source,snapshot=snapshot,rows=rows,geometry_reference_only=False,
            nominal_start_tcp=support.fk_tcp(model,initial[arm_addresses(model)]),
            minimum_desk_clearance_mm=min(s.min_desk_clearance_m for s in samples)*1000,
            branch_search=dict(attempts=attempts,rejected_routes=routes),policy=policy())
        if point is not None:
            prepared['trial_start']=dict(point,nominal_tcp=support.fk_tcp(model,current[arm_addresses(model)]))
        t.write_json(out/'precheck.json',dict(schema='rh56_ggx_actual_route_precheck/v1',passed=True,
            hardware_contacted=False,policy=policy(),branch_search=prepared['branch_search']))
        t.write_json(out/'prepared.json',prepared)
        if video:
            result=sim.CandidateResult(source['rank'],'actual_checked_route',0.,True,'actual_route',None,None,
                prepared['minimum_desk_clearance_mm'],'','',len(samples))
            sim.write_video(out/'cb3_review.mp4',model,samples,result,args,np.array(source['preflight']['object_center_m']))
        return prepared,model
    t.write_json(out/'precheck.json',dict(schema='rh56_ggx_actual_route_precheck/v1',passed=False,
        hardware_contacted=False,ik_attempts=attempts,route_failures=routes,policy=policy()))
    raise PlanningUnavailable('actual_route_search_exhausted', routes or attempts)
