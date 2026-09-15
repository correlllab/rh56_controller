#!/usr/bin/env python3
"""Offline Thumb Reflex preview from the EXACT saved GGX input point cloud.

No execute option, cameras, serial, RTDE, motion approvals or new GGX inference.
Uses the original analytical closure/command conversion and nominal CB3 model.
Per-point simulations are diagnostics, not a real-robot execution certificate.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    import math
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--ggx-run', type=Path, help='Saved inference + crop + simulation.yaml directory')
    inputs.add_argument('--crop-report', type=Path, help='Fresh saved foreground crop, no GGX inference required; not yet a matched GGX run')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/ggx_real_lab.yaml')
    parser.add_argument('--layout-config', type=Path, default=ROOT/'configs/ggx_ten_starts_lab.yaml')
    parser.add_argument('--start-snapshot', type=Path, default=ROOT/'artifacts/vision_pick_start/start_pose.json')
    parser.add_argument('--grasp-depth-mm', type=float, default=10., help='Predeclared constant below observed top, not a tuned offset')
    parser.add_argument('--round-object-yaw-deg', type=float,
        help='Explicit round-object short-axis direction in robot base; 90 faces lab +Y/P1. Rotates proxy and grasp together, not measured points.')
    parser.add_argument('--round-object-approximation', action='store_true',
        help='Operator declares a non-directional object approximation even if measured footprint has a distinct axis; requires explicit round-object yaw.')
    parser.add_argument('--fingers', type=int, choices=[2,3,4,5], default=4, help='2=thumb/index; 3 adds middle; default remains original four-finger Reflex')
    parser.add_argument('--grasp-direction', choices=['side','legacy-top'], default='legacy-top', help='Original unrotated grasp by default; side variants require explicit selection')
    parser.add_argument('--height-fraction', type=float, default=.5, help='Side grasp center as fraction of observed height')
    parser.add_argument('--side-approach-mm', type=float, default=80., help='Horizontal final approach length; fixed policy, not collision tuning')
    parser.add_argument('--side-roll', choices=['box-aligned','photo-side','thumb-index-top','wrist-above','wrist-below'], default='box-aligned', help='Default: align palm/forward/finger-spread axes to box faces; older conventions are diagnostics')
    parser.add_argument('--point', choices=['all']+[f'P{i}' for i in range(1,11)], default='all')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--pose-only', action='store_true', help='Render static side/top alignment views; no start-path simulation or motion approval')
    parser.add_argument('--target-only', action='store_true', help='Freeze original target and scene only; no trajectory or rendering')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.round_object_approximation and args.round_object_yaw_deg is None:
        parser.error('--round-object-approximation requires --round-object-yaw-deg')
    if args.fingers == 2 and args.grasp_direction != 'legacy-top':
        parser.error('Two-finger Reflex uses the original legacy-top line solver only')
    if not math.isfinite(args.grasp_depth_mm) or args.grasp_depth_mm <= 0:
        parser.error('grasp-depth-mm must be finite and positive')
    if not math.isfinite(args.height_fraction) or not 0 < args.height_fraction < 1:
        parser.error('height-fraction must be strictly between 0 and 1')
    if not math.isfinite(args.side_approach_mm) or args.side_approach_mm <= 0:
        parser.error('side-approach-mm must be finite and positive')
    if args.round_object_yaw_deg is not None:
        if not math.isfinite(args.round_object_yaw_deg) or not -180<=args.round_object_yaw_deg<=180:
            parser.error('round-object-yaw-deg must be finite and in [-180,180]')
        if args.ggx_run or args.grasp_direction!='legacy-top':
            parser.error('Round-object yaw requires a fresh crop and legacy-top direction, not a reused GGX scene')
    return args


def planning_frame(frame, round_yaw, *, round_object_approximation=False):
    """Explicit round-object convention; never mutate measured crop metadata."""
    from copy import deepcopy
    import math
    result=deepcopy(frame)
    if round_object_approximation and round_yaw is None:
        raise ValueError('Round-object approximation requires an explicit yaw')
    if round_yaw is not None:
        if not math.isfinite(round_yaw) or not -180<=round_yaw<=180:
            raise ValueError('Invalid round-object yaw')
        if not frame['yaw_ambiguous'] and not round_object_approximation:
            raise ValueError('Round-object yaw override requires an ambiguous measured footprint; do not rotate a directional object silently')
        result['short_axis_yaw_deg']=float(round_yaw)
    return result


def photo_side_rotation(tips, reference):
    """Roll about RH56 local +Z, preserving the ORIGINAL wrist heading exactly.

    The reference photo specifies posture only, not measured world coordinates.
    Solve roll to level thumb/index; do not choose another world approach heading.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    pair=np.asarray(tips['thumb'])-tips['index']
    # Z component of reference @ Rz(roll) @ pair = A cos + B sin + C.
    a=reference[2,0]*pair[0]+reference[2,1]*pair[1]
    b=-reference[2,0]*pair[1]+reference[2,1]*pair[0]
    c=reference[2,2]*pair[2]
    radius=np.hypot(a,b)
    if radius<1e-10 or abs(c)>radius+1e-10:
        raise ValueError('Cannot level thumb/index without changing the original wrist heading')
    phase=np.arctan2(b,a)
    offset=np.arccos(np.clip(-c/radius,-1.,1.))
    solutions=[]
    for roll in (phase+offset,phase-offset):
        orientation=reference@Rotation.from_rotvec([0.,0.,roll]).as_matrix()
        heights={k:float((orientation@v)[2]) for k,v in tips.items()}
        lower=[heights[k] for k in ('middle','ring','pinky') if k in heights]
        if (abs(heights['thumb']-heights['index'])<1e-9 and heights['index']>max(lower)
                and all(a>b for a,b in zip(lower,lower[1:]))):
            degrees=float((np.degrees(roll)+180.)%360.-180.)
            solutions.append((orientation,degrees,reference[:,2].copy()))
    if len(solutions)!=1:
        raise ValueError('Original hand cannot satisfy photo-side fingertip ordering by wrist roll alone')
    return solutions[0]


def build_reflex_target(frame, observed_bounds, depth_mm, source_path, *, fingers=5,
                        grasp_direction='side', height_fraction=.5, side_approach_mm=80.,side_roll='box-aligned',
                        use_native_two_finger_minimum=False):
    import numpy as np
    from rh56_controller.vision_grasp_target import VisionGraspTarget
    from rh56_controller.grasp_viz_core import GraspVizCore
    from rh56_controller.grasp_executor import GraspExecutor
    from rh56_controller.ur5_bridge import UR5Bridge
    from rh56_controller.grasp_geometry import GraspMode
    modes={2:GraspMode.LINE_2F,3:GraspMode.PLANE_3F,4:GraspMode.PLANE_4F,5:GraspMode.PLANE_5F}
    if fingers not in modes:
        raise ValueError('Reflex preview requires 2, 3, 4, or 5 fingers')
    if fingers == 2 and grasp_direction != 'legacy-top':
        raise ValueError('Two-finger Reflex requires legacy-top line geometry')
    if use_native_two_finger_minimum and (fingers != 2 or grasp_direction != 'legacy-top'):
        raise ValueError('Explicit native minimum is limited to original two-finger line geometry')
    width, length, height = map(float, observed_bounds)
    if not 0 < depth_mm < height:
        raise ValueError('Fixed grasp depth must be inside measured object height')
    center = np.asarray(frame['center_base_mm'], dtype=float).copy()
    center[2] += height*height_fraction if grasp_direction=='side' else height-depth_mm
    target = VisionGraspTarget(str(source_path), tuple(center/1000), width, 0., depth_mm, 0.)
    # Exploratory supplied-small-object mode, explicitly requested by operator.
    # Do not test its width against the constructor's default FOUR-finger range.
    # Normal measured-geometry entry and all previous experiments are unchanged.
    core = (GraspVizCore(robot_mode=True, mink_viz=False, hand_write_only=True)
            if use_native_two_finger_minimum else
            GraspVizCore(vision_target=target, mink_viz=False, hand_write_only=True))
    try:
        core._mode = modes[fingers]
        if use_native_two_finger_minimum:
            _, native_minimum, _ = core.closure._joint_closure_range(
                'index', core.fk.ctrl_max['thumb_yaw_line'])
            if width/1000 > native_minimum:
                raise ValueError('Native-minimum experiment requires an object no wider than that minimum')
            core._grasp_center_policy = target.center_policy
            core._grasp_x, core._grasp_y, core._grasp_z = target.center_base_m
            # Original solver clamps this requested width at its own coupled
            # minimum. No jaw offset, independent joint formula or forced 12 mm.
            final = core.closure.solve(core._mode, width/1000)
        else:
            final = core.closure.solve(core._mode, core._width_target_m)
        tips = final.world_tips(center_policy='antipodal')
        opposite = np.mean([v for k,v in tips.items() if k != 'thumb'], axis=0)
        direction = tips['thumb']-opposite
        # Reuse the opposition-axis alignment from preview_tabletop_grasp_center.
        yaw = np.radians(frame['short_axis_yaw_deg'])-np.arctan2(direction[1], direction[0])
        core._plane_rz = float(yaw)
        roll_correction_deg = None
        photo_forward = None
        reference_hand_rotation = core._plane_R_matrix()@final._rot_matrix(final.base_tilt_y)
        if grasp_direction=='side' and side_roll=='box-aligned':
            short_yaw=np.radians(frame['short_axis_yaw_deg'])
            forward=np.array([-np.sin(short_yaw),np.cos(short_yaw),0.])
            if np.dot(forward[:2],center[:2])<0:
                forward=-forward
            up=np.array([0.,0.,1.])
            # Model axes: +Z wrist->fingers, +Y pinky->index, +X palm normal.
            # No fingertip equality optimization and no extra wrist roll.
            desired=np.column_stack((np.cross(up,forward),up,forward))
            plane=desired@final._rot_matrix(final.base_tilt_y).T
            core._plane_rx,core._plane_ry,core._plane_rz=core._mat_to_xyz_euler(plane)
            photo_forward=forward
        elif grasp_direction=='side' and side_roll=='photo-side':
            desired,roll_correction_deg,photo_forward=photo_side_rotation(
                final.tip_positions,reference_hand_rotation)
            # GUI plane rotation is composed with the solver's automatic tilt.
            # Remove that composition here so it cannot tilt the wrist heading.
            plane=desired@final._rot_matrix(final.base_tilt_y).T
            core._plane_rx,core._plane_ry,core._plane_rz=core._mat_to_xyz_euler(plane)
            if not np.allclose(core._plane_R_matrix()@final._rot_matrix(final.base_tilt_y),desired,atol=1e-9):
                raise ValueError('GUI composition changed requested wrist heading')
        elif grasp_direction=='side':
            # Rigidly roll the existing grasp plane around its measured opposition
            # axis: width stays horizontal; finger spread becomes vertical.
            # Explicit wrist-above/below convention, NOT chosen by IK/collision
            # scores. Both orientations use the same original solver.
            axis = np.array([np.cos(np.radians(frame['short_axis_yaw_deg'])),
                             np.sin(np.radians(frame['short_axis_yaw_deg'])),0.])
            from scipy.spatial.transform import Rotation
            aligned = core._plane_R_matrix()
            choices = []
            # A footprint axis has no semantic +/- direction. Choose its polarity
            # geometrically so the wrist faces the base, while retaining the
            # requested above/below convention. Never query feasibility to choose.
            for flip in (0.,np.pi):
                polarity = Rotation.from_rotvec(np.array([0.,0.,flip])).as_matrix()
                for sign in (1.,-1.):
                    plane = Rotation.from_rotvec(axis*sign*np.pi/2).as_matrix() @ polarity @ aligned
                    core._plane_rx,core._plane_ry,core._plane_rz = core._mat_to_xyz_euler(plane)
                    candidate = core._build_world_T_hand(final)
                    above = candidate[2,3] >= center[2]/1000
                    if above == (side_roll!='wrist-below'):
                        score=np.dot(candidate[:2,3]-center[:2]/1000,-center[:2]/1000)
                        choices.append((score,plane))
            if not choices:
                raise ValueError('No side orientation satisfies the requested wrist convention')
            plane = max(choices,key=lambda pair:pair[0])[1]
            if side_roll=='thumb-index-top':
                # Correct orientation only, around the existing vertical grasp
                # plane normal. Do not change closure, center, or approach policy.
                normal = plane@np.array([0.,0.,1.])
                native = final._rot_matrix(final.base_tilt_y)
                pair = plane@native@(final.tip_positions['thumb']-final.tip_positions['index'])
                angle = np.arctan2(-pair[2],np.cross(normal,pair)[2])
                rotations=[]
                for turn in (angle,angle+np.pi):
                    rotated=Rotation.from_rotvec(normal*turn).as_matrix()@plane
                    heights={k:float((rotated@native@v)[2]) for k,v in final.tip_positions.items()}
                    lower=[heights[k] for k in ('middle','ring','pinky') if k in heights]
                    if (abs(heights['thumb']-heights['index'])<1e-9
                            and heights['index']>max(lower)
                            and all(a>b for a,b in zip(lower,lower[1:]))):
                        rotations.append((rotated,float(np.degrees(turn))))
                if len(rotations)!=1:
                    raise ValueError('Cannot orient original fingertips as thumb/index level above remaining fingers')
                plane,roll_correction_deg=rotations[0]
            core._plane_rx,core._plane_ry,core._plane_rz = core._mat_to_xyz_euler(plane)
            if not np.allclose(core._plane_R_matrix(),plane,atol=1e-9):
                raise ValueError('GUI plane Euler conversion does not preserve side orientation')
        hand = core._build_world_T_hand(final)
        if not np.allclose(hand[:3,:3] @ final.grasp_center('antipodal')+hand[:3,3], center/1000, atol=1e-9, rtol=0):
            raise ValueError('Original analytical pose does not preserve measured grasp center')
        executor = GraspExecutor(None, None, core.fk, hand_write_only=True)
        executor._validate_force_mode(0.)
        full = executor._ctrl_to_real(final.ctrl_values)
        thumb = [1000,1000,1000,1000,full[4],full[5]]
        # thumb_yaw_line is planner bookkeeping, not a seventh physical actuator.
        from tools.ggx_real_support import HAND_ORDER
        ctrl = {k:float(final.ctrl_values.get(k,core.fk.ctrl_min[k])) for k in HAND_ORDER}
        pre = {k:float(core.fk.ctrl_min[k]) for k in ctrl}
        for k in ('thumb_proximal','thumb_yaw'):
            pre[k] = ctrl[k]
        side_pregrasp_tcp = None
        approach_direction = None
        if grasp_direction=='side':
            if photo_forward is not None:
                # Preserve the existing horizontal local staging policy; this
                # projection is NOT used to alter the hand's original heading.
                outward=-photo_forward.copy()
                outward[2]=0.
            else:
                outward = core._plane_R_matrix()@np.array([0.,0.,1.])
                outward[2] = 0.
                if np.dot(outward,hand[:3,3]-center/1000)<0:
                    outward = -outward
            outward /= np.linalg.norm(outward)
            staged_hand = hand.copy()
            staged_hand[:3,3] += outward*side_approach_mm/1000
            side_pregrasp_tcp = UR5Bridge.hand_T_to_tcp_T(staged_hand).tolist()
            approach_direction = (-outward).tolist()
        return dict(schema='rh56_shared_cloud_reflex_target/v1', strategy='Thumb Reflex',
            mode=str(core._mode), hardware_contacted=False, executable=False, motion_authorized=False,
            fingers=fingers, grasp_direction=grasp_direction,
            orientation_only_roll_correction_deg=roll_correction_deg,
            fingertip_order_contract='thumb == index > middle > ring > pinky (base Z)' if side_roll in ('photo-side','thumb-index-top') and grasp_direction=='side' else None,
            hand_forward_axis_base=hand[:3,2].tolist(),
            reference_hand_rotation=reference_hand_rotation.tolist(),
            photo_interpretation='original local hand +Z preserved exactly; local wrist roll only; photo is not calibration' if side_roll=='photo-side' and grasp_direction=='side' else None,
            box_alignment_contract='hand +X parallel short axis, +Y vertical up, +Z parallel long axis; no exact thumb/index height constraint' if side_roll=='box-aligned' and grasp_direction=='side' else None,
            plane_euler_deg=np.degrees([core._plane_rx,core._plane_ry,core._plane_rz]).tolist(),
            contact_plane_normal_base=(core._plane_R_matrix()@np.array([0.,0.,1.])).tolist(),
            active_tip_positions_base_m={k:(hand[:3,:3]@v+hand[:3,3]).tolist() for k,v in final.tip_positions.items()},
            side_selection=side_roll if grasp_direction=='side' else None,
            side_axis_polarity_policy='wrist toward base within requested roll; no IK/collision selection' if grasp_direction=='side' else None,
            height_fraction=height_fraction if grasp_direction=='side' else None,
            side_pregrasp_tcp_T=side_pregrasp_tcp, side_approach_mm=side_approach_mm if grasp_direction=='side' else None,
            local_approach_direction_base=approach_direction,
            measured_width_mm=width, measured_long_mm=length, measured_height_mm=height,
            grasp_depth_mm=depth_mm, grasp_center_base_mm=center.tolist(),
            measured_short_axis_yaw_deg=frame['short_axis_yaw_deg'], hand_yaw_deg=float(np.degrees(yaw)),
            yaw_ambiguous=frame['yaw_ambiguous'], yaw_policy='same frozen measured footprint axis as GGX crop',
            achieved_width_mm=float(final.width*1000), width_residual_mm=float(final.width*1000-width),
            width_policy='operator_requested_native_two_finger_minimum' if use_native_two_finger_minimum else 'original_requested_width',
            width_offset_mm=0., closure_margin_mm=0., empirical_xy_offset_mm=[0.,0.], local_mount_correction=None,
            final_hand_T=hand.tolist(), final_tcp_T=UR5Bridge.hand_T_to_tcp_T(hand).tolist(),
            thumb_ctrl=pre, final_ctrl=ctrl, thumb_raw=thumb, final_raw=full,
            force_N=0., hand_write_only=True, adaptive_force_enabled=False,
            original_executor_timing_s=dict(thumb_wait=.2, fingers_wait=.25),
            phase_order=['thumb_position']+(['moveL_to_side_pregrasp'] if grasp_direction=='side' else [])+
                        ['blocking_moveL_to_final_pose','remaining_fingers_close'],
            thumb_contact_feedback_trigger=False,
            limitations=['Measured visible bounds, not full-object truth or supplied dimensions.',
                'Grasp depth is a predeclared policy, not inferred from GGX success or a contact sensor.',
                'No old cube empirical corrections; real command-to-gap accuracy remains unverified.',
                'Original force=0 position-control reflex; not feedback-triggered or adaptive-force reflex.'])
    finally:
        core.cleanup()


def render_alignment_views(out, model, target, snapshot, sim_args):
    """Static nominal final-pose views, explicitly NOT a path check."""
    import numpy as np
    import mujoco
    import cv2
    from tools import ggx_planning as planning, preflight_ggx_ur5_table as sim
    hand=np.asarray(target['final_hand_T'])
    seed=sim.initial_qpos(model,np.asarray(snapshot['q']))
    solved=next(planning.solutions(model,sim.target_eeff_transform(hand[:3,3],hand[:3,:3]),
                                  sim_args,[],[seed]),None)
    if solved is None:
        raise ValueError('Static final-pose IK unavailable; no alignment picture rendered')
    sim.set_hand_qpos(model,solved,target['final_ctrl'])
    data=mujoco.MjData(model);data.qpos[:]=solved;mujoco.mj_forward(model,data)
    model.vis.global_.offwidth=max(int(model.vis.global_.offwidth),720)
    model.vis.global_.offheight=max(int(model.vis.global_.offheight),480)
    renderer=mujoco.Renderer(model,height=480,width=720)
    look=(hand[:3,3]+np.asarray(target['grasp_center_base_mm'])/1000)/2
    normal=hand[:3,0]
    views={}
    try:
        for label,elevation in [('side',0.),('top',-89.)]:
            camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:]=look;camera.distance=.62
            camera.azimuth=float(np.degrees(np.arctan2(-normal[1],-normal[0])))
            camera.elevation=elevation
            renderer.update_scene(data,camera=camera)
            rgb=renderer.render().copy()
            cv2.rectangle(rgb,(0,0),(719,52),(30,30,30),-1)
            cv2.putText(rgb,'BOX ALIGNMENT ONLY - NOT A PATH / COLLISION PASS',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,230,130),1)
            cv2.putText(rgb,f'{label}: unchanged hand geometry and closure commands',(8,42),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,230,130),1)
            path=out/f'alignment_{label}.png'
            cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            views[label]=str(path.resolve())
    finally:
        renderer.close()
    return dict(status='static_pose_rendered_not_path_checked',views=views,executable=False)


def render_reflex(path, model, samples, args, center):
    """Same MuJoCo camera/MP4 backend, explicitly labeled as Reflex, not GGX."""
    import cv2
    import numpy as np
    import mujoco
    import imageio_ffmpeg
    from tools import preflight_ggx_ur5_table as sim
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.video_width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), args.video_height)
    renderer = mujoco.Renderer(model, height=args.video_height, width=args.video_width)
    camera = sim._camera(args, np.asarray(center))
    data = mujoco.MjData(model)
    writer = imageio_ffmpeg.write_frames(str(path),(args.video_width,args.video_height),fps=args.fps,
        codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=2,
        output_params=['-movflags','+faststart'])
    try:
        writer.send(None)
        for sample in samples:
            data.qpos[:] = sample.qpos
            mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera=camera)
            frame = renderer.render().copy()
            lines = ['ANALYTICAL THUMB REFLEX - OFFLINE DIAGNOSTIC ONLY',
                'NOT A COLLISION PASS / NOT APPROVED FOR HARDWARE',
                f'phase: {sample.phase}',
                f'desk gap: {sample.min_desk_clearance_m*1000:.1f} mm; target proxy gap: {sample.object_clearance_m*1000:.1f} mm',
                'Negative target gap = model overlap; contact semantics not certified']
            cv2.rectangle(frame,(0,0),(args.video_width-1,116),(35,35,35),-1)
            for i,line in enumerate(lines):
                cv2.putText(frame,line,(8,19+22*i),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,220,100),1,cv2.LINE_AA)
            writer.send(frame)
    finally:
        writer.close()
        renderer.close()


def simulate_point(model, target, point, snapshot, config, sim_args, out, video):
    import math
    import numpy as np
    import mujoco
    from tools import ggx_template as t, ggx_real_support as s, ggx_motion as motion
    from tools import ggx_planning as planning, preflight_ggx_ur5_table as sim
    required_clearance = motion.reflex_desk_clearance_m(config)
    q = sim.initial_qpos(model, np.asarray(snapshot['q']))
    sim.set_hand_qpos(model,q,sim.GGX_OPEN_CTRL)
    samples = [sim.TrajectorySample('scan_start',0.,q.copy(),0.)]
    record = dict(point_id=point['id'], status='starting', executable=False,
                  required_desk_clearance_mm=required_clearance*1000,
                  geometry_certified_for_motion=False, hardware_contacted=False,
                  execution_settings=dict(hand_speed_raw=config['hand_speed_raw'], hand_force_raw=config['hand_force_raw']),
                  phase_metrics=[], stopped_phase=None)
    phase = 'position_to_trial_start'
    try:
        # Keep the existing arm IK selection; prepare the hand at scan height.
        hint = sim.initial_qpos(model,np.asarray(point['q_nominal']))
        hand = s.UR5Bridge.tcp_T_to_hand_T(s.pose_matrix(point['tcp']))
        qpoint = next(planning.solutions(model,sim.target_eeff_transform(hand[:3,3],hand[:3,:3]),
                                        sim_args,[],[hint,q]),None)
        if qpoint is None:
            raise ValueError('Fixed start IK unresolved')
        def hand_phase(name, before, after):
            nonlocal q
            for alpha in np.linspace(0.,1.,16):
                q = q.copy()
                ctrl = {k:(1-alpha)*before[k]+alpha*after[k] for k in sim.GGX_OPEN_CTRL}
                sim.set_hand_qpos(model,q,ctrl)
                samples.append(sim.TrajectorySample(name,float(alpha),q.copy(),float(alpha)))
        phase = 'thumb_position'
        hand_phase(phase,sim.GGX_OPEN_CTRL,target['thumb_ctrl'])
        phase = 'position_to_trial_start'
        sim.set_hand_qpos(model,qpoint,target['thumb_ctrl'])
        sim._append_joint_segment(model,samples,q,qpoint,phase=phase,
                                  max_step_rad=np.radians(sim_args.path_step_deg),closure_alpha=0.)
        q = qpoint
        final_tcp = s.matrix_pose(np.asarray(target['final_tcp_T']))
        lift_tcp = list(final_tcp); lift_tcp[2] += sim_args.lift_mm/1000
        segments = []
        if target.get('side_pregrasp_tcp_T') is not None:
            segments.append(('reflex_side_pregrasp',s.matrix_pose(np.asarray(target['side_pregrasp_tcp_T'])),target['thumb_ctrl']))
        segments += [('reflex_approach',final_tcp,target['thumb_ctrl']),('lift',lift_tcp,target['final_ctrl'])]
        for phase, goal, ctrl in segments:
            if phase == 'lift':
                hand_phase('remaining_fingers_close',target['thumb_ctrl'],target['final_ctrl'])
            start = s.fk_tcp(model,q[planning.arm_addresses(model)])
            distance,angle = s.pose_error(start,goal)
            count = max(2,math.ceil(distance/2)+1,math.ceil(angle)+1)
            for alpha in np.linspace(0.,1.,count)[1:]:
                solved = t.solve_tcp(model,q,motion.pose_at(start,goal,float(alpha)),sim_args)
                solved = planning.nearest_equivalent(model,solved,q)
                if np.max(np.abs(solved[planning.arm_addresses(model)]-q[planning.arm_addresses(model)])) > .15:
                    raise ValueError('Nominal moveL IK branch discontinuity')
                q = solved
                sim.set_hand_qpos(model,q,ctrl)
                samples.append(sim.TrajectorySample(phase,float(alpha),q.copy(),0. if phase=='reflex_approach' else 1.))
        record['status'] = 'nominal_sequence_solved_contact_unverified'
    except ValueError as exc:
        record.update(status='nominal_ik_unavailable',stopped_phase=phase,error=str(exc))
    sim.check_trajectory(model,samples,required_clearance_m=required_clearance)
    data = mujoco.MjData(model)
    metrics = {}
    for sample in samples:
        row = metrics.setdefault(sample.phase,dict(phase=sample.phase,samples=0,minimum_desk_gap_mm=float('inf'),
            minimum_target_gap_mm=float('inf'),target_overlap_samples=0,desk_reject_samples=0,
            forbidden_target_overlap_samples=0,minimum_forbidden_target_gap_mm=float('inf'),
            target_contact_policy=motion.REFLEX_CONTACT_POLICY))
        data.qpos[:] = sample.qpos; mujoco.mj_forward(model,data)
        contact = motion.reflex_target_check(model,data,sample.phase)
        gap,nearest = contact['minimum_target_gap_mm']/1000,contact['nearest_target_geom']
        row['forbidden_target_overlap_samples'] += int(contact['forbidden_target_overlap'])
        row['finger_contact_allowed'] = contact['finger_contact_allowed']
        if contact['minimum_forbidden_target_gap_mm'] < row['minimum_forbidden_target_gap_mm']:
            row['minimum_forbidden_target_gap_mm'] = contact['minimum_forbidden_target_gap_mm']
            row['nearest_forbidden_target_geom'] = contact['nearest_forbidden_target_geom']
        sample.object_clearance_m = float(gap)
        sample.object_collision = bool(gap <= 1e-9)
        sample.nearest_object_geom = nearest
        if sample.object_collision and 'first_overlap' not in row:
            row['first_overlap'] = dict(phase_fraction=sample.phase_alpha, gap_mm=float(gap*1000),geom=nearest)
        row['samples'] += 1
        row['minimum_desk_gap_mm'] = min(row['minimum_desk_gap_mm'],sample.min_desk_clearance_m*1000)
        if gap*1000 < row['minimum_target_gap_mm']:
            row.update(minimum_target_gap_mm=gap*1000,nearest_target_geom=nearest)
        row['target_overlap_samples'] += int(gap <= 1e-9)
        sample.desk_collision=motion.desk_clearance_blocked(sample.min_desk_clearance_m,required_clearance)
        row['desk_reject_samples'] += int(sample.desk_collision)
    record['phase_metrics'] = list(metrics.values())
    record['sample_count'] = len(samples)
    record['nominal_goals'] = []
    if record['status']=='nominal_sequence_solved_contact_unverified' and target.get('side_pregrasp_tcp_T') is None:
        for name,phase,kind,ctrl in [('position_to_trial_start','position_to_trial_start','moveJ',target['thumb_ctrl']),
                                    ('reflex_approach','approach','moveL',target['thumb_ctrl']),
                                    ('lift','lift','moveL',target['final_ctrl'])]:
            last=[s for s in samples if s.phase==name][-1]
            arm=last.qpos[planning.arm_addresses(model)]
            record['nominal_goals'].append(dict(phase=phase,kind=kind,goal_tcp=s.fk_tcp(model,arm),
                q_nominal=arm.tolist(),closure_alpha=1. if phase=='lift' else 0.,hand_ctrl=dict(ctrl),
                target_contact_policy=motion.REFLEX_CONTACT_POLICY if phase in ('approach','lift') else None))
    record['limitations'] = ['Diagnostic distances only: intended thumb contact and unintended proxy penetration are not silently equated.',
        'Reflex approach/closure/lift finger-target contact recorded, not rejected; preparation and nonfinger contacts remain gated by runner.',
        'No camera mount/unknown fixtures or full self-collision certificate.',
        'IK/render timing is not execution timing; no physical dynamics or grasp success simulated.']
    sim._write_csv(out/'trajectory.csv',sim._trajectory_rows(model,samples))
    if video:
        render_reflex(out/'reflex_preview.mp4',model,samples,sim_args,
                      np.asarray(target['grasp_center_base_mm'])/1000)
        record['video'] = str((out/'reflex_preview.mp4').resolve())
    t.write_json(out/'summary.json',record)
    return record


def main(argv=None):
    import numpy as np
    import yaml
    from tools import run_ggx_official_filter as official, ggx_template as t, ggx_fixed_starts as fixed
    from tools import run_ggx_real as real, ggx_real_support as support, preflight_ggx_ur5_table as sim
    from tools import run_ggx_review as ui, run_ggx_ten_starts as ten
    from tools import prepare_ggx_tabletop_cloud as cloud
    from tools.preview_tabletop_target import load_perception_config
    args = parse_args(argv)
    out = (args.out or ROOT/'artifacts/reflex_comparison_previews'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')).resolve()
    out.mkdir(parents=True,exist_ok=False)
    report = dict(schema='rh56_reflex_shared_perception_preview/v1',status='starting',executable=False,
                  hardware_contacted=False,motion_authorized=False,trials=[],
                  assumptions=['Saved segmented cloud; GGX lineage verified only when --ggx-run is supplied; no inference performed.',
                    'Existing measured footprint rectangle, not a new PCA algorithm; width/height from observed bounds, not operator priors.',
                    'Original analytical closure and Thumb Reflex commands; side mode adds an explicit horizontal staging segment.',
                    'Same setup-wide fixed arm starts, initial GGX-open hand, nominal mount, object proxy, desk and lift distance.',
                    'One analytical target frozen across all points; no per-point yaw/width/center optimization.',
                    'Offline development preview only, not a matched physical benchmark result or a motion approval.'])
    try:
        directory = args.ggx_run.resolve() if args.ggx_run else None
        if directory:
            inf,_,_,_,_,pins = official.prepare_inputs(directory/'inference/summary.json')
            path = Path(inf['source_point_cloud'])
            crop = ui.read_json(path.parent/'summary.json')
        else:
            crop = ui.read_json(args.crop_report)
            path = Path(crop['point_cloud'])
            pins = ui.pin([args.crop_report,path])
        cloud.validate_foreground_report(crop)
        preview = ui.read_json(Path(crop['source_preview']))
        pc,_ = load_perception_config(Path(preview['config']))
        if not directory:
            capture = Path(preview['capture'])
            expected = {crop['source_preview']:crop['source_preview_sha256'],
                        str(capture/'camera_check_depth_u16.png'):crop['source_depth_sha256'],
                        pc['calibration']:crop['calibration_sha256'],
                        preview['config']:preview['config_sha256'],
                        str(Path(crop['source_preview']).parent/'foreground.png'):crop['source_foreground_sha256']}
            ui.verify(expected)
            pins.update(expected)
            pins.update(ui.pin([capture/'camera_check_rgb.png',capture/'camera_check.yaml',pc['desk_plane']]))
        normal = yaml.safe_load(Path(pc['desk_plane']).read_text())['plane']['normal_base']
        points = np.load(path,allow_pickle=False)
        frame,proxy,_ = cloud.foreground_geometry(points*1000,crop['dimensions_prior_mm'],normal,pc['desk_z_mm'],
                                                 freeze_reference_yaw=crop.get('reference_yaw_frozen',False))
        # Given priors affect only the shared obstacle proxy, not planner inputs.
        if not np.allclose(proxy['observed_bounds_mm'],crop['collision_proxy']['observed_bounds_mm'],atol=1e-8,rtol=0):
            raise ValueError('Saved point-cloud geometry no longer matches crop')
        target_frame=planning_frame(frame,args.round_object_yaw_deg,
                                    round_object_approximation=args.round_object_approximation)
        target = build_reflex_target(target_frame,proxy['observed_bounds_mm'],args.grasp_depth_mm,path,
            fingers=args.fingers,grasp_direction=args.grasp_direction,height_fraction=args.height_fraction,
            side_approach_mm=args.side_approach_mm,side_roll=args.side_roll)
        if args.round_object_yaw_deg is not None:
            target.update(measured_short_axis_yaw_deg=frame['short_axis_yaw_deg'],
                          round_object_yaw_deg=args.round_object_yaw_deg,
                          round_object_approximation=args.round_object_approximation,
                          yaw_policy='operator-selected round-object axis; fixed for all starts and XY relocations')
            target['limitations'].append('Rotated round-object box is an approximation, not a guaranteed enclosing box of the unchanged measured cloud.')
            report['assumptions'].append('Explicit round-object yaw changes proxy and grasp orientation together; dimensions/center/cloud/layout/thresholds unchanged. Not a per-point feasibility search or matched original GGX orientation.')
        report['target'] = target
        config = real.load_config(args.config)
        layout = fixed.build(args.layout_config,config,Path(preview['config']),
                             grasp_center_mm=target['grasp_center_base_mm'])
        t.write_json(out/'fixed_layout.json',layout)
        if directory:
            simulation_path = directory/'simulation.yaml'
        else:
            simulation_path = out/'simulation.yaml'
            # Existing crop-to-scene conversion, not copied coordinates from the old box.
            simulation = ui.crop_sim_config(crop,crop['dimensions_prior_mm'],pc,
                ui.read_json(args.start_snapshot),out/'unused_no_ggx_candidates.yml')
            if args.round_object_yaw_deg is not None:
                # Scene X is the long axis; grasp opposition uses the short axis.
                simulation['object_yaw_deg']=args.round_object_yaw_deg-90.
            simulation_path.write_text(yaml.safe_dump(simulation))
        sim_args = sim.parse_args(['--config',str(simulation_path),'--no-video'])
        sim_args.ik_position_tol_mm = sim_args.ik_orientation_tol_deg = .05
        from tools.ggx_reset import desk_source
        desk=desk_source(config,pc)
        center = [sim_args.object_x_m,sim_args.object_y_m,sim_args.table_height_m+sim_args.object_size_mm[2]/2000]
        source = dict(preflight=dict(object_center_m=center,object_size_mm=sim_args.object_size_mm,
            object_yaw_deg=sim_args.object_yaw_deg,table_height_m=sim_args.table_height_m),
            desk_envelope=config['desk_envelope'],desk_top_m=desk['preflight']['table_height_m'])
        for key in ('desk_surface_plane','desk_surface_policy'):
            if key in desk: source[key]=desk[key]
        if args.round_object_yaw_deg is not None:
            source['round_object_orientation']=dict(short_axis_yaw_deg=args.round_object_yaw_deg,
                measured_short_axis_yaw_deg=frame['short_axis_yaw_deg'],point_cloud_rotated=False,
                operator_declared_approximation=args.round_object_approximation,
                proxy_enclosure_certified=False,policy='fixed operator-selected round-object approximation')
        model = support.build_model(source)
        snapshot = ui.read_json(args.start_snapshot)
        pins.update({**ten.code_pins(args.config),**layout['source_sha256'],**ui.pin([
            __file__,simulation_path,args.start_snapshot,ROOT/'rh56_controller/grasp_viz_core.py',
            ROOT/'tools/prepare_ggx_tabletop_cloud.py'])})
        report.update(shared_cloud=str(path),shared_cloud_sha256=ui.digest(path),
                      ggx_inference=str(directory/'inference/summary.json') if directory else None,
                      ggx_input_match_verified=bool(directory),
                      input_sha256=pins,scene=source,model='CB3 UR5 + unchanged RH56',lift_mm=sim_args.lift_mm,
                      supplied_dimensions_used_for_planner=False,supplied_dimensions_used_for_shared_collision_proxy=True)
        t.write_json(out/'reflex_target.json',target)
        t.write_json(out/'summary.json',report)
        if args.target_only:
            ui.verify(pins)
            report['status']='target_frozen_not_motion_approved'
            return 0
        if args.pose_only:
            ui.verify(pins)
            report['alignment']=render_alignment_views(out,model,target,snapshot,sim_args)
            ui.verify(pins)
            report['status']='static_alignment_only_not_motion_approved'
            return 0
        for point in layout['points']:
            if args.point != 'all' and point['id'] != args.point:
                continue
            ui.verify(pins)
            folder = out/point['id'];folder.mkdir()
            trial = simulate_point(model,target,point,snapshot,config,sim_args,folder,args.video)
            report['trials'].append(trial)
            print(f'Reflex {point["id"]}: {trial["status"]}; no hardware or motion approval',flush=True)
        ui.verify(pins)
        report['status'] = 'offline_preview_complete_not_motion_approved'
        return 0
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        print(report['error'],file=sys.stderr)
        return 1
    finally:
        t.write_json(out/'summary.json',report)
        with (out/'summary.csv').open('w',newline='') as stream:
            fields=['point_id','status','stopped_phase','sample_count','error','video']
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(report['trials'])
        print(f'Reflex offline report: {out/"summary.json"}',flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
