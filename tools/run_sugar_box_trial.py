#!/usr/bin/env python3
"""Supervised known-size flat sugar-box trial using the original Plan/LiveCycle.

Offline saved-image preview by default. Hardware execution always captures afresh,
requires Enter confirmation, and never retries or releases automatically.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from tools import run_vision_pick_cycle as cycle
from tools.preview_tabletop_target import load_perception_config, main as preview_tabletop
from rh56_controller.vision_grasp_target import VisionGraspTarget


def trial_config():
    c = cycle.load_config(ROOT/'configs/ur5_iterative_cube_lab.yaml')
    # Separate experiment inputs, never edit the frozen cube files.
    c.update(object_width_mm=90., object_height_mm=40., grasp_xy_offset_base_mm=[0.,0.])
    return c


def select_target(preview, c, source):
    """Promote only a unique, complete, known-size flat-box observation."""
    if preview.get('schema') != 'rh56_tabletop_foreground_preview/v2' or preview.get('executable_target') is not False:
        raise ValueError('Expected non-executable tabletop evidence')
    rows = [r for r in preview['candidates'] if r['geometry_candidate']]
    if len(rows) != 1 or preview.get('geometry_candidate_count') != 1:
        raise ValueError('Require exactly one complete foreground candidate')
    row = rows[0]; patch = row['top_patch']
    if row['rejection_reasons'] or patch is None or patch.get('yaw_ambiguous',True):
        raise ValueError('Ambiguous or rejected tabletop surface')
    dims = cycle.finite([patch['long_mm'],patch['short_mm'],patch['height_median_mm']],(3,))
    if np.any(np.abs(dims-[176.,90.,40.]) > 5.):
        raise ValueError(f'Not the reviewed flat sugar-box dimensions: {dims.tolist()}')
    if (not np.isfinite(patch['core_spatial_height_mad_mm'])
            or patch['core_spatial_height_mad_mm'] > 3.):
        raise ValueError('Top surface depth is too noisy')
    axis = float(cycle.finite(patch['short_axis_yaw_deg'],())) % 180
    if abs(axis-90) > 10:
        raise ValueError('Box orientation differs from the reviewed near-base-Y closing direction')
    top = cycle.finite(patch['center_base_mm'],(3,))
    # Execution target-center check remains; not an all-object-pixels vision gate.
    cycle.within(top,c['target_bounds_mm'],'Measured top center')
    center = top.copy(); center[2] -= c['grasp_depth_mm']
    cycle.within(center,c['target_bounds_mm'],'Grasp center')
    target = VisionGraspTarget(str(source),tuple(center/1000),c['object_width_mm'],
                              c['width_offset_mm']-c['closure_margin_mm'],
                              c['grasp_depth_mm'],c['yaw_deg'])
    return target,axis,patch


def check_consistency(observations, max_spread_mm=3.):
    if len(observations)<2:
        raise ValueError('Require multiple fresh observations')
    patches=[o[2] for o in observations]
    points=np.array([p['center_base_mm'] for p in patches],dtype=float)
    dims=np.array([[p['long_mm'],p['short_mm'],p['height_median_mm']] for p in patches])
    axes=np.array([o[1] for o in observations])
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(dims)) or not np.all(np.isfinite(axes)):
        raise ValueError('Nonfinite observation')
    spread=float(np.max(np.linalg.norm(points[:,None]-points[None,:],axis=-1)))
    if spread>max_spread_mm or np.max(np.ptp(dims,axis=0))>3 or np.ptp(axes)>2:
        raise ValueError('Object center, dimensions or yaw changed across captures')
    return dict(center_spread_mm=spread,dimension_spans_mm=np.ptp(dims,axis=0).tolist(),
                yaw_span_deg=float(np.ptp(axes)))


class TabletopPlan:
    """Input adapter only; original core, alignment, command mapper and waypoints."""
    def __init__(self,c,target,axis_yaw=None):
        from rh56_controller.grasp_viz_core import GraspVizCore
        from rh56_controller.grasp_executor import GraspExecutor
        from tools.run_vision_plan_trial import load_local_alignment,command_hand_pose,pose,UR5Bridge
        self.core=GraspVizCore(vision_target=target,mink_viz=False,hand_write_only=True)
        try:
            final=self.core.closure.solve(self.core._mode,target.planner_width_m)
            if axis_yaw is not None:
                tips=final.world_tips(center_policy='antipodal')
                d=tips['thumb']-np.mean([p for k,p in tips.items() if k!='thumb'],axis=0)
                yaw=float(axis_yaw-np.rad2deg(np.arctan2(d[1],d[0])))
                target=replace(target,yaw_deg=yaw)
                self.core._plane_rz=np.deg2rad(yaw)
            validation=yaml.safe_load(Path(c['alignment_validation']).read_text())
            anchor=validation['hand_command_assumed']
            mount,alignment=load_local_alignment(Path(c['local_alignment']),Path(c['alignment_validation']),anchor)
            closures=self.core._compute_plan_closures(10,final)
            placement=cycle.grasp_placement_offset(c)
            self.waypoints=[]
            for r in closures:
                T=command_hand_pose(self.core._build_world_T_hand(r),mount).copy()
                T[:3,3]+=placement
                self.waypoints.append((T,r))
            mapper=GraspExecutor(None,None,self.core.fk,hand_write_only=True)
            self.commands=[mapper._ctrl_to_real(r.ctrl_values) for r in closures]
            final_yaw=self.commands[-1][5]
            for cmd in self.commands: cmd[5]=final_yaw
            if self.commands[0]!=[1000,1000,1000,1000,1000,0]:
                raise ValueError('Original opposing-thumb open approach required')
            self.tcp=[pose(UR5Bridge.hand_T_to_tcp_T(T)).tolist() for T,_ in self.waypoints]
            self.lift=cycle.lift_pose(self.tcp[-1],c['lift_mm']).tolist()
            for p in [*self.tcp,self.lift]:
                cycle.within(np.array(p[:3])*1000,c['tcp_bounds_mm'],'Planned TCP')
            self.report=dict(schema='rh56_sugar_existing_plan/v1',target=target.metadata(),
                             input_reference_width_mm=target.planner_width_m*1000,
                             returned_centroid_width_mm=final.width*1000,
                             hand_commands=self.commands,tcp_waypoints=self.tcp,lift_tcp=self.lift,
                             alignment=alignment,alignment_anchor_command=anchor,
                             final_shape_extrapolated=self.commands[-1]!=anchor,
                             grasp_xy_offset_base_mm=(placement[:2]*1000).tolist(),
                             commanded_grasp_center_base_m=(np.array(target.center_base_m)+placement).tolist(),
                             entry_policy='direct_to_original_plan_approach',
                             hardware_ik_validated=False,collision_free_verified=False)
        except BaseException:
            self.core.cleanup(); raise

    def close(self): self.core.cleanup()


class SugarCycle(cycle.LiveCycle):
    # Only scan return differs. Grasp, serial ordering and lift stay inherited.
    SCAN_JOINT_SPEED = .10  # rad/s, leading axis; supervised commissioning value
    SCAN_JOINT_ACCEL = .20  # rad/s²

    def prepare(self, value):
        start=json.loads(Path(self.c['start_pose']).read_text())
        self.scan_q=cycle.finite(start['q'],(6,)).tolist()
        self._preparing_scan=True
        try:
            return super().prepare(value)
        finally:
            self._preparing_scan=False

    def preflight(self, goals, limits, *, start_snapshot=None):
        if not getattr(self,'_preparing_scan',False):
            return super().preflight(goals,limits,start_snapshot=start_snapshot)
        # LiveCycle.prepare delegates its entry preflight here. Keep the original
        # vertical reset check, but never submit scan return to Cartesian IK.
        expected,expected_limits=cycle.scan_segments(self.c,self.scan,self.reset_tcp)
        if goals!=expected or limits!=expected_limits or start_snapshot is not None:
            raise RuntimeError('Unexpected scan entry preflight')
        origin=cycle.stationary_snapshot(self.bridge._iface.recv)
        if self.reset_tcp is not None:
            super().preflight([self.reset_tcp],[150])
            q=self.report['controller_preflights'][-1]['q_samples'][-1]
            tcp=self.reset_tcp
        else:
            self.allowed_goals,self.next_goal=[],0
            q,tcp=origin['q'],origin['tcp']
        self.check_joint_scan(q,tcp)
        cycle.check_stationary_origin(origin,cycle.stationary_snapshot(self.bridge._iface.recv))

    def check_joint_scan(self, start_q, start_tcp):
        from tools.run_vision_plan_trial import rot_error
        ctrl,recv=self.bridge._iface.ctrl,self.bridge._iface.recv
        cycle.normal(recv)
        if np.max(np.abs(cycle.finite(ctrl.getTCPOffset(),(6,))))>1e-8:
            raise RuntimeError('Expected zero TCP offset')
        q0=cycle.finite(start_q,(6,));q1=cycle.finite(self.scan_q,(6,))
        tcp0=cycle.finite(start_tcp,(6,));scan=cycle.finite(self.scan,(6,))
        distance=float(np.linalg.norm(scan[:3]-tcp0[:3])*1000)
        angle=float(np.rad2deg(rot_error(tcp0,scan)))
        _,entry_limits=cycle.scan_segments(self.c,self.scan)
        if distance>entry_limits[-1]+1e-6 or angle>self.c['max_rotation_deg']:
            raise RuntimeError('Joint scan endpoint exceeds original entry distance/rotation limits')
        # Use raw saved joint angles, never wrap to another revolution/solution.
        count=max(3,int(np.ceil(np.max(np.abs(q1-q0))/.01))+1)
        record=dict(schema='rh56_joint_scan_preflight/v1',status='checking',
                    start_q=q0.tolist(),goal_q=q1.tolist(),start_tcp=tcp0.tolist(),
                    goal_tcp=scan.tolist(),speed_rad_s=self.SCAN_JOINT_SPEED,
                    accel_rad_s2=self.SCAN_JOINT_ACCEL,sample_step_rad_max=.01,
                    collision_free_verified=False,samples=[])
        self.report.setdefault('joint_scan_preflights',[]).append(record)
        try:
            for t in np.linspace(0,1,count):
                cycle.normal(recv)
                q=(q0+t*(q1-q0)).tolist()
                if not ctrl.isJointsWithinSafetyLimits(q):
                    raise RuntimeError('Controller joint scan safety check failed')
                tcp=cycle.finite(ctrl.getForwardKinematics(q,[0.]*6),(6,))
                cycle.within(tcp[:3]*1000,self.c['tcp_bounds_mm'],'Joint scan sampled TCP')
                if not ctrl.isPoseWithinSafetyLimits(tcp.tolist()):
                    raise RuntimeError('Controller joint scan pose safety check failed')
                record['samples'].append(dict(fraction=float(t),q=q,tcp=tcp.tolist()))
            for actual,expected in ((record['samples'][0]['tcp'],tcp0),
                                    (record['samples'][-1]['tcp'],scan)):
                if np.linalg.norm(np.array(actual[:3])-expected[:3])>.001 or rot_error(actual,expected)>np.deg2rad(.5):
                    raise RuntimeError('Saved/reset joints do not match scan path TCP endpoints')
            xyz=np.array([s['tcp'][:3] for s in record['samples']])*1000
            record.update(status='passed',tcp_min_mm=xyz.min(axis=0).tolist(),
                          tcp_max_mm=xyz.max(axis=0).tolist(),
                          max_joint_travel_deg=float(np.rad2deg(np.max(np.abs(q1-q0)))))
        except BaseException as exc:
            record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            raise
        return record

    def move_scan(self, _):
        from tools.run_vision_plan_trial import rot_error
        if not self.args.execute or self.report.get('confirmation_method')!='interactive_enter':
            raise RuntimeError('Joint scan motion requires explicit execute and confirmation')
        if self.reset_tcp is not None:
            self.report['stage']='reset_lift'
            self.move_tcp(self.reset_tcp)
            self.report['low_entry_reset']['completed']=True
        self.report['stage']='move_scan'
        recv,ctrl=self.bridge._iface.recv,self.bridge._iface.ctrl
        origin=cycle.stationary_snapshot(recv)
        checked=self.check_joint_scan(origin['q'],origin['tcp'])
        cycle.check_stationary_origin(origin,cycle.stationary_snapshot(recv))
        if self.cancelled.is_set(): raise RuntimeError('Cycle cancelled before joint scan')
        self.report['hardware_commands_attempted']=True
        self.report['joint_scan_motion_attempted']=True
        # Async MoveJ permits stopJ from this same controlling thread. Never
        # leave a blocked motion worker alive or retry after an error.
        try:
            if not ctrl.moveJ(self.scan_q,self.SCAN_JOINT_SPEED,self.SCAN_JOINT_ACCEL,True):
                raise RuntimeError('Controller rejected joint scan moveJ')
            begin=time.monotonic()
            timeout=np.deg2rad(checked['max_joint_travel_deg'])/self.SCAN_JOINT_SPEED+30
            while True:
                cycle.normal(recv)
                q=cycle.finite(recv.getActualQ(),(6,))
                tcp=cycle.finite(recv.getActualTCPPose(),(6,))
                speed=cycle.finite(recv.getActualQd(),(6,))
                cycle.within(tcp[:3]*1000,self.c['tcp_bounds_mm'],'Live joint scan TCP')
                if self.telemetry is not None:
                    self.telemetry_writer.writerow([time.time(),'move_scan',*tcp,recv.getSafetyMode()])
                    self.telemetry.flush()
                if self.cancelled.is_set() or time.monotonic()-begin>timeout:
                    raise RuntimeError('Joint scan cancelled or timed out')
                if (np.max(np.abs(q-self.scan_q))<.005 and np.max(np.abs(speed))<.005
                        and np.linalg.norm(tcp[:3]-np.array(self.scan[:3]))<.001
                        and rot_error(tcp,self.scan)<np.deg2rad(.5)):
                    break
                time.sleep(.02)
            self.wait_stationary(.3)
            self.report['joint_scan_completed']=True
        except BaseException:
            self.cancelled.set()
            self.report['stopJ_requested_on_failure']=True
            ctrl.stopJ(self.SCAN_JOINT_ACCEL)
            raise

    def saved_observation(self,capture,out):
        with redirect_stdout(io.StringIO()):
            rc=preview_tabletop(['--capture',str(capture),'--background',str(self.args.background),
                                '--config',str(self.args.vision_config),'--out',str(out)])
        if rc: raise ValueError('Tabletop perception failed; see saved detection report')
        report=json.loads((out/'summary.json').read_text())
        return select_target(report,self.c,out/'summary.json')

    def detect(self,_):
        from tools import calibrate_ur5_external_camera_real as camera
        from tools.capture_tabletop_background import TABLETOP_WARMUP_FRAMES
        camera_only=getattr(self.args,'camera_check_only',False)
        if camera_only and self.args.execute:
            raise RuntimeError('Camera-only validation cannot execute a cycle')
        a=camera.parse_args(['--check-only','--camera-serial',self.c['camera_serial'],
                            '--warmup-frames',str(TABLETOP_WARMUP_FRAMES),'--settle-frames','5'])
        self.report['capture_policy']=dict(warmup_frames=TABLETOP_WARMUP_FRAMES,
            frames=self.c['frames'],settle_frames_per_capture=5,
            aggregation='none; every raw frame must pass',automatic_retry=False,
            camera_only=camera_only,captures=[])
        pipeline,align,metadata=camera._start_camera(a)
        captures=[]
        try:
            for i in range(self.c['frames']):
                if not camera_only:
                    cycle.stationary_snapshot(self.bridge._iface.recv)
                rgb,depth,stamp,frame=camera._read_aligned_frame(pipeline,align,5)
                self.capture_time=time.monotonic()
                capture=self.args.out/f'capture_{i:02}';capture.mkdir()
                camera._save_rgb(capture/'camera_check_rgb.png',rgb)
                camera._save_depth(capture/'camera_check_depth_u16.png',depth)
                check=dict(schema='rh56_realsense_camera_check/v1',camera=camera._camera_metadata_to_dict(metadata),
                           host_time_s=time.time(),camera_timestamp_ms=stamp,frame_number=frame)
                (capture/'camera_check.yaml').write_text(yaml.safe_dump(check,sort_keys=False))
                captures.append(capture)
                self.report['capture_policy']['captures'].append(dict(directory=str(capture),frame_number=frame,
                    camera_timestamp_ms=stamp,host_time_s=check['host_time_s']))
            if not camera_only:
                cycle.stationary_snapshot(self.bridge._iface.recv)
        finally: pipeline.stop()
        # Save the complete fixed burst before processing, without choosing a
        # good frame or collecting replacements for a rejected observation.
        observations=[]
        for i,capture in enumerate(captures):
            observations.append(self.saved_observation(capture,self.args.out/f'detection_{i:02}'))
        self.report['observation_consistency']=check_consistency(observations,self.c['max_center_spread_mm'])
        self.report['target_validation']=observations[-1][0].source
        return observations[-1]

    def plan(self,observation):
        target,axis,_=observation
        self.plan_data=TabletopPlan(self.c,target,axis)
        self.report['plan']=self.plan_data.report
        cycle.dump(self.args.out/'plan.json',self.plan_data.report)
        return self.plan_data


def run_reset_only(live, report, checkpoint):
    """Explicit open/scan prefix only; never enter perception or grasp stages."""
    for stage in ('prepare','open_hand','move_scan'):
        report['stage']=stage
        report['events'].append(dict(stage=stage,event='start',time_s=time.time()))
        checkpoint()
        print(f'Reset: {stage}',flush=True)
        getattr(live,stage)(None)
        report['events'].append(dict(stage=stage,event='complete',time_s=time.time()))
        checkpoint()
    report['final_snapshot']=cycle.stationary_snapshot(live.bridge._iface.recv)
    report['status']='reset_completed_no_grasp'


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True,help='New output directory')
    p.add_argument('--capture',type=Path,help='Saved capture for offline preview/controller preflight only')
    p.add_argument('--background',type=Path,default=ROOT/'artifacts/tabletop_background/empty_20260909_01')
    p.add_argument('--vision-config',type=Path,default=ROOT/'configs/tabletop_vision_lab.yaml')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true',help='Enter confirmation, then one original Plan grasp/lift cycle')
    mode.add_argument('--preflight-only',action='store_true',help='Entry-only controller IK/FK/safety checks, no motion; pick checks wait until actual scan arrival')
    mode.add_argument('--camera-check-only',action='store_true',help='Fresh three-frame detection only; no robot/hand connection or motion')
    p.add_argument('--reset-only',action='store_true',help='With --execute: low-setting open and checked scan return only, no detection/grasp/lift')
    p.add_argument('--confirm-controller-script',action='store_true',help='Permit RTDE script start/stop for preflight only')
    a=p.parse_args(argv)
    if a.reset_only and not a.execute: p.error('--reset-only requires --execute and Enter confirmation')
    if a.execute and a.capture: p.error('Execution must capture fresh images')
    if a.camera_check_only and a.capture: p.error('Camera check must capture fresh images')
    if not a.execute and not a.camera_check_only and a.capture is None: p.error('Offline/preflight mode needs --capture')
    if a.preflight_only and not a.confirm_controller_script: p.error('Preflight requires --confirm-controller-script')
    return a


def main(argv=None):
    args=parse_args(argv)
    from tools.run_iterative_cube import verify_baseline
    baseline=verify_baseline()
    c=trial_config()
    if args.reset_only:
        c.update(speed_profile='slow',hand_force_raw=100)
    vision,bounds=load_perception_config(args.vision_config)
    if bounds is not None: raise ValueError('Use the independent tabletop perception profile')
    for key in ('camera_serial','calibration','desk_plane','desk_z_mm'):
        if vision[key]!=c[key]: raise ValueError(f'Perception/planner calibration mismatch: {key}')
    args.out.mkdir(parents=True,exist_ok=False)
    args.reset_from_low=True
    args.acknowledge_shape_extrapolation=True
    report=dict(schema='rh56_sugar_box_cycle/v1',status='starting',stage='configuration',config=c,
                events=[],hardware_commands_attempted=False,grasp_success='unverified',
                automatic_release=False,automatic_retry=False,baseline_id=baseline['baseline_id'],
                assumptions=['Known lying 176x90x40 mm sugar box only; not a universal-object picker.',
                             'Operator measured thumb-middle physical gap about 90 mm for internal100, low-setting empty-hand trial.',
                             'Internal98 = nominal90 + measured approximate10 - prior2 mm total closure margin; slope/loaded contact are unverified.',
                             'No cube base-Y placement offset; use reviewed object center and measured short axis.',
                             'Only scan return uses saved-joint MoveJ at 0.10 rad/s, 0.20 rad/s²; 0.01 rad sampled controller FK, not full-arm/hand collision certification.',
                             'Original Plan/GraspExecutor grasp motion, open-loop timing and actual-pose vertical lift.',
                             'Mount correction extrapolated to new shape; controller IK is not collision certification.',
                             'No hand feedback, automatic release or retry; observed pickup required after commands.',
                             'Live camera uses reviewed 60-frame warmup; fixed raw burst, all frames must pass unchanged geometry/consistency gates.',
                             'Speed/force explicitly restored to original lab settings for trial; raw force is not measured contact force.'])
    live=None;rc=0;failed=True
    try:
        for path in [Path(__file__),args.vision_config,ROOT/'tools/preview_tabletop_target.py',
                     ROOT/'tools/capture_tabletop_background.py',args.background/'camera_check_depth_u16.png']:
            report.setdefault('source_sha256',{})[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        if args.reset_only:
            report['reset_only']=True
            report['grasp_success']='not_attempted'
            report['assumptions']=['Reset only: object supported on desk, open-hand and scan-return paths clear.',
                'Hand speed100/force100, write-only; vertical retreat if needed at 0.01 m/s.',
                'Saved-joint scan return 0.10 rad/s; existing preflight checks, not full collision certification.',
                'No camera, detection, grasp, post-grasp lift or automatic retry.']
            print('仅复位：低设置开手（speed100、force100），必要时垂直退离，低速回保存的扫描位；不检测、不抓取。',flush=True)
        elif args.camera_check_only:
            print('仅检查实时视觉：预热60帧、采集3帧、逐帧检查和一致性验证；不连接机械臂或手。',flush=True)
        else:
            print('糖盒单轮：原 Plan；内部98 mm；位置偏移0；顶面下10 mm；抬升50 mm。\n'
                  '回扫描位：MoveJ 0.10 rad/s、0.20 rad/s²。\n'
                  '抓取机械臂0.1 m/s、0.2 m/s²；手speed1000、force1000；无手反馈。',flush=True)
        if args.execute:
            if args.reset_only:
                print('确认物体受桌面支撑，人员离开运动区，开手/必要时原地上抬/关节回扫描位路径畅通。\n'
                      '允许RTDE脚本启停；全程观察并可急停；只执行开手和回位，不继续抓取、不重试。',flush=True)
            else:
                print('请移走尺子，人员离开运动区；确认空手、糖盒受桌面支撑、开手/原地上抬/回扫描位/抓取/抬升路径畅通。\n'
                      '全程观察并可急停。本轮允许RTDE脚本启停、局部安装补偿外推和必要时原地垂直退离；不自动松手或重试。',flush=True)
            if input('确认上述条件，按 Enter 执行一轮；其他输入或 Ctrl-C 取消：').strip():
                report['status']='cancelled_before_hardware';failed=False;return 2
            verify_baseline()
            report['confirmation_method']='interactive_enter'
        live=SugarCycle(c,args,report)
        if args.reset_only:
            run_reset_only(live,report,lambda:cycle.dump(args.out/'summary.json',report))
        elif args.execute:
            cycle.run_cycle(live,report,lambda:cycle.dump(args.out/'summary.json',report))
        elif args.camera_check_only:
            report['stage']='detect'
            live.detect(None)
            report['status']='live_vision_passed_no_motion'
        else:
            observation=live.saved_observation(args.capture,args.out/'detection')
            plan=live.plan(observation)
            if args.preflight_only:
                # prepare performs receive-only origin checks then entry-route IK;
                # it does not open the hand or execute entry movements.
                live.prepare(None)
                # The robot has not moved to scan. Do not pretend a pick
                # preflight from its actual pose validates scan-to-pick.
                report['status']='entry_preflight_passed_no_motion'
                report['pick_preflight_pending_at_scan']=True
            else: report['status']='offline_plan_preview'
        failed=False
    except (Exception,KeyboardInterrupt,EOFError) as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');rc=1
    finally:
        if live is not None: live.close(failed and args.execute)
        if report.get('cleanup_errors'):
            report['status']='failed';rc=1
        cycle.dump(args.out/'summary.json',report)
        with (args.out/'summary.csv').open('w',newline='') as stream:
            keys=['status','stage','hardware_commands_attempted','grasp_success']
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerow({k:report[k] for k in keys})
        print(f'Sugar trial: {report["status"]}; {args.out / "summary.json"}',flush=True)
        if 'error' in report: print(report['error'],flush=True)
    return rc


if __name__=='__main__':
    raise SystemExit(main())
