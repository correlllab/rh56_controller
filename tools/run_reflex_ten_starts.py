#!/usr/bin/env python3
"""Frozen three/four-finger Reflex across object-relative P1..P10.

Offline by default. --execute requires --capture-live, reset/placement Enter,
controller-only permission, and each actual execution video's Enter. Never
reuses a GGX closure or a saved video as motion permission. No hand sensor reads.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import prepare_reflex_comparison as preview
from tools import run_ggx_real as real, run_ggx_review as review
from tools import ggx_template as files, ggx_xy_tracking as xy, ggx_fixed_starts as fixed
from tools import ggx_motion as motion, ggx_real_support as support
from tools import preflight_ggx_ur5_table as sim, ggx_planning as planning
from tools import run_ggx_experiment as experiment, run_ggx_ten_starts as ten


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--crop-report',type=Path,help='Freeze one original Reflex target from an existing reviewed foreground crop')
    source.add_argument('--template',type=Path,help='Resume SAME object/template with fresh live XY; never change object')
    source.add_argument('--resume-from',type=Path,help='Continue an ended summary from its first unexecuted point; preserve completed results/template')
    p.add_argument('--config',type=Path,default=ROOT/'configs/reflex_real_lab.yaml')
    p.add_argument('--perception-config',type=Path,default=ROOT/'configs/tabletop_vision_lab.yaml')
    p.add_argument('--layout-config',type=Path,default=ROOT/'configs/reflex_ten_starts_object_relative_lab.yaml')
    p.add_argument('--start-snapshot',type=Path,default=ROOT/'configs/reflex_scan_wrist_flip.json')
    p.add_argument('--object-label',required=True)
    p.add_argument('--fingers',type=int,choices=[2,3,4],default=4,
        help='2=thumb/index; 3 adds middle; 4 also includes ring (default); inactive fingers stay open')
    selection=p.add_mutually_exclusive_group()
    selection.add_argument('--point',choices=['all']+[f'P{i}' for i in range(1,11)],default='all')
    selection.add_argument('--points',nargs='+',choices=[f'P{i}' for i in range(1,11)],
        help='Run only this ordered subset; does not replay omitted points or merge their old outcomes')
    p.add_argument('--capture-live',action='store_true',help='Camera-only XY measurement each trial; robot motion still requires --execute')
    p.add_argument('--trial-captures',type=Path,nargs='+',help='Offline XY replay: one capture per selected start')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--allow-high-force',action='store_true')
    p.add_argument('--video',action='store_true',help='Offline videos; live execution video is mandatory')
    p.add_argument('--open',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--hold-seconds',type=float,default=3.,help='Commanded lift hold before human scoring, not a grasp sensor')
    p.add_argument('--grasp-depth-mm',type=float,default=35.,
        help='Grasp center below observed top (default: 35 mm); must match a reused frozen template')
    p.add_argument('--round-object-yaw-deg',type=float,
        help='Explicit round-object axis, frozen across all starts; 90 aligns lab +Y/P1. Must match reused template.')
    p.add_argument('--round-object-approximation',action='store_true',
        help='Explicit non-directional object approximation despite a measured footprint axis; requires round-object yaw and must match reused template.')
    p.add_argument('--xy-search-px',type=int,default=120,
        help='RGB translation search radius (default 120 px); confidence and 40 mm metric bound unchanged')
    p.add_argument('--out',type=Path)
    args=p.parse_args(argv)
    if args.points and (len(set(args.points))!=len(args.points)
            or args.points!=sorted(args.points,key=lambda value:int(value[1:]))):
        p.error('--points must be unique and in P1 to P10 order')
    if args.round_object_approximation and args.round_object_yaw_deg is None:
        p.error('--round-object-approximation requires --round-object-yaw-deg')
    import math
    if args.execute and not args.capture_live:
        p.error('--execute requires --capture-live; no saved-XY execution')
    if args.capture_live and args.trial_captures:
        p.error('Live capture cannot be replaced by offline trial captures')
    if args.trial_captures and len(args.trial_captures)!=len(selected_point_ids(args)):
        p.error('Need one offline capture for each selected point')
    if not math.isfinite(args.hold_seconds) or not 1<=args.hold_seconds<=10:
        p.error('hold-seconds must be finite and in [1,10]')
    if not math.isfinite(args.grasp_depth_mm) or args.grasp_depth_mm<=0:
        p.error('grasp-depth-mm must be finite and positive')
    if args.round_object_yaw_deg is not None and (not math.isfinite(args.round_object_yaw_deg) or not -180<=args.round_object_yaw_deg<=180):
        p.error('round-object-yaw-deg must be finite and in [-180,180]')
    if not 5<=args.xy_search_px<=150:
        p.error('xy-search-px must be within [5,150]')
    if args.resume_from and (args.point!='all' or args.points or args.trial_captures):
        p.error('--resume-from requires --point all, no --points and no per-point offline capture list')
    for k,v in vars(args).items():
        if isinstance(v,Path): setattr(args,k,v.resolve())
    return args


def selected_point_ids(args):
    """Explicit subset preserves original point identities, never renumbers them."""
    return args.points if args.points is not None else (
        [f'P{i}' for i in range(1,11)] if args.point=='all' else [args.point])


def validate_target(target):
    fingers=target.get('fingers')
    modes={2:'GraspMode.LINE_2F',3:'GraspMode.PLANE_3F',4:'GraspMode.PLANE_4F'}
    if (target.get('schema')!='rh56_shared_cloud_reflex_target/v1'
            or fingers not in modes or target.get('mode')!=modes[fingers]
            or target.get('grasp_direction')!='legacy-top'
            or target.get('side_pregrasp_tcp_T') is not None
            or target.get('force_N')!=0 or target.get('hand_write_only') is not True
            or target.get('phase_order')!=['thumb_position','blocking_moveL_to_final_pose','remaining_fingers_close']):
        raise ValueError('Only original two/three/four-finger position-control Reflex is accepted')
    for k in ('thumb_raw','final_raw'):
        values=target[k]
        if len(values)!=6 or any(type(v) is not int or not 0<=v<=1000 for v in values):
            raise ValueError('Invalid frozen hand command')
    if target['thumb_raw'][:4]!=[1000]*4 or target['thumb_raw'][4:]!=target['final_raw'][4:]:
        raise ValueError('Frozen thumb command does not implement native Reflex')
    inactive=('pinky','ring','middle')[:5-fingers]
    if target['final_raw'][:len(inactive)]!=[1000]*len(inactive) or any(
            target['final_ctrl'][k]!=0. for k in inactive):
        raise ValueError('Inactive Reflex fingers must remain open')
    expected=set(('thumb','index','middle','ring')[:fingers])
    if set(target['active_tip_positions_base_m'])!=expected:
        raise ValueError('Frozen target active fingertips differ from selected finger mode')


def freeze(args,out):
    if args.template:
        data=review.read_json(args.template)
        path=args.template
    else:
        geometry=out/'frozen_geometry'
        code=preview.main(['--crop-report',str(args.crop_report),'--fingers',str(args.fingers),
            '--grasp-depth-mm',str(args.grasp_depth_mm),
            '--grasp-direction','legacy-top','--target-only','--config',str(args.config),
            '--layout-config',str(args.layout_config),'--start-snapshot',str(args.start_snapshot),'--out',str(geometry),
            *([] if args.round_object_yaw_deg is None else ['--round-object-yaw-deg',str(args.round_object_yaw_deg)]),
            *(['--round-object-approximation'] if args.round_object_approximation else [])])
        if code: raise ValueError('Could not freeze original Reflex geometry')
        saved=review.read_json(geometry/'summary.json')
        crop=review.read_json(args.crop_report)
        perception=review.read_json(crop['source_preview'])
        data=dict(schema='rh56_reflex_ten_template/v1',target=saved['target'],scene=saved['scene'],
            object_label=args.object_label,perception=perception,crop=crop,
            pins={**saved['input_sha256'],**review.pin([geometry/'summary.json',args.crop_report])},
            simulation_config=str(geometry/'simulation.yaml'),lift_mm=saved['lift_mm'],
            executable=False,closure_feedback=False)
        path=out/'template.json'
        files.write_json(path,data)
    if data.get('schema')!='rh56_reflex_ten_template/v1' or data['object_label']!=args.object_label:
        raise ValueError('Wrong template type/object label; a new object needs a new crop/template')
    validate_target(data['target'])
    if data['target'].get('grasp_depth_mm') != args.grasp_depth_mm:
        raise ValueError('Frozen template grasp depth differs; use a new crop-based run to change depth')
    if data['target']['fingers']!=args.fingers:
        raise ValueError('Frozen template finger mode differs; generate a new crop-based template')
    if data['target'].get('round_object_yaw_deg')!=args.round_object_yaw_deg:
        raise ValueError('Frozen round-object yaw differs; regenerate from crop to change orientation')
    if data['target'].get('round_object_approximation',False)!=args.round_object_approximation:
        raise ValueError('Frozen round-object approximation differs; regenerate from crop')
    if Path(data['perception']['config']).resolve()!=args.perception_config:
        raise ValueError('Template perception config differs')
    review.verify(data['pins'])
    return data,path


def relocate(data,measurement):
    import numpy as np
    source=dict(perception=data['perception'],crop=data['crop'],pins=dict(data['pins']),**deepcopy(data['scene']))
    if measurement.get('status')!='localized': raise ValueError('XY localization failed')
    delta=np.asarray(measurement['object_xy_mm'])/1000-np.asarray(source['preflight']['object_center_m'][:2])
    if delta.shape!=(2,) or not np.all(np.isfinite(delta)): raise ValueError('Invalid XY translation')
    source['preflight']['object_center_m'][:2]=list(np.asarray(source['preflight']['object_center_m'][:2])+delta)
    target=deepcopy(data['target'])
    target['grasp_center_base_mm'][:2]=list(np.asarray(target['grasp_center_base_mm'][:2])+delta*1000)
    for key in ('final_hand_T','final_tcp_T'):
        t=np.asarray(target[key]);t[:2,3]+=delta;target[key]=t.tolist()
    for pos in target['active_tip_positions_base_m'].values():
        pos[:2]=list(np.asarray(pos[:2])+delta)
    return source,target,delta.tolist()


def nominal_gate(record):
    if record['status']!='nominal_sequence_solved_contact_unverified':
        raise planning.PlanningUnavailable('reflex_nominal_ik',[])
    for row in record['phase_metrics']:
        if row['desk_reject_samples']:
            raise motion.PathCollision('Reflex nominal desk clearance failed',phase=row['phase'],obstacle='desk')
        if row.get('target_contact_policy') != motion.REFLEX_CONTACT_POLICY:
            raise ValueError('Reflex nominal contact classification missing; regenerate checks')
        if row['forbidden_target_overlap_samples']:
            raise motion.PathCollision('Reflex preparation or nonfinger target collision',phase=row['phase'],obstacle='object')


def check_shapes(model,target):
    """Check that rendered controls encode the exact hardware commands, no clipping."""
    from rh56_controller.grasp_executor import GraspExecutor
    import numpy as np
    ranges={k:model.actuator(k).ctrlrange for k in sim.GGX_OPEN_CTRL}
    fk=SimpleNamespace(ctrl_min={k:float(v[0]) for k,v in ranges.items()},ctrl_max={k:float(v[1]) for k,v in ranges.items()})
    executor=GraspExecutor(None,None,fk,hand_write_only=True)
    for ck,rk in [('thumb_ctrl','thumb_raw'),('final_ctrl','final_raw')]:
        ctrl=target[ck]
        if set(ctrl)!=set(ranges) or any(not np.isfinite(ctrl[k]) or not v[0]<=ctrl[k]<=v[1] for k,v in ranges.items()):
            raise ValueError('Frozen Reflex controls outside reviewed actuator ranges')
        if executor._ctrl_to_real(ctrl)!=target[rk]:
            raise ValueError('Planner and execution model raw hand mappings differ')


def checked_samples(model,target,commands):
    """Controller-resolved arm paths plus both stationary hand transitions."""
    import numpy as np
    expected=['position_to_trial_start','approach','lift']
    if [c['phase'] for c in commands]!=expected: raise ValueError('Missing Reflex arm phase')
    controls=[target['thumb_ctrl'],target['thumb_ctrl'],target['final_ctrl']]
    samples=[]
    def transition(phase,q,before,after):
        for a in np.linspace(0,1,16):
            pose=sim.initial_qpos(model,np.asarray(q))
            sim.set_hand_qpos(model,pose,{k:(1-a)*before[k]+a*after[k] for k in before})
            samples.append(sim.TrajectorySample(phase,float(a),pose,float(a)))
    for i,command in enumerate(commands):
        if command.get('hand_ctrl')!=controls[i]: raise ValueError('Checked phase uses wrong hand shape')
        if i==0:
            transition('thumb_position',command['start_q'],sim.GGX_OPEN_CTRL,target['thumb_ctrl'])
        elif i==2:
            transition('remaining_fingers_close',commands[i-1]['q'],controls[i-1],controls[i])
        for row in command['samples']:
            pose=sim.initial_qpos(model,np.asarray(row['q']))
            sim.set_hand_qpos(model,pose,controls[i])
            samples.append(sim.TrajectorySample(command['phase'],row['fraction'],pose,float(i==2)))
    return samples


def review_checked(out,model,target,commands,sim_args,config=None):
    import numpy as np
    samples=checked_samples(model,target,commands)
    required_clearance=motion.reflex_desk_clearance_m(config or {})
    sim.check_trajectory(model,samples,required_clearance_m=required_clearance)
    data=sim.mujoco.MjData(model)
    failures=[];contacts=[]
    for sample in samples:
        data.qpos[:]=sample.qpos;sim.mujoco.mj_forward(model,data)
        contact=motion.reflex_target_check(model,data,sample.phase)
        contacts.append(dict(phase=sample.phase,fraction=sample.phase_alpha,**contact))
        gap,nearest=contact['minimum_target_gap_mm']/1000,contact['nearest_target_geom']
        sample.object_clearance_m=float(gap);sample.object_collision=gap<=1e-9
        sample.nearest_object_geom=nearest
        sample.desk_collision=motion.desk_clearance_blocked(sample.min_desk_clearance_m,required_clearance)
        if sample.desk_collision: failures.append((sample.phase,'desk'))
        if contact['forbidden_target_overlap']: failures.append((sample.phase,'object'))
    result=dict(schema='rh56_reflex_checked_path/v1',passed=not failures,samples=len(samples),
        required_desk_clearance_mm=required_clearance*1000,
        failures=failures,hand_transition_samples=16,target_contacts=contacts,
        assumptions=['Original hand coupling model; not physical hand calibration.',
            'Reflex approach/closure/lift finger-target contacts recorded, not hard rejections or success evidence.',
            'Preparation and all nonfinger target pairs remain checked; desk margin comes from the pinned run profile.',
            'Desk checked throughout; unknown fixtures/camera mount and full self-collision not certified.'])
    files.write_json(out/'execution_precheck.json',result)
    if failures:
        phase,obstacle=failures[0]
        raise motion.PathCollision('Reflex controller-resolved geometry failed',phase=phase,obstacle=obstacle)
    path=out/'execution_review.mp4'
    preview.render_reflex(path,model,samples,sim_args,np.asarray(target['grasp_center_base_mm'])/1000)
    return path


def execute_sequence(backend,target,commands,hold_seconds,open_raw):
    """Native thumb -> arm -> fingers ordering; never interpolated GGX closure."""
    if [c['phase'] for c in commands]!=['position_to_trial_start','approach','lift']:
        raise ValueError('Expected checked positioning, native Reflex approach, lift')
    for command,ctrl in zip(commands,[target['thumb_ctrl'],target['thumb_ctrl'],target['final_ctrl']]):
        if command.get('revision')!=motion.REVISION or command.get('hand_ctrl')!=ctrl:
            raise ValueError('Execution hand shape differs from controller preflight')
    backend.open_hand(open_raw)
    backend.set_stage('thumb_position');backend.hand_angles(target['thumb_raw'])
    backend.dwell(.2)
    backend.move_phase(commands[0])
    backend.move_phase(commands[1])
    backend.set_stage('remaining_fingers_close')
    if hasattr(backend,'report'):
        backend.report['grasp_close_command_attempted']=True;backend.checkpoint()
    backend.hand_angles(target['final_raw'])
    backend.dwell(.25)
    backend.move_phase(commands[2])
    backend.set_stage('lift_hold');backend.dwell(hold_seconds)
    if hasattr(backend,'report'):
        backend.report['lift_hold_completed']=True;backend.checkpoint()


def run_trial(args,config,data,target,source,point,scan,out,trial,save,*,backend_factory=real.LiveBackend):
    model=support.build_model(source)
    check_shapes(model,target)
    sim_args=sim.parse_args(['--config',data['simulation_config'],'--no-video'])
    sim_args.ik_position_tol_mm=sim_args.ik_orientation_tol_deg=.05
    backend=None
    try:
        if args.execute:
            backend=backend_factory(config,out,trial,save);backend.open_receiver()
            snapshot=backend.snapshot()
            real.require_at_execution_target(review.read_json(scan),snapshot)
        else: snapshot=review.read_json(scan)
        nominal=out/'nominal';nominal.mkdir()
        record=preview.simulate_point(model,target,point,snapshot,config,sim_args,nominal,args.video and not args.execute)
        trial['nominal']=str(nominal/'summary.json');save()
        nominal_gate(record)
        goals=record['nominal_goals']
        if not args.execute:
            trial['status']='offline_path_passed_not_executed';return
        if not real.confirm('Reflex：允许控制脚本启停和控制器预检；此确认不授权运动。'):
            raise KeyboardInterrupt
        review.verify(trial['pins']);real.require_same_start(snapshot,backend.snapshot())
        backend.open_control();real.check_tcp_offset(backend.control)
        commands=motion.preflight_goals(backend,snapshot,goals,model,config,trial)
        path=review_checked(out,model,target,commands,sim_args,config)
        files.write_json(out/'execution.json',dict(commands=commands,target=target))
        execution_pins=review.pin([path,out/'execution.json',out/'execution_precheck.json'])
        trial['execution_pins']=execution_pins;save()
        review.show(path,args.open)
        print(f"Reflex {point['id']}: force={config['hand_force_raw']}, speed={config['hand_speed_raw']}; "
              f"arm={config['joint_speed_rad_s']} rad/s, {config['cartesian_speed_m_s']} m/s; "
              f"desk margin={motion.reflex_desk_clearance_m(config)*1000:g} mm; "
              f"desk surface={config.get('reflex_desk_surface','conservative-horizontal')}; "
              f"lift={data['lift_mm']} mm, hold={args.hold_seconds:g}s。\n"
              '确认物体自本轮拍摄后未动、位置/朝向正确、物体有支撑、张手与全程路径畅通，人员离开运动区，可立即急停。\n'
              f"{target['fingers']} 指位置控制，无触觉反馈；完成后保持，不自动松手。",flush=True)
        if not real.confirm('批准本轮视频并执行？'): raise KeyboardInterrupt
        trial['motion_confirmation_received']=True;save()
        review.verify(trial['pins']);review.verify(execution_pins)
        real.check_tcp_offset(backend.control)
        backend.held_origin=backend.snapshot();real.require_same_start(snapshot,backend.held_origin)
        trial['operator_outcome']='unverified';save()
        execute_sequence(backend,target,commands,args.hold_seconds,support.raw_hand_command(model,0.))
        trial['final_snapshot']=backend.snapshot()
        trial['status']='commands_completed_grasp_unverified'
    finally:
        if backend is not None:
            backend.close()
            if trial.get('cleanup_errors'): raise RuntimeError('Hardware cleanup failed; stop experiment')
        save()


def score_trial(trial):
    print('完成抬升并保持后：成功=物体完整离桌且未滑落；不确定或人工中断不可自动算成功。')
    while True:
        answer=input('记录结果：s 成功；f 失败；Enter 未确认；q 记录未确认并结束：').strip().lower()
        if answer in ('s','f','','q'): break
    trial['operator_outcome']={'s':'success','f':'fail','':'unverified','q':'unverified'}[answer]
    return answer!='q'


def resume_state(args):
    """Carry terminal outcomes forward; never replay a partially executed trial."""
    if not args.resume_from: return [],None
    old=review.read_json(args.resume_from)
    if old.get('selected_points',[f'P{i}' for i in range(1,11)])!=[f'P{i}' for i in range(1,11)]:
        raise ValueError('Subset run cannot use contiguous --resume-from; review its outcomes and use --template with explicit remaining --points')
    if (old.get('schema')!='rh56_reflex_ten_experiment/v1'
            or old.get('status') not in ('failed','stopped','interrupted','operator_stopped','completed','all_points_skipped')):
        raise ValueError('Resume requires an ended Reflex experiment report')
    if old.get('hold_seconds')!=args.hold_seconds or old.get('grasp_depth_mm')!=args.grasp_depth_mm:
        raise ValueError('Resume must keep the frozen depth and hold protocol')
    if old.get('fingers',4)!=args.fingers:
        raise ValueError('Resume must keep the original finger mode')
    if old.get('round_object_yaw_deg')!=args.round_object_yaw_deg:
        raise ValueError('Resume must keep the frozen round-object yaw')
    if old.get('round_object_approximation',False)!=args.round_object_approximation:
        raise ValueError('Resume must keep the frozen round-object approximation')
    for path in (args.config,args.layout_config,args.perception_config):
        expected=old.get('pins',{}).get(str(path))
        if expected is None or review.pin([path])[str(path)]!=expected:
            raise ValueError('Resume requires unchanged motion, layout and perception configs')
    records=old.get('prior_trials',[])+old['trials']
    completed=[]
    fields=('point_id','object_label','status','operator_outcome','motion_commands_attempted',
            'grasp_close_command_attempted','lift_hold_completed','collision_phase','obstacle','planning_reason')
    for i,row in enumerate(records):
        if row['point_id']!=f'P{i+1}': raise ValueError('Resume requires a contiguous P1..P10 history')
        if row['status'] in ('commands_completed_grasp_unverified','skipped_path'):
            completed.append(dict({k:row[k] for k in fields if k in row},carried_forward=True,
                source_report=row.get('source_report',str(args.resume_from))))
        else:
            if (i!=len(records)-1 or row.get('motion_commands_attempted') or row.get('hand_commands')
                    or row.get('motion_confirmation_received') or row.get('cleanup_errors')):
                raise ValueError('Partially executed/ambiguous trial needs manual review; will not automatically retry')
            break
    if len(completed)==10: raise ValueError('All ten points already have terminal outcomes')
    args.template=Path(old['template']).resolve()
    # Template/data hashes are checked by freeze; original report is also pinned.
    return completed,dict(summary=str(args.resume_from),input_sha256=review.pin([args.resume_from]),
                          next_point=f'P{len(completed)+1}',template=old['template'])


def run(args,out,report,checkpoint,*,reset_fn=experiment.reset_robot,capture_fn=experiment.capture_xy,
        locate_fn=xy.locate,trial_fn=run_trial,score_fn=score_trial):
    config=real.load_config(args.config)
    if args.execute: real.require_hand_setting_authorization(config,args.allow_high_force)
    prior,resume=resume_state(args)
    data,template_path=freeze(args,out)
    layout=fixed.build(args.layout_config,config,args.perception_config,
                       grasp_center_mm=data['target']['grasp_center_base_mm'])
    files.write_json(out/'fixed_layout.json',layout)
    pins={**data['pins'],**ten.code_pins(args.config),**layout['source_sha256'],**review.pin([
        __file__,preview.__file__,xy.__file__,fixed.__file__,experiment.__file__,template_path,out/'fixed_layout.json'])}
    report.update(template=str(template_path),pins=pins,trials=[],lift_mm=data['lift_mm'],hold_seconds=args.hold_seconds,
                  required_desk_clearance_mm=motion.reflex_desk_clearance_m(config)*1000,
                  desk_surface_policy=config.get('reflex_desk_surface','conservative-horizontal'))
    if resume:
        pins.update(resume['input_sha256'])
        report.update(prior_trials=prior,resume=resume)
        print(f"续跑 {resume['next_point']}–P10；已保留 {len(prior)} 个已结束点，不重跑。",flush=True)
    scan=args.start_snapshot
    selected=selected_point_ids(args)
    points=[p for p in layout['points'] if p['id'] in selected]
    if resume: points=points[len(prior):]
    need_reset=args.execute;measurement=None
    reference=dict(perception=data['perception'],crop=data['crop'],pins=data['pins'],**data['scene'])
    for index,point in enumerate(points):
        folder=out/point['id'];folder.mkdir()
        trial=dict(point_id=point['id'],object_label=args.object_label,status='starting',
            motion_commands_attempted=False,motion_confirmation_received=False,operator_outcome='not_executed',pins=dict(pins))
        report['trials'].append(trial)
        def save(): files.write_json(folder/'summary.json',trial);checkpoint()
        save()
        try:
            review.verify(pins)
            if need_reset:
                code,scan=reset_fn(args,folder/'reset')
                trial['reset_report']=str(folder/'reset/summary.json')
                if code: trial['status']='reset_stopped';report['status']='stopped';return code
                need_reset=False;measurement=None
            if measurement is None or args.trial_captures:
                capture=(capture_fn(args,reference,folder) if args.capture_live else
                         args.trial_captures[index] if args.trial_captures else Path(data['perception']['capture']))
                measurement=locate_fn(reference,capture,folder/'xy',search_px=args.xy_search_px)
                measurement['input_sha256'].update(review.pin([folder/'xy/summary.json']))
            source,target,delta=relocate(data,measurement)
            if layout['schema']=='rh56_object_relative_starts/v1':
                point=fixed.translate_point(point,delta)
            trial['pins'].update(measurement['input_sha256'])
            trial.update(xy=measurement,delta_xy_m=delta,point=point,z_yaw_frozen=True)
            files.write_json(folder/'target.json',target);save()
            trial_fn(args,config,data,target,source,point,scan,folder,trial,save)
            if args.execute:
                need_reset=True
                if not score_fn(trial): report['status']='operator_stopped';return 0
        except (motion.PathCollision,planning.PlanningUnavailable) as exc:
            if trial['motion_commands_attempted'] or trial.get('hand_commands') or trial.get('cleanup_errors'):
                trial.update(status='failed_after_device_commands',error=str(exc));raise
            trial.update(status='skipped_path',error=str(exc),collision_phase=getattr(exc,'phase',None),
                         obstacle=getattr(exc,'obstacle',None),planning_reason=getattr(exc,'reason',None))
            print(f"{point['id']}: 未执行，路径拒绝：{exc}",flush=True)
        except (KeyboardInterrupt,EOFError):
            trial['status']='interrupted';report['status']='interrupted';return 130
        except Exception as exc:
            trial.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
        finally: save()
    skipped=sum(t['status']=='skipped_path' for t in report['trials'])
    report['status']='all_points_skipped' if skipped==len(points) else 'completed'
    return 2 if skipped==len(points) else 0


def main(argv=None):
    args=parse_args(argv)
    out=args.out or ROOT/'artifacts/reflex_ten_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='rh56_reflex_ten_experiment/v1',status='starting',trials=[],
        execute_requested=args.execute,camera_requested=args.capture_live,grasp_depth_mm=args.grasp_depth_mm,
        xy_search_px=args.xy_search_px,fingers=args.fingers,round_object_yaw_deg=args.round_object_yaw_deg,
        round_object_approximation=args.round_object_approximation,
        selected_points=selected_point_ids(args),
        assumptions=[f'Original {"line2" if args.fingers==2 else "plane"+str(args.fingers)} Reflex, not GGX closure; no force feedback.',
            'One frozen template; object-relative P1..P10; target and starts share XY relocation. New object needs new crop/template.',
            'Offline default never opens devices; optional camera-only capture is explicit.',
            'Pre-motion path/IK rejection skips; robot/communication/localization faults stop.',
            'Native thumb wait .2s, post-closure wait .25s; not measured hand settling feedback.',
            'Prepare thumb at scan pose BEFORE positioning to Pi; preparation hand shape is used in every path check.',
            'Reset is existing confirmed open/retreat/return, with its known opening-sweep/fixture limitations.',
            'Reflex approach/closure/lift finger-target contacts logged, not rejected; preparation/arm/palm checks retained.',
            'No controller or model change can guarantee clearance from unmodeled camera mount/fixtures.',
            'Hold interval is explicit; prior GGX trials are not retroactively claimed to share it.',
            'Operator scores success; completed commands alone are not successful grasp evidence.'])
    def checkpoint():
        trials=report.get('prior_trials',[])+report['trials']
        report['counts']=dict(planned=len(selected_point_ids(args)),visited=len(trials),
            motion_attempted=sum(bool(t.get('motion_commands_attempted')) for t in trials),
            grasp_close_attempted=sum(bool(t.get('grasp_close_command_attempted')) for t in trials),
            lift_hold_completed=sum(bool(t.get('lift_hold_completed')) for t in trials),
            path_rejected=sum(t['status']=='skipped_path' for t in trials),
            operator_success=sum(t.get('operator_outcome')=='success' for t in trials))
        files.write_json(out/'summary.json',report)
        with (out/'summary.csv').open('w',newline='') as stream:
            fields=['object_label','point_id','status','collision_phase','obstacle','planning_reason',
                    'motion_commands_attempted','grasp_close_command_attempted','lift_hold_completed','operator_outcome','error',
                    'carried_forward','source_report']
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(trials)
    checkpoint()
    try: return run(args,out,report,checkpoint)
    except (KeyboardInterrupt,EOFError): report['status']='interrupted';return 130
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        print(report['error'],file=sys.stderr);return 1
    finally:
        checkpoint();print(f"Reflex 十轮：{report['status']}；{out/'summary.json'}",flush=True)


if __name__=='__main__': raise SystemExit(main())
