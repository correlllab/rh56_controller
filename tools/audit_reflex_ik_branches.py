#!/usr/bin/env python3
"""Offline Reflex arm-branch audit; never connects hardware or changes the grasp.

Enumerate existing solver seeds at the frozen endpoint, then trace the same
Cartesian approach backward to Pi to obtain a branch hint. Recheck the FULL
forward sequence, including scan-to-Pi and lift, with the existing predicates.
"""
import argparse
import csv
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_reflex_ten_starts as runner, ggx_planning as planning
from tools import ggx_template as files, ggx_motion as motion, ggx_real_support as support
from tools import preflight_ggx_ur5_table as sim, run_ggx_review as review


def reverse_hint(model, endpoint, point, target, args):
    """Use unchanged moveL interpolation and IK; no endpoint-only shortcut."""
    import math
    q=endpoint.copy()
    start=support.matrix_pose(np.asarray(target['final_tcp_T']))
    end=point['tcp']
    distance,angle=support.pose_error(start,end)
    count=max(2,math.ceil(distance/2)+1,math.ceil(angle)+1)
    for alpha in np.linspace(0,1,count)[1:]:
        solved=files.solve_tcp(model,q,motion.pose_at(start,end,float(alpha)),args)
        solved=planning.nearest_equivalent(model,solved,q)
        if np.max(np.abs(solved[planning.arm_addresses(model)]-q[planning.arm_addresses(model)]))>.15:
            raise ValueError('Reverse approach IK branch discontinuity')
        q=solved
    return q


def wrist_equivalents(model,solutions):
    """Equivalent angle representations within existing limits, not new poses."""
    addresses=planning.arm_addresses(model)
    for branch,q in enumerate(solutions):
        for fourth in (0.,-2*np.pi,2*np.pi):
            for sixth in (0.,-2*np.pi,2*np.pi):
                variant=q.copy();variant[addresses[3]]+=fourth;variant[addresses[5]]+=sixth
                if all(not model.jnt_limited[model.joint(sim.ARM_JOINTS[i]).id] or
                       model.jnt_range[model.joint(sim.ARM_JOINTS[i]).id,0]<=variant[addresses[i]]<=
                       model.jnt_range[model.joint(sim.ARM_JOINTS[i]).id,1] for i in (3,5)):
                    yield branch,variant


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True,help='Saved Reflex offline run containing template and fixed_layout')
    p.add_argument('--point',choices=[f'P{i}' for i in range(1,11)],default='P2')
    p.add_argument('--config',type=Path,default=ROOT/'configs/reflex_real_lab.yaml')
    p.add_argument('--start-snapshot',type=Path,default=ROOT/'artifacts/vision_pick_start/start_pose.json')
    p.add_argument('--video',action='store_true',help='Render endpoint-passing branches, including rejected preparation paths')
    p.add_argument('--branch',type=int,help='Inspect one deterministic audit branch index only (not robot execution)')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=False)
    saved=review.read_json(a.run_dir/'template.json');review.verify(saved['pins'])
    target=saved['target'];runner.validate_target(target)
    layout=review.read_json(a.run_dir/'fixed_layout.json')
    point=next(x for x in layout['points'] if x['id']==a.point)
    config=runner.real.load_config(a.config)
    model=support.build_model(saved['scene']);runner.check_shapes(model,target)
    args=sim.parse_args(['--config',saved['simulation_config'],'--no-video'])
    args.ik_position_tol_mm=args.ik_orientation_tol_deg=.05
    snapshot=review.read_json(a.start_snapshot)
    hand=np.asarray(target['final_hand_T'])
    transform=sim.target_eeff_transform(hand[:3,3],hand[:3,:3])
    seed=sim.initial_qpos(model,np.asarray(snapshot['q']))
    attempts=[];rows=[]
    # Numerical wrist-flip seeds around the first endpoint solution. Never
    # command these guesses; every result still comes from the original IK.
    initial_attempts=[]
    first=next(planning.solutions(model,transform,args,initial_attempts,[seed]),None)
    hints=[seed]
    if first is not None:
        addresses=planning.arm_addresses(model)
        for fourth in (-np.pi,np.pi):
            for sixth in (-np.pi,np.pi):
                guess=first.copy()
                guess[addresses[3]]+=fourth;guess[addresses[4]]*=-1;guess[addresses[5]]+=sixth
                hints.append(guess)
    report=dict(schema='rh56_reflex_ik_branch_audit/v1',hardware_contacted=False,executable=False,
        point=a.point,branches=rows,ik_attempts=attempts,target=target,
        initial_ik_attempts=initial_attempts,seed_hints_rad=[h[planning.arm_addresses(model)].tolist() for h in hints],
        input_sha256=review.pin([a.run_dir/'template.json',a.run_dir/'fixed_layout.json',a.config,a.start_snapshot,__file__]),
        assumptions=['No target/hand command/orientation/desk/threshold changes.',
            'Existing deterministic IK seeds; same straight approach, scan positioning and lift.',
            'Nominal branches only; no controller or physical collision certification.'])
    def save():
        files.write_json(a.out/'summary.json',report)
        with (a.out/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['branch','endpoint_passed','route_passed','minimum_endpoint_desk_mm','error'],extrasaction='ignore')
            writer.writeheader();writer.writerows(rows)
    save()
    for index,(solver_branch,q) in enumerate(wrist_equivalents(model,planning.solutions(model,transform,args,attempts,hints))):
        if a.branch is not None and index!=a.branch:
            continue
        row=dict(branch=index,solver_branch=solver_branch,q_rad=q[planning.arm_addresses(model)].tolist(),endpoint_passed=True,route_passed=False,
                 endpoint_checks=[],minimum_endpoint_desk_mm=float('inf'))
        rows.append(row)
        data=sim.mujoco.MjData(model)
        for label,ctrl in [('thumb',target['thumb_ctrl']),('closed',target['final_ctrl'])]:
            data.qpos[:]=q;sim.set_hand_qpos(model,data.qpos,ctrl);sim.mujoco.mj_forward(model,data)
            gap,nearest=sim.desk_clearance(model,data)
            contact=motion.reflex_target_check(model,data,'approach')
            row['endpoint_checks'].append(dict(hand=label,desk_mm=gap*1000,nearest_desk=nearest,**contact))
            row['minimum_endpoint_desk_mm']=min(row['minimum_endpoint_desk_mm'],gap*1000)
            row['endpoint_passed'] &= gap>=.002 and not contact['forbidden_target_overlap']
        if row['endpoint_passed']:
            try:
                hint=reverse_hint(model,q,point,target,args)
                alternate=deepcopy(point);alternate['q_nominal']=hint[planning.arm_addresses(model)].tolist()
                directory=a.out/f'branch_{index:02d}';directory.mkdir()
                row.update(point=alternate,nominal_report=str(directory/'summary.json'))
                files.write_json(directory/'point.json',alternate)
                record=runner.preview.simulate_point(model,target,alternate,snapshot,config,args,directory,a.video)
                runner.nominal_gate(record)
                row['route_passed']=True
            except (ValueError,motion.PathCollision,planning.PlanningUnavailable) as exc:
                row['error']=f'{type(exc).__name__}: {exc}'
        print(f"branch {index}: endpoint={row['endpoint_passed']}, desk={row['minimum_endpoint_desk_mm']:.2f} mm, route={row['route_passed']}",flush=True)
        save()
    report['status']='nominal_alternative_found' if any(r['route_passed'] for r in rows) else 'no_checked_alternative_found'
    save();return 0 if report['status']=='nominal_alternative_found' else 2


if __name__=='__main__':
    raise SystemExit(main())
