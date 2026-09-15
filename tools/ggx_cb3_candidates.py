"""Official endpoint filter + existing CB3 path checks, with separate attribution.

No new collision solver. Unknown target-proxy contacts are NOT automatic passes
or measured grasp failures. Legacy visual approvals retain their original gates.
"""
from dataclasses import asdict
import csv
from pathlib import Path
import time
import numpy as np

SCHEMA='rh56_ggx_cb3_scene_preflight/v1'


def load_filter(path, candidate_path):
    from tools.run_ggx_review import read_json,verify,digest
    from tools.run_ggx_official_filter import SCHEMA as FILTER_SCHEMA
    data=read_json(path)
    if (data.get('schema')!=FILTER_SCHEMA or data.get('status')!='completed'
            or data.get('executable') is not False or data.get('hardware_contacted') is not False
            or data.get('candidate_sha256')!=digest(candidate_path)
            or Path(data['candidate_file']).resolve()!=candidate_path.resolve()):
        raise ValueError('Missing/mismatched official scene filter')
    verify(data['source_sha256'])
    rows=data['candidates']
    if [r['rank'] for r in rows]!=list(range(len(rows))) or not rows:
        raise ValueError('Official filter changed original rank order')
    if any(type(r.get('official_scene_clear')) is not bool for r in rows):
        raise ValueError('Missing official scene-filter decision')
    return data


def classify(result, samples, bounds_ok):
    """Keep numerical rejection separate from physical failure and uncertainty."""
    if result.reason.startswith('ik_failed'):
        return 'rejected_ik', result.reason
    if not bounds_ok:
        return 'rejected_workspace', 'sampled_tcp_outside_workspace'
    if result.first_collision_phase:
        penetration=any(s.min_desk_clearance_m < -1e-9 for s in samples)
        return 'rejected_desk', 'desk_model_penetration' if penetration else 'desk_clearance_below_2mm'
    if any(s.object_collision for s in samples):
        return 'uncertain_target_proxy', 'target_proxy_contact_not_physical_failure'
    if not result.passed:
        return 'rejected_local_check', result.reason
    return 'passed_current_checks', 'pass'


def simulate(args,out,report,config_path,source_pins,*,filter_path=None):
    if getattr(args, 'candidate_scope', 'grasp') == 'grasp':
        yield from geometry_candidates(args,out,report,config_path,source_pins,filter_path=filter_path)
        return
    from tools import run_ggx_review as ui, run_ggx_official_filter as official
    from tools import ggx_real_support as support, preflight_ggx_ur5_table as sim
    from tools import ggx_reset as reset, run_ggx_real as real, run_ggx_ten_starts as ten
    from tools.preview_tabletop_target import load_perception_config
    ui.verify(source_pins)
    if filter_path is None:
        preparation=out/'scene_input';preparation.mkdir()
        scene_input=official.prepare_scene(getattr(args,'inference_report',out/'inference/summary.json'),preparation)
        ui.run_stage(out,'official_scene_filter','run_ggx_official_filter.py',
            ['--scene-input',scene_input,'--threshold-mm',args.scene_collision_mm,
             '--out',out/'official_scene_filter'],python=args.ggx_python)
        filter_path=out/'official_scene_filter/summary.json'
    sim_args=sim.parse_args(['--config',str(config_path),'--no-video','--no-rejected-video'])
    sim_args.desk_clearance_mm=2.  # Same fixed local threshold as continuous CB3 preflight.
    filtered=load_filter(filter_path,sim_args.graspgenx_yaml)
    if filtered['threshold_mm'] != args.scene_collision_mm:
        raise ValueError('Saved official threshold differs from declared protocol')
    config=real.load_config(args.real_config)
    if not config.get('desk_envelope',{}).get('confirmed'):
        raise ValueError('Official/CB3 protocol requires the confirmed fixed desk envelope')
    perception,_=load_perception_config(args.config)
    fixed=reset.desk_source(config,perception)
    center=np.array([sim_args.object_x_m,sim_args.object_y_m,
        sim_args.table_height_m+sim_args.object_size_mm[2]/2000])
    source=dict(preflight=dict(object_center_m=center.tolist(),object_size_mm=sim_args.object_size_mm,
        object_yaw_deg=sim_args.object_yaw_deg,table_height_m=sim_args.table_height_m),
        desk_envelope=config['desk_envelope'],desk_edge_y_m=None,
        desk_top_m=fixed['preflight']['table_height_m'])
    model=support.build_model(source)
    start=sim.initial_qpos(model,np.radians(sim_args.start_q_deg))
    candidates=sim.load_isaac_grasp_yaml(sim_args.graspgenx_yaml)
    if len(candidates)!=len(filtered['candidates']):
        raise ValueError('Official filter candidate count mismatch')
    pins={**source_pins,**filtered['source_sha256'],**ten.code_pins(args.real_config),
          **ui.pin([__file__,filter_path,config_path,args.config,perception['desk_plane'],perception['calibration']])}
    report.update(source_sha256=pins,official_scene_filter=str(filter_path),
        feasibility_protocol='official_scene_plus_cb3/v1',
        feasibility_stages=[],simulation=dict(mode='cb3_fixed_desk',model_loads=1,stages=[]))
    def checkpoint():
        states=[r.get('local_state','running') for r in report['feasibility_stages']]
        report['feasibility_counts']=dict(candidates=len(states),
            official_rejected=sum(not r['official_scene_clear'] for r in report['feasibility_stages']),
            current_checks_passed=states.count('passed_current_checks'),uncertain_target_proxy=states.count('uncertain_target_proxy'),
            local_rejected=sum(s.startswith('rejected_') for s in states))
        ui.save_report(out,report)
        with (out/'feasibility.csv').open('w',newline='') as stream:
            fields=['rank','confidence','official_scene_clear','local_state','local_reason',
                    'target_proxy_status','known_desk_clearance_mm','proxy_gap_mm','report']
            w=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
            w.writeheader();w.writerows(report['feasibility_stages'])
    checkpoint()
    for rank,candidate in enumerate(candidates[:args.max_candidates]):
        if getattr(args,'candidate_rank',None) is not None and rank!=args.candidate_rank:
            continue
        ui.verify(pins)
        official_row=filtered['candidates'][rank]
        row=dict(rank=rank,confidence=candidate.confidence,official_scene_clear=official_row['official_scene_clear'])
        report['feasibility_stages'].append(row)
        if not official_row['official_scene_clear']:
            row.update(local_state='not_evaluated',local_reason='rejected_by_official_scene_filter')
            report['reviews'].append(dict(rank=rank,decision='official_scene_reject',
                reason='rejected_by_official_scene_filter',report=str(filter_path)))
            print(f'rank {rank}: GGX 官方场景距离筛选拒绝（{filtered["threshold_mm"]:g} mm）。',flush=True)
            checkpoint();continue
        folder=out/f'sim_rank_{rank:03d}';folder.mkdir()
        started=time.monotonic()
        result,samples=sim.evaluate_candidate(model,start,
            sim.candidate_for_object_origin(candidate,'ur-base',center),rank,sim_args,center)
        bounds=np.asarray(config['tcp_bounds_mm'])/1000
        addresses=[sim._joint_qpos_address(model,n) for n in sim.ARM_JOINTS]
        positions=np.array([support.fk_tcp(model,s.qpos[addresses])[:3] for s in samples])
        bounds_ok=bool(np.all((positions>=bounds[0]) & (positions<=bounds[1])))
        state,reason=classify(result,samples,bounds_ok)
        passed=state=='passed_current_checks'
        row.update(local_state=state,local_reason=reason,
            target_proxy_status='not_fully_evaluated' if result.reason.startswith('ik_failed') else
                'uncertain' if any(s.object_collision for s in samples) else 'clear_in_model',
            known_desk_clearance_mm=result.min_desk_clearance_mm,proxy_gap_mm=result.min_object_clearance_mm,
            report=str(folder/'summary.json'))
        result_row=asdict(result);result_row['passed']=passed
        sim._write_csv(folder/'candidates.csv',[result_row])
        sim._write_csv(folder/'trajectory.csv',sim._trajectory_rows(model,samples))
        sim._write_csv(folder/f'trajectory_rank_{rank:03d}.csv',sim._trajectory_rows(model,samples))
        summary=dict(schema=SCHEMA,simulation_only=True,passed=passed,
            automatic_pass_ranks=[rank] if passed else [],uncertain_ranks=[rank] if state=='uncertain_target_proxy' else [],
            feasibility=row,object_collision_gate=dict(revision=sim.OBJECT_GATE_REVISION),
            candidate_frame='ur-base',candidate_yaml=str(sim_args.graspgenx_yaml.resolve()),
            candidate_yaml_sha256=ui.digest(sim_args.graspgenx_yaml),
            model=str(support.URDF.resolve()),model_sha256=ui.digest(support.URDF),robot_model='CB3 UR5',
            config=str(config_path.resolve()),config_sha256=ui.digest(config_path),
            object_center_m=center.tolist(),object_size_mm=sim_args.object_size_mm,
            object_yaw_deg=sim_args.object_yaw_deg,table_height_m=sim_args.table_height_m,
            desk_top_m=source['desk_top_m'],desk_envelope=config['desk_envelope'],desk_clearance_mm=2.,
            real_config=str(args.real_config),perception_config=str(args.config),
            official_filter_report=str(filter_path.resolve()),source_sha256=pins,
            arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(sim_args).items()},
            timings=dict(evaluation_s=time.monotonic()-started,render_s=0.),
            assumptions=['Official endpoint scene-distance result is separate from CB3 sampled path/IK results.',
                'Fixed independently calibrated desk envelope/top; same builder as real execution.',
                'Target box contacts are uncertain proxy observations, not measured physical collisions or grasp failures.',
                'Uncertain candidates are NOT automatic passes; independent safety review remains required.',
                'No complete self-collision, hidden-obstacle, opening sweep or dynamic grasp certification.'])
        from tools.ggx_template import write_json
        write_json(folder/'summary.json',summary)
        print(f'rank {rank}: official PASS → CB3 {state} ({reason})',flush=True)
        checkpoint()
        if not passed:
            report['reviews'].append(dict(rank=rank,decision=state,reason=reason,report=str(folder/'summary.json')))
            checkpoint();continue
        video=folder/f'review_rank_{rank:03d}.mp4'
        render=not args.execute and getattr(args,'render_candidates',True)
        if render:
            started=time.monotonic()
            sim.write_video(video,model,samples,result,sim_args,center)
            summary['timings']['render_s']=time.monotonic()-started
            write_json(folder/'summary.json',summary)
        review_pins={**pins,**ui.pin([folder/'summary.json',folder/'candidates.csv',folder/f'trajectory_rank_{rank:03d}.csv',
                                    *([video] if render else [])])}
        yield rank,video,review_pins
    checkpoint()
    if (report['feasibility_counts']['current_checks_passed']==0
            and report['feasibility_counts']['uncertain_target_proxy']>0):
        report['status']='uncertain_candidates_require_independent_review'
        checkpoint()


def geometry_candidates(args,out,report,config_path,source_pins,*,filter_path=None):
    """Official scene screen + start-independent geometry, no fictitious travel."""
    from tools import run_ggx_review as ui, run_ggx_official_filter as official
    from tools import ggx_real_support as support, preflight_ggx_ur5_table as sim
    from tools import ggx_reset as reset, run_ggx_real as real, run_ggx_ten_starts as ten
    from tools import ggx_planning as planning
    from tools.preview_tabletop_target import load_perception_config
    from tools.ggx_template import write_json
    ui.verify(source_pins)
    if filter_path is None:
        folder=out/'scene_input';folder.mkdir()
        scene_input=official.prepare_scene(getattr(args,'inference_report',out/'inference/summary.json'),folder)
        ui.run_stage(out,'official_scene_filter','run_ggx_official_filter.py',
            ['--scene-input',scene_input,'--threshold-mm',args.scene_collision_mm,
             '--out',out/'official_scene_filter'],python=args.ggx_python)
        filter_path=out/'official_scene_filter/summary.json'
    sim_args=sim.parse_args(['--config',str(config_path),'--no-video'])
    sim_args.desk_clearance_mm=2.
    filtered=load_filter(filter_path,sim_args.graspgenx_yaml)
    if filtered['threshold_mm'] != args.scene_collision_mm:
        raise ValueError('Saved official threshold differs from declared protocol')
    config=real.load_config(args.real_config)
    if not config.get('desk_envelope',{}).get('confirmed'):
        raise ValueError('Geometry protocol requires confirmed fixed desk')
    pc,_=load_perception_config(args.config)
    fixed=reset.desk_source(config,pc)
    center=np.array([sim_args.object_x_m,sim_args.object_y_m,sim_args.table_height_m+sim_args.object_size_mm[2]/2000])
    source=dict(preflight=dict(object_center_m=center.tolist(),object_size_mm=sim_args.object_size_mm,
        object_yaw_deg=sim_args.object_yaw_deg,table_height_m=sim_args.table_height_m),
        desk_envelope=config['desk_envelope'],desk_top_m=fixed['preflight']['table_height_m'])
    model=support.build_model(source)
    candidates=sim.load_isaac_grasp_yaml(sim_args.graspgenx_yaml)
    if len(candidates)!=len(filtered['candidates']):
        raise ValueError('Official candidate count mismatch')
    pins={**source_pins,**filtered['source_sha256'],**ten.code_pins(args.real_config),
          **ui.pin([__file__,planning.__file__,filter_path,config_path,args.config,pc['desk_plane'],pc['calibration']])}
    report.update(source_sha256=pins,official_scene_filter=str(filter_path),
        feasibility_protocol=planning.REVISION,selection_scope='grasp_geometry',policy=planning.policy(),
        workspace_bounds_mm=config['tcp_bounds_mm'],feasibility_stages=[])
    def checkpoint():
        rows=report['feasibility_stages']
        report['feasibility_counts']=dict(candidates=len(rows),
            official_rejected=sum(not r['official_scene_clear'] for r in rows),
            geometry_eligible=sum(r.get('local_state')=='geometry_eligible' for r in rows),
            local_rejected=sum(r['official_scene_clear'] and r.get('local_state')!='geometry_eligible' for r in rows),
            target_proxy_uncertain=sum(r.get('target_proxy_uncertain',False) for r in rows),
            actual_travel_checked=0)
        ui.save_report(out,report)
        with (out/'feasibility.csv').open('w',newline='') as f:
            fields=['rank','confidence','official_scene_clear','local_state','standoff_mm','target_proxy_uncertain','report']
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    checkpoint()
    for rank,candidate in enumerate(candidates[:args.max_candidates]):
        if getattr(args,'candidate_rank',None) is not None and rank!=args.candidate_rank:
            continue
        ui.verify(pins)
        row=dict(rank=rank,confidence=candidate.confidence,official_scene_clear=filtered['candidates'][rank]['official_scene_clear'])
        report['feasibility_stages'].append(row)
        if not row['official_scene_clear']:
            row['local_state']='official_scene_reject';checkpoint();continue
        folder=out/f'sim_rank_{rank:03d}';folder.mkdir()
        start=time.monotonic()
        detail,samples=planning.select_geometry(model,
            sim.candidate_for_object_origin(candidate,'ur-base',center),rank,sim_args,center,config)
        eligible=detail['state']=='geometry_eligible'
        row.update(local_state=detail['state'],standoff_mm=detail.get('standoff_mm'),
            target_proxy_uncertain=detail.get('target_proxy_uncertain',False),report=str(folder/'summary.json'))
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(sim_args).items()}
        if eligible:arguments['standoff_mm']=detail['standoff_mm']
        summary=dict(schema=planning.SCHEMA,simulation_only=True,executable=False,passed=False,
            automatic_pass_ranks=[],geometry_review_ranks=[rank] if eligible else [],
            selection_scope='grasp_geometry',actual_travel_checked=False,policy=planning.policy(),
            geometry=detail,feasibility=row,object_collision_gate=dict(revision=sim.OBJECT_GATE_REVISION),
            candidate_frame='ur-base',candidate_yaml=str(sim_args.graspgenx_yaml),candidate_yaml_sha256=ui.digest(sim_args.graspgenx_yaml),
            model=str(support.URDF),model_sha256=ui.digest(support.URDF),robot_model='CB3 UR5',
            config=str(config_path),config_sha256=ui.digest(config_path),arguments=arguments,
            **source['preflight'],desk_top_m=source['desk_top_m'],desk_envelope=config['desk_envelope'],
            desk_clearance_mm=2.,real_config=str(args.real_config),perception_config=str(args.config),
            official_filter_report=str(filter_path),source_sha256=pins,timings=dict(evaluation_s=time.monotonic()-start),
            local_reference=dict(rows=planning.exact_reference_rows(model,samples,
                sim.candidate_for_object_origin(candidate,'ur-base',center),center,sim_args,detail['standoff_mm']) if eligible else [],
                minimum_desk_clearance_mm=min((s.min_desk_clearance_m for s in samples),default=0)*1000),
            assumptions=['Geometry selection only: NO scan-to-grasp path and NO trial motion approval.',
                'Fixed multi-seed and 80/60/40/20 mm retreat policy; original ranked grasp unchanged.',
                'Same local target/desk/workspace predicates as execution; unresolved local paths are not normal review candidates.',
                'Actual trial travel remains separately checked; local pass is not motion approval.'])
        write_json(folder/'summary.json',summary)
        sim._write_csv(folder/'candidates.csv',[dict(rank=rank,passed=False,geometry_eligible=eligible,reason=detail['state'])])
        sim._write_csv(folder/f'trajectory_rank_{rank:03d}.csv',sim._trajectory_rows(model,samples))
        print(f'rank {rank}: official PASS -> {detail["state"]}; standoff={detail.get("standoff_mm")} mm; travel not checked',flush=True)
        checkpoint()
        if not eligible:continue
        video=folder/f'review_rank_{rank:03d}.mp4'
        render=not args.execute and getattr(args,'render_candidates',True)
        if render:
            result=sim.CandidateResult(**detail['result']);result.passed=False;result.reason='GEOMETRY ONLY - travel not checked'
            sim.write_video(video,model,samples,result,sim_args,center)
        review_pins={**pins,**ui.pin([folder/'summary.json',folder/'candidates.csv',folder/f'trajectory_rank_{rank:03d}.csv',*([video] if render else [])])}
        yield rank,video,review_pins
    checkpoint()
