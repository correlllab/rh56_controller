#!/usr/bin/env python3
"""Frozen GGX diagnosis. NEVER camera, serial, RTDE, approval or motion.

freeze: preserve scene/evidence/current checker assets, without rewriting history.
replay: repeat saved trajectory and same-candidate checks.
batch: trace ALL archived candidates through ranking, official filter, local checks
       and nominal routes, including counterfactual candidates outside top-k.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def csv_rows(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def source_files(value):
    """Inventory existing referenced files, never follow directories or devices."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield from source_files(k); yield from source_files(v)
    elif isinstance(value, list):
        for v in value:
            yield from source_files(v)
    elif isinstance(value, str) and value.startswith(str(ROOT)+'/'):
        p = Path(value)
        if p.is_file():
            yield p


def historical_hashes(value):
    if isinstance(value, dict):
        for k, v in value.items():
            if (isinstance(k, str) and k.startswith(str(ROOT)+'/')
                    and isinstance(v, str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)):
                yield k, v
            else:
                yield from historical_hashes(v)
    elif isinstance(value, list):
        for v in value:
            yield from historical_hashes(v)


def freeze(args, report):
    import numpy as np
    import mujoco
    from tools import ggx_real_support as support, run_ggx_real as real
    from tools import ggx_fixed_starts as fixed, ggx_reset as reset
    from tools.preview_tabletop_target import load_perception_config
    from tools import preflight_ggx_ur5_table as sim
    review, real_run = read(args.review_run/'summary.json'), read(args.real_run/'summary.json')
    rank = review['selected_rank']
    if (review['status']!='review_approved' or real_run['selected_rank']!=rank
            or Path(real_run['approval']).resolve()!=args.review_run/'summary.json'):
        raise ValueError('Review and real evidence do not identify the same candidate')
    prepared = read(args.real_run/'prepared.json')
    if prepared['rank']!=rank:
        raise ValueError('Prepared candidate mismatch')
    inf = read(args.review_run/'inference/summary.json')
    files = set()
    records = []
    for directory in (args.review_run, args.real_run):
        files.update(p for p in directory.rglob('*') if p.is_file())
        records.extend(read(p) for p in directory.rglob('*.json'))
    for record in records:
        files.update(source_files(record))
    # Include current implementation dependencies as a separately identified
    # checker version, NOT replacements for missing historical source bytes.
    from tools.run_ggx_ten_starts import code_pins
    files.update(Path(p) for p in code_pins(args.config))
    files.update([Path(__file__), args.config, args.layout, args.perception,
                  ROOT/'tools/run_ggx_cube_inference.py', ROOT/'tools/run_ggx_frozen_inference.py'])
    # Preserve complete Python packages, not just entrypoint hashes. This lets
    # --frozen-code replay after the working tree changes, without restoring it.
    for package in ('tools','rh56_controller'):
        files.update(p for p in (ROOT/package).rglob('*.py') if '__pycache__' not in p.parts)
    pc, _ = load_perception_config(args.perception)
    files.update(Path(pc[k]) for k in ('calibration','desk_plane'))
    for record in records:
        for key in ('background',):
            value = record.get(key) if isinstance(record,dict) else None
            if value and Path(value).is_dir() and Path(value).is_relative_to(ROOT):
                files.update(p for p in Path(value).rglob('*') if p.is_file())
    for p in args.official_filter.parent.glob('*.npy'):
        files.add(p)
    files.add(args.official_filter)
    official_root = ROOT/'artifacts/ggx_lab/GraspGenX'
    files.update(p for p in (official_root/'graspgenx').rglob('*.py'))
    files.update(p for p in (official_root/'scripts').glob('*.py'))
    descriptor = official_root/'ext/gripper_descriptions/gripper_descriptions/assets/x_grippers/inspire_hand'
    files.update(p for p in descriptor.rglob('*') if p.is_file())
    expected = set(h for record in records for h in historical_hashes(record))
    mismatches=[]
    for p, h in sorted(expected):
        current = digest(p) if Path(p).is_file() else None
        if current!=h:
            mismatches.append(dict(path=p,historical_sha256=h,current_sha256=current))
    for p in sorted(files):
        if not p.resolve().is_relative_to(ROOT):
            continue  # Interpreter symlinks are environment metadata, not scene files.
        relative = p.resolve().relative_to(ROOT)
        target = args.out/'files'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    candidates = sim.load_isaac_grasp_yaml(args.review_run/'inference/graspgenx_candidates.yml')
    chosen = candidates[rank]
    poses = np.repeat(np.eye(4)[None], len(candidates), axis=0)
    for i,c in enumerate(candidates):
        poses[i,:3,:3], poses[i,:3,3] = c.rotation,c.position
    np.savez(args.out/'reference_candidates.npz', poses=poses,
             scores=[c.confidence for c in candidates], selected_rank=rank)
    config = real.load_config(args.config)
    desk = reset.desk_source(config, pc)
    old = prepared['source']['preflight']
    source = dict(rank=rank, preflight=old, desk_envelope=config['desk_envelope'],
                  desk_top_m=desk['preflight']['table_height_m'], desk_edge_y_m=None)
    model = support.build_model(source)
    mujoco.mj_saveModel(model, str(args.out/'current_model.mjb'))
    mujoco.mj_saveModel(support.build_model(prepared['source']), str(args.out/'historical_scene_model.mjb'))
    layout = fixed.build(args.layout, config, args.perception)
    inputs = dict(original_workspace=str(ROOT),source=source, config=config, layout=layout, snapshot=real_run['start_snapshot'],
                  original_seed=inf['seed'],num_grasps=inf['num_grasps'],topk=inf['topk'],
                  inference_point_cloud=str(Path('files')/Path(inf['source_point_cloud']).relative_to(ROOT)),
                  review_run=str(args.review_run.relative_to(ROOT)),real_run=str(args.real_run.relative_to(ROOT)),
                  official_filter=str(Path('files')/args.official_filter.relative_to(ROOT)),
                  current_source_sha256={str(p):digest(p) for p in files if p.suffix=='.py'},
                  current_checker_not_historical=True)
    import importlib.metadata
    inputs['environment']={name:importlib.metadata.version(name) for name in ('numpy','scipy','mujoco','mink')}
    write(args.out/'inputs.json',inputs)
    weights=[]
    for p in sorted((official_root/'ext/graspgenx_checkpoints/release').rglob('*')):
        if p.is_file():
            weights.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
    write(args.out/'weights_inventory.json',weights)
    write(args.out/'historical_mismatches.json',mismatches)
    manifest = {str(p.relative_to(args.out)):digest(p) for p in args.out.rglob('*') if p.is_file()}
    write(args.out/'manifest.json', manifest)
    report.update(status='frozen',rank=rank,validation_level='real_approach_completed_not_grasp_lift',
        historical_status=real_run['status'],requested_phase=real_run['requested_phase'],
        historical_scene_policy=real_run.get('scene_policy'),
        historical_code_or_input_mismatches=len(mismatches),copied_files=len(files),
        historical_execution_reproducible_bitwise=not bool(mismatches),
        source_point_cloud_sha256=digest(args.out/inputs['inference_point_cloud']),
        limitations=['Historical RGB-D is preserved; historical execution used operator fixed-scene confirmation, not a fresh verified capture.',
            'Historical source hashes with unavailable original bytes are reported, not overwritten or forged.',
            'Compiled historical-scene and current-scene models use CURRENT builder; saved motion samples are replayed without IK.',
            'Weights pinned in external inventory, not duplicated; replay verifies them before new inference.',
            'Approach completion is not grasp/lift success or an executable authorization.'])


def bundle_inputs(bundle):
    for rel, expected in read(bundle/'manifest.json').items():
        if digest(bundle/rel)!=expected:
            raise ValueError(f'Frozen file changed: {rel}')
    inputs=read(bundle/'inputs.json')
    for original, expected in inputs['current_source_sha256'].items():
        path=Path(original)
        if ROOT.resolve()==(bundle/'files').resolve():
            path=bundle/'files'/path.relative_to(inputs['original_workspace'])
        if digest(path)!=expected:
            raise ValueError(f'Checker/source changed since freeze; create a new versioned comparison: {original}')
    return inputs


def context(bundle):
    import mujoco
    import numpy as np
    from tools import preflight_ggx_ur5_table as sim
    inputs=bundle_inputs(bundle)
    model=mujoco.MjModel.from_binary_path(str(bundle/'current_model.mjb'))
    argv=['--config',str(bundle/'files'/inputs['review_run']/'simulation.yaml'),'--no-video']
    args=sim.parse_args(argv)
    args.desk_clearance_mm=2.
    return inputs,model,args,np.array(inputs['source']['preflight']['object_center_m'])


def local_check(model,candidate,rank,args,center,config):
    """Record the current UI verdict AND unchanged execution local predicates.

    Do not search extra standoffs/branches to improve an audit result. The same
    sampled local approach is checked with the execution target gate enabled.
    """
    from tools import ggx_planning as plan, preflight_ggx_ur5_table as sim
    candidate=sim.candidate_for_object_origin(candidate,'ur-base',center)
    detail,samples=plan.select_geometry(model,candidate,rank,args,center,config)
    eligible=detail['state']=='geometry_eligible'
    failures=plan.geometry_failures(model,samples,config,target_gate=True) if eligible else []
    return detail,samples,dict(initial_review_eligible=eligible,
        unified_local_pass=bool(eligible and not failures),
        first_local_loss=detail['state'] if not eligible else failures[0]['kind'] if failures else None,
        failures=failures)


def replay(args,report):
    import numpy as np
    import mujoco
    from tools import ggx_planning as plan, preflight_ggx_ur5_table as sim
    from tools import ggx_real_support as support
    inputs,model,simargs,center=context(args.bundle)
    candidates=sim.load_isaac_grasp_yaml(args.bundle/'files'/inputs['review_run']/'inference/graspgenx_candidates.yml')
    rank=inputs['source']['rank']
    execution=read(args.bundle/'files'/inputs['real_run']/'execution.json')
    historical=mujoco.MjModel.from_binary_path(str(args.bundle/'historical_scene_model.mjb'))
    repetitions=[]
    for index in range(args.repeats):
        results={}
        for name,m in [('historical_scene_current_checker',historical),('current_scene_current_checker',model)]:
            samples=[]
            for command in execution['commands']:
                for row in command['samples']:
                    q=sim.initial_qpos(m,np.asarray(row['q']))
                    sim.set_hand_qpos(m,q,sim._lerp_ctrl(command['closure_alpha']))
                    samples.append(sim.TrajectorySample(command['phase'],row['fraction'],q,command['closure_alpha']))
            results[name]=dict(samples=len(samples),failures=plan.geometry_failures(m,samples,inputs['config'],target_gate=True))
        detail,samples,local=local_check(model,candidates[rank],rank,simargs,center,inputs['config'])
        rows=plan.exact_reference_rows(model,samples,sim.candidate_for_object_origin(candidates[rank],'ur-base',center),
            center,simargs,detail['standoff_mm']) if samples else []
        check=dict(repetition=index,saved_path=results,local=local,geometry=detail,rows=rows)
        write(args.out/f'repeat_{index}.json',check)
        stable={k:v for k,v in check.items() if k!='repetition'}
        repetitions.append(dict(repetition=index,result_sha256=hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest(),
                                unified_local_pass=local['unified_local_pass']))
        print(f'Frozen replay {index+1}/{args.repeats}: local={local["unified_local_pass"]}',flush=True)
    report.update(status='completed',rank=rank,repetitions=repetitions,
                  checks_bitwise_identical=len({r['result_sha256'] for r in repetitions})==1)
    csv_rows(args.out/'summary.csv',repetitions)


def compare_reference(poses, reference):
    import numpy as np
    from scipy.spatial.transform import Rotation
    distance=np.linalg.norm(poses[:,:3,3]-reference[:3,3],axis=1)*1000
    angle=np.degrees(Rotation.from_matrix(np.swapaxes(poses[:,:3,:3],1,2)@reference[:3,:3]).magnitude())
    return distance,angle


def exported_candidates(inference):
    """Use the exact YAML read-back convention used by review/real execution."""
    import numpy as np
    from rh56_controller.graspgenx_baseline import load_isaac_grasp_yaml
    archive=np.load(inference/'all_candidates.npz',allow_pickle=False)
    candidates=load_isaac_grasp_yaml(inference/'all_candidates.yml')
    order=archive['rank_to_generation_id']
    if len(candidates)!=len(order):raise ValueError('Incomplete all-candidate YAML')
    poses=np.repeat(np.eye(4)[None],len(order),axis=0)
    for rank,raw_id in enumerate(order):
        c=candidates[rank]
        if c.confidence!=float(archive['confidence'][raw_id]):raise ValueError('Exported rank lineage changed')
        poses[raw_id,:3,:3],poses[raw_id,:3,3]=c.rotation,c.position
    return candidates,poses


def batch(args,report):
    import numpy as np
    from rh56_controller.graspgenx_baseline import GraspGenXCandidate
    from tools import ggx_planning as plan, preflight_ggx_ur5_table as sim
    from tools import ggx_real_support as support
    inputs,model,simargs,center=context(args.bundle)
    archive=np.load(args.inference/'all_candidates.npz',allow_pickle=False)
    inf=read(args.inference/'summary.json')
    if inf['source_point_cloud_sha256']!=digest(args.bundle/inputs['inference_point_cloud']):
        raise ValueError('Inference does not use the frozen input cloud')
    scores=archive['confidence']
    candidates,poses=exported_candidates(args.inference)
    order=archive['rank_to_generation_id']
    official=read(args.official_mask)
    if official['candidate_archive_sha256']!=digest(args.inference/'all_candidates.npz'):
        raise ValueError('Official mask does not bind archived candidates')
    if official.get('pose_convention')!='official_export_yaml_readback':
        raise ValueError('Use official filter on the execution YAML convention, not raw diffusion rotation matrices')
    frozen_filter=(args.bundle/inputs['official_filter']).parent
    if (official['scene_sha256']!=digest(frozen_filter/'scene_without_target_ur_base_m.npy')
            or official['surface_sha256']!=digest(frozen_filter/'official_hand_surface_m.npy')
            or official['threshold_mm']!=read(args.bundle/inputs['official_filter'])['threshold_mm']):
        raise ValueError('Official mask changed frozen geometry/threshold')
    free=np.asarray(official['clear_by_generation_id'],dtype=bool)
    if free.shape!=(len(scores),):raise ValueError('Official mask size mismatch')
    reference=np.load(args.bundle/'reference_candidates.npz')['poses'][inputs['source']['rank']]
    distance,angle=compare_reference(poses,reference)
    rows=[]
    def save():
        report.update(candidates=len(scores),evaluated=len(rows),rows=rows)
        write(args.out/'summary.json',report)
        csv_rows(args.out/'summary.csv',rows)
    for rank,raw_id in enumerate(order):
        if rank<args.rank_start or (args.rank_stop is not None and rank>=args.rank_stop):continue
        if args.limit is not None and rank>=args.limit:break
        raw_id=int(raw_id)
        row=dict(rank=rank,generation_id=raw_id,retained_topk=rank<inputs['topk'],confidence=float(scores[raw_id]),
            reference_distance_mm=float(distance[raw_id]),reference_rotation_deg=float(angle[raw_id]),
            reference_near_1mm_1deg=bool(distance[raw_id]<=1 and angle[raw_id]<=1),
            reference_near_5mm_5deg=bool(distance[raw_id]<=5 and angle[raw_id]<=5),
            official_clear=bool(free[raw_id]),initial_review_eligible=False,unified_local_pass=False,
            route_checked=False,nominal_route_pass=False,route_pass_points='',first_loss='official_scene_filter')
        rows.append(row)
        if free[raw_id]:
            c=candidates[rank]
            detail,samples,local=local_check(model,c,rank,simargs,center,inputs['config'])
            row.update({k:v for k,v in local.items() if k!='failures'})
            row['first_loss']=local['first_local_loss']
            folder=args.out/f'rank_{rank:04}';folder.mkdir()
            write(folder/'local.json',dict(geometry=detail,shared_execution_predicates=local))
            if local['unified_local_pass'] and args.routes:
                row['route_checked']=True
                common=plan.exact_reference_rows(model,samples,sim.candidate_for_object_origin(c,'ur-base',center),center,simargs,detail['standoff_mm'])
                source=dict(inputs['source']);source['rank']=rank
                source['pins']={str(args.bundle/'manifest.json'):digest(args.bundle/'manifest.json'),
                                str(args.inference/'all_candidates.npz'):digest(args.inference/'all_candidates.npz')}
                close=[r for r in common if r['phase']=='close'][-1]
                ref=dict(schema='rh56_ggx_cb3_prepared/v1',executable=False,rows=common,
                         closure_alpha=close['closure_alpha'])
                results=[]
                for point in inputs['layout']['points']:
                    pdir=folder/point['id'];pdir.mkdir()
                    try:
                        plan.execution(model,source,ref,common,inputs['snapshot'],inputs['config'],pdir,point=point)
                        results.append(dict(point=point['id'],status='nominal_route_pass'))
                    except plan.PlanningUnavailable as exc:
                        results.append(dict(point=point['id'],status='planning_unavailable',reason=exc.reason,attempts=exc.attempts))
                write(folder/'routes.json',results)
                passed=[r['point'] for r in results if r['status']=='nominal_route_pass']
                row.update(nominal_route_pass=bool(passed),route_pass_points=','.join(passed),
                           first_loss=None if passed else 'actual_nominal_route')
        row['pipeline_first_loss']='topk_truncation' if not row['retained_topk'] else row['first_loss']
        # Outside-top-k checks are counterfactual, not originally executed steps.
        if (rank+1)%10==0:save();print(f'Frozen batch {args.inference.name}: {rank+1}/{len(scores)}',flush=True)
    report.update(status='completed',all_candidates_evaluated=len(rows)==len(scores),
        covered_ranks=[r['rank'] for r in rows],pose_convention='official_export_yaml_readback',
        topk_local_available=sum(r['unified_local_pass'] and r['retained_topk'] for r in rows),
        all_local_available=sum(r['unified_local_pass'] for r in rows),
        topk_route_available=sum(r['nominal_route_pass'] and r['retained_topk'] for r in rows),
        all_route_available=sum(r['nominal_route_pass'] for r in rows))
    save()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['freeze','replay','batch'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--bundle',type=Path)
    p.add_argument('--frozen-code',action='store_true',help='Replay using copied Python sources, not the live worktree')
    p.add_argument('--review-run',type=Path,default=ROOT/'artifacts/ggx_review_runs/20260911T194809_266833Z')
    p.add_argument('--real-run',type=Path,default=ROOT/'artifacts/ggx_real_runs/20260911T210031_032481Z')
    p.add_argument('--official-filter',type=Path,default=ROOT/'artifacts/ggx_official_audit_20260911/reference53/official_scene_filter/summary.json')
    p.add_argument('--config',type=Path,default=ROOT/'configs/ggx_real_lab.yaml')
    p.add_argument('--perception',type=Path,default=ROOT/'configs/tabletop_vision_lab.yaml')
    p.add_argument('--layout',type=Path,default=ROOT/'configs/ggx_ten_starts_lab.yaml')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--inference',type=Path)
    p.add_argument('--official-mask',type=Path)
    p.add_argument('--routes',action='store_true',help='Offline nominal routes only, for local-clear candidates')
    p.add_argument('--limit',type=int,help='Explicit incomplete smoke test; omitted evaluates all archived candidates')
    p.add_argument('--rank-start',type=int,default=0,help='Offline sharding only; original ranks unchanged')
    p.add_argument('--rank-stop',type=int,help='Exclusive original-rank end for an offline shard')
    a=p.parse_args(argv)
    if a.action!='freeze' and not a.bundle:p.error('--bundle required')
    if a.frozen_code and a.action=='freeze':p.error('--frozen-code requires an existing bundle')
    if a.action=='batch' and (not a.inference or not a.official_mask):p.error('batch needs --inference and --official-mask')
    if a.repeats<2 or (a.limit is not None and a.limit<1):p.error('repeats>=2 and positive limit required')
    if a.rank_start<0 or (a.rank_stop is not None and a.rank_stop<=a.rank_start):p.error('Invalid rank shard')
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    return a


def main(argv=None):
    a=parse_args(argv)
    if a.frozen_code:
        entry=a.bundle/'files/tools/run_ggx_frozen_audit.py'
        manifest=read(a.bundle/'manifest.json')
        if digest(entry)!=manifest['files/tools/run_ggx_frozen_audit.py']:
            raise ValueError('Frozen audit entrypoint changed')
        command=[sys.executable,str(entry),*(x for x in (sys.argv[1:] if argv is None else argv) if x!='--frozen-code')]
        return subprocess.call(command)
    a.out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='rh56_ggx_frozen_audit/v1',status='running',action=a.action,
        hardware_contacted=False,executable=False,started_epoch_s=time.time(),
        assumptions=['Saved inputs only; no hardware API or approval path.',
            'No proxy/model/clearance/standoff/seed-search tuning to improve acceptance.',
            'Current UI and execution local predicates reported separately; uncertainty is not a physical failure.',
            'Near-reference 1mm/1deg and 5mm/5deg are diagnostics, NOT identity or acceptance thresholds.',
            'Nominal route availability is not controller validation or real grasp success.'])
    code=0
    try:globals()[a.action](a,report)
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');code=1
        print(report['error'],file=sys.stderr)
    finally:
        report['elapsed_s']=time.time()-report['started_epoch_s']
        write(a.out/'summary.json',report)
        if not (a.out/'summary.csv').exists():csv_rows(a.out/'summary.csv',[{k:v for k,v in report.items() if not isinstance(v,(dict,list))}])
        print(f'Frozen audit {a.action}: {report["status"]}; {a.out}',flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
