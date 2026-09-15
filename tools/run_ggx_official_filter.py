#!/usr/bin/env python3
"""Offline official GGX scene-distance filter; no camera, robot or hand access.

Run in the independent GGX environment. Uses the official function and Inspire
collision mesh, original rank order, and the reviewed target mask. This is an
endpoint scene-distance screen, NOT a trajectory or target-contact certificate.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_ggx_review import read_json, pin, verify, digest

SCHEMA = 'rh56_ggx_official_scene_filter/v1'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--inference', type=Path)
    inputs.add_argument('--scene-input', type=Path, help='Prepared calibrated scene; GGX environment does not need OpenCV')
    p.add_argument('--prepare-only', action='store_true', help='Control/sim environment: prepare scene, no Torch/GGX import')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--graspgenx-root', type=Path, default=ROOT/'artifacts/ggx_lab/GraspGenX')
    p.add_argument('--threshold-mm', type=float, default=20., help='Protocol parameter; official default 20 mm, never auto-tuned')
    p.add_argument('--seed', type=int, default=20260909)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    a = p.parse_args(argv)
    if a.prepare_only and not a.inference:
        p.error('--prepare-only requires --inference')
    if not np.isfinite(a.threshold_mm) or not 0 < a.threshold_mm <= 100:
        p.error('--threshold-mm must be positive and <=100; zero is not an uncertainty policy')
    if not 0 <= a.seed < 2**32:
        p.error('Invalid seed')
    for key, value in vars(a).items():
        if isinstance(value, Path): setattr(a, key, value.resolve())
    return a


def scene_points(depth, foreground, component, calibration, scale, roi):
    """Calibrated visible scene; exclude ONLY the selected foreground component."""
    import cv2
    from tools.analyze_ur5_camera_desk_plane import deproject_aligned_depth, transform_points
    if depth.shape != foreground.shape or depth.dtype != np.uint16:
        raise ValueError('Missing/mismatched reference depth and foreground mask')
    _, labels = cv2.connectedComponents((foreground > 0).astype(np.uint8))
    target = labels == component
    if type(component) is not int or component < 1 or target.sum() < 100:
        raise ValueError('Selected target mask missing; never exclude the whole scene')
    intr = calibration['selected_intrinsics']
    camera, pixels = deproject_aligned_depth(depth, intr['camera_matrix'],
        intr['distortion_coefficients'], scale, roi, 1, (.05, 2.))
    uv = pixels.astype(int)
    keep = ~target[uv[:,1], uv[:,0]]
    xyz = transform_points(calibration['transforms']['base_from_camera']['matrix'], camera[keep])
    if len(xyz) < 100:
        raise ValueError('Insufficient observed non-target scene; NOT a collision-free scene')
    x0,y0,x1,y1 = roi
    known = (depth[y0:y1,x0:x1]*scale > .05) & (depth[y0:y1,x0:x1]*scale < 2.)
    outside = ~target[y0:y1,x0:x1]
    return xyz, target, dict(target_pixels=int(target.sum()), scene_points_before_sampling=len(xyz),
        non_target_valid_depth_fraction=float(known[outside].mean()), roi=list(roi),
        missing_depth_policy='unobserved, not certified free; fixed-desk path checks remain separate')


def prepare_inputs(inference_path):
    import cv2
    import yaml
    from tools.preview_tabletop_target import load_perception_config
    from tools.run_vision_pick_cycle import validate_capture
    inf = read_json(inference_path)
    if (inf.get('schema') != 'rh56_ggx_point_cloud_inference/v1'
            or inf.get('input_frame') != 'ur_base'
            or inf.get('status') != 'candidates_generated_not_motion_validated'):
        raise ValueError('Expected successful measured UR-base inference')
    cloud, candidates = Path(inf['source_point_cloud']), Path(inf['candidate_file'])
    if digest(cloud) != inf['source_point_cloud_sha256'] or digest(candidates) != inf['candidate_sha256']:
        raise ValueError('Inference inputs/candidates changed')
    crop_path = cloud.parent/'summary.json'
    crop = read_json(crop_path)
    preview_path = Path(crop['source_preview'])
    if digest(preview_path) != crop['source_preview_sha256']:
        raise ValueError('Reference perception changed')
    preview = read_json(preview_path)
    c, _ = load_perception_config(Path(preview['config']))
    capture = Path(preview['capture'])
    metadata = validate_capture(c, capture)
    if digest(capture/'camera_check_depth_u16.png') != crop['source_depth_sha256']:
        raise ValueError('Reference depth changed')
    if digest(Path(c['calibration'])) != crop['calibration_sha256']:
        raise ValueError('Reference calibration changed')
    foreground_path = preview_path.parent/'foreground.png'
    if crop.get('source_foreground_sha256') and digest(foreground_path) != crop['source_foreground_sha256']:
        raise ValueError('Reference target mask changed')
    component = crop['reviewed_component_id']
    selected = [r for r in preview['candidates'] if r['component_id'] == component]
    if len(selected) != 1 or not selected[0]['geometry_candidate'] or selected[0]['rejection_reasons']:
        raise ValueError('Expected one explicitly selected geometry component')
    depth = cv2.imread(str(capture/'camera_check_depth_u16.png'), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(foreground_path), cv2.IMREAD_GRAYSCALE)
    if depth is None or mask is None or depth.shape != (720,1280) or mask.shape != depth.shape:
        raise ValueError('Missing calibrated scene images')
    points, target, stats = scene_points(depth, mask, component, yaml.safe_load(Path(c['calibration']).read_text()),
        metadata['camera']['depth']['scale_m_per_unit'], c['roi'])
    pins = pin([inference_path, cloud, candidates, crop_path, preview_path, foreground_path,
        c['calibration'], c['desk_plane'], preview['config'], capture/'camera_check.yaml',
        capture/'camera_check_depth_u16.png'])
    return inf, candidates, points, target, stats, pins


def prepare_scene(inference, out):
    import cv2
    inf, candidates, scene, mask, stats, pins = prepare_inputs(inference)
    np.save(out/'observed_scene_ur_base_m.npy',scene)
    cv2.imwrite(str(out/'excluded_target_mask.png'),mask.astype(np.uint8)*255)
    pins.update(pin([__file__,out/'observed_scene_ur_base_m.npy',out/'excluded_target_mask.png',
                    ROOT/'tools/analyze_ur5_camera_desk_plane.py']))
    payload=dict(schema='rh56_ggx_official_scene_input/v1',frame='ur_base',units='metres',
        inference=str(inference.resolve()),candidate_file=str(candidates.resolve()),
        scene_points=str((out/'observed_scene_ur_base_m.npy').resolve()),scene=stats,source_sha256=pins)
    (out/'scene_input.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    return out/'scene_input.json'


def run(args, report):
    import trimesh
    scene_input = args.scene_input or prepare_scene(args.inference,args.out)
    inputs=read_json(scene_input)
    if (inputs.get('schema')!='rh56_ggx_official_scene_input/v1' or inputs.get('frame')!='ur_base'
            or inputs.get('units')!='metres'):
        raise ValueError('Invalid prepared scene input')
    pins={**inputs['source_sha256'],**pin([scene_input])}
    verify(pins)
    for key in ('inference','candidate_file','scene_points'):
        if str(Path(inputs[key]).resolve()) not in pins:
            raise ValueError(f'Unbound scene input: {key}')
    candidate_path=Path(inputs['candidate_file'])
    scene=np.load(inputs['scene_points'],allow_pickle=False)
    if scene.ndim!=2 or scene.shape[1]!=3 or len(scene)<100 or not np.all(np.isfinite(scene)):
        raise ValueError('Missing/nonfinite observed scene; not automatically collision-free')
    stats=inputs['scene']
    root = args.graspgenx_root
    descriptor = root/'ext/gripper_descriptions/gripper_descriptions/assets/x_grippers/inspire_hand'
    mesh_path = descriptor/'coll_mesh.obj'
    official = root/'graspgenx/utils/collision_filter.py'
    if not mesh_path.is_file() or not official.is_file():
        raise FileNotFoundError('Official filter/Inspire collision mesh missing; no substitute geometry')
    # Same mesh loaded by resolve_gripper_info; do not load a model/checkpoint.
    os.environ['GRASPGENX_GRIPPER_CFG_DIR'] = str(root/'ext/gripper_descriptions')
    os.environ['GRASPGENX_CHECKPOINT_DIR'] = str(root/'ext/graspgenx_checkpoints')
    sys.path.insert(0, str(root))
    import torch
    from graspgenx.utils.collision_filter import filter_colliding_grasps
    from rh56_controller.graspgenx_baseline import load_isaac_grasp_yaml
    from tools.run_ggx_cube_inference import revision
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent protocol/device change')
    rng = np.random.default_rng(args.seed)
    if len(scene) > 8192:
        scene = scene[rng.choice(len(scene),8192,replace=False)]
    mesh = trimesh.load(mesh_path, force='mesh')
    if not isinstance(mesh,trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError('Invalid official collision mesh')
    surface, _ = trimesh.sample.sample_surface(mesh, 2000, seed=args.seed)
    candidates = load_isaac_grasp_yaml(candidate_path)
    poses = np.repeat(np.eye(4)[None],len(candidates),axis=0)
    for i,c in enumerate(candidates):
        poses[i,:3,:3], poses[i,:3,3] = c.rotation, c.position
    pins.update(pin([__file__,official,mesh_path,ROOT/'tools/analyze_ur5_camera_desk_plane.py',
                     ROOT/'rh56_controller/graspgenx_baseline.py']))
    np.save(args.out/'scene_without_target_ur_base_m.npy',scene)
    np.save(args.out/'official_hand_surface_m.npy',surface)
    pins.update(pin([args.out/'scene_without_target_ur_base_m.npy',args.out/'official_hand_surface_m.npy']))
    verify(pins)
    started = time.monotonic()
    free = filter_colliding_grasps(scene_pc=scene, grasp_poses=poses,
        gripper_surface_points=surface.astype(np.float32), collision_threshold=args.threshold_mm/1000,
        batch_size=16, device=device)
    if np.asarray(free).shape != (len(candidates),) or np.asarray(free).dtype != np.bool_:
        raise ValueError('Invalid official filter output')
    report.update(status='completed', candidate_file=str(candidate_path), candidate_sha256=digest(candidate_path),
        inference=inputs['inference'], source_sha256=pins, official_revision=revision(root),
        descriptor_revision=revision(root/'ext/gripper_descriptions'), official_function=str(official),
        scene=stats, sampled_scene_points=len(scene), hand_surface_samples=len(surface),
        device=device, elapsed_s=time.monotonic()-started,
        candidates=[dict(rank=i,name=c.name,confidence=c.confidence,official_scene_clear=bool(free[i]),
            decision='pass_scene_distance' if free[i] else 'rejected_by_official_scene_filter') for i,c in enumerate(candidates)])
    verify(pins)


def main(argv=None):
    a = parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema=SCHEMA, status='starting', executable=False, hardware_contacted=False,
        threshold_mm=a.threshold_mm, seed=a.seed, candidates=[],
        assumptions=['Official filter, original confidence ranks; endpoint only, not full GGX deployment.',
            'Observed calibrated ROI scene excludes only the selected target; missing depth is unknown.',
            'Official 20 mm default unless explicitly predeclared otherwise; no automatic threshold tuning.',
            'Official sampling counts: 2000 hand points, at most 8192 scene points; saved seeded realizations.',
            'Filter rejection is not a measured grasp failure; filter pass is not motion authorization.',
            'CB3 desk/path checks and target-contact uncertainty are separate.'])
    code=0
    try:
        if a.prepare_only:
            path=prepare_scene(a.inference,a.out)
            report.update(status='scene_prepared',scene_input=str(path))
        else:
            run(a,report)
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');code=1
    finally:
        (a.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        with (a.out/'summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['rank','name','confidence','official_scene_clear','decision'])
            w.writeheader();w.writerows(report['candidates'])
        print(f"Official GGX scene filter: {report['status']}; {a.out/'summary.json'}",flush=True)
        if report.get('error'): print(report['error'],file=sys.stderr)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
