#!/usr/bin/env python3
"""Generate GGX Inspire-Hand candidates for a cube or saved point cloud, offline only.

Run with the independent GraspGenX .venv, not the robot-control environment.
Reuses official sampler/config/export APIs; never imports robot control.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graspgenx-root', type=Path, default=ROOT / 'artifacts/ggx_lab/GraspGenX')
    parser.add_argument('--out', type=Path, required=True, help='New output directory, never overwritten')
    parser.add_argument('--num-grasps', type=int, default=200, help='Diffusion samples (200 for the initial 8 GB GPU smoke test)')
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260909)
    parser.add_argument('--save-all-candidates', action=argparse.BooleanOptionalAction, default=True,
                        help='Archive sampler output BEFORE sorting/top-k, without changing inference')
    parser.add_argument('--point-cloud',type=Path,help='Optional finite Nx3 .npy in metres; replaces synthetic cube')
    parser.add_argument('--input-frame',help='Required with --point-cloud, e.g. ur_base; exported poses use this same frame')
    args = parser.parse_args(argv)
    if not 0 < args.topk <= args.num_grasps or not 0 <= args.seed < 2**32:
        parser.error('Require 0 < topk <= num-grasps and seed in [0, 2**32)')
    if bool(args.point_cloud)!=bool(args.input_frame):
        parser.error('--point-cloud and --input-frame must be supplied together')
    args.graspgenx_root = args.graspgenx_root.resolve()
    args.out = args.out.resolve()
    return args


def revision(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def plain_point_cloud(points):
    """Normalize trimesh array subclasses for the official sampler's exact type check."""
    import numpy as np
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 or not np.all(np.isfinite(xyz)):
        raise ValueError('Expected nonempty finite Nx3 point cloud')
    return xyz


def inference_points(args, report):
    import numpy as np
    if args.point_cloud is not None:
        xyz=plain_point_cloud(np.load(args.point_cloud,allow_pickle=False))
        if len(xyz)<100:
            raise ValueError('Require at least 100 measured points')
        if not .005 < np.linalg.norm(np.ptp(xyz,axis=0)) < 2.:
            raise ValueError('Implausible object extent; point cloud must be in metres')
        report['source_point_cloud']=str(args.point_cloud.resolve())
        report['source_point_cloud_sha256']=hashlib.sha256(args.point_cloud.read_bytes()).hexdigest()
        np.save(args.out/'object_points_input_frame_m.npy',xyz)
    else:
        import trimesh
        mesh=trimesh.creation.box(extents=[.04,.04,.04])
        mesh.export(args.out/'cube_40mm.obj')
        xyz,_=trimesh.sample.sample_surface(mesh,3500,seed=args.seed)
        xyz=plain_point_cloud(xyz)
        np.save(args.out/'object_points_mesh_frame_m.npy',xyz)
    report['input_points']=len(xyz)
    return xyz


def archive_candidates(out, batches, mean, selected_poses, selected_scores, topk):
    """Record the unmodified official sample return and its stable rank lineage.

    This is instrumentation, not another sampler/ranker. Verify that the tap
    reproduces the official selected output before claiming complete lineage.
    """
    import numpy as np
    from graspgenx.dataset.eval_utils import save_to_isaac_grasp_format
    if len(batches) != 1:
        raise ValueError('Expected the declared single sampler pass')
    raw, scores = batches[0]
    raw = np.array(raw, copy=True)
    scores = np.asarray(scores)
    if raw.shape != (len(scores), 4, 4) or not np.isfinite(raw).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid pre-top-k sampler output')
    matrices = raw.copy()
    matrices[:, 3, 3] = 1.
    matrices[:, :3, 3] += mean
    order = np.argsort(-scores, kind='stable')
    selected = order[:topk]
    if not (np.array_equal(matrices[selected], selected_poses)
            and np.array_equal(scores[selected], selected_scores)):
        raise ValueError('Archived lineage differs from official top-k; do not silently rerank')
    np.savez(out/'all_candidates.npz', poses_sampler_frame=raw,
             poses_input_frame=matrices, confidence=scores, rank_to_generation_id=order,
             selected_generation_ids=selected, restored_mean_input_frame_m=mean)
    save_to_isaac_grasp_format(matrices[order], scores[order], str(out/'all_candidates.yml'))
    inverse = np.empty(len(order), dtype=int)
    inverse[order] = np.arange(len(order))
    with (out/'candidate_lineage.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['generation_id','confidence','rank','retained_topk'])
        writer.writeheader()
        writer.writerows(dict(generation_id=i, confidence=float(scores[i]), rank=int(inverse[i]),
                              retained_topk=bool(inverse[i] < topk)) for i in range(len(scores)))
    return dict(raw_count=len(raw), raw_file=str(out/'all_candidates.npz'),
                all_candidate_file=str(out/'all_candidates.yml'),
                raw_sha256=hashlib.sha256((out/'all_candidates.npz').read_bytes()).hexdigest(),
                lineage_file=str(out/'candidate_lineage.csv'), topk_verified_identical=True,
                capture_stage='official sample return, threshold=-1, before run_inference sorting/top-k')


def infer(args, report):
    root = args.graspgenx_root
    if not (root / 'graspgenx/grasp_server.py').is_file():
        raise FileNotFoundError(f'Missing official GGX checkout: {root}')
    checkpoint = root / 'ext/graspgenx_checkpoints/release'
    descriptors = root / 'ext/gripper_descriptions'
    for p in [checkpoint / 'gen/config.yaml', checkpoint / 'dis/config.yaml',
              descriptors / 'gripper_descriptions/assets/x_grippers/inspire_hand/config.json']:
        if not p.is_file():
            raise FileNotFoundError(f'Missing GGX dependency: {p}')
    import os
    os.environ['GRASPGENX_CHECKPOINT_DIR'] = str(checkpoint.parent)
    os.environ['GRASPGENX_GRIPPER_CFG_DIR'] = str(descriptors)
    sys.path[:0] = [str(ROOT), str(root), str(root / 'scripts')]
    import numpy as np
    import torch
    import trimesh
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use the independent GGX environment with GPU access')
    from demo_object_pc import load_model_cfg
    from graspgenx.grasp_server import GraspGenXSampler
    from graspgenx.dataset.eval_utils import save_to_isaac_grasp_format
    from rh56_controller.graspgenx_baseline import load_isaac_grasp_yaml

    report.update(graspgenx_revision=revision(root), checkpoint_revision=revision(checkpoint.parent),
                  descriptor_revision=revision(descriptors), torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  python=sys.version, checkpoint_root=str(checkpoint))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    xyz = inference_points(args,report)
    mean = xyz.mean(axis=0)
    pc = xyz - mean
    cloud_name='object_points_mesh_frame_m.npy' if args.point_cloud is None else 'object_points_input_frame_m.npy'
    report['point_cloud_sha256'] = hashlib.sha256((args.out / cloud_name).read_bytes()).hexdigest()
    report['sample_mean_input_frame_m'] = mean.tolist()
    if args.point_cloud is None: report['sample_mean_mesh_frame_m'] = mean.tolist()
    cfg = load_model_cfg(str(checkpoint / 'gen'), str(checkpoint / 'dis'), None, None)
    sampler = GraspGenXSampler(cfg, 'inspire_hand', assets_dir=str(root / 'assets'))
    batches = []
    if args.save_all_candidates:
        original_sample = sampler.sample
        def tapped_sample(*a, **kw):
            output = original_sample(*a, **kw)
            if len(output[0]):
                batches.append((output[0].detach().cpu().numpy().copy(),
                                output[1].detach().cpu().numpy().copy()))
            return output  # Same tensors/order; no RNG calls or sorting here.
        sampler.sample = tapped_sample
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    poses, confidence = GraspGenXSampler.run_inference(
        pc, sampler, grasp_threshold=-1., num_grasps=args.num_grasps,
        topk_num_grasps=args.topk, min_grasps=1, max_tries=1, remove_outliers=False)
    torch.cuda.synchronize()
    report['inference_seconds'] = time.perf_counter() - begin
    report['peak_gpu_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024**2
    if len(poses) == 0:
        raise RuntimeError('Official sampler returned no candidates; no automatic retry')
    matrices = poses.detach().cpu().numpy()
    scores = confidence.detach().cpu().numpy()
    matrices[:, 3, 3] = 1.
    matrices[:, :3, 3] += mean  # Undo centering exactly once, in the declared input frame.
    if not np.all(np.isfinite(matrices)) or not np.all(np.isfinite(scores)):
        raise RuntimeError('Nonfinite inference output')
    if args.save_all_candidates:
        report['pre_topk_archive'] = archive_candidates(args.out, batches, mean, matrices, scores, args.topk)
    output = args.out / 'graspgenx_candidates.yml'
    save_to_isaac_grasp_format(matrices, scores, str(output))
    candidates = load_isaac_grasp_yaml(output)
    report.update(status='candidates_generated_not_motion_validated', candidates=len(candidates),
                  confidence_min=float(scores.min()), confidence_max=float(scores.max()),
                  candidate_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  candidate_file=str(output))


def main(argv=None):
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='rh56_ggx_cube_inference/v1', status='starting',
                  time_utc=datetime.now(timezone.utc).isoformat(), seed=args.seed,
                  num_grasps=args.num_grasps, topk=args.topk, candidates=0,
                  cube_width_mm=40, hardware_commands_sent=False,
                  assumptions=['Complete centered 40 mm cube mesh, NOT camera perception.',
                               'Official Inspire-Hand diffusion sampler; single inference pass, no retry.',
                               'Seed and exact input cloud recorded; CUDA is not guaranteed bitwise deterministic.',
                               'Output in input mesh frame, metres; no iterative placement/width offsets.',
                               'Candidates are not robot trajectories or collision/success certification.'])
    report['input_frame']=args.input_frame or 'mesh'
    if args.point_cloud is not None:
        report.update(schema='rh56_ggx_point_cloud_inference/v1',cube_width_mm=None,
            assumptions=['Saved segmented point cloud in metres; input may be partial and manually cropped.',
                         'Official Inspire-Hand diffusion sampler; one pass, no automatic retry.',
                         'Seed/exact cloud recorded; CUDA is not guaranteed bitwise deterministic.',
                         'Exported poses in declared input frame, point mean restored once; no iterative offsets.',
                         'No full geometry completion, robot IK, collision or grasp-success validation.'])
    print(json.dumps(report, indent=2), flush=True)
    result = 0
    try:
        infer(args, report)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        print(report['error'], file=sys.stderr)
        result = 1
    except KeyboardInterrupt:
        report.update(status='interrupted')
        result = 130
    finally:
        (args.out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        with (args.out / 'summary.csv').open('w', newline='') as stream:
            keys = ['status', 'seed', 'num_grasps', 'topk', 'candidates', 'hardware_commands_sent']
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerow({k: report[k] for k in keys})
        print(f"GGX: {report['status']}; {args.out / 'summary.json'}", flush=True)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
