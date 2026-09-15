#!/usr/bin/env python3
"""Independent GPU worker for saved GGX inputs ONLY; no hardware interfaces.

Runs fresh inference processes for identical-seed repeats, then distinct seeds.
Uses the frozen official scene/hand samples for ALL pre-top-k candidates.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_ggx_frozen_audit import read,write,digest,csv_rows,bundle_inputs


def filter_archive(bundle,inputs,inference,out):
    import numpy as np
    from tools.run_ggx_frozen_audit import exported_candidates
    from graspgenx.utils.collision_filter import filter_colliding_grasps
    folder=(bundle/inputs['official_filter']).parent
    scene=np.load(folder/'scene_without_target_ur_base_m.npy',allow_pickle=False)
    surface=np.load(folder/'official_hand_surface_m.npy',allow_pickle=False).astype(np.float32)
    threshold=read(bundle/inputs['official_filter'])['threshold_mm']
    _,poses=exported_candidates(inference)
    free=filter_colliding_grasps(scene_pc=scene,grasp_poses=poses,
        gripper_surface_points=surface,collision_threshold=threshold/1000,batch_size=16,device='cuda')
    write(out,dict(schema='rh56_ggx_frozen_official_mask/v1',
        candidate_archive_sha256=digest(inference/'all_candidates.npz'),
        candidate_yaml_sha256=digest(inference/'all_candidates.yml'),
        scene_sha256=digest(folder/'scene_without_target_ur_base_m.npy'),
        surface_sha256=digest(folder/'official_hand_surface_m.npy'),
        pose_convention='official_export_yaml_readback',threshold_mm=threshold,device='cuda',
        clear_by_generation_id=free.tolist()))
    return free


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--seeds',type=int,nargs='+',default=[20260909,20260910,20260911])
    p.add_argument('--filter-only',type=Path,nargs='+',help='Saved inference directories; no new inference')
    a=p.parse_args(argv)
    if a.repeats<2 or len(set(a.seeds))!=len(a.seeds) or any(not 0<=s<2**32 for s in a.seeds):
        p.error('repeats>=2 and distinct uint32 seeds required')
    a.bundle=a.bundle.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='rh56_ggx_frozen_inference/v1',status='running',hardware_contacted=False,
        executable=False,runs=[],seed_list=a.seeds,repeats=a.repeats,
        assumptions=['Identical frozen input in fresh processes; default CUDA behavior retained, not forced deterministic.',
            'Original top-k retained verbatim; pre-truncation tap records all sampler output.',
            'Official filter uses identical saved surface/scene points and threshold for every candidate.',
            'Candidate availability still requires the separate CPU local/path audit.'])
    code=0
    try:
        import numpy as np
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU inference substitution')
        inputs=bundle_inputs(a.bundle)
        if not a.filter_only:
            for item in read(a.bundle/'weights_inventory.json'):
                if digest(item['path'])!=item['sha256']:raise ValueError('Checkpoint changed since freeze')
        root=ROOT/'artifacts/ggx_lab/GraspGenX';sys.path.insert(0,str(root))
        from graspgenx.utils.collision_filter import filter_colliding_grasps
        if a.filter_only:
            for folder in a.filter_only:
                free=filter_archive(a.bundle,inputs,folder.resolve(),a.out/(folder.name+'.json'))
                report['runs'].append(dict(path=str(folder),official_pass=int(free.sum()),count=len(free)))
            report['status']='completed'
            return 0
        folder=(a.bundle/inputs['official_filter']).parent
        scene=np.load(folder/'scene_without_target_ur_base_m.npy',allow_pickle=False)
        surface=np.load(folder/'official_hand_surface_m.npy',allow_pickle=False).astype(np.float32)
        threshold=read(a.bundle/inputs['official_filter'])['threshold_mm']
        reference=np.load(a.bundle/'reference_candidates.npz')
        ref_free=filter_colliding_grasps(scene_pc=scene,grasp_poses=reference['poses'],
            gripper_surface_points=surface,collision_threshold=threshold/1000,batch_size=16,device='cuda')
        ref_free_again=filter_colliding_grasps(scene_pc=scene,grasp_poses=reference['poses'],
            gripper_surface_points=surface,collision_threshold=threshold/1000,batch_size=16,device='cuda')
        old=np.array([r['official_scene_clear'] for r in read(a.bundle/inputs['official_filter'])['candidates']])
        report['official_reference_replay']=dict(repeated_identical=bool(np.array_equal(ref_free,ref_free_again)),
            differs_from_saved_ranks=np.flatnonzero(ref_free!=old).tolist(),
            selected_reference_clear=bool(ref_free[inputs['source']['rank']]))
        schedule=[a.seeds[0]]*a.repeats+a.seeds[1:]
        first=None
        for index,seed in enumerate(schedule):
            folder=a.out/f'run_{index:02}_seed_{seed}'
            command=[sys.executable,str(ROOT/'tools/run_ggx_cube_inference.py'),
                '--point-cloud',str(a.bundle/inputs['inference_point_cloud']),'--input-frame','ur_base',
                '--num-grasps',str(inputs['num_grasps']),'--topk',str(inputs['topk']),
                '--seed',str(seed),'--save-all-candidates','--out',str(folder)]
            with (a.out/f'run_{index:02}.log').open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'Inference failed; see run_{index:02}.log')
            archive=np.load(folder/'all_candidates.npz',allow_pickle=False)
            poses,scores=archive['poses_input_frame'],archive['confidence']
            order=archive['rank_to_generation_id'];selected=archive['selected_generation_ids']
            if len(poses)!=inputs['num_grasps']:raise ValueError('Pre-top-k archive is incomplete')
            free=filter_archive(a.bundle,inputs,folder,folder/'official_mask.json')
            row=dict(index=index,seed=seed,path=str(folder),raw_count=len(poses),
                topk_count=len(selected),official_raw_pass=int(free.sum()),official_topk_pass=int(free[selected].sum()),
                historical_topk_poses_bitwise_equal=bool(np.array_equal(poses[selected],reference['poses'])),
                historical_topk_scores_bitwise_equal=bool(np.array_equal(scores[selected],reference['scores'])),
                historical_topk_max_position_difference_mm=float(np.max(np.linalg.norm(poses[selected,:3,3]-reference['poses'][:,:3,3],axis=1))*1000))
            if first is None:first=(poses.copy(),scores.copy(),order.copy())
            if seed==a.seeds[0]:
                row.update(same_seed_raw_poses_equal=bool(np.array_equal(poses,first[0])),
                    same_seed_raw_scores_equal=bool(np.array_equal(scores,first[1])),
                    same_seed_order_equal=bool(np.array_equal(order,first[2])),
                    same_seed_max_pose_element_difference=float(np.max(np.abs(poses-first[0]))),
                    same_seed_max_score_difference=float(np.max(np.abs(scores-first[1]))))
            report['runs'].append(row)
            write(a.out/'summary.json',report);csv_rows(a.out/'summary.csv',report['runs'])
            print(f'Frozen inference {index+1}/{len(schedule)}: seed={seed}; archived={len(poses)}; original top-k identical={row["historical_topk_poses_bitwise_equal"]}',flush=True)
        report['status']='completed'
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');code=1
        print(report['error'],file=sys.stderr)
    finally:
        write(a.out/'summary.json',report)
        if not (a.out/'summary.csv').exists():csv_rows(a.out/'summary.csv',[dict(status=report['status'])])
    return code


if __name__=='__main__':raise SystemExit(main())
