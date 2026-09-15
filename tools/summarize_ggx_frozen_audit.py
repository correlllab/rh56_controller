#!/usr/bin/env python3
"""Merge complete disjoint frozen-audit rank shards. Offline evidence only."""
import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_ggx_frozen_audit import read,write,digest,csv_rows,compare_reference


def merge_shards(paths, count):
    rows=[]
    for path in paths:
        shard=read(path)
        if shard['status']!='completed' or shard.get('pose_convention')!='official_export_yaml_readback':
            raise ValueError(f'Incomplete/noncanonical shard: {path}')
        rows.extend(shard['rows'])
    if sorted(r['rank'] for r in rows)!=list(range(count)):
        raise ValueError('Missing or duplicated original ranks; cannot report a complete acquisition rate')
    return sorted(rows,key=lambda r:r['rank'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace-root',type=Path,required=True)
    p.add_argument('--inference-root',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    from scipy.spatial.transform import Rotation
    inf=read(a.inference_root/'summary.json');inputs=read(a.bundle/'inputs.json')
    report=dict(schema='rh56_ggx_frozen_localization/v1',status='running',hardware_contacted=False,
                executable=False,seeds=[],input_sha256=digest(a.bundle/inputs['inference_point_cloud']),
                assumptions=['Same frozen scene, no new perception and no hardware motion.',
                    'Availability means a local-clear candidate with >=1 of the ten NOMINAL routes; not controller or real grasp success.',
                    'Counterfactual outside-top-k checks never change operational top-k=100.',
                    'Three predeclared distinct seeds are a small pilot, not a general success probability.',
                    'Early v1/v2 incomplete/raw-matrix traces are excluded; only complete canonical YAML shards are merged.'])
    code=0
    try:
        same=[r for r in inf['runs'] if r['seed']==inf['seed_list'][0]]
        arrays=[np.load(Path(r['path'])/'all_candidates.npz',allow_pickle=False) for r in same]
        first=arrays[0];comparisons=[]
        for index,other in enumerate(arrays):
            delta=np.linalg.norm(first['poses_input_frame'][:,:3,3]-other['poses_input_frame'][:,:3,3],axis=1)*1000
            Ra=Rotation.from_matrix(first['poses_input_frame'][:,:3,:3]).as_matrix()
            Rb=Rotation.from_matrix(other['poses_input_frame'][:,:3,:3]).as_matrix()
            angle=np.degrees(Rotation.from_matrix(np.swapaxes(Ra,1,2)@Rb).magnitude())
            comparisons.append(dict(repetition=index,
                raw_poses_bitwise_equal=bool(np.array_equal(first['poses_input_frame'],other['poses_input_frame'])),
                raw_scores_bitwise_equal=bool(np.array_equal(first['confidence'],other['confidence'])),
                topk_ids_and_order_equal=bool(np.array_equal(first['selected_generation_ids'],other['selected_generation_ids'])),
                full_rank_changed_positions=int(np.sum(first['rank_to_generation_id']!=other['rank_to_generation_id'])),
                max_translation_mm=float(delta.max()),max_rotation_deg=float(angle.max()),
                max_confidence_difference=float(np.abs(first['confidence']-other['confidence']).max())))
        report['same_seed_comparison']=comparisons
        report['saved_candidate_replay']=read(a.replay/'summary.json')
        report['reference_validation']=read(a.bundle/'summary.json')
        allrows=[]
        for seed in inf['seed_list']:
            paths=sorted((a.trace_root/f'seed_{seed}').glob('*/summary.json'))
            rows=merge_shards(paths,inputs['num_grasps'])
            top=[r for r in rows if r['retained_topk']]
            available=[r for r in rows if r['nominal_route_pass']]
            seedrow=dict(seed=seed,generated=len(rows),topk=len(top),
                official_all=sum(r['official_clear'] for r in rows),official_topk=sum(r['official_clear'] for r in top),
                review_eligible_all=sum(r['initial_review_eligible'] for r in rows),
                review_eligible_topk=sum(r['initial_review_eligible'] for r in top),
                shared_local_all=sum(r['unified_local_pass'] for r in rows),
                shared_local_topk=sum(r['unified_local_pass'] for r in top),
                nominal_available_all=len(available),nominal_available_topk=sum(r['nominal_route_pass'] for r in top),
                available_ranks=[r['rank'] for r in available],
                lost_at_topk_ranks=[r['rank'] for r in available if not r['retained_topk']],
                topk_first_loss_counts=dict(Counter(r['first_loss'] or 'nominal_available' for r in top)),
                counterfactual_all_first_loss_counts=dict(Counter(r['first_loss'] or 'nominal_available' for r in rows)))
            report['seeds'].append(seedrow)
            allrows.extend(dict(seed=seed,**r) for r in rows)
        report['seed_acquisition']=dict(distinct_seeds=len(report['seeds']),
            with_topk_nominal_candidate=sum(r['nominal_available_topk']>0 for r in report['seeds']),
            with_any_generated_nominal_candidate=sum(r['nominal_available_all']>0 for r in report['seeds']))
        report['status']='completed'
        csv_rows(a.out/'summary.csv',report['seeds']);csv_rows(a.out/'all_candidate_stages.csv',allrows)
        report['input_reports_sha256']={str(path.resolve()):digest(path) for path in
            [a.inference_root/'summary.json',a.bundle/'manifest.json',a.replay/'summary.json',*a.trace_root.glob('seed_*/*/summary.json')]}
        print(report['seed_acquisition'])
        for row in report['seeds']:print(row)
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');print(report['error'],file=sys.stderr);code=1
    finally:
        write(a.out/'summary.json',report)
    return code


if __name__=='__main__':raise SystemExit(main())
