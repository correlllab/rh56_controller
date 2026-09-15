#!/usr/bin/env python3
"""Offline attribution of frozen top-100 target-proxy rejections; no hardware.

Select the highest original rank in each recorded contact-location category.
Replay unchanged frozen code, separate endpoint from approach, and record vertex
witnesses independently of mj_geomDistance. Never changes an acceptance gate.
"""
import argparse
from pathlib import Path
import sys


def representatives(rows, details):
    groups = {}
    for row in sorted(rows, key=lambda r: r['rank']):
        if row['first_loss'] != 'local_approach_unresolved':
            continue
        failures = details[row['rank']]['geometry']['local_failures']
        if not failures or any(f['kind'] != 'target_proxy_uncertain' for f in failures):
            raise ValueError('Unexpected rejection mix: define attribution categories explicitly')
        label = failures[0]['geom'].split('/')[0]
        category = ('pregrasp_overlap' if any(f['phase'] != 'approach' for f in failures)
                    else 'thumb' if label.startswith('thumb')
                    else 'attachment' if label == 'gripper_attachment' else 'other_finger')
        groups.setdefault(category, []).append(row['rank'])
    return groups


def point_in_box_witness(vertices, position, rotation, size):
    """Positive margin: an exact mesh/box vertex is inside the proxy, in metres."""
    import numpy as np
    margins = np.min(size - np.abs((vertices - position) @ rotation), axis=1)
    index = int(np.argmax(margins))
    return dict(max_vertex_inside_margin_mm=float(margins[index] * 1000),
                inside_vertex_count=int(np.sum(margins > 1e-7)),
                witness_world_m=vertices[index].tolist())


def observed_points_in_convex_geom(points, vertices):
    """Measured surface inside existing convex geom, not proof of real contact."""
    import numpy as np
    from scipy.spatial import ConvexHull
    planes = ConvexHull(vertices).equations
    margins = np.concatenate([-np.max(chunk @ planes[:, :3].T + planes[:, 3], axis=1)
                             for chunk in np.array_split(points, max(1, len(points)//256))])
    return dict(observed_points_inside_geom=int(np.sum(margins > 1e-7)),
                deepest_observed_point_margin_mm=float(max(0., margins.max()) * 1000))


def common_interior_witness(first, second):
    """Diagnostic only: positive verified interior radius proves model overlap.

    Failed/zero LP results do NOT prove separation. This function is not called
    by any planning or execution gate. Recheck actual inequalities at the point
    rather than accepting the solver status or its objective as a certificate.
    """
    import numpy as np
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull
    planes = np.concatenate([ConvexHull(v).equations for v in (first, second)])
    planes = planes / np.linalg.norm(planes[:, :3], axis=1)[:, None]
    solution = linprog([0., 0., 0., -1.], A_ub=np.c_[planes[:, :3], np.ones(len(planes))],
                       b_ub=-planes[:, 3], bounds=[(None, None)]*3+[(0., None)], method='highs')
    result = dict(solver_status=int(solution.status), positive_interior_witness=False)
    if solution.success:
        point = solution.x[:3]
        margin = float(np.min(-(planes[:, :3] @ point + planes[:, 3])))
        result.update(positive_interior_witness=margin > 1e-7,
                      verified_interior_radius_mm=max(0., margin-1e-7)*1000,
                      witness_world_m=point.tolist())
    return result


def plot_endpoint(path, cloud, vertices, box_position, box_rotation, box_size, rank, geom):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    observed = (cloud-box_position) @ box_rotation*1000
    shape = (vertices-box_position) @ box_rotation*1000
    size = box_size*1000
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (i,j) in zip(axes, [(0,1),(0,2),(1,2)]):
        ax.add_patch(Rectangle((-size[i],-size[j]),2*size[i],2*size[j],
                               fill=False, edgecolor='black', linestyle='--', label='Unchanged box proxy'))
        ax.scatter(observed[::3,i], observed[::3,j], s=1, c='gray', alpha=.35, label='Observed target surface')
        ax.scatter(shape[:,i], shape[:,j], s=2, c='tab:orange', alpha=.6, label='Selected local collision geom')
        ax.set(xlabel=f'Object-local {"XYZ"[i]} (mm)', ylabel=f'Object-local {"XYZ"[j]} (mm)')
        ax.set_aspect('equal'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(f'Rank {rank}: approach endpoint / {geom}\nProjections only; missing depth is not free space')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--inference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    args.bundle = args.bundle.resolve()
    sys.path.insert(0, str(args.bundle/'files'))
    import numpy as np
    import mujoco
    from tools.run_ggx_frozen_audit import context, local_check, exported_candidates, read, write, digest, csv_rows
    from tools import ggx_motion as motion, preflight_ggx_ur5_table as sim
    inputs, model, simargs, center = context(args.bundle)
    batch = read(args.batch/'summary.json')
    if batch['status'] != 'completed' or sorted(r['rank'] for r in batch['rows']) != list(range(100)):
        raise ValueError('Requires complete original top-100 batch')
    if inputs['topk'] != 100:
        raise ValueError('Protocol is frozen at top-100')
    inf = read(args.inference/'summary.json')
    if inf['source_point_cloud_sha256'] != digest(args.bundle/inputs['inference_point_cloud']):
        raise ValueError('Inference cloud is not the frozen scene')
    if inf['pre_topk_archive']['raw_sha256'] != digest(args.inference/'all_candidates.npz'):
        raise ValueError('Raw candidate archive changed')
    candidates, _ = exported_candidates(args.inference)
    archive = np.load(args.inference/'all_candidates.npz', allow_pickle=False)
    for row in batch['rows']:
        rank = row['rank']; raw_id = int(archive['rank_to_generation_id'][rank])
        if row['generation_id'] != raw_id or row['confidence'] != candidates[rank].confidence:
            raise ValueError('Candidate lineage changed')
    details = {r['rank']: read(args.batch/f"rank_{r['rank']:04}/local.json") for r in batch['rows']
               if r['first_loss'] == 'local_approach_unresolved'}
    groups = representatives(batch['rows'], details)
    args.out.mkdir(parents=True, exist_ok=False)
    cloud = np.load(args.bundle/inputs['inference_point_cloud'])
    obj = model.geom('preflight_object_visual').id
    if model.geom_type[obj] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError('This frozen diagnostic requires a box proxy; no model substitution')
    labels = {sim._geom_label(model, i): i for i in range(model.ngeom)}
    report = dict(schema='rh56_ggx_rejection_attribution/v1', hardware_contacted=False, executable=False,
                  status='running', original_topk=100, groups=groups, representatives=[],
                  selection_rule='highest original rank per first recorded body category; pregrasp contact takes priority',
                  source_sha256={str(p): digest(p) for p in [args.bundle/'manifest.json', args.batch/'summary.json',
                      args.inference/'all_candidates.npz', args.inference/'all_candidates.yml', Path(__file__)]},
                  assumptions=['Frozen code/model/inputs; no reranking, new inference or alternate path search.',
                      'Positive inside-vertex margin independently proves overlap with the MODEL proxy, not the real object.',
                      'Point cloud is a partial noisy visible surface, not a complete volume or free-space certificate.',
                      'Endpoint denotes the final approach sample before closure, not an evaluated grasp outcome.'])
    write(args.out/'summary.json', report)
    for category, ranks in groups.items():
        rank = ranks[0]
        detail, samples, local = local_check(model, candidates[rank], rank, simargs, center, inputs['config'])
        if detail != details[rank]['geometry']:
            raise ValueError(f'Frozen replay differs for rank {rank}')
        rows = []; data = mujoco.MjData(model)
        for index, sample in enumerate(samples):
            if sample.phase not in motion.OBJECT_PHASES:
                continue
            data.qpos[:] = sample.qpos; data.qvel[:] = 0.; mujoco.mj_forward(model, data)
            gap, label = motion.object_clearance(model, data); geom = labels[label]
            row = dict(index=index, phase=sample.phase, clearance_mm=float(gap*1000), geom=label,
                       hand_closure_alpha=sample.closure_alpha,
                       pair_contacts_mm=[float(c.dist*1000) for c in data.contact
                                         if {int(c.geom1), int(c.geom2)} == {obj, geom}])
            vertices = sim._world_convex_vertices(model, data, geom)
            if vertices is not None:
                row.update(point_in_box_witness(vertices, data.geom_xpos[obj],
                                                data.geom_xmat[obj].reshape(3, 3), model.geom_size[obj]))
            rows.append(row)
        first = next(r for r in rows if r['clearance_mm'] <= 1e-6)
        worst = min(rows, key=lambda r: r['clearance_mm'])
        endpoint = [r for r in rows if r['phase'] == 'approach'][-1]
        for row in (worst, endpoint):
            data.qpos[:] = samples[row['index']].qpos; mujoco.mj_forward(model, data)
            vertices = sim._world_convex_vertices(model, data, labels[row['geom']])
            if vertices is not None:
                row.update(observed_points_in_convex_geom(cloud, vertices))
                witness = common_interior_witness(vertices, sim._world_convex_vertices(model, data, obj))
                if witness['positive_interior_witness']:
                    # Separate direct box-coordinate membership check.
                    direct = point_in_box_witness(np.array([witness['witness_world_m']]),
                        data.geom_xpos[obj], data.geom_xmat[obj].reshape(3,3), model.geom_size[obj])
                    if not direct['inside_vertex_count']:
                        raise ValueError('Interior witness fails direct box membership')
                row['model_common_interior'] = witness
        data.qpos[:] = samples[endpoint['index']].qpos; mujoco.mj_forward(model, data)
        vertices = sim._world_convex_vertices(model, data, labels[endpoint['geom']])
        if vertices is not None:
            plot_endpoint(args.out/f'rank_{rank:04}_endpoint.png', cloud, vertices, data.geom_xpos[obj],
                          data.geom_xmat[obj].reshape(3,3), model.geom_size[obj], rank, endpoint['geom'])
        result = dict(rank=rank, category=category, sample_count=len(samples),
                      saved_local_reproduced=True, first=first, worst=worst, endpoint=endpoint,
                      terminal_query_rejected=endpoint['clearance_mm'] <= 1e-6,
                      approach_only_query_rejected=endpoint['clearance_mm'] > 1e-6,
                      samples=rows)
        write(args.out/f'rank_{rank:04}.json', result)
        np.savez(args.out/f'rank_{rank:04}_samples.npz', qpos=[s.qpos for s in samples], phases=[s.phase for s in samples])
        report['representatives'].append({k:v for k,v in result.items() if k != 'samples'})
        write(args.out/'summary.json', report)
        print(f"rank {rank}: first={first['phase']}, endpoint={endpoint['clearance_mm']:.3f} mm, "
              f"worst={worst['clearance_mm']:.3f} mm", flush=True)
    report['status'] = 'completed'
    report['all_proxy_rejections_have_negative_query'] = all(
        any(f.get('clearance_mm', 0) < 0 for f in d['geometry']['local_failures']) for d in details.values())
    write(args.out/'summary.json', report)
    csv_rows(args.out/'summary.csv', [dict(rank=r['rank'], category=r['category'],
        terminal_query_rejected=r['terminal_query_rejected'], first_phase=r['first']['phase'],
        endpoint_clearance_mm=r['endpoint']['clearance_mm'], worst_clearance_mm=r['worst']['clearance_mm'],
        witness_inside_margin_mm=r['worst'].get('max_vertex_inside_margin_mm'),
        observed_points_inside_worst_geom=r['worst'].get('observed_points_inside_geom')) for r in report['representatives']])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
