#!/usr/bin/env python3
"""Draw confidence-ranked GGX open-hand candidates over saved camera data.

Official descriptor point cloud, not iterative Plan or executable hand commands.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inference',type=Path,required=True,help='Point-cloud inference summary.json')
    p.add_argument('--crop',type=Path,required=True,help='Measured crop summary.json')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--topk',type=int,default=3)
    a=p.parse_args(argv)
    if not 1<=a.topk<=10:p.error('topk must be 1..10')
    a.out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='rh56_ggx_camera_candidate_preview/v1',status='starting',
        hardware_contacted=False,executable_target=False,candidates=[],
        assumptions=['Confidence-ranked poses, not collision-filtered or approved for execution.',
                     'Official descriptor OPEN-hand points transformed in UR base; no iterative correction.',
                     'Camera overlay is projection, not occlusion-aware rendering or contact verification.',
                     'Yellow cross is the descriptor sweep-volume center, not a measured contact.',
                     '3D view uses partial observed object points only; hidden geometry remains unknown.'])
    rc=0
    try:
        import cv2
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from rh56_controller.graspgenx_baseline import load_isaac_grasp_yaml
        from tools.preview_tabletop_grasp_center import project_base_mm
        inf=json.loads(a.inference.read_text());crop=json.loads(a.crop.read_text())
        if (inf.get('schema')!='rh56_ggx_point_cloud_inference/v1' or inf.get('input_frame')!='ur_base'
                or inf.get('status')!='candidates_generated_not_motion_validated'
                or crop.get('schema')!='rh56_ggx_measured_crop/v1' or crop.get('frame')!='ur_base'):
            raise ValueError('Expected UR-base measured crop and successful GGX inference')
        cloud_path=Path(crop['point_cloud'])
        if hashlib.sha256(cloud_path.read_bytes()).hexdigest()!=inf['source_point_cloud_sha256']:
            raise ValueError('Input cloud differs from inference')
        points=np.load(cloud_path,allow_pickle=False)
        colors=np.load(cloud_path.parent/'object_colors_rgb.npy',allow_pickle=False)/255.
        source=json.loads(Path(crop['source_preview']).read_text())
        if hashlib.sha256(Path(crop['source_preview']).read_bytes()).hexdigest()!=crop['source_preview_sha256']:
            raise ValueError('Reviewed perception report changed since crop')
        if hashlib.sha256(Path(source['calibration']).read_bytes()).hexdigest()!=crop['calibration_sha256']:
            raise ValueError('Calibration changed since crop')
        cal=yaml.safe_load(Path(source['calibration']).read_text())
        rgb=cv2.imread(str(Path(source['capture'])/'camera_check_rgb.png'))
        descriptor=Path(inf['checkpoint_root']).parents[2]/'ext/gripper_descriptions/gripper_descriptions/assets/x_grippers/inspire_hand'
        hand_file=descriptor/'points.json';hand=np.asarray(json.loads(hand_file.read_text())['open'])[::7]
        config=json.loads((descriptor/'config.json').read_text())
        report['open_hand_points_sha256']=hashlib.sha256(hand_file.read_bytes()).hexdigest()
        if hashlib.sha256(Path(inf['candidate_file']).read_bytes()).hexdigest()!=inf['candidate_sha256']:
            raise ValueError('Candidate file changed since inference')
        candidates=load_isaac_grasp_yaml(inf['candidate_file'])[:a.topk]
        fig=plt.figure(figsize=(5*len(candidates),5))
        for i,c in enumerate(candidates):
            world=hand@c.rotation.T+c.position
            sweep=c.position+c.rotation@np.asarray(config['sweep_volume']['offset'])
            ax=fig.add_subplot(1,len(candidates),i+1,projection='3d')
            ax.scatter(*(points[::4]*1000).T,c=colors[::4],s=2,alpha=.8)
            ax.scatter(*(world*1000).T,c='#0077bb',s=1,alpha=.32)
            ax.scatter(*(sweep*1000),c='red',s=35,marker='+')
            ax.quiver(*(c.position*1000),*(c.approach_direction*70),color='red',arrow_length_ratio=.15)
            allp=np.vstack([points,world])*1000;mid=(allp.min(axis=0)+allp.max(axis=0))/2
            radius=np.max(np.ptp(allp,axis=0))/2+10
            ax.set(xlim=(mid[0]-radius,mid[0]+radius),ylim=(mid[1]-radius,mid[1]+radius),zlim=(mid[2]-radius,mid[2]+radius),
                   xlabel='Base X (mm)',ylabel='Base Y (mm)',zlabel='Base Z (mm)',title=f'Rank {i}: score {c.confidence:.3f}')
            ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=-65)
            overlay=rgb.copy();uv=project_base_mm(world*1000,cal)
            valid=np.all(np.isfinite(uv),axis=1)&np.all(uv>=0,axis=1)&np.all(uv<[rgb.shape[1],rgb.shape[0]],axis=1)
            for x,y in np.rint(uv[valid]).astype(int):cv2.circle(overlay,(x,y),1,(255,170,0),-1)
            mark=project_base_mm([sweep*1000],cal)[0]
            if np.all(np.isfinite(mark)) and np.all(mark>=0) and np.all(mark<[rgb.shape[1],rgb.shape[0]]):
                cv2.drawMarker(overlay,tuple(np.rint(mark).astype(int)),(0,255,255),cv2.MARKER_CROSS,24,2)
            cv2.putText(overlay,f'GGX rank {i} score {c.confidence:.3f} | OPEN hand projection | NOT EXECUTABLE',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.62,(0,255,255),2)
            cv2.imwrite(str(a.out/f'rank_{i:02}_overlay.png'),overlay)
            report['candidates'].append(dict(rank=i,confidence=c.confidence,root_position_base_m=c.position.tolist(),
                approach_axis_base=c.approach_direction.tolist(),sweep_center_base_m=sweep.tolist(),
                open_hand_point_min_base_z_mm=float(world[:,2].min()*1000)))
        fig.suptitle('GGX official open-hand points + measured partial object | no collision / motion validation')
        fig.tight_layout();fig.savefig(a.out/'top_candidates_3d.png',dpi=150);plt.close(fig)
        report['status']='offline_candidates_visualized'
    except Exception as e:
        report.update(status='failed',error=f'{type(e).__name__}: {e}');rc=1
    finally:
        (a.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        with (a.out/'summary.csv').open('w',newline='') as f:
            keys=['status','hardware_contacted','executable_target'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerow({k:report[k] for k in keys})
        print(json.dumps(report,indent=2),flush=True)
    return rc


if __name__=='__main__':raise SystemExit(main())
