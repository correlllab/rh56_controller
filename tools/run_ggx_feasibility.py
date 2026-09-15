#!/usr/bin/env python3
"""Recheck a saved GGX batch: official scene filter + fixed CB3, NEVER hardware.

Does not re-infer, rerank, request approval or execute. Saves separate stage
counts and feasibility.csv; an optional video is only an offline diagnostic.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_ggx_review as ui


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-run',type=Path,required=True,help='Saved review/initial_ggx directory with inference and simulation.yaml')
    p.add_argument('--official-filter',type=Path,help='Reuse matching completed filter summary; never refit or rerank')
    p.add_argument('--scene-collision-mm',type=float,default=20.)
    p.add_argument('--ggx-python',type=Path,default=ROOT/'artifacts/ggx_lab/GraspGenX/.venv/bin/python')
    p.add_argument('--real-config',type=Path,default=ROOT/'configs/ggx_real_lab.yaml')
    p.add_argument('--max-candidates',type=int,default=100)
    p.add_argument('--candidate-rank',type=int,help='Offline diagnostic: inspect only this ORIGINAL rank, never rerank')
    p.add_argument('--candidate-scope',choices=['grasp','legacy-path'],default='grasp',
                   help='grasp: start-independent selection; legacy-path: reproduce old scan-route screen')
    p.add_argument('--video',action='store_true')
    p.add_argument('--out',type=Path)
    a=p.parse_args(argv)
    import math
    if not math.isfinite(a.scene_collision_mm) or not 0<a.scene_collision_mm<=100 or not 1<=a.max_candidates<=100:
        p.error('Invalid threshold/candidate count')
    if a.candidate_rank is not None and not 0<=a.candidate_rank<a.max_candidates:
        p.error('candidate-rank must be within max-candidates')
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.absolute() if k=='ggx_python' else v.resolve())
    return a


def main(argv=None):
    a=parse_args(argv)
    out=a.out or ROOT/'artifacts/ggx_feasibility_audits'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='rh56_ggx_feasibility_audit/v1',status='running',executable=False,
        robot_commands_sent=False,hardware_contacted=False,selected_rank=None,reviews=[],
        assumptions=['Offline saved batch only; no new GGX inference, rank changes or hardware access.',
            'Official endpoint scene decision and local CB3 path/target uncertainty reported separately.',
            'A pass is not physical success; uncertain candidates require independent review, never automatic execution.'])
    code=0
    try:
        from tools import run_ggx_official_filter as official,ggx_cb3_candidates as cb3
        a.inference_report=a.review_run/'inference/summary.json'
        inf,candidates,scene,mask,stats,pins=official.prepare_inputs(a.inference_report)
        crop=ui.read_json(Path(inf['source_point_cloud']).parent/'summary.json')
        perception=ui.read_json(Path(crop['source_preview']))
        a.config=Path(perception['config'])
        a.execute=False
        a.render_candidates=a.video
        config_path=a.review_run/'simulation.yaml'
        pins.update(ui.pin([config_path]))
        if a.official_filter:
            result=cb3.load_filter(a.official_filter,candidates)
            if result['threshold_mm']!=a.scene_collision_mm:
                raise ValueError('Saved official threshold differs from declared protocol')
        for rank,video,checks in cb3.simulate(a,out,report,config_path,pins,filter_path=a.official_filter):
            report.setdefault('eligible_ranks',[]).append(rank)
        if report['status']=='running':report['status']='audit_completed'
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');code=1
        print(report['error'],file=sys.stderr)
    finally:
        ui.save_report(out,report)
        print(f"GGX offline audit: {report['status']}; {out/'summary.json'}",flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
