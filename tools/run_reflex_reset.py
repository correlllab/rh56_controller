#!/usr/bin/env python3
"""Open/retreat/return to the saved scan pose only; no capture or grasp.

Default records no action. --execute and the existing reset Enter confirmation
are both required. Reuses the ten-trial reset, including its path checks/logs.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_ggx_experiment as experiment, ggx_template as files


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/reflex_real_lab.yaml')
    p.add_argument('--perception-config',type=Path,default=ROOT/'configs/tabletop_vision_lab.yaml')
    p.add_argument('--start-snapshot',type=Path,default=ROOT/'configs/reflex_scan_wrist_flip.json')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--allow-high-force',action='store_true')
    p.add_argument('--out',type=Path)
    args=p.parse_args(argv)
    for k,v in vars(args).items():
        if isinstance(v,Path): setattr(args,k,v.resolve())
    return args


def main(argv=None):
    args=parse_args(argv)
    out=args.out or ROOT/'artifacts/reflex_reset_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if not args.execute:
        out.mkdir(parents=True,exist_ok=False)
        files.write_json(out/'summary.json',dict(schema='rh56_reflex_reset_request/v1',
            status='not_executed',hardware_contacted=False,
            assumptions=['No device access without --execute; live reset also requires Enter.',
                         'Reset only: no camera, template, perception or grasp.']))
        print(f'未连接硬件。复位需要 --execute 和现场 Enter 确认；{out / "summary.json"}')
        return 0
    return experiment.reset_robot(args,out)[0]


if __name__=='__main__':
    raise SystemExit(main())
