#!/usr/bin/env python3
"""Capture a tabletop RGB-D burst; camera only, no robot/hand connection.

Save raw frames, a valid-depth temporal median and per-pixel stability statistics.
Empty-table status still requires visual review; this is not camera calibration.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Shared with the sugar trial: 30-frame startup produced fragmented box depth;
# retain the 60-frame warmup used for the reviewed tabletop reference captures.
TABLETOP_WARMUP_FRAMES = 60


def summarize_depth(stack, scale_m):
    values = np.asarray(stack)
    if values.ndim != 3 or values.dtype != np.uint16 or len(values) < 3:
        raise ValueError('Expected at least three uint16 depth frames')
    if not np.isfinite(scale_m) or scale_m <= 0:
        raise ValueError('Depth scale must be positive and finite')
    valid = values > 0
    count = valid.sum(axis=0)
    masked = np.where(valid, values.astype(float), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)  # All-invalid pixels handled explicitly.
        median = np.nanmedian(masked, axis=0)
        mad_mm = np.nanmedian(np.abs(masked-median), axis=0)*scale_m*1000
    supported = count >= np.ceil(.8*len(values))
    depth = np.where(supported, np.nan_to_num(median), 0).round().astype(np.uint16)
    return depth, mad_mm, count


def depth_quality(depth, mad, roi, *, required=True):
    x0, y0, x1, y1 = roi
    supported = depth[y0:y1, x0:x1] > 0
    samples = mad[y0:y1, x0:x1][supported]
    if not len(samples) and required:
        raise RuntimeError('No sufficiently supported background depth in ROI')
    return dict(roi_supported_depth_fraction=float(np.mean(supported)),
                roi_temporal_mad_median_mm=float(np.median(samples)) if len(samples) else None,
                roi_temporal_mad_p95_mm=float(np.percentile(samples, 95)) if len(samples) else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New output directory; never overwritten')
    parser.add_argument('--camera-serial', required=True)
    parser.add_argument('--frames', type=int, default=9)
    parser.add_argument('--scene', choices=['empty-table', 'object'], default='empty-table',
                        help='Label the operator-prepared scene; no automatic scene classification')
    parser.add_argument('--roi', nargs=4, type=int, default=[250,50,1150,680])
    parser.add_argument('--rgb-only-quality', action='store_true',
                        help='Object XY tracking only: retain RGB-D recording, but do not require valid depth')
    args = parser.parse_args(argv)
    if args.rgb_only_quality and args.scene != 'object':
        parser.error('--rgb-only-quality requires --scene object; background/GGX still need depth')
    if not 3 <= args.frames <= 60:
        parser.error('--frames must be in [3, 60]')
    x0,y0,x1,y1 = args.roi
    if not 0 <= x0 < x1 <= 1280 or not 0 <= y0 < y1 <= 720:
        parser.error('ROI must lie within 1280x720')
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='rh56_tabletop_background/v1' if args.scene == 'empty-table' else 'rh56_tabletop_object_burst/v1',
                  status='starting', scene=args.scene,
                  camera_serial=args.camera_serial, roi=args.roi, requested_frames=args.frames,
                  captures=[], ur_contacted=False, hand_contacted=False, hardware_motion_commands_sent=False,
                  valid_depth_required=not args.rgb_only_quality,
                  visually_verified_empty=False if args.scene == 'empty-table' else None,
                  assumptions=[f'Operator requested {args.scene} capture; visual review still required.',
                               'Camera, desk and mat remain fixed; this does not update calibration.',
                               'Temporal median requires valid depth in at least 80% of frames.',
                               'MAD measures short-term repeatability, not absolute depth accuracy.',
                               'Current camera emitter/laser settings are read and retained, not changed.'])
    if args.rgb_only_quality:
        report['assumptions'].append('RGB XY tracking capture: depth recorded but not required; not valid GGX input by itself.')
    pipeline = None
    result = 0
    try:
        import yaml
        from tools import calibrate_ur5_external_camera_real as camera
        options = camera.parse_args(['--check-only','--camera-serial',args.camera_serial,
                                     '--warmup-frames',str(TABLETOP_WARMUP_FRAMES),'--settle-frames','5'])
        pipeline, align, metadata = camera._start_camera(options)
        frames = []
        middle_rgb = None
        for index in range(args.frames):
            rgb, depth, timestamp, number = camera._read_aligned_frame(pipeline, align, 5)
            capture = args.out / f'capture_{index:02d}'
            capture.mkdir()
            camera._save_rgb(capture/'camera_check_rgb.png', rgb)
            camera._save_depth(capture/'camera_check_depth_u16.png', depth)
            payload = dict(schema='rh56_realsense_camera_check/v1',
                           camera=camera._camera_metadata_to_dict(metadata), host_time_s=time.time(),
                           camera_timestamp_ms=timestamp, frame_number=number,
                           safety=dict(ur_contacted=False, hardware_motion_commands_sent=False))
            (capture/'camera_check.yaml').write_text(yaml.safe_dump(payload, sort_keys=False))
            report['captures'].append(dict(directory=capture.name, frame_number=number,
                                           camera_timestamp_ms=timestamp))
            frames.append(depth)
            if index == args.frames//2:
                middle_rgb = rgb
            print(f'RGB-D frame {index+1}/{args.frames} saved ({args.scene})', flush=True)
        pipeline.stop()
        pipeline = None
        median, mad, count = summarize_depth(np.stack(frames), metadata.depth_scale_m)
        camera._save_rgb(args.out/'camera_check_rgb.png', middle_rgb)
        camera._save_depth(args.out/'camera_check_depth_u16.png', median)
        camera._save_rgb(args.out/'camera_check_depth_preview.png', camera._depth_preview(median))
        np.savez_compressed(args.out/'depth_statistics.npz', mad_mm=mad.astype(np.float32),
                            valid_count=count.astype(np.uint16))
        payload.update(depth_aggregation='temporal median; RGB is the middle frame',
                       frame_number=None, camera_timestamp_ms=None,
                       source_frame_numbers=[c['frame_number'] for c in report['captures']])
        (args.out/'camera_check.yaml').write_text(yaml.safe_dump(payload, sort_keys=False))
        report.update(status='captured_needs_visual_review',
                      **depth_quality(median, mad, args.roi, required=not args.rgb_only_quality),
                      background_depth='camera_check_depth_u16.png',
                      camera=camera._camera_metadata_to_dict(metadata))
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        result = 1
    except KeyboardInterrupt:
        report.update(status='interrupted')
        result = 130
    finally:
        try:
            if pipeline is not None:
                pipeline.stop()
        finally:
            (args.out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            with (args.out/'summary.csv').open('w', newline='') as stream:
                keys = ['status','requested_frames','ur_contacted','hand_contacted',
                        'hardware_motion_commands_sent','roi_supported_depth_fraction',
                        'roi_temporal_mad_median_mm','roi_temporal_mad_p95_mm']
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerow({k:report.get(k) for k in keys})
            print(json.dumps(report, indent=2), flush=True)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
