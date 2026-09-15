"""Offline video preparation for an unchanged, official-selected GGX candidate.

A strict local check remains strict. A target-proxy-only rejection may produce
a clearly labelled local diagnostic video, never a prepared motion approval.
No camera, serial, RTDE connection, or robot backend is constructed here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import os
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SCHEMA = 'rh56_ggx_quick_local_preview/v1'


def _target_only(detail):
    failures = detail.get('local_failures', [])
    return (detail.get('state') == 'local_approach_unresolved'
            and detail.get('local_execution_check', {}).get('passed') is False
            and detail.get('target_proxy_uncertain') is True
            and bool(failures)
            and all(item.get('kind') == 'target_proxy_uncertain' for item in failures))


def prepare_preview(source, snapshot, out):
    """Return/write a non-executable report; preserve every strict rejection.

    ``out`` must be new or empty. The first attempt uses the unmodified actual
    saved-start preparation. Only a target-only local-geometry rejection may
    fall back to recovering the original local path using the existing fixed
    policy once. That second path does not begin at ``snapshot``.
    """
    import numpy as np
    from tools import ggx_planning as planning, ggx_real_support as support
    from tools import preflight_ggx_ur5_table as sim
    from tools.ggx_template import sim_args, write_json
    from tools.run_ggx_real import load_config
    from tools.run_ggx_ten_starts import code_pins

    if source.get('selection_scope') != support.OFFICIAL_SELECTION_SCOPE:
        raise ValueError('Quick local preview requires an official-only candidate source')
    support.verify(source['pins'])
    q = np.asarray(snapshot['q'], dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError('Expected finite saved six-joint start snapshot')
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Quick local preview output must be new or empty')
    report = dict(schema=SCHEMA, status='running', rank=source['rank'],
        executable=False, usable_for_motion=False, hardware_contacted=False,
        motion_authorized=False, physical_hand_verified=False,
        official_selection_unchanged=True, source_sha256=source['pins'],
        code_config_sha256={**code_pins(Path(source['checker_config'])), **support.pin([__file__])},
        policy=planning.policy(), strict_check_failed=None, video=None,
        path_scope=None, sample_recovery_attempts=0,
        assumptions=[
            'Original candidate/rank, local model, target proxy, thresholds and policy are unchanged.',
            'Strict preparation includes the saved start, not a currently observed real robot pose.',
            'A target-contact warning video shows only local pregrasp/approach/close/lift, not travel from the saved start.',
            'Closure/lift are kinematic previews; a static proxy is not a simulated successful pickup.',
            'Inherited nominal checks are not a dynamic, self-collision, or real-controller certification.',
            'Neither result authorizes hardware or verifies physical hand mapping.'])
    summary = out / 'summary.json'
    write_json(summary, report)

    try:
        prepared, _ = support.prepare(source, snapshot, out, video=True)
    except planning.PlanningUnavailable as exc:
        report.update(strict_check_failed=True, strict_reason=exc.reason,
                      strict_attempts=exc.attempts, status='local_preview_unavailable')
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(summary, report)
        raise
    else:
        video = out / 'cb3_review.mp4'
        if not video.is_file():
            raise RuntimeError('Strict preparation completed without its requested video')
        report.update(status='strict_nominal_path_ready', strict_check_failed=False,
            video=str(video), path_scope='saved_start_to_grasp_lift',
            actual_travel_checked=True, prepared_report=str(out / 'prepared.json'),
            minimum_desk_clearance_mm=prepared['minimum_desk_clearance_mm'])
        support.verify(report['code_config_sha256'])
        write_json(summary, report)
        return report

    # The strict API deliberately does not return rejected trajectory samples.
    # Read its retained attribution before repeating only its local fixed-policy
    # search. Do not downgrade IK, workspace, desk, or actual-travel failures.
    assessment_path = out / 'local_assessment.json'
    if assessment_path.is_file():
        assessment = json.loads(assessment_path.read_text())
        report['strict_local_assessment'] = str(assessment_path)
        detail = assessment.get('geometry', {})
    else:
        detail = {}
    if (report['strict_reason'] != 'selected_official_candidate_local_geometry_unavailable'
            or not _target_only(detail)):
        write_json(summary, report)
        return report

    support.verify(source['pins'])
    model = support.build_model(source)
    args = sim_args(source)
    args.desk_clearance_mm = 2.  # Same as prepare_official_candidate, not a preview tolerance.
    config = load_config(Path(source['checker_config']))
    center = np.asarray(source['preflight']['object_center_m'])
    candidate = sim.load_isaac_grasp_yaml(source['preflight']['candidate_yaml'])[source['rank']]
    candidate = sim.candidate_for_object_origin(candidate, 'ur-base', center)
    report['sample_recovery_attempts'] = 1
    recovered, samples = planning.select_geometry(model, candidate, source['rank'], args, center, config)
    report['recovered_geometry'] = recovered
    if (not _target_only(recovered)
            or recovered.get('standoff_mm') != detail.get('standoff_mm')):
        report['preview_unavailable_reason'] = 'fixed_policy_recovery_changed_or_not_target_only'
        write_json(summary, report)
        return report

    failures = planning.geometry_failures(model, samples, config, target_gate=True)
    complete = (bool(samples)
        and samples[0].phase == 'start'
        and samples[-1].phase == 'lift'
        and abs(samples[-1].phase_alpha - 1.) < 1e-9
        and all(any(s.phase == phase for s in samples) for phase in ('approach', 'close', 'lift'))
        and all(np.all(np.isfinite(s.qpos)) for s in samples))
    result_data = recovered.get('result', {})
    if (not complete or result_data.get('reason') != 'object_collision_before_closure'
            or not failures or any(f['kind'] != 'target_proxy_uncertain' for f in failures)):
        report.update(preview_unavailable_reason='incomplete_or_non_target_local_failure',
                      recovered_failures=failures)
        write_json(summary, report)
        return report

    # The existing renderer places an explicit local-only/no-motion banner on
    # every frame for GEOMETRY ONLY results. Keep passed=False and the negative
    # target distances; do not hide the proxy or alter model contact settings.
    result = sim.CandidateResult(**result_data)
    result.passed = False
    result.reason = 'GEOMETRY ONLY - TARGET CONTACT WARNING - LOCAL ONLY'
    video = out / 'local_target_contact_warning.mp4'
    sim.write_video(video, model, samples, result, args, center)
    report.update(status='target_contact_warning_preview', video=str(video),
        path_scope='local_only', actual_travel_checked=False,
        strict_local_target_clearance_passed=False, target_contact_warning=True,
        result=asdict(result),
        minimum_desk_clearance_mm=min(s.min_desk_clearance_m for s in samples) * 1000.,
        warning='Target proxy overlap remains unresolved. This local-only video is not a saved-start execution path or motion approval.')
    support.verify(source['pins'])
    support.verify(report['code_config_sha256'])
    write_json(summary, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approval', type=Path, required=True,
                        help='Saved official-only selection; never a motion approval')
    parser.add_argument('--snapshot', type=Path, required=True,
                        help='Saved robot start JSON; no live robot connection')
    parser.add_argument('--out', type=Path, required=True, help='New or empty artifact directory')
    args = parser.parse_args(argv)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    from tools import ggx_real_support as support
    try:
        source = support.read_approval(args.approval.resolve())
        report = prepare_preview(source, json.loads(args.snapshot.read_text()), args.out)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    print(f"GGX preview: {report['status']}; {args.out.resolve() / 'summary.json'}")
    if report['video']:
        print(report['video'])
    print('Offline only; no motion authorized. Target warnings are not passes.')
    return 0 if report['video'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
