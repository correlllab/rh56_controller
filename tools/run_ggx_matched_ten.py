#!/usr/bin/env python3
"""GGX frozen top-100, first feasible candidate per saved Reflex start.

Default: offline generation/screening/ten-route review. --validate-only only
checks shared inputs. Devices require --execute --capture-live, reset Enter,
RTDE-only authorization and checked-video Enter. Never retry after motion.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import ggx_matched_protocol as protocol, run_ggx_review as ui
from tools import run_ggx_real as real, ggx_template as template, ggx_xy_tracking as xy
from tools import run_ggx_experiment as experiment, run_ggx_ten_starts as ten
from tools import ggx_planning as planning

SCHEMA = 'rh56_ggx_matched_pool_ten/v1'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--reflex-run', type=Path, help='Recorded Reflex Cartesian starts/cloud/settings; common new wrist seed and scan for all objects')
    source.add_argument('--resume-from', type=Path, help='Ended matched GGX summary; no retry of partly executed trials')
    pool = p.add_mutually_exclusive_group()
    pool.add_argument('--pool-report', type=Path, help='Reuse complete, current, same-cloud top-100 screening')
    pool.add_argument('--candidate-run', type=Path, help='Reuse same-cloud inference; new full screening, no inference')
    p.add_argument('--official-filter', type=Path, help='Reuse matching official filter with --candidate-run')
    p.add_argument('--ggx-python', type=Path, default=ROOT/'artifacts/ggx_lab/GraspGenX/.venv/bin/python')
    p.add_argument('--seed', type=int, default=20260909)
    p.add_argument('--resume-after-reset-fix', action='store_true',
                   help='Explicit migration from the pinned pre-fix reset version; re-screen same pool, preserve completed trials')
    p.add_argument('--recover-from-current', action='store_true',
                   help='First reset only: operator confirms hand already open/object supported; preflight return from current elevated pose, no opening/extra retreat')
    p.add_argument('--point', choices=['all']+[f'P{i}' for i in range(1, 11)], default='all')
    p.add_argument('--validate-only', action='store_true', help='No inference, IK or devices; save shared-condition audit')
    p.add_argument('--capture-live', action='store_true')
    p.add_argument('--trial-captures', type=Path, nargs='+')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--allow-high-force', action='store_true')
    p.add_argument('--video', action='store_true', help='Offline videos; live execution video is always required')
    p.add_argument('--open', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--out', type=Path)
    a = p.parse_args(argv)
    if a.resume_after_reset_fix and not a.resume_from:
        p.error('--resume-after-reset-fix requires --resume-from')
    if a.recover_from_current and not (a.resume_from and a.execute and a.capture_live):
        p.error('--recover-from-current requires live --execute resume and fresh recovery confirmation')
    if a.execute and not a.capture_live:
        p.error('--execute requires fresh live XY (--capture-live)')
    if a.validate_only and (a.execute or a.capture_live):
        p.error('--validate-only never contacts devices')
    if a.capture_live and a.trial_captures:
        p.error('Cannot mix live and saved trial captures')
    if a.official_filter and not a.candidate_run:
        p.error('--official-filter requires --candidate-run')
    if a.resume_from and (a.pool_report or a.candidate_run or a.point != 'all' or a.trial_captures or a.validate_only):
        p.error('Resume freezes inputs and continues remaining points with fresh live XY or offline reference')
    if not 0 <= a.seed < 2**32:
        p.error('Invalid seed')
    if a.trial_captures and len(a.trial_captures) != (10 if a.point == 'all' else 1):
        p.error('Supply one saved capture per requested point')
    for k, v in vars(a).items():
        if isinstance(v, Path):
            setattr(a, k, v.absolute() if k == 'ggx_python' else v.resolve())
    return a


def attempted(record):
    return bool(record.get('motion_commands_attempted') or record.get('hand_commands')
                or record.get('motion_confirmation_received') or record.get('cleanup_errors'))


def resume(args):
    if not args.resume_from:
        return []
    old = ui.read_json(args.resume_from)
    if old.get('schema') != SCHEMA or old.get('status') not in ('failed', 'stopped', 'interrupted', 'completed'):
        raise ValueError('Resume requires an ended matched-pool report')
    if old.get('execute_requested') != args.execute:
        raise ValueError('Cannot resume an offline run as real execution (or vice versa)')
    if getattr(args, 'resume_after_reset_fix', False):
        allowed = {str(ROOT/'tools/ggx_reset.py'): {'8038c5a7a2372d7f9d112679389b4fe953df5cc9713595716d1dc523bda6439b',
                                                '1a592f5f2e185de388d6e488ffae4e26e765a6c4bf953611579de8af92e13bb1'},
                   str(Path(__file__).resolve()): {'6144aad51931c3e2ed0111b7bbbab08c786059b77333a5b2b47d1052a56f9942',
                                                 '5838a71a3889ac1edb1b2455bc4896e83db91aa2afe3c48e5046e90b0f495daf'}}
        changes=[]
        for path, expected in old['pins'].items():
            actual=ui.digest(path)
            if actual != expected:
                if expected not in allowed.get(path, set()):
                    raise ValueError(f'Reset migration cannot waive other changed inputs: {path}')
                changes.append(dict(path=path, previous_sha256=expected, current_sha256=actual))
        args.reset_migration=dict(changes=changes, previous_report=str(args.resume_from),
            policy='reset-only source migration; no previous motion approval or local pass inherited')
    else:
        ui.verify(old['pins'])
    records = old.get('prior_trials', [])+old['trials']
    carried = []
    for i, row in enumerate(records):
        if row['point_id'] != f'P{i+1}':
            raise ValueError('Resume requires contiguous P1–P10 history')
        if row['status'] in ('commands_completed_grasp_unverified', 'no_usable_candidate', 'offline_prepared'):
            carried.append(dict(row, carried_forward=True, source_report=row.get('source_report', str(args.resume_from))))
        else:
            if i != len(records)-1 or any(attempted(r) for r in row.get('attempts', [])):
                raise ValueError('Partial/authorized trial needs manual review; no automatic retry')
            if row.get('reset_report'):
                reset = ui.read_json(row['reset_report'])
                preflight_only_failure = (reset.get('schema') == 'rh56_ggx_reset/v1'
                    and reset.get('status') == 'failed' and reset.get('motion_commands_attempted') is False
                    and not reset.get('hand_commands') and reset.get('cleanup_errors') == []
                    and reset.get('controller_preflight_timing', {}).get('status') == 'failed'
                    and all(t.get('stage') in ('connect_controller', 'preflight') for t in reset.get('timings', [])))
                reviewed_manual_recovery = (getattr(args, 'recover_from_current', False)
                    and reset.get('schema') == 'rh56_ggx_reset/v1' and reset.get('status') == 'failed'
                    and reset.get('motion_commands_attempted') is False and reset.get('cleanup_errors') == [])
                if reset.get('status') != 'reset_completed' and not preflight_only_failure and not reviewed_manual_recovery:
                    raise ValueError('Incomplete reset needs manual review before resume')
            break
    if len(carried) >= 10:
        raise ValueError('All ten points already have terminal results')
    args.reflex_run = Path(old['reflex_report'])
    args.pool_report = Path(old['pool_report'])
    if getattr(args, 'resume_after_reset_fix', False):
        pool=ui.read_json(args.pool_report)
        args.official_filter=Path(pool['official_scene_filter'])
        official=ui.read_json(args.official_filter)
        args.candidate_run=Path(official['candidate_file']).parent.parent
        args.pool_report=None
    args.previous_lift_report = None if getattr(args, 'recover_from_current', False) else last_lift_report(carried, args.resume_from.parent)
    return carried


def last_lift_report(trials, root):
    for trial in reversed(trials):
        if trial.get('status') == 'commands_completed_grasp_unverified':
            origin = Path(trial.get('source_report', root/'summary.json')).parent
            path = origin/trial['point_id']/f"rank_{trial['selected_rank']:03d}"/'summary.json'
            from tools.ggx_reset import completed_lift_target
            completed_lift_target(path)
            return path
    return None


def backend_with_hold(seconds):
    class MatchedBackend(real.LiveBackend):
        def set_stage(self, stage):
            if stage == 'close':
                self.report['grasp_close_command_attempted'] = True
            return super().set_stage(stage)

        def move_phase(self, command):
            super().move_phase(command)
            if command['phase'] == 'lift':
                self.set_stage('lift_hold')
                self.dwell(seconds)
                self.report['lift_hold_completed'] = True
                self.checkpoint()
    return MatchedBackend


def try_candidates(args, contract, pool, candidates, point, scan, measurement, folder, trial, save,
                   *, runner=real.run):
    """Only pre-motion feasibility rejections may advance the original rank."""
    authorization = {}
    trial.update(attempts=[], candidate_ledger=[], selected_rank=None)
    live_args = SimpleNamespace(execute=args.execute, preflight_only=False, checked_review=args.execute,
        static_scene=args.execute, start_snapshot=scan, phase='grasp-lift', video=args.video or args.execute, open=args.open)
    for row in pool['feasibility_stages']:
        rank = row['rank']
        entry = dict(rank=rank, confidence=row['confidence'], local_state=row['local_state'])
        trial['candidate_ledger'].append(entry)
        if rank not in candidates:
            entry['status'] = 'saved_screen_rejection'; continue
        if trial['selected_rank'] is not None:
            entry['status'] = 'not_tested_after_selected'; continue
        data, original, reference = candidates[rank]
        source, delta = template.relocate(data, original, reference,
            xy_mm=measurement['object_xy_mm'], lock_z=True)
        source.update(trial_config=contract['config'], desk_envelope=contract['config']['desk_envelope'],
                      desk_edge_y_m=contract['config']['desk_edge_y_m'])
        source['pins'].update(trial['pins'])
        model, canonical = template.canonical(data, source, reference, delta)
        destination = folder/f'rank_{rank:03d}'; destination.mkdir()
        record = dict(rank=rank, status='starting', motion_commands_attempted=False,
            motion_confirmation_received=False, code_config_sha256=trial['pins'], operator_outcome='not_executed',
            hold_seconds=contract['hold_seconds'], lift_mm=80.)
        trial['attempts'].append(record)
        def checkpoint():
            template.write_json(destination/'summary.json', record); save()
        def prepare(s, snapshot, out, video=False):
            result = template.prepare_trial(data, s, reference, model, canonical, point, snapshot, out, video=video)
            # One authorization explicitly covers this point's candidate search;
            # the legacy prompt only covered a single candidate. Motion still
            # needs the selected candidate's separate final video confirmation.
            authorize_point_checks(args.execute, point, authorization)
            return result
        try:
            code = runner(live_args, contract['config'], source, destination, record, checkpoint,
                prepare_fn=prepare, backend_factory=backend_with_hold(contract['hold_seconds']),
                controller_authorization=authorization)
            if record.get('cleanup_errors'):
                raise RuntimeError('Controller cleanup failed; cannot advance candidate')
            if code:
                # A safety veto may select next; cancellation/scene change stops.
                if not attempted(record) and record['status'] in ('cb3_candidate_rejected', 'cb3_review_vetoed'):
                    entry['status'] = record['status']; continue
                trial['status'] = 'stopped'; return code
            trial.update(selected_rank=rank, status=record['status'],
                         motion_commands_attempted=bool(record.get('motion_commands_attempted')),
                         grasp_close_command_attempted=bool(record.get('grasp_close_command_attempted')),
                         lift_hold_completed=bool(record.get('lift_hold_completed')))
            entry['status'] = 'selected'
        except (planning.PlanningUnavailable, real.motion.PathCollision) as exc:
            if attempted(record):
                raise
            record.update(status='rejected_before_motion', reason=str(exc),
                planning_reason=getattr(exc, 'reason', None), details=getattr(exc, 'attempts', None),
                collision_phase=getattr(exc, 'phase', None), obstacle=getattr(exc, 'obstacle', None))
            entry['status'] = 'rejected_before_motion'
        finally:
            checkpoint()
    if trial['selected_rank'] is None:
        trial['status'] = 'no_usable_candidate'
    return 0


def authorize_point_checks(execute, point, authorization):
    if execute and not authorization.get('confirmed', False):
        if not real.confirm(f"{point['id']}：允许本起点按原排名尝试候选的 RTDE 脚本启停和控制器 IK/路径预检；不允许运动"):
            raise KeyboardInterrupt
        authorization.update(confirmed=True, point_id=point['id'], scope='candidate_search_no_motion')


def run(args, out, report, save):
    from tools.run_reflex_ten_starts import score_trial
    prior = resume(args)
    contract = protocol.load_reference(args.reflex_run)
    args.config, args.perception_config = contract['config_path'], Path(contract['data']['perception']['config'])
    args.start_snapshot = contract['scan']
    template.write_json(out/'fixed_layout.json', contract['layout'])
    pins = {**contract['pins'], **ten.code_pins(args.config), **ui.pin([
        __file__, protocol.__file__, xy.__file__, ROOT/'tools/run_reflex_ten_starts.py', out/'fixed_layout.json'])}
    report.update(reflex_report=str(args.reflex_run), object_label=contract['object_label'],
        pins=pins, prior_trials=prior, shared_conditions=dict(fixed_layout=str(out/'fixed_layout.json'),
            reference_cloud=contract['data']['crop']['point_cloud'], config=contract['config'],
            scan=str(contract['scan']), xy_search_px=contract['xy_search_px'], hold_seconds=contract['hold_seconds'],
            lift_mm=80., z_yaw_frozen=True, wrist_branch_policy=contract['wrist_branch_policy']), method_differences=contract['differences'],
        historical_code_drift=contract['historical_code_drift'])
    if getattr(args, 'reset_migration', None): report['reset_migration']=args.reset_migration
    for message in contract['differences']:
        print('Protocol difference: '+message, flush=True)
    save()
    if args.validate_only:
        report['status'] = 'reference_validated_no_inference_or_devices'; return 0
    pool_path, pool, pool_pins = protocol.freeze_pool(args, contract, out)
    pins.update(pool_pins)
    if args.resume_from: pins.update(ui.pin([args.resume_from]))
    report.update(pool_report=str(pool_path), pool_counts=pool['feasibility_counts'],
                  inference_requested=not bool(args.pool_report or args.candidate_run))
    templates = out/'templates'; templates.mkdir()
    candidates = protocol.candidate_templates(pool, contract, pins, templates)
    if args.execute: real.require_hand_setting_authorization(contract['config'], args.allow_high_force)
    points = [p for p in contract['layout']['points'] if args.point == 'all' or p['id'] == args.point][len(prior):]
    scan, measurement, need_reset = contract['scan'], None, args.execute
    previous_lift = getattr(args, 'previous_lift_report', None)
    recover_current = getattr(args, 'recover_from_current', False)
    report['manual_current_recovery_requested'] = recover_current
    for index, original_point in enumerate(points):
        folder = out/original_point['id']; folder.mkdir()
        trial = dict(point_id=original_point['id'], object_label=contract['object_label'], status='starting',
                     operator_outcome='not_executed', pins=dict(pins), attempts=[])
        report['trials'].append(trial)
        def checkpoint():
            template.write_json(folder/'summary.json', trial); save()
        try:
            ui.verify(pins)
            # No candidates: record all ten as candidate unavailability, with no reset/camera/motion.
            if not candidates:
                trial.update(status='no_usable_candidate', failure_layer='saved_pool_screen',
                             candidate_ledger=pool['feasibility_stages']); continue
            if need_reset:
                from tools.ggx_reset import run_reset
                code, scan = run_reset(SimpleNamespace(reset_first=True, capture_live=True,
                    real_config=args.config, config=args.perception_config, start_snapshot=args.start_snapshot,
                    allow_high_force=args.allow_high_force, completed_lift_report=previous_lift,
                    recover_from_current=recover_current), folder/'reset')
                trial['reset_report'] = str(folder/'reset/summary.json')
                if code: trial['status']='reset_stopped';report['status']='stopped';return code
                recover_current = False
                measurement = None; need_reset = False
            if measurement is None or args.trial_captures:
                capture = (experiment.capture_xy(args, contract['reference'], folder) if args.capture_live else
                    args.trial_captures[index] if args.trial_captures else Path(contract['reference']['perception']['capture']))
                measurement = xy.locate(contract['reference'], capture, folder/'xy', search_px=contract['xy_search_px'])
                if measurement.get('status') != 'localized': raise ValueError('XY localization failed')
                measurement['input_sha256'].update(ui.pin([folder/'xy/summary.json']))
            point, delta = protocol.matched_point(contract['layout'], original_point['id'],
                contract['reference']['preflight']['object_center_m'], measurement['object_xy_mm'])
            trial['pins'].update(measurement['input_sha256'])
            trial.update(point=point, delta_xy_m=delta, xy=measurement)
            code = try_candidates(args, contract, pool, candidates, point, scan, measurement, folder, trial, checkpoint)
            if code: report['status']='stopped';return code
            if args.execute and trial['status'] == 'commands_completed_grasp_unverified':
                previous_lift = folder/f"rank_{trial['selected_rank']:03d}"/'summary.json'
                need_reset = True
                if not score_trial(trial): report['status']='stopped';return 0
        except (KeyboardInterrupt, EOFError):
            trial['status']='interrupted';raise
        except Exception as exc:
            trial.update(status='failed', error=f'{type(exc).__name__}: {exc}');raise
        finally:
            checkpoint()
    report['status'] = 'completed'
    return 0


def main(argv=None):
    args = parse_args(argv)
    out = args.out or ROOT/'artifacts/ggx_matched_ten_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema=SCHEMA, status='starting', execute_requested=args.execute, trials=[],
        selection_policy='frozen_original_top100_first_feasible_per_start_one_physical_attempt',
        assumptions=['No reranking, top-100 backfill or physical retry; safety vetoes are logged.',
            'Prior Reflex results are not rewritten; shared input/layout does not imply identical hand/contact strategy.',
            'Unknown fixtures/camera mount are not certified by the local model.',
            'Only newly confirmed reset and checked-video execution may move hardware.'])
    def save():
        records = report.get('prior_trials', [])+report['trials']
        report['counts'] = dict(planned=10 if args.point == 'all' else 1, visited=len(records),
            no_usable_candidate=sum(t['status']=='no_usable_candidate' for t in records),
            motion_attempted=sum(any(a.get('motion_commands_attempted') for a in t.get('attempts', [])) for t in records),
            operator_success=sum(t.get('operator_outcome')=='success' for t in records),
            lift_hold_completed=sum(bool(t.get('lift_hold_completed')) for t in records))
        template.write_json(out/'summary.json', report)
        with (out/'summary.csv').open('w', newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=['object_label','point_id','status','selected_rank','operator_outcome',
                'motion_commands_attempted','grasp_close_command_attempted','lift_hold_completed','error'],extrasaction='ignore')
            w.writeheader();w.writerows(records)
    try:
        return run(args, out, report, save)
    except (KeyboardInterrupt, EOFError):
        report['status']='interrupted';return 130
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        print(report['error'], file=sys.stderr);return 1
    finally:
        save()
        print(f"GGX matched: {report['status']}; {out/'summary.json'}", flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
