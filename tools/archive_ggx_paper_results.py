#!/usr/bin/env python3
"""Archive the 2026-09-13 five-object GGX session without touching hardware.

Copies evidence/current source, retains historical hash drift, emits CSV tables
and a checksummed tarball. This is a local archive, not an off-machine backup.
Historical experiment reports and motion approvals are never rewritten.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_ggx_frozen_audit import read, write, digest, source_files, historical_hashes


def table(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def command(args, cwd=ROOT):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    return dict(command=list(map(str, args)), returncode=result.returncode,
                stdout=result.stdout, stderr=result.stderr)


def verify_bundle(out):
    with (out/'manifest.csv').open(newline='') as stream:
        entries = list(csv.DictReader(stream))
    for entry in entries:
        target = (out/entry['bundle_path']).resolve()
        if not target.is_relative_to(out.resolve()) or digest(target) != entry['sha256']:
            raise ValueError(f'Archive checksum mismatch: {entry["bundle_path"]}')
    summary = read(out/'summary.json')
    if digest(out/'manifest.csv') != summary['manifest_sha256']:
        raise ValueError('Manifest checksum differs from summary')
    return len(entries)


def build(args, out):
    experiment_runs = sorted((ROOT/'artifacts/ggx_experiments').glob('20260913*'))
    roots = [*experiment_runs,
        ROOT/'artifacts/ggx_quick_preview_20260912/fresh_scene',
        ROOT/'artifacts/ggx_quick_preview_20260912/current_capture',
        ROOT/'artifacts/ggx_quick_preview_20260913/top100_completion',
        ROOT/'artifacts/ggx_quick_preview_20260913/rank20_preview',
        ROOT/'artifacts/ggx_start_stage_audits/20260913_sugar_rank20',
        ROOT/'artifacts/tabletop_background/empty_20260909_01',
        ROOT/'h1_mujoco/inspire',
        ROOT/'magpie_force_control/src/ur_description/meshes/ur5/collision',
        ROOT/'artifacts/ggx_lab/GraspGenX/ext/gripper_descriptions/gripper_descriptions/assets/x_grippers/inspire_hand']
    files = {p.resolve() for root in roots if root.is_dir() for p in root.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and not p.is_symlink()}
    for folder in ('tools', 'rh56_controller', 'tests'):
        files.update(p.resolve() for p in (ROOT/folder).rglob('*.py') if '__pycache__' not in p.parts)
    for folder in ('configs', 'docs'):
        files.update(p.resolve() for p in (ROOT/folder).rglob('*') if p.is_file())
    files.update(p.resolve() for name in ('AGENTS.md', 'pyproject.toml', 'uv.lock', '.gitignore')
                 if (p := ROOT/name).is_file())
    official = ROOT/'artifacts/ggx_lab/GraspGenX'
    for folder in ('graspgenx', 'scripts'):
        files.update(p.resolve() for p in (official/folder).rglob('*.py'))
    files.update(p.resolve() for p in official.glob('*') if p.is_file() and
                 p.suffix.lower() in ('.toml', '.yaml', '.yml', '.txt', '.md', '.lock'))

    # Follow file references in JSON metadata transitively, preserving the full
    # scene/model lineage rather than just the selected run directory.
    processed, expected, excluded, referenced = set(), set(), [], set()
    while True:
        pending = [p for p in files-processed if p.suffix == '.json']
        if not pending:
            break
        for path in pending:
            processed.add(path)
            try:
                record = read(path)
            except (UnicodeError, json.JSONDecodeError):
                continue
            expected.update(historical_hashes(record))
            referenced.update(source_files(record))
            if isinstance(record, dict) and record.get('checkpoint_root'):
                excluded.append(dict(path=record['checkpoint_root'], reason='Model checkpoint directory; revision retained in inference report'))
        for path in referenced:
            p = path.resolve()
            if not p.is_relative_to(ROOT):
                continue
            if any(part.startswith('.venv') for part in p.parts) or 'graspgenx_checkpoints' in p.parts:
                excluded.append(dict(path=str(p), reason='Environment or checkpoint payload not copied'))
            else:
                files.add(p)
    print(f'Collecting {len(files)} evidence/source files; no devices accessed.', flush=True)
    current_hashes, manifest = {}, []
    for index, path in enumerate(sorted(files)):
        if not path.is_relative_to(ROOT):
            raise ValueError(f'Unexpected external dependency: {path}')
        target = out/'files'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        before = digest(path)
        shutil.copy2(path, target)
        if digest(target) != before or digest(path) != before:
            raise ValueError(f'Source changed during archival copy: {path}')
        current_hashes[str(path)] = before
        manifest.append(dict(original_path=str(path), bundle_path=str(target.relative_to(out)),
                             bytes=target.stat().st_size, sha256=before))
        if index and index % 1000 == 0:
            print(f'Copied and verified {index} files.', flush=True)
    mismatches = [dict(path=name, historical_sha256=sha, archived_current_sha256=current_hashes.get(name),
                       status='historical_bytes_not_at_original_path')
                  for name, sha in sorted(expected) if current_hashes.get(name) != sha]
    archived_hashes = {row['sha256']: row['bundle_path'] for row in manifest}
    for mismatch in mismatches:
        mismatch['historical_bytes_found_elsewhere_in_bundle'] = archived_hashes.get(mismatch['historical_sha256'])
    write(out/'historical_hash_drift.json', mismatches)
    write(out/'excluded_dependencies.json', list({(r['path'], r['reason']): r for r in excluded}.values()))

    run_rows, trial_rows, batches = [], [], []
    for run in experiment_runs:
        p = out/'files'/run.relative_to(ROOT)/'summary.json'
        if not p.exists():
            continue
        r = read(p)
        run_rows.append(dict(run=run.name, object=r.get('object_label'), execute_requested=r.get('execute_requested'),
            status=r.get('status'), rank=r.get('rank'), selection_policy=r.get('selection_policy'),
            inference_runs=r.get('ggx_inference_runs'), recorded_point_trials=len(r.get('trials', [])), error=r.get('error'),
            evidence=str(p.relative_to(out))))
        for trial in r.get('trials', []):
            trial_rows.append(dict(run=run.name, object=r.get('object_label'),
                **{k: trial.get(k) for k in ('point_id', 'rank', 'status', 'planning_reason', 'execute_requested',
                    'motion_commands_attempted', 'motion_confirmation_received', 'stage', 'error')},
                physical_success='unverified_by_software', evidence=str(p.relative_to(out))))
    batch_roots = [ROOT/'artifacts/ggx_quick_preview_20260912/fresh_scene',
                   *[p/'initial_ggx' for p in experiment_runs]]
    for directory in batch_roots:
        frozen = out/'files'/directory.relative_to(ROOT)
        if not (frozen/'inference/summary.json').exists():
            continue
        inference = read(frozen/'inference/summary.json')
        selection = read(frozen/'summary.json')
        filtered = read(frozen/'official_scene_filter/summary.json') if (frozen/'official_scene_filter/summary.json').exists() else {}
        counts = selection.get('feasibility_counts', {}) or {}
        candidates = filtered.get('candidates', [])
        batches.append(dict(batch=str(directory.relative_to(ROOT)), seed=inference.get('seed'),
            raw_candidates=inference.get('pre_topk_archive', {}).get('raw_count'), topk=inference.get('topk'),
            official_evaluated=len(candidates), official_pass=sum(x['official_scene_clear'] for x in candidates),
            local_rank_prefix_examined=counts.get('candidates'), local_geometry_pass_in_examined_prefix=counts.get('geometry_eligible'),
            selection_status=selection.get('status'), selected_rank=selection.get('selected_rank'),
            evidence=str((frozen/'summary.json').relative_to(out))))
    table(out/'runs.csv', run_rows, ['run','object','execute_requested','status','rank','selection_policy',
        'inference_runs','recorded_point_trials','error','evidence'])
    table(out/'trials.csv', trial_rows, ['run','object','point_id','rank','execute_requested','status','planning_reason',
        'motion_commands_attempted','motion_confirmation_received','stage','physical_success','error','evidence'])
    table(out/'candidate_batches.csv', batches, ['batch','seed','raw_candidates','topk','official_evaluated',
        'official_pass','local_rank_prefix_examined','local_geometry_pass_in_examined_prefix','selection_status','selected_rank','evidence'])
    write(out/'operator_observations.json', dict(schema='rh56_ggx_operator_observations/v1',
        source='User statements in this conversation; not sensor-derived and no transcript timestamp invented',
        observations=[
            dict(object='sugar_box',run='20260913T203244_138758Z',point='P2',rank=20,
                 reported_grasp_success=True,manual_stop=True,camera_mount_near_collision=True,
                 full_lift_hold_completion='not_confirmed',user_requested_summary='整套流程成功 1/10；执行后成功 1/1；路径拒绝 9/10'),
            dict(object='cube',user_requested_summary='GGX: no_candidate_approved'),
            dict(object='mustard_bottle',run='20260913T204646_815984Z',point='P2',rank=72,
                 reported_grasp_success=False,failure_mechanism='not_reported',user_requested_summary='0/10'),
            dict(object='charger',user_requested_summary='0/10'),
            dict(object='can',user_requested_summary='0/10',counting_policy_for_two_batches='not_fixed'),
            dict(scope='comparison_protocol',statement='User states all other methods use the same pose',
                 independent_cross_method_verification=False)]))
    git = {name: command(['git', *cmd]) for name, cmd in {
        'head':['rev-parse','HEAD'], 'branch':['branch','--show-current'], 'status':['status','--short'],
        'diff':['diff','--binary','HEAD'], 'staged_diff':['diff','--cached','--binary']}.items()}
    write(out/'git_state.json', git)
    environments = {}
    snippet = 'import sys,json,importlib.metadata as m; print(json.dumps({"python":sys.version,"executable":sys.executable,"packages":sorted([(d.metadata.get("Name",""),d.version) for d in m.distributions()])}))'
    for name, python in [('controller',ROOT/'.venv312/bin/python'), ('ggx',official/'.venv/bin/python')]:
        environments[name] = command([str(python), '-c', snippet])
    write(out/'environments.json', environments)
    if args.attachment:
        target = out/'discussion_attachment.txt'
        shutil.copy2(args.attachment, target)
        if digest(target) != digest(args.attachment):
            raise ValueError('Discussion attachment checksum mismatch')
    shutil.copy2(out/'files/docs/ggx_results_20260913_zh.md', out/'README_zh.md')
    # Include generated tables/notes in integrity verification, not just raw evidence.
    copied_paths = {row['bundle_path'] for row in manifest}
    for path in sorted(out.rglob('*')):
        if path.is_file() and str(path.relative_to(out)) not in copied_paths:
            manifest.append(dict(original_path='', bundle_path=str(path.relative_to(out)),
                                 bytes=path.stat().st_size, sha256=digest(path)))
    table(out/'manifest.csv', manifest, ['original_path','bundle_path','bytes','sha256'])
    summary = dict(schema='rh56_ggx_paper_evidence_bundle/v1', status='archived',
        created_utc=datetime.now(timezone.utc).isoformat(), hardware_contacted=False,
        originals_modified=False, motion_authorized=False, local_only=True,
        source_roots=[str(p) for p in roots],file_count=len(manifest),total_bytes=sum(r['bytes'] for r in manifest),
        historical_hash_mismatches=len(mismatches), historical_hashes_rewritten=False,
        manifest_sha256=digest(out/'manifest.csv'),
        counts=dict(experiment_run_directories=len(run_rows),recorded_point_trials=len(trial_rows),
                    successful_inference_batches=len(batches),
                    real_trials_with_motion_attempt=sum(r.get('execute_requested') is True and
                        r.get('motion_commands_attempted') is True for r in trial_rows)),
        limitations=['No hardware access, scene recapture, new inference or motion approval.',
            'Current source snapshot is not all historical source revisions; see historical_hash_drift.json.',
            'Virtual environments, GPU drivers and model checkpoint payloads excluded; versions/revisions retained.',
            'Inline colleague figure is transcribed in README only; original image/PDF unavailable.',
            'Original absolute paths are preserved in copied reports, not rewritten into portable motion approvals.',
            'Manual success observations are separate from program status and unverified full-task completion.',
            'A local verified archive is not an off-site backup or independent replication.'])
    write(out/'summary.json', summary)
    print(f'Verifying {verify_bundle(out)} files, then compressing archive.', flush=True)
    archive = out.with_suffix('.tar.gz')
    with tarfile.open(archive, 'x:gz', compresslevel=3) as tar:
        tar.add(out, arcname=out.name)
    archive_digest = digest(archive)
    (out.parent/(archive.name+'.sha256')).write_text(f'{archive_digest}  {archive.name}\n')
    print(f'Archive complete: {out}; {archive}; SHA-256 {archive_digest}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts/ggx_paper_evidence/20260913_session')
    parser.add_argument('--attachment', type=Path, help='User-provided discussion text; stored as non-authoritative context')
    parser.add_argument('--verify', type=Path, help='Only verify an existing bundle; no copying or hardware')
    args = parser.parse_args(argv)
    if args.verify:
        print(f'Verified {verify_bundle(args.verify.resolve())} manifest entries.')
        return 0
    out = args.out.resolve()
    if out.exists() or out.with_suffix('.tar.gz').exists():
        parser.error('Output directory/archive already exists; choose a new path, never overwrite evidence')
    if not out.is_relative_to(ROOT/'artifacts/ggx_paper_evidence'):
        parser.error('Archive output must be under artifacts/ggx_paper_evidence/')
    out.mkdir(parents=True)
    try:
        build(args, out)
        return 0
    except Exception as exc:
        write(out/'archive_error.json', dict(status='archive_failed',error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
