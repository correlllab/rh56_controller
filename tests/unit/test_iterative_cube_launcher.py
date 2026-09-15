"""Launcher safety/forwarding tests: no robot, serial device or camera required."""
import hashlib
import json

import pytest

from tools import run_iterative_cube as launcher
from tools import run_vision_pick_cycle as cycle


@pytest.fixture
def runner(monkeypatch):
    calls = []
    manifest = dict(config='configs/ur5_iterative_cube_lab.yaml', preview_capture='saved/capture')
    monkeypatch.setattr(launcher, 'verify_baseline', lambda: manifest)
    monkeypatch.setattr(cycle, 'main', lambda args: calls.append(cycle.parse_args(args)) or 0)
    return calls


def test_default_is_offline_and_never_asks_to_move(runner, monkeypatch, tmp_path):
    monkeypatch.setattr('builtins.input', lambda _: pytest.fail('Offline must not ask for motion'))
    assert launcher.main(['--out', str(tmp_path / 'run')]) == 0
    assert not runner[0].execute and not runner[0].reset_from_low
    assert runner[0].capture == launcher.ROOT / 'saved/capture'


@pytest.mark.parametrize('answers', [[], ['cancel'], [cycle.CONFIRMATION]])
def test_cancel_before_any_cycle_call(runner, monkeypatch, tmp_path, answers):
    answers = iter(answers)
    def read(_):
        try:
            return next(answers)
        except StopIteration:
            raise EOFError
    monkeypatch.setattr('builtins.input', read)
    assert launcher.main(['--execute', '--out', str(tmp_path / 'run')]) == 2
    assert not runner


def test_confirmed_cycle_forwards_explicit_authorizations(runner, monkeypatch, tmp_path):
    # Only one prompt; a second would exhaust the iterator and fail this test.
    answers = iter([''])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    assert launcher.main(['--execute', '--out', str(tmp_path / 'run')]) == 0
    args = runner[0]
    assert args.execute and args.reset_from_low and args.confirm_high_force
    assert args.acknowledge_open_loop and args.acknowledge_shape_extrapolation
    assert args.capture is None


def test_ctrl_c_cancels_before_hardware(runner, monkeypatch, tmp_path):
    def interrupt(_):
        raise KeyboardInterrupt
    monkeypatch.setattr('builtins.input', interrupt)
    assert launcher.main(['--execute', '--out', str(tmp_path / 'run')]) == 2
    assert not runner


def test_baseline_is_rechecked_after_enter(runner, monkeypatch, tmp_path):
    def read(_):
        def changed():
            raise ValueError('Baseline changed during confirmation')
        monkeypatch.setattr(launcher, 'verify_baseline', changed)
        return ''
    monkeypatch.setattr('builtins.input', read)
    with pytest.raises(ValueError, match='during confirmation'):
        launcher.main(['--execute', '--out', str(tmp_path / 'run')])
    assert not runner


@pytest.mark.parametrize('extra', [['--execute', '--capture', 'saved'], []])
def test_invalid_or_existing_output_never_calls_cycle(runner, tmp_path, extra):
    with pytest.raises(SystemExit):
        launcher.main([*extra, '--out', str(tmp_path)])
    assert not runner


def test_pinned_file_changes_and_missing_files_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    config = tmp_path / 'config.yaml'
    config.write_text('original')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema='rh56_iterative_cube_baseline/v1',
        config='config.yaml', sha256={'config.yaml': hashlib.sha256(config.read_bytes()).hexdigest()})))
    assert launcher.verify_baseline(manifest)['config'] == 'config.yaml'
    config.write_text('changed')
    with pytest.raises(ValueError, match='Baseline changed'):
        launcher.verify_baseline(manifest)
    config.unlink()
    with pytest.raises(ValueError, match='Baseline file missing'):
        launcher.verify_baseline(manifest)


def test_changed_baseline_blocks_before_confirmation(runner, monkeypatch, tmp_path):
    def changed():
        raise ValueError('Baseline changed')
    monkeypatch.setattr(launcher, 'verify_baseline', changed)
    monkeypatch.setattr('builtins.input', lambda _: pytest.fail('Must verify before asking'))
    with pytest.raises(SystemExit):
        launcher.main(['--execute', '--out', str(tmp_path / 'run')])
    assert not runner


def test_per_run_manifest_is_written(runner, monkeypatch, tmp_path):
    def fake_cycle(argv):
        args = cycle.parse_args(argv)
        args.out.mkdir()
        (args.out / 'summary.json').write_text('{}')
        return 0
    monkeypatch.setattr(cycle, 'main', fake_cycle)
    out = tmp_path / 'run'
    launcher.main(['--out', str(out)])
    record = json.loads((out / 'launcher.json').read_text())
    assert record['schema'] == 'rh56_iterative_cube_launch/v1'
    assert record['execute'] is False and record['confirmations_completed'] is False
    assert record['confirmation_method'] == 'none'
