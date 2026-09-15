"""Figure bookkeeping tests with synthetic records; no devices or lab artifacts."""
import csv
import json

import pytest

from tools import plot_real_vision_comparison as plot


@pytest.fixture
def records(tmp_path):
    for path in plot.REFLEX:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=['point_id', 'operator_outcome'])
            writer.writeheader()
            writer.writerows({'point_id': f'P{i}', 'operator_outcome': 'success' if i <= 5 else 'fail'} for i in range(1, 11))
    target = tmp_path / plot.GGX
    target.parent.mkdir(parents=True)
    with target.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['point_id', *plot.OBJECTS])
        writer.writeheader()
        writer.writerows({'point_id': f'P{i}', **{obj: int(i <= 2) for obj in plot.OBJECTS}} for i in range(1, 11))
    target = tmp_path / plot.RASPBERRY_REFLEX
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({'status': 'completed', 'trials': [
        {'point_id': f'P{i}', 'operator_outcome': 'success' if i <= 5 else 'not_executed'} for i in range(1, 11)]}))
    target = tmp_path / plot.RASPBERRY_FILTER
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({'status': 'completed', 'candidates': [
        {'rank': i, 'official_scene_clear': False} for i in range(100)]}))
    return tmp_path


def test_original_five_unchanged(records):
    data = plot.load_results(records)
    assert len(data['Reflex']) == len(data['GGX']) == 5
    assert sum(map(sum, data['Reflex'])) == 25


def test_six_object_denominator_and_caption(records, monkeypatch):
    monkeypatch.setattr(plot, 'render', lambda *args: None)
    out = records / 'figure'
    assert plot.main(['--root', str(records), '--out', str(out), '--layout', 'combined', '--include-raspberry']) == 0
    rows = list(csv.DictReader((out / 'summary.csv').open()))
    overall = {row['method']: row for row in rows if row['object'] == 'overall'}
    assert overall['Reflex']['planned_starts'] == '60'
    assert overall['Reflex']['success_rate'] == '0.5'
    assert overall['GGX']['successes'] == '10'
    caption = (out / 'caption.txt').read_text()
    assert 'six objects' in caption and '60 per method' in caption
    assert 'zero physical' in caption
    metadata = json.loads((out / 'metadata.json').read_text())
    assert metadata['display_order'][-1] == 'raspberry'
    assert plot.RASPBERRY_FILTER in metadata['sources_sha256']


def test_eligible_candidate_cannot_be_recorded_as_empty_pool(records):
    target = records / plot.RASPBERRY_FILTER
    report = json.loads(target.read_text())
    report['candidates'][0]['official_scene_clear'] = True
    target.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='fully rejected'):
        plot.load_results(records, include_raspberry=True)


def test_nut_status_does_not_invent_trials_or_change_rates(records, monkeypatch):
    monkeypatch.setattr(plot, 'render', lambda *args: None)
    source = records / plot.NUT_DIAGNOSTIC
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({'status': 'requested_endpoint_desk_rejected',
                                  'hardware_contacted': False}))
    out = records / 'nut_figure'
    assert plot.main(['--root', str(records), '--out', str(out), '--layout', 'combined',
                      '--include-raspberry', '--include-nut-status']) == 0
    rows = list(csv.DictReader((out / 'summary.csv').open()))
    assert not any(row['object'] == 'hex_nut' for row in rows)
    assert all(row['planned_starts'] == '60' for row in rows if row['object'] == 'overall')
    statuses = list(csv.DictReader((out / 'exploratory_status.csv').open()))
    assert statuses[1]['status'] == 'not_tested'
    assert all(row['physical_attempts'] == '0' for row in statuses)
    assert 'GGX was not tested' in (out / 'caption.txt').read_text()
