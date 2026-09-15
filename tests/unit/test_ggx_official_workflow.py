"""Official selection is independent from local motion eligibility; no devices."""
import json
from types import SimpleNamespace

import pytest

from tools import run_ggx_review as review


def test_official_prompt_states_endpoint_selection_not_motion(tmp_path, monkeypatch):
    image = tmp_path / 'official.png'
    image.write_bytes(b'image')
    prompts = []
    monkeypatch.setattr(review, 'show', lambda *a: None)
    monkeypatch.setattr('builtins.input', lambda prompt: prompts.append(prompt) or '')
    assert review.decision(image, review.pin([image]), False, official_only=True) == ('approve', '')
    assert '未检查本地路径，不执行实机' in prompts[0]


@pytest.mark.parametrize('live_flag', ['--execute', '--reset-first'])
def test_official_new_generation_cannot_accidentally_move(live_flag):
    with pytest.raises(SystemExit):
        review.parse_args(['--capture-live', '--dimensions-mm', '40', '90', '176',
                           '--feasibility-mode', 'official-only', live_flag])


def test_reuse_does_not_need_dimensions_or_run_inference(tmp_path, monkeypatch):
    run = tmp_path / 'saved'
    (run / 'inference').mkdir(parents=True)
    (run / 'simulation.yaml').write_text('saved config')
    (run / 'inference/summary.json').write_text('{}')
    args = review.parse_args(['--candidate-run', str(run), '--feasibility-mode', 'official-only'])
    assert args.max_candidates == 100
    def simulate(args, out, report, config, pins):
        assert config == run / 'simulation.yaml'
        assert args.inference_report == run / 'inference/summary.json'
        review.verify(pins)
        yield 7, None, pins
    monkeypatch.setattr(review, 'simulate_candidates', simulate)
    monkeypatch.setattr(review, 'run_stage', lambda *a, **k: pytest.fail('No capture/inference'))
    report = {}
    assert next(review.reuse_candidates(args, tmp_path, report))[0] == 7
    assert not report['reused_candidates']['old_approval_inherited']
    assert not report['reused_candidates']['inference_rerun']


def test_official_selection_records_original_rank_and_distinct_scope(tmp_path, monkeypatch):
    directory = tmp_path / 'candidate'
    directory.mkdir()
    image = directory / 'review_official.png'
    image.write_bytes(b'image')
    summary = directory / 'summary.json'
    summary.write_text(json.dumps(dict(schema='rh56_ggx_official_candidate_selection/v1')))
    pins = review.pin([image, summary])
    # The real generator has close(); retain that protocol in the mock.
    def candidates(*a):
        yield 53, image, pins
    monkeypatch.setattr(review, 'reuse_candidates', candidates)
    monkeypatch.setattr(review, 'decision', lambda *a, **k: ('approve', ''))
    out = tmp_path / 'review'
    assert review.main(['--candidate-run', 'saved', '--feasibility-mode', 'official-only',
                        '--no-open', '--out', str(out)]) == 0
    report = review.read_json(out / 'summary.json')
    assert report['selected_rank'] == 53
    assert report['status'] == 'official_candidate_selected'
    assert report['reviews'][0]['scope'] == 'official_ggx_candidate'
    assert not report['executable'] and not report['robot_commands_sent']


@pytest.mark.parametrize('execute,preflight', [(True, False), (False, True)])
def test_official_unverified_hand_stops_before_every_device(tmp_path, execute, preflight):
    pytest.importorskip('mujoco')
    pytest.importorskip('mink')
    from tools import run_ggx_real as real
    args = SimpleNamespace(execute=execute, preflight_only=preflight)
    report = {}
    def forbidden(*a, **k):
        pytest.fail('No hardware, capture, IK or hand commands before hand contract is resolved')
    assert real.run(args, {}, {'selection_scope': 'official_ggx_candidate'}, tmp_path,
                    report, lambda: None, backend_factory=forbidden, scene_check=forbidden,
                    prepare_fn=forbidden) == 2
    assert report['status'] == 'physical_hand_mapping_unverified'
    assert report['official_candidate_retained']


def test_template_cannot_upgrade_official_selection(tmp_path, monkeypatch):
    pytest.importorskip('mujoco')
    pytest.importorskip('mink')
    from tools import ggx_template as template
    (tmp_path / 'approval.json').write_text('{}')
    monkeypatch.setattr(template.support, 'read_approval', lambda *a: {
        'selection_scope': 'official_ggx_candidate'})
    monkeypatch.setattr(template.support, 'prepare', lambda *a, **k: pytest.fail('No automatic upgrade'))
    with pytest.raises(ValueError, match='physical OPEN'):
        template.create(tmp_path / 'approval.json', None, {}, tmp_path)
