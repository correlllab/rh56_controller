"""Model reuse / lazy review / video policy. No camera, robot, GPU or encoder."""
from types import SimpleNamespace
from dataclasses import replace
import csv

import numpy as np
import pytest
import yaml

pytest.importorskip('mujoco')
from tools import preflight_ggx_ur5_table as sim, run_ggx_review as review


@pytest.fixture
def batch(tmp_path, monkeypatch):
    xml = tmp_path / 'model.xml'
    xml.write_text('fake model; build_model is injected')
    candidate_file = tmp_path / 'candidates.yml'
    # Deliberately not in confidence order on disk.
    entries = {f'grasp_{i}': dict(confidence=score, position=[0., 0., .2],
               orientation=dict(xyz=[0., 0., 0.], w=1.))
               for i, score in enumerate([.7, .9, .6, .8])}
    candidate_file.write_text(yaml.safe_dump(dict(format='isaac_grasp', grasps=entries)))
    config = tmp_path / 'sim.yaml'
    config.write_text(yaml.safe_dump(dict(xml=str(xml), graspgenx_yaml=str(candidate_file), video=True)))
    built, evaluated, rendered = [], [], []
    def build(*a, **kw):
        model = object()
        built.append(model)
        return model
    monkeypatch.setattr(sim, 'build_model', build)
    monkeypatch.setattr(sim, 'initial_qpos', lambda m,q: np.zeros(6))
    def evaluate(model, start, candidate, rank, args, center):
        assert not start.flags.writeable
        np.testing.assert_array_equal(start, np.zeros(6))
        evaluated.append((model, rank, candidate.name, start.copy()))
        sample = sim.TrajectorySample('approach', 1., start.copy() + rank, 0.)
        passed = rank != 0
        result = sim.CandidateResult(rank, candidate.name, candidate.confidence, passed,
            'pass' if passed else 'object_collision_before_closure', 0., 0., 10., 'palm', '', 1)
        return result, [sample]
    monkeypatch.setattr(sim, 'evaluate_candidate', evaluate)
    monkeypatch.setattr(sim, '_trajectory_rows',
                        lambda m,s: [dict(q=list(sample.qpos), phase=sample.phase) for sample in s])
    def render(path, model, samples, result, args, center):
        rendered.append((model, result.rank, path))
        path.write_bytes(f'fake video rank {result.rank}'.encode())
    monkeypatch.setattr(sim, 'write_video', render)
    return SimpleNamespace(config=config, xml=xml, candidate_file=candidate_file,
                           built=built, evaluated=evaluated, rendered=rendered)


def arguments(batch, out, rank=0, rejected=False):
    return ['--config', str(batch.config), '--candidate-rank', str(rank), '--out', str(out),
            '--rejected-video' if rejected else '--no-rejected-video']


def test_scene_load_sorts_without_evaluating_and_reuses_model(batch, tmp_path):
    args = arguments(batch, tmp_path/'unused')
    scene = sim.load_scene(sim.parse_args(args))
    assert [c.name for c in scene.candidates] == ['grasp_1', 'grasp_3', 'grasp_0', 'grasp_2']
    assert not batch.evaluated
    for rank in range(3):
        out = tmp_path/f'rank{rank}'
        assert sim.main(arguments(batch, out, rank), scene=scene) == (2 if rank == 0 else 0)
        assert review.read_json(out/'summary.json')['timings']['model_reused']
    assert len(batch.built) == 1
    assert [r[1] for r in batch.evaluated] == [0, 1, 2]
    assert all(r[0] is scene.model for r in batch.evaluated)


@pytest.mark.parametrize('rejected', [False, True])
def test_rejection_reports_unchanged_video_optional(batch, tmp_path, rejected):
    out = tmp_path/'rejected'
    assert sim.main(arguments(batch, out, rejected=rejected)) == 2
    report = review.read_json(out/'summary.json')
    assert not report['passed'] and report['automatic_pass_ranks'] == []
    assert report['object_collision_gate']['revision'] == sim.OBJECT_GATE_REVISION
    assert report['timings']['rendered_videos'] == int(rejected)
    assert (out/'review.mp4').exists() == rejected
    assert len(batch.rendered) == int(rejected)
    assert (out/'trajectory.csv').exists() and (out/'summary.csv').exists()
    with (out/'candidates.csv').open() as stream:
        assert next(csv.DictReader(stream))['reason'] == 'object_collision_before_closure'
    assert list(review.reviewed_candidates(out)) == []


def test_standalone_rejected_video_default_is_preserved():
    assert sim.parse_args([]).rejected_video


def test_pass_video_rendered_once_and_alias_is_exact_copy(batch, tmp_path):
    out = tmp_path/'passed'
    assert sim.main(arguments(batch, out, rank=1)) == 0
    assert len(batch.rendered) == 1
    assert (out/'review.mp4').read_bytes() == (out/'review_rank_001.mp4').read_bytes()
    assert (out/'trajectory.csv').read_bytes() == (out/'trajectory_rank_001.csv').read_bytes()
    assert review.read_json(out/'summary.json')['timings']['rendered_videos'] == 1
    assert next(review.reviewed_candidates(out))[0] == 1


def test_no_video_still_never_calls_renderer(batch, tmp_path):
    out = tmp_path/'no_video'
    assert sim.main([*arguments(batch, out, rank=1), '--no-video']) == 0
    assert not batch.rendered and not list(out.glob('*.mp4'))


def test_shared_and_fresh_scene_have_identical_candidate_and_trajectory_reports(batch, tmp_path):
    scene = sim.load_scene(sim.parse_args(arguments(batch, tmp_path/'unused')))
    for rank in range(3):
        cold, warm = tmp_path/f'cold{rank}', tmp_path/f'warm{rank}'
        assert sim.main(arguments(batch, cold, rank)) == sim.main(arguments(batch, warm, rank), scene=scene)
        for name in ('candidates.csv', 'trajectory.csv', 'summary.csv'):
            assert (cold/name).read_bytes() == (warm/name).read_bytes()


@pytest.mark.parametrize('option,value', [('--object-x-m', '.5'), ('--desk-clearance-mm', '3'),
                                        ('--path-step-deg', '2')])
def test_reuse_rejects_changed_scene_or_check_parameters(batch, tmp_path, option, value):
    args = arguments(batch, tmp_path/'unused')
    scene = sim.load_scene(sim.parse_args(args))
    with pytest.raises(ValueError, match='arguments changed'):
        sim.main([*args, option, value], scene=scene)
    assert not batch.evaluated


@pytest.mark.parametrize('name', ['config', 'xml', 'candidate_file'])
def test_reuse_rejects_changed_input_files(batch, tmp_path, name):
    args = arguments(batch, tmp_path/'unused')
    scene = sim.load_scene(sim.parse_args(args))
    path = getattr(batch, name)
    path.write_text(path.read_text()+'\n# modified\n')
    with pytest.raises(ValueError, match='input changed'):
        sim.main(args, scene=scene)
    assert not batch.evaluated


def iterator(batch, out):
    out.mkdir()
    args = SimpleNamespace(max_candidates=4)
    report = dict(status='running', reviews=[])
    return review.simulate_candidates(args, out, report, batch.config, review.pin([batch.config])), report


def test_interactive_checks_are_lazy_without_candidate_subprocesses(batch, tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'run_stage', lambda *a, **kw: pytest.fail('No subprocess'))
    candidates, report = iterator(batch, tmp_path/'review')
    assert not batch.built and not batch.evaluated
    assert next(candidates)[0] == 1
    assert [r[1] for r in batch.evaluated] == [0, 1]
    assert [r[1] for r in batch.rendered] == [1]
    assert report['reviews'][0]['reason'] == 'object_collision_before_closure'
    assert next(candidates)[0] == 2
    assert [r[1] for r in batch.evaluated] == [0, 1, 2]
    assert len(batch.built) == 1
    candidates.close()
    assert not (tmp_path/'review/sim_rank_003').exists()


def test_live_candidate_screen_does_not_render_or_require_a_ur5e_video(batch,tmp_path):
    out = tmp_path/'live';out.mkdir()
    args = SimpleNamespace(max_candidates=4,execute=True)
    report = dict(status='running',reviews=[])
    candidates = review.simulate_candidates(args,out,report,batch.config,review.pin([batch.config]))
    rank,video,pins = next(candidates)
    assert rank == 1 and not video.exists()
    assert not batch.rendered
    assert not review.review_is_manual_only(video,pins)
    assert str((video.parent/'summary.json').resolve()) in pins
    assert str((video.parent/'trajectory_rank_001.csv').resolve()) in pins
    candidates.close()


def test_interactive_changed_inputs_stop_before_next_candidate(batch, tmp_path):
    candidates, report = iterator(batch, tmp_path/'review')
    next(candidates)
    batch.candidate_file.write_text(batch.candidate_file.read_text()+'\n# changed\n')
    with pytest.raises(ValueError, match='changed'):
        next(candidates)
    assert [r[1] for r in batch.evaluated] == [0, 1]


def test_interactive_unexpected_error_stops_and_retains_log(batch, tmp_path, monkeypatch):
    candidates, report = iterator(batch, tmp_path/'review')
    def fail(*a, **kw):
        raise RuntimeError('test solver error')
    monkeypatch.setattr(sim, 'evaluate_candidate', fail)
    with pytest.raises(RuntimeError, match='test solver error'):
        next(candidates)
    assert report['simulation']['stages'][0]['status'] == 'failed'
    assert 'test solver error' in (tmp_path/'review/sim_rank_000.log').read_text()
    assert not (tmp_path/'review/sim_rank_001').exists()


@pytest.mark.parametrize('answer,code,status', [('', 0, 'review_approved'), ('q', 2, 'cancel'),
                                               ('r', 2, 'scene_invalidated')])
def test_review_approval_or_cancel_never_computes_next_candidate(batch, tmp_path, monkeypatch, answer, code, status):
    monkeypatch.setattr(review, 'generate_candidates', lambda args,out,report:
        review.simulate_candidates(args, out, report, batch.config, review.pin([batch.config])))
    monkeypatch.setattr('builtins.input', lambda prompt: answer)
    out = tmp_path/'run'
    assert review.main(['--capture', str(tmp_path), '--dimensions-mm', '40', '90', '176',
                        '--no-open', '--out', str(out)]) == code
    assert review.read_json(out/'summary.json')['status'] == status
    assert [r[1] for r in batch.evaluated] == [0, 1]
    assert [r[1] for r in batch.rendered] == [1]


def reject_entire_batch(monkeypatch):
    original = sim.evaluate_candidate
    def rejected(model,start,candidate,rank,args,center):
        result,_ = original(model,start,candidate,rank,args,center)
        reason = {1:'ik_failed_pregrasp',3:'desk_clearance'}.get(rank,'object_collision_before_closure')
        desk = reason == 'desk_clearance'
        result = replace(result,passed=False,reason=reason,
            ik_position_error_mm=10. if rank==1 else 0.,
            min_desk_clearance_mm=-5. if desk else 10.,
            first_collision_phase='approach' if desk else '',
            min_object_clearance_mm=-5.,first_object_collision_phase='approach',samples_checked=3)
        samples = [sim.TrajectorySample(phase,0. if phase=='start' else 1.,start.copy()+rank,0.,
            min_desk_clearance_m=-.005 if desk else .01,desk_collision=desk,
            object_clearance_m=-.005,object_collision=True)
            for phase in ('start','start_to_pregrasp','approach')]
        return result,samples
    monkeypatch.setattr(sim,'evaluate_candidate',rejected)


def test_zero_pass_fallback_waits_for_whole_batch_then_renders_in_confidence_order(batch,tmp_path,monkeypatch):
    reject_entire_batch(monkeypatch)
    candidates,report = iterator(batch,tmp_path/'manual')
    rank,video,pins = next(candidates)
    assert rank == 0
    assert [x[1] for x in batch.evaluated] == [0,1,2,3]
    assert [x[1] for x in batch.rendered] == [0]
    assert len(batch.built) == 1
    assert report['manual_fallback']['eligible_ranks'] == [0,2]
    assert review.review_is_manual_only(video,pins)
    manifest = review.read_json(video.parent/'summary.json')
    assert not manifest['automatic_pass'] and not manifest['executable_target']
    assert manifest['original_result']['reason'] == 'object_collision_before_closure'
    assert not manifest['original_result']['passed']
    assert next(candidates)[0] == 2
    assert [x[1] for x in batch.evaluated] == [0,1,2,3]  # Never re-solve to render fallback.
    assert [x[1] for x in batch.rendered] == [0,2]
    assert list(candidates) == []


def test_fallback_can_be_disabled_without_rendering_or_changing_rejections(batch,tmp_path,monkeypatch):
    reject_entire_batch(monkeypatch)
    out = tmp_path/'disabled';out.mkdir()
    args = SimpleNamespace(max_candidates=4,fallback_review=False)
    report = dict(status='running',reviews=[])
    assert list(review.simulate_candidates(args,out,report,batch.config,review.pin([batch.config]))) == []
    assert [x[1] for x in batch.evaluated] == [0,1,2,3]
    assert not batch.rendered and 'manual_fallback' not in report


def test_no_fallback_after_operator_vetoes_an_automatic_pass(batch,tmp_path):
    candidates,report = iterator(batch,tmp_path/'passes')
    assert [x[0] for x in candidates] == [1,2,3]
    assert 'manual_fallback' not in report


def test_all_ik_or_desk_rejects_finish_without_a_manual_video(batch,tmp_path,monkeypatch):
    reject_entire_batch(monkeypatch)
    original = sim.evaluate_candidate
    def no_eligible(*args, **kwargs):
        result, samples = original(*args, **kwargs)
        if result.reason == 'object_collision_before_closure':
            result = replace(result, reason='ik_failed_approach')
        return result, samples
    monkeypatch.setattr(sim, 'evaluate_candidate', no_eligible)
    candidates, report = iterator(batch, tmp_path/'no_eligible')
    assert list(candidates) == []
    assert len(batch.evaluated) == 4
    assert not batch.rendered and 'manual_fallback' not in report


@pytest.mark.parametrize('changed', ['source', 'rejection'])
def test_inputs_changed_between_manual_candidates_stop_without_more_rendering(batch,tmp_path,monkeypatch,changed):
    reject_entire_batch(monkeypatch)
    candidates,report = iterator(batch,tmp_path/'manual')
    next(candidates)
    path = (batch.candidate_file if changed == 'source' else
            tmp_path/'manual/sim_rank_002/candidates.csv')
    path.write_text(path.read_text()+'\n# changed\n')
    with pytest.raises(ValueError,match='changed'):
        next(candidates)
    assert [x[1] for x in batch.rendered] == [0]


def test_manual_render_failure_stops_without_approval_or_real_handoff(batch,tmp_path,monkeypatch):
    from tools import run_ggx_real as real
    reject_entire_batch(monkeypatch)
    monkeypatch.setattr(review, 'generate_candidates', lambda args,out,report:
        review.simulate_candidates(args,out,report,batch.config,review.pin([batch.config])))
    def fail_render(*args, **kwargs):
        raise RuntimeError('test renderer unavailable')
    monkeypatch.setattr(sim, 'write_video', fail_render)
    monkeypatch.setattr(real, 'main', lambda *a: pytest.fail('No real handoff'))
    monkeypatch.setattr('builtins.input', lambda *a: pytest.fail('No video to approve'))
    out = tmp_path/'failed_render'
    # Manual collision diagnostics are now offline-only; execution suppresses
    # them entirely (covered separately), so exercise the renderer without --execute.
    assert review.main(['--capture',str(tmp_path),'--dimensions-mm','40','90','176',
                        '--no-open','--out',str(out)]) == 1
    report = review.read_json(out/'summary.json')
    assert report['status'] == 'failed' and 'renderer unavailable' in report['error']
    assert report['selected_rank'] is None and not report['robot_commands_sent']
    assert report['manual_fallback']['shown_ranks'] == []


@pytest.mark.parametrize('fault',['ik','desk','nan','incomplete','sample_desk','bad_orientation'])
def test_manual_fallback_never_includes_bad_ik_or_desk_paths(batch,monkeypatch,fault):
    reject_entire_batch(monkeypatch)
    args = sim.parse_args([])
    # Seed/test model are injected; this checks the production eligibility predicate.
    start = np.zeros(6); start.setflags(write=False)
    result,samples = sim.evaluate_candidate(object(),start,
        SimpleNamespace(name='test',confidence=.9),0,args,np.zeros(3))
    assert sim.manual_object_review_eligible(result,samples,args)
    if fault=='ik': result=replace(result,reason='ik_failed_approach')
    if fault=='desk': result=replace(result,min_desk_clearance_mm=1.)
    if fault=='nan': result=replace(result,ik_position_error_mm=float('nan'))
    if fault=='incomplete': samples=samples[:-1]
    if fault=='sample_desk': samples[1].desk_collision=True
    if fault=='bad_orientation': result=replace(result,ik_orientation_error_deg=5.)
    assert not sim.manual_object_review_eligible(result,samples,args)


@pytest.mark.parametrize('answers,status', [([''],'manual_review_accepted_not_executable'),
    (['n',''],'manual_review_accepted_not_executable'),(['q'],'cancel'),(['r'],'scene_invalidated')])
def test_fallback_ui_acceptance_or_cancel_cannot_handoff_to_real(batch,tmp_path,monkeypatch,answers,status):
    from tools import run_ggx_real as real
    reject_entire_batch(monkeypatch)
    monkeypatch.setattr(review,'generate_candidates',lambda args,out,report:
        review.simulate_candidates(args,out,report,batch.config,review.pin([batch.config])))
    monkeypatch.setattr(real,'main',lambda *a:pytest.fail('Fallback must never enter real runner'))
    choices=iter(answers);prompts=[]
    def answer(prompt):
        prompts.append(prompt)
        return next(choices)
    monkeypatch.setattr('builtins.input',answer)
    out=tmp_path/'run'
    code=review.main(['--capture',str(tmp_path),'--dimensions-mm','40','90','176',
                     '--no-open','--out',str(out)])
    report=review.read_json(out/'summary.json')
    assert code == (0 if status=='manual_review_accepted_not_executable' else 2)
    assert report['status']==status and not report['robot_commands_sent']
    assert all('不执行实机' in p for p in prompts)
    assert report['reviews'][-1]['scope']=='manual_object_collision'
    assert len(batch.evaluated)==4  # Only one batch, no second inference or solve.
    if status=='manual_review_accepted_not_executable':
        assert report['selected_rank']==(2 if len(answers)>1 else 0)
        with pytest.raises(ValueError,match='not a real-motion approval'):
            real.support.read_approval(out/'summary.json')


def test_manual_video_watermark_never_calls_collision_a_pass(monkeypatch):
    pytest.importorskip('PIL')
    from PIL import ImageDraw
    labels=[]
    original=ImageDraw.ImageDraw.text
    def record(self,xy,text,*args,**kwargs):
        labels.append(text)
        return original(self,xy,text,*args,**kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw,'text',record)
    result=sim.CandidateResult(6,'rank6',.8,False,'manual_object_collision_review',
        1.,1.,10.,'finger','',3,min_object_clearance_mm=-10.)
    sample=sim.TrajectorySample('approach',1.,np.zeros(6),0.,min_desk_clearance_m=.01,
        object_clearance_m=-.01,object_collision=True)
    frame=sim.annotate_review_frame(np.zeros((540,720,3),np.uint8),sample,result,.01)
    assert frame.shape==(540,720,3)
    assert any('MANUAL REVIEW ONLY' in t for t in labels)
    assert any('NOT approved for real motion' in t for t in labels)
    assert not any('PRECHECK PASS' in t for t in labels)
