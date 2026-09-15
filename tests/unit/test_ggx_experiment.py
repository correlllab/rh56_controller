"""Fixed-start experimental orchestration with injected, hardware-free backends."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
pytest.importorskip('mujoco')
pytest.importorskip('mink')
pytest.importorskip('cv2')
from tools import run_ggx_experiment as experiment
from tools import ggx_template as template, ggx_fixed_starts as fixed


def test_default_is_offline_and_new_generation_needs_capture_dimensions():
    args=experiment.parse_args(['--template','template.json'])
    assert not args.execute and not args.capture_live and args.point=='all'
    for flags in ([],['--capture-live'],['--template','t.json','--execute'],
                  ['--template','t.json','--dimensions-mm','40','40','40']):
        with pytest.raises(SystemExit): experiment.parse_args(flags)


def test_ggx_venv_interpreter_symlink_not_resolved(tmp_path):
    path=tmp_path/'venv_python';path.symlink_to('/usr/bin/python3')
    args=experiment.parse_args(['--capture-live','--dimensions-mm','40','40','40','--ggx-python',str(path)])
    assert args.ggx_python == path.absolute()


def test_fixed_layout_never_depends_on_object_or_candidate(tmp_path):
    import yaml
    real=experiment.real
    c=real.load_config(real.ROOT/'configs/ggx_real_lab.yaml')
    # Synthetic setup fixture: never depend on untracked lab calibration artifacts.
    plane=tmp_path/'plane.yaml';plane.write_text('plane:\n  normal_base: [0, 0, 1]\n')
    cal=tmp_path/'calibration.yaml';cal.write_text('fixture: setup source hash only\n')
    perception=tmp_path/'perception.yaml'
    perception.write_text(yaml.safe_dump(dict(schema='rh56_tabletop_vision_config/v1',camera_serial='test',
        calibration=str(cal),desk_plane=str(plane),desk_z_mm=43.5,roi=[0,0,1280,720])))
    a=fixed.build(real.ROOT/'configs/ggx_ten_starts_lab.yaml',c,perception)
    b=fixed.build(real.ROOT/'configs/ggx_ten_starts_lab.yaml',c,perception)
    assert a['points']==b['points']
    assert len(a['points'])==10
    points=np.array([p['tcp'][:3] for p in a['points']])*1000
    offsets=np.array([p['offset_mm'] for p in template.paper_points()])
    assert np.allclose(points[:,None]-points[None,:],offsets[:,None]-offsets[None,:],atol=1e-10)
    assert a['points'][0]['thumb_clearance_mm']==pytest.approx(25.,abs=.05)
    assert all(p['setup_desk_clearance_mm'] >= 2 for p in a['points'])
    assert all(p['tcp'][3:]==a['points'][0]['tcp'][3:] for p in a['points'])


@pytest.mark.parametrize('state',[{'motion_commands_attempted':True},{'hand_commands':[{}]},
                                {'motion_confirmation_received':True},{'cleanup_errors':['bad']}])
def test_collision_cannot_skip_after_motion_or_cleanup_fault(state):
    exc=experiment.real.motion.PathCollision('collision',phase='approach',obstacle='object')
    assert not experiment.skip_collision(exc,state)


def test_generic_runtime_error_never_skips_even_if_text_mentions_collision():
    assert not experiment.skip_collision(RuntimeError('collision'),{})


@pytest.fixture
def flow(tmp_path,monkeypatch):
    args=experiment.parse_args(['--template','t.json','--capture-live','--execute','--allow-high-force'])
    plane=tmp_path/'plane.yaml';plane.write_text('plane:\n  normal_base: [0.01, 0.02, 1]\n')
    events=[]; faults={}; seen=[]; xy_calls=[]
    source=dict(rank=53,pins={},preflight=dict(object_center_m=[-.2,-.6,.14],object_yaw_deg=12.,
        object_size_mm=[90,40,200],arguments={}),perception=dict(config=str(args.perception_config),
        capture=str(tmp_path/'reference'),desk_plane=str(plane)))
    data=dict(open_grasp_center_in_tcp_m=[0,0,0],
              reference_object_T=template.object_matrix([-.2,-.6,.14],12.).tolist())
    reference=dict(rows=[dict(phase='approach',tcp=[-.2,-.6,.3,0,0,0])])
    points=[dict(id=f'P{i}',tcp=[-.2,-.4,.3+i*.01,0,0,0],center_base_mm=[-200,-400,300+i*10]) for i in range(1,11)]
    monkeypatch.setattr(fixed,'build',lambda *a:dict(points=deepcopy(points),source_sha256={}))
    monkeypatch.setattr(experiment.ten,'code_pins',lambda *a:{})
    monkeypatch.setattr(experiment.ten,'plot_layout',lambda *a:None)
    monkeypatch.setattr(template,'load',lambda p:(deepcopy(data),deepcopy(source),deepcopy(reference)))
    monkeypatch.setattr(template,'create',lambda *a:events.append('template'))
    monkeypatch.setattr(template,'canonical',lambda *a:(None,[]))
    def prepare(data,source,reference,model,common,point,*a,**kw):
        seen.append((deepcopy(point),deepcopy(source)))
    monkeypatch.setattr(template,'prepare_trial',prepare)
    def reset(args,out): events.append('reset'); return 0,args.start_snapshot
    def generate(args,out,snapshot): events.append('ggx');return tmp_path/'approval.json'
    def capture(args,source,out): events.append('capture');return tmp_path/'capture'
    def locate(source,capture,out):
        events.append('xy');xy_calls.append(out)
        if faults.get('xy'):raise ValueError('XY uncertain')
        value=dict(status='localized',object_xy_mm=[-200+len(xy_calls),-600],input_sha256={})
        out.mkdir();template.write_json(out/'summary.json',value);return value
    def real_run(a,c,s,o,r,checkpoint,*,prepare_fn):
        p=r['point_id'];events.append(p)
        prepare_fn(s,{},o)
        if p in faults:
            if faults[p]=='after_motion':
                r['motion_commands_attempted']=True
                raise experiment.real.motion.PathCollision('collision',phase='approach',obstacle='object')
            raise faults[p]
        r.update(status='commands_completed_grasp_unverified' if a.execute else 'offline_prepared',
            motion_confirmation_received=a.execute,motion_commands_attempted=a.execute)
        return 0
    report={}; out=tmp_path/'run';out.mkdir()
    def run():
        return experiment.run(args,out,report,lambda:None,reset_fn=reset,generate_fn=generate,
                              locate_fn=locate,capture_fn=capture,real_run=real_run)
    return SimpleNamespace(args=args,run=run,events=events,faults=faults,report=report,seen=seen,points=points)


def test_one_generation_then_ten_trials_remeasure_after_each_motion(flow):
    flow.args.template=None;flow.args.dimensions_mm=[40,40,40]
    assert flow.run()==0
    assert flow.events.count('ggx')==1 and flow.events.count('template')==1
    assert flow.events.count('reset')==10 and flow.events.count('xy')==10
    assert [p for p,s in flow.seen]==flow.points
    for p,source in flow.seen:
        assert source['preflight']['object_center_m'][2]==.14
        assert source['preflight']['object_yaw_deg']==12.
        assert source['preflight']['object_size_mm']==[90,40,200]


def test_collision_skips_without_extra_reset_capture_or_new_ggx(flow):
    flow.faults['P1']=experiment.real.motion.PathCollision('collision',phase='start_to_pregrasp',obstacle='object')
    flow.faults['P2']=experiment.real.motion.PathCollision('collision',phase='position_to_trial_start',obstacle='desk')
    assert flow.run()==0
    assert [r['status'] for r in flow.report['trials'][:2]]==['skipped_collision']*2
    assert flow.report['skipped_points']==2
    assert flow.events.count('reset')==8 and flow.events.count('xy')==8
    assert flow.events.count('ggx')==0


@pytest.mark.parametrize('fault',[RuntimeError('controller failed'),ValueError('IK failed'),'after_motion'])
def test_non_skippable_failure_stops_without_next_point(flow,fault):
    flow.faults['P1']=fault
    with pytest.raises((RuntimeError,ValueError)):flow.run()
    assert 'P2' not in flow.events


def test_failed_xy_stops_before_any_trial_no_cached_fallback(flow):
    flow.faults['xy']=True
    with pytest.raises(ValueError,match='uncertain'):flow.run()
    assert 'P1' not in flow.events


def test_offline_replay_never_resets_or_captures(flow):
    flow.args.execute=flow.args.capture_live=False
    assert flow.run()==0
    assert 'reset' not in flow.events and 'capture' not in flow.events


def test_invalid_saved_template_stops_before_reset(flow,monkeypatch):
    def invalid(path): raise ValueError('stale template')
    monkeypatch.setattr(template,'load',invalid)
    with pytest.raises(ValueError,match='stale'):flow.run()
    assert flow.events==[]


def test_saved_perception_mismatch_stops_before_reset(flow,monkeypatch):
    original_load=template.load
    def mismatch(path):
        data,source,reference=original_load(path)
        source['perception']['config']='/tmp/other_perception_config.yaml'
        return data,source,reference
    monkeypatch.setattr(template,'load',mismatch)
    with pytest.raises(ValueError,match='perception config differ'):flow.run()
    assert flow.events==[]


def test_trial_capture_does_not_require_current_depth(flow,monkeypatch,tmp_path):
    from tools import preview_tabletop_target as preview
    calls=[]
    monkeypatch.setattr(preview,'load_perception_config',lambda *a:({'camera_serial':'unused','roi':[0,0,1280,720]},None))
    monkeypatch.setattr(experiment.real,'confirm',lambda *a:True)
    monkeypatch.setattr(experiment.review,'run_stage',lambda *a:calls.append(a))
    experiment.capture_xy(flow.args,{},tmp_path)
    assert '--rgb-only-quality' in calls[0][-1]


def test_all_collisions_are_reported_not_success(flow):
    flow.faults.update({f'P{i}':experiment.real.motion.PathCollision('collision',phase='approach',obstacle='object') for i in range(1,11)})
    assert flow.run()==2
    assert flow.report['status']=='all_points_skipped'
    assert len(flow.report['trials'])==10


def test_generation_never_passes_execute_to_review(tmp_path):
    args=experiment.parse_args(['--capture-live','--dimensions-mm','40','40','40'])
    commands=[];out=tmp_path/'ggx';out.mkdir()
    def generate(command):
        commands.extend(command)
        template.write_json(out/'summary.json',dict(status='auto_candidate_selected',
                            selected_candidate=str(out/'selected_candidate.json')))
        return 0
    experiment.generate_once(args,out,args.start_snapshot,review_main=generate)
    assert '--execute' not in commands and '--freeze-reference-yaw' in commands
    assert commands[commands.index('--feasibility-mode')+1]=='official-cb3'
    assert commands[commands.index('--real-config')+1]==str(args.config)
    assert '--auto-select' in commands and '--no-fallback-review' in commands
    assert commands[commands.index('--max-candidates')+1]=='100'


def test_saved_batch_selection_does_not_recapture_or_regenerate(tmp_path):
    args=experiment.parse_args(['--candidate-run', str(tmp_path/'batch'),
        '--official-filter',str(tmp_path/'filter.json'),'--capture-live','--execute','--allow-high-force'])
    out=tmp_path/'selection';out.mkdir(); commands=[]
    def select(command):
        commands.extend(command)
        template.write_json(out/'summary.json',dict(status='auto_candidate_selected',
                            selected_candidate=str(out/'selected_candidate.json')))
        return 0
    assert experiment.generate_once(args,out,args.start_snapshot,review_main=select)==out/'selected_candidate.json'
    assert '--candidate-run' in commands and '--official-filter' in commands
    assert not {'--execute','--capture-live','--dimensions-mm','--background'} & set(commands)


@pytest.mark.parametrize('extra', [['--execute'], ['--reset-first'], ['--max-candidates','50'],
                                  ['--feasibility-mode','official-only']])
def test_automatic_selector_cannot_enable_motion_or_change_budget(extra):
    with pytest.raises(SystemExit):
        experiment.review.parse_args(['--candidate-run','batch','--auto-select',
                                      '--feasibility-mode','official-cb3',*extra])


def test_saved_batch_selection_failure_happens_before_reset(flow, tmp_path):
    flow.args.template=None
    flow.args.candidate_run=Path('batch')
    def invalid(*a):
        raise ValueError('no strict pass')
    with pytest.raises(ValueError,match='no strict pass'):
        experiment.run(flow.args,tmp_path,{},lambda:None,
                       generate_fn=invalid,reset_fn=lambda *a:pytest.fail('reset before selection'))


@pytest.mark.parametrize('saved', [False, True])
def test_manual_selection_uses_strict_review_without_motion(tmp_path, saved):
    flags = (['--candidate-run',str(tmp_path/'batch'),'--official-filter',str(tmp_path/'filter.json')]
             if saved else ['--dimensions-mm','40','90','176'])
    args=experiment.parse_args([*flags,'--manual-select','--capture-live','--execute','--allow-high-force'])
    out=tmp_path/'selection';out.mkdir()
    def select(command):
        parsed=experiment.review.parse_args(command)
        assert not parsed.execute and not parsed.reset_first and not parsed.auto_select
        assert not parsed.fallback_review and parsed.max_candidates==100
        assert parsed.feasibility_mode=='official-cb3'
        assert bool(parsed.candidate_run)==saved
        template.write_json(out/'summary.json',dict(status='review_approved',selected_rank=26))
        return 0
    assert experiment.generate_once(args,out,args.start_snapshot,review_main=select)==out/'summary.json'


@pytest.mark.parametrize('status', ['manual_review_accepted_not_executable','cancelled','no_candidate_approved'])
def test_manual_selection_does_not_promote_diagnostic_or_cancelled_review(tmp_path,status):
    args=experiment.parse_args(['--capture-live','--dimensions-mm','40','90','176','--manual-select'])
    def select(command):
        template.write_json(tmp_path/'summary.json',dict(status=status))
        return 0
    with pytest.raises(RuntimeError,match='No strict GGX template'):
        experiment.generate_once(args,tmp_path,args.start_snapshot,review_main=select)


def test_manual_selection_cannot_change_frozen_template():
    with pytest.raises(SystemExit):
        experiment.parse_args(['--template','old.json','--manual-select'])
