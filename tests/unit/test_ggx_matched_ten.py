from types import SimpleNamespace
import json
import numpy as np
import pytest

from tools import run_ggx_matched_ten as m, ggx_matched_protocol as protocol


def fixture(monkeypatch):
    monkeypatch.setattr(m.template, 'relocate', lambda *a, **k: ({'pins': {}}, np.eye(4)))
    monkeypatch.setattr(m.template, 'canonical', lambda *a: (object(), []))
    args = SimpleNamespace(execute=True, video=False, open=False)
    contract = dict(config={'desk_envelope': {}, 'desk_edge_y_m': None}, hold_seconds=3.)
    pool = dict(feasibility_stages=[dict(rank=i, confidence=1-i/100,
        local_state='geometry_eligible' if i in (20, 26, 31) else 'local_rejected') for i in range(100)])
    candidates = {i: (None, None, None) for i in (20, 26, 31)}
    return args, contract, pool, candidates


def test_rank_order_first_feasible_and_no_extra_physical_attempt(monkeypatch, tmp_path):
    args, contract, pool, candidates = fixture(monkeypatch)
    calls = []
    def runner(a, c, s, out, record, checkpoint, **kw):
        calls.append(record['rank'])
        if record['rank'] == 20:
            raise m.planning.PlanningUnavailable('no_branch', [])
        record.update(status='commands_completed_grasp_unverified', motion_commands_attempted=True)
        return 0
    trial = dict(pins={})
    code=m.try_candidates(args,contract,pool,candidates,{},'scan',{'object_xy_mm':[0,0]},
                          tmp_path,trial,lambda:None,runner=runner)
    assert code == 0 and calls == [20,26]
    assert trial['selected_rank']==26
    assert len(trial['candidate_ledger']) == 100
    assert trial['candidate_ledger'][31]['status']=='not_tested_after_selected'


@pytest.mark.parametrize('flag', ['motion_commands_attempted','motion_confirmation_received','hand_commands','cleanup_errors'])
def test_never_retry_after_commands_authorization_or_cleanup_error(monkeypatch,tmp_path,flag):
    args,contract,pool,candidates=fixture(monkeypatch)
    calls=[]
    def runner(a,c,s,out,record,checkpoint,**kw):
        calls.append(record['rank']);record[flag]=True
        raise m.planning.PlanningUnavailable('failure_after_command', [])
    with pytest.raises(m.planning.PlanningUnavailable):
        m.try_candidates(args,contract,pool,candidates,{},'scan',{'object_xy_mm':[0,0]},
                         tmp_path,{'pins':{}},lambda:None,runner=runner)
    assert calls==[20]


def test_controller_or_localization_error_is_not_candidate_rejection(monkeypatch,tmp_path):
    args,contract,pool,candidates=fixture(monkeypatch)
    def runner(*a,**kw): raise RuntimeError('UR5 must be stationary')
    with pytest.raises(RuntimeError,match='stationary'):
        m.try_candidates(args,contract,pool,candidates,{},'scan',{'object_xy_mm':[0,0]},
                         tmp_path,{'pins':{}},lambda:None,runner=runner)


def test_cancellation_does_not_advance_rank(monkeypatch,tmp_path):
    args,contract,pool,candidates=fixture(monkeypatch)
    calls=[]
    def runner(a,c,s,out,record,checkpoint,**kw):
        calls.append(record['rank']);record['status']='cancelled_before_motion';return 2
    trial={'pins':{}}
    assert m.try_candidates(args,contract,pool,candidates,{},'scan',{'object_xy_mm':[0,0]},
                            tmp_path,trial,lambda:None,runner=runner)==2
    assert calls==[20] and trial['status']=='stopped'


def test_matching_translates_target_and_all_start_coordinates_only():
    point=dict(id='P1',tcp=[.1,.2,.3,1.,2.,3.],center_base_mm=[100.,200.,300.],q_nominal=[1]*6)
    layout=dict(points=[point])
    new,delta=protocol.matched_point(layout,'P1',[.2,.3,.4],[210.,280.])
    assert np.allclose(delta,[.01,-.02])
    assert np.allclose(new['tcp'][:3],[.11,.18,.3])
    assert new['tcp'][3:]==point['tcp'][3:]
    assert new['q_nominal']==point['q_nominal']
    assert point['tcp']==[.1,.2,.3,1.,2.,3.]


def test_shared_cloud_must_be_identical_path_and_hash(tmp_path):
    a,b=tmp_path/'a.npy',tmp_path/'b.npy'
    a.write_bytes(b'same');b.write_bytes(b'same')
    candidate=tmp_path/'c.yml';candidate.write_text('candidate')
    info=dict(status='candidates_generated_not_motion_validated',input_frame='ur_base',
              source_point_cloud=str(a),source_point_cloud_sha256=m.ui.digest(a),
              candidate_file=str(candidate),candidate_sha256=m.ui.digest(candidate))
    protocol.validate_cloud(info,a)
    with pytest.raises(ValueError,match='identical pinned Reflex cloud'):
        protocol.validate_cloud(info,b)


def test_hold_occurs_only_after_lift_and_records_completion(monkeypatch,tmp_path):
    events=[]
    monkeypatch.setattr(m.real.LiveBackend,'move_phase',lambda self,c:events.append(c['phase']))
    monkeypatch.setattr(m.real.LiveBackend,'dwell',lambda self,s:events.append(s))
    report={}
    backend=m.backend_with_hold(3.)({},tmp_path,report,lambda:None)
    backend.move_phase({'phase':'approach'})
    assert not report.get('lift_hold_completed')
    backend.set_stage('close')
    backend.move_phase({'phase':'lift'})
    assert events==['approach','lift',3.]
    assert report['grasp_close_command_attempted'] and report['lift_hold_completed']


def test_resume_does_not_retry_physical_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'last_lift_report',lambda *a: None)
    path=tmp_path/'summary.json'
    path.write_text(json.dumps(dict(schema=m.SCHEMA,status='stopped',execute_requested=True,pins={},
        reflex_report='reflex.json',pool_report='pool.json',trials=[dict(point_id='P1',
        status='commands_completed_grasp_unverified',operator_outcome='fail',attempts=[dict(motion_commands_attempted=True)])])))
    args=SimpleNamespace(resume_from=path,execute=True)
    carried=m.resume(args)
    assert len(carried)==1 and carried[0]['operator_outcome']=='fail'


def test_resume_blocks_partial_attempt(tmp_path):
    path=tmp_path/'summary.json'
    path.write_text(json.dumps(dict(schema=m.SCHEMA,status='failed',execute_requested=True,pins={},
        trials=[dict(point_id='P1',status='failed',attempts=[dict(hand_commands=[1])])])) )
    with pytest.raises(ValueError,match='Partial/authorized'):
        m.resume(SimpleNamespace(resume_from=path,execute=True))


def test_readonly_cli_default_and_live_requirements():
    a=m.parse_args(['--reflex-run','run.json','--validate-only'])
    assert not a.execute and not a.capture_live
    with pytest.raises(SystemExit):m.parse_args(['--reflex-run','run.json','--execute'])
    with pytest.raises(SystemExit):m.parse_args(['--reflex-run','run.json','--validate-only','--capture-live'])


def test_controller_authorization_covers_one_point_without_motion(monkeypatch):
    prompts=[]
    monkeypatch.setattr(m.real,'confirm',lambda prompt:prompts.append(prompt) or True)
    authority={}
    m.authorize_point_checks(False,{'id':'P1'},authority)
    assert not prompts
    m.authorize_point_checks(True,{'id':'P1'},authority)
    m.authorize_point_checks(True,{'id':'P1'},authority)
    assert len(prompts)==1 and authority['scope']=='candidate_search_no_motion'
    assert '不允许运动' in prompts[0]


def test_new_wrist_changes_only_seeds_not_cartesian_starts():
    from copy import deepcopy
    old=dict(settings={'seed_q':[0]*6},points=[dict(id=f'P{i}',tcp=[.1,.2,i/100,1.,2.,3.],
        q_nominal=[0]*6,center_base_mm=[1.,2.,i],offset_mm=[3.,4.,i]) for i in range(1,11)])
    original=deepcopy(old)
    seed=[.3,-1.7,1.9,.4,.3,-3.7]
    changed,meta=protocol.uniform_wrist_layout(old,dict(seed_q=seed,tcp_rotvec=[1.,2.,3.]),dict(q=seed))
    assert old==original
    assert changed['settings']['seed_q']==seed
    assert meta['original_q_nominal']=={f'P{i}':[0]*6 for i in range(1,11)}
    for before,after in zip(old['points'],changed['points']):
        assert after['q_nominal']==seed
        for key in ('tcp','center_base_mm','offset_mm','id'): assert before[key]==after[key]


def test_wrist_policy_rejects_scan_seed_or_tcp_orientation_change():
    layout=dict(settings={},points=[dict(id='P1',tcp=[0,0,0,1,2,3],q_nominal=[0]*6)])
    with pytest.raises(ValueError,match='seeds differ'):
        protocol.uniform_wrist_layout(layout,dict(seed_q=[0]*6,tcp_rotvec=[1,2,3]),dict(q=[1]*6))
    with pytest.raises(ValueError,match='TCP orientation'):
        protocol.uniform_wrist_layout(layout,dict(seed_q=[0]*6,tcp_rotvec=[0,0,0]),dict(q=[0]*6))


@pytest.mark.parametrize('commands', [False,True])
def test_resume_reset_preflight_failure_only_if_no_commands(tmp_path,commands):
    reset=tmp_path/'reset.json'
    reset.write_text(json.dumps(dict(schema='rh56_ggx_reset/v1',status='failed',
        motion_confirmation_received=True,motion_commands_attempted=commands,cleanup_errors=[],
        controller_preflight_timing={'status':'failed'},timings=[{'stage':'connect_controller'},{'stage':'preflight'}])))
    report=tmp_path/'summary.json'
    report.write_text(json.dumps(dict(schema=m.SCHEMA,status='stopped',execute_requested=True,pins={},
        reflex_report='reflex.json',pool_report='pool.json',trials=[dict(point_id='P1',status='reset_stopped',
        attempts=[],reset_report=str(reset))])))
    args=SimpleNamespace(resume_from=report,execute=True)
    if commands:
        with pytest.raises(ValueError,match='Incomplete reset'):m.resume(args)
    else: assert m.resume(args)==[]


@pytest.mark.parametrize('recover, motion, cleanup, allowed', [
    (False, False, [], False), (True, False, [], True),
    (True, True, [], False), (True, False, ['disconnect failed'], False)])
def test_explicit_manual_recovery_allows_hand_only_failed_reset(tmp_path, recover, motion, cleanup, allowed):
    reset = tmp_path/'reset.json'
    reset.write_text(json.dumps(dict(schema='rh56_ggx_reset/v1', status='failed',
        motion_commands_attempted=motion, hand_commands=[dict(kind='angles')], cleanup_errors=cleanup)))
    report = tmp_path/'summary.json'
    report.write_text(json.dumps(dict(schema=m.SCHEMA, status='stopped', execute_requested=True,
        pins={}, reflex_report='reflex.json', pool_report='pool.json', trials=[dict(
            point_id='P1', status='reset_stopped', attempts=[], reset_report=str(reset))])))
    args = SimpleNamespace(resume_from=report, execute=True, recover_from_current=recover)
    if allowed:
        assert m.resume(args) == []
        assert args.previous_lift_report is None
    else:
        with pytest.raises(ValueError, match='Incomplete reset'): m.resume(args)


def test_manual_recovery_cli_requires_live_execute_resume():
    for flags in [[], ['--execute', '--capture-live']]:
        with pytest.raises(SystemExit):
            m.parse_args(['--reflex-run', 'run.json', '--recover-from-current', *flags])
    args = m.parse_args(['--resume-from', 'run.json', '--recover-from-current', '--execute', '--capture-live'])
    assert args.recover_from_current
