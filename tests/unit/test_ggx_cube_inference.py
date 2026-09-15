"""GGX runner contracts without installing torch or contacting hardware."""
import json

import pytest

from tools import run_ggx_cube_inference as runner


def test_trimesh_style_subclass_normalized_for_official_sampler():
    import numpy as np
    class TrackedArray(np.ndarray):
        pass
    original = np.zeros((5,3)).view(TrackedArray)
    assert type(runner.plain_point_cloud(original)) is np.ndarray
    with pytest.raises(ValueError):
        runner.plain_point_cloud([[float('nan'),0,0]])


@pytest.mark.parametrize('extra', [['--topk','201'], ['--num-grasps','0'], ['--seed','-1']])
def test_bad_args(extra):
    with pytest.raises(SystemExit):
        runner.parse_args(['--out','unused',*extra])


def test_failure_writes_summary_without_loading_gpu(tmp_path):
    out = tmp_path/'run'
    assert runner.main(['--graspgenx-root',str(tmp_path/'missing'),'--out',str(out)]) == 1
    report = json.loads((out/'summary.json').read_text())
    assert report['status']=='failed' and not report['hardware_commands_sent']
    assert (out/'summary.csv').is_file()


def test_success_summary_and_no_overwrite(tmp_path, monkeypatch):
    def infer(args, report):
        report.update(status='candidates_generated_not_motion_validated', candidates=100)
    monkeypatch.setattr(runner, 'infer', infer)
    out = tmp_path/'run'
    assert runner.main(['--out',str(out)]) == 0
    report = json.loads((out/'summary.json').read_text())
    assert report['candidates']==100 and report['cube_width_mm']==40
    with pytest.raises(FileExistsError):
        runner.main(['--out',str(out)])


@pytest.mark.parametrize('extra',[['--point-cloud','x.npy'],['--input-frame','ur_base']])
def test_point_cloud_requires_explicit_frame(extra):
    with pytest.raises(SystemExit):runner.parse_args(['--out','unused',*extra])


def test_measured_cloud_preserves_base_frame_and_points(tmp_path):
    import numpy as np
    points=np.random.default_rng(0).uniform([-.3,-.65,.15],[-.2,-.6,.22],(120,3))
    source=tmp_path/'input.npy';np.save(source,points)
    out=tmp_path/'out';out.mkdir()
    a=runner.parse_args(['--out',str(out),'--point-cloud',str(source),'--input-frame','ur_base'])
    report={};actual=runner.inference_points(a,report)
    np.testing.assert_array_equal(actual,points)
    np.testing.assert_array_equal(np.load(out/'object_points_input_frame_m.npy'),points)
    assert not (out/'cube_40mm.obj').exists()
    assert report['input_points']==120


@pytest.mark.parametrize('fault',['few','nan','millimetres','object_array'])
def test_bad_measured_cloud_rejected(tmp_path,fault):
    import numpy as np
    points=np.random.default_rng(0).uniform(0,.1,(120,3))
    if fault=='few':points=points[:5]
    if fault=='nan':points[0,0]=np.nan
    if fault=='millimetres':points*=1000
    if fault=='object_array':points=points.astype(object)
    source=tmp_path/'input.npy';np.save(source,points)
    a=runner.parse_args(['--out',str(tmp_path),'--point-cloud',str(source),'--input-frame','ur_base'])
    with pytest.raises(ValueError):runner.inference_points(a,{})


def test_point_cloud_summary_never_claims_synthetic_cube(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'infer',lambda a,r:r.update(status='test_only',candidates=3))
    out=tmp_path/'out'
    assert runner.main(['--out',str(out),'--point-cloud','fake.npy','--input-frame','ur_base'])==0
    r=json.loads((out/'summary.json').read_text())
    assert r['schema']=='rh56_ggx_point_cloud_inference/v1' and r['cube_width_mm'] is None
    assert r['input_frame']=='ur_base' and not r['hardware_commands_sent']
