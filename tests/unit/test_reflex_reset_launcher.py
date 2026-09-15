from tools import run_reflex_reset as reset


def test_default_reset_launcher_has_no_hardware_access(tmp_path,monkeypatch):
    assert reset.parse_args([]).start_snapshot==reset.ROOT/'configs/reflex_scan_wrist_flip.json'
    monkeypatch.setattr(reset.experiment,'reset_robot',lambda *a:(_ for _ in ()).throw(AssertionError('no hardware')))
    assert reset.main(['--out',str(tmp_path/'out')])==0
    assert (tmp_path/'out/summary.json').is_file()


def test_explicit_reset_delegates_only_to_existing_confirmed_reset(tmp_path,monkeypatch):
    calls=[]
    def run(args,out):
        calls.append((args.execute,args.allow_high_force,args.config,out))
        return 2,None
    monkeypatch.setattr(reset.experiment,'reset_robot',run)
    assert reset.main(['--execute','--allow-high-force','--out',str(tmp_path/'out')])==2
    assert calls==[(True,True,reset.ROOT/'configs/reflex_real_lab.yaml',tmp_path/'out')]
