"""Background aggregation never needs camera or robot access."""
import numpy as np
import pytest

from tools.capture_tabletop_background import summarize_depth, depth_quality, main


def test_median_ignores_invalid_depth_and_single_outliers():
    stack=np.full((9,2,2),600,dtype=np.uint16)
    stack[0,0,0]=0
    stack[1,0,0]=100
    depth,mad,count=summarize_depth(stack,.001)
    assert depth[0,0]==600 and mad[0,0]==0 and count[0,0]==8


def test_unsupported_pixels_remain_invalid():
    stack=np.full((9,2,2),600,dtype=np.uint16)
    stack[:,0,0]=0
    stack[:2,1,1]=0
    depth,mad,count=summarize_depth(stack,.001)
    assert depth[0,0]==0 and depth[1,1]==0
    assert count[0,0]==0 and np.isnan(mad[0,0])


def test_mad_units():
    stack=np.array([598,599,600,601,602],dtype=np.uint16).reshape(5,1,1)
    depth,mad,_=summarize_depth(stack,.001)
    assert depth.item()==600 and mad.item()==pytest.approx(1)


def test_existing_output_not_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        main(['--out',str(tmp_path),'--camera-serial','unused'])


def test_rgb_tracking_tolerates_missing_depth_without_changing_default():
    depth=np.zeros((10,10),np.uint16);mad=np.full((10,10),np.nan)
    with pytest.raises(RuntimeError,match='No sufficiently supported'):
        depth_quality(depth,mad,[0,0,10,10])
    result=depth_quality(depth,mad,[0,0,10,10],required=False)
    assert result['roi_supported_depth_fraction']==0
    assert result['roi_temporal_mad_median_mm'] is None


def test_rgb_tracking_mode_cannot_be_used_for_empty_background(tmp_path):
    out=tmp_path/'invalid'
    with pytest.raises(SystemExit):
        main(['--out',str(out),'--camera-serial','unused','--rgb-only-quality'])
    assert not out.exists()
