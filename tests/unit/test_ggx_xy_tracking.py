"""RGB translation and metric XY tests; no camera or robot access."""
import numpy as np
import pytest
cv2 = pytest.importorskip('cv2')
pytest.importorskip('mujoco')
pytest.importorskip('mink')
from tools import ggx_xy_tracking as xy


def scene():
    image = np.full((180,220,3), 30, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[65:110,85:130] = 1
    image[mask != 0] = [40,200,230]
    cv2.line(image,(92,73),(117,97),(10,20,40),3)
    return image, mask


@pytest.mark.parametrize('shift', [(0,0), (9,-6), (-7,5)])
def test_translation_without_rotation_or_scale(shift):
    image,mask=scene()
    current=cv2.warpAffine(image,np.array([[1.,0,shift[0]],[0,1.,shift[1]]]),
                           (image.shape[1],image.shape[0]),borderValue=(30,30,30))
    measured,quality=xy.match_translation(image,current,mask,search_px=25)
    assert measured == pytest.approx(shift,abs=.1)
    assert quality['correlation'] > .99


def test_uniform_cube_uses_silhouette_not_whole_background():
    image,mask=scene()
    image[mask != 0] = [40,200,230]
    current=cv2.warpAffine(image,np.array([[1.,0,7],[0,1.,4]]),(220,180),borderValue=(30,30,30))
    measured,_=xy.match_translation(image,current,mask,search_px=25)
    assert measured == pytest.approx([7,4],abs=.1)


def test_large_translation_needs_larger_search_not_lower_confidence():
    image=np.full((400,600,3),30,np.uint8)
    mask=np.zeros(image.shape[:2],np.uint8);mask[180:220,280:320]=1
    image[mask>0]=[40,200,230]
    cv2.line(image,(285,185),(315,215),(10,20,40),3)
    current=cv2.warpAffine(image,np.array([[1.,0,-82],[0,1.,-1]]),(600,400),borderValue=(30,30,30))
    with pytest.raises(ValueError,match='uncertain'):
        xy.match_translation(image,current,mask,search_px=60)
    shift,quality=xy.match_translation(image,current,mask,search_px=120)
    assert shift==pytest.approx([-82,-1],abs=.1)
    assert quality['correlation']>.99


def test_missing_object_cannot_reuse_old_xy():
    image,mask=scene()
    with pytest.raises(ValueError,match='uncertain'):
        xy.match_translation(image,np.full_like(image,30),mask,search_px=25)


def test_blank_reference_cannot_be_localized():
    image,mask=scene()
    with pytest.raises(ValueError,match='contrast'):
        xy.match_translation(np.full_like(image,30),image,mask,search_px=25)


def test_reference_depth_and_fixed_height_give_metric_xy():
    # Downward camera, 0.8 m depth. No current depth input exists in this API.
    T=np.diag([1.,-1.,-1.,1.]); T[2,3]=1.
    pixels=np.c_[np.linspace(50,100,50),np.linspace(60,110,50)]
    K=np.array([[500,0,100],[0,500,90],[0,0,1.]])
    delta,spread=xy.metric_translation(pixels,np.full(50,.8),[5,-2],K,np.zeros(5),T)
    assert delta == pytest.approx([8.,3.2],abs=.001)
    assert spread < .001


@pytest.mark.parametrize('depths', [np.ones(29),np.zeros(30),np.full(30,np.nan)])
def test_invalid_reference_depth_rejected(depths):
    with pytest.raises(ValueError,match='Invalid reference'):
        xy.metric_translation(np.ones((30,2)),depths,[1,2],np.eye(3),np.zeros(5),np.eye(4))


def test_full_xy_locator_never_reads_current_depth(tmp_path,monkeypatch):
    import yaml
    from tools import preview_tabletop_target as preview
    ref,cur=tmp_path/'reference',tmp_path/'current'
    ref.mkdir();cur.mkdir()
    image=np.full((720,1280,3),30,np.uint8)
    mask=np.zeros((720,1280),np.uint8);mask[320:370,600:650]=255
    image[mask>0]=[40,200,230]
    shifted=cv2.warpAffine(image,np.array([[1.,0,5],[0,1.,-2]]),(1280,720),borderValue=(30,30,30))
    cv2.imwrite(str(ref/'camera_check_rgb.png'),image)
    cv2.imwrite(str(cur/'camera_check_rgb.png'),shifted)
    cv2.imwrite(str(ref/'camera_check_depth_u16.png'),np.full((720,1280),800,np.uint16))
    cv2.imwrite(str(ref/'foreground.png'),mask)
    T=np.diag([1.,-1.,-1.,1.]);T[2,3]=1.
    K=[[500,0,640],[0,500,360],[0,0,1]]
    camera=dict(serial='test',stream=dict(width=1280,height=720),
                factory_color_intrinsics={},depth=dict(scale_m_per_unit=.001))
    calibration=tmp_path/'calibration.yaml'
    calibration.write_text(yaml.safe_dump(dict(schema='rh56_ur5_external_camera_calibration/v2',
        camera=camera,selected_intrinsics=dict(camera_matrix=K,distortion_coefficients=[0]*5),
        transforms=dict(base_from_camera=dict(matrix=T.tolist())))))
    for directory in (ref,cur):
        (directory/'camera_check.yaml').write_text(yaml.safe_dump(dict(
            schema='rh56_realsense_camera_check/v1',camera=camera)))
    monkeypatch.setattr(preview,'load_perception_config',lambda *a:(dict(calibration=str(calibration),camera_serial='test'),None))
    source=dict(pins={},perception=dict(config='config.yaml',capture=str(ref),
        candidates=[dict(component_id=1,geometry_candidate=True,rejection_reasons=[])]),
        crop=dict(reviewed_component_id=1,source_preview=str(ref/'summary.json')),
        preflight=dict(object_center_m=[.1,.2,.2]))
    assert not (cur/'camera_check_depth_u16.png').exists()
    result=xy.locate(source,cur,tmp_path/'xy')
    assert result['status']=='localized' and result['current_depth_used'] is False
    assert result['object_xy_mm']==pytest.approx([108.,203.2],abs=.15)
    assert str(cur/'camera_check_depth_u16.png') not in result['input_sha256']


def test_symmetric_cube_can_freeze_initial_axis_without_claiming_measured_yaw():
    from tools import prepare_ggx_tabletop_cloud as cloud
    y,x=np.mgrid[-20:21,-20:21]
    points=np.c_[x.ravel(),y.ravel(),np.full(x.size,74)]
    frame,proxy,warnings=cloud.foreground_geometry(points,[40,40,40],[0,0,1],34,
                                                  freeze_reference_yaw=True)
    assert frame['yaw_ambiguous'] is True and frame['yaw_policy']=='frozen_reference_axis'
    assert warnings and np.all(np.array(proxy['dimensions_mm']) >= [40,40,40])
    report=dict(schema=cloud.FOREGROUND_SCHEMA,crop_mode='foreground',
        status='measured_crop_ready_for_offline_inference',hardware_contacted=False,
        executable_target=False,object_frame=frame,collision_proxy=proxy,
        dimensions_prior_mm=[40,40,40],reference_yaw_frozen=True)
    cloud.validate_foreground_report(report)
    report['reference_yaw_frozen']=False
    with pytest.raises(ValueError,match='Invalid foreground'):
        cloud.validate_foreground_report(report)
