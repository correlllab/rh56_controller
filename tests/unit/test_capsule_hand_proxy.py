import numpy as np

from rh56_controller.capsule_hand_proxy import Capsule, sample_linear_path_collisions


def test_near_final_ignore_does_not_hide_palm_collision():
    half_extents = np.array([0.05, 0.05, 0.05])
    capsules = [
        Capsule(
            name="thumb_test",
            group="thumb",
            p0=np.array([0.0, 0.0, 0.0]),
            p1=np.array([0.02, 0.0, 0.0]),
            radius=0.01,
            source="test",
        ),
        Capsule(
            name="palm_test",
            group="palm",
            p0=np.array([0.0, 0.0, 0.0]),
            p1=np.array([0.0, 0.02, 0.0]),
            radius=0.01,
            source="test",
        ),
    ]

    rows = sample_linear_path_collisions(
        capsules,
        start=np.zeros(3),
        final=np.zeros(3),
        rotation=np.eye(3),
        half_extents=half_extents,
        path_samples=2,
        final_ignore_m=1.0,
        final_ignore_groups=("thumb",),
    )

    assert rows[0]["raw_collision_count"] == 2
    assert rows[0]["ignored_collision_count"] == 1
    assert rows[0]["active_collision_count"] == 1
    assert rows[0]["collision"] is True
    assert rows[0]["nearest_group"] == "palm"


def test_near_final_ignore_can_mask_fingertip_contact_only():
    half_extents = np.array([0.05, 0.05, 0.05])
    capsules = [
        Capsule(
            name="thumb_test",
            group="thumb",
            p0=np.array([0.0, 0.0, 0.0]),
            p1=np.array([0.02, 0.0, 0.0]),
            radius=0.01,
            source="test",
        )
    ]

    rows = sample_linear_path_collisions(
        capsules,
        start=np.zeros(3),
        final=np.zeros(3),
        rotation=np.eye(3),
        half_extents=half_extents,
        path_samples=2,
        final_ignore_m=1.0,
        final_ignore_groups=("thumb",),
    )

    assert rows[0]["raw_collision_count"] == 1
    assert rows[0]["ignored_collision_count"] == 1
    assert rows[0]["active_collision_count"] == 0
    assert rows[0]["collision"] is False
    assert rows[0]["ignored_for_final_contact"] is True


def test_swept_path_detects_collision_between_endpoints():
    half_extents = np.array([0.02, 0.02, 0.02])
    capsules = [
        Capsule(
            name="index_test",
            group="index",
            p0=np.array([0.0, 0.0, 0.0]),
            p1=np.array([0.0, 0.0, 0.005]),
            radius=0.005,
            source="test",
        )
    ]

    rows = sample_linear_path_collisions(
        capsules,
        start=np.array([-0.08, 0.0, 0.0]),
        final=np.array([0.08, 0.0, 0.0]),
        rotation=np.eye(3),
        half_extents=half_extents,
        path_samples=2,
        final_ignore_m=0.0,
        final_ignore_groups=("thumb", "index", "middle", "ring", "pinky"),
    )

    assert rows[0]["collision"] is False
    assert rows[1]["path_check"] == "swept_interval"
    assert rows[1]["collision"] is True
    assert 0.0 < rows[1]["alpha"] < 1.0


def test_shifted_aabb_center_is_used_for_tabletop_object():
    half_extents = np.array([0.02, 0.02, 0.02])
    capsules = [
        Capsule(
            name="palm_test",
            group="palm",
            p0=np.array([0.0, 0.0, 0.03]),
            p1=np.array([0.0, 0.0, 0.035]),
            radius=0.005,
            source="test",
        )
    ]

    rows = sample_linear_path_collisions(
        capsules,
        start=np.zeros(3),
        final=np.zeros(3),
        rotation=np.eye(3),
        half_extents=half_extents,
        aabb_center=np.array([0.0, 0.0, 0.02]),
        path_samples=2,
        final_ignore_m=0.0,
        final_ignore_groups=("thumb", "index", "middle", "ring", "pinky"),
    )

    assert rows[0]["collision"] is True
    assert rows[0]["nearest_group"] == "palm"


def test_cylinder_proxy_does_not_include_aabb_corners():
    half_extents = np.array([0.05, 0.05, 0.10])
    capsules = [
        Capsule(
            name="index_test",
            group="index",
            p0=np.array([0.045, 0.045, 0.0]),
            p1=np.array([0.045, 0.045, 0.01]),
            radius=0.005,
            source="test",
        )
    ]

    box_rows = sample_linear_path_collisions(
        capsules,
        start=np.zeros(3),
        final=np.zeros(3),
        rotation=np.eye(3),
        half_extents=half_extents,
        object_shape="box",
        path_samples=2,
        final_ignore_m=0.0,
        final_ignore_groups=(),
    )
    cylinder_rows = sample_linear_path_collisions(
        capsules,
        start=np.zeros(3),
        final=np.zeros(3),
        rotation=np.eye(3),
        half_extents=half_extents,
        object_shape="cylinder",
        path_samples=2,
        final_ignore_m=0.0,
        final_ignore_groups=(),
    )

    assert box_rows[0]["collision"] is True
    assert cylinder_rows[0]["collision"] is False
    assert cylinder_rows[0]["clearance_m"] > 0.0


def test_sphere_proxy_does_not_include_aabb_corners():
    half_extents = np.array([0.05, 0.05, 0.05])
    capsules = [
        Capsule(
            name="thumb_test",
            group="thumb",
            p0=np.array([0.04, 0.04, 0.04]),
            p1=np.array([0.04, 0.04, 0.04]),
            radius=0.005,
            source="test",
        )
    ]

    rows = sample_linear_path_collisions(
        capsules,
        start=np.zeros(3),
        final=np.zeros(3),
        rotation=np.eye(3),
        half_extents=half_extents,
        object_shape="sphere",
        path_samples=2,
        final_ignore_m=0.0,
        final_ignore_groups=(),
    )

    assert rows[0]["collision"] is False
    assert rows[0]["clearance_m"] > 0.0


def test_swept_cylinder_path_detects_between_endpoint_collision():
    half_extents = np.array([0.02, 0.02, 0.04])
    capsules = [
        Capsule(
            name="index_test",
            group="index",
            p0=np.zeros(3),
            p1=np.array([0.0, 0.0, 0.005]),
            radius=0.005,
            source="test",
        )
    ]

    rows = sample_linear_path_collisions(
        capsules,
        start=np.array([-0.08, 0.0, 0.0]),
        final=np.array([0.08, 0.0, 0.0]),
        rotation=np.eye(3),
        half_extents=half_extents,
        object_shape="cylinder",
        path_samples=2,
        final_ignore_m=0.0,
    )

    assert rows[0]["collision"] is False
    assert rows[1]["collision"] is True
    assert 0.0 < rows[1]["alpha"] < 1.0


def test_numerical_tangency_is_not_counted_as_penetration():
    half_extents = np.array([0.02, 0.02, 0.02])
    capsules = [
        Capsule(
            name="thumb_contact",
            group="thumb",
            p0=np.array([0.025 - 5e-6, 0.0, 0.0]),
            p1=np.array([0.025 - 5e-6, 0.0, 0.0]),
            radius=0.005,
            source="test",
        )
    ]

    rows = sample_linear_path_collisions(
        capsules,
        start=np.zeros(3),
        final=np.zeros(3),
        rotation=np.eye(3),
        half_extents=half_extents,
        path_samples=2,
        final_ignore_m=0.0,
        final_ignore_groups=(),
    )

    assert -1e-5 < rows[0]["clearance_m"] < 0.0
    assert rows[0]["collision"] is False
