import numpy as np
import pytest

from tools.run_h12_closure_plane_sensitivity import (
    detect_knee_width,
    nearest_width_index,
    reference_specs,
)


def test_reference_specs_perturb_each_local_axis_symmetrically():
    specs = reference_specs(5.0)

    assert specs[0].name == "baseline"
    np.testing.assert_array_equal(specs[0].offset_local_m, np.zeros(3))
    offsets_mm = np.vstack([spec.offset_local_m for spec in specs[1:]]) * 1000.0
    assert {tuple(row) for row in offsets_mm} == {
        (-5.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
        (0.0, -5.0, 0.0),
        (0.0, 5.0, 0.0),
        (0.0, 0.0, -5.0),
        (0.0, 0.0, 5.0),
    }


def test_nearest_width_index_supports_descending_closure_schedule():
    widths = np.array([110.0, 80.0, 50.0, 39.0, 20.0])

    assert nearest_width_index(widths, 40.0) == 3


def test_detect_knee_returns_first_threshold_crossing_in_closure_order():
    widths = np.array([100.0, 80.0, 60.0, 40.0, 30.0, 20.0])
    rotation = np.array([0.0, 4.0, 8.0, 12.0, 24.0, 44.0])

    knee, rate = detect_knee_width(
        widths,
        rotation,
        threshold_deg_per_mm=1.0,
    )

    assert knee == pytest.approx(30.0)
    assert rate.shape == widths.shape
