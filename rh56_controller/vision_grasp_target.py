"""Adapt saved colored-target measurements to the existing UR5 Plan inputs.

No camera, robot, or hand interfaces are imported here. Width and yaw are
operator inputs: the colored-target detector only measures a top-center point.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class VisionGraspTarget:
    source: str
    center_base_m: tuple[float, float, float]
    object_width_mm: float
    width_offset_mm: float
    depth_below_top_mm: float
    yaw_deg: float
    center_policy: str = "antipodal"

    @property
    def planner_width_m(self) -> float:
        return (self.object_width_mm + self.width_offset_mm) / 1000.0

    def metadata(self) -> dict:
        return {
            **asdict(self),
            "planner_width_m": self.planner_width_m,
            "frame": "UR base",
            "assumptions": [
                "Saved measurement; loading does not check current scene or camera drift.",
                "Upright object: grasp center is a base-Z offset below the measured top.",
                "Object width and base-frame yaw are supplied, not estimated by the detector.",
                "Width offset is explicit; it is not a universal physical calibration.",
            ],
        }


def load_vision_target(path, *, object_width_mm, yaw_deg,
                       width_offset_mm=0.0, depth_below_top_mm=None):
    path = Path(path)
    report = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("schema") != "rh56_ur5_camera_colored_target_validation/v1":
        raise ValueError("Expected a colored-target target_validation.yaml (v1)")
    if report.get("quality_pass") is not True:
        raise ValueError("Camera target measurement did not pass its quality checks")
    top = np.asarray(report["estimated_target_top_center_base_mm"], dtype=float)
    height = float(report["physical_reference"]["target_height_mm"])
    depth = height / 2.0 if depth_below_top_mm is None else float(depth_below_top_mm)
    width, offset, yaw = map(float, (object_width_mm, width_offset_mm, yaw_deg))
    if top.shape != (3,) or not np.all(np.isfinite(top)):
        raise ValueError("Target top center must contain three finite coordinates")
    if not np.all(np.isfinite([height, depth, width, offset, yaw])):
        raise ValueError("Target dimensions and yaw must be finite")
    if height <= 0 or width <= 0 or width + offset <= 0:
        raise ValueError("Height, object width, and planner width must be positive")
    if not 0 < depth < height:
        raise ValueError("Grasp depth must lie strictly between top and bottom")
    center = top.copy()
    center[2] -= depth
    yaw = (yaw + 180.0) % 360.0 - 180.0
    return VisionGraspTarget(str(path.resolve()), tuple(center / 1000.0),
                             width, offset, depth, yaw)


def add_vision_target_arguments(parser):
    parser.add_argument("--vision-target", help="Saved colored-target target_validation.yaml")
    parser.add_argument("--object-width-mm", type=float,
                        help="Physical grasp width; required with --vision-target")
    parser.add_argument("--vision-yaw-deg", type=float,
                        help="Grasp-plane yaw in UR base; required with --vision-target")
    parser.add_argument("--planner-width-offset-mm", type=float, default=0.0,
                        help="Explicit addition to object width (default: 0; no implicit 20 mm)")
    parser.add_argument("--grasp-depth-mm", type=float,
                        help="Base-Z depth below target top (default: half reported height)")


def vision_target_from_args(args):
    if args.vision_target is None:
        if any(v is not None for v in (args.object_width_mm, args.vision_yaw_deg, args.grasp_depth_mm)) or args.planner_width_offset_mm != 0:
            raise ValueError("Vision dimensions/yaw require --vision-target")
        return None
    if args.object_width_mm is None or args.vision_yaw_deg is None:
        raise ValueError("--vision-target requires --object-width-mm and --vision-yaw-deg")
    return load_vision_target(
        args.vision_target, object_width_mm=args.object_width_mm,
        yaw_deg=args.vision_yaw_deg, width_offset_mm=args.planner_width_offset_mm,
        depth_below_top_mm=args.grasp_depth_mm,
    )
