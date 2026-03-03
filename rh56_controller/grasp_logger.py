"""
grasp_logger.py — Structured JSONL execution log for grasp experiments.

Each GRASP! execution creates one file:  logs/<name>_<YYYYMMDD_HHMMSS>.jsonl

Event types written (one JSON object per line):
  meta        — grasp parameters at execution start
  waypoint    — per-step arm pose + EEF error + sensor readings
  force_iter  — per-iteration force control state
  done        — final status on completion or abort
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class GraspLogger:
    """Write structured JSONL logs for a single grasp execution."""

    def __init__(self, name: str, log_dir: str = "logs"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        self.path = Path(log_dir) / f"{safe_name}_{ts}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w")
        self._t0   = time.monotonic()

    def log(self, event: str, **data):
        """Write one event record to the log (thread-safe per file)."""
        rec = {"t": round(time.monotonic() - self._t0, 4), "event": event}
        rec.update(data)
        self._file.write(json.dumps(rec, default=_json_default) + "\n")
        self._file.flush()

    def log_meta(self, *, mode: str, width_target_m: float, grasp_z: float,
                 grasp_x: float = 0.0, grasp_y: float = 0.0,
                 plane_rx: float = 0.0, plane_ry: float = 0.0, plane_rz: float = 0.0,
                 force_N: float = 0.0, step_mm: float = 10.0,
                 approach_m: Optional[float] = None,
                 strategy: str = "Plan", name: str = ""):
        self.log("meta",
                 mode=mode, width_target_m=width_target_m,
                 grasp_z=grasp_z, grasp_x=grasp_x, grasp_y=grasp_y,
                 plane_rx=plane_rx, plane_ry=plane_ry, plane_rz=plane_rz,
                 force_N=force_N, step_mm=step_mm,
                 approach_m=approach_m, strategy=strategy, name=name)

    def log_waypoint(self, *, step_i: int, n_steps: int,
                     desired_T=None, actual_T=None,
                     pos_err_mm: Optional[float] = None,
                     rot_err_deg: Optional[float] = None,
                     joint_q=None, finger_angles=None, finger_forces=None):
        def _mat(T):
            if T is None:
                return None
            import numpy as np
            T = np.asarray(T)
            return {"pos": T[:3, 3].tolist(), "mat": T[:3, :3].tolist()}
        self.log("waypoint",
                 step_i=step_i, n_steps=n_steps,
                 desired=_mat(desired_T), actual=_mat(actual_T),
                 pos_err_mm=pos_err_mm, rot_err_deg=rot_err_deg,
                 joint_q=list(joint_q) if joint_q is not None else None,
                 finger_angles=list(finger_angles) if finger_angles is not None else None,
                 finger_forces=list(finger_forces) if finger_forces is not None else None)

    def log_force_iter(self, *, iteration: int, forces, angles,
                       thresholds=None, forces_N=None, thresholds_N=None):
        self.log("force_iter",
                 iteration=iteration,
                 forces=list(forces) if forces is not None else None,
                 forces_N=[round(v, 4) for v in forces_N] if forces_N is not None else None,
                 angles=list(angles) if angles is not None else None,
                 thresholds=list(thresholds) if thresholds is not None else None,
                 thresholds_N=[round(v, 4) for v in thresholds_N] if thresholds_N is not None else None)

    def log_done(self, *, strategy: str, status: str):
        self.log("done", strategy=strategy, status=status)

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass


def _json_default(obj):
    """Fallback JSON serialiser for numpy arrays and similar."""
    try:
        return obj.tolist()
    except AttributeError:
        return str(obj)
