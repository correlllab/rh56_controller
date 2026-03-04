"""grasp_executor.py — Grasp strategy execution for UR5 + Inspire RH56 hand.

Three strategies:
  Naive       — Fixed 5-finger pinch; arm moves to pose, then fingers close.
  Plan        — Chunked trajectory: arm and fingers move proportionally in
                step_mm increments; optional adaptive force phase at end.
  Thumb Reflex — Arm moves to final pose; thumb positions first; then all
                 fingers close; optional adaptive force phase at end.

All execution runs in a background thread so the tkinter UI stays
responsive.  Status messages and live force readings are delivered via
callbacks (called from the executor thread — the UI must be thread-safe,
e.g. drain messages via a queue + timer as done in GraspVizUI).

Arm + force coordination:
    moveL is always called with blocking=True, so each arm chunk finishes
    before the fingers are sent.  The adaptive force phase runs AFTER
    the arm has reached its final position — there is no concurrent
    arm+force loop.  This is intentional given the single-connection
    constraint and the conservative safety philosophy.
"""

import threading
import time
from typing import Optional, Callable, Dict, List

import numpy as np


_ACTUATOR_ORDER = [
    "pinky", "ring", "middle", "index", "thumb_proximal", "thumb_yaw",
]

# Force calibration constants (same as real2sim_viz._FORCE_CALIB)
# Finger index → (a, b) such that F_N = a*raw + b, clamped ≥ 0
_FORCE_CALIB = {
    0: (0.006452,  0.018),   # pinky
    1: (0.006452,  0.018),   # ring
    2: (0.006452,  0.018),   # middle
    # 3: (0.006452,  0.018),   # index, overwriting index-specific calibration
    3: (0.007478, -0.414),   # index
    4: (0.012547,  0.384),   # thumb (bend)
    5: (0.012547,  0.384),   # thumb (yaw)
}

# Inverse calibration: Newton → raw integer [0..1000]
# raw = (F_N - b) / a, clamped to [0, 1000]
_FORCE_CALIB_INV = {
    3: lambda N: max(0, min(1000, int((N - (-0.414)) / 0.007478))),  # index
    2: lambda N: max(0, min(1000, int((N - 0.018) / 0.006452))),     # middle
    4: lambda N: max(0, min(1000, int((N - 0.384) / 0.012547))),     # thumb_bend
}

# Fallback for fingers without calibration (ring, pinky)
def _force_N_to_raw(finger_idx: int, force_N: float) -> int:
    if finger_idx in _FORCE_CALIB_INV:
        return _FORCE_CALIB_INV[finger_idx](force_N)
    return int(force_N * 1000 / 9.81)


# ---------------------------------------------------------------------------
# Antipodal load distribution
# ---------------------------------------------------------------------------
# Ad-hoc force distribution coefficients keyed by N = number of non-thumb
# active fingers.  Order is pinky→index (ascending finger index, 0..3).
# F_i = _ANTIPODAL_COEFFS[N][i] * force_N / N  (coefficients sum to N so
# the total force on the finger side equals force_N, matching the thumb).
_ANTIPODAL_COEFFS: Dict[int, List[float]] = {
    1: [1.0],                       # 2-finger: thumb vs. 1 finger — equal
    2: [1.0, 1.0],                  # 3-finger: thumb vs. middle+index — halves
    3: [0.5, 1.75, 1.75],            # 4-finger: ring, middle, index
    4: [0.5, 0.5, 1.9, 1.9],    # 5-finger: pinky, ring, middle, index
}

# Approximate finger lateral positions (m) in the hand base frame, indexed
# by finger index (0=pinky, 1=ring, 2=middle, 3=index).  Used for the
# geometric moment-balance distribution.  Tune for your specific hardware.
_FINGER_LATERAL_POS: List[float] = [0.054, 0.036, 0.018, 0.000]

# Thumb opposition lateral position (m) for the geometric distribution.
# This is the y-coordinate that the resultant finger force must pass through.
_THUMB_LATERAL_POS: float = 0.018


def _raw_to_N(forces_raw) -> list:
    """Convert raw force readings [0..1000] → Newtons for all 6 fingers."""
    result = []
    for i, raw in enumerate(forces_raw):
        if i in _FORCE_CALIB:
            a, b = _FORCE_CALIB[i]
            result.append(round(max(0.0, a * raw + b), 4))
        else:
            result.append(round(max(0.0, raw / 1000.0 * 9.81), 4))
    return result


class GraspExecutor:
    """
    Background-thread grasp executor coordinating UR5 arm + RH56 hand.

    Parameters
    ----------
    arm      : UR5Bridge  (from ur5_bridge.py)
    hand     : RH56Hand   (from rh56_hand.py)
    fk       : InspireHandFK  (for ctrl_min / ctrl_max bounds)
    status_cb: Callable[[str], None]   — receives status/log messages
    force_cb : Callable[[list], None]  — receives raw force readings
    """

    def __init__(
        self,
        arm,
        hand,
        fk,
        status_cb: Optional[Callable] = None,
        force_cb:  Optional[Callable] = None,
    ):
        self._arm     = arm
        self._hand    = hand
        self._fk      = fk
        self._status  = status_cb or (lambda s: print(f"[GraspExec] {s}"))
        self._forcecb = force_cb  or (lambda f: None)
        self._abort   = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Toggle between moment-balanced geometric distribution and the
        # ad-hoc _ANTIPODAL_COEFFS heuristic in _adaptive_force_phase().
        # False (default) = ad-hoc coefficients.
        self.use_geometric_force_dist: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        """True if a grasp strategy is currently executing."""
        return self._thread is not None and self._thread.is_alive()

    def abort(self):
        """
        Request abort.  The arm decelerates; the hand freezes at its current
        position.  Returns immediately (does not join the thread).
        """
        self._abort.set()
        if self._arm is not None:
            self._arm.stop()

    def execute_naive(
        self,
        world_T_hand: np.ndarray,
        force_N: float = 0.0,
        move_arm: bool = True,
        active_fingers: Optional[List[int]] = None,
        logger=None,
    ):
        """Start a Naive grasp in a background thread."""
        self._start(self._run_naive, world_T_hand, force_N, move_arm,
                    active_fingers, logger)

    def execute_plan(
        self,
        closure_result,
        world_T_hand: np.ndarray,
        force_N: float = 0.0,
        step_mm: float = 10.0,
        move_arm: bool = True,
        active_fingers: Optional[List[int]] = None,
        logger=None,
    ):
        """Start a Plan grasp in a background thread."""
        self._start(
            self._run_plan,
            closure_result, world_T_hand, force_N, step_mm, move_arm,
            active_fingers, logger,
        )

    def execute_plan_waypoints(
        self,
        waypoints,
        force_N: float = 0.0,
        move_arm: bool = True,
        active_fingers: Optional[List[int]] = None,
        logger=None,
    ):
        """Start a width-space Plan grasp from pre-computed waypoints.

        waypoints: list of (world_T_hand: np.ndarray, closure_result) pairs,
                   ordered from approach width down to final grip width.
        """
        self._start(self._run_plan_waypoints, waypoints, force_N, move_arm,
                    active_fingers, logger)

    def execute_thumb_reflex(
        self,
        closure_result,
        world_T_hand: np.ndarray,
        force_N: float = 0.0,
        move_arm: bool = True,
        active_fingers: Optional[List[int]] = None,
        logger=None,
    ):
        """Start a Thumb Reflex grasp in a background thread."""
        self._start(
            self._run_thumb_reflex,
            closure_result, world_T_hand, force_N, move_arm,
            active_fingers, logger,
        )

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------
    def _start(self, fn, *args):
        if self.is_running():
            self._status("Executor is busy — abort current strategy first.")
            return
        self._abort.clear()
        self._thread = threading.Thread(
            target=fn, args=args, daemon=True, name="grasp-exec",
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Angle conversion helper
    # ------------------------------------------------------------------
    def _ctrl_to_real(self, ctrl_values: Dict[str, float]) -> List[int]:
        """
        Convert ctrl_values dict (rad) to real-hand angle_set list [0..1000].
        Uses the INVERTED convention:
            real = round((1 - clamp((ctrl - min) / (max - min))) * 1000)
        Order: [pinky, ring, middle, index, thumb_proximal, thumb_yaw]
        """
        ctrl_min = np.array([self._fk.ctrl_min[a] for a in _ACTUATOR_ORDER])
        ctrl_max = np.array([self._fk.ctrl_max[a] for a in _ACTUATOR_ORDER])
        rng = ctrl_max - ctrl_min
        fc  = np.array([ctrl_values.get(a, 0.0) for a in _ACTUATOR_ORDER])
        real = np.round(
            (1.0 - np.clip(
                (fc - ctrl_min) / np.where(rng > 0, rng, 1.0),
                0.0, 1.0,
            )) * 1000
        ).astype(int)
        return real.tolist()

    def _open_cmd(self) -> List[int]:
        """Return all-open finger command (1000 = fully open)."""
        return [1000] * 6

    # ------------------------------------------------------------------
    # EEF error helpers
    # ------------------------------------------------------------------
    def _eef_error(self, desired_T: np.ndarray, actual_T: np.ndarray):
        """
        Compute position error (mm) and rotation error (deg) between two 4×4
        homogeneous transforms.
        """
        pos_err = float(np.linalg.norm(actual_T[:3, 3] - desired_T[:3, 3]) * 1000)
        R_err   = actual_T[:3, :3].T @ desired_T[:3, :3]
        cos_a   = float(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
        rot_err = float(np.degrees(np.arccos(cos_a)))
        return pos_err, rot_err

    def _report_eef_error(self, desired_T: np.ndarray, label: str = ""):
        """
        Read the current arm EEF pose, compute error vs desired_T, log it.

        Returns (actual_T, pos_err_mm, rot_err_deg).  All three are None if
        the arm is unavailable.
        """
        if self._arm is None:
            return None, None, None
        actual_T = self._arm.get_hand_pose_4x4()
        if actual_T is None:
            return None, None, None
        pos_err, rot_err = self._eef_error(desired_T, actual_T)
        sev = ("OK"   if pos_err < 5  and rot_err < 3
               else "WARN" if pos_err < 15
               else "HIGH")
        self._status(
            f"EEF [{label}] pos={pos_err:.1f}mm rot={rot_err:.1f}° [{sev}]"
        )
        return actual_T, pos_err, rot_err

    # ------------------------------------------------------------------
    # Naive strategy
    # ------------------------------------------------------------------
    def _run_naive(
        self,
        world_T_hand: np.ndarray,
        force_N: float,
        move_arm: bool,
        active_fingers: Optional[List[int]],
        logger,
    ):
        """
        Naive: move arm to pose (optional), then send fixed 5-finger pinch.
        Finger posture is hardcoded (no grasp geometry solver).
        """
        if move_arm and not self._abort.is_set():
            self._status("Naive: moving arm to grasp pose...")
            warns = self._arm.move_to_hand_pose(world_T_hand, blocking=True)
            for w in warns:
                self._status(w)
            actual_T, pos_err, rot_err = self._report_eef_error(
                world_T_hand, label="Naive"
            )
            if logger is not None:
                joint_q = self._arm.get_joint_angles() if self._arm else None
                fa = self._hand.angle_read() if self._hand else None
                ff = self._hand.force_act()   if self._hand else None
                logger.log_waypoint(
                    step_i=0, n_steps=1,
                    desired_T=world_T_hand, actual_T=actual_T,
                    pos_err_mm=pos_err, rot_err_deg=rot_err,
                    joint_q=joint_q,
                    finger_angles=fa if fa is not None else [],
                    finger_forces=ff if ff is not None else [],
                )

        if self._abort.is_set():
            if logger is not None:
                logger.log_done(strategy="Naive", status="aborted")
                logger.close()
            self._status("Naive: aborted.")
            return

        # Fixed naive posture [pinky, ring, middle, index, thumb_bend, thumb_yaw]
        naive_cmd = [504, 496, 467, 500, 479, 0]
        self._status("Naive: closing fingers to pinch posture...")
        try:
            if force_N > 0.0 and not self._abort.is_set():
                self._hand.force_set([_force_N_to_raw(i, force_N) for i in range(6)])
            self._hand.angle_set(naive_cmd)
        except Exception as e:
            self._status(f"Naive: finger error: {e}")
            if logger is not None:
                logger.log_done(strategy="Naive", status=f"error: {e}")
                logger.close()
            return

        if force_N > 0.0 and not self._abort.is_set():
            self._status(f"Naive: monitoring force (threshold {force_N:.1f} N)...")
            self._naive_force_monitor(force_N, active_fingers=active_fingers)

        if logger is not None:
            logger.log_done(strategy="Naive", status="complete")
            logger.close()
        self._status("Naive: complete.")

    def _naive_force_monitor(
        self,
        threshold_N: float,
        poll_s: float = 0.05,
        active_fingers: Optional[List[int]] = None,
    ):
        """Poll force_act until any active calibrated finger reaches threshold."""
        while not self._abort.is_set():
            raw = self._hand.force_act()
            if raw is None:
                time.sleep(poll_s)
                continue
            self._forcecb(raw)
            for idx, (a, b) in _FORCE_CALIB.items():
                if active_fingers is not None and idx not in active_fingers:
                    continue
                f_N = max(0.0, a * raw[idx] + b)
                if f_N >= threshold_N:
                    self._status(
                        f"Naive: force threshold reached "
                        f"(finger[{idx}]={f_N:.2f} N)"
                    )
                    return
            time.sleep(poll_s)

    # ------------------------------------------------------------------
    # Plan strategy
    # ------------------------------------------------------------------
    def _run_plan(
        self,
        closure_result,
        world_T_hand: np.ndarray,
        force_N: float,
        step_mm: float,
        move_arm: bool,
        active_fingers: Optional[List[int]],
        logger,
    ):
        """
        Plan grasp: chunked arm + proportional finger trajectory, then
        optional adaptive force phase.
        """
        finger_final = self._ctrl_to_real(closure_result.ctrl_values)

        if move_arm:
            current_T = self._arm.get_hand_pose_4x4()
            if current_T is None:
                self._status("Plan: cannot read arm pose — using final pose only.")
                current_T = world_T_hand
            waypoints = self._compute_waypoints(
                current_T, world_T_hand, step_mm / 1000.0,
            )
            self._status(f"Plan: {len(waypoints)} chunk(s) × {step_mm:.0f} mm")
        else:
            waypoints = [world_T_hand]

        n = len(waypoints)
        for i, wp_T in enumerate(waypoints):
            if self._abort.is_set():
                if logger is not None:
                    logger.log_done(strategy="Plan", status="aborted")
                    logger.close()
                self._status("Plan: aborted.")
                return

            t   = (i + 1) / n
            cmd = [max(0, int(1000 - t * (1000 - v))) for v in finger_final]

            if move_arm:
                warns = self._arm.move_to_hand_pose(wp_T, blocking=True)
                for w in warns:
                    self._status(w)
                actual_T, pos_err, rot_err = self._report_eef_error(
                    wp_T, label=f"chunk {i+1}"
                )
                if logger is not None:
                    joint_q = self._arm.get_joint_angles() if self._arm else None
                    fa = self._hand.angle_read() if self._hand else None
                    ff = self._hand.force_act()   if self._hand else None
                    logger.log_waypoint(
                        step_i=i, n_steps=n,
                        desired_T=wp_T, actual_T=actual_T,
                        pos_err_mm=pos_err, rot_err_deg=rot_err,
                        joint_q=joint_q,
                        finger_angles=fa if fa is not None else [],
                        finger_forces=ff if ff is not None else [],
                    )
            try:
                self._hand.angle_set(cmd)
            except Exception as e:
                self._status(f"Plan: finger error at chunk {i+1}: {e}")

            self._status(f"Plan: chunk {i+1}/{n} (t={t:.2f})")

        if force_N > 0.0 and not self._abort.is_set():
            self._status(f"Plan: entering force phase ({force_N:.1f} N)...")
            self._adaptive_force_phase(
                closure_result.ctrl_values, force_N,
                active_fingers=active_fingers, logger=logger,
            )

        if logger is not None:
            logger.log_done(strategy="Plan", status="complete")
            logger.close()
        self._status("Plan: complete.")

    # ------------------------------------------------------------------
    # Plan strategy — width-space waypoints
    # ------------------------------------------------------------------
    def _run_plan_waypoints(
        self,
        waypoints,
        force_N: float,
        move_arm: bool,
        active_fingers: Optional[List[int]],
        logger,
    ):
        """
        Width-space Plan grasp.

        waypoints[0]   = approach: fingers go to approach config first (single
                         angle_set), then arm moves to approach pose (single moveL).
        waypoints[1:]  = steps: arm moveL + finger set per step, approach→target.

        Thumb yaw is pre-set to its final value and held fixed throughout.
        """
        if not waypoints:
            self._status("Plan: no waypoints — aborted.")
            if logger is not None:
                logger.log_done(strategy="Plan", status="no waypoints")
                logger.close()
            return

        final_cmd    = self._ctrl_to_real(waypoints[-1][1].ctrl_values)
        approach_T, approach_r = waypoints[0]
        approach_cmd = self._ctrl_to_real(approach_r.ctrl_values)
        approach_cmd[5] = final_cmd[5]  # thumb yaw fixed

        # Phase 1: pre-set thumb yaw (fingers open), then approach finger config
        self._status("Plan: opening fingers to approach config...")
        try:
            if self._abort.is_set():
                if logger is not None:
                    logger.log_done(strategy="Plan", status="aborted")
                    logger.close()
                self._status("Plan: aborted.")
                return
            self._hand.angle_set(approach_cmd)
            time.sleep(0.3)
        except Exception as e:
            self._status(f"Plan: approach finger error: {e}")

        # Phase 2: arm moves to approach pose (single moveL)
        if move_arm and not self._abort.is_set():
            self._status(
                f"Plan: moving arm to approach pose "
                f"(width={approach_r.width*1000:.1f}mm)..."
            )
            warns = self._arm.move_to_hand_pose(approach_T, blocking=True)
            for w in warns:
                self._status(w)
            actual_T, pos_err, rot_err = self._report_eef_error(
                approach_T, label="approach"
            )
            if logger is not None:
                joint_q = self._arm.get_joint_angles() if self._arm else None
                fa = self._hand.angle_read() if self._hand else None
                ff = self._hand.force_act()   if self._hand else None
                logger.log_waypoint(
                    step_i=0, n_steps=len(waypoints),
                    desired_T=approach_T, actual_T=actual_T,
                    pos_err_mm=pos_err, rot_err_deg=rot_err,
                    joint_q=joint_q,
                    finger_angles=fa if fa is not None else [],
                    finger_forces=ff if ff is not None else [],
                )

        # Phase 3: step from approach → target
        steps   = waypoints[1:]
        n_steps = len(steps)
        self._status(
            f"Plan: stepping {n_steps} step(s) → "
            f"{waypoints[-1][1].width*1000:.1f}mm"
        )
        for i, (wp_T, r_i) in enumerate(steps):
            if self._abort.is_set():
                if logger is not None:
                    logger.log_done(strategy="Plan", status="aborted")
                    logger.close()
                self._status("Plan: aborted.")
                return
            cmd    = self._ctrl_to_real(r_i.ctrl_values)
            cmd[5] = final_cmd[5]
            if move_arm:
                warns = self._arm.move_to_hand_pose(wp_T, blocking=True)
                for w in warns:
                    self._status(w)
                actual_T, pos_err, rot_err = self._report_eef_error(
                    wp_T, label=f"step {i+1}"
                )
                if logger is not None:
                    joint_q = self._arm.get_joint_angles() if self._arm else None
                    fa = self._hand.angle_read() if self._hand else None
                    ff = self._hand.force_act()   if self._hand else None
                    logger.log_waypoint(
                        step_i=i + 1, n_steps=len(waypoints),
                        desired_T=wp_T, actual_T=actual_T,
                        pos_err_mm=pos_err, rot_err_deg=rot_err,
                        joint_q=joint_q,
                        finger_angles=fa if fa is not None else [],
                        finger_forces=ff if ff is not None else [],
                    )
            try:
                self._hand.angle_set(cmd)
            except Exception as e:
                self._status(f"Plan: finger error at step {i+1}: {e}")
            self._status(
                f"Plan: step {i+1}/{n_steps} (width={r_i.width*1000:.1f}mm)"
            )

        if force_N > 0.0 and not self._abort.is_set():
            self._status(f"Plan: entering force phase ({force_N:.1f} N)...")
            self._adaptive_force_phase(
                waypoints[-1][1].ctrl_values, force_N,
                active_fingers=active_fingers, logger=logger,
            )

        if logger is not None:
            logger.log_done(strategy="Plan", status="complete")
            logger.close()
        self._status("Plan: complete.")

    # ------------------------------------------------------------------
    # Thumb Reflex strategy
    # ------------------------------------------------------------------
    def _run_thumb_reflex(
        self,
        closure_result,
        world_T_hand: np.ndarray,
        force_N: float,
        move_arm: bool,
        active_fingers: Optional[List[int]],
        logger,
    ):
        """
        Thumb Reflex:
          1. Thumb (bend + yaw) positions first; other fingers stay fully open.
          2. Wait 0.5 s for thumb to settle.
          3. Arm moves to final grasp pose (single moveL — no stepping).
          4. All remaining fingers close to final config.
          5. Optional adaptive force phase.
        """
        full_cmd   = self._ctrl_to_real(closure_result.ctrl_values)
        thumb_only = [1000, 1000, 1000, 1000, full_cmd[4], full_cmd[5]]
        self._status("Thumb Reflex: positioning thumb (fingers open)...")
        try:
            self._hand.angle_set(thumb_only)
        except Exception as e:
            self._status(f"Thumb Reflex: thumb error: {e}")
            if logger is not None:
                logger.log_done(strategy="Thumb Reflex", status=f"error: {e}")
                logger.close()
            return

        time.sleep(0.2)

        if self._abort.is_set():
            if logger is not None:
                logger.log_done(strategy="Thumb Reflex", status="aborted")
                logger.close()
            self._status("Thumb Reflex: aborted.")
            return

        if move_arm:
            self._status("Thumb Reflex: moving arm to grasp pose...")
            warns = self._arm.move_to_hand_pose(world_T_hand, blocking=True)
            for w in warns:
                self._status(w)
            actual_T, pos_err, rot_err = self._report_eef_error(
                world_T_hand, label="Thumb Reflex"
            )
            if logger is not None:
                joint_q = self._arm.get_joint_angles() if self._arm else None
                fa = self._hand.angle_read() if self._hand else None
                ff = self._hand.force_act()   if self._hand else None
                logger.log_waypoint(
                    step_i=0, n_steps=1,
                    desired_T=world_T_hand, actual_T=actual_T,
                    pos_err_mm=pos_err, rot_err_deg=rot_err,
                    joint_q=joint_q,
                    finger_angles=fa if fa is not None else [],
                    finger_forces=ff if ff is not None else [],
                )

        if self._abort.is_set():
            if logger is not None:
                logger.log_done(strategy="Thumb Reflex", status="aborted")
                logger.close()
            self._status("Thumb Reflex: aborted.")
            return

        self._status("Thumb Reflex: closing remaining fingers to target width...")
        try:
            self._hand.angle_set(full_cmd)
        except Exception as e:
            self._status(f"Thumb Reflex: finger error: {e}")
            if logger is not None:
                logger.log_done(strategy="Thumb Reflex", status=f"error: {e}")
                logger.close()
            return

        # Wait for fingers to reach target position at full speed before
        # switching to the slow adaptive force phase.
        time.sleep(0.25)

        if force_N > 0.0 and not self._abort.is_set():
            self._status(f"Thumb Reflex: entering force phase ({force_N:.1f} N)...")
            self._adaptive_force_phase(
                closure_result.ctrl_values, force_N,
                active_fingers=active_fingers, logger=logger,
            )

        if logger is not None:
            logger.log_done(strategy="Thumb Reflex", status="complete")
            logger.close()
        self._status("Thumb Reflex: complete.")

    # ------------------------------------------------------------------
    # Antipodal force distribution helpers
    # ------------------------------------------------------------------
    def _compute_finger_force_targets(
        self,
        force_N: float,
        active_fingers: Optional[List[int]],
        use_geometric: bool = False,
    ) -> List[float]:
        """
        Per-finger Newton targets for an antipodal grasp.

        Thumb (idx 4) receives ``force_N``; non-thumb active fingers share
        the same total force using an antipodal load distribution so the
        resultant finger force matches the thumb in magnitude and line of
        action.  Thumb yaw (idx 5) is always excluded.

        Parameters
        ----------
        force_N        : target squeeze force (N)
        active_fingers : participating finger indices, or None (= all).
        use_geometric  : if True use pseudoinverse moment-balance;
                         if False use _ANTIPODAL_COEFFS heuristic.

        Returns
        -------
        List[float] of length 6, order [pinky, ring, middle, index,
        thumb_bend, thumb_yaw].
        """
        targets: List[float] = [0.0] * 6

        if active_fingers is None:
            non_thumb   = [0, 1, 2, 3]
            thumb_active = True
        else:
            non_thumb    = sorted(f for f in active_fingers if f not in (4, 5))
            thumb_active = 4 in active_fingers

        if thumb_active:
            targets[4] = force_N

        N = len(non_thumb)
        if N == 0:
            return targets

        if use_geometric:
            weights = self._geometric_antipodal_weights(non_thumb)
        else:
            coeffs  = _ANTIPODAL_COEFFS.get(N, [1.0] * N)
            weights = [c / N for c in coeffs]   # sum(weights) = 1

        for rank, fi in enumerate(non_thumb):
            targets[fi] = weights[rank] * force_N

        return targets

    def _geometric_antipodal_weights(
        self, non_thumb_fingers: List[int]
    ) -> List[float]:
        """
        Compute moment-balanced force weights via min-norm pseudoinverse.

        Solves  A @ w = b  (minimum ||w||₂ subject to force + moment balance):
            A = [[1,  1,  ...,  1 ],   (total force = 1)
                 [y₁, y₂, ..., yₙ]]   (moment about origin = y_thumb)
            b = [1, _THUMB_LATERAL_POS]

        Any negative weights are clipped to 0 and the result is renormalised
        so that sum(weights) = 1.  Falls back to equal weights on degeneracy.

        Tune _FINGER_LATERAL_POS and _THUMB_LATERAL_POS for your hardware.
        """
        ys  = np.array([_FINGER_LATERAL_POS[fi] for fi in non_thumb_fingers])
        y_t = _THUMB_LATERAL_POS
        N   = len(ys)

        A = np.vstack([np.ones(N), ys])   # shape (2, N)
        b = np.array([1.0, y_t])

        try:
            # Minimum-norm (pseudoinverse) solution: w = Aᵀ (AAᵀ)⁻¹ b
            w = A.T @ np.linalg.solve(A @ A.T, b)
        except np.linalg.LinAlgError:
            return [1.0 / N] * N

        w = np.maximum(w, 0.0)
        s = float(w.sum())
        if s < 1e-6:
            return [1.0 / N] * N
        return (w / s).tolist()

    # ------------------------------------------------------------------
    # Adaptive force phase (shared by Plan + Thumb Reflex)
    # ------------------------------------------------------------------
    def _adaptive_force_phase(
        self,
        ctrl_values: dict,
        force_N: float,
        active_fingers: Optional[List[int]] = None,
        logger=None,
    ):
        """
        Adaptive force phase: close active fingers iteratively until each
        reaches its individual force target, then hold.

        Antipodal load distribution
        ---------------------------
        Thumb (idx 4) receives ``force_N``.  Non-thumb active fingers share
        the same total so the resultant finger force opposes the thumb.
        Toggle ``self.use_geometric_force_dist`` (default False) to switch
        between:
          - False : ad-hoc _ANTIPODAL_COEFFS heuristic
          - True  : pseudoinverse moment-balance (_geometric_antipodal_weights)

        Per-finger saturation
        ---------------------
        Once a finger's measured force ≥ its individual target the executor
        stops commanding further position changes for that finger (holds its
        last angle).  Other fingers continue closing independently.
        """
        max_iterations = 20
        poll_dt        = 0.10

        # --- 1. Per-finger Newton targets -----------------------------------
        targets_N = self._compute_finger_force_targets(
            force_N, active_fingers,
            use_geometric=self.use_geometric_force_dist,
        )
        dist_mode = "geometric" if self.use_geometric_force_dist else "ad-hoc"
        # Normalised per-finger weights (proportion of force_N; sum ≤ 1 per side).
        weights_norm = [
            round(t / force_N, 4) if force_N > 0 else 0.0 for t in targets_N
        ]
        self._status(
            f"Force phase: targets={[f'{v:.2f}' for v in targets_N]} N"
            f"  weights={weights_norm}  [{dist_mode}]"
        )

        # --- 2. Raw targets for firmware force_set() ------------------------
        targets_raw = [
            _force_N_to_raw(i, targets_N[i]) if targets_N[i] > 0 else 0
            for i in range(6)
        ]

        # --- 3. Angle targets and per-finger step sizes ---------------------
        plan_angles  = self._ctrl_to_real(ctrl_values)
        min_closure  = self._ctrl_to_real(
            {a: self._fk.ctrl_max[a] for a in _ACTUATOR_ORDER}
        )
        target_angles = np.array([
            min_closure[i] if targets_raw[i] > 0 else plan_angles[i]
            for i in range(6)
        ], dtype=float)
        step_sizes = np.array([
            max(1.0, abs(plan_angles[i] - min_closure[i]) / max_iterations)
            if targets_raw[i] > 0 else 0.0
            for i in range(6)
        ], dtype=float)

        # --- 4. Initial setup -----------------------------------------------
        try:
            self._hand.speed_set([75] * 6)
        except Exception as e:
            self._status(f"Force phase: speed_set error: {e}")
            return

        # Inactive fingers (target=0) and thumb_yaw (idx 5) are immediately
        # considered satisfied so they are never commanded further.
        satisfied = [targets_raw[i] == 0 for i in range(6)]

        current_angles = np.array(
            self._hand.angle_read() or [1000] * 6, dtype=float
        )

        # --- 5. Control loop ------------------------------------------------
        for iteration in range(max_iterations):
            if self._abort.is_set():
                self._status("Force phase: aborted.")
                return

            # Refresh firmware force protection each iteration.
            try:
                self._hand.force_set(targets_raw)
            except Exception as e:
                self._status(f"Force phase: force_set error: {e}")
                break

            # Step unsatisfied fingers toward closure; hold satisfied ones.
            next_angles = current_angles.copy()
            for i in range(6):
                if not satisfied[i]:
                    delta = target_angles[i] - current_angles[i]
                    next_angles[i] += float(
                        np.clip(delta, -step_sizes[i], step_sizes[i])
                    )

            try:
                self._hand.angle_set(np.round(next_angles).astype(int).tolist())
            except Exception as e:
                self._status(f"Force phase: angle_set error: {e}")
                break

            time.sleep(poll_dt)

            # Read back actual angles (firmware may have stopped a motor early).
            readback = self._hand.angle_read()
            current_angles = np.array(
                readback if readback else next_angles, dtype=float
            )

            # Read forces and check per-finger saturation.
            raw_forces = self._hand.force_act()
            if raw_forces is None:
                continue
            self._forcecb(raw_forces)

            newly_satisfied = [
                i for i in range(6)
                if not satisfied[i] and raw_forces[i] >= targets_raw[i]
            ]
            for i in newly_satisfied:
                satisfied[i] = True
            if newly_satisfied:
                f_N = _raw_to_N(raw_forces)
                self._status(
                    f"Force phase: finger(s) {newly_satisfied} satisfied"
                    f" @ {[f'{f_N[i]:.2f}' for i in newly_satisfied]} N"
                )

            self._status(f"Force phase: iter {iteration + 1}/{max_iterations}")
            if logger is not None:
                logger.log_force_iter(
                    iteration=iteration + 1,
                    forces=raw_forces,
                    angles=current_angles.tolist(),
                    thresholds=targets_raw,
                    forces_N=_raw_to_N(raw_forces),
                    thresholds_N=targets_N,
                    # Log distribution metadata on the first iteration only.
                    weights=weights_norm if iteration == 0 else None,
                    dist_mode=dist_mode if iteration == 0 else None,
                )

            if all(satisfied):
                break

        # --- 6. Final report ------------------------------------------------
        final_raw = self._hand.force_act()
        final_N   = _raw_to_N(final_raw) if final_raw else [0.0] * 6
        errs      = [round(final_N[i] - targets_N[i], 3) for i in range(6)]
        self._status(
            f"Force phase: done. "
            f"Final={[f'{v:.2f}' for v in final_N]} N  "
            f"Error={[f'{e:+.2f}' for e in errs]} N"
        )

    # ------------------------------------------------------------------
    # Waypoint interpolation
    # ------------------------------------------------------------------
    def _compute_waypoints(
        self,
        start_T: np.ndarray,
        end_T: np.ndarray,
        step_m: float,
    ) -> List[np.ndarray]:
        """
        Compute N linearly-interpolated waypoints between two 4×4 poses.
        Position: linear.  Rotation: SLERP via axis-angle.
        The last waypoint equals end_T exactly.
        """
        dist = float(np.linalg.norm(end_T[:3, 3] - start_T[:3, 3]))
        n    = max(1, int(np.ceil(dist / max(step_m, 1e-4))))
        waypoints = []
        for i in range(1, n + 1):
            t    = i / n
            T_wp = np.eye(4)
            T_wp[:3, 3] = (1 - t) * start_T[:3, 3] + t * end_T[:3, 3]

            R_rel  = start_T[:3, :3].T @ end_T[:3, :3]
            cos_a  = float(np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0))
            angle  = float(np.arccos(cos_a))
            if abs(angle) < 1e-6:
                T_wp[:3, :3] = end_T[:3, :3]
            else:
                k  = np.array([
                    R_rel[2, 1] - R_rel[1, 2],
                    R_rel[0, 2] - R_rel[2, 0],
                    R_rel[1, 0] - R_rel[0, 1],
                ]) / (2 * np.sin(angle))
                a_t  = angle * t
                c, s = np.cos(a_t), np.sin(a_t)
                v    = 1 - c
                kx, ky, kz = k
                R_t = np.array([
                    [kx*kx*v + c,    kx*ky*v - kz*s, kx*kz*v + ky*s],
                    [kx*ky*v + kz*s, ky*ky*v + c,    ky*kz*v - kx*s],
                    [kx*kz*v - ky*s, ky*kz*v + kx*s, kz*kz*v + c   ],
                ])
                T_wp[:3, :3] = start_T[:3, :3] @ R_t
            waypoints.append(T_wp)
        return waypoints
