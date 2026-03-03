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
    3: (0.007478, -0.414),   # index
    2: (0.006452,  0.018),   # middle
    4: (0.012547,  0.384),   # thumb (bend)
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
        time.sleep(0.5)

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
        Use RH56Hand.adaptive_force_control_iter (generator) to step-close
        until the desired force threshold is reached.

        Per-finger calibrated Newton→raw conversion.  Non-active fingers
        get threshold=0 (firmware treats them as already done).
        """
        target_forces = [
            (_force_N_to_raw(i, force_N)
             if (active_fingers is None or i in active_fingers)
             else 0)
            for i in range(6)
        ]
        plan_angles = self._ctrl_to_real(ctrl_values)
        # Minimum-closure position: FK ctrl_max → real = 0 (most physically closed).
        # Active fingers close from plan_angles toward this target; inactive fingers
        # hold at their plan position.
        max_iterations = 20
        min_closure_real = self._ctrl_to_real(
            {a: self._fk.ctrl_max[a] for a in _ACTUATOR_ORDER}
        )
        force_target_angles = [
            min_closure_real[i] if target_forces[i] > 0 else plan_angles[i]
            for i in range(6)
        ]
        # Per-finger step size: spread the closing delta evenly over max_iterations.
        step_sizes = [
            max(1, (plan_angles[i] - min_closure_real[i]) // max_iterations)
            if target_forces[i] > 0 else 50
            for i in range(6)
        ]
        target_forces_N = _raw_to_N(target_forces)
        try:
            gen = self._hand.adaptive_force_control_iter(
                target_forces, force_target_angles,
                step_size=step_sizes,
                max_iterations=max_iterations,
                speed=50,
            )
            for state in gen:
                if self._abort.is_set():
                    self._status("Force phase: aborted.")
                    break
                if state.get("done"):
                    final_raw = state["final_forces"]
                    final_N   = _raw_to_N(final_raw)
                    errs = [
                        round(final_N[i] - target_forces_N[i], 3)
                        for i in range(6)
                    ]
                    self._status(
                        f"Force phase: done. "
                        f"Final={[f'{v:.2f}' for v in final_N]} N  "
                        f"Error={[f'{e:+.2f}' for e in errs]} N"
                    )
                else:
                    raw_forces = state.get("forces", [])
                    self._forcecb(raw_forces)
                    self._status(f"Force phase: iter {state['iteration']}")
                    if logger is not None:
                        logger.log_force_iter(
                            iteration=state["iteration"],
                            forces=raw_forces,
                            angles=state.get("angles", []),
                            thresholds=target_forces,
                            forces_N=_raw_to_N(raw_forces),
                            thresholds_N=target_forces_N,
                        )

        except Exception as e:
            self._status(f"Force phase: error: {e}")
            self._hand.speed_set([1000] * 6)

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
