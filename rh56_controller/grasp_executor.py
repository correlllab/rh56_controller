"""grasp_executor.py — Grasp strategy execution for UR5 + Inspire RH56 hand.

Three strategies:
  Naive       — Fixed 5-finger pinch; arm moves to pose, then fingers close.
  Plan        — Chunked trajectory: arm and fingers move proportionally in
                step_mm increments; optional adaptive force phase at end.
  Thumb Reflex — Arm moves to final pose; thumb positions first; then all
                 fingers close; optional adaptive force phase at end.

All execution runs in a background thread so the matplotlib UI stays
responsive.  Status messages and live force readings are delivered via
callbacks (called from the executor thread — the UI must be thread-safe,
e.g. drain messages via a queue + timer as done in GraspViz).

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
    ):
        """Start a Naive grasp in a background thread."""
        self._start(self._run_naive, world_T_hand, force_N, move_arm)

    def execute_plan(
        self,
        closure_result,
        world_T_hand: np.ndarray,
        force_N: float = 0.0,
        step_mm: float = 10.0,
        move_arm: bool = True,
    ):
        """Start a Plan grasp in a background thread."""
        self._start(
            self._run_plan,
            closure_result, world_T_hand, force_N, step_mm, move_arm,
        )

    def execute_plan_waypoints(
        self,
        waypoints,
        force_N: float = 0.0,
        move_arm: bool = True,
    ):
        """Start a width-space Plan grasp from pre-computed waypoints.

        waypoints: list of (world_T_hand: np.ndarray, closure_result) pairs,
                   ordered from approach width down to final grip width.
        """
        self._start(self._run_plan_waypoints, waypoints, force_N, move_arm)

    def execute_thumb_reflex(
        self,
        closure_result,
        world_T_hand: np.ndarray,
        force_N: float = 0.0,
        move_arm: bool = True,
    ):
        """Start a Thumb Reflex grasp in a background thread."""
        self._start(
            self._run_thumb_reflex,
            closure_result, world_T_hand, force_N, move_arm,
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
    # Naive strategy
    # ------------------------------------------------------------------
    def _run_naive(
        self,
        world_T_hand: np.ndarray,
        force_N: float,
        move_arm: bool,
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

        if self._abort.is_set():
            self._status("Naive: aborted.")
            return

        # Fixed naive posture [pinky, ring, middle, index, thumb_bend, thumb_yaw]
        # 750 ≈ 75% closed for fingers; 740 ≈ 74% for thumb_bend;
        # 0 = thumb_yaw max adduction (INVERTED: 0 → ctrl_max)
        naive_cmd = [750, 750, 750, 750, 740, 0]
        self._status("Naive: closing fingers to pinch posture...")
        try:
            self._hand.angle_set(naive_cmd)
        except Exception as e:
            self._status(f"Naive: finger error: {e}")
            return

        if force_N > 0.0 and not self._abort.is_set():
            self._status(f"Naive: monitoring force (threshold {force_N:.1f} N)...")
            self._naive_force_monitor(force_N)

        self._status("Naive: complete.")

    def _naive_force_monitor(self, threshold_N: float, poll_s: float = 0.05):
        """Poll force_act until any calibrated finger reaches threshold."""
        while not self._abort.is_set():
            raw = self._hand.force_act()
            if raw is None:
                time.sleep(poll_s)
                continue
            self._forcecb(raw)
            for idx, (a, b) in _FORCE_CALIB.items():
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
    ):
        """
        Plan grasp: chunked arm + proportional finger trajectory, then
        optional adaptive force phase.

        Each chunk:
          1. Arm moves to intermediate pose (blocking).
          2. Fingers sent to proportionally interpolated position.
        Finger interpolation: open (1000) at chunk 0 → final at chunk N.
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
            self._status(
                f"Plan: {len(waypoints)} chunk(s) × {step_mm:.0f} mm"
            )
        else:
            waypoints = [world_T_hand]

        n = len(waypoints)
        for i, wp_T in enumerate(waypoints):
            if self._abort.is_set():
                self._status("Plan: aborted.")
                return

            # Proportional finger close: t goes 0→1 over the chunks
            t   = (i + 1) / n
            cmd = [max(0, int(1000 - t * (1000 - v))) for v in finger_final]

            if move_arm:
                warns = self._arm.move_to_hand_pose(wp_T, blocking=True)
                for w in warns:
                    self._status(w)
            try:
                self._hand.angle_set(cmd)
            except Exception as e:
                self._status(f"Plan: finger error at chunk {i+1}: {e}")

            self._status(f"Plan: chunk {i+1}/{n} (t={t:.2f})")

        if force_N > 0.0 and not self._abort.is_set():
            self._status(f"Plan: entering force phase ({force_N:.1f} N)...")
            self._adaptive_force_phase(closure_result.ctrl_values, force_N)

        self._status("Plan: complete.")

    # ------------------------------------------------------------------
    # Plan strategy — width-space waypoints
    # ------------------------------------------------------------------
    def _run_plan_waypoints(
        self,
        waypoints,
        force_N: float,
        move_arm: bool,
    ):
        """
        Width-space Plan grasp: each waypoint is a (world_T_hand, closure_result)
        pair, ordered from approach width down to final grip width.

        Execution per step:
          1. Arm moves to intermediate pose (blocking).
          2. Fingers sent to config at that width.

        Thumb yaw is set to its final value BEFORE the first arm move and held
        fixed throughout — only flex joints (pinky/ring/middle/index/thumb_bend)
        vary with width.
        """
        if not waypoints:
            self._status("Plan: no waypoints — aborted.")
            return

        n = len(waypoints)
        final_cmd = self._ctrl_to_real(waypoints[-1][1].ctrl_values)

        # Pre-set thumb yaw to the final value before any arm motion.
        thumb_init = [1000, 1000, 1000, 1000, 1000, final_cmd[5]]
        self._status("Plan: pre-setting thumb yaw...")
        try:
            self._hand.angle_set(thumb_init)
            time.sleep(0.3)
        except Exception as e:
            self._status(f"Plan: thumb yaw init error: {e}")

        self._status(
            f"Plan: {n} width-space step(s), "
            f"width {waypoints[0][1].width*1000:.1f}→{waypoints[-1][1].width*1000:.1f} mm"
        )

        for i, (wp_T, r_i) in enumerate(waypoints):
            if self._abort.is_set():
                self._status("Plan: aborted.")
                return

            cmd = self._ctrl_to_real(r_i.ctrl_values)
            cmd[5] = final_cmd[5]   # thumb yaw stays fixed at final value

            if move_arm:
                warns = self._arm.move_to_hand_pose(wp_T, blocking=True)
                for w in warns:
                    self._status(w)
            try:
                self._hand.angle_set(cmd)
            except Exception as e:
                self._status(f"Plan: finger error at step {i+1}: {e}")

            self._status(
                f"Plan: step {i+1}/{n} (width={r_i.width*1000:.1f}mm)"
            )

        if force_N > 0.0 and not self._abort.is_set():
            self._status(f"Plan: entering force phase ({force_N:.1f} N)...")
            self._adaptive_force_phase(waypoints[-1][1].ctrl_values, force_N)

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
    ):
        """
        Thumb Reflex:
          1. Arm moves to final grasp pose (single moveL).
          2. Thumb (bend + yaw) positions first; other fingers stay open.
          3. Wait 0.5 s for thumb to settle.
          4. All remaining fingers close.
          5. Optional adaptive force phase.
        """
        if move_arm and not self._abort.is_set():
            self._status("Thumb Reflex: moving arm to grasp pose...")
            warns = self._arm.move_to_hand_pose(world_T_hand, blocking=True)
            for w in warns:
                self._status(w)

        if self._abort.is_set():
            self._status("Thumb Reflex: aborted.")
            return

        full_cmd = self._ctrl_to_real(closure_result.ctrl_values)
        # Send thumb only — pinky/ring/middle/index stay open (1000)
        thumb_only = [1000, 1000, 1000, 1000, full_cmd[4], full_cmd[5]]
        self._status("Thumb Reflex: positioning thumb...")
        try:
            self._hand.angle_set(thumb_only)
        except Exception as e:
            self._status(f"Thumb Reflex: thumb error: {e}")
            return

        time.sleep(0.5)  # wait for thumb to reach position

        if self._abort.is_set():
            self._status("Thumb Reflex: aborted.")
            return

        self._status("Thumb Reflex: closing remaining fingers...")
        try:
            self._hand.angle_set(full_cmd)
        except Exception as e:
            self._status(f"Thumb Reflex: finger error: {e}")
            return

        if force_N > 0.0 and not self._abort.is_set():
            self._status(f"Thumb Reflex: entering force phase ({force_N:.1f} N)...")
            self._adaptive_force_phase(closure_result.ctrl_values, force_N)

        self._status("Thumb Reflex: complete.")

    # ------------------------------------------------------------------
    # Adaptive force phase (shared by Plan + Thumb Reflex)
    # ------------------------------------------------------------------
    def _adaptive_force_phase(self, ctrl_values: dict, force_N: float):
        """
        Use RH56Hand.adaptive_force_control_iter (generator) to step-close
        until the desired force threshold is reached.
        Converts force_N to raw gram units (F_N = raw/1000 * 9.81 fallback).
        """
        target_raw_g  = int(force_N * 1000 / 9.81)
        target_forces = [target_raw_g] * 6
        target_angles = self._ctrl_to_real(ctrl_values)
        try:
            gen = self._hand.adaptive_force_control_iter(
                target_forces, target_angles,
                step_size=50, max_iterations=20,
            )
            for state in gen:
                if self._abort.is_set():
                    self._status("Force phase: aborted.")
                    break
                if state.get("done"):
                    self._status(
                        f"Force phase: done. "
                        f"Final forces={state['final_forces']}"
                    )
                else:
                    self._forcecb(state.get("forces", []))
                    self._status(f"Force phase: iter {state['iteration']}")
        except Exception as e:
            self._status(f"Force phase: error: {e}")

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

            # SLERP between the two rotation matrices
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
