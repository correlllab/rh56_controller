"""ur5_bridge.py — Thin wrapper around UR5_Interface for grasp_viz integration.

Uses magpie_control (UR5_Interface + SE3) for all real robot communication.
All transforms are in the UR5 world frame (Z up, robot base at origin).

Threading note:
    The UR5 only supports a single RTDE connection.  Do NOT call methods
    from multiple threads simultaneously.  All callers must use a single
    controlling thread (the GraspExecutor background thread) or hold
    self._lock before each call.  Reads (get_tcp_pose_4x4, get_joint_angles)
    are fast and safe to call from any thread as long as no motion is in
    progress, because RTDE receive is separate from control.

Usage:
    bridge = UR5Bridge(ip="192.168.0.4")
    if bridge.connect():
        T_hand = ...  # 4x4 numpy
        warnings = bridge.move_to_hand_pose(T_hand, blocking=True)
        q = bridge.snapshot_joints(shared_array)
    bridge.disconnect()
"""

import logging
import threading
import time
from typing import Optional

import numpy as np

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixed TCP ↔ hand-base body transform
# (constants duplicated from grasp_viz.py — kept here to avoid circular import)
#
# _T_HAND_IN_TCP  :  hand_base expressed in TCP (wrist3) frame
# _T_TCP_IN_HAND  :  TCP (wrist3) expressed in hand_base frame  (inverse)
# ---------------------------------------------------------------------------
# _WRIST3_TO_HAND_POS  = np.array([0.0, 0.156, 0.0])
_WRIST3_TO_HAND_POS  = np.array([0.0, 0.0, 0.156])
# _WRIST3_TO_HAND_POS  = np.array([-0.070, -0.016, 0.155])
# _WRIST3_TO_HAND_QUAT = np.array([-0.5, 0.5, -0.5, -0.5])
_WRIST3_TO_HAND_QUAT = np.array([0.7071068, 0, 0, 0.7071068])


def _quat_wxyz_to_R(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Convert a unit quaternion (w,x,y,z) to a 3×3 rotation matrix."""
    return np.array([
        [1 - 2*(y**2 + z**2),  2*(x*y - w*z),      2*(x*z + w*y)     ],
        [2*(x*y + w*z),        1 - 2*(x**2 + z**2), 2*(y*z - w*x)     ],
        [2*(x*z - w*y),        2*(y*z + w*x),       1 - 2*(x**2 + y**2)],
    ])


def _build_T(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Build 4×4 homogeneous transform from position + wxyz quaternion."""
    T = np.eye(4)
    T[:3, :3] = _quat_wxyz_to_R(*quat_wxyz)
    T[:3, 3]  = pos
    return T


_T_HAND_IN_TCP = _build_T(_WRIST3_TO_HAND_POS, _WRIST3_TO_HAND_QUAT)
_T_TCP_IN_HAND = np.linalg.inv(_T_HAND_IN_TCP)


# ---------------------------------------------------------------------------
# UR5Bridge
# ---------------------------------------------------------------------------
class UR5Bridge:
    """
    Thin wrapper around magpie_control's UR5_Interface for grasp_viz.

    Workspace limits (UR5e):
        MAX_REACH_M   = 0.850 m  — outer reach limit
        MIN_RADIUS_M  = 0.150 m  — inner singularity avoidance (XY radius)
        MIN_Z_M       = 0.000 m  — floor (robot mounted on table, Z ≥ 0)
    """

    MAX_REACH_M   = 0.850
    MIN_RADIUS_M  = 0.150
    MIN_Z_M       = 0.000
    DEFAULT_SPEED = 0.10   # m/s  (conservative for precision grasping)
    DEFAULT_ACCEL = 0.20   # m/s²

    def __init__(
        self,
        ip: str = "192.168.0.4",
        speed: Optional[float] = None,
        accel: Optional[float] = None,
    ):
        self._ip    = ip
        self._speed = speed if speed is not None else self.DEFAULT_SPEED
        self._accel = accel if accel is not None else self.DEFAULT_ACCEL

        self._iface       = None   # UR5_Interface (set in connect())
        self._SE3         = None   # spatialmath.SE3 class (set in connect())
        self._connected   = False
        self._teach_mode  = False
        self._lock        = threading.Lock()
        self._last_q: Optional[np.ndarray] = None  # cached joint angles

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        """
        Connect to the UR5 robot via RTDE (uses magpie_control).
        Returns True on success, False on failure.
        Reads initial joint angles on success.
        """
        try:
            from magpie_control.ur5 import UR5_Interface   # type: ignore
            from spatialmath import SE3                      # type: ignore
            self._SE3   = SE3
            self._iface = UR5_Interface(robotIP=self._ip)
            self._iface.start()
            # Cache initial joint state for the sim viewer snapshot
            self._last_q  = np.array(self._iface.get_joint_angles())
            self._connected = True
            _log.info("Connected to UR5 at %s", self._ip)
            return True
        except Exception as e:
            _log.error("Connect failed: %s", e)
            self._connected = False
            return False

    def disconnect(self):
        """Cleanly disconnect from the UR5.

        Drops the _iface reference after stop() so CPython's refcount immediately
        closes any underlying sockets, freeing the RTDE slot for another client.
        """
        if self._iface is not None:
            try:
                self._iface.stop()
            except Exception as e:
                _log.warning("Error during disconnect: %s", e)
            self._iface = None  # drop ref → sockets closed by refcount GC
        self._connected = False
        _log.info("Disconnected.")

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def in_teach_mode(self) -> bool:
        return self._teach_mode

    # ------------------------------------------------------------------
    # Static coordinate transform helpers
    # ------------------------------------------------------------------
    @staticmethod
    def hand_T_to_tcp_T(world_T_hand: np.ndarray) -> np.ndarray:
        """Convert world_T_hand (4×4) → world_T_tcp (4×4)."""
        return world_T_hand @ _T_TCP_IN_HAND

    @staticmethod
    def tcp_T_to_hand_T(world_T_tcp: np.ndarray) -> np.ndarray:
        """Convert world_T_tcp (4×4) → world_T_hand (4×4)."""
        return world_T_tcp @ _T_HAND_IN_TCP

    # ------------------------------------------------------------------
    # Workspace check (returns warnings, does NOT block execution)
    # ------------------------------------------------------------------
    def check_pose_workspace(self, world_T_hand: np.ndarray) -> list:
        """
        Check workspace bounds for a hand pose.
        Returns a (possibly empty) list of human-readable warning strings.
        The caller decides whether to proceed despite warnings.
        """
        pos  = world_T_hand[:3, 3]
        xy_r = float(np.hypot(pos[0], pos[1]))
        warns = []
        if xy_r > self.MAX_REACH_M:
            warns.append(
                f"WARN: XY reach {xy_r*1000:.0f}mm exceeds {self.MAX_REACH_M*1000:.0f}mm limit")
        if xy_r < self.MIN_RADIUS_M:
            warns.append(
                f"WARN: XY reach {xy_r*1000:.0f}mm below inner {self.MIN_RADIUS_M*1000:.0f}mm limit (singularity)")
        if float(pos[2]) < self.MIN_Z_M:
            warns.append(
                f"WARN: Z={pos[2]*1000:.0f}mm is below table (Z ≥ 0 required)")
        return warns

    # ------------------------------------------------------------------
    # State reads
    # ------------------------------------------------------------------
    def get_tcp_pose_4x4(self) -> Optional[np.ndarray]:
        """Read current TCP pose as a 4×4 numpy matrix (world frame)."""
        if not self._connected:
            return None
        with self._lock:
            return np.array(self._iface.getPose().A)

    def get_hand_pose_4x4(self) -> Optional[np.ndarray]:
        """Read current hand-base pose as a 4×4 numpy matrix (world frame)."""
        tcp = self.get_tcp_pose_4x4()
        return None if tcp is None else self.tcp_T_to_hand_T(tcp)

    def get_joint_angles(self) -> Optional[np.ndarray]:
        """Read current joint angles as (6,) numpy array in radians."""
        if not self._connected:
            return None
        with self._lock:
            return np.array(self._iface.get_joint_angles())

    def snapshot_joints(self, real_q_arr=None) -> Optional[np.ndarray]:
        """
        Read current joint angles and (optionally) write to a shared-memory
        multiprocessing.Array so the robot viewer subprocess can display
        the real robot state.  Call this after each arm move completes.
        Returns the joint angle array, or None if not connected.
        """
        q = self.get_joint_angles()
        if q is not None:
            self._last_q = q
            if real_q_arr is not None:
                real_q_arr[:] = q
        return q

    # ------------------------------------------------------------------
    # Motion  (call only from a single controlling thread)
    # ------------------------------------------------------------------
    def move_to_hand_pose(
        self,
        world_T_hand: np.ndarray,
        speed: Optional[float] = None,
        accel: Optional[float] = None,
        blocking: bool = True,
    ) -> list:
        """
        MoveL to the TCP pose derived from world_T_hand.

        Returns the list of workspace warning strings (move executes
        regardless — caller can check warnings and decide whether to stop).
        blocking=True  : waits for motion to finish before returning.
        blocking=False : returns immediately (use p_moving() to poll).

        Silently does nothing if not connected or in teach mode.
        """
        if not self._connected:
            return ["NOT CONNECTED — arm move skipped"]
        if self._teach_mode:
            return ["TEACH MODE active — arm move blocked"]

        warns        = self.check_pose_workspace(world_T_hand)
        world_T_tcp  = self.hand_T_to_tcp_T(world_T_hand)
        spd = speed if speed is not None else self._speed
        acc = accel if accel is not None else self._accel

        # Orthogonalize R before constructing SE3 to avoid floating-point
        # det-drift that causes spatialmath's validity check to reject it.
        U, _, Vt = np.linalg.svd(world_T_tcp[:3, :3])
        T_clean = world_T_tcp.copy()
        T_clean[:3, :3] = U @ Vt
        with self._lock:
            self._iface.moveL(
                self._SE3(T_clean),
                linSpeed=spd, linAccel=acc, asynch=not blocking,
            )
        if blocking:
            self._wait_done()
        return warns

    def p_moving(self) -> bool:
        """True if the arm is currently in motion."""
        if not self._connected:
            return False
        with self._lock:
            return self._iface.p_moving()

    def _wait_done(self, poll_s: float = 0.05):
        while self.p_moving():
            time.sleep(poll_s)

    def stop(self, decel: float = 2.0):
        """Deceleration stop (emergency use)."""
        if not self._connected:
            return
        try:
            with self._lock:
                self._iface.ctrl.stopL(decel)
        except Exception as e:
            _log.error("stop() error: %s", e)

    # ------------------------------------------------------------------
    # Teach mode
    # ------------------------------------------------------------------
    def enable_teach_mode(self):
        """Enable teach mode (robot can be guided by hand)."""
        if not self._connected:
            _log.warning("Not connected — cannot enable teach mode.")
            return
        with self._lock:
            self._iface.toggle_teach_mode()
        self._teach_mode = True
        _log.info("Teach mode ENABLED.")

    def disable_teach_mode(self):
        """Exit teach mode and return to normal control."""
        if not self._connected:
            _log.warning("Not connected — cannot disable teach mode.")
            return
        with self._lock:
            self._iface.toggle_teach_mode()
        self._teach_mode = False
        _log.info("Teach mode DISABLED.")

    # ------------------------------------------------------------------
    # Set Pose from Robot — decode TCP → grasp planner parameters
    # ------------------------------------------------------------------
    def decode_tcp_to_grasp_params(self, current_result=None) -> dict:
        """
        Read the current TCP pose and decode it into grasp_viz_core parameters.

        Returns dict with keys:
            grasp_x, grasp_y, grasp_z   (metres, UR5 world frame)
            plane_rx, plane_ry, plane_rz (radians, plane orientation)

        If current_result (ClosureResult) is supplied, the plane_rx/ry/rz are
        recovered by inverting the auto-tilt transform.  Otherwise zeroed.

        Math (inverse of _build_ctrl_array in grasp_viz_core.py):
            R_full  = plane_R @ tilt_R
            plane_R = R_full @ tilt_R.T
            => extract XYZ Euler angles from plane_R
        """
        world_T_hand = self.get_hand_pose_4x4()
        if world_T_hand is None:
            return {}

        grasp_x = float(world_T_hand[0, 3])
        grasp_y = float(world_T_hand[1, 3])
        grasp_z = float(world_T_hand[2, 3])
        plane_rx = plane_ry = plane_rz = 0.0

        if current_result is not None:
            try:
                from .grasp_geometry import ClosureResult
                from .grasp_viz_core import GraspVizCore
                R_hand  = world_T_hand[:3, :3]
                R_tilt  = ClosureResult._rot_matrix(current_result.base_tilt_y)
                R_plane = R_hand @ R_tilt.T
                plane_rx, plane_ry, plane_rz = GraspVizCore._mat_to_xyz_euler(R_plane)

                # Invert _build_world_T_hand: hand_base = [-mid_w[0] + gx,
                #   -mid_w[1] + gy, gz - mid_w[2]].  Recover slider coords:
                #   gx = hand_base_X + mid_w[0], etc.
                mid_w = R_hand @ current_result.midpoint
                grasp_x = float(world_T_hand[0, 3]) + mid_w[0]
                grasp_y = float(world_T_hand[1, 3]) + mid_w[1]
                grasp_z = float(world_T_hand[2, 3]) + mid_w[2]
            except Exception as e:
                _log.warning("Plane orientation decode failed: %s", e)

        return {
            "grasp_x":   grasp_x,
            "grasp_y":   grasp_y,
            "grasp_z":   grasp_z,
            "plane_rx":  plane_rx,
            "plane_ry":  plane_ry,
            "plane_rz":  plane_rz,
        }
