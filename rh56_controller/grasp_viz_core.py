"""
grasp_viz_core.py — Core state and logic for GraspViz (no UI dependencies).

GraspVizCore manages:
  - FK model + closure geometry
  - Grasp state (mode, width, z/x/y, plane angles, current result)
  - MuJoCo viewer subprocesses (hand and robot)
  - Mink comparison planner (background thread)
  - Real arm (UR5Bridge) + hand (RH56Hand) connections
  - GraspExecutor (background thread execution)
  - Thread-safe status queue

Subclass GraspVizCore with a UI layer (e.g. GraspVizUI in grasp_viz_ui.py)
to add interactive controls.
"""

import logging
import math
import multiprocessing
import queue
import threading
import time
from typing import Optional, Dict

import numpy as np

from .grasp_geometry import (
    InspireHandFK, ClosureGeometry, ClosureResult, GraspMode,
    CTRL_MAX, NON_THUMB_FINGERS, _DEFAULT_XML, GRASP_FINGER_SETS,
)

_log = logging.getLogger(__name__)
from .grasp_viz_workers import (
    _ACTUATOR_ORDER, _DEFAULT_ROBOT_Z, _VIEWER_CTRL_LEN, _VIEWER_STATE_LEN,
    _VIEWER_FINGER_ORDER, _GRASP_SCENE, _ROBOT_SCENE, _RIGHT_SCENE,
    _DEFAULT_ROBOT_X, _DEFAULT_ROBOT_Y,
    _DEFAULT_H12_X, _DEFAULT_H12_Y, _DEFAULT_H12_Z,
    _H12_SCENE,
    _IK_DT, _IK_MAX_ITERS, _IK_POS_THR, _IK_ORI_THR, _EEFF_LOCAL,
    MODES,
    _worker_jnt_map, _worker_apply_qpos, _worker_add_geoms,
    _hand_viewer_worker, _robot_viewer_worker, _h12_robot_viewer_worker,
    _mat_to_xyz_euler,
)


class GraspVizCore:
    """
    Core state and geometry logic for the grasp visualizer.

    No UI imports — can be subclassed by GraspVizUI (tkinter) or tested
    without a display.
    """

    def __init__(
        self,
        xml_path: str = _DEFAULT_XML,
        rebuild: bool = False,
        port: Optional[str] = None,
        robot_mode: bool = False,
        h12_mode: bool = False,
        send_real: bool = False,
        mink_viz: bool = True,
        real_robot: bool = False,
        ur5_ip: Optional[str] = None,
        ur5_speed: float = 0.10,
        ros_sync: bool = False,
        ros_publish_hz: float = 20.0,
        ros_send_hand_cmd: bool = False,
        rerun_viz: bool = False,
    ) -> None:
        _log.info("Initialising FK model...")
        self.fk      = InspireHandFK(xml_path=xml_path, rebuild=rebuild)
        self.closure = ClosureGeometry(self.fk)

        # Grasp state
        self._mode    = GraspMode.PLANE_4F
        self._width_m = 0.040
        # Decoupled target used by GRASP! (can differ from slider after manual edit)
        self._width_target_m      = 0.040
        self._width_target_edited = False   # True once user has manually set the target
        self._grasp_z = 0.0
        self._result: Optional[ClosureResult] = None
        self._width_range = (0.015, 0.090)

        self._real_robot_mode = real_robot
        self._grasp_hand_locked = False  # True after executor runs; blocks slider→hand sends
        if real_robot:
            robot_mode = True

        self._robot_mode = robot_mode
        self._h12_mode   = h12_mode
        if h12_mode:
            self._grasp_x = _DEFAULT_H12_X
            self._grasp_y = _DEFAULT_H12_Y
            self._grasp_z = _DEFAULT_H12_Z
        elif robot_mode:
            self._grasp_x = _DEFAULT_ROBOT_X
            self._grasp_y = _DEFAULT_ROBOT_Y
            self._grasp_z = _DEFAULT_ROBOT_Z
        else:
            self._grasp_x = 0.0
            self._grasp_y = 0.0
            self._grasp_z = 0.0

        self._plane_rx = 0.0
        self._plane_ry = 0.0
        self._plane_rz = 0.0

        self._wrist3_pos: Optional[np.ndarray] = None
        self._wrist3_mat: Optional[np.ndarray] = None

        # Real hand
        self._hand      = None
        self._send_real = send_real
        if port is not None:
            self._init_hand(port)

        # ---- Multiprocessing viewer state ----
        _mp = multiprocessing.get_context("fork")
        self._mp_ctx = _mp

        self._custom_ctrl_arr  = _mp.Array("d", _VIEWER_CTRL_LEN)
        self._mink_ctrl_arr    = _mp.Array("d", _VIEWER_CTRL_LEN)
        self._viewer_state_arr = _mp.Array("d", _VIEWER_STATE_LEN)

        self._ctrl_open_fingers = np.array([self.fk.ctrl_min[a] for a in _ACTUATOR_ORDER])
        self._sim_grasp_t       = _mp.Value("d", 1.0)
        self._sim_grasp_gen     = 0

        self._hand_ours_stop  = _mp.Event()
        self._hand_mink_stop  = _mp.Event()
        self._robot_ours_stop = _mp.Event()
        self._robot_mink_stop = _mp.Event()
        self._h12_stop        = _mp.Event()

        self._hand_ours_proc:  Optional[multiprocessing.Process] = None
        self._hand_mink_proc:  Optional[multiprocessing.Process] = None
        self._robot_ours_proc: Optional[multiprocessing.Process] = None
        self._robot_mink_proc: Optional[multiprocessing.Process] = None
        self._h12_proc:        Optional[multiprocessing.Process] = None

        # ---- Mink grasp planner (background thread, not subprocess) ----
        self._state_lock        = threading.Lock()
        self._mink_enabled      = False
        self._mink_planner      = None
        self._mink_result       = None
        self._mink_lock         = threading.Lock()
        self._mink_solve_event  = threading.Event()
        self._mink_solve_stop   = threading.Event()
        self._mink_solve_thread: Optional[threading.Thread] = None

        self._init_mink_planner(mink_viz)

        # ---- Real robot arm (UR5) ----
        self._arm      = None
        self._executor = None
        self._teach_mode            = False
        self._manual_teach_override = False
        self._approach_width_m: Optional[float] = None
        self._grasp_strategy = "Plan"
        self._grasp_force_N  = 0.0
        self._grasp_step_mm  = 10.0

        _mp2 = multiprocessing.get_context("fork")
        self._real_q_arr    = _mp2.Array("d", 6)
        self._real_tracking = _mp2.Value("b", 0)

        # Thread-safe status queue (executor/arm threads → UI poll loop)
        self._status_queue: queue.Queue = queue.Queue()

        self._init_real_arm(real_robot, ur5_ip, ur5_speed)
        self._init_executor()

        # Optional in-process ROS2 bridge (state publish + command subscribe)
        self._ros_bridge = None
        if ros_sync:
            try:
                from .grasp_viz_ros import GraspVizRosBridge

                self._ros_bridge = GraspVizRosBridge(
                    core=self,
                    publish_hz=ros_publish_hz,
                    send_hand_cmd=ros_send_hand_cmd,
                    rerun_enabled=rerun_viz,
                )
                ok = self._ros_bridge.start()
                if ok:
                    _log.info("ROS bridge enabled.")
                else:
                    err = getattr(self._ros_bridge, "last_error", "")
                    if err:
                        _log.warning("ROS bridge requested but failed to start: %s", err)
                    else:
                        _log.warning("ROS bridge requested but failed to start.")
            except Exception as exc:
                _log.warning("ROS bridge setup failed: %s", exc)

        if self._real_robot_mode:
            self._start_viewer_monitor()

        self._recompute()

    def _init_hand(self, port: str) -> None:
        """Connect to the real RH56 hand on the given serial port."""
        try:
            from .rh56_hand import RH56Hand
            self._hand = RH56Hand(port=port)
            _log.info("Connected to real hand on %s", port)
        except Exception as exc:
            _log.warning("Could not connect to real hand: %s", exc)

    def _init_mink_planner(self, mink_viz: bool) -> None:
        """Load the Mink comparison planner and start its solve thread."""
        if not mink_viz:
            return
        try:
            from .mink_grasp_planner import MinkGraspPlanner
            _log.info("Loading mink comparison planner...")
            self._mink_planner = MinkGraspPlanner(
                _RIGHT_SCENE, dt=0.005, max_iters=150, conv_thr=5e-3,
            )
            self._mink_enabled = True
            self._mink_solve_thread = threading.Thread(
                target=self._mink_solve_loop, daemon=True, name="mink-solve",
            )
            self._mink_solve_thread.start()
            _log.info("Mink planner ready.")
        except Exception as exc:
            _log.warning("Could not load mink planner: %s", exc)

    def _init_real_arm(self, real_robot: bool, ur5_ip: Optional[str],
                       ur5_speed: float) -> None:
        """Connect to the UR5 arm if in real-robot mode."""
        if not real_robot or ur5_ip is None:
            return
        try:
            from .ur5_bridge import UR5Bridge
            self._arm = UR5Bridge(ip=ur5_ip, speed=ur5_speed)
            if self._arm.connect():
                self._arm.snapshot_joints(self._real_q_arr)
                self._real_tracking.value = 1
                _log.info("UR5 arm connected at %s", ur5_ip)
            else:
                self._arm = None
        except Exception as exc:
            _log.warning("UR5 arm setup failed: %s", exc)
            self._arm = None

    def _init_executor(self) -> None:
        """Create GraspExecutor when both arm and hand are available."""
        if self._arm is None or self._hand is None:
            return
        try:
            from .grasp_executor import GraspExecutor
            self._executor = GraspExecutor(
                arm=self._arm,
                hand=self._hand,
                fk=self.fk,
                status_cb=lambda s: self._status_queue.put(s),
                force_cb=lambda f: self._status_queue.put(f"forces: {f}"),
            )
            _log.info("GraspExecutor ready.")
        except Exception as exc:
            _log.warning("GraspExecutor setup failed: %s", exc)

    def cleanup(self) -> None:
        """Hard shutdown of all hardware and subprocesses."""
        _log.info("Cleaning up resources...")

        # 1. Stop Mink thread
        self._mink_solve_stop.set()
        self._mink_solve_event.set() # Wake it up so it can exit

        # 2. Terminate MuJoCo viewer processes
        for stop_event in [self._hand_ours_stop, self._hand_mink_stop,
                           self._robot_ours_stop, self._robot_mink_stop]:
            stop_event.set()

        # 2b. Stop ROS bridge executor thread
        if self._ros_bridge is not None:
            try:
                self._ros_bridge.stop()
            except Exception as e:
                _log.warning("Error stopping ROS bridge: %s", e)

        # 3. Close the Real Robot connection (Crucial for UR5)
        if self._arm is not None:
            try:
                _log.info("Closing UR5 connection...")
                self._arm.disconnect()
            except Exception as e:
                _log.error("Error closing arm: %s", e)

        # # 4. Close the Real Hand connection
        # if self._hand is not None:
        #     try:
        #         print("[GraspViz] Closing Hand serial port...")
        #         # self._hand is typically a serial-based object
        #         self._hand.close()
        #     except Exception as e:
        #         print(f"Error closing hand: {e}")

        # 5. Join processes briefly
        for p in [self._hand_ours_proc, self._hand_mink_proc,
                  self._robot_ours_proc, self._robot_mink_proc]:
            if p and p.is_alive():
                p.terminate()

    # ------------------------------------------------------------------
    # Status (thread-safe — all paths go through the queue)
    # ------------------------------------------------------------------
    def _update_status(self, msg: str):
        """Enqueue a status message for the UI poll loop."""
        self._status_queue.put(msg)

    # ------------------------------------------------------------------
    # Closure
    # ------------------------------------------------------------------
    def _recompute(self) -> None:
        """Recompute ClosureResult for current mode + width."""
        try:
            result = self.closure.solve(self._mode, self._width_m)
        except Exception as exc:
            _log.warning("solve failed: %s", exc)
            result = None

        with self._state_lock:
            self._result = result

        if result is not None:
            self._push_viewer_ctrl()
            self._send_real_hand()
            if self._mink_enabled:
                self._mink_solve_event.set()

    # ------------------------------------------------------------------
    # Sim2Real
    # ------------------------------------------------------------------
    def _send_real_hand(self) -> None:
        if self._hand is None or not self._send_real or self._result is None:
            return
        if self._grasp_hand_locked:
            return
        r = self._result
        finger_ctrl = np.array([
            r.ctrl_values.get("pinky",           0.0),
            r.ctrl_values.get("ring",            0.0),
            r.ctrl_values.get("middle",          0.0),
            r.ctrl_values.get("index",           0.0),
            r.ctrl_values.get("thumb_proximal",  0.0),
            r.ctrl_values.get("thumb_yaw",       0.0),
        ])
        ctrl_min = np.array([self.fk.ctrl_min[a] for a in _ACTUATOR_ORDER])
        ctrl_max = np.array([self.fk.ctrl_max[a] for a in _ACTUATOR_ORDER])
        rng = ctrl_max - ctrl_min
        real_cmd = np.round(
            (1.0 - np.clip((finger_ctrl - ctrl_min) / np.where(rng > 0, rng, 1.0),
                           0.0, 1.0)) * 1000
        ).astype(int).tolist()

        if (self._real_robot_mode
                and self._grasp_strategy == "Thumb Reflex"
                and not (self._executor is not None and self._executor.is_running())):
            real_cmd[0] = 1000
            real_cmd[1] = 1000
            real_cmd[2] = 1000
            real_cmd[3] = 1000

        try:
            self._hand.angle_set(real_cmd)
        except Exception as exc:
            _log.warning("angle_set failed: %s", exc)

    # ------------------------------------------------------------------
    # Plane / transform helpers
    # ------------------------------------------------------------------
    def _plane_R_matrix(self) -> np.ndarray:
        return ClosureResult._plane_rot(self._plane_rx, self._plane_ry, self._plane_rz)

    @staticmethod
    def _mat_to_xyz_euler(R: np.ndarray):
        return _mat_to_xyz_euler(R)

    def _build_world_T_hand(self, r: ClosureResult) -> np.ndarray:
        """Build 4×4 world_T_hand from closure result + current slider state."""
        gz    = self._grasp_z
        wbase = r.world_base(gz, self._plane_rx, self._plane_ry, self._plane_rz)
        if self._robot_mode or self._h12_mode:
            wbase = wbase + np.array([self._grasp_x, self._grasp_y, 0.0])
        R_full = self._plane_R_matrix() @ ClosureResult._rot_matrix(r.base_tilt_y)
        T = np.eye(4)
        T[:3, :3] = R_full
        T[:3, 3]  = wbase
        return T

    # ------------------------------------------------------------------
    # Viewer monitor (background thread)
    # ------------------------------------------------------------------
    def _start_viewer_monitor(self) -> None:
        """Background thread: disable teach mode when all sim viewers close."""
        def _monitor():
            while True:
                time.sleep(2.0)
                if not self._teach_mode or self._arm is None or self._manual_teach_override:
                    continue
                any_open = any(
                    p is not None and p.is_alive()
                    for p in [self._robot_ours_proc, self._robot_mink_proc,
                               self._hand_ours_proc,  self._hand_mink_proc]
                )
                if not any_open:
                    try:
                        self._arm.disable_teach_mode()
                    except Exception as e:
                        _log.warning("monitor teach-disable error: %s", e)
                    self._status_queue.put("Teach mode disabled (all viewers closed)")
                    self._status_queue.put("__reset_teach_btn__")
        t = threading.Thread(target=_monitor, daemon=True, name="viewer-monitor")
        t.start()

    # ------------------------------------------------------------------
    # Real-hand sync
    # ------------------------------------------------------------------
    def _sync_real_hand_to_ctrl(self, ctrl_arr) -> None:
        if self._hand is None:
            return
        try:
            angles = self._hand.angle_read()
            ctrl_min = np.array([self.fk.ctrl_min[a] for a in _ACTUATOR_ORDER])
            ctrl_max = np.array([self.fk.ctrl_max[a] for a in _ACTUATOR_ORDER])
            rng      = ctrl_max - ctrl_min
            real_ctrl = ctrl_min + (1.0 - np.clip(
                np.array(angles, dtype=float) / 1000.0, 0.0, 1.0)) * rng
            ctrl_arr[6:12] = real_ctrl
        except Exception as exc:
            _log.warning("_sync_real_hand_to_ctrl: %s", exc)

    def _is_cylinder_bad(self) -> bool:
        return self._mode == "cylinder" and self._width_m * 2 < 0.071

    # ------------------------------------------------------------------
    # ROS bridge helpers
    # ------------------------------------------------------------------
    def ros_set_mode(self, label: str) -> None:
        """Set closure mode from ROS command input."""
        if label not in MODES:
            self._update_status(f"ROS ignored invalid mode: {label}")
            return
        self._mode = label
        n = int(label[0]) if label[0].isdigit() else 4
        wrange = self.closure.width_range(label, n_fingers=n)
        self._width_range = wrange
        self._width_m = float(np.clip(self._width_m, wrange[0], wrange[1]))
        self._width_target_m = float(np.clip(self._width_target_m, wrange[0], wrange[1]))
        self._recompute()
        self._update_status(f"ROS mode set: {label}")

    def ros_set_width_m(self, width_m: float) -> None:
        """Set active solve width from ROS command input."""
        wmin, wmax = self._width_range
        self._width_m = float(np.clip(width_m, wmin, wmax))
        if not self._width_target_edited:
            self._width_target_m = self._width_m
        self._recompute()

    def ros_set_target_width_m(self, width_m: float) -> None:
        """Set grasp execution target width from ROS command input."""
        wmin, wmax = self._width_range
        self._width_target_m = float(np.clip(width_m, wmin, wmax))
        self._width_target_edited = True

    def ros_set_target_pose(self, x_m: float, y_m: float, z_m: float) -> None:
        """Set target Cartesian hand-base translation from ROS command input."""
        self._grasp_x = float(x_m)
        self._grasp_y = float(y_m)
        self._grasp_z = float(z_m)
        self._push_viewer_ctrl()
        self._update_status(
            f"ROS pose set: x={self._grasp_x*1000:.0f} y={self._grasp_y*1000:.0f} z={self._grasp_z*1000:.0f} mm"
        )

    def get_ros_snapshot(self) -> Optional[Dict]:
        """Thread-safe snapshot for ROS2/rerun publishers."""
        with self._state_lock:
            r = self._result
        if r is None:
            return None

        wtips = r.world_tips(self._grasp_z, self._plane_rx, self._plane_ry, self._plane_rz)
        ctrl = [
            float(r.ctrl_values.get("pinky", 0.0)),
            float(r.ctrl_values.get("ring", 0.0)),
            float(r.ctrl_values.get("middle", 0.0)),
            float(r.ctrl_values.get("index", 0.0)),
            float(r.ctrl_values.get("thumb_proximal", 0.0)),
            float(r.ctrl_values.get("thumb_yaw", 0.0)),
        ]

        return {
            "mode": str(r.mode),
            "width_m": float(r.width),
            "target_width_m": float(self._width_target_m),
            "finger_span_m": float(r.finger_span),
            "tilt_deg": float(r.tilt_deg),
            "target_pose": {
                "x": float(self._grasp_x),
                "y": float(self._grasp_y),
                "z": float(self._grasp_z),
            },
            "tips_world_m": {k: [float(v[0]), float(v[1]), float(v[2])] for k, v in wtips.items()},
            "ctrl_rad": ctrl,
        }

    # ------------------------------------------------------------------
    # Shared-memory ctrl/state builders
    # ------------------------------------------------------------------
    def _build_ctrl_array(self, r: ClosureResult,
                           finger_ctrl: Optional[Dict] = None) -> np.ndarray:
        gz    = self._grasp_z
        wbase = r.world_base(gz, self._plane_rx, self._plane_ry, self._plane_rz)
        if self._robot_mode or self._h12_mode:
            wbase = wbase + np.array([self._grasp_x, self._grasp_y, 0.0])
        fc = finger_ctrl if finger_ctrl is not None else r.ctrl_values
        R_full = self._plane_R_matrix() @ ClosureResult._rot_matrix(r.base_tilt_y)
        rot_x, rot_y, rot_z = _mat_to_xyz_euler(R_full)
        return np.array([
            wbase[0], wbase[1], wbase[2],
            rot_x, rot_y, rot_z,
            fc.get("pinky",          0.0),
            fc.get("ring",           0.0),
            fc.get("middle",         0.0),
            fc.get("index",          0.0),
            fc.get("thumb_proximal", 0.0),
            fc.get("thumb_yaw",      0.0),
        ], dtype=float)

    def _build_state_array(self, r: ClosureResult) -> np.ndarray:
        gz       = self._grasp_z
        mode_idx = MODES.index(r.mode) if r.mode in MODES else 0
        wtips    = r.world_tips(gz, self._plane_rx, self._plane_ry, self._plane_rz)
        if self._robot_mode or self._h12_mode:
            xy_off = np.array([self._grasp_x, self._grasp_y, 0.0])
            wtips  = {f: p + xy_off for f, p in wtips.items()}
        state = np.full(_VIEWER_STATE_LEN, np.nan)
        state[0] = gz
        state[1] = float(mode_idx)
        state[2] = r.cylinder_radius
        state[3] = self._grasp_x if (self._robot_mode or self._h12_mode) else 0.0
        state[4] = self._grasp_y if (self._robot_mode or self._h12_mode) else 0.0
        for i, fname in enumerate(_VIEWER_FINGER_ORDER):
            if fname in wtips:
                state[5 + i * 3: 5 + i * 3 + 3] = wtips[fname]
        return state

    def _push_viewer_ctrl(self) -> None:
        with self._state_lock:
            r = self._result
        if r is None:
            return
        ctrl  = self._build_ctrl_array(r)
        state = self._build_state_array(r)
        self._custom_ctrl_arr[:]  = ctrl
        self._viewer_state_arr[:] = state

    def _push_mink_viewer_ctrl(self) -> None:
        with self._state_lock:
            r = self._result
        with self._mink_lock:
            m_res = self._mink_result
        if r is None or m_res is None:
            return
        mink_fc = {
            "pinky":          float(m_res.ctrl[0]),
            "ring":           float(m_res.ctrl[1]),
            "middle":         float(m_res.ctrl[2]),
            "index":          float(m_res.ctrl[3]),
            "thumb_proximal": float(m_res.ctrl[4]),
            "thumb_yaw":      float(m_res.ctrl[5]),
        }
        ctrl = self._build_ctrl_array(r, finger_ctrl=mink_fc)
        self._mink_ctrl_arr[:] = ctrl

    # ------------------------------------------------------------------
    # Viewer process launch helpers
    # ------------------------------------------------------------------
    def _launch_hand_viewer_ours(self) -> None:
        if self._hand_ours_proc is not None and self._hand_ours_proc.is_alive():
            _log.debug("Hand viewer (Ours) already open.")
            return
        self._hand_ours_stop.clear()
        self._push_viewer_ctrl()
        self._sync_real_hand_to_ctrl(self._custom_ctrl_arr)
        proc = self._mp_ctx.Process(
            target=_hand_viewer_worker,
            args=(_GRASP_SCENE, self._custom_ctrl_arr,
                  self._viewer_state_arr, self._hand_ours_stop, "Ours"),
            daemon=True,
        )
        proc.start()
        self._hand_ours_proc = proc
        _log.info("Hand viewer (Ours) launched.")

    def _launch_hand_viewer_mink(self) -> None:
        if not self._mink_enabled or self._mink_planner is None:
            _log.warning("Mink planner not available.")
            return
        if self._hand_mink_proc is not None and self._hand_mink_proc.is_alive():
            _log.debug("Hand viewer (Mink) already open.")
            return
        self._hand_mink_stop.clear()
        self._push_mink_viewer_ctrl()
        proc = self._mp_ctx.Process(
            target=_hand_viewer_worker,
            args=(_GRASP_SCENE, self._mink_ctrl_arr,
                  self._viewer_state_arr, self._hand_mink_stop, "Mink"),
            daemon=True,
        )
        proc.start()
        self._hand_mink_proc = proc
        _log.info("Hand viewer (Mink) launched.")

    def _launch_robot_viewer_ours(self) -> None:
        if self._robot_ours_proc is not None and self._robot_ours_proc.is_alive():
            _log.debug("Robot viewer (Ours) already open.")
            return
        self._robot_ours_stop.clear()
        self._push_viewer_ctrl()
        self._sync_real_hand_to_ctrl(self._custom_ctrl_arr)
        proc = self._mp_ctx.Process(
            target=_robot_viewer_worker,
            args=(_ROBOT_SCENE, self._custom_ctrl_arr,
                  self._viewer_state_arr, self._robot_ours_stop),
            kwargs=dict(ik_dt=_IK_DT, ik_max_iters=_IK_MAX_ITERS,
                        ik_pos_thr=_IK_POS_THR, ik_ori_thr=_IK_ORI_THR,
                        eeff_local=tuple(_EEFF_LOCAL),
                        real_q_arr=self._real_q_arr,
                        real_tracking=self._real_tracking,
                        sim_grasp_t=self._sim_grasp_t,
                        ctrl_open_fingers=tuple(self._ctrl_open_fingers)),
            daemon=True,
        )
        proc.start()
        self._robot_ours_proc = proc
        _log.info("Robot viewer (Ours) launched.")

    def _launch_robot_viewer_mink(self) -> None:
        if not self._mink_enabled or self._mink_planner is None:
            _log.warning("Mink planner not available.")
            return
        if self._robot_mink_proc is not None and self._robot_mink_proc.is_alive():
            _log.debug("Robot viewer (Mink) already open.")
            return
        self._robot_mink_stop.clear()
        self._push_mink_viewer_ctrl()
        proc = self._mp_ctx.Process(
            target=_robot_viewer_worker,
            args=(_ROBOT_SCENE, self._mink_ctrl_arr,
                  self._viewer_state_arr, self._robot_mink_stop),
            kwargs=dict(ik_dt=_IK_DT, ik_max_iters=_IK_MAX_ITERS,
                        ik_pos_thr=_IK_POS_THR, ik_ori_thr=_IK_ORI_THR,
                        eeff_local=tuple(_EEFF_LOCAL),
                        real_q_arr=self._real_q_arr,
                        real_tracking=self._real_tracking,
                        sim_grasp_t=self._sim_grasp_t,
                        ctrl_open_fingers=tuple(self._ctrl_open_fingers)),
            daemon=True,
        )
        proc.start()
        self._robot_mink_proc = proc
        _log.info("Robot viewer (Mink) launched.")

    def _launch_h12_viewer(self) -> None:
        if self._h12_proc is not None and self._h12_proc.is_alive():
            _log.debug("H1-2 viewer already open.")
            return
        self._h12_stop.clear()
        self._push_viewer_ctrl()
        proc = self._mp_ctx.Process(
            target=_h12_robot_viewer_worker,
            args=(_H12_SCENE, self._custom_ctrl_arr,
                  self._viewer_state_arr, self._h12_stop),
            kwargs=dict(ik_dt=_IK_DT, ik_max_iters=40,
                        ik_pos_thr=_IK_POS_THR, ik_ori_thr=_IK_ORI_THR,
                        ctrl_open_fingers=tuple(self._ctrl_open_fingers)),
            daemon=True,
        )
        proc.start()
        self._h12_proc = proc
        _log.info("H1-2 viewer launched.")

    def _launch_viewer(self) -> None:
        if self._h12_mode:
            self._launch_h12_viewer()
        elif self._robot_mode:
            self._launch_robot_viewer_ours()
        else:
            self._launch_hand_viewer_ours()

    # ------------------------------------------------------------------
    # Mink solve loop (background thread)
    # ------------------------------------------------------------------
    def _mink_solve_loop(self) -> None:
        while not self._mink_solve_stop.is_set():
            triggered = self._mink_solve_event.wait(timeout=0.1)
            if not triggered:
                continue
            self._mink_solve_event.clear()
            with self._state_lock:
                result = self._result
            if result is None or self._mink_planner is None:
                continue
            try:
                m_res = self._run_mink_for_result(result)
                with self._mink_lock:
                    self._mink_result = m_res
                self._push_mink_viewer_ctrl()
            except Exception as exc:
                _log.warning("Mink solve error: %s", exc)

    def _run_mink_for_result(self, result: ClosureResult):
        mode = result.mode
        tips = result.tip_positions
        if mode == "2-finger line":
            return self._mink_planner.solve_line(
                result.width,
                thumb_target=tips["thumb"],
                index_target=tips["index"],
            )
        elif "plane" in mode:
            n       = int(mode[0])
            fingers = GRASP_FINGER_SETS[n] + ["thumb"]
            targets = {f: tips[f] for f in fingers if f in tips}
            return self._mink_planner.solve_plane(result.width, targets)
        elif mode == "cylinder":
            fingers = NON_THUMB_FINGERS + ["thumb"]
            targets = {f: tips[f] for f in fingers if f in tips}
            return self._mink_planner.solve_cylinder(result.width, targets)
        raise ValueError(f"Unknown mode: {mode}")

    # ------------------------------------------------------------------
    # Static viewer geometry helpers (kept for in-process use)
    # ------------------------------------------------------------------
    @staticmethod
    def _setup_jnt_map(model) -> dict:
        return _worker_jnt_map(model)

    @staticmethod
    def _apply_qpos(jm, data, ctrl, model):
        _worker_apply_qpos(jm, data, ctrl, model)

    def _add_viewer_geoms(self, viewer, result, gz: float):
        if result is None:
            return
        state = self._build_state_array(result)
        _worker_add_geoms(viewer, state)

    # ------------------------------------------------------------------
    # Plan computation
    # ------------------------------------------------------------------
    def _compute_plan_closures(self, step_mm: float, r_target,
                               approach_m: float = None):
        """Compute list of ClosureResult for width-space Plan waypoints."""
        width_end = r_target.width
        if approach_m is not None:
            width_start = approach_m
        elif self._approach_width_m is not None:
            width_start = self._approach_width_m
        else:
            width_start = self._width_range[1]

        width_start = min(width_start, self._width_range[1])
        width_start = max(width_start, width_end + 1e-4)

        try:
            r_approach = self.closure.solve(self._mode, width_start)
        except Exception:
            r_approach = None

        closures = []
        if r_approach is not None:
            closures.append(r_approach)

        step_m = max(step_mm / 1000.0, 1e-4)
        N = max(1, int(math.ceil((width_start - width_end) / step_m)))
        for i in range(1, N + 1):
            t   = i / N
            w_i = width_start + t * (width_end - width_start)
            try:
                r_i = self.closure.solve(self._mode, w_i)
            except Exception:
                r_i = None
            if r_i is not None:
                closures.append(r_i)

        if not closures or closures[-1] is not r_target:
            closures.append(r_target)
        return closures
