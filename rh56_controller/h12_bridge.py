"""
h12_bridge.py — ROS2 client bridge for real H1-2 arm control.

Wraps the h12_ros2_controller ROS2 action servers:
  - /frame_task    (FrameTask action)   — single arm: right or left
  - /dual_arm      (DualArm action)     — bimanual (both arms simultaneously)
  - /named_config  (NamedConfig action) — go to a named configuration

Because rclpy is only available for Python 3.10 and uv runs Python 3.12,
this bridge spawns h12_ros_proxy.py as a subprocess under Python 3.10
(with the ROS2 environment sourced) and communicates via newline-delimited
JSON over stdin/stdout.

Usage (from grasp_viz_core):
    bridge = H12Bridge(bimanual=False)
    bridge.connect()                   # spawns proxy; returns True on success
    bridge.send_arm("right_wrist_yaw_link", T_4x4)
    bridge.send_named_config("home")
    bridge.disconnect()
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional
import numpy as np

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locate proxy script and Python 3.10 interpreter
# ---------------------------------------------------------------------------

_PROXY_SCRIPT = str(Path(__file__).parent / "h12_ros_proxy.py")

# ROS environment needed by the subprocess
_ROS_SETUP = "/opt/ros/humble/setup.bash"
_WS_CTRL_SETUP = os.path.expanduser("~/ws_ctrl/install/setup.bash")

_PYTHON310 = "/usr/bin/python3.10"


def _build_proxy_env() -> dict:
    """Return an os.environ copy with ROS2 paths injected."""
    # Source both setup files in a subshell and extract the environment
    cmd = (
        f"source {_ROS_SETUP} 2>/dev/null && "
        f"source {_WS_CTRL_SETUP} 2>/dev/null && "
        "env"
    )
    try:
        out = subprocess.check_output(
            ["bash", "-c", cmd], text=True, timeout=10
        )
    except Exception as exc:
        _log.warning("H12Bridge: could not source ROS env: %s", exc)
        return dict(os.environ)

    env = {}
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            env[k] = v
    # Merge with current env so PATH etc. are not lost
    merged = dict(os.environ)
    merged.update(env)
    return merged


# ---------------------------------------------------------------------------
# H12Bridge
# ---------------------------------------------------------------------------

class H12Bridge:
    """
    Thin proxy client for H1-2 arm control.

    Spawns h12_ros_proxy.py under Python 3.10 and communicates via JSON
    over stdin/stdout.  All action calls are synchronous within their own
    thread — callers should use threading.Thread to avoid blocking the UI.
    """

    def __init__(self, bimanual: bool = False) -> None:
        self._bimanual   = bimanual
        self._proc: Optional[subprocess.Popen] = None
        self._lock       = threading.Lock()
        self._pending: dict[int, threading.Event] = {}
        self._results:  dict[int, dict] = {}
        self._msg_id    = 0
        self._connected = False
        self.last_error = ""
        self._reader_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Spawn the proxy subprocess and send 'connect'.  Returns True on success."""
        try:
            env = _build_proxy_env()
            self._proc = subprocess.Popen(
                [_PYTHON310, "-u", _PROXY_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1,  # line-buffered
            )
            self._connected = True

            # Background thread to read proxy stdout
            self._reader_thread = threading.Thread(
                target=self._reader_loop, daemon=True, name="h12-proxy-reader")
            self._reader_thread.start()

            # Background thread to log proxy stderr
            threading.Thread(
                target=self._stderr_loop, daemon=True, name="h12-proxy-stderr"
            ).start()

            ok = self._call({"cmd": "connect"}, timeout=15.0)
            if ok:
                _log.info("H12Bridge: proxy connected (ROS2).")
            else:
                _log.warning("H12Bridge: proxy connect failed: %s", self.last_error)
                self._connected = False
            return ok
        except Exception as exc:
            self.last_error = str(exc)
            _log.warning("H12Bridge.connect failed: %s", exc)
            return False

    def disconnect(self) -> None:
        if self._connected and self._proc is not None:
            try:
                self._call({"cmd": "disconnect"}, timeout=3.0)
            except Exception:
                pass
        self._connected = False
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=3.0)
            except Exception:
                pass
            self._proc = None

    def _reader_loop(self):
        """Read JSON replies from proxy stdout and wake waiting callers."""
        try:
            for line in self._proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    _log.debug("H12Bridge: bad proxy reply: %r", line)
                    continue
                msg_id = msg.get("id", -1)
                with self._lock:
                    self._results[msg_id] = msg
                    ev = self._pending.get(msg_id)
                if ev is not None:
                    ev.set()
        except Exception as exc:
            _log.debug("H12Bridge reader loop exited: %s", exc)
        finally:
            self._connected = False

    def _stderr_loop(self):
        try:
            for line in self._proc.stderr:
                _log.info("h12-proxy: %s", line.rstrip())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # RPC helper
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        with self._lock:
            self._msg_id += 1
            return self._msg_id

    def _call_raw(self, msg: dict, timeout: float = 15.0) -> Optional[dict]:
        """Send a command and return the full reply dict, or None on error/timeout."""
        if self._proc is None or self._proc.poll() is not None:
            self.last_error = "proxy process not running"
            return None

        msg_id = self._next_id()
        msg["id"] = msg_id

        ev = threading.Event()
        with self._lock:
            self._pending[msg_id] = ev

        try:
            line = json.dumps(msg) + "\n"
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except Exception as exc:
            self.last_error = str(exc)
            with self._lock:
                self._pending.pop(msg_id, None)
            return None

        if not ev.wait(timeout=timeout + 5.0):
            self.last_error = f"timeout waiting for id={msg_id}"
            with self._lock:
                self._pending.pop(msg_id, None)
                self._results.pop(msg_id, None)
            return None

        with self._lock:
            result = self._results.pop(msg_id, {})
            self._pending.pop(msg_id, None)

        if not result.get("ok", False):
            self.last_error = result.get("error", "unknown error")
        return result

    def _call(self, msg: dict, timeout: float = 15.0) -> bool:
        """Send a command to the proxy and wait for its reply."""
        result = self._call_raw(msg, timeout)
        return result is not None and result.get("ok", False)

    # ------------------------------------------------------------------
    # Single-arm: FrameTask action
    # ------------------------------------------------------------------

    def send_arm(self, frame_name: str, T: np.ndarray,
                 timeout: float = 15.0) -> bool:
        """
        Move a single arm so that `frame_name` reaches pose `T` (4×4 matrix,
        world frame).  Blocks until the action completes or times out.

        frame_name: "right_wrist_yaw_link" or "left_wrist_yaw_link"
        """
        if not self._connected:
            _log.warning("H12Bridge not connected.")
            return False
        return self._call({
            "cmd": "send_arm",
            "frame_name": frame_name,
            "T": T.tolist(),
            "timeout": timeout,
        }, timeout=timeout)

    # ------------------------------------------------------------------
    # Dual-arm: DualArm action
    # ------------------------------------------------------------------

    def send_dual_arm(self,
                      right_T: Optional[np.ndarray],
                      left_T:  Optional[np.ndarray],
                      timeout: float = 15.0) -> bool:
        """
        Move both arms simultaneously via the /dual_arm action.
        Pass None to leave an arm at its current pose.
        """
        if not self._connected:
            _log.warning("H12Bridge not connected.")
            return False
        return self._call({
            "cmd": "send_dual_arm",
            "right_T": right_T.tolist() if right_T is not None else None,
            "left_T":  left_T.tolist()  if left_T  is not None else None,
            "timeout": timeout,
        }, timeout=timeout)

    # ------------------------------------------------------------------
    # Pose query
    # ------------------------------------------------------------------

    def get_wrist_pose(self, frame_name: str,
                       base_frame: str = "world",
                       timeout: float = 5.0) -> Optional[np.ndarray]:
        """
        Return the current 4×4 world→wrist transform via ROS2 TF, or None on failure.

        frame_name: "right_wrist_yaw_link" or "left_wrist_yaw_link"
        """
        if not self._connected:
            _log.warning("H12Bridge not connected.")
            return None
        result = self._call_raw({
            "cmd": "get_wrist_pose",
            "frame_name": frame_name,
            "base_frame": base_frame,
            "timeout": timeout,
        }, timeout=timeout)
        if result is None or not result.get("ok"):
            return None
        try:
            return np.array(result["T"])
        except Exception as exc:
            _log.warning("H12Bridge.get_wrist_pose: bad T in reply: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Named configuration (e.g. "home", "rest")
    # ------------------------------------------------------------------

    def send_named_config(self, name: str, timeout: float = 15.0) -> bool:
        """Send the robot to a named configuration defined in h12_ros2_controller."""
        if not self._connected:
            _log.warning("H12Bridge not connected.")
            return False
        return self._call({
            "cmd": "send_named_config",
            "name": name,
            "timeout": timeout,
        }, timeout=timeout)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._connected
