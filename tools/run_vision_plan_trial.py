#!/usr/bin/env python3
"""Explicit-speed trial using the existing Plan executor; offline preview default.

Hardware requires --execute --confirm-plan. No lift or automatic release.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "magpie_control/src"))

from rh56_controller.vision_grasp_target import add_vision_target_arguments, vision_target_from_args
from rh56_controller.grasp_viz_core import GraspVizCore
from rh56_controller.grasp_executor import GraspExecutor
from rh56_controller.grasp_logger import GraspLogger
from rh56_controller.ur5_bridge import UR5Bridge


def pose(T):
    return np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()]


def rot_error(a, b):
    return (Rotation.from_rotvec(a[3:]).inv() * Rotation.from_rotvec(b[3:])).magnitude()


def validate_rigid_transform(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("Alignment must be a finite 4x4 transform")
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-6)):
        raise ValueError("Alignment must be a proper rigid transform")
    return matrix


def load_local_alignment(candidate_path, validation_path, final_command):
    """Require the exact frozen candidate and matching fixed-shape holdout."""
    candidate_bytes = candidate_path.read_bytes()
    candidate = json.loads(candidate_bytes)
    validation = json.loads(validation_path.read_text())
    if candidate.get("schema") != "rh56_two_finger_local_alignment/v1":
        raise ValueError("Unsupported local alignment schema")
    if validation.get("schema") != "rh56_frozen_two_finger_holdout/v1":
        raise ValueError("Unsupported alignment holdout schema")
    if (validation.get("candidate_sha256") != hashlib.sha256(candidate_bytes).hexdigest()
            or validation.get("candidate_unchanged") is not True
            or validation.get("refit_performed") is not False
            or validation.get("both_fingers_under_5mm_proxy_gate") is not True):
        raise ValueError("Alignment lacks matching frozen holdout validation")
    if validation.get("hand_command_assumed") != list(final_command):
        raise ValueError("Local alignment was not checked for this final hand command")
    correction = validate_rigid_transform(candidate["correction_left_multiply_in_tcp"])
    if not np.allclose(correction, validation["fixed_correction"], atol=1e-10):
        raise ValueError("Holdout correction differs from candidate")
    mount = validate_rigid_transform(candidate["candidate_tcp_from_hand_current_shape_only"])
    nominal_mount = UR5Bridge.tcp_T_to_hand_T(np.eye(4))
    if not np.allclose(mount, correction @ nominal_mount, atol=1e-6):
        raise ValueError("Candidate was computed for a different nominal hand mount")
    return mount, {"candidate": str(candidate_path), "validation": str(validation_path),
                   "sha256": hashlib.sha256(candidate_bytes).hexdigest(),
                   "scope": "Local final-hand-shape fit; other closures are unvalidated extrapolation",
                   "candidate_tcp_from_hand": mount.tolist()}


def command_hand_pose(desired_hand, local_mount):
    """Adapt only the frame mapping; keep original Plan closure generation."""
    if local_mount is None:
        return desired_hand
    nominal_mount = UR5Bridge.tcp_T_to_hand_T(np.eye(4))
    corrected_tcp = desired_hand @ np.linalg.inv(local_mount)
    return corrected_tcp @ nominal_mount


def inspection_hand_pose(desired_hand, hover_mm):
    """Raise a single inspection target in UR base Z, retaining orientation."""
    if not np.isfinite(hover_mm) or not 0 <= hover_mm <= 100:
        raise ValueError("Inspection hover must be finite and between 0 and 100 mm")
    result = desired_hand.copy()
    result[2, 3] += hover_mm/1000
    return result


def plan_prefix(closures, stop_after_step):
    """Select an unchanged prefix: waypoint 0 is approach, steps start at 1.

    This is a fresh approach-to-step run, not a resume operation. Stopping does
    not issue a release, lift, or the remaining closure commands.
    """
    if stop_after_step is None:
        return closures
    if (isinstance(stop_after_step, bool) or not isinstance(stop_after_step, int)
            or not 1 <= stop_after_step < len(closures)):
        raise ValueError(f"stop-after-step must be an integer in [1, {len(closures)-1}]")
    return closures[:stop_after_step + 1]


def resume_plan_window(closures, resume_after_step, stop_after_step):
    """Include the completed step as a hold anchor, never the open approach."""
    if (isinstance(resume_after_step, bool) or not isinstance(resume_after_step, int)
            or not 1 <= resume_after_step < len(closures) - 1
            or stop_after_step is None or stop_after_step <= resume_after_step):
        raise ValueError("Resume requires a completed closure step and an explicit later stop step")
    return plan_prefix(closures, stop_after_step)[resume_after_step:]


def validate_resume_checkpoint(previous, current, resume_after_step):
    """Match the last commanded hand shape and allow only an explicit upward Z shift.

    A checkpoint proves command history, not physical hand state. The current
    arm must independently match the raised anchor before any serial writes.
    """
    if (previous.get("schema") != "rh56_existing_plan_trial/v1"
            or previous.get("status") != "commands_completed"
            or previous.get("executed_last_plan_step") != resume_after_step
            or previous.get("full_closure_steps") != current["full_closure_steps"]):
        raise ValueError("Resume checkpoint does not identify the completed Plan step")
    if previous.get("hand_commands", [None])[-1] != current["hand_commands"][0]:
        raise ValueError("Resume anchor hand command differs from checkpoint")
    for key in ("source", "object_width_mm", "width_offset_mm", "yaw_deg", "center_policy"):
        if previous["input"].get(key) != current["input"].get(key):
            raise ValueError(f"Resume input changed: {key}")
    if previous.get("local_alignment", {}).get("sha256") != current.get("local_alignment", {}).get("sha256"):
        raise ValueError("Resume alignment changed")
    old = np.asarray(previous["waypoints"][-1], dtype=float)
    new = np.asarray(current["waypoints"][0], dtype=float)
    if old.shape != (6,) or new.shape != (6,) or not np.all(np.isfinite([old, new])):
        raise ValueError("Resume anchor must contain finite TCP poses")
    dz = float(new[2] - old[2])
    center_delta = np.array(current["input"]["center_base_m"]) - previous["input"]["center_base_m"]
    if (np.linalg.norm(new[:2] - old[:2]) > 1e-8 or rot_error(old, new) > 1e-7
            or not -1e-8 <= dz <= .020 + 1e-8
            or not np.allclose(center_delta, [0., 0., dz], rtol=0., atol=1e-8)):
        raise ValueError("Resume permits only the same path or an explicit upward base-Z shift up to 20 mm")
    return dz


class SerialLog:
    def __init__(self, serial, log):
        self.serial, self.log = serial, log

    def write(self, data):
        self.log.write(json.dumps({"time": time.time(), "tx_hex": data.hex()}) + "\n")
        self.log.flush()
        count = self.serial.write(data)
        if count != len(data):
            raise RuntimeError("Incomplete hand serial write")
        return count

    def read(self, size):
        data = self.serial.read(size)
        self.log.write(json.dumps({"time": time.time(), "rx_hex": data.hex()}) + "\n")
        self.log.flush()
        return data

    def __getattr__(self, name):
        return getattr(self.serial, name)


class CheckedBridge(UR5Bridge):
    """Keep original blocking bridge motion, checking state at segment boundaries."""

    def move_to_hand_pose(self, T, **kwargs):
        recv = self._iface.recv
        if recv.getSafetyMode() != 1 or recv.getRobotMode() != 7:
            raise RuntimeError("Robot not in normal state before segment")
        warnings = self.check_pose_workspace(T)
        if warnings:
            raise RuntimeError(str(warnings))
        result = super().move_to_hand_pose(T, **kwargs)
        actual = self.get_hand_pose_4x4()
        if np.linalg.norm(actual[:3, 3] - T[:3, 3]) > .002:
            raise RuntimeError("Arm segment did not reach its position target")
        if rot_error(pose(actual), pose(T)) > np.deg2rad(1):
            raise RuntimeError("Arm segment did not reach its orientation target")
        if recv.getSafetyMode() != 1:
            raise RuntimeError("Robot safety state changed")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_vision_target_arguments(parser)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-snapshot", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-plan", "--confirm-slow-plan", dest="confirm_plan", action="store_true",
                        help="Confirm RH56 closure, UR5 Plan movement and stopScript on exit")
    parser.add_argument("--speed-profile", choices=("slow", "planner"), default="slow",
                        help="slow: 0.01 m/s, 0.02 m/s2, hand 100; planner: 0.10 m/s, 0.20 m/s2, hand 1000; force defaults to 100 independently")
    parser.add_argument("--hand-force-limit-raw", type=int, default=100,
                        help="Explicit per-channel force limit, 0..1000; default 100. Values above 100 need --confirm-high-force when executing")
    parser.add_argument("--confirm-high-force", action="store_true",
                        help="Confirm the explicitly requested force limit above 100; not implied by selecting original planner speed")
    parser.add_argument("--phase", choices=("plan", "approach-only"), default="plan",
                        help="approach-only executes only original Plan waypoint 0 and its open approach hand command")
    parser.add_argument("--stop-after-step", type=int,
                        help="Only with plan: run original approach plus closure steps 1..N, then stop; not a resume command")
    parser.add_argument("--resume-after-step", type=int,
                        help="Use this completed closure as a hold anchor, then continue to the explicit stop step without reopening")
    parser.add_argument("--resume-summary", type=Path,
                        help="Completed trial summary proving the last commanded hand shape; required with resume-after-step")
    parser.add_argument("--approach-hover-mm", type=float, default=0.,
                        help="Only with approach-only: stop this far above the original approach in base Z (0..100 mm), without a later descent")
    parser.add_argument("--local-alignment", type=Path)
    parser.add_argument("--alignment-validation", type=Path)
    parser.add_argument("--acknowledge-shape-extrapolation", action="store_true",
                        help="Acknowledge that applying the final-shape fit to open/intermediate hand shapes is experimentally unvalidated")
    args = parser.parse_args()
    if not args.vision_target:
        parser.error("--vision-target is required")
    if args.execute and (not args.confirm_plan or args.start_snapshot is None):
        parser.error("Motion requires --confirm-plan and --start-snapshot")
    if not 0 <= args.hand_force_limit_raw <= 1000:
        parser.error("--hand-force-limit-raw must be in [0, 1000]")
    if args.execute and args.hand_force_limit_raw > 100 and not args.confirm_high_force:
        parser.error("Force limits above 100 require --confirm-high-force")
    if bool(args.local_alignment) != bool(args.alignment_validation):
        parser.error("--local-alignment and --alignment-validation must be supplied together")
    if args.execute and args.local_alignment and not args.acknowledge_shape_extrapolation:
        parser.error("Local alignment motion requires --acknowledge-shape-extrapolation")
    if not np.isfinite(args.approach_hover_mm) or not 0 <= args.approach_hover_mm <= 100:
        parser.error("--approach-hover-mm must be finite and in [0, 100]")
    if args.approach_hover_mm and args.phase != "approach-only":
        parser.error("--approach-hover-mm requires --phase approach-only")
    if args.stop_after_step is not None and (args.phase != "plan" or args.stop_after_step < 1):
        parser.error("--stop-after-step requires --phase plan and a positive step number")
    if (args.resume_after_step is None) != (args.resume_summary is None):
        parser.error("--resume-after-step and --resume-summary must be supplied together")
    if args.resume_after_step is not None and (args.phase != "plan" or args.stop_after_step is None):
        parser.error("Resume requires --phase plan and an explicit --stop-after-step")
    speed, accel, hand_speed = ((UR5Bridge.DEFAULT_SPEED, UR5Bridge.DEFAULT_ACCEL, 1000)
                                if args.speed_profile == "planner" else (.01, .02, 100))
    target = vision_target_from_args(args)
    core = GraspVizCore(vision_target=target, mink_viz=False, hand_write_only=True)
    args.out.mkdir(parents=True, exist_ok=False)
    bridge = hand = executor = logger = serial_log = None
    statuses = []
    failure = None
    summary = {"schema": "rh56_existing_plan_trial/v1", "input": target.metadata(),
               "status": "preview", "hardware_motion_commands_sent": False,
               "speed_m_s": speed, "accel_m_s2": accel,
               "speed_profile": args.speed_profile, "phase": args.phase,
               "approach_hover_mm": args.approach_hover_mm,
               "requested_stop_after_step": args.stop_after_step,
               "resume_after_step": args.resume_after_step,
               "hand_speed_raw": hand_speed, "hand_force_limit_raw": args.hand_force_limit_raw,
               "feedback_force_N": 0, "hand_write_only": True,
               "sequence": "existing execute_plan_waypoints; unchanged width steps and sleeps",
               "grasp_success": "unverified", "automatic_lift_or_release": False}

    def status(message):
        if args.resume_after_step is not None:
            if message == "Plan: opening fingers to approach config...":
                message = "Plan: reasserting completed-step hand command; not reopening..."
            elif message.startswith("Plan: moving arm to approach pose"):
                message = "Plan: checking completed-step hold anchor; not returning to original approach..."
        statuses.append(message)
        print(message, flush=True)

    try:
        result = core.closure.solve(core._mode, core._width_target_m)
        closures = core._compute_plan_closures(10, result)
        summary["full_closure_steps"] = len(closures) - 1
        final_command = GraspExecutor(None, None, core.fk)._ctrl_to_real(result.ctrl_values)
        local_mount = None
        if args.local_alignment:
            local_mount, alignment_metadata = load_local_alignment(args.local_alignment, args.alignment_validation, final_command)
            summary["local_alignment"] = alignment_metadata
        if args.phase == "approach-only":
            closures = closures[:1]
        elif args.resume_after_step is not None:
            closures = resume_plan_window(closures, args.resume_after_step, args.stop_after_step)
        else:
            closures = plan_prefix(closures, args.stop_after_step)
        first_step = args.resume_after_step or 0
        summary["selected_plan_step_indices"] = list(range(first_step, first_step + len(closures)))
        summary["selected_last_plan_step"] = first_step + len(closures) - 1
        summary["remaining_closure_steps"] = summary["full_closure_steps"] - summary["selected_last_plan_step"]
        summary["is_partial_plan"] = summary["remaining_closure_steps"] > 0
        desired_waypoints = [(core._build_world_T_hand(r), r) for r in closures]
        if args.phase == "approach-only":
            desired_waypoints = [(inspection_hand_pose(T, args.approach_hover_mm), r) for T, r in desired_waypoints]
        waypoints = [(command_hand_pose(T, local_mount), r) for T, r in desired_waypoints]
        summary["desired_physical_hand_poses"] = [T.tolist() for T, r in desired_waypoints]
        summary["bridge_command_frame_note"] = "Executor reports nominal bridge-frame pose; corrected physical hand pose is inferred, not directly measured"
        summary["hand_commands"] = [GraspExecutor(None, None, core.fk)._ctrl_to_real(r.ctrl_values) for T, r in waypoints]
        summary["waypoints"] = [pose(UR5Bridge.hand_T_to_tcp_T(T)).tolist() for T, r in waypoints]
        summary["waypoint_count"] = len(waypoints)
        if args.resume_after_step is not None:
            checkpoint_bytes = args.resume_summary.read_bytes()
            dz = validate_resume_checkpoint(json.loads(checkpoint_bytes), summary, args.resume_after_step)
            summary["resume_checkpoint"] = {"path": str(args.resume_summary),
                "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                "base_z_shift_from_checkpoint_mm": dz * 1000,
                "hand_state_source": "Last commanded shape only; no physical readback",
                "anchor_action": "Reassert same hand command, then existing arm-first/hand-second steps"}
        print(json.dumps(summary, indent=2), flush=True)
        if not args.execute:
            return

        from rh56_controller.rh56_hand import RH56Hand
        bridge = CheckedBridge(ip="192.168.0.4", speed=speed, accel=accel)
        if not bridge.connect():
            raise RuntimeError("Original UR5Bridge connection failed")
        control, recv = bridge._iface.ctrl, bridge._iface.recv
        if recv.getRobotMode() != 7 or recv.getSafetyMode() != 1 or recv.isProtectiveStopped():
            raise RuntimeError("Robot not in normal state")
        if np.max(np.abs(control.getTCPOffset())) > 1e-8:
            raise RuntimeError("Trial requires zero active TCP offset")
        start = np.asarray(recv.getActualTCPPose())
        q = np.asarray(recv.getActualQ())
        expected = json.loads(args.start_snapshot.read_text())
        if args.resume_after_step is not None:
            if not 0 <= time.time() - expected["time_s"] <= 300:
                raise RuntimeError("Resume snapshot must be less than five minutes old")
            anchor = summary["waypoints"][0]
            if (np.linalg.norm(start[:3] - np.array(anchor)[:3]) > .001
                    or rot_error(start, anchor) > np.deg2rad(.5)
                    or np.linalg.norm(recv.getActualTCPSpeed()) > .001):
                raise RuntimeError("Robot must already be stationary at the resume anchor; no automatic repositioning")
        if np.linalg.norm(start[:3] - np.array(expected["tcp"])[:3]) > .002 or rot_error(start, expected["tcp"]) > np.deg2rad(1):
            raise RuntimeError("Start pose changed since the read-only snapshot")
        if np.max(np.abs(q - expected["q"])) > .01:
            raise RuntimeError("Start joints changed")
        summary["initial_tcp_pose"] = start.tolist()
        summary["initial_q_rad"] = q.tolist()

        # Check controller IK along every Cartesian segment before any hand or
        # arm movement. Controller IK uses the real CB3, not the UR5e viewer.
        previous = start
        total_distance = 0.0
        ik_rows = []
        for idx, (T, _) in enumerate(waypoints):
            goal = pose(bridge.hand_T_to_tcp_T(T))
            distance = float(np.linalg.norm(goal[:3] - previous[:3]))
            total_distance += distance
            if distance > (.30 if idx == 0 else .05):
                raise RuntimeError(f"Unexpected segment distance: {distance}")
            interpolation = Slerp([0, 1], Rotation.from_rotvec([previous[3:], goal[3:]]))
            for alpha in np.linspace(0, 1, max(3, int(np.ceil(distance / .005)) + 1)):
                candidate = np.r_[previous[:3] + alpha*(goal[:3]-previous[:3]), interpolation(alpha).as_rotvec()]
                if not control.isPoseWithinSafetyLimits(candidate.tolist()):
                    raise RuntimeError("Controller pose safety check failed")
                solved = np.asarray(control.getInverseKinematics(candidate.tolist(), q.tolist(), 1e-5, 1e-5))
                if solved.shape != (6,) or not np.all(np.isfinite(solved)) or np.max(np.abs(solved-q)) > .15:
                    raise RuntimeError("Controller IK failed or changed branch")
                q = solved
            ik_rows.append({"waypoint": idx, "distance_m": distance, "q_rad": q.tolist()})
            previous = goal
        summary["controller_ik"] = ik_rows
        summary["total_tcp_distance_m"] = total_distance
        print(f"Controller preflight passed: {len(waypoints)} waypoints, {total_distance*1000:.1f} mm total. Starting {args.phase} at {speed} m/s.", flush=True)

        hand = RH56Hand("/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0", hand_id=1)
        hand.ser.exclusive = True
        serial_log = (args.out / "hand_serial.jsonl").open("w")
        hand.ser = SerialLog(hand.ser, serial_log)
        # Same initialization spacing as the GUI. Speed is explicit; restoring
        # original speed does not authorize restoring the GUI maximum force.
        hand.clear_errors()
        time.sleep(.125)
        hand.speed_set([hand_speed]*6)
        time.sleep(.125)
        hand.force_set([args.hand_force_limit_raw]*6)
        time.sleep(.125)
        logger = GraspLogger("vision_plan", log_dir=str(args.out))
        logger.log("input_context", **summary)
        executor = GraspExecutor(bridge, hand, core.fk, status_cb=status, hand_write_only=True)
        summary["hardware_motion_commands_sent"] = True
        executor.execute_plan_waypoints(waypoints, force_N=0, logger=logger)
        t0 = time.monotonic()
        with (args.out / "telemetry.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["elapsed_s", "x", "y", "z", "rx", "ry", "rz", "safety_mode"])
            while executor.is_running():
                elapsed = time.monotonic()-t0
                writer.writerow([elapsed, *recv.getActualTCPPose(), recv.getSafetyMode()])
                handle.flush()
                if recv.getSafetyMode() != 1 or elapsed > total_distance/speed + 30:
                    executor._abort.set()
                    control.stopL(.2)
                    raise RuntimeError("Safety change or trial timeout")
                time.sleep(.05)
        if "Plan: complete." not in statuses:
            raise RuntimeError("Existing Plan did not report completion")
        summary.update(status="commands_completed", elapsed_s=time.monotonic()-t0,
                       executed_last_plan_step=summary["selected_last_plan_step"],
                       final_tcp_pose=recv.getActualTCPPose(), final_q_rad=recv.getActualQ(),
                       final_safety_mode=recv.getSafetyMode())
        if summary["is_partial_plan"]:
            print(f"Stopped after original Plan step {summary['executed_last_plan_step']}/{summary['full_closure_steps']}; no further closure, lift or release commanded.", flush=True)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        summary.update(status="failed", error=failure)
        if executor is not None:
            executor._abort.set()
            bridge._iface.ctrl.stopL(.2)
            executor._thread.join(timeout=3)
    finally:
        if bridge is not None:
            bridge.disconnect()
            summary["bridge_disconnect_called"] = True
        if hand is not None:
            hand.ser.close()
        if serial_log is not None:
            serial_log.close()
        if logger is not None and not logger._file.closed:
            logger.close()
        core.cleanup()
        summary["statuses"] = statuses
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps(summary, indent=2), flush=True)
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
